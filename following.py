# %%
import itertools
import os
import sys
import random
import time
import warnings
from dataclasses import dataclass, astuple
from typing import Literal
import numpy as np
from utils import *

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch as t
import torch.nn as nn
import torch.optim as optim
import wandb
from IPython.display import HTML, display
from jaxtyping import Bool, Float, Int
from matplotlib.animation import FuncAnimation
from numpy.random import Generator
from torch import Tensor
from torch.distributions.categorical import Categorical
from torch.optim.optimizer import Optimizer
from tqdm import tqdm

warnings.filterwarnings("ignore")
Arr = np.ndarray
device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")

@dataclass
class PPOArgs:
    seed: int = 1

    # Wandb / logging
    use_wandb: bool = False
    wandb_project_name: str = "swanPPO"
    wandb_entity: str = None

    # Duration of different phases
    total_timesteps: int = 500_000
    num_envs: int = 4
    num_steps_per_rollout: int = 128
    num_minibatches: int = 4
    batches_per_learning_phase: int = 4

    # Optimization hyperparameters
    lr: float = 2.5e-4
    max_grad_norm: float = 0.5

    # RL hyperparameters
    gamma: float = 0.99

    # PPO-specific hyperparameters
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.25

    def __post_init__(self):
        self.batch_size = self.num_steps_per_rollout * self.num_envs

        assert self.batch_size % self.num_minibatches == 0, "batch_size must be divisible by num_minibatches"
        self.minibatch_size = self.batch_size // self.num_minibatches
        self.total_phases = self.total_timesteps // self.batch_size
        self.total_training_steps = self.total_phases * self.batches_per_learning_phase * self.num_minibatches


args = PPOArgs(num_minibatches=2)

# %%

def layer_init(layer: nn.Linear, std=np.sqrt(2), bias_const=0.0):
    t.nn.init.orthogonal_(layer.weight, std)
    t.nn.init.constant_(layer.bias, bias_const)
    return layer

ACTION_NONE = 0
ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTIONS = [ACTION_NONE, ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]
NUM_ACTIONS = len(ACTIONS)
OBS_ME_X = 0
OBS_ME_Y = 1
OBS_ME_VX = 2
OBS_ME_VY = 3
OBS_BALL_X = 4
OBS_BALL_Y = 5
OBS_BALL_VX = 6
OBS_BALL_VY = 7
NUM_OBS = 8
TIME_STEP = 0.05


def get_actor_and_critic() -> tuple[nn.Module, nn.Module]:
    critic = nn.Sequential(
        layer_init(nn.Linear(NUM_OBS, 64)),
        nn.Tanh(),
        layer_init(nn.Linear(64, 64)),
        nn.Tanh(),
        layer_init(nn.Linear(64, 1), std=1.0),
    )

    actor = nn.Sequential(
        layer_init(nn.Linear(NUM_OBS, 64)),
        nn.Tanh(),
        layer_init(nn.Linear(64, 64)),
        nn.Tanh(),
        layer_init(nn.Linear(64, NUM_ACTIONS), std=0.01),
    )
    return actor.to(device), critic.to(device)

@t.inference_mode()
def compute_advantages(
    next_value: Float[Tensor, "num_envs"],
    next_terminated: Bool[Tensor, "num_envs"],
    rewards: Float[Tensor, "buffer_size num_envs"],
    values: Float[Tensor, "buffer_size num_envs"],
    terminated: Bool[Tensor, "buffer_size num_envs"],
    gamma: float,
    gae_lambda: float,
) -> Float[Tensor, "buffer_size num_envs"]:
    """
    Compute advantages using Generalized Advantage Estimation.
    """
    T = values.shape[0]
    terminated = terminated.float()
    next_terminated = next_terminated.float()

    # Get tensors of V(s_{t+1}) and d_{t+1} for all t = 0, 1, ..., T-1
    next_values = t.concat([values[1:], next_value[None, :]])
    next_terminated = t.concat([terminated[1:], next_terminated[None, :]])

    # Compute deltas: \delta_t = r_t + (1 - d_{t+1}) \gamma V(s_{t+1}) - V(s_t)
    deltas = rewards + gamma * next_values * (1.0 - next_terminated) - values

    # Compute advantages using the recursive formula, starting with advantages[T-1] = deltas[T-1] and working backwards
    advantages = t.zeros_like(deltas)
    advantages[-1] = deltas[-1]
    for s in reversed(range(T - 1)):
        advantages[s] = deltas[s] + gamma * gae_lambda * (1.0 - terminated[s + 1]) * advantages[s + 1]

    return advantages


def get_minibatch_indices(rng: Generator, batch_size: int, minibatch_size: int) -> list[np.ndarray]:
    """
    Return a list of length `num_minibatches`, where each element is an array of `minibatch_size` and the union of all
    the arrays is the set of indices [0, 1, ..., batch_size - 1] where `batch_size = num_steps_per_rollout * num_envs`.
    """
    assert batch_size % minibatch_size == 0
    num_minibatches = batch_size // minibatch_size
    indices = rng.permutation(batch_size).reshape(num_minibatches, minibatch_size)
    return list(indices)

@dataclass
class ReplayMinibatch:
    """
    Samples from the replay memory, converted to PyTorch for use in neural network training.

    Data is equivalent to (s_t, a_t, logpi(a_t|s_t), A_t, A_t + V(s_t), d_{t+1})
    """

    obs: Float[Tensor, "minibatch_size *obs_shape"]
    actions: Int[Tensor, "minibatch_size *action_shape"]
    logprobs: Float[Tensor, "minibatch_size"]
    advantages: Float[Tensor, "minibatch_size"]
    returns: Float[Tensor, "minibatch_size"]
    terminated: Bool[Tensor, "minibatch_size"]


class ReplayMemory:
    """
    Contains buffer; has a method to sample from it to return a ReplayMinibatch object.
    """

    rng: Generator
    obs: Float[Arr, "buffer_size num_envs *obs_shape"]
    actions: Int[Arr, "buffer_size num_envs *action_shape"]
    logprobs: Float[Arr, "buffer_size num_envs"]
    values: Float[Arr, "buffer_size num_envs"]
    rewards: Float[Arr, "buffer_size num_envs"]
    terminated: Bool[Arr, "buffer_size num_envs"]

    def __init__(
        self,
        num_envs: int,
        obs_shape: tuple,
        action_shape: tuple,
        batch_size: int,
        minibatch_size: int,
        batches_per_learning_phase: int,
        seed: int = 42,
    ):
        self.num_envs = num_envs
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.batches_per_learning_phase = batches_per_learning_phase
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        """Resets all stored experiences, ready for new ones to be added to memory."""
        self.obs = np.empty((0, self.num_envs, *self.obs_shape), dtype=np.float32)
        self.actions = np.empty((0, self.num_envs, *self.action_shape), dtype=np.int32)
        self.logprobs = np.empty((0, self.num_envs), dtype=np.float32)
        self.values = np.empty((0, self.num_envs), dtype=np.float32)
        self.rewards = np.empty((0, self.num_envs), dtype=np.float32)
        self.terminated = np.empty((0, self.num_envs), dtype=bool)

    def add(
        self,
        obs: Float[Arr, "num_envs *obs_shape"],
        actions: Int[Arr, "num_envs *action_shape"],
        logprobs: Float[Arr, "num_envs"],
        values: Float[Arr, "num_envs"],
        rewards: Float[Arr, "num_envs"],
        terminated: Bool[Arr, "num_envs"],
    ) -> None:
        """Add a batch of transitions to the replay memory."""
        # Check shapes & datatypes
        for data, expected_shape in zip(
            [obs, actions, logprobs, values, rewards, terminated], [self.obs_shape, self.action_shape, (), (), (), ()]
        ):
            assert_equal(type(data), np.ndarray)
            assert_equal(data.shape, (self.num_envs, *expected_shape))

        # Add data to buffer (not slicing off old elements)
        self.obs = np.concatenate((self.obs, obs[None, :]))
        self.actions = np.concatenate((self.actions, actions[None, :]))
        self.logprobs = np.concatenate((self.logprobs, logprobs[None, :]))
        self.values = np.concatenate((self.values, values[None, :]))
        self.rewards = np.concatenate((self.rewards, rewards[None, :]))
        self.terminated = np.concatenate((self.terminated, terminated[None, :]))

    def get_minibatches(
        self, next_value: Tensor, next_terminated: Tensor, gamma: float, gae_lambda: float
    ) -> list[ReplayMinibatch]:
        """
        Returns a list of minibatches. Each minibatch has size `minibatch_size`, and the union over all minibatches is
        `batches_per_learning_phase` copies of the entire replay memory.
        """
        # Convert everything to tensors on the correct device
        obs, actions, logprobs, values, rewards, terminated = (
            t.tensor(x, device=device, dtype=t.float)
            for x in [self.obs, self.actions, self.logprobs, self.values, self.rewards, self.terminated]
        )

        # Compute advantages & returns
        advantages = compute_advantages(next_value, next_terminated, rewards, values, terminated, gamma, gae_lambda)
        returns = advantages + values

        # Return a list of minibatches
        minibatches = []
        for _ in range(self.batches_per_learning_phase):
            for indices in get_minibatch_indices(self.rng, self.batch_size, self.minibatch_size):
                minibatches.append(
                    ReplayMinibatch(
                        *[
                            data.flatten(0, 1)[indices]
                            for data in [obs, actions, logprobs, advantages, returns, terminated]
                        ]
                    )
                )

        # Reset memory (since we only need to call this method once per learning phase)
        self.reset()

        return minibatches
    
@dataclass
class StepResult:
    obs: Float[Arr, "num_envs *obs_shape"]
    rewards: Float[Arr, "num_envs"]
    terminated: Bool[Arr, "num_envs"]

ME_RADIUS = 0.05
BALL_RADIUS = 0.4

class EnvVector:
    """
    Vectorized environment.
    """
    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.obs = np.zeros((NUM_OBS, self.num_envs))
        self.obs[OBS_BALL_X] = -3
        self.obs[OBS_BALL_Y] = -2
        self.obs[OBS_BALL_VX] = 0
        self.obs[OBS_BALL_VY] = 2
        self.obs[OBS_ME_X] = 0
        self.obs[OBS_ME_Y] = 0
        self.obs[OBS_ME_VX] = 0
        self.obs[OBS_ME_VY] = 0

    def step(self, actions: Int[Arr, "num_envs"]) -> StepResult:
        obs = self.obs
        rewards = np.full(self.num_envs, 1.0)
        me_a = np.stack([np.where(actions == ACTION_RIGHT, 1, 0) - np.where(actions == ACTION_LEFT, 1, 0), np.where(actions == ACTION_UP, 1, 0) - np.where(actions == ACTION_DOWN, 1, 0)], axis=0)
        me_p, me_v = obs[OBS_ME_X:OBS_ME_Y+1], obs[OBS_ME_VX:OBS_ME_VY+1]
        ball_p, ball_v = obs[OBS_BALL_X:OBS_BALL_Y+1], obs[OBS_BALL_VX:OBS_BALL_VY+1]
        # ball accellerates to me with constant acceleration
        ball_acc_factor = 1.0
        ball_a = (me_p - ball_p) / np.linalg.norm(me_p - ball_p, axis=0) * ball_acc_factor
        self._move_object(me_p, me_v, me_a, clip=True)
        self._move_object(ball_p, ball_v, ball_a, clip=False)
        
        terminated = np.linalg.norm(me_p - ball_p, axis=0) < ME_RADIUS + BALL_RADIUS
        return StepResult(self.obs, rewards, terminated)
    
    def _move_object(self, p: Float[Arr, "2 num_envs"], v: Float[Arr, "2 num_envs"], a: Float[Arr, "2 num_envs"], clip: bool = False):
        a = a - 0.05 * v # friction
        v += a * TIME_STEP
        p += v * TIME_STEP + 0.5 * a * TIME_STEP**2
        if clip:
            size = 1.0
             # Zero out velocity in direction of wall collision
            v[0] = np.where(p[0] < -size, 0, np.where(p[0] > size, 0, v[0]))
            v[1] = np.where(p[1] < -size, 0, np.where(p[1] > size, 0, v[1]))
            # Clip position
            p = np.clip(p, -size, size)


class PPOAgent:
    critic: nn.Module
    actor: nn.Module

    def __init__(self, envs: EnvVector, actor: nn.Module, critic: nn.Module, memory: ReplayMemory):
        super().__init__()
        self.envs = envs
        self.actor = actor
        self.critic = critic
        self.memory = memory

        self.step = 0  # Tracking number of steps taken (across all environments)
        self.next_obs = t.tensor(envs.obs.T, device=device, dtype=t.float)  # need starting obs (in tensor form)
        self.next_terminated = t.zeros(envs.num_envs, device=device, dtype=t.bool)  # need starting termination=False

    def play_step(self) -> list[dict]:
        """
        Carries out a single interaction step between the agent and the environment, and adds results to the replay memory.

        returns info list dict
        """
        # Get newest observations (i.e. where we're starting from)
        obs = self.next_obs
        terminated = self.next_terminated

        # Compute logits based on newest observation, and use it to get an action distribution we sample from
        with t.inference_mode():
            logits = self.actor(obs)
        dist = Categorical(logits=logits)
        actions = dist.sample() # samples one action for each env

        # Step environment based on the sampled action
        next_obs, rewards, next_terminated = astuple(self.envs.step(actions.cpu().numpy()))
        next_obs = next_obs.T

        # Calculate logprobs and values, and add this all to replay memory
        logprobs = dist.log_prob(actions).cpu().numpy()
        with t.inference_mode():
            values = self.critic(obs).flatten().cpu().numpy()
        self.memory.add(obs.cpu().numpy(), actions.cpu().numpy(), logprobs, values, rewards, terminated.cpu().numpy())

        # Set next observation & termination state
        self.next_obs = t.from_numpy(next_obs).to(device, dtype=t.float)
        self.next_terminated = t.from_numpy(next_terminated).to(device, dtype=t.float)

        self.step += self.envs.num_envs
        return []

    def get_minibatches(self, gamma: float, gae_lambda: float) -> list[ReplayMinibatch]:
        """
        Gets minibatches from the replay memory, and resets the memory
        """
        with t.inference_mode():
            next_value = self.critic(self.next_obs).flatten()
        minibatches = self.memory.get_minibatches(next_value, self.next_terminated, gamma, gae_lambda)
        self.memory.reset()
        return minibatches


def calc_clipped_surrogate_objective(
    probs: Categorical,
    mb_action: Int[Tensor, "minibatch_size"],
    mb_advantages: Float[Tensor, "minibatch_size"],
    mb_logprobs: Float[Tensor, "minibatch_size"],
    clip_coef: float,
    eps: float = 1e-8,
) -> Float[Tensor, ""]:
    """Return the clipped surrogate objective, suitable for maximisation with gradient ascent.

    probs:
        a distribution containing the actor's unnormalized logits of shape (minibatch_size, num_actions)
    mb_action:
        what actions actions were taken in the sampled minibatch
    mb_advantages:
        advantages calculated from the sampled minibatch
    mb_logprobs:
        logprobs of the actions taken in the sampled minibatch (according to the old policy)
    clip_coef:
        amount of clipping, denoted by epsilon in Eq 7.
    eps:
        used to add to std dev of mb_advantages when normalizing (to avoid dividing by zero)
    """
    assert mb_action.shape == mb_advantages.shape == mb_logprobs.shape
    logits_diff = probs.log_prob(mb_action) - mb_logprobs

    prob_ratio = t.exp(logits_diff)

    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + eps)

    non_clipped = prob_ratio * mb_advantages
    clipped = t.clip(prob_ratio, 1 - clip_coef, 1 + clip_coef) * mb_advantages

    return t.minimum(non_clipped, clipped).mean()


def calc_value_function_loss(
    values: Float[Tensor, "minibatch_size"], mb_returns: Float[Tensor, "minibatch_size"], vf_coef: float
) -> Float[Tensor, ""]:
    """Compute the value function portion of the loss function.

    values:
        the value function predictions for the sampled minibatch (using the updated critic network)
    mb_returns:
        the target for our updated critic network (computed as `advantages + values` from the old network)
    vf_coef:
        the coefficient for the value loss, which weights its contribution to the overall loss. Denoted by c_1 in the paper.
    """
    assert values.shape == mb_returns.shape

    return vf_coef * (values - mb_returns).pow(2).mean()


def calc_entropy_bonus(dist: Categorical, ent_coef: float):
    """Return the entropy bonus term, suitable for gradient ascent.

    dist:
        the probability distribution for the current policy
    ent_coef:
        the coefficient for the entropy loss, which weights its contribution to the overall objective function. Denoted by c_2 in the paper.
    """
    return ent_coef * dist.entropy().mean()



class PPOScheduler:
    def __init__(self, optimizer: Optimizer, initial_lr: float, end_lr: float, total_phases: int):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.end_lr = end_lr
        self.total_phases = total_phases
        self.n_step_calls = 0

    def step(self):
        """Implement linear learning rate decay so that after `total_phases` calls to step, the learning rate is end_lr.

        Do this by directly editing the learning rates inside each param group (i.e. `param_group["lr"] = ...`), for each param
        group in `self.optimizer.param_groups`.
        """
        self.n_step_calls += 1
        frac = self.n_step_calls / self.total_phases
        assert frac <= 1
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.initial_lr + frac * (self.end_lr - self.initial_lr)

def make_optimizer(
    actor: nn.Module, critic: nn.Module, total_phases: int, initial_lr: float, end_lr: float = 0.0
) -> tuple[optim.Adam, PPOScheduler]:
    """
    Return an appropriately configured Adam with its attached scheduler.
    """
    optimizer = optim.AdamW(
        itertools.chain(actor.parameters(), critic.parameters()), lr=initial_lr, eps=1e-5, maximize=True
    )
    scheduler = PPOScheduler(optimizer, initial_lr, end_lr, total_phases)
    return optimizer, scheduler

def set_global_seeds(seed):
    """Sets random seeds in several different ways (to guarantee reproducibility)"""
    t.manual_seed(seed)
    t.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    t.backends.cudnn.deterministic = True

class PPOTrainer:
    def __init__(self, args: PPOArgs):
        set_global_seeds(args.seed)
        self.args = args
        self.run_name = f"following__{args.wandb_project_name}__seed{args.seed}__{time.strftime('%Y%m%d-%H%M%S')}"
        self.envs = EnvVector(args.num_envs)

        # Define some basic variables from our environment
        self.num_envs = args.num_envs
        self.action_shape = ()
        self.obs_shape = (NUM_OBS,)

        # Create our replay memory
        self.memory = ReplayMemory(
            self.num_envs,
            self.obs_shape,
            self.action_shape,
            args.batch_size,
            args.minibatch_size,
            args.batches_per_learning_phase,
            args.seed,
        )

        # Create our networks & optimizer
        self.actor, self.critic = get_actor_and_critic()
        self.optimizer, self.scheduler = make_optimizer(self.actor, self.critic, args.total_training_steps, args.lr)

        # Create our agent
        self.agent = PPOAgent(self.envs, self.actor, self.critic, self.memory)

    def rollout_phase(self) -> None:
        for _ in range(self.args.num_steps_per_rollout):
            self.agent.play_step()

    def learning_phase(self) -> None:
        minibatches = self.agent.get_minibatches(self.args.gamma, self.args.gae_lambda)
        for minibatch in minibatches:
            objective_fn = self.compute_ppo_objective(minibatch)
            objective_fn.backward()
            nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) + list(self.critic.parameters()), self.args.max_grad_norm
            )
            self.optimizer.step()
            self.optimizer.zero_grad()
        self.scheduler.step()

    def compute_ppo_objective(self, minibatch: ReplayMinibatch) -> Float[Tensor, ""]:
        logits = self.actor(minibatch.obs)
        dist = Categorical(logits=logits)
        values = self.critic(minibatch.obs).squeeze()

        clipped_surrogate_objective = calc_clipped_surrogate_objective(
            dist, minibatch.actions, minibatch.advantages, minibatch.logprobs, self.args.clip_coef
        )
        value_loss = calc_value_function_loss(values, minibatch.returns, self.args.vf_coef)
        entropy_bonus = calc_entropy_bonus(dist, self.args.ent_coef)

        total_objective_function = clipped_surrogate_objective - value_loss + entropy_bonus
        return total_objective_function

    def train(self) -> None:
        if self.args.use_wandb:
            wandb.init(
                project=self.args.wandb_project_name,
                entity=self.args.wandb_entity,
                name=self.run_name,
                monitor_gym=False,
            )
            wandb.watch([self.actor, self.critic], log="all", log_freq=50)

        for phase in tqdm(range(self.args.total_phases)):
            self.rollout_phase()
            self.learning_phase()

        if self.args.use_wandb:
            wandb.finish()

print("starting...")
args = PPOArgs(use_wandb=False)
trainer = PPOTrainer(args)
trainer.train()


# %%


