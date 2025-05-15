import os
import random
import sys
from pathlib import Path
from typing import TypeAlias, List, Tuple, Optional
from enum import Enum, auto

import pygame
from jaxtyping import Bool, Float, Int
import torch as t
from torch import Tensor

from abc import ABC, abstractmethod
from utils import *

class GameResult(Enum):
    ONGOING = auto()
    WON = auto()
    LOST = auto()
    DRAW = auto()

EMPTY = 0
MINE = 1
THEIRS = 2

class DiscreteGame(ABC):
    """
    Every action is an int.
    """

    @abstractmethod
    def action_space_size(self) -> int: pass

    @abstractmethod
    def compute_game_result(self) -> GameResult: pass

    @abstractmethod
    def step(self, action: int): pass

    @abstractmethod
    def valid_moves(self) -> Int[Tensor, "n_valid_moves"]: pass

    @abstractmethod
    def render(self): pass

    @abstractmethod
    def reset(self): pass

class Connect4Game(DiscreteGame):
    SIZE: int = 9
    LINE_LENGTH: int = 4

    def __init__(self):
        super().__init__()
        self.grid = t.zeros((self.SIZE, self.SIZE), dtype=t.int8)
        self.step_count = 0
        self.player_turn = True
        self.game_over = False
        self.winner = None

    def action_space_size(self) -> int:
        return self.SIZE ** 2

    def compute_game_result(self) -> GameResult:
        for player in [MINE, THEIRS]:
            if check_line_exists(self.grid, player, self.LINE_LENGTH):
                return GameResult.WON if player == MINE else GameResult.LOST
        if self.valid_moves().size == 0:
            return GameResult.DRAW
        return GameResult.ONGOING

    def valid_moves(self) -> Int[Tensor, "n_valid_moves"]:
        return t.where(self.grid.flatten() == EMPTY)[0]
    



# Piece values
EMPTY = 0
PLAYER = 1  # Black
AI = 2      # White








# %%




# %%




# %%




# %%




# %%




# %%




# %%








