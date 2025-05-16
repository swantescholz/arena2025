# %%
import torch as t
from jaxtyping import Bool, Float, Int
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from dataclasses import dataclass

model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

@dataclass
class Args:
    prompt: str = "Once upon a time"
    max_length: int = 100
    temperature: float = 1.0
    skip_top_p: float = 0.3
    skip_bottom_p: float = 0.1

# %%

def custom_sampling(logits: Float[t.Tensor, "n_vocab"], args: Args) -> int:
    # Convert logits to probabilities if needed
    if args.temperature != 1.0:
        logits = logits / args.temperature
    
    probs = t.softmax(logits, dim=-1)
    probs /= probs.sum()

    sorted_probs, sorted_indices = t.sort(probs, descending=True)
    cumsum = t.cumsum(sorted_probs, dim=0)

    mask = (cumsum > args.skip_top_p) & t.cat((t.tensor([True]), (cumsum <= (1 - args.skip_bottom_p))[1:] ), dim=0)
    valid_indices = sorted_indices[mask]
    if valid_indices.numel() == 0:
        print(sum(mask))
        print(cumsum)
        raise ValueError("No valid indices found")

    valid_probs = probs[valid_indices]
    valid_probs = valid_probs / valid_probs.sum()

    sampled = t.multinomial(valid_probs, 1).item()
    return valid_indices[sampled]


print("Generating text...")
args = Args(prompt="Once upon a time", max_length=150, skip_bottom_p=0.1, skip_top_p=0.0, temperature=1.2)

# Encode the prompt
input_ids = tokenizer.encode(args.prompt, return_tensors='pt')[0]

print(f"{tokenizer.decode(input_ids)}", end="", flush=True)

# Generate text
generated = input_ids
with t.no_grad():
    for _ in range(args.max_length):
        # Get model outputs
        outputs = model(generated)
        next_token_logits = outputs.logits[-1, :]
        
        # Sample using our custom function
        next_token = custom_sampling(next_token_logits, args=args)
        
        # Print the new token
        print(f"{tokenizer.decode(next_token)}", end="", flush=True)
        
        # Append to generated sequence
        generated = t.cat((generated, t.tensor([next_token])), dim=-1)
        
        # Stop if we generate the end of sequence token
        if next_token == tokenizer.eos_token_id:
            break

# Decode and return the generated text
print()
