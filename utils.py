# %%
import os
import random
import sys
from pathlib import Path
from typing import TypeAlias

import einops
from einops import einsum
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch as t
from abc import ABC, abstractmethod
from jaxtyping import Bool, Float, Int
from torch import Tensor

def check_line_exists(grid: Int[Tensor, "h w"], target_value: int, line_length: int) -> bool:
    # Convert grid to CPU for numpy-like operations
    grid = (grid == target_value)
    h, w = grid.shape
    lm1 = line_length - 1
    
    # Check horizontal
    for i in range(h):
        for j in range(w - lm1):
            if t.all(grid[i, j:j+line_length]):
                return True
    
    # Check vertical
    for i in range(h - lm1):
        for j in range(w):
            if t.all(grid[i:i+line_length, j]):
                return True
    
    # Check diagonal (top-left to bottom-right)
    for i in range(h - lm1):
        for j in range(w - lm1):
            if t.all(Tensor([grid[i+k, j+k] for k in range(line_length)])):
                return True
    
    # Check diagonal (top-right to bottom-left)
    for i in range(h - lm1):
        for j in range(lm1, w):
            if t.all(Tensor([grid[i+k, j-k] for k in range(line_length)])):
                return True
    
    return False


def assert_equal(a, b, custom_prefix: str = "assert_equal failed"):
    if a != b:
        raise AssertionError(f"{custom_prefix}: {a} != {b}")