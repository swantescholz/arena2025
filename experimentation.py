# %%
from abc import ABC, abstractmethod

class Animal(ABC):
    @abstractmethod
    def make_sound(self, x: int) -> bool:
        pass

class Dog(Animal):
    def make_sound(self):
        print("Woof!")

class Cat(Animal):
    def make_soundss(self):1
        print("Meow!")

# Dog() works, but this fails:
# class Cat(Animal): pass  ❌ TypeError

Dog().make_sound()
Cat().make_sound()

# %%
from enum import Enum, auto

class Color(Enum):
    RED = auto()
    GREEN = auto()
    BLUE = auto()

print(Color.RED.value == 1)
print(Color.RED.value)

# %%

import torch

mask = torch.tensor([True, False, True, False, False, True])
indices = torch.nonzero(mask, as_tuple=True)[0]

print(torch.where(mask))  # tensor([0, 2, 5])

# %%





# %%





# %%





# %%





# %%





# %%





# %%





# %%





# %%





# %%





# %%





# %%





# %%





# %%





# %%





# %%





# %%





# %%





# %%





# %%





# %%





