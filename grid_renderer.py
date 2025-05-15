import pygame
from jaxtyping import Bool, Float, Int
from pygame.locals import (
    QUIT,
    KEYDOWN,
    K_ESCAPE,
    K_SPACE,
    MOUSEBUTTONDOWN,
)
import torch as t
from torch import Tensor
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

from enum import Enum, auto

Color = Tuple[int, int, int] | None

class UserEvent: pass
class QuitEvent(UserEvent): pass
class ResetEvent(UserEvent): pass
class MoveEvent(UserEvent):
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

# Constants
_CELL_SIZE = 80
_FPS = 60

# Colors
_BLACK = (0, 0, 0)
_WHITE = (255, 255, 255)
_GREY = (64, 64, 64)
_ORANGE = (255, 165, 0)

@dataclass
class GameState:
    grid: Int[Tensor, "h w"]
    winner: Optional[str]

class GridRenderer:
    def __init__(self, width: int, height: int):
        """Initialize the renderer with a given grid size."""
        pygame.init()
        self.width = width
        self.height = height
        self.window_width = self.width * _CELL_SIZE
        self.window_height = self.height * _CELL_SIZE
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Game")
        self.circle_color_mappings: Dict[int, Color] = {
            0: None,
            1: _BLACK,
            2: _WHITE,
        }

    def await_user_input(self) -> UserEvent:
        while True:
            event = pygame.event.wait()
            if event.type == QUIT:
                self.cleanup()
                return QuitEvent()
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    self.cleanup()
                    return QuitEvent()
                elif event.key == K_SPACE:
                    return ResetEvent()
            elif event.type == MOUSEBUTTONDOWN:
                # Get mouse position and convert to grid coordinates
                mouse_pos = pygame.mouse.get_pos()
                clicked_row = mouse_pos[1] // _CELL_SIZE
                clicked_col = mouse_pos[0] // _CELL_SIZE
                if 0 <= clicked_row < self.height and 0 <= clicked_col < self.width:
                    return MoveEvent(clicked_col, clicked_row)
            else:
                pass # Ignore other events

    def render(self, state: GameState) -> None:
        """Render the current game state."""
        self.screen.fill(_GREY)
        for y in range(self.height):
            for x in range(self.width):
                # Draw cell
                rect = pygame.Rect(x * _CELL_SIZE, y * _CELL_SIZE, _CELL_SIZE, _CELL_SIZE)
                pygame.draw.rect(self.screen, _WHITE, rect, 2)
                
                # Draw pieces
                center = (x * _CELL_SIZE + _CELL_SIZE // 2, y * _CELL_SIZE + _CELL_SIZE // 2)
                piece = state.grid[y, x].item()
                if piece == 1:  # MINE
                    pygame.draw.circle(self.screen, _BLACK, center, _CELL_SIZE // 2 - 5)
                elif piece == 2:  # THEIRS
                    pygame.draw.circle(self.screen, _WHITE, center, _CELL_SIZE // 2 - 5)
                    # Draw a black border for white pieces to make them visible
                    # pygame.draw.circle(self.screen, _BLACK, center, _CELL_SIZE // 2 - 5, 2)
        
        # Draw game over message
        if state.winner is not None:
            font = pygame.font.Font(None, 74)
            text = font.render(f"{'Draw!' if state.winner == 'Draw' else state.winner + ' wins!'}", True, _ORANGE)
            text_rect = text.get_rect(center=(self.window_width // 2, self.window_height // 2))
            # Add a dark background for better visibility
            bg_rect = text_rect.copy()
            bg_rect.inflate_ip(20, 20)
            pygame.draw.rect(self.screen, _BLACK, bg_rect)
            self.screen.blit(text, text_rect)
        
        pygame.display.flip()

    def cleanup(self):
        """Clean up pygame resources."""
        pygame.quit()

def main():
    renderer = GridRenderer(10, 10)
    while True:
        renderer.render(GameState(t.randint(0, 3, (10, 10)), "test"))
        event = renderer.await_user_input()
        if isinstance(event, QuitEvent):
            break
            

if __name__ == "__main__":
    main()