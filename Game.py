import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame as pg
import random
from enum import Enum
from collections import namedtuple

BLOCK_SIZE = 20
SPEED = 10


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple("Point", "x y")

# RGB colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
DGREEN = (0, 153, 0)
RED = (255, 0, 0)


class SnakeGame:
    """
    User controlled environment for the snake game
    """

    def __init__(self, w=32, h=26):
        """
        parameters:
            h [int] height
            w [int] width
            both must be expressed in row numbers as they will be multiplied by BLOCK_SIZE
        """
        self.w = w
        self.h = h

        # init display
        self.display = pg.display.set_mode((self.w * BLOCK_SIZE + BLOCK_SIZE,
                                            self.h * BLOCK_SIZE + BLOCK_SIZE
                                            ))
        pg.display.set_caption("Snake")
        self.clock = pg.time.Clock()

        # init game state
        self.direction = Direction.DOWN
        # snake
        self.head = Point(self.w // 2, self.h // 2)  # start in the center
        self.snake = [self.head,
                      Point(self.head.x, self.head.y - 1),
                      Point(self.head.x, self.head.y - 2),
                      ]
        self.score = 0
        self.food = None
        self._place_food()

    def _place_food(self):
        """
        helper function to randomly place food on the grid
        """
        x = random.randint(0, self.w)
        y = random.randint(0, self.h)
        self.food = Point(x, y)

        # check if it is a valid position
        if self.food in self.snake:
            self._place_food()

    def play_step(self):
        """
        iteration of the game
        returns:
            game_over [bool] check if entered an invalid block
            score [int] current score
        """
        # input phase
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                quit()
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_LEFT:
                    self.direction = Direction.LEFT
                elif event.key == pg.K_RIGHT:
                    self.direction = Direction.RIGHT
                elif event.key == pg.K_UP:
                    self.direction = Direction.UP
                elif event.key == pg.K_DOWN:
                    self.direction = Direction.DOWN

        # movement phase
        self._move(self.direction)  # updates the head
        self.snake.insert(0, self.head)
        # if no food is eaten then the last body piece will be popped

        # check reward/block
        game_over = False
        if self._is_collision():
            game_over = True
            return game_over, self.score
        if self.head == self.food:  # ate foot
            self.score += 1
            self._place_food()
        else:
            self.snake.pop()

        # update ui
        self._update_ui()
        self.clock.tick(SPEED)

        # end step
        return game_over, self.score

    def _move(self, direction):
        """
        helper function to move the snake according to provided direction
        """
        # get head coordinates
        x = self.head.x
        y = self.head.y

        # update according to direction
        if direction == Direction.RIGHT:
            x += 1
        elif direction == Direction.LEFT:
            x += -1
        elif direction == Direction.UP:
            y += -1  # display goes from top (0) to bottom (h)
        elif direction == Direction.DOWN:
            y += 1
        self.head = Point(x, y)

    def _is_collision(self):
        """
        helper function to determine if the movement is valid
        """
        # print("x: {}, y: {}, w: {}, h: {}".format(self.head.x, self.head.y, self.w, self.h))
        # print("head:", self.head, "\nbody:", self.snake)

        # only head is checked since body must have already been in a valid position
        # check boundaries
        if self.head.x > self.w or self.head.x < 0 or self.head.y > self.h or self.head.y < 0:
            return True
        # check if it hit itself
        if self.head in self.snake[1:]:  # head is in position 0
            return True
        # else no collision
        return False

    def _update_ui(self):
        """
        helper function to update user interface
        """
        self.display.fill(WHITE)
        # draw snake
        for pt in self.snake:
            pg.draw.rect(self.display,
                         GREEN,
                         pg.Rect(pt.x * BLOCK_SIZE, pt.y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
                         )
        pg.draw.rect(self.display,
                     DGREEN,
                     pg.Rect(self.head.x * BLOCK_SIZE, self.head.y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
                     )

        pg.draw.rect(self.display,
                     RED,
                     pg.Rect(self.food.x * BLOCK_SIZE, self.food.y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
                     )

        # display score
        text = font.render("Score: {}".format(str(self.score)), True, BLACK)
        self.display.blit(text, [0, 0])

        pg.display.flip()


if __name__ == "__main__":
    # init
    pg.init()
    font = pg.font.Font('arial.ttf', 25)
    game = SnakeGame(32, 26)
    # print("-----game started-----")

    # game loop
    while True:
        game_over, score = game.play_step()

        # game over check
        if game_over:
            break

    # terminate
    print("Final score: {}".format(score))
    pg.quit()
