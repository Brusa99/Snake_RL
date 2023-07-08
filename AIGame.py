import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame as pg
import numpy as np
import random
from enum import Enum
from collections import namedtuple

BLOCK_SIZE = 40
SPEED = 40
MAX_ITER = 100


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


# actions 3

LEFT_TURN = 0
STRAIGHT = 1
RIGHT_TURN = 2

# actions 4

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

Point = namedtuple("Point", "x y")

# RGB colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
DGREEN = (0, 153, 0)
RED = (255, 0, 0)

pg.init()
font = pg.font.Font('arial.ttf', 25)


class SnakeGameAI:
    """
    AI controlled environment for the snake game
    """

    def __init__(self, w=32, h=26, agent_type=3):
        """
        parameters:
            h [int] height
            w [int] width
            both must be expressed in row numbers as they will be multiplied by BLOCK_SIZE
        """
        self.w = w
        self.h = h

        assert agent_type in (3, 4), "invalid agent type"
        self.agent_type = agent_type

        # init environment
        self.display = pg.display.set_mode((self.w * BLOCK_SIZE + BLOCK_SIZE,
                                            self.h * BLOCK_SIZE + BLOCK_SIZE
                                            ))
        pg.display.set_caption("Snake AI")
        self.clock = pg.time.Clock()
        self.reset()

    def reset(self):
        """
        Reset playground
        """
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
        self.iteration = 0

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

    def play_step(self, action):
        """
        iteration of the AI game
        inputs:
            action in [turn left, forward, turn right]
        returns:
            reward [int] : +1 for eating, -1 for game over, 0 else
            game_over [bool] : determines if failed
            score [int] : game score
        """
        # update iterations
        self.iteration += 1

        # user quit
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                quit()

        # movement phase
        if self.agent_type == 3:
            self._move3(action)  # updates the head
        else:
            self._move4(action)  # updates the head
        self.snake.insert(0, self.head)
        # if no food is eaten then the last body piece will be popped

        # check reward/block
        reward = -0.1  # might try -0.001
        game_over = False

        # collision check
        if self.is_collision():
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # if snake is doing nothing for too long => game over
        if self.iteration > MAX_ITER * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # reward
        if self.head == self.food:  # ate foot
            self.score += 1
            reward = 15
            self._place_food()
        else:
            self.snake.pop()

        # update ui
        self._update_ui()
        self.clock.tick(SPEED)

        # end step
        return reward, game_over, self.score

    def _move3(self, action):
        """
        helper function to move the snake according to 3-action [turn left, straight, turn right]
        Determine in which direction the snake should move.
        """
        # determine direction based on action ~ maps action to direction
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        if np.array_equal(action, LEFT_TURN):
            new_dir = clock_wise[(idx - 1) % 4]
        elif np.array_equal(action, STRAIGHT):
            new_dir = clock_wise[idx]
        else:  # np.array_equal(action, RIGHT_TURN):
            new_dir = clock_wise[(idx + 1) % 4]

        self.direction = new_dir

        # get head coordinates
        x = self.head.x
        y = self.head.y

        # update according to direction
        if self.direction == Direction.RIGHT:
            x += 1
        elif self.direction == Direction.LEFT:
            x += -1
        elif self.direction == Direction.UP:
            y += -1  # display goes from top (0) to bottom (h)
        elif self.direction == Direction.DOWN:
            y += 1
        self.head = Point(x, y)
        return

    def _move4(self, action):
        """
        helper function to move the snake according to 4-action [up, down, left, right]
        """
        # get head coordinates
        x = self.head.x
        y = self.head.y

        # update according to action
        if np.array_equal(action, UP):
            y += -1
        elif np.array_equal(action, DOWN):
            y += 1
        elif np.array_equal(action, LEFT):
            x += -1
        elif np.array_equal(action, RIGHT):
            x += 1

        self.head = Point(x, y)
        return

    def is_collision(self, pt=None):
        """
        helper function to determine if position pt is valid
        """
        if pt is None:
            pt = self.head

        # check boundaries
        if pt.x > self.w or pt.x < 0 or pt.y > self.h or pt.y < 0:
            return True
        # check if it hit itself
        if pt in self.snake[1:]:  # head is in position 0
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
