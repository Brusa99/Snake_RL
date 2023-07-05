import torch as th
import random
import numpy as np
from collections import deque

from AIGame import SnakeGameAI, Direction, Point
from model import LinearQNet, QTrainer, TDControl

MAX_MEMORY = 100_000
BATCH_SIZE = 1_000
LEARNING_RATE = .001
DISCOUNT_RATE = 0.9
EPSILON_ZERO = 0.5  # starting exploration parameter


class Agent:
    """
    Agent that learns and implements a policy on the environment AIGame
    """

    def __init__(self):
        """
        n_games [int] : how many games the agent has played
        epsilon [float] : greedy policy control
        gamma [float] : discount rate
        memory [deque] : agent memory
        """
        self.n_games = 0
        self.epsilon = EPSILON_ZERO  # exploration parameter
        self.memory = deque(maxlen=MAX_MEMORY)  # deque auto removes (FIFO) elements when len exceeds max parameter

        self.model = TDControl((2,)*11, 3, DISCOUNT_RATE, LEARNING_RATE)

    def get_state(self, game: SnakeGameAI):
        """
        This function returns State an 11-dimensional tuple:
            [danger_left, danger_straight, danger_right,
            direction_left, direction_right, direction_up, direction_down,
            food_left, food_right, food_up, food_down]
        based on the environment.

        Note that danger is calculated only in the immediate proximity while
        food is considered wrt to all the board.
        """
        food = game.food

        # boolean direction
        dir_up = game.direction == Direction.UP
        dir_down = game.direction == Direction.DOWN
        dir_left = game.direction == Direction.LEFT
        dir_right = game.direction == Direction.RIGHT

        # relative point
        head = game.head
        pt_up = Point(head.x, head.y - 1)
        pt_down = Point(head.x, head.y + 1)
        pt_left = Point(head.x - 1, head.y)
        pt_right = Point(head.x + 1, head.y)

        # state
        state = (
            # danger left
            (dir_up and game.is_collision(pt_left)) or
            (dir_down and game.is_collision(pt_right)) or
            (dir_left and game.is_collision(pt_down)) or
            (dir_right and game.is_collision(pt_up)),
            # danger straight
            (dir_up and game.is_collision(pt_up)) or
            (dir_down and game.is_collision(pt_down)) or
            (dir_left and game.is_collision(pt_left)) or
            (dir_right and game.is_collision(pt_right)),
            # danger right
            (dir_up and game.is_collision(pt_right)) or
            (dir_down and game.is_collision(pt_left)) or
            (dir_left and game.is_collision(pt_up)) or
            (dir_right and game.is_collision(pt_down)),

            # directions
            dir_left,
            dir_right,
            dir_up,
            dir_down,

            # food
            food.x < head.x,  # left
            food.x > head.x,  # right
            food.y < head.y,  # up
            food.y > head.y,  # down
        )
        # print("\nDangers:", *state[0:3], "\nDirections:", *state[3:7], "\nFood:", *state[7:])  # debug
        return np.array(state, dtype=int)  # int dtype to convert in 0,1 matrix

    def get_state_plus(self, game: SnakeGameAI):
        """
        Like get_state but:
        gives coordinate of food in [0,1]
        gives coordinate of head in [0,1]
        """
        # get coordinates
        food = game.food
        food_x = food.x / game.w
        food_y = food.y / game.h

        head = game.head
        head_x = head.x / game.w
        head_y = head.y / game.h

        # boolean direction
        dir_up = game.direction == Direction.UP
        dir_down = game.direction == Direction.DOWN
        dir_left = game.direction == Direction.LEFT
        dir_right = game.direction == Direction.RIGHT

        # relative point
        pt_up = Point(head.x, head.y - 1)
        pt_down = Point(head.x, head.y + 1)
        pt_left = Point(head.x - 1, head.y)
        pt_right = Point(head.x + 1, head.y)

        # state
        state = [
            # danger left
            (dir_up and game.is_collision(pt_left)) or
            (dir_down and game.is_collision(pt_right)) or
            (dir_left and game.is_collision(pt_down)) or
            (dir_right and game.is_collision(pt_up)),
            # danger straight
            (dir_up and game.is_collision(pt_up)) or
            (dir_down and game.is_collision(pt_down)) or
            (dir_left and game.is_collision(pt_left)) or
            (dir_right and game.is_collision(pt_right)),
            # danger right
            (dir_up and game.is_collision(pt_right)) or
            (dir_down and game.is_collision(pt_left)) or
            (dir_left and game.is_collision(pt_up)) or
            (dir_right and game.is_collision(pt_down)),

            # directions
            dir_left,
            dir_right,
            dir_up,
            dir_down,

            # coordinates
            food_x,
            food_y,
            head_x,
            head_y
        ]
        return np.array(state)

    def get_state_matrix(self, game: SnakeGameAI):
        """
        This function returns State pg_size+2-dimensional tuple that consist of:
            a boolean for each cell of the playground that represents if the cell is blocked
            food.x, food.y
            head.x, head.y

        The idea is that given full knowledge of the world the snake won't bottle itself
        """
        # find blocks
        snake = np.zeros((self.pg_x + 2, self.pg_y + 2))
        for pt in game.snake:
            snake[pt.x, pt.y] = 11
        snake = snake[1:-1, 1:-1]

        # find food
        food = np.zeros((self.pg_x + 2, self.pg_y + 2))
        food[game.food.x, game.food.y] = 1
        food = food[1:-1, 1:-1]

        # find head
        head = np.zeros((self.pg_x + 2, self.pg_y + 2))
        head[game.head.x, game.head.y] = 1
        head = head[1:-1, 1:-1]

        # get state
        state = np.concatenate((snake.flatten(), food.flatten(), head.flatten()))
        return state

    def remember(self, state, action, reward, next_state, next_action, game_over):
        """
        Updates the memory of the Agent.
        The Agent has to keep track of the action a he took from state s to move to state s',
        along with the reward r and extra info regarding reaching the terminal state (game_over)
        """
        self.memory.append((state, action, reward, next_state, next_action, game_over))
        return None

    def train_lm(self):
        """
        Trains long memory which refers to memory between different games.
        When game_over state is reached, the agent trains long term memory.
        Also known as experience replay or batch updating.
        Should prevent variance.
        """
        # We use the const BATCH_SIZE to determine how many tuples (S, A, R, S', GO) to use
        # If we have more than necessary we reduce by random sampling
        if len(self.memory) > BATCH_SIZE:
            sample = random.sample(self.memory, BATCH_SIZE)
        else:
            sample = self.memory

        # extract components by type
        states, actions, new_states, rewards, next_action, game_overs = zip(*sample)  # unpacks sample and groups

        # train
        self.model.train_step(states, actions, new_states, rewards, next_action, game_overs)
        return

    def train_sm(self, state, action, reward, next_state, next_action, game_over):
        """
        Trains the model on a single step.
        Also known as online learning.
        """
        self.model.train_step(state, action, reward, next_state, next_action, game_over)

    def get_action(self, state):
        """
        Returns the action decided by the Agent.
        """
        action = self.model.get_action(state, self.epsilon)
        # update epsilon
        self.epsilon = self.epsilon * 0.99
        return action


