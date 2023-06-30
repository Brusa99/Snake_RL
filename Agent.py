import torch as th
import random
import numpy as np
from collections import deque

from AIGame import SnakeGameAI, Direction, Point

MAX_MEMORY = 100_000
BATCH_SIZE = 1_000
LEARNING_RATE = .001
EPSILON_ZERO = 50  # starting exploration parameter


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
        self.epsilon = 0  # exploration parameter
        self.gamma = 0
        self.memory = deque(maxlen=MAX_MEMORY)  # deque auto removes (FIFO) elements when len exceeds max parameter

        self.model = None
        self.trainer = None

    @staticmethod
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

            # food
            food.x < head.x,  # left
            food.x > head.x,  # right
            food.y < head.y,  # up
            food.y > head.y,  # down
        ]
        return np.array(state, dtype=int)  # int dtype to convert in 0,1 matrix

    def remember(self, state, action, reward, next_state, game_over):
        """
        Updates the memory of the Agent.
        The Agent has to keep track of the action a he took from state s to move to state s',
        along with the reward r and extra info regarding reaching the terminal state (game_over)
        """
        self.memory.append((state, action, reward, next_state, game_over))
        return None

    def train_lm(self):
        """
        Trains long memory which refers to memory between different games.
        When game_over state is reached, the agent trains long term memory.
        Also known as experience replay.

        Note that this function doesn't need any parameters since it accesses the agent's memory.
        """
        # We use the const BATCH_SIZE to determine how many tuples (S, A, R, S', GO) to use
        # If we have more than necessary we reduce by random sampling
        if len(self.memory) > BATCH_SIZE:
            sample = random.sample(self.memory, BATCH_SIZE)
        else:
            sample = self.memory

        # extract components by type
        states, actions, new_states, rewards, game_overs = zip(*sample)  # unpacks sample and groups

        # train
        self.trainer.train_step(states, actions, new_states, rewards, game_overs)
        return

    def train_sm(self, state, action, reward, next_state, game_over):
        """
        Trains short memory, which refers to the current game.
        """
        self.trainer.train_step(state, action, reward, next_state, game_over)

    def get_action(self, state):
        """
        Returns the action decided by the Agent.
        The chance of doing a random action (EXPLORATION) is decided by the class attribute epsilon
        """
        action = [0, 0, 0]

        # update epsilon
        self.epsilon = EPSILON_ZERO - (self.n_games / 2)  # note that epsilon can become negative => no more exploration

        # randomly take action with probability epsilon
        if random.randint(0, 100) < self.epsilon:
            move = random.randint(0, 2)
            action[move] = 1

        # or choose best action
        else:
            state = th. tensor(state, dtype=th.float)  # convert to tensor for the nn model
            prediction = self.model(state)  # outputs raw value
            move = th.argmax(prediction).item()
            action[move] = 1
        return action