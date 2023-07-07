import torch as th
import numpy as np

from AIGame import SnakeGameAI, Direction, Point
from model import TDControl

LEARNING_RATE = .001
DISCOUNT_RATE = 0.9
EPSILON_ZERO = 0.5  # starting exploration parameter


class Agent:
    """
    Agent that learns and implements a policy on the environment AIGame
    """

    def __init__(self, alg="SARSA"):
        """
        n_games [int] : how many games the agent has played
        epsilon [float] : greedy policy control
        gamma [float] : discount rate
        memory [deque] : agent memory
        """
        self.n_games = 0
        self.epsilon = EPSILON_ZERO  # exploration parameter

        self.model = TDControl((2,) * 11, 3, DISCOUNT_RATE, LEARNING_RATE, alg=alg)  # classic state

        # To use whole map for states, use this model (also have to change in train driver which agent_type + get_state)
        # moreover it won't work since max dimension for a ndarray is 32
        # self.model = TDControl((4,) * 64, 4, DISCOUNT_RATE, LEARNING_RATE, alg=alg)  # material state

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

    def get_state_matrix(self, game: SnakeGameAI):
        """
        This function return an alternative state.
        The state is given by the whole map/playground.
        Each entry of the state can assume one of 4 values:
            0 : empty
            1 : block
            2 : head
            3 : food
        """
        world = np.zeros((self.pg_x + 2, self.pg_y + 2))

        # find blocks
        for pt in game.snake:
            world[pt.x, pt.y] = 1

        # find food and head
        world[game.food.x, game.food.y] = 3
        world[game.head.x, game.head.y] = 2

        state = world[1:-1, 1:-1]
        return state

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

    def get_action_greedy(self, state):
        """
        Returns the greedy action chosen by the agent given a state
        """
        return self.model.get_action(state)