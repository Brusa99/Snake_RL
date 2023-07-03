import torch as th
import random
import numpy as np
from collections import deque

from AIGame import SnakeGameAI, Direction, Point
from model import LinearQNet, QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 1_000
LEARNING_RATE = .001
EPSILON_ZERO = 80  # starting exploration parameter


class SuperAgent:
    """
    Agent that learns and implements a policy on the environment AIGame
    """

    def __init__(self, pg_x, pg_y):
        """
        n_games [int] : how many games the agent has played
        epsilon [float] : greedy policy control
        gamma [float] : discount rate
        memory [deque] : agent memory
        """
        self.pg_x = pg_x
        self.pg_y = pg_y
        self.n_games = 0
        self.epsilon = EPSILON_ZERO  # exploration parameter
        self.gamma = 0.9  # must be in (0,1)
        self.memory = deque(maxlen=MAX_MEMORY)  # deque auto removes (FIFO) elements when len exceeds max parameter

        # as a model we take a ffnn, input size is |S|, output size is |A|.
        pg_size = pg_x * pg_y
        self.model = LinearQNet(3 * pg_size, (pg_size*10)//8 * 8, 4)  # hidden layer size may be changed
        self.trainer = QTrainer(model=self.model, learning_rate=LEARNING_RATE, gamma=self.gamma)

    def get_state(self, game: SnakeGameAI):
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
        states, actions, new_states, rewards, game_overs = zip(*sample)  # unpacks sample and groups

        # train
        self.trainer.train_step(states, actions, new_states, rewards, game_overs)
        return

    def train_sm(self, state, action, reward, next_state, game_over):
        """
        Trains the model on a single step.
        Also known as online learning.
        """
        self.trainer.train_step(state, action, reward, next_state, game_over)

    def get_action(self, state):
        """
        Returns the action decided by the Agent.
        The chance of doing a random action (EXPLORATION) is decided by the class attribute epsilon
        """
        action = [0, 0, 0, 0]

        # update epsilon
        # method 1
        # self.epsilon = EPSILON_ZERO - self.n_games  # note that epsilon can become negative => no more exploration
        # method 2
        self.epsilon = self.epsilon*0.999

        # randomly take action with probability epsilon
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 3)
            action[move] = 1

        # or choose best action
        else:
            state = th.tensor(state, dtype=th.float)  # convert to tensor for the nn model
            prediction = self.model(state)  # outputs raw value
            move = th.argmax(prediction).item()
            action[move] = 1

        return action
