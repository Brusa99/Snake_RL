import os
import torch as th
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


class TDControl:
    """
    Implements in-policy TD-control through linear approximation.
    Algorithms can be chosen between: "SARSA", "expSARSA", "Q-Learning"
    """

    def __init__(self, state_size, action_size, gamma, lr, alg="SARSA"):
        self.gamma = gamma  # discount factor
        self.lr = lr  # learning rate
        self.state_size = state_size  # as a tuple
        self.action_size = action_size

        # TD error
        assert alg in ("SARSA", "expSARSA", "Q-Learning"), \
            "invalid algorithm valid arguments are: SARSA, expSARSA, Q-Learning"
        self.alg = alg

        # approximations of Q-values
        self.values = np.zeros((*self.state_size, self.action_size))

    def train_step(self, state, action, reward, new_state, new_action, game_over):
        """
        Single training step with the SARSA algorithm.
        """
        if self.alg == "SARSA":
            # SARSA
            # Q(s,a) <- Q(s,a) + lr * [ r + Q(s',a') - Q(s,a) ]
            if game_over:
                delta = reward - self.values[(*state, action)]  # value of the terminal state is 0
            else:
                delta = reward + self.gamma * self.values[(*new_state, new_action)] - self.values[(*state, action)]
        if self.alg == "expSARSA":
            # Expected SARSA
            if game_over:
                delta = reward - self.values[(*state, action)]
            else:
                delta = reward + \
                        self.gamma * np.dot(self.values[(*new_state,)], self.get_policy(new_state)) - \
                        self.values[(*state, action)]
        if self.alg == "Q-Learning":
            if game_over:
                delta = reward - self.values[(*state, action)]
            else:
                delta = reward + self.gamma * np.max(self.values[(*new_state,)]) - self.values[(*state, action)]

        # update
        self.values[(*state, action)] += self.lr * delta

    def get_action(self, state, epsilon=0):
        """
        Returns the action with an epsilon-greedy policy.
        For epsilon=0 always returns the best action
        """
        # epsilon probability to take a random action
        if np.random.rand() < epsilon:
            prob_actions = np.ones(self.action_size) / self.action_size
        # else take the best action
        else:
            best_value = np.max(self.values[(*state,)])
            best_actions = (self.values[(*state,)] == best_value)
            prob_actions = best_actions / np.sum(best_actions)
        action = np.random.choice(self.action_size, p=prob_actions)
        return action

    def get_policy(self, state):
        """
        like get_action but returns vector of probabilities for each action
        """
        policy = np.ones(self.action_size) / self.action_size
        best_value = np.max(self.values[(*state,)])
        best_actions = (self.values[(*state,)] == best_value)
        policy += best_actions / np.sum(best_actions)
        return policy


class LinearQNet(nn.Module):
    """
    Feed forward network with one hidden layer.
    Neurons apply linear transformation y = xA^T + b
    """

    def __init__(self, input_size, hidden_size, output_size):
        """
        in our implementation:
        the input size is the dimension of the state space
        the output size is the number of action (which equals the dimension of the action space since we implemented
            as boolean)
        """
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)  # input layer
        self.linear2 = nn.Linear(hidden_size, output_size)  # hidden layer

    def forward(self, x):
        """
        Implementation of the forward function.
        Applies all the layers to the input x through the rectified linear activation function.

        parameters:
            x [th.tensor] : input to which apply the transformation

        In our implementation:
        given a state s it returns a |A|-dim vector corresponding to the value Q(s,a) of each action a, given the state.
        """
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name):
        """
        saves the model to the specified file_name
        """
        th.save(self.state_dict(), file_name)
        return None


class QTrainer:
    """
    implements the Q-Learning algorithm to a linear neural network
    """

    def __init__(self, model, learning_rate, gamma):
        self.lr = learning_rate
        self.model = model
        self.gamma = gamma

        # optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()  # mean squared error

    def train_step(self, state, action, reward, next_state, next_action, game_over):
        """
        step of the training phase
        next_action is not used in the algorithm, it's present to make it compatible
        """
        # convert everything to tensors
        state = th.tensor(np.array(state), dtype=th.float)
        action = th.tensor(np.array(action), dtype=th.long)
        reward = th.tensor(np.array(reward), dtype=th.float)
        next_state = th.tensor(np.array(next_state), dtype=th.float)

        # function must be able to handle multiple sizes
        # we want the tensors in the form (NUMBER_OF_BATCHES, x)
        # if the tensors are given as multiples as input they already are in the desired form
        if len(state.shape) == 1:
            # If the input is given as a single state/action.. we must convert it
            state = th.unsqueeze(state, 0)
            action = th.unsqueeze(action, 0)
            next_state = th.unsqueeze(next_state, 0)
            reward = th.unsqueeze(reward, 0)
            game_over = (game_over,)

        # implement the Q learning algorithm
        # Q_new += R + gamma * max(Q(S',A') - Q(S,A))
        pred_action = self.model(state)
        target = pred_action.clone()
        for idx in range(len(game_over)):  # the size of all tensors is the same
            Q_new = reward[idx]
            if not game_over[idx]:  # do not apply the algorithm if it led to game over
                Q_new = reward[idx] + self.gamma * th.max(self.model(next_state[idx]))
            target[idx][th.argmax(action).item()] = Q_new

        # optimization step
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred_action)
        loss.backward()
        self.optimizer.step()
