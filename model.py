import os
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
            # Q(s,a) <- Q(s,a) + lr * [ r + y * Q(s',a') - Q(s,a) ]
            if game_over:
                delta = reward - self.values[(*state, action)]  # value of the terminal state is 0
            else:
                delta = reward + self.gamma * self.values[(*new_state, new_action)] - self.values[(*state, action)]
        if self.alg == "expSARSA":
            # Expected SARSA
            # Q(s,a) <- Q(s,a) + lr * [ r + y * SUM_a' pi(a'|s')Q(s',a') - Q(s,a) ]
            if game_over:
                delta = reward - self.values[(*state, action)]
            else:
                delta = reward + \
                        self.gamma * np.dot(self.values[(*new_state,)], self.get_policy(new_state)) - \
                        self.values[(*state, action)]
        if self.alg == "Q-Learning":
            # Q-Learning
            # Q(s,a) <- Q(s,a) + lr * [ r + y * max_a'{Q(s',a')} ]
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

    def save(self, file):
        """
        saves the matrix in a .npy format
        """
        np.save(file, self.values)
