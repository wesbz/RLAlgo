"""
File to complete. Contains the agents
"""
import numpy as np
import math


class Agent(object):
    """Agent base class. DO NOT MODIFY THIS CLASS
    """

    def __init__(self, mdp):
        # Init the random policy
        super(Agent, self).__init__()
        self.policy = np.zeros((4, mdp.size[0], mdp.size[1])) + 0.25

        self.mdp = mdp
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        self.discount = 0.9

        self.V = np.zeros(mdp.size)

        # For others: Q values
        self.Q = np.zeros((4, mdp.size[0], mdp.size[1]))

    def update(self):
        # DO NOT MODIFY
        raise NotImplementedError

    def action(self):
        self.last_position = self.mdp.position
        return self.actions[np.random.choice(range(len(self.actions)),
                                             p=self.policy[:, self.last_position[0],
                                                            self.last_position[1]])]

class VEvalTemporalDifferencing(Agent):
    def __init__(self, mdp):
        super(VEvalTemporalDifferencing, self).__init__(mdp)
        self.learning_rate = 0.1

    def update(self):
        # If the last episode finish, the reward is the one from the last episode
        reward = self.mdp.get_last_reward()
        self.V[self.last_position] += self.learning_rate * \
            (reward + self.discount *
             self.V[self.mdp.position] - self.V[self.last_position])

    def action(self):
        # YOU CAN MODIFY
        return super(VEvalTemporalDifferencing, self).action()

class VEvalMonteCarlo(Agent):
    def __init__(self, mdp):
        super(VEvalMonteCarlo, self).__init__(mdp)

    def update(self):
        # TO IMPLEMENT
        raise NotImplementedError

    def action(self):
        # YOU CAN MODIFY
        return super(VEvalMonteCarlo, self).action()


class ValueIteration(Agent):
    def __init__(self, mdp):
        super(ValueIteration, self).__init__(mdp)

    def update(self):
        # TO IMPLEMENT
        reward = self.mdp.get_last_reward()
        self.V[self.last_position] = reward + self.discount * 99999999
        raise NotImplementedError

    def action(self):
        # YOU CAN MODIFY
        return super(ValueIteration, self).action()


class PolicyIteration(Agent):
    def __init__(self, mdp):
        super(PolicyIteration, self).__init__(mdp)

    def update(self):
        # TO IMPLEMENT
        raise NotImplementedError

    def action(self):
        # YOU CAN MODIFY
        return super(PolicyIteration, self).action()


class QLearning(Agent):
    def __init__(self, mdp):
        super(QLearning, self).__init__(mdp)

    def update(self):
        # TO IMPLEMENT
        raise NotImplementedError

    def action(self):
        # YOU CAN MODIFY
        return super(QLearning, self).action()


class SARSA(Agent):
    def __init__(self, mdp):
        super(SARSA, self).__init__(mdp)

    def update(self):
        # TO IMPLEMENT
        raise NotImplementedError

    def action(self):
        # YOU CAN MODIFY
        return super(SARSA, self).action()
