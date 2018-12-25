"""
File to complete. Contains the agents
"""
import numpy as np
import math


class Agent(object):
    """Agent base class. DO NOT MODIFY THIS CLASS
    """

    def __init__(self, mdp):
        super(Agent, self).__init__()
        # Init with a random policy
        self.policy = np.zeros((4, mdp.env.observation_space.n)) + 0.25
        self.mdp = mdp
        self.discount = 0.9

        # Intialize V or Q depends on your agent
        # self.V = np.zeros(self.mdp.env.observation_space.n)
        # self.Q = np.zeros((4, self.mdp.env.observation_space.n))

    def update(self, observation, action, reward):
        # DO NOT MODIFY. This is an example
        pass

    def action(self, observation):
        # DO NOT MODIFY. This is an example
        return self.mdp.env.action_space.sample()


class QLearning(Agent):
    def __init__(self, mdp):
        super(QLearning, self).__init__(mdp)
        self.Q_table = np.zeros((self.mdp.env.nS,self.mdp.env.nA))
        self.alpha = 0.5
        self.epsilon = 0.1 # Value of epsilon for the epsilon-greedy policy

    def update(self, observation, action, reward):
        observation_after_action = self.mdp.env.s
        self.Q_table[observation,action] += self.alpha*(reward+self.discount*np.max(self.Q_table[observation_after_action,:])-self.Q_table[observation,action])


    def make_epsilon_greedy_policy(self, Q, epsilon, nA):
        """
        Creates an epsilon-greedy policy based on a given Q-function and epsilon.

        Args:
            Q: A dictionary that maps from state -> action-values.
                Each value is a numpy array of length nA (see below)
            epsilon: The probability to select a random action . float between 0 and 1.
            nA: Number of actions in the environment.

        Returns:
            A function that takes the observation as an argument and returns
            the probabilities for each action in the form of a numpy array of length nA.

        """
        def policy_fn(observation):
            A = np.ones(nA, dtype=float) * epsilon / nA
            best_action = np.argmax(Q[observation,:])
            A[best_action] += (1.0 - epsilon)
            return A
        return policy_fn

    def action(self, observation):
        policy = self.make_epsilon_greedy_policy(self.Q_table, self.epsilon, self.mdp.env.action_space.n)
        action_proba = policy(observation) # Probabilité de chaque action lorsqu'on est dans l'état 'observation'
        action = np.random.choice(np.arange(len(action_proba)), p=action_proba) # On tire aléatoirement une action. Le poids de chaque action est la probabilité donnée par la policy
        return action
        # return super(QLearning, self).action(observation)


class SARSA(Agent):
    def __init__(self, mdp):
        super(SARSA, self).__init__(mdp)
        self.Q_table = np.zeros((self.mdp.env.nS,self.mdp.env.nA))
        self.alpha = 0.9
        self.epsilon = 0.1 # Value of epsilon for the epsilon-greedy policy
        self.policy = self.make_epsilon_greedy_policy(self.Q_table, self.epsilon, self.mdp.env.action_space.n)


    def update(self, observation, action, reward):
        observation_after_action = self.mdp.env.s
        next_action = self.action(observation_after_action)
        self.Q_table[observation,action] += self.alpha*(reward+self.discount*self.Q_table[observation_after_action,next_action]-self.Q_table[observation,action])


    def make_epsilon_greedy_policy(self, Q, epsilon, nA):
        """
        Creates an epsilon-greedy policy based on a given Q-function and epsilon.

        Args:
            Q: A dictionary that maps from state -> action-values.
                Each value is a numpy array of length nA (see below)
            epsilon: The probability to select a random action . float between 0 and 1.
            nA: Number of actions in the environment.

        Returns:
            A function that takes the observation as an argument and returns
            the probabilities for each action in the form of a numpy array of length nA.

        """
        def policy_fn(observation):
            A = np.ones(nA, dtype=float) * epsilon / nA
            best_action = np.argmax(Q[observation,:])
            A[best_action] += (1.0 - epsilon)
            return A
        return policy_fn


    def action(self, observation):
        action_proba = self.policy(observation) # Probabilité de chaque action lorsqu'on est dans l'état 'observation'
        action = np.random.choice(np.arange(len(action_proba)), p=action_proba) # On tire aléatoirement une action. Le poids de chaque action est la probabilité donnée par la policy
        return action
        # return super(SARSA, self).action(observation)


class ValueIteration:
    def __init__(self, mdp):
        self.mdp = mdp
        self.gamma = 0.9
        self.theta = 0.01

    def optimal_value_function(self):
        """1 step of value iteration algorithm
            Return: State Value V
        """
        # Intialize random V
        V = np.zeros(self.mdp.env.nS)

        while True: # do ...
            delta = 0
            for s in range(self.mdp.env.nS):
                v = V[s]
                Q_s = np.zeros(self.mdp.env.nA)
                for a in range(self.mdp.env.nA):
                    for prob, next_state, reward, _ in self.mdp.env.P[s][a]:
                        """ Les transitions sont déterministes, donc on n'a qu'un seul next_state au plus à explorer par action, un seul tuple au plus dans P[s][a].
                        Pour le cas non déterministe, il faudrait itérer sur tous les tuples de self.mdp.env.P[s][a]
                        """
                        Q_s[a] += prob*(reward+self.gamma*V[next_state])
                V[s] = np.max(Q_s)
                delta = max(delta, np.abs(v - V[s]))
            if delta < self.theta: # until delta < theta
                break
        return V

    def optimal_policy_extraction(self, V):
        """2 step of policy iteration algorithm
            Return: the extracted policy
        """
        policy = np.zeros([self.mdp.env.nS, self.mdp.env.nA])

        for s in range(self.mdp.env.nS):
            Q_s = np.zeros(self.mdp.env.nA)
            for a in range(self.mdp.env.nA):
                for prob, next_state, reward, _ in self.mdp.env.P[s][a]:
                    Q_s[a] += prob*(reward+self.gamma*V[next_state])
            best_action = np.argmax(Q_s)
            policy[s] = [i == best_action for i in range(len(policy[s]))] # Like a one-hot encoding
        return policy

    def value_iteration(self):
        """This is the main function of value iteration algorithm.
            Return:
                final policy
                (optimal) state value function V
        """
        V = self.optimal_value_function()
        policy = self.optimal_policy_extraction(V)

        return policy, V


class PolicyIteration:
    def __init__(self, mdp):
        self.mdp = mdp
        self.gamma = 0.9
        self.theta = 0.01

    def policy_evaluation(self, policy):
        """1 step of policy iteration algorithm
            Return: State Value V
        """
        V = np.zeros(self.mdp.env.nS) # intialize V to 0's

        while True: # do ...
            delta = 0
            for s in range(self.mdp.env.nS):
                v = 0
                for a, action_proba in enumerate(policy[s]):
                    for prob, next_state, reward, _ in self.mdp.env.P[s][a]:
                        v += action_proba*prob*(reward+self.gamma*V[next_state])
                delta = max(delta, np.abs(v - V[s]))
                V[s] = v
            if delta < self.theta: # until delta < theta
                break
        return np.array(V)

    def policy_improvement(self, V, policy):
        """2 step of policy iteration algorithm
            Return: the improved policy
        """
        policy_stable = True
        for s in range(self.mdp.env.nS):
            chosen_a = np.argmax(policy[s]) # this action is chosen according to the policy
            Q_s = np.zeros(self.mdp.env.nA)
            for a in range(self.mdp.env.nA):
                for (proba, next_state, reward, _) in self.mdp.env.P[s][a]:
                    Q_s[a] += proba*(reward + self.gamma*V[next_state]) # We compute the Q function values at state s for all a.
            best_action = np.argmax(Q_s)
            policy[s] = policy[s] = np.eye(self.mdp.env.nA)[best_action] # [i == best_action for i in range(len(policy[s]))] # Like a one-hot encoding
            if best_action != chosen_a: # If the policy is not stable for at least one state...
                policy_stable = False # The policy is not stable at all.
        return policy, policy_stable


    def policy_iteration(self):
        """This is the main function of policy iteration algorithm.
            Return:
                final policy
                (optimal) state value function V
        """
        # Start with a random policy
        policy = np.ones([self.mdp.env.nS, self.mdp.env.nA]) / self.mdp.env.nA
        policy_stable = False
        n_iteration = 0
        while (not policy_stable) and (n_iteration < 1000): # Iteration until the policy becomes stable. It might never happen if, for example, the policy is stuck between two policies that are equally good. We add a constraint on the number of iterations.
            V = self.policy_evaluation(policy)
            policy, policy_stable = self.policy_improvement(V, policy)
            n_iteration += 1
        return policy, V
