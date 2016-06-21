import numpy as np
from mdptoolbox.mdp import PolicyIteration

from .learner import Learner

class MLMDP(Learner):
    '''
    Maximum likelihood markov decision process. Learner that builds a representation of an MDP
    based on the maximum likelihood estimate for the reward and transition functions from the data
    observed.
    '''
    
    def __init__(self, discount_factor=0.86, learning_rate=0.99, normalize_count=None):
        Learner.__init__(self, discount_factor, learning_rate)
        self._transition_count = {}
        self._transition_history = {}
        self._reward_history = {}
        self.normalize_count = 1 if normalize_count is None else normalize_count
        self.normalize_count_double = normalize_count is None

        self._max_history_len = 0
        self._transition_prior = {}
        self._reward_prior = {}


    def _learn_incr(self, prev_state, action, reward, curr_state):
        ''' Incrementally update the value estimates after observing a transition between states '''
        
        key1 = (prev_state, action, curr_state)
        key2 = (prev_state, action)
        self._transition_count[key1] = self._transition_count.get(key1, 0) + 1
        self._transition_history[key2] = self._transition_history.get(key2, []) + [curr_state]
        self._reward_history[key1] = self._reward_history.get(key1, []) + [reward]

        if self.normalize_count > 0 and self._transition_count[key1] >= self.normalize_count:
            self.converge()
            self._transition_count = {}
            if self.normalize_count_double:
                self.normalize_count *= 2
                        
            
    def _calc_matrices(self):
        n_states = max(self.get_states()) + 1
        n_actions = max(self.get_actions()) + 1
        T = np.zeros((n_actions, n_states, n_states))
        R = np.zeros((n_actions, n_states, n_states))

        # Compute the estimated transition probabilities
        for (s1, a), s_list in self._transition_history.items():
            s_count = float(len(s_list))
            for s2 in s_list:
                T[a, s1, s2] += 1 / s_count

        # Compute the estimated reward value
        for (s1, a, s2), r_list in self._reward_history.items():
            r_count = len(r_list)
            for r in r_list:
                R[a, s1, s2] += float(r) / r_count

        # Combine with the prior belief in accordance to the current learning rate
        for a in range(n_actions):
            for s1 in range(n_states):
                for s2 in range(n_states):

                    if (s1, a, s2) in self._transition_prior:
                        T[a, s1, s2] = T[a, s1, s2] * self._learning_rate + \
                            self._transition_prior.get((s1, a, s2), 0) * (1.0 - self._learning_rate)
                    if self._max_history_len > 0 and len(self._transition_history) > self._max_history_len:
                        self._transition_history[(s1, a)] = []
                        self._transition_prior[(s1, a, s2)] = T[a, s1, s2]

                    if (s1, a, s2) in self._reward_prior:
                        R[a, s1, s2] = R[a, s1, s2] * self._learning_rate + \
                            self._reward_prior.get((s1, a, s2), 0) * (1.0 - self._learning_rate)
                    if self._max_history_len > 0 and len(self._reward_history) > self._max_history_len:
                        self._reward_history[(s1, a, s2)] = []
                        self._reward_prior[(s1, a, s2)] = R[a, s1, s2]

                        
        # Make sure that the transition matrix is stochastic by providing uniform weight to unseen transition rows
        for a in range(n_actions):
            for s in range(n_states):
                diff = 1.0 - T[a, s].sum()
                if abs(diff) > 1E-3:
                    T[a, s] += diff / n_states
                
        return T, R
            
        
    def converge(self, atol=1E-3, max_iter=1000, max_time=0):
        ''' Train over already fitted data over and over until convergence '''
        T, R = self._calc_matrices()

        V = self._value_iteration(T, R, atol, max_iter, max_time)

        n_states = max(self.get_states()) + 1
        n_actions = max(self.get_actions()) + 1
        for a in range(n_actions):
            for s in range(n_states):
                self._set_value(s, a, V[a, s])
