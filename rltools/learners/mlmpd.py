import numpy as np

from rltools.learners import Learner

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
        ''' Incrementally update value estimates after observing a transition between states '''

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
        for (state1, action), s_list in self._transition_history.items():
            s_count = float(len(s_list))
            for state2 in s_list:
                T[action, state1, state2] += 1 / s_count

        # Compute the estimated reward value
        for (state1, action, state2), r_list in self._reward_history.items():
            r_count = len(r_list)
            for reward in r_list:
                R[action, state1, state2] += float(reward) / r_count

        # Combine with the prior belief in accordance to the current learning rate
        for action in range(n_actions):
            for state1 in range(n_states):
                for state2 in range(n_states):

                    if (state1, action, state2) in self._transition_prior:
                        T[action, state1, state2] = \
                            T[action, state1, state2] * self._learning_rate + \
                                self._transition_prior.get((state1, action, state2), 0) * \
                                (1.0 - self._learning_rate)
                    if self._max_history_len > 0 and \
                        len(self._transition_history) > self._max_history_len:
                        self._transition_history[(state1, action)] = []
                        self._transition_prior[(state1, action, state2)] = \
                            T[action, state1, state2]

                    if (state1, action, state2) in self._reward_prior:
                        R[action, state1, state2] = \
                            R[action, state1, state2] * self._learning_rate + \
                                self._reward_prior.get((state1, action, state2), 0) * \
                                (1.0 - self._learning_rate)
                    if self._max_history_len > 0 and \
                        len(self._reward_history) > self._max_history_len:
                        self._reward_history[(state1, action, state2)] = []
                        self._reward_prior[(state1, action, state2)] = R[action, state1, state2]


        # Make sure that the transition matrix is stochastic by providing uniform weight to unseen
        # transition rows
        for action in range(n_actions):
            for state in range(n_states):
                diff = 1.0 - T[action, state].sum()
                if abs(diff) > 1E-3:
                    T[action, state] += diff / n_states

        return T, R


    def converge(self, atol=1E-3, max_iter=1000, max_time=0):
        ''' Train over already fitted data over and over until convergence '''
        T, R = self._calc_matrices()

        V = self._value_iteration(T, R, atol, max_iter, max_time)

        n_states = max(self.get_states()) + 1
        n_actions = max(self.get_actions()) + 1
        for action in range(n_actions):
            for state in range(n_states):
                self._set_value(state, action, V[action, state])
