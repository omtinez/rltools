import random
import pickle
import numpy as np

from rltools.strategies import Strategy
from rltools.learners import Learner, MLMDP, TemporalDifferenceLearner

class RMaxStrategy(Strategy):

    def __init__(self, learner, valid_actions=None, confidence_fn=None):
        Strategy.__init__(self, learner, valid_actions)

        # Initialize ourselves as an instance of a learner so we can keep track of the states and
        # populate transition counts without relying on the learner
        Learner.__init__(self)  

        self._confidence = confidence_fn if confidence_fn else lambda c: 1.0 - (1.0 / max(1, c) ** 0.5)
        self._transition_count = {}
        self._max_reward_seen = 0

        # Internally, this strategy keeps its own maximum-likelihood MDP
        #self.optimistic_learner = TemporalDifferenceLearner(l=0)
        self.optimistic_learner = MLMDP(normalize_count=0)


    def _learn_incr(self, prev_state, action, reward, curr_state):
        key = (prev_state, action)
        self._transition_count[key] = self._transition_count.get(key, 0) + 1


    def fit(self, X):
        Learner.fit(self, X)
        Strategy.fit(self, X)
        self.optimistic_learner.fit(X)


    def policy(self, state, valid_actions=None):
        valid_actions = self._parse_valid_actions(valid_actions)

        # Early exit: only one action, nothing to choose
        if len(valid_actions) == 1:
            return valid_actions[0]

        # Save the current state of the optimistic learner
        p_optimistic_learner = pickle.dumps(self.optimistic_learner)
        
        # Update the value of the states estimated by learner based on our confidence and
        # optimistic expectation of unseen transitions
        num_updates = 0
        confidence_atol = 1E-3
        n_states = max(self.optimistic_learner.get_states()) + 1
        n_actions = max(self.optimistic_learner.get_actions()) + 1
        for a in range(n_actions):
            for s in range(n_states):
                confidence = self._confidence(
                    self._transition_count.get((s, a), 0))
                if confidence < 1.0 - confidence_atol:
                    num_updates += 1
                    self.optimistic_learner._last_state = s
                    self.optimistic_learner.fit(
                        (s, a,  (1.0 - confidence) * self._max_reward_seen))
       
        # If we are confident enough about all values, no need to solve the MDP
        if num_updates > 0:
            self.optimistic_learner.converge()
            value_fn = self.optimistic_learner.val
        else:
            value_fn = None

        # Restore the state of the optimistic learner prior to messing with the rewards
        self.optimistic_learner = pickle.loads(p_optimistic_learner)

        # Simle greedy policy
        return self._greedy_policy(state, valid_actions, value_fn)
