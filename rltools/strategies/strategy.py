import random

class Strategy(object):

    def __init__(self, learner, valid_actions=None):
        self.learner = learner
        self.valid_actions = valid_actions or []


    def fit(self, X):  # pylint: disable=invalid-name
        self.learner.fit(X)


    def init_episode(self, *args, **kwargs):
        self.learner.init_episode(*args, **kwargs)


    def converge(self, *args, **kwargs):
        self.learner.converge(*args, **kwargs)


    def _parse_valid_actions(self, valid_actions=None):
        # Valid actions chosen, by priority:
        # 1. action set passed to this function
        # 2. action set passed to initialization of strategy object
        # 3. all actions observed by learner
        return valid_actions or self.valid_actions or list(self.learner.get_actions())


    def _greedy_policy(self, state, valid_actions=None, value_fn=None):
        valid_actions = self._parse_valid_actions(valid_actions)
        value_fn = value_fn or self.learner.val

        # Pick the action with highest value
        action_values = [(value_fn(state, action), action) for action in valid_actions]
        sorted_values = sorted(action_values, key=lambda x: -x[0])

        # In case of a tie, choose randomly
        atol = 1E-3
        equal_values = [v for v in sorted_values if abs(v[0] - sorted_values[0][0]) < atol]
        return equal_values[random.randint(0, len(equal_values) - 1)][1]



    def policy(self, state, valid_actions=None):
        return self._greedy_policy(state, valid_actions)
