import random

class EpsilonGreedyStrategy(object):

    def __init__(self, learner, valid_actions=None):
        self.learner = learner
        self.valid_actions = valid_actions or []
        self.prediction_count = 0


    def _epsilon(self, t):
        return 0.2 / (1 + t/1000)


    def fit(self, X):
        self.learner.fit(X)


    def init_episode(self):
        self.learner.init_episode()


    def policy(self, state, valid_actions=None):
        self.prediction_count += 1

        # Valid actions chosen, by priority:
        # 1. action set passed to this function
        # 2. action set passed to initialization of strategy object
        # 3. all actions observed by learner
        valid_actions = valid_actions or self.valid_actions or list(self.learner.get_actions())
        
        # Every 1/e times, pick random action
        roll = random.random()
        if roll < self._epsilon(self.prediction_count):
            return valid_actions[random.randint(0, len(valid_actions) - 1)]

        # Otherwise, pick the action with highest value
        action_values = [(self.learner.val(state, action), action) for action in valid_actions]
        return sorted(action_values, key=lambda x: x[0])[-1][1]
