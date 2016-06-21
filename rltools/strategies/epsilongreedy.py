import random

from rltools.strategies import Strategy

class EpsilonGreedyStrategy(Strategy):
    '''
    Very simple strategy in which the best action is chosen by a greedy policy with probability of
    `(1 - epsilon)` times and a random action is chosen with probability `epsilon` times. Epsilon
    is gradually decayed as more episodes are learned.
    '''

    def __init__(self, learner, valid_actions=None):
        Strategy.__init__(self, learner, valid_actions)


    def _epsilon(self, episode):
        return 0.2 / (1 + episode / 1E3)


    def policy(self, state, valid_actions=None):
        valid_actions = self._parse_valid_actions(valid_actions)

        # Every 1/e times, pick random action
        roll = random.random()
        if roll < self._epsilon(self.learner._curr_episode):  # pylint: disable=protected-access
            return valid_actions[random.randint(0, len(valid_actions) - 1)]

        # Otherwise, pick the action with highest value
        return self._greedy_policy(state, valid_actions)
