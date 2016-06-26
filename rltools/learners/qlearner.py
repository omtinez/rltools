from rltools.learners import Learner

class QLearner(Learner):
    '''
    Simple learner implementing Q-Learning. For each observation, update the estimated value of a
    state proportionally to the learning rate and taking into account the (discounted) estimated
    value of future states.
    '''

    def __init__(self, learning_rate=0.2, discount_factor=0.9):
        Learner.__init__(self, discount_factor, learning_rate)
                 
            
    def _learn_incr(self, prev_state, action, reward, curr_state):  # pylint: disable=unused-argument
        ''' Incrementally update the value estimates after observing a transition between states '''
        future_reward = max([self.val(curr_state, action_) for action_ in self.get_actions()])
        self._update_value(
            prev_state, action, reward + self._discount_factor * future_reward)
