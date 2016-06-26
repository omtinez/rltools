from rltools.learners import Learner


def norm(vec_a, vec_b):
    num = min(len(vec_a), len(vec_b))
    return sum([(vec_a[i] - vec_b[i]) ** 2 for i in range(num)]) ** 0.5


def linear_combination(weights, values):
    num = min(len(weights), len(values))
    return sum([weights[i] * values[i] for i in range(num)]) / float(num)


class ValueFunctionApproximation(Learner):
    '''
    Value function approximation learner. Instead of a discrete state space, assume that each state
    is a vector of features and that the value of any state can be estimated by a parametrized
    function.
    '''

    def __init__(self, dof, discount_factor=0.75, learning_rate=0.9, value_fn=None):
        Learner.__init__(self, discount_factor, learning_rate)

        self.dof = dof
        self.params = {}
        self.value_fn = linear_combination if value_fn is None else value_fn


    def val(self, state, action):
        params = self.params.get(action, [0 for _ in range(self.dof)])
        return self.value_fn(params, state)


    def _learn_incr(self, prev_state, action, reward, curr_state):
        ''' Incrementally update value estimates after observing a transition between states '''

        # Keep a set of parameters for every action
        if action not in self.params:
            # Make dimensionality of parameters equal to the provided degrees of freedom
            self.params[action] = [random.random() for _ in range(self.dof)]

        # The estimated value is just the output from the value function approximation
        estimated_value = self.value_fn(self.params[action], curr_state)

        # Future reward is estimated as the discounted largest value of future state over all
        # possible actions
        future_reward = self._discount_factor * \
            max([self.val(curr_state, action_) for action_ in self.get_actions()])

        # The total error will be reward + discounted future value - estimated current value,
        # decayed by the learning rate
        total_error = self._learning_rate * (reward + future_reward - estimated_value)

        # Update the weights (parameters) based on the amount of error by computing dv/dw for each
        # weight
        weight_deltas = self.derivative_params(prev_state, action)
        for i in range(self.dof):
            self.params[action][i] += total_error * weight_deltas[i]
            self.params[action][i] = max(-1, min(1, self.params[action][i]))


    def derivative_params(self, state, action):
        deltas = []
        params = self.params[action]
        for i in range(self.dof):
            only_one = [params[j] if j == i else 0 for j in range(self.dof)]
            deltas.append(self.value_fn(only_one, state))
        return deltas


    def converge(self, atol=1E-3, max_iter=1000, max_time=0):
        ''' Train over already fitted data over and over until convergence '''
        # TODO