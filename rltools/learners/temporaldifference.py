import time

from rltools.learners import Learner

class TemporalDifferenceLearner(Learner):
    '''
    Learner that uses the temporal difference methods described by Richard Sutton in
    `Learning to Predict by the Methods of Temporal Differences`.
    '''

    def __init__(self, l=0.6, discount_factor=1, learning_rate=0.9):
        Learner.__init__(self, discount_factor, learning_rate)
        self._lambda = l
        self._episode_list = []


    def init_episode(self):
        '''
        Called after a terminal state is reached and a new episode is started. See `Learner.fit()`
        for more details.
        '''
        if len(self._episode_list) == 0 or len(self._episode_list[-1]) > 0:
            Learner.init_episode(self)
            self._episode_list.append([])
            self._prev_values = self._copy_values()


    def _learn_incr(self, prev_state, action, reward, curr_state):
        ''' Incrementally update the value estimates after observing a transition between states '''

        # Call init_episode() if this is the first item
        if len(self._values) == 0:
            self.init_episode()
        self._episode_list[-1].append((prev_state, action, reward, curr_state, 1))

        # For this state, compute value as the max of values for all possible actions
        estimated_value = max([self.val(prev_state, a) for a in self._all_actions])

        # Keep track of the value estimate based on prior iterations
        previous_value = self._prev_values.get((curr_state, action), 0)

        # Compute the value added to each of the states
        delta = reward + self._discount_factor * previous_value - estimated_value
        value = self._learning_rate * delta

        for ix, vec in enumerate(self._episode_list[-1]):
            state_, action_, _, _, eligibility = vec
            self._set_value(state_, action_, value * eligibility \
                + self.val(state_, action_))  # update value in additive way
            self._episode_list[-1][ix] = vec[:-1] + (vec[-1] * self._discount_factor * \
                self._lambda,)  # decay eligibility


    def _learn_episode(self, ix):
        ''' Helper method to re-learn a specific episode given its index '''

        # Insert new episode
        self.init_episode()

        # Copy the old one and re-learn it
        for vec in self._episode_list[ix]:
            state1, action, reward, state2, _ = tuple(vec)
            self._learn_incr(state1, action, reward, state2)

        # Remove inserted episode
        self._episode_list.pop()


    @staticmethod
    def _vec_diff(vec_a, vec_b):
        return sum([(vec_a.get(key, 0) - vec_b.get(key, 0)) ** 2 for key in \
            list(vec_a.keys()) + list(vec_b.keys())])


    def converge(self, atol=1E-3, max_iter=1000, max_time=0):
        ''' Train over already fitted data over and over until convergence '''

        stopwatch = time.time() + max_time
        def give_up():
            raise RuntimeError('Convergence not achieved after %d iterations and %.03f seconds,'
                               'current squared diff: %f'
                               % (max_iter, time.time() - stopwatch + max_time, curr_diff))

        curr_diff = atol
        prev_vals = {key: 0 for key in self._values.keys()}
        for _ in range(max_iter):

            for j in range(len(self._episode_list)):
                self._learn_episode(j)

            try:
                curr_diff = self._vec_diff(prev_vals, self._values)
            except OverflowError:
                pass
            prev_vals = self._copy_values()
            if curr_diff < atol:
                return self._values
            elif max_time > 0 and stopwatch < time.time():
                give_up()

        give_up()
            