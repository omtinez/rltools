import time
import numpy as np

class Learner(object):
    '''
    Define interface of methods that must be implemented by all inhereting classes.
    '''

    def __init__(self, discount_factor=1, learning_rate=1):
        self._values = {}
        self._prev_values = {}
        self._discount_factor = discount_factor
        self._learning_rate = learning_rate
        self._learning_discount = learning_rate
        self._curr_episode = 0
        self._last_state = None

        self._all_states = set()
        self._all_actions = set()


    def _set_value(self, state, action, val):
        ''' Helper method to override the value of specific <state, action> '''
        self._all_states.add(state)
        self._all_actions.add(action)
        self._values[(state, action)] = val


    def _update_value(self, state, action, val):
        '''
        Helper method to add/subtract value estimated for specific <state, action> using the
        current learning rate
        '''
        self._all_states.add(state)
        self._all_actions.add(action)
        self._values[(state, action)] = val * self._learning_rate + \
            (1.0 - self._learning_rate) * self._values.get((state, action), 0)


    def _copy_values(self):
        return {k:v for k, v in self._values.items()}


    def get_states(self):
        return set(self._all_states)


    def get_actions(self):
        return set(self._all_actions)


    def val(self, state, action):
        '''
        Retrieves the estimated value of taking an `action` from this `state`, or zero if this
        <state, action> has no learned value.

        Parameters
        ----------
        state : int
            Integer used to identify a unique state.
        action : int
            Integer used to identify a unique action.

        Returns
        -------
        val : float
            Estimated value of the `state` reached after taking `action`.
        '''
        return self._values.get((state, action), 0)


    def init_episode(self, init_state=None):
        '''
        Called after a terminal state is reached and a new episode is started. This method is
        automatically called when the first observation is recorded using `fit()` or an observation
        is recorded with a state value of `None`.

        Examples
        --------
        >>> learner = Learner()
        >>>
        >>> # Fitting an entire episode at once will automatically call `init_episode()`:
        >>> learner.fit([(0, 0, 0), (1, 0, 0.1), (2, 0, 0.5), (3, 0, -1)])
        >>>
        >>> # The prior line is equivalent to:
        >>> learner.init_episode()
        >>> learner.fit([(0, 0, 0), (2, 0, 0.3), (3, 0, -1)])
        >>>
        >>> # A new episode can alternatively be initialized by passing a state with value `None`:
        >>> learner.fit((None, 0, 0))
        >>>
        >>> # This also works for passing multiple episodes as a single iterable:
        >>> learner.fit([
        ...     (0, 0, 0), (1, 0, 0.1), (2, 0, 0.5), (3, 0, -1),
        ...     (None, 0, 0),
        ...     (0, 0, 0), (2, 0, 0.3), (3, 0, -1),
        ... ])
        '''
        self._curr_episode += 1
        self._learning_rate = self._learning_discount ** self._curr_episode
        self._last_state = init_state


    def fit(self, X):  # pylint: disable=invalid-name
        '''
        Fit the learner with the provided data in the form of <state, action, reward>. The reward
        and action for the first state in an episode are always ignored.

        Parameters
        ----------
        X : Tuple of <state, action, reward> or array-like of <state, action, reward>, where
            `state` is an integer representing the current state (that the agent just landed on),
            `action` is an integer representing the action taken by the agent to migrate from the
            previews state to `state`, and reward is a number (int or float type) representing the
            reward given for reaching the `state` after taking `action`.

        Examples
        --------
        >>> learner1 = Learner()
        >>>
        >>> # Fitting an entire episode at once:
        >>> learner1.fit([(0, 0, 0), (1, 0, 0.1), (2, 0, 0.5), (3, 0, -1)])
        >>>
        >>> # Equivalent to fitting each transition within the episode separately:
        >>> learner2 = Learner()
        >>> learner2.fit((0, 0, 0))
        >>> learner2.fit((1, 0, 0.1))
        >>> learner2.fit((2, 0, 0.5))
        >>> learner2.fit((3, 0, -1))
        >>>
        >>> all([learner1.val(i, 0) == learner2.val(i, 0) for i in range(4)])
        ... True
        '''
        if not hasattr(X, '__iter__'):
            raise ValueError('Parameter must be tuple of <state, reward> or iterable of tuples'
                             'of <state, reward>')
        elif all((hasattr(tup, '__iter__') and len(tup) == 3 for tup in X)):
            self.init_episode()
        else:
            if self._last_state is None:
                self.init_episode()
            X = [X]

        for (state, action, reward) in X:
            self._all_states.add(state)
            self._all_actions.add(action)
            if self._last_state is not None:
                self._learn_incr(self._last_state, action, reward, state)
            self._last_state = state


    def _learn_incr(self, prev_state, action, reward, curr_state):  # pylint: disable=unused-argument
        ''' Incrementally update the value estimates after observing a transition between states '''
        self._update_value(prev_state, action, reward * self._learning_rate)


    def _value_iteration(self, T, R, atol=1E-3, max_iter=1000, max_time=0):  # pylint: disable=too-many-arguments, invalid-name
        '''
        Given transition matrix T and reward matrix R, compute value of <state, action> vectors
        using value iteration algorithm.
        '''

        n_states = max(self.get_states()) + 1
        n_actions = max(self.get_actions()) + 1
        curr_values = np.zeros((n_actions, n_states))

        # Iterate and increase state values by the discounted adjacent state values
        stopwatch = time.time() + max_time
        prev_values = curr_values.copy()
        for i in range(max_iter):  # pylint: disable=unused-variable
            for action in range(n_actions):
                for state1 in range(n_states):
                    curr_values[action, state1] = (T[action, state1] * R[action, state1]).sum()
                    for state2 in range(n_states):
                        curr_values[action, state1] += T[action, state1, state2] * \
                            self._discount_factor * prev_values[action, state2]

            if ((prev_values - curr_values) ** 2).mean() < atol or \
                (max_time > 0 and stopwatch < time.time()):
                break

            prev_values = curr_values.copy()

        return curr_values

    def converge(self, atol=1E-5, max_iter=1000, max_time=0):
        ''' Train over already fitted data over and over until convergence '''
        raise NotImplementedError(
            'Classes inhereting from Learner must override Learner.converge()')
            