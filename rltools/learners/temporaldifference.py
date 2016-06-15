class TemporalDifferenceLearner(object):
    '''
    Learner that uses the temporal difference methods described by Richard Sutton in
    `Learning to Predict by the Methods of Temporal Differences`.
    '''
    
    def __init__(self, l=0.6, discount_factor=0):
        self._values = {}
        self._prev_values = {}
        self._gamma = 1.0 - discount_factor
        self._lambda = l
        self._curr_episode = 0
        self._episode_list = []
        self._last_state = None

        self._all_states = set()
        self._all_actions = set()


    def get_states(self):
        return set(self._all_states)


    def get_actions(self):
        return set(self._all_actions)


    def val(self, state, action):
        '''
        Retrieves the estimated value of taking an `action` from this `state`, or zero if this
        <state, action> has never been encountered before.

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

        
    def _alpha(self, T):
        '''
        By default, make alpha (the learning rate) decrease as more episodes are reached. Users are
        encouraged to override this function in subclasses or specific instances.
        '''
        return 1 / T
    
    
    def init_episode(self):
        '''
        Called after a terminal state is reached and a new episode is started. This method is
        automatically called when the first observation is recorded using `fit()` or an observation
        is recorded with a state value of `None`.

        Examples
        --------
        >>> td = TemporalDifferenceLearner()
        >>>
        >>> # Fitting an entire episode at once will automatically call `init_episode()`:
        >>> td.fit([(0, 0, 0), (1, 0, 0.1), (2, 0, 0.5), (3, 0, -1)])
        >>>
        >>> # The prior line is equivalent to:
        >>> td.init_episode()
        >>> td.fit([(0, 0, 0), (2, 0, 0.3), (3, 0, -1)])
        >>> 
        >>> # A new episode can alternatively be initialized by passing a state with value `None`:
        >>> td.fit((None, 0, 0))
        >>> 
        >>> # This also works for passing multiple episodes as a single iterable:
        >>> td.fit([
        ...     (0, 0, 0), (1, 0, 0.1), (2, 0, 0.5), (3, 0, -1),
        ...     (None, 0, 0),
        ...     (0, 0, 0), (2, 0, 0.3), (3, 0, -1),
        ... ])
        '''
        if len(self._episode_list) == 0 or len(self._episode_list[-1]) > 0:
            self._curr_episode += 1
            self._last_state = None
            self._episode_list.append([])
            self._prev_values = self._copy_values()
        
        
    def _set_value(self, state, action, val):
        ''' Helper method to override the value of specific <state, action> '''
        self._values[(state, action)] = val


    def _update_value(self, state, action, val):
        ''' Helper method to add/subtract value estimated for specific <state, action> '''
        self._values[(state, action)] = val + self._values.get((state, action), 0)


    def _copy_values(self):
        return {k:v for k, v in self._values.items()}


    def fit(self, X):
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
        >>> td1 = TemporalDifferenceLearner()
        >>>
        >>> # Fitting an entire episode at once:
        >>> td1.fit([(0, 0, 0), (1, 0, 0.1), (2, 0, 0.5), (3, 0, -1)])
        >>>
        >>> # Equivalent to fitting each transition within the episode separately:
        >>> td2 = TemporalDifferenceLearner()
        >>> td2.fit((0, 0, 0))
        >>> td2.fit((1, 0, 0.1))
        >>> td2.fit((2, 0, 0.5))
        >>> td2.fit((3, 0, -1))
        >>>
        >>> all([td1.val(i, 0) == td2.val(i, 0) for i in range(4)])
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
            if self._last_state is not None:
                self._learn_incr(self._last_state, action, reward, state)
            self._all_states.add(state)
            self._all_actions.add(action)
            self._last_state = state
    

    def _learn_incr(self, prev_state, action, reward, curr_state):
        ''' Incrementally update the value estimates after observing a transition between states '''
        
        # Call init_episode() if this is the first item
        if len(self._values) == 0:
            self.init_episode()
        self._episode_list[-1].append((prev_state, action, reward, curr_state, 1))
        
        # For the previous state, compute value as the max of values for all possible actions
        prev_state_val = max([self.val(prev_state, a) for a in self._all_actions])

        # Compute the value added to each of the states
        delta = reward + (self._gamma * self._prev_values.get((curr_state, action), 0) - prev_state_val)
        value = self._alpha(self._curr_episode) * delta

        for ix, vec in enumerate(self._episode_list[-1]):
            state1, action_, reward_, state2, eligibility = vec
            self._update_value(state1, action_, value * eligibility)  # update value
            self._episode_list[-1][ix] = vec[:-1] + (vec[-1] * self._gamma * self._lambda,)  # decay eligibility
                        
            
    def _learn_episode(self, ix):
        ''' Helper method to re-learn a specific episode given its index '''
        
        # Insert new episode
        self.init_episode()
        
        # Copy the old one and re-learn it
        for vec in self._episode_list[ix]:
            state1, action, reward, state2, eligibility = tuple(vec)
            self._learn_incr(state1, action, reward, state2)
            
        # Remove inserted episode
        self._episode_list.pop()
  
            
    @staticmethod
    def _vec_diff(a, b):
        return sum([(a.get(k, 0) - b.get(k, 0)) ** 2 for k in (list(a.keys()) + list(b.keys()))])
            
        
    def _converge(self, atol=1E-3, max_iter=1000):
        ''' Train over already fitted data over and over until convergence '''
        
        curr_diff = atol
        prev_vals = {k:0 for k in self._values.keys()}
        for i in range(max_iter):
            
            for j in range(len(self._episode_list)):
                self._learn_episode(j)
            
            try:
                curr_diff = self._vec_diff(prev_vals, self._values)
            except OverflowError as e:
                pass
            prev_vals = self._copy_values()
            if curr_diff < atol:
                return self._values
            
        raise RuntimeError('Convergence not achieved after %d iterations, current squared diff: %f' 
                           % (max_iter, curr_diff))
            