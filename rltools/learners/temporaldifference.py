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
        
        # For the previous state, compute value as the max of values for all possible actions
        prev_state_val = max([self.val(prev_state, a) for a in self._all_actions])

        # Compute the value added to each of the states
        delta = reward + (self._discount_factor * self._prev_values.get((curr_state, action), 0) - prev_state_val)
        value = self._learning_rate * delta

        for ix, vec in enumerate(self._episode_list[-1]):
            state1, action_, reward_, state2, eligibility = vec
            self._update_value(state1, action_, value * eligibility)  # update value
            self._episode_list[-1][ix] = vec[:-1] + (vec[-1] * self._discount_factor * self._lambda,)  # decay eligibility
                        
            
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
            
        
    def converge(self, atol=1E-3, max_iter=1000, max_time=0):
        ''' Train over already fitted data over and over until convergence '''

        stopwatch = time.time() + max_time
        def give_up():
            raise RuntimeError('Convergence not achieved after %d iterations and %.03f seconds,'
                               'current squared diff: %f' 
                                % (max_iter, time.time() - stopwatch + max_time, curr_diff))
        
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
            elif max_time > 0 and stopwatch < time.time():
                give_up()
            
        give_up()
            