import random

class RandomWalk(object):
    ACTIONS = [0]

    def __init__(self):
        self.num_states = 7
        self.current_state = 3

    def take_action(self, action):
        if action > 0:
            raise ValueError('Only action "0" supported in this domain')
        if self.current_state == 0 or self.current_state == self.num_states:
            raise RuntimeError('Terminal state reached')

        roll = random.random()
        if roll > 0.5:
            self.current_state += 1
        else:
            self.current_state -= 1

        reward = 1 if self.current_state == self.num_states - 1 else 0
        return action, reward, self.current_state


def play(strategy, iterations=100, converge=False):
    strategy.valid_actions = RandomWalk.ACTIONS
    mydomain = RandomWalk()
    strategy.fit((0, 0, 0))

    count = 0
    while count < iterations:
        action = strategy.policy(mydomain.current_state)
        try:
            a, r, s = mydomain.take_action(action)
            strategy.fit((s, a, r))

        except RuntimeError:
            count += 1
            mydomain.current_state = 3
            strategy.init_episode() 
            strategy.fit((0, 0, 0))

    if converge:
        strategy.converge(max_time=60)

    true_prob = [1./6, 1./3, 1./2, 2./3, 5./6]
    print('Estimated probabilities:', ['%.5f' % strategy.learner.val(i, 0) for i in range(1,6)])
    print('Expected probabilities:', ['%.5f' % p for p in true_prob])

    rmse = sum([(strategy.learner.val(i, 0) - true_prob[i-1]) ** 2 for i in range(1,6)]) / len(true_prob)
    print('RMSE:', rmse)
    return rmse

