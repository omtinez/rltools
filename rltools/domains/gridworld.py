import random

class GridWorld(object):
    ACTIONS = [0, 1, 2, 3]

    def __init__(self, grid, rewards, noise=0):
        self.grid = grid
        self.rewards = rewards
        self._state_map = {}
        self._reverse_map = {}
        self.initial_state = 0
        self._success_prob = 1 - noise

        for i in range(len(grid)):
            for j in range(len(grid[i])):
                self._state_map[(i, j)] = len(self._state_map)
                self._reverse_map[self._state_map[(i, j)]] = (i, j)

        self.current_state = self.initial_state


    def take_action(self, action):
        coords = self._reverse_map[self.current_state]
        if self.grid[coords[0]][coords[1]] == -1:
            raise RuntimeError('Terminal state reached')

        roll = random.random()
        if roll > self._success_prob:
            if action == 0 or action == 1:
                action = 2 if random.random() > 0.5 else 3
            elif action == 2 or action == 3:
                action = 0 if random.random() > 0.5 else 1

        if action == 0:  # 0 is left
            moveto = (coords[0], max(0, coords[1] - 1))
        elif action == 1:  # 1 is right
            moveto = (coords[0], min(len(self.grid[coords[0]]) - 1, coords[1] + 1))
        elif action == 2:  # 2 is up
            moveto = (max(0, coords[0] - 1), coords[1])
        elif action == 3:  # 3 is down
            moveto = (min(len(self.grid) - 1, coords[0] + 1), coords[1])

        if self.grid[moveto[0]][moveto[1]] == 1:
            moveto = coords

        reward = self.rewards[moveto[0]][moveto[1]]
        self.current_state = self._state_map[moveto]

        return action, reward, self.current_state


def play(strategy, iterations=1000, converge=False, max_steps_per_game=1000):
    strategy.valid_actions = GridWorld.ACTIONS

    grid_matrix = [[0, 0, 0, -1],
                   [0, 1, 0, -1],
                   [0, 0, 0, 0]]

    reward_matrix = [[-0.04, -0.04, -0.04, 10],
                     [-0.04, -0.04, -0.04, -10],
                     [-0.04, -0.04, -0.04, -0.04]]

    mygrid = GridWorld(grid_matrix, reward_matrix, 0.2)
    strategy.fit((0, 0, 0))

    count = 0
    steps = 0
    while count < iterations:
        action = strategy.policy(mygrid.current_state)
        try:
            action, reward, state = mygrid.take_action(action)
            strategy.fit((state, action, reward))
            steps += 1

            if steps > max_steps_per_game:
                raise RuntimeError('Maximum number of steps reached')

        except RuntimeError:
            count += 1
            mygrid.current_state = 0
            strategy.init_episode() 
            strategy.fit((0, 0, 0))

    if converge:
        strategy.converge()

    action_names = {0: '<', 1: '>', 2: '^', 3: 'v'}

    print('')
    for i in range(len(grid_matrix)):
        row = ''
        for j in range(len(grid_matrix[i])):
            s = mygrid._state_map[(i, j)]
            o = action_names.get(strategy._greedy_policy(s)) if grid_matrix[i][j] == 0 \
                else str(grid_matrix[i][j])
            row += o + '\t'
        print(row)

    for a in GridWorld.ACTIONS:
        print('')
        print('Action: %s' % action_names.get(a))
        for i in range(len(grid_matrix)):
            row = ''
            for j in range(len(grid_matrix[i])):
                s = mygrid._state_map[(i,j)]
                o = '%.3f' % strategy.learner.val(s, a) if grid_matrix[i][j] == 0 \
                    else str(grid_matrix[i][j])
                row += o + '\t'
            print(row)

        if hasattr(strategy, '_transition_count'):
            print('')
            print('Action count: %s' % action_names.get(a))
            for i in range(len(grid_matrix)):
                row = ''
                for j in range(len(grid_matrix[i])):
                    s = mygrid._state_map[(i,j)]
                    o = '%d' % strategy._transition_count.get((s, a), 0)
                    row += o + '\t'
                print(row)
