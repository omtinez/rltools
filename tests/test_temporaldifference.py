#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_temporaldifference
----------------------------------

Tests for `temporaldifference` module.
"""

import unittest2
import random
import os

from tests.test_learner import TestLearner
from rltools.learners import TemporalDifferenceLearner
from rltools.strategies import Strategy
from rltools.domains import randomwalk

class TestTemporalDifferenceLearner(TestLearner):


    def setUp(self):
        self.cls = TemporalDifferenceLearner


    def tearDown(self):
        pass


    def test_000_td_0(self):
        u = [random.random(), random.random(), random.random(), random.random()]
        r = [0, random.random(), random.random(), random.random()]
        td = TemporalDifferenceLearner(learning_rate=random.random())
        td._lambda = 0
    
        # Set initial values
        for i in range(len(u)):
            td._set_value(i, 0, u[i])
    
        # Incrementally fit random data
        td.fit([(i, 0, r[i]) for i in range(len(r))])
        
        s = [
            u[0] + td._learning_rate * (r[1] + td._discount_factor * u[1] - u[0]),
            u[1] + td._learning_rate * (r[2] + td._discount_factor * u[2] - u[1]),
            u[2] + td._learning_rate * (r[3] + td._discount_factor * u[3] - u[2]),
            u[3]
        ]
    
        for i in range(len(s)):
            self.assertAlmostEqual(s[i], td.val(i, 0))


    def test_001_td_1(self):
        u = [random.random(), random.random(), random.random(), random.random()]
        r = [0, random.random(), random.random(), random.random()]
        td = TemporalDifferenceLearner(learning_rate=random.random())
        td._lambda = 1
    
        # Set initial values
        for i in range(len(u)):
            td._set_value(i, 0, u[i])
    
        # Incrementally fit random data
        td.fit([(i, 0, r[i]) for i in range(len(r))])
        
        s = [
            u[0] + td._learning_rate * (r[1] + td._discount_factor * r[2] + td._discount_factor ** 2 * r[3] + td._discount_factor ** 3 * u[3] - u[0]),
            u[1] + td._learning_rate * (r[2] + td._discount_factor * r[3] + td._discount_factor ** 2 * u[3] - u[1]),
            u[2] + td._learning_rate * (r[3] + td._discount_factor * u[3] - u[2]),
            u[3]
        ]
    
        for i in range(len(s)):
            self.assertAlmostEqual(s[i], td.val(i, 0))


    @unittest2.skipIf("TRAVIS" in os.environ and os.environ["TRAVIS"], "Skipping this test on Travis CI.")
    def test_002_td_converge_stockastic(self):
    
        r = [7.9,-5.1,2.5,-7.2,9.0,0.0,1.6]
        u = [0.0,4.0,25.7,0.0,20.1,12.2,0.0]
        p = 0.81
    
        # Stockastic results, try at most 3 times
        atol = 1E-3
        diff = 1E-3
        for i in range(3):

            state_zero_values = []
            for l in [1, 0]:
        
                td = TemporalDifferenceLearner(learning_rate=1, discount_factor=1)
                td._lambda = l

                for i in range(len(u)):
                    td._set_value(i, 0, u[i])

                for i in range(int(1E5)):
                    rnd = random.random()
                    td.fit([
                        (0, 0, 0),
                        (1, 0, r[0]) if rnd < p else (3, 0, r[2]),
                        (2, 0, r[1]) if rnd < p else (3, 0, r[3]),
                        (4, 0, r[4]),
                        (5, 0, r[5]),
                        (6, 0, r[6])
                    ])
                state_zero_values.append(td.val(0, 0))

            diff = abs(state_zero_values[0] - state_zero_values[1])
            if diff < atol: break

        self.assertAlmostEqual(state_zero_values[0], state_zero_values[1])


    def test_003_play_random_walk(self):
        agent = Strategy(TemporalDifferenceLearner(l=0))
        rmse = randomwalk.play(agent)
        self.assertLess(rmse, 0.1)


if __name__ == '__main__':
    import sys
    sys.exit(unittest2.main())
