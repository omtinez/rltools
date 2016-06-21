#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_mldmp
----------------------------------

Tests for `maximum likelihood MDP` module.
"""

import unittest2
import random
import os

from tests.test_learner import TestLearner
from rltools.learners import MLMDP
from rltools.strategies import Strategy
from rltools.domains import randomwalk

class TestMLMDP(TestLearner):


    def setUp(self):
        self.cls = MLMDP


    def tearDown(self):
        pass


    def test_000_deterministic(self):
        learner = MLMDP(discount_factor=0.75, learning_rate=1, normalize_count=0)

        learner.init_episode()
        learner.fit((0, 0, 0))
        for i in range(1000):
            learner.fit((1, 1, 10))
            learner.fit((0, 0, 0))
            learner.fit((2, 2, -10))
            learner.fit((0, 0, 0))

        learner.converge()

        self.assertEqual(learner.val(0, 0), 0)
        self.assertEqual(learner.val(1, 0), 0)
        self.assertEqual(learner.val(2, 0), 0)

        self.assertEqual(learner.val(0, 1), -learner.val(0, 2))
        self.assertEqual(learner.val(1, 1), -learner.val(1, 2))
        self.assertEqual(learner.val(2, 1), -learner.val(2, 2))


    def test_001_biased(self):
        learner = MLMDP(discount_factor=0.75, learning_rate=1, normalize_count=0)

        learner.init_episode()
        learner.fit((0, 0, 0))
        for i in range(1000):
            learner.fit((1, 1, 10 - random.random()))
            learner.fit((0, 0, 0))
            learner.fit((2, 2, -10))
            learner.fit((0, 0, 0))

        learner.converge()

        self.assertEqual(learner.val(0, 0), 0)
        self.assertEqual(learner.val(1, 0), 0)
        self.assertEqual(learner.val(2, 0), 0)

        self.assertLess(learner.val(0, 1), -learner.val(0, 2))
        self.assertLess(learner.val(1, 1), -learner.val(1, 2))
        self.assertLess(learner.val(2, 1), -learner.val(2, 2))


    def test_004_play_random_walk(self):
        agent = Strategy(MLMDP())
        rmse = randomwalk.play(agent, converge=True)
        self.assertLess(rmse, 0.2)


if __name__ == '__main__':
    import sys
    sys.exit(unittest2.main())
