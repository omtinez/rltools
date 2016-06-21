#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_learner
----------------------------------

Tests for `Learner` interface.
"""

import random
import pickle
import unittest2

from rltools.learners import Learner


class TestLearner(unittest2.TestCase):
    # pylint: disable=protected-access, invalid-name


    def setUp(self):
        self.cls = Learner


    def tearDown(self):
        pass


    def test_000_set_value(self):
        learner = self.cls(discount_factor=random.random(), learning_rate=random.random())
        learner._set_value(0, 0, 1)
        self.assertEqual(learner.val(0, 0), 1)
        learner._set_value(0, 0, 2)
        self.assertEqual(learner.val(0, 0), 2)


    def test_001_update_value(self):
        learner = self.cls(discount_factor=random.random(), learning_rate=random.random())
        learner._update_value(0, 0, 1)
        self.assertEqual(learner.val(0, 0), 1)
        learner._update_value(0, 0, 1)
        self.assertEqual(learner.val(0, 0), 2)


    def test_002_update_sets(self):
        learner = self.cls(discount_factor=random.random(), learning_rate=random.random())
        learner._set_value(0, 0, 0)
        learner._set_value(0, 1, 0)
        learner._set_value(1, 2, 0)
        self.assertEqual(learner.get_states(), set([0, 1]))
        self.assertEqual(learner.get_actions(), set([0, 1, 2]))


    def test_003_copy_values(self):
        learner = self.cls(discount_factor=random.random(), learning_rate=random.random())
        learner._set_value(0, 0, 1)
        self.assertEqual(learner._copy_values(), {(0, 0): 1})
        learner._set_value(0, 0, 2)
        self.assertEqual(learner._copy_values(), {(0, 0): 2})


    def test_004_init_episode(self):
        learning_rate = random.random()
        discount_factor = random.random()
        learner = self.cls(discount_factor=discount_factor, learning_rate=learning_rate)
        learner.init_episode()
        self.assertIsNone(learner._last_state)
        self.assertEqual(learner._learning_rate, learning_rate)
        self.assertEqual(learner._learning_discount, learning_rate)
        self.assertEqual(learner._discount_factor, discount_factor)


    def test_005_fit(self):
        learning_rate = random.random()
        discount_factor = random.random()
        learner1 = self.cls(discount_factor=discount_factor, learning_rate=learning_rate)
        learner2 = self.cls(discount_factor=discount_factor, learning_rate=learning_rate)

        learner1.fit([(0, 0, 0), (1, 0, 0.1), (2, 0, 0.5), (3, 0, -1)])

        learner2.fit((0, 0, 0))
        learner2.fit((1, 0, 0.1))
        learner2.fit((2, 0, 0.5))
        learner2.fit((3, 0, -1))

        self.assertEqual([learner1.val(i, 0) for i in range(4)],
                         [learner2.val(i, 0) for i in range(4)])


    def test_006_pickle(self):
        learning_rate = random.random()
        discount_factor = random.random()
        learner1 = self.cls(discount_factor=discount_factor, learning_rate=learning_rate)
        learner2 = self.cls(discount_factor=discount_factor, learning_rate=learning_rate)

        learner1.fit((0, 0, 0))
        learner1.fit((1, 0, 0.1))
        learner1.fit((2, 0, 0.5))
        learner1.fit((3, 0, -1))

        learner2.fit((0, 0, 0))
        learner2.fit((1, 0, 0.1))
        p_learner2 = pickle.dumps(learner2)
        up_learner2 = pickle.loads(p_learner2)
        up_learner2.fit((2, 0, 0.5))
        up_learner2.fit((3, 0, -1))

        self.assertEqual([learner1.val(i, 0) for i in range(4)],
                         [up_learner2.val(i, 0) for i in range(4)])


if __name__ == '__main__':
    import sys
    sys.exit(unittest2.main())
