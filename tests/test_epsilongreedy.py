#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_epsilongreedy
----------------------------------

Tests for `epsilongreedy` module.
"""

import unittest2

from rltools.learners import TemporalDifferenceLearner
from rltools.strategies import EpsilonGreedyStrategy


class TestEpsilonGreedyStrategy(unittest2.TestCase):
    # pylint: disable=protected-access, invalid-name


    def setUp(self):
        pass


    def tearDown(self):
        pass


    def test_000_interface(self):
        strategy = EpsilonGreedyStrategy(TemporalDifferenceLearner())
        tup = (0, 0, 0)
        strategy.fit(tup)
        strategy.fit([tup])
        strategy.fit([tup for i in range(10)])
        strategy.policy(0)

    # TODO: Epsilon greedy strategy tests


if __name__ == '__main__':
    import sys
    sys.exit(unittest2.main())
