#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_epsilongreedy
----------------------------------

Tests for `epsilongreedy` module.
"""

import unittest2
import random

from rltools.learners import TemporalDifferenceLearner
from rltools.strategies import EpsilonGreedyStrategy


class TestEpsilonGreedyStrategy(unittest2.TestCase):


    def setUp(self):
        pass


    def tearDown(self):
        pass


    # TODO: Epsilon greedy strategy tests


if __name__ == '__main__':
    import sys
    sys.exit(unittest2.main())
