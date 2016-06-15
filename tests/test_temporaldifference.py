#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_temporaldifference
----------------------------------

Tests for `temporaldifference` module.
"""

import unittest2
import random

from rltools.learners import TemporalDifferenceLearner


class TestTemporalDifferenceLearner(unittest2.TestCase):


    def setUp(self):
        pass


    def tearDown(self):
        pass


    def test_000_interface(self):
        td = TemporalDifferenceLearner()
        tup = (0, 0, 0)
        td.fit(tup)
        td.fit([tup])
        td.fit([tup for i in range(10)])


    def test_001_fit_incrementally(self):

        # Fitting an entire episode at once:
        td1 = TemporalDifferenceLearner()
        td1.fit([(0, 0, 0), (1, 0, 0.1), (2, 0, 0.5), (3, 0, -1)])
        
        # Is equivalent to fitting each transition within the episode separately:
        td2 = TemporalDifferenceLearner()
        td2.fit((0, 0, 0))
        td2.fit((1, 0, 0.1))
        td2.fit((2, 0, 0.5))
        td2.fit((3, 0, -1))
        
        self.assertTrue(all([td1.val(i, 0) == td2.val(i, 0) for i in range(4)]))


    def test_002_td_0(self):
        u = [random.random(), random.random(), random.random(), random.random()]
        r = [0, random.random(), random.random(), random.random()]
        td = TemporalDifferenceLearner()
        td._gamma = random.random()
        td._lambda = 0
        td._alpha = lambda x: 1
    
        # Set initial values
        for i in range(len(u)):
            td._set_value(i, 0, u[i])
    
        # Incrementally fit random data
        td.fit([(i, 0, r[i]) for i in range(len(r))])
        
        s = [
            u[0] + td._alpha(1) * (r[1] + td._gamma * u[1] - u[0]),
            u[1] + td._alpha(1) * (r[2] + td._gamma * u[2] - u[1]),
            u[2] + td._alpha(1) * (r[3] + td._gamma * u[3] - u[2]),
            u[3]
        ]
    
        for i in range(len(s)):
            self.assertAlmostEqual(s[i], td.val(i, 0))


    def test_003_td_1(self):
        u = [random.random(), random.random(), random.random(), random.random()]
        r = [0, random.random(), random.random(), random.random()]
        td = TemporalDifferenceLearner()
        td._gamma = random.random()
        td._lambda = 1
        td._alpha = lambda x: 1
    
        # Set initial values
        for i in range(len(u)):
            td._set_value(i, 0, u[i])
    
        # Incrementally fit random data
        td.fit([(i, 0, r[i]) for i in range(len(r))])
        
        s = [
            u[0] + td._alpha(1) * (r[1] + td._gamma * r[2] + td._gamma ** 2 * r[3] + td._gamma ** 3 * u[3] - u[0]),
            u[1] + td._alpha(1) * (r[2] + td._gamma * r[3] + td._gamma ** 2 * u[3] - u[1]),
            u[2] + td._alpha(1) * (r[3] + td._gamma * u[3] - u[2]),
            u[3]
        ]
    
        for i in range(len(s)):
            self.assertAlmostEqual(s[i], td.val(i, 0))


    def test_004_td_converge_stockastic(self):
    
        r = [7.9,-5.1,2.5,-7.2,9.0,0.0,1.6]
        u = [0.0,4.0,25.7,0.0,20.1,12.2,0.0]
        p = 0.81
    
        state_zero_values = []
        for l in [1, 0]:
        
            td = TemporalDifferenceLearner()
            td._gamma = 1
            td._lambda = l
            td._alpha = lambda x: 1

            for i in range(len(u)):
                td._set_value(i, 0, u[i])

            for i in range(int(5E4)):
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

        self.assertAlmostEqual(state_zero_values[0], state_zero_values[1])


if __name__ == '__main__':
    import sys
    sys.exit(unittest2.main())
