# -*- coding: utf-8 -*-
import domains.gridworld
import domains.randomwalk
from strategies import EpsilonGreedyStrategy, RMaxStrategy
from learners import TemporalDifferenceLearner, MLMDP

agent = lambda: EpsilonGreedyStrategy(TemporalDifferenceLearner(l=0.6, discount_factor=0.9))
#agent = lambda: RMaxStrategy(TemporalDifferenceLearner(l=0.6))

#agent = lambda: EpsilonGreedyStrategy(MLMDP(normalize_count=0))
#agent = lambda: RMaxStrategy(MLMDP(normalize_count=100))

domains.randomwalk.play(agent(), converge=True)

#domains.gridworld.play(agent(), converge=True)

domains.gridworld.play(agent())

#domains.gridworld.test()
