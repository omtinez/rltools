# -*- coding: utf-8 -*-
import rltools.domains.gridworld
import rltools.domains.randomwalk
from rltools.strategies import EpsilonGreedyStrategy, RMaxStrategy
from rltools.learners import TemporalDifferenceLearner, MLMDP

#agent = lambda: EpsilonGreedyStrategy(TemporalDifferenceLearner(l=0.6, discount_factor=0.9))
#agent = lambda: RMaxStrategy(TemporalDifferenceLearner(l=0.6))

agent = lambda: EpsilonGreedyStrategy(MLMDP(normalize_count=100))
#agent = lambda: RMaxStrategy(MLMDP(normalize_count=100))

rltools.domains.randomwalk.play(agent(), converge=True)

#domains.gridworld.play(agent(), converge=True)

rltools.domains.gridworld.play(agent())

#domains.gridworld.test()
