# -*- coding: utf-8 -*-
import domains.gridworld
import domains.randomwalk
from strategies import EpsilonGreedyStrategy
from learners import TemporalDifferenceLearner

agent = lambda: EpsilonGreedyStrategy(TemporalDifferenceLearner(l=0))
domains.randomwalk.play(agent())
domains.gridworld.play(agent())
