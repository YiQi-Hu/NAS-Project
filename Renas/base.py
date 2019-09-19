"""
All basic data structure in this file.
PLEASE DO NOT USE 'from .base import *' !!!
"""


class NetworkUnit:
    pre_block = []

    def __init__(self, graph_part=[[]], cell_list=[]):
        self.graph_part = graph_part
        self.graph_full_list = []  # to store the structure ever used including skipping layers
        self.cell_list = cell_list
        self.score_list = []
        self.spl = None
        self.opt = None
        self.table = []

    def init_sample(self, pattern="Global", block_num=0):
        from .optimizer import Optimizer
        if pattern == "Block":
            from .sampler_block import Sampler
            self.spl = Sampler(self.graph_part, len(self.graph_part), block_num)
        else:
            from .sampler_global import Sampler
            self.spl = Sampler(self.graph_part, len(self.graph_part))
        self.opt = Optimizer(self.spl.get_dim(), self.spl.get_parametets_subscript())
        self.table = []  # to store the encoding derived from opt for sampling
        sample_size = 3  # the instance number of sampling in an iteration
        budget = 20000  # budget in online style
        positive_num = 2  # the set size of PosPop
        rand_probability = 0.99  # the probability of sample in model
        uncertain_bit = 3  # the dimension size that is sampled randomly
        # set hyper-parameter for optimization, budget is useless for single step optimization
        self.opt.set_parameters(ss=sample_size, bud=budget, pn=positive_num, rp=rand_probability, ub=uncertain_bit)
        # clear optimization model
        self.opt.clear()
        return