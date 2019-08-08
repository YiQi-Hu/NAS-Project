"""
All basic data structure in this file.
PLEASE DO NOT USE 'from .base import *' !!!
"""
from .optimizer import Optimizer
from .sampler import Sampler_table


class NetworkUnit:
    pre_block=[]
    def __init__(self, graph_part=[[]], cell_list=[]):
        self.graph_part = graph_part
        self.graph_full  # to store the structure including skipping layers
        self.cell_list = cell_list
        self.spl = Sampler_table()
        self.opt = Optimizer(self.spl.get_dim(), self.spl.get_parametets_subscript())
        self.pros  # to store the probability space derived from opt for sampling
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

class Dataset():
    def __init__(self):
        self.feature = None
        self.label = None
        self.shape = None
        return
        
    def load_from(self, path=""):
        # TODO
        return