"""
All basic data structure in this file.
PLEASE DO NOT USE 'from .base import *' !!!
"""
from sampler import Sampler

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

    def init_sample(self, block_num=0):
        self.spl = Sampler(self.graph_part, block_num)
        return