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
        self.table_list = []
        self.score_list = []
        self.best_score = 0  # to store best score it has ever met
        self.best_index = -1  # to store index of best score it has ever met
        self.spl = None

    def init_sample(self, pattern="Global", block_num=0, spl_setting=None, skipping_max_dist=3,
                    skipping_max_num=2, ops_space=None):
        from .sampler import Sampler
        self.spl = Sampler(self.graph_part, skipping_max_dist, skipping_max_num, block_num, pattern,
                           spl_setting, ops_space)
        return
    
    def update_best_score(self, score):
        if score > self.best_score:
            self.best_score = score
            self.best_index = len(self.score_list) - 1


class Dataset():
    def __init__(self):
        self.feature = None
        self.label = None
        self.shape = None
        return
        
    def load_from(self, path=""):
        # TODO
        return
