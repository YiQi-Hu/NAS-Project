"""
All basic data structure in this file.
PLEASE DO NOT USE 'from .base import *' !!!
"""

class NetworkUnit:
    pre_block=[]
    def __init__(self, graph_part=[[]], cell_list=[]):
        self.graph_part = graph_part
        self.cell_list = cell_list
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