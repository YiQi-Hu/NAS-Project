"""
All basic data structure in this file.
PLEASE DO NOT USE \'from .base import \*\' !!!
"""
from info_str import NAS_CONFIG

from multiprocessing import Pool
class Network(object):
    pre_block = []

    def __init__(self, _id, graph_tmp):
        self.id = _id
        self.graph_template = graph_tmp
        self.item_list = []
        self.spl = None

class NetworkItem(object):
    def __init__(self, _id=0, graph=[], cell_list=[], code=[]):
        self.id = _id
        self.graph = graph
        self.cell_list = cell_list
        self.code = code
        self.score = 0

class Cell(tuple):
    """class Cell inheirt from tuple
    Attributes:
        1. cell_type (string, 'conv' or 'pooling')
    Example:
        conv_cell = Cell('conv', 48, 7, 'relu')
        pooling_cell = Cell('pooling', 'avg', 9)
        print(conv_cell.type) # 'conv'
        print(pooling_cell.type) # 'pooling'
        print(conv_cell) # ('conv', 48, 7, 'relu')
        print(pooling_cell) # ('pooling', 'avg', 9)
    """
    __keys_index = {}
    __SPACE_CONFIG = NAS_CONFIG['spl']['space']
    for key in __SPACE_CONFIG.keys():
        __keys_index[key] = tuple(__SPACE_CONFIG[key].keys())
    
    @classmethod
    def get_format(cls, cell_type):
        """Get keys order of cell according to cell's type.
        Args:
            cell_type: string, type of cell defined in configuration
        Returns:
            format: tuple, keys order
        """
        return cls.__keys_index[cell_type]

    def __init__(self, *args):
        tuple.__init__(self)

    def __new__(self, *args):
        if len(args) == 1 and isinstance(args[0], tuple):
            args = args[0]
        return tuple.__new__(self, args)

    def __getnewargs__(self):
        return tuple.__getnewargs__(self)[0]

    def __getstate__(self):
        return [i for i in self]
    
    def __setstate__(self, state):
        self = Cell(*state)

    def __getattr__(self, key):
        """Get items though meaningful name
        if key is 'type', get the type of the cell.
        otherwise, get the value according to the type and the search space.
            
        """
        cell_type = self.__getitem__(0)
        if key == 'type':
            return cell_type
        key_i = self.__keys_index[cell_type].index(key) + 1
        return self.__getitem__(key_i)

if __name__ == "__main__":
    conv_space = NAS_CONFIG['spl']['space']['conv']
    cfmt = Cell.get_format('conv')
    tmp_list = ['conv']
    for ks_v in cfmt:
        ks = conv_space[ks_v]
        if isinstance(ks[0], list):
            tmp_list.append(ks[0][0])
        else:
            tmp_list.append(ks[0])
    cell = Cell(*tmp_list)
    print(cell)
