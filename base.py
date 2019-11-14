"""
All basic data structure in this file.
PLEASE DO NOT USE 'from .base import *' !!!
"""


class Network(object):
    pre_block = []

    def __init__(self, _id, graph_tmp):
        self.id = _id
        self.graph_template = graph_tmp
        self.item_list = []
        self.pre_block = []
        self.spl = None

        return


class NetworkItem(object):
    def __init__(self, _id, graph, cell_list, code):
        self.id = _id
        self.graph = graph
        self.cell_list = cell_list
        self.code = code
        self.score = 0

        return


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
    __keys_index = {
        "conv":{
            "filter_size": 1,
            "kernel_size": 2,
            "activation": 3
        },
        "pooling":{
            "ptype": 1,
            "kernel_size": 2
        }
    }

    def __init__(self, *args):
        tuple.__init__(self)
    
    def __new__(self, *args):
        Cell._check_vaild(args)
        return tuple.__new__(self, args)
    
    def __getattr__(self, key):
        """Get items though meaningful name
        if key is 'type':
            cell[0] == cell.type
        if type is 'conv':
            cell[1] == cell.filter_size
            cell[2] == cell.kernel_size
            cell[3] == cell.activation
        if type is 'pooling':
            cell[1] == cell.ptype
            cell[2] == cell.kernel_size
        Note: Other Keys will raise KeyError.
        """
        cell_type = self.__getitem__(0)
        if key == 'type':
            return cell_type
        return self.__getitem__(self.__keys_index[cell_type][key])

    @staticmethod
    def _check_vaild(args):
        cell_type = args[0]
        type_args = args[1:]
        if cell_type is 'conv':
            Cell._conv_vaild(type_args)
        elif cell_type is 'pooling':
            Cell._pool_valid(type_args)
        else:
            raise CellInitError('type error')
        return
    
    @staticmethod
    def _conv_vaild(args):
        err_msg = 'cell type \'conv\' %s.'

        if (len(args) > 3):
            raise CellInitError(err_msg % 'args num > 3')
        # fileter_size, kernel_size, activation
        fs, ks, at = args
        # '_tv' -> 'type valid'
        fs_tv = isinstance(fs, int)
        ks_tv = isinstance(ks, int)
        at_tv = isinstance(at, str)

        Cell._check_condition(fs_tv, ks_tv, at_tv, err_msg % 'arg type invalid') 
        # '_rv' -> 'range valid'
        fs_rv = (fs in range(1, 1025))
        ks_rv = (ks % 2 == 1) and (ks in range(1, 10))
        at_rv = (at in ['relu', 'tanh', 'sigmoid', 'identity', 'leakrelu'])

        Cell._check_condition(fs_rv, ks_rv, at_rv, err_msg % 'arg type invalid') 
        return
    
    @staticmethod
    def _pool_valid(args):
        err_msg = 'cell type \'pooling\' %s'
        if (len(args) > 2):
            return CellInitError(err_msg % 'args num > 2')
        # ptype, kernel_size
        pt, ks = args 
        # '_tv' -> 'type valid'
        pt_tv = isinstance(pt, str)
        ks_tv = isinstance(ks, int)

        Cell._check_condition(pt_tv, ks_tv, err_msg % 'arg type invalid')
        # '_rv' -> 'range valid'
        pt_rv = (pt in ['avg', 'max', 'global'])
        ks_rv = (ks in range(1, 11))

        Cell._check_condition(pt_rv, ks_rv, err_msg % 'arg range invalid')
        return

    @staticmethod
    def _check_condition(*args):
        msg = args[-1]
        if False in args[:-1]:
            raise CellInitError(msg)
        return


class CellInitError(Exception):
    def __init__(self, msg):
        Exception.__init__(self, msg)


class NetworkUnit(object):
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
        from optimizer import Optimizer
        if pattern == "Block":
            from sampler_block import Sampler
            self.spl = Sampler(self.graph_part, len(self.graph_part), block_num)
        else:
            from sampler_global import Sampler
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

if __name__ == "__main__":
    cc = Cell('conv', 1024, 7, 'relu') # conv cell
    pc = Cell('pooling', 'avg', 10) # pooling cell

    print(cc.kernel_size)
    print(pc.type)
    print(cc)
    print(pc)