import random
from .sampling.load_configuration import load_conf
from .optimizer import Dimension
import pickle
from optimizer import Optimizer
from optimizer import RacosOptimization
import numpy as np
from evaluater import Evaluater
from multiprocessing import Process,Pool
import multiprocessing
from base import NetworkUnit
import time
import pickle
import copy

class Sampler_table:

    def __init__(self, graph_part):

        '''
        :param nn: NetworkUnit
        '''

        # 设置结构大小
        self.node_number = len(graph_part)
        # 读取配置表得到操作的对应映射
        self.setting, self.pros, self.parameters_subscript_node, = load_conf()
        self.dic_index = self._init_dict()

        # print(len(self.pros))
        self.p = []

        # 设置优化Dimension
        # 设置优化的参数
        self.__region, self.__type = self.opt_parameters()
        self.dim = Dimension()
        self.dim.set_dimension_size(len(self.__region))
        self.dim.set_regions(self.__region, self.__type)
        self.parameters_subscript = []  #

    # 更新p
    def renewp(self, newp):
        self.p = newp

    # 基于操作的对应映射得到优化参数
    def opt_parameters(self):
        __type_tmp = []
        __region_tmp = [[0, 0] for _ in range(len(self.dic_index))]
        for key in self.dic_index:
            tmp = int(self.dic_index[key][1] - self.dic_index[key][0])
            __region_tmp[self.dic_index[key][-1]] = [0, tmp]
            __type_tmp.append(2)

        __region = []
        __type = []
        for i in range(self.node_number):
            __region = __region + __region_tmp
            __type.extend(__type_tmp)
        return __region, __type

    def sample(self):
        res = []
        # 基于节点的搜索结构参数
        for num in range(self.node_number):
            # 取一个节点大小的概率
            p_node = self.p[num*len(self.dic_index):(num+1)*len(self.dic_index)]
            first = p_node[self.dic_index['conv'][-1]]
            # first = p_node[0]
            tmp = ()
            # 首位置确定 conv 还是 pooling
            if first == 0:
                # 搜索conv下的操作
                # 基于操作的对应映射取配置所在的地址，进行取值
                tmp = tmp + ('conv',)
                struct_conv = ['conv filter_size', 'conv kernel_size', 'conv activation']
                for key in struct_conv:
                    tmp = tmp + (self.setting['conv'][key.split(' ')[-1]]['val'][p_node[self.dic_index[key][-1]]],)
            else:
                # 搜索pooling下的操作
                # 基于操作的对应映射取配置所在的地址，进行取值
                tmp = tmp + ('pooling',)
                struct_pooling = ['pooling pooling_type', 'pooling kernel_size']
                for key in struct_pooling:
                    tmp = tmp + (self.setting['pooling'][key.split(' ')[-1]]['val'][p_node[self.dic_index[key][-1]]],)
            res.append(tmp)
        return res

    # 基于opt.sample()所得结果，基于位置得到操作
    def _init_dict(self):
        dic = {}
        dic['conv'] = (0, 1, 0)
        cnt = 1
        num = 1
        for key in self.setting:
            for k in self.setting[key]:
                tmp = len(self.setting[key][k]['val']) - 1
                dic[key + ' ' + k] = (cnt, cnt + tmp, num)
                num += 1
                cnt += tmp

        return dic

    def get_dim(self):
        return self.dim

    def get_parametets_subscript(self):
        return self.parameters_subscript

    # log
    def get_cell_log(self, POOL, PATH, date):
        for i, j in enumerate(POOL):
            s = 'nn_param_' + str(i) + '_' + str(date)
            fp = open(PATH + s, "wb")
            # print(s)
            pickle.dump(j.cell_list, fp)

