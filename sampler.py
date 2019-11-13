import random
from optimizer import Dimension
import pickle
from optimizer import Optimizer
from optimizer import RacosOptimization
import numpy as np
from multiprocessing import Process,Pool
import multiprocessing
# from .base import NetworkUnit
import time
import pickle
import copy
from queue import Queue
import json
import os

from info_str import NAS_CONFIG

SPL_CONFIG = NAS_CONFIG['spl']
SPACE_CONFIG = NAS_CONFIG['space']


class Sampler:

    def __init__(self, graph_part, block_id):
        """
            Generate adjacency of network topology.
            Sampling network operation based on sampled value(table).
            The sampling value is updated by the optimization module
            based on the output given by the evaluation module.
            Attributes:
                graph_part: a Network Topology(Adjacency table).
                block_id: The stage of neural network search.
                Other important operation information and parameters of optimization module
                are given by folder 'parameters'.
        """
        # initializing the table value in Sampler.
        self.p_table = []
        self.graph_part = graph_part
        # get the node number in graph part.
        self.node_number = len(self.graph_part)

        # Parameter setting based on search method
        self.pattern = SPL_CONFIG['pattern']
        self.crosslayer_dis = SPACE_CONFIG['skipping_max_dist']
        self.cross_node_number = SPACE_CONFIG['skipping_max_num']
        # self.cross_node_range = cross_node_range

        # Set the possible cross layer connection for each node of the network topology
        self.crosslayer = self._get_crosslayer()

        # Read parameter table to get operation dictionary in stage(block_id)
        self.setting = copy.deepcopy(SPACE_CONFIG['ops'])

        if self.pattern == "Block":
            # TODO The number of words in this line is over 80 characters.
            self.setting['conv']['filter_size'] = self.setting['conv']['filter_size'][block_id]

        self.dic_index = self._init_dict()  # check

        # Set parameters of optimization module based on the above results
        self.__region, self.__type = self._opt_parameters()
        self.dim = Dimension()
        self.dim.set_dimension_size(len(self.__region))    # 10%
        self.dim.set_regions(self.__region, self.__type)
        self.parameters_subscript = []  #

        self.opt = Optimizer(self.dim, self.parameters_subscript)
        opt_para = SPL_CONFIG["opt_para"]
        sample_size = opt_para["sample_size"]  # the instance number of sampling in an iteration
        budget = opt_para["budget"]  # budget in online style
        positive_num = opt_para["positive_num"]  # the set size of PosPop
        rand_probability = opt_para["rand_probability"]  # the probability of sample in model
        uncertain_bit = opt_para["uncertain_bit"]  # the dimension size that is sampled randomly
        # set hyper-parameter for optimization, budget is useless for single step optimization
        # TODO The number of words in this line is over 80 characters.
        self.opt.set_parameters(ss=sample_size, bud=budget, pn=positive_num, rp=rand_probability, ub=uncertain_bit)
        self.opt.clear()  # clear optimization model

    def sample(self):
        """
        Get table based on the optimization module sampling,
        update table in Sampler,
        and sample the operation configuration.
        No Args.
        Retruns:
            1. cell (1d Cell list)
            2. graph_full (2d int list, as NetworkItem.graph_full)
            3. table (1d int list, depending on dimension)
        """

        table = self.opt.sample()
        cell, graph = self.convert(table)
        return cell, graph, table

    def update_opt_model(self, table, score):
        # TODO Please give the function's discrption.
        """
        As title.
        Args:
            1. table (1d int list, depending on dimension)
            2. score（float, 0 ~ 1.0)
        No returns.
        """
        # TODO optimier.update_model returns NOTHING ???????????? What is 'result' ?
        result = self.opt.update_model(table, score)  # here "-" represent that we minimize the loss
        if result:
            pos, neg = result
            print("###################pos  set#######################")
            for sam in pos:
                print(self.convert(sam))
            print("###################neg  set#######################")
            for sam in neg:
                print(self.convert(sam))

    def _get_crosslayer(self):
        """
        utilizing breadth-first search to set the possible cross layer
        connection for each node of the network topology.
        """
        cl = []
        for i in range(len(self.graph_part)):
            cl.append(self._bfs(i))
        return cl

    def _bfs(self, node_id):
        res_list = []
        q = Queue()
        q.put([node_id, 0])
        v = []
        for i in range(len(self.graph_part) + 1):
            v.append(0)
        # v = [0 for ii in range(len(self.graph_part) + 1)]

        while q.empty() is False:
            f = q.get()
            if f[1] >= 2:
                if f[1] <= self.crosslayer_dis:
                    res_list.append(f[0])
                else:
                    continue

            for j in self.graph_part[f[0]]:
                if v[j] == 0:
                    q.put([j, f[1] + 1])
                    v[j] = 1

        return res_list
    #
    def _region_cross_type(self, __region_tmp, __type_tmp, i):
        region_tmp = copy.copy(__region_tmp)
        type_tmp = copy.copy(__type_tmp)
        for j in range(self.cross_node_number):

            region_tmp.append([0, len(self.crosslayer[i])])
            type_tmp.append(2)

        return region_tmp, type_tmp

    def _opt_parameters(self):
        """Get the parameters of optimization module based on parameter document."""
        __type_tmp = []
        __region_tmp = []
        for i in range(len(self.dic_index)):
            __region_tmp.append([0, 0])
        # __region_tmp = [[0, 0] for _ in range(len(self.dic_index))]
        for key in self.dic_index:
            tmp = int(self.dic_index[key][1] - self.dic_index[key][0])
            __region_tmp[self.dic_index[key][-1]] = [0, tmp]
            __type_tmp.append(2)

        __region = []
        __type = []
        for i in range(self.node_number):
            # TODO Over 80 characters
            __region_cross, __type_cross = self._region_cross_type(__region_tmp, __type_tmp, i)

            __region = __region + __region_cross
            __type.extend(__type_cross)

        return __region, __type

    def convert(self, table_tmp):
        """Search corresponding operation configuration based on table."""
        self.p_table = copy.deepcopy(table_tmp)
        res = []
        l = 0
        r = 0
        graph_part_sample = copy.deepcopy(self.graph_part)

        for num in range(self.node_number):
            # Take the search space of a node
            l = r
            r = l + len(self.dic_index) + self.cross_node_number
            p_node = self.p_table[l:r]

            # print(p_node)
            # print(p_node[len(self.dic_index):])
            node_cross_tmp = []
            for i in range(len(p_node[len(self.dic_index):])):
                node_cross_tmp.append(p_node[len(self.dic_index):][i])

            node_cross_tmp = list(set(node_cross_tmp))
            # print('##', self.crosslayer[num])
            # print(node_cross_tmp)

            # Getting cross layer connection of graph_part based on table
            for i in node_cross_tmp:
                if len(self.crosslayer[num]) > i:
                    graph_part_sample[num].append(self.crosslayer[num][i])

            first = p_node[self.dic_index['conv'][-1]]
            tmp = ()
            # TODO Please write in other way between line 215 to 229.
            if self.pattern == "Block":
                first = 0
            if first == 0:
                # Search operation under conv
                tmp = tmp + ('conv',)
                struct_conv = ['conv filter_size', 'conv kernel_size', 'conv activation']
                for key in struct_conv:
                    tmp = tmp + (self.setting['conv'][key.split(' ')[-1]][p_node[self.dic_index[key][-1]]],)
            else:
                # Search operation under pooling
                tmp = tmp + ('pooling',)
                struct_pooling = ['pooling pooling_type', 'pooling kernel_size']
                for key in struct_pooling:
                    tmp = tmp + (self.setting['pooling'][key.split(' ')[-1]][p_node[self.dic_index[key][-1]]],)
            res.append(tmp)

        return res, graph_part_sample

    def _init_dict(self):
        """Operation space dictionary based on parameter file."""
        dic = {}
        dic['conv'] = (0, 1, 0)
        cnt = 1
        num = 1
        for key in self.setting:
            for k in self.setting[key]:
                tmp = len(self.setting[key][k]) - 1
                dic[key + ' ' + k] = (cnt, cnt + tmp, num)
                num += 1
                cnt += tmp

        return dic

    # log
    def get_cell_log(self, POOL, PATH, date):
        for i, j in enumerate(POOL):
            s = 'nn_param_' + str(i) + '_' + str(date)
            fp = open(PATH + s, "wb")
            # print(s)
            pickle.dump(j.cell_list, fp)

    # TODO(pjs) move to predictor(advice: duplicate the variable('self.')
    #  and We can move inti_p out completely)
    def ops2table(self, ops, table_tmp):
        """
        set the table under the output in predictor
        the output in predictor looks like:
        [['64', '7'], ['pooling'], ['64', '3'], ['256', '3'], ['1024', '1'],
        ['1024', '1'], ['1024', '3'], ['1024', '3'], ['1024', '3'], ['512', '1'],
        ['128', '5'], ['64', '3'], ['1024', '1'], ['1024', '1'], ['256', '3']]
        """
        self.p_table = copy.deepcopy(table_tmp)
        table = []
        l = 0
        r = 0
        for num in range(self.node_number):
            # Take the search space of a node
            l = r
            r = l + len(self.dic_index) + self.cross_node_number
            p_node = self.p_table[l:r]  # Sample value of the current node
            if len(ops[num]) != 1:
                # struct_conv = ['conv filter_size', 'conv kernel_size', 'conv activation']
                a = -1
                b = -1
                c = -1
                for j, i in enumerate(self.setting['conv']['filter_size']):
                    if str(i) == ops[num][0]:
                        a = j
                for j, i in enumerate(self.setting['conv']['kernel_size']):
                    if str(i) == ops[num][1]:
                        b = j
                for j, i in enumerate(self.setting['conv']['activation']):
                    if i == 'relu':
                        c = j
                p_node[self.dic_index['conv'][-1]] = 0
                if a != -1:
                    p_node[self.dic_index['conv filter_size'][-1]] = a
                if b != -1:
                    p_node[self.dic_index['conv kernel_size'][-1]] = b
                if c != -1:
                    p_node[self.dic_index['conv activation'][-1]] = c
                table = table + p_node
            else:
                if self.pattern == "Global":
                    p_node[self.dic_index['conv'][-1]] = 1
                table = table + p_node

        return table


if __name__ == '__main__':
    # os.chdir("../")
    PATTERN = "Global"
    setting = load_config(PATTERN)
    spl_setting = setting["spl_para"]
    skipping_max_dist = SPACE_CONFIG["skipping_max_dist"]
    ops_space = SPACE_CONFIG["ops"]
    graph_part = [[1], [2], [3], [4], [5], [6], [7], [8], [9], []]

    # network.init_sample(self.__pattern, block_num, self.spl_setting, self.skipping_max_dist, self.ops_space)

    spl = Sampler(graph_part, 0)

    res, graph_part_sample, table_present = spl.sample()
    region, type = spl._opt_parameters()
    print(spl.dic_index)
    print(len(region), len(type))
    print(region, type)
    print(len(spl.p_table))
    print(spl.p_table)
    print(res)
    print(graph_part_sample)

    init = [['64', '7'], ['pooling'], ['64', '3'], ['256', '3'], ['1024', '1'],
            ['1024', '1'], ['1024', '3'], ['1024', '3'], ['1024', '3'], ['512', '1'],
            ['128', '5'], ['64', '3'], ['1024', '1'], ['1024', '1'], ['256', '3']
            ]
    table_present = spl.ops2table(init, table_present)
    print(spl.p_table)

    res, graph_part_sample = spl.convert(table_present)
    print(res)
    print(graph_part_sample)

    print('##############################')
    score = -0.199
    spl.update_opt_model([table_present], [score]) # score +-
    res, graph_part_sample, table_present = spl.sample()

    res, graph_part_sample = spl.convert(table_present)
    print(res)
    print(graph_part_sample)