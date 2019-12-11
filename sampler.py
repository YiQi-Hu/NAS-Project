from optimizer import Dimension
from optimizer import Optimizer
import pickle
import copy
from queue import Queue
from base import Cell
from info_str import NAS_CONFIG

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
        self._p_table = []  # initializing the table value in Sampler.
        self._graph_part = graph_part
        self._node_number = len(self._graph_part)
        self._pattern = NAS_CONFIG['nas_main']['pattern']  #  Parameter setting based on search method
        self._crosslayer_dis = NAS_CONFIG['spl']['skip_max_dist'] + 1  # dis control
        self._cross_node_number = NAS_CONFIG['spl']['skip_max_num']
        self._graph_part_invisible_node = self.graph_part_add_invisible_node()
        self._crosslayer = self._get_crosslayer()
        # Read parameter table to get operation dictionary in stage(block_id)
        self._setting = dict()
        self._setting['conv'] = copy.deepcopy(NAS_CONFIG['spl']['conv_space'])
        self._setting['pooling'] = copy.deepcopy(NAS_CONFIG['spl']['pool_space'])
        self._setting['sep_conv'] = copy.deepcopy(NAS_CONFIG['spl']['sep_conv_space'])
        if self._pattern == "Block":
            self._setting['conv']['filter_size'] = \
                self._setting['conv']['filter_size'][block_id]
            self._setting['sep_conv']['filter_size'] = \
                self._setting['sep_conv']['filter_size'][block_id]
        self._dic_index = self._init_dict()  # check
        # Set parameters of optimization module based on the above results
        self.__region, self.__type = self._opt_parameters()
        self.__dim = Dimension()
        self.__dim.set_dimension_size(len(self.__region))    # 10%
        self.__dim.set_regions(self.__region, self.__type)
        self.__parameters_subscript = []  #
        self.opt = Optimizer(self.__dim, self.__parameters_subscript)
        opt_para = copy.deepcopy(NAS_CONFIG["opt"])
        __sample_size = opt_para["sample_size"]  # the instance number of sampling in an iteration
        __budget = opt_para["budget"]  # budget in online style
        __positive_num = opt_para["positive_num"]  # the set size of PosPop
        __rand_probability = opt_para["rand_probability"]  # the probability of sample in model
        __uncertain_bit = opt_para["uncertain_bit"]  # the dimension size that is sampled randomly
        self.opt.set_parameters(ss=__sample_size, bud=__budget,
                                pn=__positive_num, rp=__rand_probability, ub=__uncertain_bit)
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
        """
        Optimization of sampling space based on Evaluation and optimization method.
        Args:
            1. table (1d int list, depending on dimension)
            2. score（float, 0 ~ 1.0)
        No returns.
        """
        self.opt.update_model(table, score)

    def graph_part_add_invisible_node(self):
        graph_part_tmp = []
        for i in self._graph_part:
            if not i:
                graph_part_tmp.append([self._node_number])
            else:
                graph_part_tmp.append(i)
        graph_part_tmp.append([])
        return graph_part_tmp

    def _get_crosslayer(self):
        """
        utilizing breadth-first search to set the possible cross layer
        connection for each node of the network topology.
        """
        cl = []
        for i in range(self._node_number):
            cl.append(self._bfs(i))
        return cl

    def _bfs(self, node_id):
        res_list = []
        q = Queue()
        q.put([node_id, 0])
        v = []
        for i in range(self._node_number + 1):
            v.append(0)

        while q.empty() is False:
            f = q.get()
            if f[1] >= 2:
                if f[1] <= self._crosslayer_dis:
                    res_list.append(f[0])
                else:
                    continue

            # for j in self._graph_part[f[0]]:
            for j in self._graph_part_invisible_node[f[0]]:
                if v[j] == 0:
                    q.put([j, f[1] + 1])
                    v[j] = 1

        return res_list
    #
    def _region_cross_type(self, __region_tmp, __type_tmp, i):
        region_tmp = copy.copy(__region_tmp)
        type_tmp = copy.copy(__type_tmp)
        for j in range(self._cross_node_number):

            region_tmp.append([0, len(self._crosslayer[i])])
            type_tmp.append(2)

        return region_tmp, type_tmp

    def _opt_parameters(self):
        """Get the parameters of optimization module based on parameter document."""
        __type_tmp = []
        __region_tmp = []
        for i in range(len(self._dic_index)):
            __region_tmp.append([0, 0])
        # __region_tmp = [[0, 0] for _ in range(len(self._dic_index))]
        for key in self._dic_index:
            tmp = int(self._dic_index[key][1] - self._dic_index[key][0])
            __region_tmp[self._dic_index[key][-1]] = [0, tmp]
            __type_tmp.append(2)

        __region = []
        __type = []
        for i in range(self._node_number):
            __region_cross, __type_cross = \
                self._region_cross_type(__region_tmp, __type_tmp, i)

            __region = __region + __region_cross
            __type.extend(__type_cross)

        return __region, __type

    def convert(self, table_tmp):
        """Search corresponding operation configuration based on table."""
        self._p_table = copy.deepcopy(table_tmp)
        res = []
        l = 0
        r = 0
        graph_part_sample = copy.deepcopy(self._graph_part)
        for num in range(self._node_number):
            l = r
            r = l + len(self._dic_index) + self._cross_node_number
            p_node = self._p_table[l:r]  # Take the search space of a node
            node_cross_tmp = list(set(copy.deepcopy(p_node[len(self._dic_index):])))
            for i in node_cross_tmp:
                if i != 0:
                    graph_part_sample[num].append(self._crosslayer[num][i - 1])
            if not graph_part_sample[num]:
                graph_part_sample[num].append(self._node_number)

            first = p_node[self._dic_index['type'][-1]]
            tmp = ()
            if self._pattern == "Block" and first == 1:
                first = 0
            if first == 0 or first == 2:  # Search operation under conv
                if first == 0:
                    conv_type = 'conv'
                else:
                    conv_type = 'sep_conv'
                tmp = tmp + (conv_type,)
                space_conv = ['filter_size', 'kernel_size', 'activation']
                for key in space_conv:
                    tmp = tmp + (self._setting[conv_type][key][p_node[self._dic_index[conv_type + ' ' + key][-1]]],)
                tmp = Cell(tmp[0], tmp[1], tmp[2], tmp[3])
            else:  # Search operation under pooling
                tmp = tmp + ('pooling',)
                space_pool = ['pooling_type', 'kernel_size']
                for key in space_pool:
                # for key in self._setting['pooling']:
                    tmp = tmp + (self._setting['pooling'][key][p_node[self._dic_index['pooling ' + key][-1]]],)
                tmp = Cell(tmp[0], tmp[1], tmp[2])
            res.append(tmp)
        return res, graph_part_sample

    def _init_dict(self):
        """Operation space dictionary based on parameter file."""
        dic = {}
        dic['type'] = (0, len(self._setting)-1, 0)
        cnt = 1
        num = 1
        for key in self._setting:
            for k in self._setting[key]:
                tmp = len(self._setting[key][k]) - 1
                dic[key + ' ' + k] = (cnt, cnt + tmp, num)
                num += 1
                cnt += tmp

        return dic

    # log
    def _get_cell_log(self, POOL, PATH, date):
        for i, j in enumerate(POOL):
            s = 'nn_param_' + str(i) + '_' + str(date)
            fp = open(PATH + s, "wb")
            # print(s)
            pickle.dump(j.cell_list, fp)

    def ops2table(self, ops, table_tmp):
        """
        set the table under the output in predictor
        the output in predictor looks like:
        [['64', '7'], ['pooling'], ['64', '3'], ['256', '3'], ['1024', '1'],
        ['1024', '1'], ['1024', '3'], ['1024', '3'], ['1024', '3'], ['512', '1'],
        ['128', '5'], ['64', '3'], ['1024', '1'], ['1024', '1'], ['256', '3']]
        """
        self._p_table = copy.deepcopy(table_tmp)
        table = []
        l = 0
        r = 0
        for num in range(self._node_number):  # Take the search space of a node
            l = r
            r = l + len(self._dic_index) + self._cross_node_number
            p_node = self._p_table[l:r]  # Sample value of the current node
            if len(ops[num]) != 1:
                p_node = self._p_table[l:r]  # Sample value of the current node
                p_node[self._dic_index['type'][-1]] = 0
                for j, i in enumerate(self._setting['conv']['filter_size']):
                    if str(i) == ops[num][0]:
                        p_node[self._dic_index['conv filter_size'][-1]] = j
                for j, i in enumerate(self._setting['conv']['kernel_size']):
                    if str(i) == ops[num][1]:
                        p_node[self._dic_index['conv kernel_size'][-1]] = j
                for j, i in enumerate(self._setting['conv']['activation']):
                    if i == 'relu':
                        p_node[self._dic_index['conv activation'][-1]] = j
                table = table + p_node
            else:
                if self._pattern == "Global":
                    p_node[self._dic_index['type'][-1]] = 1
                table = table + p_node
        return table


if __name__ == '__main__':
    # os.chdir("../")
    graph_part = [[1], [2], [3, 7, 9], [4], [5], [6], [], [8], [5], [10], [11], [6]]
    # graph_part = [[1], [2], [3], [4], [5], [6], []]

    # network.init_sample(self.__pattern, block_num, self.spl_setting, self.skipping_max_dist, self.ops_space)

    spl = Sampler(graph_part, 0)

    res, graph_part_sample, table_present = spl.sample()
    print('skip_max_dist:', NAS_CONFIG['spl']['skip_max_dist'])
    print(spl._crosslayer)

    region, type = spl._opt_parameters()
    print(spl._dic_index)
    print(len(region), len(type))
    print(region, type)
    print(len(spl._p_table))
    print(spl._p_table)
    print(res)
    print(graph_part_sample)

    init = [['64', '7'], ['pooling'], ['64', '3'], ['256', '3'], ['1024', '1'],
            ['1024', '1'], ['1024', '3'], ['1024', '3'], ['1024', '3'], ['512', '1'],
            ['128', '5'], ['64', '3'], ['1024', '1'], ['1024', '1'], ['256', '3']
            ]
    table_present = spl.ops2table(init, table_present)
    print(spl._p_table)

    res, graph_part_sample = spl.convert(table_present)
    print(res)
    print(graph_part_sample)

    print('##############################'*10)
    score = -0.199
    spl.update_opt_model([table_present], [score])  # score +-
    res, graph_part_sample, table_present = spl.sample()
    print(res)
    print(graph_part_sample)
    print('##############################'*10)
    spl.update_opt_model([table_present], [score])  # score +-
    res, graph_part_sample, table_present = spl.sample()
    print(res)
    print(graph_part_sample)