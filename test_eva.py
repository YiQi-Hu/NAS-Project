import unittest
from info_str import NAS_CONFIG
import random
from enumerater import Enumerater
from base import Cell
from evaluator import Evaluator
from base import NetworkItem
import copy
from info_str import NAS_CONFIG
from ddt import data, ddt

def _random_get_cell(_num):
    _dic = dict()
    _dic['conv'] = copy.deepcopy(NAS_CONFIG['spl']['conv_space'])
    tmp = []
    for i in NAS_CONFIG['spl']['conv_space']['filter_size']:
        tmp.extend(i)
    _dic['conv']['filter_size'] = tmp
    _dic['pooling'] = copy.deepcopy(NAS_CONFIG['spl']['pool_space'])
    res = []
    for _ in range(_num):
        _type = random.randint(0, 1)
        if _type == 0:
            tmp = ('conv', )
            for key in _dic['conv']:
                _r = random.randint(1, len(_dic['conv'][key]))
                tmp = tmp + (_dic['conv'][key][_r-1], )
            res.append(Cell(tmp[0], tmp[1], tmp[2], tmp[3]))
            # res.append(tmp)
        else:
            tmp = ('pooling', )
            for key in _dic['pooling']:
                _r = random.randint(1, len(_dic['pooling'][key]))
                tmp = tmp + (_dic['pooling'][key][_r-1],)
            res.append(Cell(tmp[0], tmp[1], tmp[2]))
            # res.append(tmp)
    return res

@ddt
class Test_eva(unittest.TestCase):
    global test_info
    test_info = []
    for i in range(10):
        _depth = random.randint(3, 25)
        _width = random.randint(0, 1)
        _max_depth = random.randint(0, _depth)
        # print('##', self._depth, self._width, self._max_depth)
        NAS_CONFIG['enum']['depth'] = _depth
        NAS_CONFIG['enum']['width'] = _width
        NAS_CONFIG['enum']['max_depth'] = _max_depth
        enum = Enumerater()
        _network_list = enum.enumerate()
        ind = random.randint(0, len(_network_list)-1)
        _graph_part = _network_list[ind].graph_template
        # for i in self._network_list[ind].graph_template:
        #     if i:
        #         self._graph_part.append(i)
        #     else:
        #         self._graph_part.append([len(self._network_list[ind].graph_template)])
        # print(self._graph_part)
        _cell_list = _random_get_cell(len(_graph_part))
        test_info.append((_graph_part, _cell_list))

    with open('./test/test_eva.txt', 'w') as op:
        for i in test_info:
            op.writelines('graph:' + str(i[0]) + '\n')
            op.writelines('cell_list:' + str(i[1]) + '\n')
            op.writelines('###'*20 + '\n')

    def _run_module(self, _graph_part, _cell_list):
        eva = Evaluator()
        eva.add_data(500)
        tmp = NetworkItem(0, _graph_part, _cell_list, "")
        return eva.evaluate(tmp, is_bestNN=True)

    def _judge_score(self, score):
        self.assertEqual(float, type(score))

    @data(*test_info)
    def test_res(self, para):
        print(para[0])
        print(para[1])
        score = self._run_module(para[0], para[1])
        self._judge_score(score)


if __name__ == "__main__":
    unittest.main()
