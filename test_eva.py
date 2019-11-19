import unittest
from info_str import NAS_CONFIG
import random
from enumerater import Enumerater
from base import Cell
from evaluator import Evaluator
from base import NetworkItem


def _random_get_cell(_num):
    _dic = dict()
    _dic['conv'] = NAS_CONFIG['spl']['conv_space']
    tmp = []
    for i in NAS_CONFIG['spl']['conv_space']['filter_size']:
        tmp.extend(i)
    _dic['conv']['filter_size'] = tmp
    _dic['pooling'] = NAS_CONFIG['spl']['pool_space']
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


class Test_eva(unittest.TestCase):
    def setUp(self):
        self._depth = random.randint(0, 25)
        self._width = random.randint(0, 1)
        self._max_depth = random.randint(0, self._depth)
        # print('##', self._depth, self._width, self._max_depth)
        NAS_CONFIG['enum']['depth'] = self._depth
        NAS_CONFIG['enum']['width'] = self._width
        NAS_CONFIG['enum']['max_depth'] = self._max_depth
        enum = Enumerater()
        self._network_list = enum.enumerate()
        ind = random.randint(0, len(self._network_list))
        self._graph_part = self._network_list[ind].graph_template
        # for i in self._network_list[ind].graph_template:
        #     if i:
        #         self._graph_part.append(i)
        #     else:
        #         self._graph_part.append([len(self._network_list[ind].graph_template)])
        # print(self._graph_part)
        self._cell_list = _random_get_cell(len(self._graph_part))


    def _run_module(self):
        eva = Evaluator()
        eva.add_data(500)
        tmp = NetworkItem(0, self._graph_part, self._cell_list, "")
        return eva.evaluate(tmp, is_bestNN=True)

    def _judge_score(self, score):
        self.assertEqual(float, type(score))

    def test_res(self):
        score = self._run_module()
        self._judge_score(score)


if __name__ == "__main__":
    unittest.main()
