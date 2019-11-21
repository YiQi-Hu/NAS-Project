import unittest
from enumerater import Enumerater
import random
from predictor import Predictor
import os
from ddt import ddt, data
import copy
from info_str import NAS_CONFIG

@ddt
class Test_pred(unittest.TestCase):
    global test_info
    test_info = []
    for i in range(10):
        test_info.append(i)

    def _data_sample(self, network_pool):
        block_n = random.randint(0,3)
        graph_id = random.randint(1, len(network_pool))
        blocks = []
        for i in range(block_n):
            id = random.randint(0,len(network_pool))
            blocks.append(network_pool[id].graph_template)
        graph = network_pool[graph_id].graph_template
        return blocks, graph

    @data(*test_info)
    def test_predictor(self):
        pred = Predictor()
        _depth = random.randint(0, 25)
        _width = random.randint(0, 1)
        _max_depth = random.randint(0, _depth)
        # print('##', self._depth, self._width, self._max_depth)
        NAS_CONFIG['enum']['depth'] = _depth
        NAS_CONFIG['enum']['width'] = _width
        NAS_CONFIG['enum']['max_depth'] = _max_depth
        enum = Enumerater()
        network_pool = enum.enumerate()
        blocks, graph = self._data_sample(network_pool)
        ops = pred.predictor(blocks, graph)
        self.assertEqual(list, type(ops))
        self.assertTrue(len(ops) == len(graph))
        for op in ops:
            if op[0] != 'pooling':
                self.assertGreater(int(op[0]), 1)
                self.assertLessEqual(int(op[0]), 1024)
                self.assertGreater(int(op[1]), 0)
                self.assertLessEqual(int(op[1]), 11)


if __name__ == '__main__':
    for i in range(1):
        unittest.main()
