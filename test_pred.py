import unittest
from enumerater import Enumerater
import random
from predictor import Predictor
import os
class Test_pred(unittest.TestCase):

    def setUp(self):
        pwd = os.path.abspath(os.path.dirname(os.getcwd()))
        os.chdir(pwd)

    def _data_sample(self, network_pool):
        block_n = random.randint(0,3)
        graph_id = random.randint(1, len(network_pool))
        blocks = []
        for i in range(block_n):
            id = random.randint(0,len(network_pool))
            blocks.append(network_pool[id].graph_template)
        graph = network_pool[graph_id].graph_template
        return blocks, graph

    def test_predictor(self):
        pred = Predictor()
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
    unittest.main()
