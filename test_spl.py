import unittest
from info_str import NAS_CONFIG
import random
from enumerater import Enumerater
from base import Network
from base import Cell
from sampler import Sampler

class Test_spl(unittest.TestCase):
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

    def _run_module(self):
        spl = Sampler(self._graph_part, 0)
        return spl.sample()

    def _judge_cell(self, cell):
        self.assertEqual(list, type(cell))
        for i in cell:
            self.assertEqual(Cell, type(i))

    def _judge_graph(self, graph):
        self.assertEqual(list, type(graph))

    def _judge_table(self, table):
        self.assertEqual(list, type(table))

    def test_res(self):
        cell, graph, table = self._run_module()
        self._judge_cell(cell)
        self._judge_graph(graph)
        self._judge_table(table)


if __name__ == "__main__":
    unittest.main()


