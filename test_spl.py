import unittest
from info_str import NAS_CONFIG
import random
from enumerater import Enumerater
from base import Network
from base import Cell
from sampler import Sampler
from ddt import data, ddt
# from predictor import Predictor


@ddt
class Test_spl(unittest.TestCase):
    global test_info
    test_info = []
    for i in range(1000):
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
        test_info.append([_graph_part])

    with open('./test/test_spl.txt', 'w') as op:
        for i in test_info:
            op.writelines('graph:' + str(i[0]) + '\n')
            op.writelines('###'*20 + '\n')

    def _run_module(self, graph_part):
        spl = Sampler(graph_part, 0)
        cell, graph, table = spl.sample()
        # pred = Predictor()
        # ops = pred.predictor([], graph_part)
        # table_ops = spl.ops2table(ops, table)
        # return cell, graph, table, table_ops
        return cell, graph, table

    def _judge_cell(self, cell):
        self.assertEqual(list, type(cell))
        for i in cell:
            self.assertEqual(Cell, type(i))

    def _judge_graph(self, graph):
        self.assertEqual(list, type(graph))

    def _judge_table(self, table):
        self.assertEqual(list, type(table))

    @data(*test_info)
    def test_res(self, para):
        # print('####', para[0])
        # cell, graph, table, table_ops = self._run_module(para[0])
        cell, graph, table = self._run_module(para[0])
        self._judge_cell(cell)
        self._judge_graph(graph)
        self._judge_table(table)
        # self._judge_table(table_ops)


if __name__ == "__main__":
    unittest.main()


