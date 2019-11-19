import unittest
from info_str import NAS_CONFIG
import random
from enumerater import Enumerater
from base import Network


class Test_enum(unittest.TestCase):
    def setUp(self):
        self._depth = random.randint(0, 25)
        self._width = random.randint(0, 1)
        self._max_depth = random.randint(0, self._depth)
        # print('##', self._depth, self._width, self._max_depth)

    def _run_module(self):
        NAS_CONFIG['enum']['depth'] = self._depth
        NAS_CONFIG['enum']['width'] = self._width
        NAS_CONFIG['enum']['max_depth'] = self._max_depth
        enum = Enumerater()
        return enum.enumerate()

    def _judge_res(self, res):
        self.assertEqual(list, type(res))
        for _net in res:
            self.assertEqual(Network, type(_net))
            for _graph in _net.graph_template:
                self.assertEqual(list, type(_graph))
                for i in _graph:
                    self.assertEqual(int, type(i))

    def test_res(self):
        self._judge_res(self._run_module())


if __name__ == "__main__":
    unittest.main()



