import unittest
from info_str import NAS_CONFIG
import random
from enumerater import Enumerater
from base import Network
from ddt import data, ddt


@ddt
class Test_enum(unittest.TestCase):
    global test_info
    test_info = []
    for i in range(100):
        _depth = random.randint(0, 25)
        _width = random.randint(0, 1)
        _max_depth = random.randint(0, _depth)
        test_info.append((_depth, _width, _max_depth))

    with open('./test/test_enum.txt', 'w') as op:
        for i in test_info:
            op.writelines('depth:' + str(i[0]) + '\n')
            op.writelines('width:' + str(i[1]) + '\n')
            op.writelines('max_depth:' + str(i[2]) + '\n')
            op.writelines('###'*20 + '\n')


    def _run_module(self):
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

    @data(*test_info)
    def test_res(self, para):
        NAS_CONFIG['enum']['depth'] = para[0]
        NAS_CONFIG['enum']['width'] = para[1]
        NAS_CONFIG['enum']['max_depth'] = para[2]
        print(para[0], para[1], para[2])
        self._judge_res(self._run_module())


if __name__ == "__main__":
    unittest.main()




