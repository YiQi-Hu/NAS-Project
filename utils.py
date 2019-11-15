import queue
import sys
import os
import traceback
import multiprocessing

from info_str import NAS_CONFIG
import info_str as ifs


def list_swap(ls, i, j):
    cpy = ls[i]
    ls[i] = ls[j]
    ls[j] = cpy


class Communication:
    def __init__(self):
        self.task = queue.Queue()
        self.result = queue.Queue()
        self.idle_gpuq = multiprocessing.Queue()
        for gpu in range(NAS_CONFIG['num_gpu']):
            self.idle_gpuq.put(gpu)

class Logger(object):
    def __init__(self):
        # File object '_funcname_log'
        # All function in every module has 1 log at most.
        self.__subproc_eva_log = open(ifs.evalog_path, 'w')
        self._winner_log = open(ifs.winner_log_path, 'w')
        self._network_log = open(ifs.network_info_path, 'w')
        self._module_log = open('memory/module_log.txt', 'w')

    def __del__(self):
        self.__subproc_eva_log.close()
        self._winner_log.close()
        self._network_log.close()
        self._module_log.close()

    @staticmethod
    def _get_where_called():
        last_stack = traceback.extract_stack()[-3]
        # absolute path -> last file name
        where_file = os.path.split(last_stack[0])[-1]
        where_func = last_stack[2]
        # get rid of '.py'
        where_module = os.path.splitext(where_file)[0]

        return where_module, where_func

    @staticmethod
    def _get_action(args):
        if isinstance(args, str) and len(args):
            return args, ()
        elif isinstance(args, tuple) and len(args):
            return args[0], args[1:]
        else:
            raise Exception("empty or wrong log args")
        return

    def _log_output(self, func, context):
        func = func.strip('<>_ \'\"')
        try:
            log_name = '_%s_log' % func
            output = self.__getattribute__(log_name)
            output.write(context)
            output.write('\n')
        except:
            print(context)
        return

    def __lshift__(self, args):
        """
        Wrtie log or print system information.
        The specified log templeate is defined in info_str.py
        Args:
            args (string or tuple, non-empty)
                When it's tuple, its value is string.
                The first value must be action.
        Return: 
            None
        Example:
            NAS_LOG = LOGGER() # 'Nas.run' func in nas.py 
            NAS_LOG << 'enuming'
        """
        module, func = Logger._get_where_called()
        act, others = Logger._get_action(args)
        temp = ifs.MF_TEMP[module][func][act]

        self._log_output(func, temp % others)

NAS_LOG = Logger()

if __name__ == '__main__':
    NAS_LOG << ('hello', 'I am bread', 'hello world!')
    NAS_LOG << 'enuming'