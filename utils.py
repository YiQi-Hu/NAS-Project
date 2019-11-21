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
        self.idle_gpuq = multiprocessing.Manager().Queue()
        for gpu in range(NAS_CONFIG['nas_main']['num_gpu']):
            self.idle_gpuq.put(gpu)

class Logger(object):
    def __init__(self):
        self._eva_log = open(ifs.evalog_path, 'a')
        self._sub_proc_log = open(ifs.subproc_log_path, 'a')
        self._network_log = open(ifs.network_info_path, 'a')
        self._nas_log = open(ifs.naslog_path, 'a')

        self._log_map = { # module x func -> log
            'nas': {
                '_subproc_eva': self._sub_proc_log,
                'eliminate': self._network_log,
                'train_winner': self._network_log
            },
            'eva':{
                '_eval': self._eva_log
            }
        }

    def __del__(self):
        self._eva_log.close()
        self._sub_proc_log.close()
        self._network_log.close()
        self._nas_log.close()
        
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

    def _log_output(self, module, func, context):
        try:
            output = self._log_map[module][func]
        except:
            # if can't find func's log, search module default log
            # and print context.
            default_log = '_%s_log' % module
            if default_log in dir(self):
                output = self.__getattribute__(default_log)
            print(context)
        if output:
            output.write(context)
            output.write('\n')
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
            NAS_LOG = Logger() # 'Nas.run' func in nas.py 
            NAS_LOG << 'enuming'
        """
        module, func = Logger._get_where_called()
        act, others = Logger._get_action(args)
        temp = ifs.MF_TEMP[module][func][act]

        self._log_output(module, func, temp % others)


NAS_LOG = Logger()

if __name__ == '__main__':
    NAS_LOG << ('hello', 'I am bread', 'hello world!')
    NAS_LOG << 'enuming'