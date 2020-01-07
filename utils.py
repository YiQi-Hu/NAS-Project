import queue
import sys
import os
import traceback
import multiprocessing
from base import Network, NetworkItem, Cell
from info_str import NAS_CONFIG
import info_str as ifs


def list_swap(ls, i, j):
    cpy = ls[i]
    ls[i] = ls[j]
    ls[j] = cpy


class DataSize:
    def __init__(self, eva):
        self.eva = eva
        self.round_count = 0
        self.mode = NAS_CONFIG['nas_main']['add_data_mode']
        #  data size control for game
        self.add_data_per_rd = NAS_CONFIG['nas_main']['add_data_per_round']
        self.init_lr = NAS_CONFIG['nas_main']['init_data_size']
        self.scale = NAS_CONFIG['nas_main']['data_increase_scale']
        #  data size control for confirm train
        self.data_for_confirm_train = NAS_CONFIG['nas_main']['add_data_for_confirm_train']

    def _game_data_ctrl(self):
        if self.mode == "linear":
            self.round_count += 1
            self.eva.set_data_size(self.round_count * self.add_data_per_rd)
        elif self.mode == "scale":
            dsize = int(self.init_lr * (self.scale ** self.round_count))
            self.eva.set_data_size(dsize)
            self.round_count += 1
        else:
            raise ValueError("signal error: mode, it must be one of linear, scale")

    def control(self, stage="game"):
        """Increase the dataset's size in different way

        :param stage: must be one of "game", "confirm"
        :return:
        """
        if stage == "game":
            self._game_data_ctrl()
        elif stage == "confirm":
            self.eva.set_data_size(self.data_for_confirm_train)
        else:
            raise ValueError("signal error: stage, it must be one of game, confirm")


def _epoch_ctrl(eva=None, stage="game"):
    """

    :param eva:
    :param stage: must be one of "game", "confirm", "retrain"
    :return:
    """
    if stage == "game":
        eva.set_epoch(NAS_CONFIG['eva']['search_epoch'])
    elif stage == "confirm":
        eva.set_epoch(NAS_CONFIG['eva']['confirm_epoch'])
    elif stage == "retrain":
        eva.set_epoch(NAS_CONFIG['eva']['retrain_epoch'])
    else:
        raise ValueError("signal error: stage, it must be one of game, confirm, retrain")


class Communication:
    def __init__(self):
        self.task = queue.Queue()
        self.result = queue.Queue()
        self.idle_gpuq = multiprocessing.Manager().Queue()
        self.net_pool = ""
        self.tables = []
        self.round = 0
        self.tw_count = NAS_CONFIG['nas_main']['num_opt_best'] - NAS_CONFIG['nas_main']['num_gpu']
        for gpu in range(NAS_CONFIG['nas_main']['num_gpu']):
            self.idle_gpuq.put(gpu)

    def wake_up_train_winner(self, res):
        print("train_winner wake up")
        score, time_cost, nn_id, spl_id = res
        print("nn_id spl_id item_list_length", nn_id, spl_id, len(self.net_pool[nn_id - 1].item_list))
        self.net_pool[nn_id - 1].item_list[spl_id - 1].score = score
        self.net_pool[nn_id - 1].spl.update_opt_model(self.net_pool[nn_id - 1].item_list[spl_id - 1].code,
                                                      -self.net_pool[nn_id - 1].item_list[spl_id - 1].score)
        item_id = len(self.net_pool[nn_id - 1].item_list) + 1
        cnt = 0
        while cnt < 500:
            cell, graph, table = self.net_pool[nn_id - 1].spl.sample()
            if table not in self.tables:
                self.tables.append(table)
                print("sample success", cnt)
                break
            cnt += 1
        if self.tw_count > 0:
            self.net_pool[nn_id - 1].item_list.append(NetworkItem(item_id, graph, cell, table))
            item = self.net_pool[nn_id - 1].item_list[-1]
            task_param = [
                item, self.net_pool[nn_id - 1].pre_block, self.round, nn_id, 1, item_id,
                NAS_CONFIG['nas_main']['spl_network_round'] * (self.round - 1) + NAS_CONFIG['nas_main']['num_opt_best'],
                True, True
            ]
            print("train winner new task put")
            self.task.put(task_param)
        self.tw_count -= 1


class Logger(object):
    def __init__(self):
        self._eva_log = open(ifs.evalog_path, 'a')
        self._sub_proc_log = open(ifs.subproc_log_path, 'a')
        self._network_log = open(ifs.network_info_path, 'a')
        self._nas_log = open(ifs.naslog_path, 'a')

        self._log_map = {  # module x func -> log
            'nas': {
                '_subproc_eva': self._sub_proc_log,
                '_save_net_info': self._network_log,
                'default': self._nas_log
            },
            'evaluator': {
                'default': self._eva_log
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
        output = None
        try:
            if func not in self._log_map[module].keys():
                func = ''
            output = self._log_map[module][func]
        except:
            # if can't find func's log, search module default log
            # and print context.
            default_log = '_%s_log' % module
            if hasattr(self, default_log):
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
        # print(module, func, temp, others)
        # print(module, func, temp.format(others))
        if func != "_save_net_info":
            print(temp.format(others))
        self._log_output(module, func, temp.format(others))


NAS_LOG = Logger()

if __name__ == '__main__':
    NAS_LOG << ('hello', 'I am bread', 'hello world!')
    NAS_LOG << 'enuming'