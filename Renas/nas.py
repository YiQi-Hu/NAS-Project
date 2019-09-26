import json
import time
import random
import os
from multiprocessing import Queue, Pool
from tensorflow import set_random_seed

from enumerater import Enumerater
from evaluator import Evaluator
from communicator import Communicator
import corenas

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# File path
__NAS_CONFIG_PATH = 'nas_config.json'
__EVALOG_PATH_TEM = os.path.join('memory', 'evaluating_log_with_gpu{}.txt')
__NETWORK_INFO_PATH = os.path.join('memory', 'network_info.txt')
__WINNER_LOG_PATH = os.path.join('memory', 'train_winner_log.txt')

# Log content
__LOG_EVAINFO_TEM = 'block_num:{} round:{} network_index:{}/{}\n'
__LOG_EVAFAIL = 'evaluating failed and we will try again...\n'
__LOG_WINNER_TEM = 'block_num:{} sample_count:{}/{}\n'

# System information
__SYS_EVAFAIL = 'NAS: ' + __LOG_EVAFAIL
__SYS_EVA_RESULT_TEM = 'NAS: network_index:{} score:{} time_cost:{} '
__SYS_ELIINFO_TEM = 'NAS: eliminating {}, remaining {}...'
__SYS_INIT_ING = 'NAS: Initializing...'
__SYS_I_AM_PS = 'NAS: I am ps.'
__SYS_I_AM_WORKER = 'NAS: I am worker.'
__SYS_ENUM_ING = 'NAS: Enumerating all possible networks!'
__SYS_SEARCH_BLOCK_TEM = 'NAS: Searching for block {}/{}...'
__SYS_WORKER_DONE = 'NAS: all of the blocks have been evaluated, please go to the ps manager to view the result...'
__SYS_WAIT_FOR_TASK = 'NAS: waiting for assignment of next round...'
__SYS_CONFIG_ING = 'NAS: Configuring the networks in the first round...'
__SYS_GET_WINNER = 'NAS: We got a WINNER!'
__SYS_BEST_AND_SCORE_TEM = 'NAS: We have got the best network and its score is {}'
__SYS_START_GAME_TEM = 'NAS: Now we have {} networks. Start game!'
__SYS_CONFIG_OPS_ING = "NAS: Configuring ops and skipping for the best structure and training them..."

# Global variables
NAS_CONFIG = json.load(open(__NAS_CONFIG_PATH, encoding='utf-8'))
IDLE_GPUQ = Queue()

def __gpu_eva(params):
    graph, cell, nn_pb, _, p_, ft_sign, pl_, eva, ngpu = params
    # params = (graph, cell, nn_preblock, round, pos, 
    # finetune_signal, pool_len, eva, ngpu)
    start_time = time.time()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(params.ngpu)
    with open(__EVALOG_PATH_TEM.format(ngpu)) as f:
        f.write(__LOG_EVAINFO_TEM.format(
            len(nn_pb)+1, round, p_, pl_
        ))
        while True:
            try:
                score = eva.evaluate(graph, cell, nn_pb, False, ft_sign, f)
                break
            except:
                print(__SYS_EVAFAIL)
                f.write(__LOG_EVAFAIL)
        IDLE_GPUQ.put(ngpu)

    end_time = time.time()
    start_time = time.time()
    time_cost = end_time - start_time
    return score, time_cost, p_


def __module_init():
    enu = Enumerater(
        depth=NAS_CONFIG.__depth,
        width=NAS_CONFIG.__width,
        max_branch_depth=NAS_CONFIG.__max_bdepth)
    eva = Evaluator()

    return enu, eva

def __filln_queue(q, n):
    for i in range(n):
        q.put(i)

def __wait_for_event(event_func):
    # TODO replaced by multiprocessing.Event
    while event_func():
        print(__SYS_WAIT_FOR_TASK)
        time.sleep(20)
    return

class Nas():
    def __init__(self, job_name, ps_host):
        self.__is_ps = (job_name == 'ps')
        self.__ps_host = ps_host

        set_random_seed(NAS_CONFIG.randseed)
        return 

    def __ps_run(self, enum, eva, cmnct):
        __filln_queue(IDLE_GPUQ, NAS_CONFIG.num_gpu)
        print(__SYS_ENUM_ING)
        network_pool_tem = enum.enumerate()

        for i in range(NAS_CONFIG.block_num):
            print(__SYS_SEARCH_BLOCK_TEM.format(i+1, NAS_CONFIG.block_num))
            block, best_index = corenas.Corenas(i, eva, cmnct, network_pool_tem)
            block.pre_block.append([
                block.graph_part, 
                block.graph_full_list[best_index], 
                block.cell_list[best_index]
            ])
        cmnct.end_flag.put(1) # TODO find other way to stop workers

        return block.pre_block

    # TODO ps -> worker
    # @staticmethod 
    def __worker_run(self, eva, cmnct):
        __filln_queue(IDLE_GPUQ, NAS_CONFIG.num_gpu)
        pool = Pool(processes=NAS_CONFIG.num_gpu)
        while cmnct.end_flag.empty():
            __wait_for_event(cmnct.data_sync.empty)
            
            cmnct.data_count += 1
            data_count_ps = cmnct.data_sync.get(timeout=1)
            eva.add_data(1600*(data_count_ps-cmnct.data_count+1))
            result_list = []

            while not cmnct.task.empty():
                gpu = IDLE_GPUQ.get()
                try:
                    task_params = cmnct.task.get(timeout=1)
                except:
                    IDLE_GPUQ.put(gpu)
                    break
                result = pool.apply_async(__gpu_eva, args=task_params)
                result_list.append(result)
            for r_ in result_list:
                score, time_cost, network_index = r_.get()
                print(__SYS_EVA_RESULT_TEM.format(network_index, score, time_cost))
                cmnct.result.put([score, network_index, time_cost])

            __wait_for_event(cmnct.task.empty)
        pool.close()
        pool.join()
        return

    def run(self):
        print(__SYS_INIT_ING)
        enum, eva = __module_init()
        cmnct = Communicator(self.__is_ps, self.__ps_host)
        if self.__is_ps:
            print(__SYS_I_AM_PS)
            return self.__ps_run(enum, eva, cmnct)
        else:
            print(__SYS_I_AM_WORKER)
            self.__worker_run(eva, cmnct)
            print(__SYS_WORKER_DONE)

        return