import json
import time
import random
import os
from multiprocessing import Queue, Pool
# from tensorflow import set_random_seed

import copy
from multiprocessing import Pool
from numpy import zeros
from enumerater import Enumerater
from communicator import Communicator
from evaluator import Evaluator
from sampler import Sampler
from info_str import (
    NAS_CONFIG,
    CUR_VER_DIR,
    EVALOG_PATH_TEM,
    NETWORK_INFO_PATH,
    WINNER_LOG_PATH,
    LOG_EVAINFO_TEM,
    LOG_EVAFAIL,
    LOG_WINNER_TEM,
    SYS_EVAFAIL,
    SYS_EVA_RESULT_TEM,
    SYS_ELIINFO_TEM,
    SYS_INIT_ING,
    SYS_I_AM_PS,
    SYS_I_AM_WORKER,
    SYS_ENUM_ING,
    SYS_SEARCH_BLOCK_TEM,
    SYS_WORKER_DONE,
    SYS_WAIT_FOR_TASK,
    SYS_CONFIG_ING,
    SYS_GET_WINNER,
    SYS_BEST_AND_SCORE_TEM,
    SYS_START_GAME_TEM,
    SYS_CONFIG_OPS_ING
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# # TODO Fatal: Corenas can not get items in IDLE_GPUQ (Queue)
# IDLE_GPUQ = Queue()

def _subproc_eva(params, eva, gpuq):
    print("before i get!!!")
    ngpu = gpuq.get()
    print("i get!!!")
    start_time = time.time()

    try:
        # return score and pos
        if NAS_CONFIG['eva_debug']:
            raise Exception() # return random result
        score, pos = _gpu_eva(params, eva, ngpu)
    except:
        score = random.random()
        # params = (graph, cell, nn_preblock, pos, ...)
        pos = params[3]
    finally:
        gpuq.put(ngpu)

    end_time = time.time()
    time_cost = end_time - start_time

    return score, time_cost, pos


def _gpu_eva(params, eva, ngpu, cur_bt_score=0):
    graph, cell, nn_pb, p_, r_, ft_sign, pl_ = params
    # params = (graph, cell, nn_preblock, pos,
    # round, finetune_signal, pool_len)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(ngpu)
    with open(EVALOG_PATH_TEM.format(ngpu), 'w') as f:
        f.write(LOG_EVAINFO_TEM.format(
            len(nn_pb)+1, r_, p_, pl_
        ))
        # try infinitely ?
        # while True:
        print("%d training..." % ngpu)
        score = eva.evaluate(graph, cell, nn_pb, cur_best_score=cur_bt_score, is_bestNN=False,
                             update_pre_weight=ft_sign, log_file=f)
            # try:
            #     score = eva.evaluate(graph, cell, nn_pb, cur_best_score=cur_bt_score, is_bestNN=False, update_pre_weight=ft_sign, log_file=f)
            #     break
            # except Exception as e:
            #     print(SYS_EVAFAIL, e)
            #     f.write(LOG_EVAFAIL)
    print("eva completed")
    return score, p_


def _module_init():
    enu = Enumerater(
        depth=NAS_CONFIG["depth"],
        width=NAS_CONFIG["width"],
        max_branch_depth=NAS_CONFIG["max_depth"])
    eva = Evaluator()

    return enu, eva

def _filln_queue(q, n):
    for i in range(n):
        q.put(i)

def _wait_for_event(event_func):
    # TODO replaced by multiprocessing.Event
    while event_func():
        print(SYS_WAIT_FOR_TASK)
        time.sleep(20)
    return

def _do_task(pool, cmnct, eva):
    result_list = []

    while not cmnct.task.empty():
        try:
            task_params = cmnct.task.get(timeout=1)
        except:
            break
        if NAS_CONFIG['subp_debug']:
            result = _subproc_eva(task_params, eva, cmnct.idle_gpuq)
        else:
            result = pool.apply_async(
                _subproc_eva,
                (task_params, eva, cmnct.idle_gpuq))
        result_list.append(result)

    return result_list

# TODO too slow
def _arrange_result(result_list, cmnct):
    _cnt = 0
    for r_ in result_list:
        _cnt += 1
        cplt_r = _cnt / len(result_list) * 100
        print("\r_arrange_result Completed: {} %".format(cplt_r), end='')
        score, time_cost, network_index = r_.get()
        # print(SYS_EVA_RESULT_TEM.format(network_index, score, time_cost))
        cmnct.result.put((score, network_index, time_cost))
    print('done!')
    return

# wait to be _save_info
def _save_info(path, network, round, original_index, network_num):
    tmpa = 'number of scheme: {}\n'
    tmpb = 'graph_part: {}\n'
    tmpc = '    graph_full: {}\n    cell_list: {}\n    score: {}\n'
    s = LOG_EVAINFO_TEM.format(len(network.pre_block)+1, round, original_index, network_num)
    s = s + tmpa.format(len(network.score_list))
    s = s + tmpb.format(str(network.graph_part))

    for item in zip(network.graph_full_list, network.cell_list, network.score_list):
        s = s + tmpc.format(str(item[0]), str(item[1]), str(item[2]))

    with open(path, 'a') as f:
        f.write(s)

    return


def _list_swap(ls, i, j):
    cpy = ls[i]
    ls[i] = ls[j]
    ls[j] = cpy


def _eliminate(net_pool=None, scores=[], round=0):
    '''
    Eliminates the worst 50% networks in net_pool depending on scores.
    '''
    scores_cpy = scores.copy()
    scores_cpy.sort()
    original_num = len(scores)
    mid_index = original_num // 2
    mid_val = scores_cpy[mid_index]
    original_index = [i for i in range(len(scores))]  # record the

    i = 0
    while i < len(net_pool):
        if scores[i] < mid_val:
            _list_swap(net_pool, i, len(net_pool) - 1)
            _list_swap(scores, i, len(scores) - 1)
            _list_swap(original_index, i, len(original_index) - 1)
            _save_info(NETWORK_INFO_PATH, net_pool.pop(), round, original_index.pop(), original_num)
            scores.pop()
        else:
            i += 1
    print(SYS_ELIINFO_TEM.format(original_num - len(scores), len(scores)))
    return mid_val

def _datasize_ctrl(eva=None):
    '''
    Increase the dataset's size in different way
    '''
    # TODO different datasize control policy (?)
    nxt_size = eva.get_train_size() * 2
    eva.set_train_size(nxt_size)
    return

def _game_assign_task(net_pool, scores, com, round, pool_len, eva):
    finetune_sign = (pool_len < NAS_CONFIG['finetune_threshold'])
    for nn, score, i in zip(net_pool, scores, range(pool_len)):
        if round == 1:
            cell, graph = nn.cell_list[-1], nn.graph_full_list[-1]
        else:
            nn.opt.update_model(nn.table, score)
            nn.table = nn.opt.sample()
            nn.spl.renewp(nn.table)
            cell, graph = nn.spl.sample()
            nn.graph_full_list.append(graph)
            nn.cell_list.append(cell)
        task_param = [
            graph, cell, nn.pre_block, i,
            round, finetune_sign, pool_len
        ]
        com.task.put(task_param)
    # TODO data size control
    com.data_sync.put(com.data_count)  # for multi host
    eva.add_data(1600)
    return


def _arrange_score(net_pool, scores, com):
    while not com.result.empty():
        # score, index, time_cost
        s_, i_, tc_ = com.result.get()
        scores[i_] = s_

    for nn, score in zip(net_pool, scores):
        nn.score_list.append(score)
    return scores


def _game(eva, net_pool, scores, com, round):
    pool_len = len(net_pool)
    print(SYS_START_GAME_TEM.format(pool_len))
    # put all the network in this round into the task queue
    _game_assign_task(net_pool, scores, com, round, pool_len, eva)
    # For corenas can not get IDLE_GPUQ from nas, so
    # we fill it here.
    pool = Pool(processes=NAS_CONFIG["num_gpu"])
    _filln_queue(com.idle_gpuq, NAS_CONFIG["num_gpu"])
    # Do the tasks
    result_list = _do_task(pool, com, eva)
    pool.close()
    pool.join()
    _arrange_result(result_list, com)
    # TODO replaced by multiprocessing.Event
    _wait_for_event(lambda: com.result.qsize() != pool_len)
    # fill the score list
    _arrange_score(net_pool, scores, com)
    return scores


def _train_winner(net_pool, round):
    """

    Args:
        net_pool: list of NetworkUnit, and its length equals to 1
        round: the round number of game
    Returns:
        best_nn: object of Class NetworkUnit
    """
    best_nn = net_pool[0]
    eva = Evaluator()
    eva.add_data(-1)
    print(SYS_CONFIG_OPS_ING)
    for i in range(NAS_CONFIG['opt_best_k']):
        best_nn.sample()
        with open(WINNER_LOG_PATH, 'a') as f:
            f.write(LOG_WINNER_TEM.format(len(best_nn.pre_block) + 1, i, NAS_CONFIG['opt_best_k']))
            p = Pool(1)
            params = (best_nn.graph_full[-1], best_nn.cell_list[-1], best_nn.pre_block, 0, 0, True, 1)
            # params = (graph, cell, nn_preblock, pos, round, finetune_signal, pool_len)
            opt_score, _ = p.apply(_gpu_eva, args=(params, eva, 0, best_nn.best_score))
        best_nn.update_score(opt_score)
    print(SYS_BEST_AND_SCORE_TEM.format(best_nn.best_score))
    _save_info(NETWORK_INFO_PATH, best_nn, round, 0, 1)
    return best_nn

# Debug function
import pickle
_OPS_PNAME = 'pcache\\ops_%d-%d-%d.pickle' % (
    NAS_CONFIG["depth"], NAS_CONFIG["width"], NAS_CONFIG["max_depth"])
def _get_ops_copy():
    with open(_OPS_PNAME, 'rb') as f:
        pool = pickle.load(f)
    return pool

def _save_ops_copy(pool):
    with open(_OPS_PNAME, 'wb') as f:
        pickle.dump(pool, f)
    return

def _init_ops(net_pool):
    """Generates ops and skipping for every Network,

    Args:
        net_pool (list of NetworkUnit)
    Returns:
        net_pool (list of NetworkUnit)
        scores (list of score, and its length equals to that of net_pool)
    """
    scores = zeros(len(net_pool))
    scores = scores.tolist()

    # for debug
    if NAS_CONFIG['ops_debug']:
        try:
            return scores, _get_ops_copy()
        except:
            print('Nas: _get_ops_copy failed')

    for nn in net_pool:
        nn.init_sample()

    # for debug
    if NAS_CONFIG['ops_debug']:
        _save_ops_copy(net_pool)

    return net_pool, scores

def _init_npool_sampler(netpool, block_num):
    for nw in netpool:
        nw.spl = Sampler(nw.graph_part, block_num)
    return

def Corenas(block_num, eva, com, npool_tem):
    # Different code is ran on different machine depends on whether it's a ps host or a worker host.
    # PS host is used for generating all the networks and collect the final result of evaluation.
    # And PS host together with worker hosts(as long as they all have GPUs) train as well as
    # evaluate all the networks asynchronously inside one round and synchronously between rounds

    # implement the copy when searching for every block
    net_pool = copy.deepcopy(npool_tem)
    # initialize sampler
    _init_npool_sampler(net_pool, block_num)

    # Step 2: Search best structure
    print(SYS_CONFIG_ING)
    scores, net_pool = _init_ops(net_pool)
    round = 0
    while (len(net_pool) > 1):
        # Step 3: Sample, train and evaluate every network
        com.data_count += 1
        round += 1
        scores = _game(eva, net_pool, scores, com, round)
        # Step 4: Eliminate half structures and increase dataset size
        _eliminate(net_pool, scores, round)
        # _datasize_ctrl('same', epic)
    print(SYS_GET_WINNER)
    # Step 5: Global optimize the best network
    best_nn, best_index = _train_winner(net_pool, round+1)

    return best_nn, best_index




class Nas():
    def __init__(self, job_name, ps_host=''):
        self.__is_ps = (job_name == 'ps')
        self.__ps_host = ps_host

        return

    def _ps_run(self, enum, eva, cmnct):
        print(SYS_ENUM_ING)

        network_pool_tem = enum.enumerate()
        for i in range(NAS_CONFIG["block_num"]):
            print(SYS_SEARCH_BLOCK_TEM.format(i+1, NAS_CONFIG["block_num"]))
            block, best_index = Corenas(i, eva, cmnct, network_pool_tem)
            block.pre_block.append([
                block.graph_part,
                block.graph_full_list[best_index],
                block.cell_list[best_index]
            ])
        cmnct.end_flag.put(1) # TODO find other way to stop workers

        return block.pre_block

    def _worker_run(eva, cmnct):
        _filln_queue(cmnct.idle_gpuq, NAS_CONFIG["num_gpu"])
        pool = Pool(processes=NAS_CONFIG["num_gpu"])
        while cmnct.end_flag.empty():
            _wait_for_event(cmnct.data_sync.empty)

            cmnct.data_count += 1
            data_count_ps = cmnct.data_sync.get(timeout=1)
            eva.add_data(1600*(data_count_ps-cmnct.data_count+1))

            result_list = _do_task(pool, cmnct, eva)
            _arrange_result(result_list, cmnct)

            _wait_for_event(cmnct.task.empty)

        pool.close()
        pool.join()
        return

    def run(self):
        print(SYS_INIT_ING)
        cmnct = Communicator(self.__is_ps, self.__ps_host)
        enum, eva = _module_init()

        if self.__is_ps:
            print(SYS_I_AM_PS)
            return self._ps_run(enum, eva, cmnct)
        else:
            print(SYS_I_AM_WORKER)
            self._worker_run(eva, cmnct)
            print(SYS_WORKER_DONE)

        return

if __name__ == '__main__':
    nas = Nas('ps', '127.0.0.1:5000')
    nas.run()
