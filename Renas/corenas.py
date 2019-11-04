import copy
from multiprocessing import Pool

from numpy import zeros

from nas import Nas
from nas import _gpu_eva
from nas import _wait_for_event
from nas import _do_task
from nas import _arrange_result
from nas import _filln_queue

from info_str import NAS_CONFIG
from info_str import (
    NETWORK_INFO_PATH,
    WINNER_LOG_PATH,
    LOG_EVAINFO_TEM,
    LOG_WINNER_TEM,
    SYS_CONFIG_ING,
    SYS_GET_WINNER,
    SYS_ELIINFO_TEM,
    SYS_BEST_AND_SCORE_TEM,
    SYS_START_GAME_TEM,
    SYS_CONFIG_OPS_ING
)

CORE_CONFIG = NAS_CONFIG['core']

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
    finetune_sign = (pool_len < CORE_CONFIG['finetune_threshold'])
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
    _filln_queue(com.idle_gpuq, NAS_CONFIG["num_gpu"])
    # Do the tasks
    with Pool(processes=NAS_CONFIG['num_gpu']) as pool:
        result_list = _do_task(pool, com, eva)
        _arrange_result(result_list, com)
    # TODO replaced by multiprocessing.Event
    _wait_for_event(lambda: com.result.qsize() != pool_len)
    # fill the score list
    _arrange_score(net_pool, scores, com)
    return scores

# TODO understand this code
def _train_winner(net_pool, round, eva):
    best_nn = net_pool[0]
    best_opt_score = 0
    best_cell_i = 0
    eva.add_data(100)  # -1 represent that we add all data for training
    print(SYS_CONFIG_OPS_ING)
    for i in range(CORE_CONFIG['opt_best_k']):
        best_nn.table = best_nn.opt.sample()
        best_nn.spl.renewp(best_nn.table)
        cell, graph = best_nn.spl.sample()
        best_nn.graph_full_list.append(graph)
        best_nn.cell_list.append(cell)
        with open(WINNER_LOG_PATH, 'a') as f:
            f.write(LOG_WINNER_TEM.format(len(best_nn.pre_block) + 1, i, CORE_CONFIG['opt_best_k']))
            with Pool(1) as p:
                params = (graph, cell, best_nn.pre_block, 0, 0, True, 1)
                # params = (graph, cell, nn_preblock, pos,
                # round, finetune_signal, pool_len)
                opt_score, _ = p.apply(_gpu_eva, args=(params, eva, 0, best_cell_i))
        best_nn.score_list.append(opt_score)
        if opt_score > best_opt_score:
            best_opt_score = opt_score
            best_cell_i = i
    print(SYS_BEST_AND_SCORE_TEM.format(best_opt_score))
    best_index = best_cell_i - CORE_CONFIG["opt_best_k"]

    _save_info(NETWORK_INFO_PATH, best_nn, round, 0, 1)
    return best_nn, best_index

# Debug function
import pickle
_OPS_PNAME = 'pcache\\ops_%d-%d-%d.pickle' % (
    NAS_CONFIG['enum']["depth"],
    NAS_CONFIG['enum']["width"],
    NAS_CONFIG['enum']["max_depth"]
)

def _get_ops_copy():
    with open(_OPS_PNAME, 'rb') as f:
        pool = pickle.load(f)
    return pool

def _save_ops_copy(pool):
    with open(_OPS_PNAME, 'wb') as f:
        pickle.dump(pool, f)
    return

# TODO understand this code
def _init_ops(net_pool):
    scores = zeros(len(net_pool))
    scores = scores.tolist()

    # for debug
    if CORE_CONFIG['ops_debug']:
        try:
            return scores, _get_ops_copy()
        except:
            print('Nas: _get_ops_copy failed')

    # copied from initialize_ops_subprocess(self, NETWORK_POOL):
    _cnt = 0
    for nn in net_pool:  # initialize the full network by adding the skipping and ops to graph_part
        _cnt += 1
        print("\r_init_ops Completed: %f %%" % (_cnt / len(net_pool) * 100), end='')  # for debug
        nn.table = nn.spl.opt.sample()
        nn.spl.renewp(nn.table)
        cell, graph = nn.spl.sample()
        # blocks = []
        # for block in nn.pre_block:  # get the graph_full adjacency list in the previous blocks
        #     blocks.append(block[0])  # only get the graph_full in the pre_bock
        # pred_ops = Predictor().predictor(blocks, graph)
        #
        # table = nn.spl.init_p(pred_ops)  # spl refer to the pred_ops
        # nn.spl.renewp(table)
        # cell, graph = nn.spl.sample()  # sample again after renew the table
        nn.graph_full_list.append(graph)  # graph from first sample and second sample are the same, so that we don't have to assign network.graph_full at first time
        nn.cell_list.append(cell)

    # for debug
    if CORE_CONFIG['ops_debug']:
        _save_ops_copy(net_pool)
    print()

    return scores, net_pool

def Corenas(block_num, eva, com, npool_tem):
    # Different code is ran on different machine depends on whether it's a ps host or a worker host.
    # PS host is used for generating all the networks and collect the final result of evaluation.
    # And PS host together with worker hosts(as long as they all have GPUs) train as well as
    # evaluate all the networks asynchronously inside one round and synchronously between rounds

    # implement the copy when searching for every block
    net_pool = copy.deepcopy(npool_tem)
    for network in net_pool:  # initialize the sample module
        network.init_sample(block_num)

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
    best_nn, best_index = _train_winner(net_pool, round+1, eva)

    return best_nn, best_index
