import json
import time
import random
import os
from multiprocessing import Queue, Pool
# from tensorflow import set_random_seed

import copy
from multiprocessing import Pool
import numpy as np
from base import NetworkItem
from enumerater import Enumerater
from utils import Communication, list_swap
from evaluator import Evaluator
from sampler import Sampler

from info_str import NAS_CONFIG
import info_str as ifs
from utils import NAS_LOG

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def _subproc_eva(params, eva, gpuq):
    ngpu = gpuq.get()
    start_time = time.time()

    # return score and pos
    if NAS_CONFIG['eva_debug']:
        score = random.uniform(0, 0.1)
    else:
        item, nn_pb, rd, nn_id, pl_len, spl_id, bt_nm, blk_wnr, ft_sign = params
        os.environ['CUDA_VISIBLE_DEVICES'] = str(ngpu)
        score = eva.evaluate(item, nn_pb, is_bestNN=blk_wnr,
                                 update_pre_weight=ft_sign)
    gpuq.put(ngpu)
    time_cost = time.time() - start_time

    NAS_LOG << ('eva_result', nn_id, score, time_cost)
    return score, time_cost, nn_id, spl_id


def _do_task(pool, cmnct, eva):
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
        cmnct.result.put(result)


def _arrange_result(cmnct, net_pl, block_winner=False):
    while not cmnct.result.empty():
        r_ = cmnct.result.get()
        score, time_cost, nn_id, spl_id = r_.get()
        print(ifs.eva_result_tem.format(nn_id, spl_id, score, time_cost))
        # mark down the score
        net_pl[nn_id - 1].item_list[-spl_id].score = score
    # TODO remove other model
    if block_winner:
        pass


def _datasize_ctrl(eva=None, in_game=False):
    """
    Increase the dataset's size in different way
    """
    if in_game:
        eva.add_data(NAS_CONFIG['add_data_every_round'])
    else:
        eva.add_data(NAS_CONFIG['add_data_for_winner'])


def _gpu_batch_update_model(nn, batch_num=NAS_CONFIG['spl_round_net']):
    """

    :param nn:
    :param batch_num:equals to num of gpu
    :return:
    """
    for spl_id in range(1, batch_num + 1):
        nn.spl.update_opt_model(nn.item_list[-spl_id].code, nn.score_list[-spl_id].score)


def _gpu_batch_spl(nn, batch_num=NAS_CONFIG['spl_round_net']):
    """

    :param nn:
    :param batch_num:
    :return:
    """
    for spl_id in range(1, batch_num + 1):
        cell, graph, table = nn.spl.sample()
        nn.item_list.append(NetworkItem(spl_id, graph, cell, table))


def _gpu_batch_task_inqueue(para):
    """

    :param para:
    :return:
    """
    nn, com, round, nn_id, pool_len, batch_num, block_winner, finetune_sign = para
    for spl_id in range(1, batch_num + 1):
        item = nn.item_list[-spl_id]
        task_param = [
            item, nn.pre_block, round, nn_id,
            pool_len, spl_id, batch_num, block_winner, finetune_sign
        ]
        com.task.put(task_param)


def _assign_task(net_pool, com, round, batch_num=NAS_CONFIG['spl_round_net'], block_winner=False):
    pool_len = len(net_pool)
    finetune_sign = True if NAS_CONFIG['pattern'] == "Global" else \
        (pool_len < NAS_CONFIG['finetune_threshold'])
    for nn, nn_id in zip(net_pool, range(1, pool_len+1)):
        if round > 1:
            _gpu_batch_update_model(nn, batch_num)
            _gpu_batch_spl(nn, batch_num)
        para = nn, com, round, nn_id, pool_len, \
            batch_num, block_winner, finetune_sign
        _gpu_batch_task_inqueue(para)


def _game(eva, net_pool, com, round, process_pool):
    _assign_task(net_pool, com, round)
    _datasize_ctrl(eva, in_game=True)
    _do_task(process_pool, com, eva)
    _arrange_result(com, net_pool)


def _eliminate(net_pool=None, round=0):
    """
    Eliminates the worst 50% networks in net_pool depending on scores.
    """
    if NAS_CONFIG['eliminate_policy'] == "best_score_have_ever_met":
        policy = max
    elif NAS_CONFIG['eliminate_policy'] == "average_score_this_round":
        policy = np.mean
    scores = [policy([x.score for x in net_pool[nn_id].item_list[-NAS_CONFIG['spl_round_net']:]])
              for nn_id in range(len(net_pool))]
    scores_cpy = scores.copy()
    scores_cpy.sort()
    original_num = len(scores)
    mid_index = original_num // 2
    mid_val = scores_cpy[mid_index]
    # record the number of the removed network in the pool
    original_index = [i for i in range(len(scores))]

    i = 0
    while i < len(net_pool):
        if scores[i] < mid_val:
            list_swap(net_pool, i, len(net_pool) - 1)
            list_swap(scores, i, len(scores) - 1)
            list_swap(original_index, i, len(original_index) - 1)
            scores.pop()
        else:
            i += 1
    print(ifs.eliinfo_tem.format(original_num - len(scores), len(scores)))


def _train_winner(net_pl, com, pro_pl, round):
    """

    Args:
        net_pool: list of NetworkUnit, and its length equals to 1
        round: the round number of game
    Returns:
        best_nn: object of Class NetworkUnit
    """
    print(ifs.config_ops_ing)
    start_train_winner = time.time()
    eva_winner = Evaluator()
    _datasize_ctrl(eva_winner)

    for i in range(NAS_CONFIG['opt_best_k'] // NAS_CONFIG['num_gpu'] + 1):
        if (i + 1) * NAS_CONFIG['num_gpu'] > NAS_CONFIG['opt_best_k']:
            task_num = NAS_CONFIG['opt_best_k'] - NAS_CONFIG['num_gpu'] * i
        else:
            task_num = NAS_CONFIG['num_gpu']
        if task_num:
            if NAS_CONFIG['pattern'] == "Global":
                round = i + 1
                block_winner = False
            elif NAS_CONFIG['pattern'] == "Block":
                block_winner = True
            _assign_task(net_pl, com, round, task_num, block_winner=block_winner)
            _do_task(pro_pl, com, eva_winner)
            _arrange_result(com, net_pl, block_winner=block_winner)
    best_nn = net_pl[0]
    scores = [x.score for x in best_nn.item_list[-NAS_CONFIG['opt_best_k']:]]
    best_index = scores.index(max(scores)) - len(scores)
    print(ifs.train_winner_tem.format(time.time() - start_train_winner))
    return best_nn, best_index

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

    # for debug
    if NAS_CONFIG['ops_debug']:
        try:
            return _get_ops_copy()
        except:
            print('Nas: _get_ops_copy failed')

    for nn in net_pool:
        nn.first_sample()

    # for debug
    if NAS_CONFIG['ops_debug']:
        _save_ops_copy(net_pool)


def _init_npool_sampler(netpool, block_num):
    for nw in netpool:
        nw.spl = Sampler(nw.graph_part, block_num)
    return


def algo(block_num, eva, com, npool_tem, process_pool):
    """evaluate all the networks asynchronously inside one round and synchronously between rounds

    :param block_num:
    :param eva:
    :param com:
    :param npool_tem:
    :param process_pool:
    :return:
    """
    net_pool = copy.deepcopy(npool_tem)
    print(ifs.start_game_tem.format(len(net_pool)))
    _init_npool_sampler(net_pool, block_num)

    print(ifs.config_ing)
    _init_ops(net_pool)
    round = 0
    start_game = time.time()
    while len(net_pool) > 1:
        start_round = time.time()
        round += 1
        _game(eva, net_pool, com, round, process_pool)
        _eliminate(net_pool, round)
        print(ifs.round_over.format(time.time() - start_round))
    print(ifs.get_winner.format(time.time() - start_game))
    best_nn, best_index = _train_winner(net_pool, com, process_pool, round+1)

    return best_nn, best_index


class Nas:
    def __init__(self, pool):
        print(ifs.init_ing)
        self.enu = Enumerater()
        self.eva = Evaluator()
        self.com = Communication()
        self.pool = pool

    def run(self):
        NAS_LOG << 'enuming'
        network_pool_tem = self.enu.enumerate()
        start_search = time.time()
        for i in range(NAS_CONFIG["block_num"]):
            NAS_LOG << ('search_blk', (i+1) / NAS_CONFIG["block_num"])
            start_block = time.time()
            block, best_index = algo(i, self.eva, self.com, network_pool_tem, self.pool)
            block.pre_block.append([
                block.graph_template,
                block.item_list[best_index]
            ])
            NAS_LOG << ('search_blk_end', time.time() - start_block)
        NAS_LOG << ('nas_end', time.time() - start_search)
        return block.pre_block


if __name__ == '__main__':
    pool = Pool(processes=NAS_CONFIG["num_gpu"])
    nas = Nas(pool)
    nas.run()
