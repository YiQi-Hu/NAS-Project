import json
import time
import random
import os
from multiprocessing import Queue, Pool
# from tensorflow import set_random_seed
import re
import copy
from multiprocessing import Pool
import numpy as np
from base import Network, NetworkItem
from enumerater import Enumerater
from utils import Communication, list_swap
from evaluator import Evaluator
from sampler import Sampler

from info_str import NAS_CONFIG
import info_str as ifs
from utils import NAS_LOG


MAIN_CONFIG = NAS_CONFIG['nas_main']
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def _subproc_eva(params, eva, gpuq):
    ngpu = gpuq.get()
    start_time = time.time()

    item, rd, nn_id, pl_len, spl_id, bt_nm, blk_wnr, ft_sign = params
    # return score and pos
    if MAIN_CONFIG['eva_debug']:
        score = random.uniform(0, 0.1)
    else:
        item.graph.append([])
        os.environ['CUDA_VISIBLE_DEVICES'] = str(ngpu)
        score = eva.evaluate(item, is_bestNN=blk_wnr,
                                 update_pre_weight=ft_sign)
    gpuq.put(ngpu)
    time_cost = time.time() - start_time

    NAS_LOG << ('eva_ing', len(Network.pre_block)+1, rd, nn_id,
                pl_len, spl_id, bt_nm, score, time_cost, os.getpid())
    return score, time_cost, nn_id, spl_id


def _save_net_info(net, *args):
    net_info_temp = "elim_net_info"
    sche_info_temp = "scheme_info"
    blk_num, rd, net_lft, net_id, sche_num = args
    NAS_LOG << (net_info_temp, blk_num, rd, net_lft, net_id, sche_num, net.graph_template)
    for scheme in net.item_list:
        NAS_LOG << (sche_info_temp, str(scheme.graph), str(scheme.cell_list), str(scheme.code),
                    str(scheme.score))


def _do_task(pool, cmnct, eva):
    while not cmnct.task.empty():
        try:
            task_params = cmnct.task.get(timeout=1)
        except:
            break
        if MAIN_CONFIG['subp_debug']:
            result = _subproc_eva(task_params, eva, cmnct.idle_gpuq)
        else:
            result = pool.apply_async(
                _subproc_eva,
                (task_params, eva, cmnct.idle_gpuq))
        cmnct.result.put(result)


def _arrange_result(cmnct, net_pl):
    while not cmnct.result.empty():
        r_ = cmnct.result.get()
        if MAIN_CONFIG['subp_debug']:
            score, time_cost, nn_id, spl_id = r_
        else:
            score, time_cost, nn_id, spl_id = r_.get()
        NAS_LOG << ('eva_result', nn_id, spl_id, score, time_cost)
        # mark down the score
        net_pl[nn_id - 1].item_list[-spl_id].score = score


def _datasize_ctrl(eva=None, in_game=False):
    """
    Increase the dataset's size in different way
    """
    if in_game:
        eva.add_data(MAIN_CONFIG['add_data_per_round'])
    else:
        eva.add_data(MAIN_CONFIG['add_data_for_winner'])


def _init_ops_dup_chk(network, pred, task_num=MAIN_CONFIG['spl_network_round']):
    """init ops with duplicate check

    :return:
    """
    tables = []
    cells = []
    graphs = []
    spl_index = 0
    cnt = 0
    while spl_index < task_num:
        cnt += 1
        if cnt > 500:
            NAS_LOG << ('no_dim_ini', spl_index)
            raise ValueError("sample error")
        cell, graph, table = network.spl.sample()
        graph, cell, table = _pred_ops(network, pred, graph, table)
        if table not in tables:
            tables.append(table)
            cells.append(cell)
            graphs.append(graph)
            spl_index += 1
    return cells, graphs, tables


def _spl_dup_chk(network, task_num=MAIN_CONFIG['spl_network_round']):
    """sample with duplicate check

    :param network:
    :param task_num:
    :return:
    """
    tables = []
    cells = []
    graphs = []
    spl_index = 0
    cnt = 0
    while spl_index < task_num:
        cnt += 1
        if cnt > 500:
            NAS_LOG << ('no_dim_spl', spl_index)
            raise ValueError("sample error")
        cell, graph, table = network.spl.sample()
        if table not in tables:
            tables.append(table)
            cells.append(cell)
            graphs.append(graph)
            spl_index += 1
    return cells, graphs, tables


def _gpu_batch_update_model(nn, batch_num=MAIN_CONFIG['spl_network_round']):
    """

    :param nn:
    :param batch_num:equals to num of gpu
    :return:
    """
    for spl_id in range(1, batch_num + 1):
        nn.spl.update_opt_model(nn.item_list[-spl_id].code, nn.item_list[-spl_id].score)


def _gpu_batch_init(nn, pred, batch_num=MAIN_CONFIG['spl_network_round']):
    """

    :param nn:
    :param batch_num:
    :return:
    """
    cells, graphs, tables = _init_ops_dup_chk(nn, pred, batch_num)
    # cells, graphs, tables = _spl_dup_chk(nn, batch_num)
    for cell, graph, table, spl_id in zip(cells, graphs, tables, range(1, batch_num + 1)):
        nn.item_list.append(NetworkItem(spl_id, graph, cell, table))


def _gpu_batch_spl(nn, batch_num=MAIN_CONFIG['spl_network_round']):
    """

    :param nn:
    :param batch_num:
    :return:
    """
    cells, graphs, tables = _spl_dup_chk(nn, batch_num)
    for cell, graph, table, spl_id in zip(cells, graphs, tables, range(1, batch_num + 1)):
        nn.item_list.append(NetworkItem(spl_id, graph, cell, table))


def _pred_ops(nn, pred, graph, table):
    """

    :param nn:
    :param pred:
    :param spl_id:
    :return:
    """
    pre_block = Network.pre_block.copy()
    pre_block = [elem[-1].graph for elem in pre_block]
    for block in pre_block:
        if block[-1]:
            block.append([])
    graph.append([])  # add the virtual node
    pred_ops = pred.predictor(pre_block, graph)
    pred_ops = pred_ops[:-1]  # remove the ops of virtual node
    table = nn.spl.ops2table(pred_ops, table)
    cell, graph = nn.spl.convert(table)
    return graph, cell, table


def _gpu_batch_task_inqueue(para):
    """

    :param para:
    :return:
    """
    nn, com, round, nn_id, pool_len, batch_num, block_winner, finetune_sign = para
    for spl_id in range(1, batch_num + 1):
        item = nn.item_list[-spl_id]
        task_param = [
            item, round, nn_id, pool_len, spl_id, batch_num, block_winner, finetune_sign
        ]
        com.task.put(task_param)


def _assign_task(net_pool, com, round, batch_num=MAIN_CONFIG['spl_network_round'], block_winner=False):
    pool_len = len(net_pool)
    finetune_sign = True if MAIN_CONFIG['pattern'] == "Global" else \
        (pool_len < MAIN_CONFIG['finetune_threshold'])
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
    if MAIN_CONFIG['eliminate_policy'] == "best":
        policy = max
    elif MAIN_CONFIG['eliminate_policy'] == "over_average":
        policy = np.mean
    scores = [policy([x.score for x in net_pool[nn_id].item_list[-MAIN_CONFIG['spl_network_round']:]])
              for nn_id in range(len(net_pool))]
    scores_cpy = scores.copy()
    scores_cpy.sort()
    original_num = len(scores)
    mid_index = original_num // 2
    mid_val = scores_cpy[mid_index]

    i = 0
    while i < len(net_pool):
        if scores[i] < mid_val:
            list_swap(net_pool, i, len(net_pool) - 1)
            list_swap(scores, i, len(scores) - 1)
            net = net_pool.pop()
            scores.pop()
            NAS_LOG << ("elim_net", len(Network.pre_block)+1, round, len(net_pool),
                        net.id, len(net.item_list))
            # TODO record the info of the network removed
            _save_net_info(net, len(Network.pre_block)+1,
                           round, len(net_pool), net.id, len(net.item_list))
        else:
            i += 1
    NAS_LOG << ('eliinfo_tem', original_num - len(scores), len(scores))


def _rm_other_model(best_index):
    models = os.listdir(NAS_CONFIG['eva']['model_path'])
    models = [model for model in models if not re.search("model"+str(best_index), model)]
    for model in models:
        os.remove(os.path.join(NAS_CONFIG['eva']['model_path'], model))


def _global_train(net_pl, com, pro_pl, eva_winner):
    for i in range(MAIN_CONFIG['num_opt_best'] // MAIN_CONFIG['num_gpu'] + 1):
        if (i + 1) * MAIN_CONFIG['num_gpu'] > MAIN_CONFIG['num_opt_best']:
            task_num = MAIN_CONFIG['num_opt_best'] - MAIN_CONFIG['num_gpu'] * i
        else:
            task_num = MAIN_CONFIG['num_gpu']
        if task_num:
            round = i + 1
            _assign_task(net_pl, com, round, task_num)
            _do_task(pro_pl, com, eva_winner)
            _arrange_result(com, net_pl)


def _train_winner(net_pl, com, pro_pl, round):
    """

    Args:
        net_pool: list of NetworkUnit, and its length equals to 1
        round: the round number of game
    Returns:
        best_nn: object of Class NetworkUnit
    """
    NAS_LOG << "config_ops_ing"
    start_train_winner = time.time()
    eva_winner = Evaluator()
    _datasize_ctrl(eva_winner)
    
    if MAIN_CONFIG['pattern'] == "Block":
        _assign_task(net_pl, com, round, batch_num=MAIN_CONFIG['num_opt_best'], block_winner=True)
        _do_task(pro_pl, com, eva_winner)
        _arrange_result(com, net_pl)
    elif MAIN_CONFIG['pattern'] == "Global":
        _global_train(net_pl, com, pro_pl, eva_winner)
    best_nn = net_pl[0]
    _save_net_info(best_nn, len(Network.pre_block) + 1,
                   round, len(net_pl), best_nn.id, len(best_nn.item_list))
    scores = [x.score for x in best_nn.item_list[-MAIN_CONFIG['num_opt_best']:]]
    best_index = scores.index(max(scores))
    best_item_index = best_index - len(scores)
    if MAIN_CONFIG['pattern'] == "Block":
        _rm_other_model(best_index)
    NAS_LOG << ("train_winner_tem", time.time()-start_train_winner)
    return best_nn, best_item_index

# Debug function
import pickle
_OPS_PNAME = 'pcache\\ops_%d-%d-%d.pickle' % (
    NAS_CONFIG["enum"]["depth"], NAS_CONFIG["enum"]["width"], NAS_CONFIG["enum"]["max_depth"])


def _get_ops_copy():
    with open(_OPS_PNAME, 'rb') as f:
        pool = pickle.load(f)
    return pool


def _save_ops_copy(pool):
    with open(_OPS_PNAME, 'wb') as f:
        pickle.dump(pool, f)
    return


def _subproc_init_ops(net_pool, task_num):
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    from predictor import Predictor
    pred = Predictor()
    for nn in net_pool:
        _gpu_batch_init(nn, pred, task_num)
    return net_pool


def _init_ops(net_pool, process_pool):
    """Generates ops and skipping for every Network,

    Args:
        net_pool (list of NetworkUnit)
    Returns:
        net_pool (list of NetworkUnit)
        scores (list of score, and its length equals to that of net_pool)
    """
    # for debug
    if MAIN_CONFIG['ops_debug']:
        try:
            return _get_ops_copy()
        except:
            print('Nas: _get_ops_copy failed')
    with Pool(1) as p:
        net_pool = p.apply(_subproc_init_ops, args=(net_pool, MAIN_CONFIG['spl_network_round']))
    # net_pool = process_pool.apply_async(_subproc_init_ops, args=(net_pool, MAIN_CONFIG['spl_network_round']))
    # net_pool = net_pool.get()
    # for debug
    if MAIN_CONFIG['ops_debug']:
        _save_ops_copy(net_pool)
    return net_pool


def _init_npool_sampler(netpool, block_num):
    for nw in netpool:
        nw.spl = Sampler(nw.graph_template, block_num)
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
    NAS_LOG << ('start_game', len(net_pool))
    _init_npool_sampler(net_pool, block_num)

    NAS_LOG << 'config_ing'
    net_pool = _init_ops(net_pool, process_pool)
    round = 0
    start_game = time.time()
    while len(net_pool) > 1:
        start_round = time.time()
        round += 1
        _game(eva, net_pool, com, round, process_pool)
        _eliminate(net_pool, round)
        NAS_LOG << ('round_over', time.time()-start_round)
    NAS_LOG << ('get_winner', time.time()-start_game)
    best_nn, best_index = _train_winner(net_pool, com, process_pool, round+1)

    return best_nn, best_index


def _retrain():
    retrain_eva = Evaluator()
    _datasize_ctrl(retrain_eva)
    score = retrain_eva.evaluate([], is_bestNN=True, update_pre_weight=True)
    return score


class Nas:
    def __init__(self, pool):
        NAS_LOG << "init_ing"
        self.enu = Enumerater()
        self.eva = Evaluator()
        self.com = Communication()
        self.pool = pool

    def run(self):
        NAS_LOG << 'enuming'
        network_pool_tem = self.enu.enumerate()
        start_search = time.time()
        for i in range(MAIN_CONFIG["block_num"]):
            NAS_LOG << ('search_blk', i+1, MAIN_CONFIG["block_num"])
            start_block = time.time()
            block, best_index = algo(i, self.eva, self.com, network_pool_tem, self.pool)
            Network.pre_block.append([
                block.graph_template,
                block.item_list[best_index]
            ])
            NAS_LOG << ('search_blk_end', time.time() - start_block)
        NAS_LOG << ('nas_end', time.time() - start_search)
        start_retrain = time.time()
        retrain_score = random.random()
        # retrain_score = _retrain()
        NAS_LOG << ('retrain_end', retrain_score, time.time() - start_retrain)
        return Network.pre_block, retrain_score


if __name__ == '__main__':
    pool = Pool(processes=MAIN_CONFIG["num_gpu"])
    nas = Nas(pool)
    nas.run()
