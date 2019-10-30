import random
import time
import os
import sys
import copy
import socket
import numpy as np
import tensorflow as tf
import multiprocessing
from multiprocessing import Process, Pool
from .utils import Communication, _list_swap, try_secure
from .logger import Logger
from .enumerater import Enumerater
from .evaluator import Evaluator
# from .optimizer import Optimizer
# from .sampler import Sampler

gpu_list = multiprocessing.Queue()


def _gpu_eva(params, spl_times, eva, ngpu):
    graph, cell, nn_pb, cur_best_score, round, pos, spl_index, is_bestnn, finetune_signal, pool_len = params
    os.environ["CUDA_VISIBLE_DEVICES"] = str(ngpu)
    with open("memory/evaluating_log_with_gpu{}.txt".format(ngpu), "a") as f:
        print("\nblock_num:{} round:{} network_index:{}/{} spl_index: {}/{} eva_pid: {} gpu: {}"
              .format(len(nn_pb)+1, round, pos, pool_len, spl_index, spl_times, os.getpid(), ngpu))
        print("graph:", graph)
        print("cell:", cell)
        f.write("\nblock_num:{} round:{} network_index:{}/{} spl_index: {}/{} eva_pid: {} gpu: {}\n"
                .format(len(nn_pb)+1, round, pos, pool_len, spl_index, spl_times, os.getpid(), ngpu))
        f.write(str(graph)+"\n"+str(cell)+"\n"+str(nn_pb)+"\n")
        start_time = time.time()
        score = try_secure("eva", eva.evaluate, args=(graph, cell, nn_pb, cur_best_score, is_bestnn,
                                                      finetune_signal, f), f=f)
        # score = eva.evaluate(graph, cell, nn_preblock, False, finetune_signal, f)
        end_time = time.time()
    time_cost = end_time - start_time
    gpu_list.put(ngpu)
    return score, time_cost, pos, spl_index


def _init_ops(net_pl, spl_times):
    from .predictor import Predictor
    pred = Predictor()
    for network in net_pl:  # initialize the full network by adding the skipping and ops to graph_part
        for i in range(spl_times):
            result = try_secure("spl", network.spl.sample, args=(), in_child=True)
            if result:
                cell, graph = result
                table = network.spl.p
            else:
                return
            # network.spl.sample()
            # network.graph_full_list.append(graph)
            blocks = []
            for block in network.pre_block:  # get the graph_full adjacency list in the previous blocks
                blocks.append(block[1])  # only get the graph_full in the pre_bock

            if i % 2 == 0:
                # process the ops from predictor
                pred_ops = pred.predictor(blocks, graph)
                table = network.spl.init_p(pred_ops)  # spl refer to the pred_ops
                network.spl.renewp(table)
                cell, graph = network.spl.convert()  # convert the table

            # graph from first sample and second sample are the same,
            # so that we don't have to assign network.graph_full at first time
            network.graph_full_list.append(graph)
            network.cell_list.append(cell)
            network.table_list.append(table)
    return net_pl


class Nas:
    def __init__(self, pattern, setting, search_space):
        self.__pattern = pattern

        self.num_gpu = setting["num_gpu"]
        if self.__pattern == "Block":
            self.__finetune_threshold = setting["finetune_threshold"]
            self.spl_round_net = setting["spl_one_round_one_network"]
            self.eliminate_policy = setting["eliminate_policy"]
            self.add_data_every_round = setting["add_data_every_round"]
        self.__opt_best_k = setting["opt_best_k"]
        self.add_data_for_winner = setting["add_data_for_winner"]
        self.__block_num = search_space["block_num"]
        self.eva_para = setting["eva_para"]

        print("NAS: Initializing eva...")
        self.eva = Evaluator(self.eva_para) if self.__pattern == "Block" else None
        print("NAS: Initializing enu...")
        self.enu = Enumerater(search_space["graph"])
        print("NAS: Initializing com...")
        self.com = Communication()
        print("NAS: Initializing log...")
        self.log = Logger("network_info.txt")

        for gpu in range(self.num_gpu):
            gpu_list.put(gpu)

        self.spl_setting = setting["spl_para"]
        self.skipping_max_dist = search_space["skipping_max_dist"]
        self.skipping_max_num = search_space["skipping_max_num"]
        self.ops_space = search_space["ops"]

        if setting["rand_seed"] is not -1:
            random.seed(setting["rand_seed"])
            tf.set_random_seed(setting["rand_seed"])

        return

    def __eliminate(self, net_pl=None, round=0):
        """
        Eliminates the worst 50% networks in network_pool depending on scores.
        """
        if self.eliminate_policy == "best_score_have_ever_met":
            scores = [net_pl[nn_index].best_score for nn_index in range(len(net_pl))]
        if self.eliminate_policy == "average_score_this_round":
            scores = [np.mean(net_pl[nn_index].score_list[-self.spl_round_net:])
                      for nn_index in range(len(net_pl))]
        print("scores:", scores)
        scores_cpy = copy.deepcopy(scores)
        scores_cpy.sort()
        original_num = len(scores)
        mid_index = original_num // 2
        mid_val = scores_cpy[mid_index]
        print("mid_val:", mid_val)
        # record the number of the removed network in the pool
        original_index = [i for i in range(len(scores))]

        i = 0
        while i < len(scores):
            if scores[i] < mid_val:
                _list_swap(net_pl, i, len(net_pl) - 1)
                _list_swap(original_index, i, len(original_index) - 1)
                _list_swap(scores, i, len(scores) - 1)
                self.log.save_info(net_pl.pop(), round, original_index.pop(), original_num)
                scores.pop()
            else:
                i += 1
        print("NAS: eliminating {}, remaining {}...".format(original_num - len(net_pl), len(net_pl)))
        return mid_val

    def __assign_task(self, net_pl, com, rd, i, task_num, in_game):
        pool_len = len(net_pl)
        network_index = 1
        is_bestnn = not in_game
        pre_rd = i  # 0 if in game

        # put all the network in this round into the task queue
        if self.__pattern == "Block":
            finetune_signal = (pool_len < self.__finetune_threshold)  # used in Block mode
        else:
            finetune_signal = True
        for nn in net_pl:
            if rd == 1:
                cell_l, graph_l = nn.cell_list, nn.graph_full_list
                for spl_index in range(task_num):
                    graph, cell = graph_l[spl_index], cell_l[spl_index]
                    com.task.put([graph, cell, nn.pre_block, nn.best_score, rd, network_index,
                                  pre_rd * self.num_gpu + spl_index + 1, is_bestnn, finetune_signal, pool_len])
                    # cell, graph = nn.spl.sample()
            else:
                for index in range(1, 1 + task_num):
                    nn.spl.update_opt_model(nn.table_list[-index], nn.score_list[-index])  # -1
                for spl_index in range(task_num):
                    cell, graph = try_secure("spl", nn.spl.sample, args=())
                    # cell, graph = nn.spl.sample()
                    nn.graph_full_list.append(graph)
                    nn.cell_list.append(cell)
                    nn.table_list.append(nn.spl.p)
                    com.task.put([graph, cell, nn.pre_block, nn.best_score, rd, network_index,
                                  pre_rd * self.num_gpu + spl_index + 1, is_bestnn, finetune_signal, pool_len])
            network_index += 1

    def __datasize_ctrl(self, eva, in_game):
        if self.__pattern == "Block" and in_game:
            eva.add_data(self.add_data_every_round)
        else:
            eva.add_data(self.add_data_for_winner)

    def __do_task(self, com, eva, task_num):
        pool = Pool(self.num_gpu)
        # as long as there's still task available
        while not com.task.empty():  # not beautiful in sync
            gpu = gpu_list.get()  # get a gpu resource
            try:  # try except for multi host
                task_params = com.task.get(timeout=1)
            except:  # if failed, give back gpu resource and break out the loop, this might indicates this round is over
                gpu_list.put(gpu)
                break
            graph, cell, nn_pre_block, _, round, network_index, spl_index, _, finetune_signal, pool_len = task_params
            print("#########start to evaluate, block_num:{} round:{} network_index:{}/{} spl_index: {}/{} ##########"
                  .format(len(nn_pre_block)+1, round, network_index, pool_len, spl_index, task_num))
            eva_result = pool.apply_async(_gpu_eva, args=(task_params, task_num, eva, gpu))
            com.result.put(eva_result)
        pool.close()
        pool.join()

    def __arrange_result(self, com, net_pl, task_num):
        assert com.result.qsize() == task_num, \
            "The number of scores collected do not equal to that of tasks..."

        # fill the score list
        print("network_index spl_index score time_cost")
        while not com.result.empty():
            eva_result = com.result.get()
            score, time_cost, network_index, spl_index = eva_result.get()
            print([network_index, spl_index, score, time_cost])
            # mark down the score it has met
            net_pl[network_index-1].score_list.append(score)
            # mark down the best score and the index it has met
            net_pl[network_index-1].update_best_score(score)

    def __game(self, eva, net_pl, com, round):
        in_game = True
        self.__assign_task(net_pl, com, round, 0, self.spl_round_net, in_game)
        self.__datasize_ctrl(eva, in_game)
        self.__do_task(com, eva, self.spl_round_net)
        task_num = len(net_pl)*self.spl_round_net
        self.__arrange_result(com, net_pl, task_num)

    def __train_winner(self, net_pl, com, round):
        print("NAS: We start to train winner!")
        start_train_winner = time.time()
        in_game = False
        eva_winner = Evaluator(self.eva_para)
        self.__datasize_ctrl(eva_winner, in_game)

        for i in range(self.__opt_best_k//self.num_gpu+1):
            if (i+1)*self.num_gpu > self.__opt_best_k:
                task_num = self.__opt_best_k - self.num_gpu * i
            else:
                task_num = self.num_gpu
            if task_num:
                if self.__pattern == "Global":
                    round = i + 1
                self.__assign_task(net_pl, com, round, i, task_num, in_game)
                self.__do_task(com, eva_winner, self.__opt_best_k)
                self.__arrange_result(com, net_pl, task_num)
        best_nn = net_pl[0]
        self.log.save_info(best_nn, round, 0, 1)
        print("NAS: Train winner finished and cost time: ", time.time() - start_train_winner)
        return best_nn

    def initialize_ops(self, net_pl):
        print("NAS: Configuring the networks in the first round...")
        start_init = time.time()
        init_num = self.spl_round_net if self.__pattern == "Block" else self.num_gpu
        with Pool(1) as p:
            net_pl = p.apply(_init_ops, args=(net_pl, init_num))
        if not net_pl:
            sys.exit(0)
        print("NAS: Configure finished, cost time: ", time.time() - start_init)
        return net_pl

    def algorithm(self, block_num, eva, com, net_pl_tmp):
        # implement the copy when searching for every block
        net_pl = copy.deepcopy(net_pl_tmp)
        for network in net_pl:  # initialize the sample module
            network.init_sample(self.__pattern, block_num, self.spl_setting, self.skipping_max_dist,
                                self.skipping_max_num, self.ops_space)

        net_pl = self.initialize_ops(net_pl)
        round = 0
        start_game = time.time()
        print("NAS: We start the game!")
        while (len(net_pl) > 1):
            start_round = time.time()
            # Step 3: Sample, train and evaluate every network
            round += 1
            self.__game(eva, net_pl, com, round)
            # Step 4: Eliminate half structures and increase dataset size
            self.__eliminate(net_pl, round)
            print("NAS: The round is over, cost time: ", time.time() - start_round)
            # self.__datasize_ctrl("same", epic)
        print("NAS: We got a WINNER and game cost: ", time.time() - start_game)
        # Step 5: Global optimize the best network
        best_nn = self.__train_winner(net_pl, com, round+1)
        # self.__save_log("", opt, spl, enu, eva)
        return best_nn

    def run(self):
        print("NAS: Enumerating all possible networks!")
        net_pool_template = self.enu.enumerate()
        start_search = time.time()
        for i in range(self.__block_num):
            print("NAS: Searching for block {}/{}...".format(i + 1, self.__block_num))
            start_block = time.time()
            block = self.algorithm(i, self.eva, self.com, net_pool_template)
            block.pre_block.append([block.graph_part, block.graph_full_list[block.best_index],
                                    block.cell_list[block.best_index], block.best_score])
            print("NAS: Search current block finished and cost time: ", time.time() - start_block)
            # or NetworkUnit.pre_block.append()
        print("NAS: Search finished and cost time: ", time.time() - start_search)
        return block.pre_block  # or NetworkUnit.pre_block


if __name__ == '__main__':
    nas = Nas()
    print(nas.run())
