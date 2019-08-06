import random
import time
import os
import socket
import numpy as np
import tensorflow as tf
import multiprocessing
from multiprocessing import Process, Pool
from multiprocessing.managers import BaseManager

from .enumerater import Enumerater
from .evaluater import Evaluater
from .optimizer import Optimizer
from .sampler import Sampler

NETWORK_POOL = []
gpu_list = multiprocessing.Queue()


def run_proc(NETWORK_POOL, spl, eva, scores):
    for i, nn in enumerate(NETWORK_POOL):
        try:
            spl_list = spl.sample(len(nn.graph_part))
            nn.cell_list.append(spl_list)
            score = eva.evaluate(nn)
            scores.append(score)
        except Exception as e:
            print(e)
            return i


def call_eva(eva, nn, pos, ngpu, start_time):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(ngpu)
    score, time_cost = eva.evaluate(nn, pos, start_time)
    gpu_list.put(ngpu)
    return score, time_cost


class QueueManager(BaseManager):
    pass


class Nas:
    def __init__(self, ps_host, worker_host, job_name, task_index, m_best=1, opt_best_k=5, randseed=-1, depth=6,
                 width=3, max_branch_depth=6, num_gpu=1):
        self.__m_best = m_best
        self.__m_pool = []
        self.__opt_best_k = opt_best_k
        self.__depth = depth
        self.__width = width
        self.__max_bdepth = max_branch_depth
        self.num_gpu = num_gpu
        self.ps_host = ps_host
        self.worker_host = worker_host
        self.job_name = job_name
        self.task_index = task_index

        if randseed is not -1:
            random.seed(randseed)
            tf.set_random_seed(randseed)

        return

    def __list_swap(self, ls, i, j):
        cpy = ls[i]
        ls[i] = ls[j]
        ls[j] = cpy

    def __eliminate(self, network_pool=None, scores=[]):
        """
        Eliminates the worst 50% networks in network_pool depending on scores.
        """
        scores_cpy = scores.copy()
        scores_cpy.sort()
        mid_val = scores_cpy[len(scores) // 2]

        i = 0
        while (i < len(network_pool)):
            if scores[i] < mid_val:
                # del network_pool[i]   # TOO SLOW !!
                # del scores[i]
                self.__list_swap(network_pool, i, len(network_pool) - 1)
                self.__list_swap(scores, i, len(scores) - 1)
                network_pool.pop()
                scores.pop()
            else:
                i += 1
        return mid_val

    def __datasize_ctrl(self, type="", eva=None):
        """
        Increase the dataset's size in different way
        """
        # TODO Where is Class Dataset?
        cur_train_size = eva.get_train_size()
        if type.lower() == 'same':
            nxt_size = cur_train_size * 2
        else:
            raise Exception("NAS: Invalid datasize ctrl type")

        eva.set_train_size(nxt_size)
        return

    def __save_log(self, path="",
                   optimizer=None,
                   sampler=None,
                   enumerater=None,
                   evaluater=None):
        with open(path, 'w') as file:
            file.write("-------Optimizer-------")
            file.write(optimizer.log)
            file.write("-------Sampler-------")
            file.write(sampler.log)
            file.write("-------Enumerater-------")
            file.write(enumerater.log)
            file.write("-------Evaluater-------")
            file.write(evaluater.log)
        return

    def __run_init(self):
        enu = Enumerater(
            depth=self.__depth,
            width=self.__width,
            max_branch_depth=self.__max_bdepth)
        eva = Evaluater()
        spl = Sampler()
        opt = Optimizer(spl.dim, spl.parameters_subscript)

        sample_size = 3  # the instance number of sampling in an iteration
        budget = 20000  # budget in online style
        positive_num = 2  # the set size of PosPop
        rand_probability = 0.99  # the probability of sample in model
        uncertain_bit = 3  # the dimension size that is sampled randomly
        # set hyper-parameter for optimization, budget is useless for single step optimization
        opt.set_parameters(ss=sample_size, bud=budget, pn=positive_num, rp=rand_probability, ub=uncertain_bit)
        # clear optimization model
        opt.clear()

        return enu, eva, spl, opt

    def __game(self, pros, spl, eva, NETWORK_POOL, task, result, round, best_score):
        print("NAS: Now we have {0} networks. Start game!".format(len(NETWORK_POOL)))
        scores = np.zeros(len(NETWORK_POOL))
        result_list = []

        spl.renewp(pros)
        eva.add_data(800)

        network_index = -1
        # put all the network in this round into the task queue
        for nn in NETWORK_POOL:
            spl_list = spl.sample(len(nn.graph_part))
            nn.cell_list.append(spl_list)
            network_index += 1
            task.put([nn, network_index, round])

        # TODO data size control
        eva.add_data(1280)
        for gpu in range(self.num_gpu):
            gpu_list.put(gpu)

        pool = Pool(processes=self.num_gpu)
        # as long as there's still task available
        while not task.empty():
            gpu = gpu_list.get()  # get a gpu resource
            try:
                nn, network_index, _ = task.get(timeout=1)
            except:  # if failed, give back gpu resource and break out the loop, this might indicates this round is over
                gpu_list.put(gpu)
                break
            eva_result = pool.apply(call_eva, args=(eva, nn, network_index, gpu, time.time()))
            result_list.append([eva_result, network_index, nn])

        for res in result_list:
            tmp = res[0].get()
            # The order of result is evaluation score,network index, network configuration,duration time
            result.put(tmp[0], res[1], res[2], tmp[1])

        pool.close()
        pool.join()

        network_count = 0
        while network_count < len(NETWORK_POOL):
            tmp = result.get()
            scores[tmp[1]] = tmp[0]
            if tmp[0] > best_score:
                # TODO save the best score so far
                best_score = tmp[0]

        return scores.tolist(), best_score

    def __train_winner(self, pros, spl, eva, NETWORK_POOL):
        eva.add_data(-1)
        best_nn = NETWORK_POOL[0]
        best_opt_score = eva.evaluate(best_nn, 0)
        l = len(best_nn.cell_list)
        best_cell_i = l
        spl.renewp(pros)

        for i in range(self.__opt_best_k):
            best_nn.cell_list.append(spl.sample(len(best_nn.graph_part)))
            opt_score = eva.evaluate(best_nn, i + 1)
            if opt_score > best_opt_score:
                best_opt_score = opt_score
                best_cell_i = i + l
        print(best_opt_score)
        return best_nn, best_cell_i

    def run(self):
        """
        Algorithm Main Function
        """
        # Step 0: Initialize
        print("NAS start running...")
        enu, eva, spl, opt = self.__run_init()
        best_score = 0

        # Different code is ran on different machine depends on whether it's a ps host or a worker host.
        # PS host is used for generating all the networks and collect the final result of evaluation.
        # And PS host together with worker hosts(as long as they all have GPUs) train as well as evaluate all the networks asynchronously inside one round and synchronously between rounds
        if self.job_name == 'ps':
            # Step 1: Brute Enumerate all possible network structures
            NETWORK_POOL = enu.enumerate()
            print("NAS: Enumerated all possible networks!")

            # Step 2: Search best structure
            pros = opt.sample()

            # Step 3: Sample, train and evaluate every network
            task_queue = multiprocessing.Queue()
            result_queue = multiprocessing.Queue()
            flag = multiprocessing.Queue()

            QueueManager.register('get_task_queue', callable=lambda: task_queue)
            QueueManager.register('get_result_queue', callable=lambda: result_queue)
            QueueManager.register('get_flag', callable=lambda: flag)

            # TODO there might be other ways to get the IP address
            server_addr = socket.gethostbyname(self.ps_host.split(".")[0])
            manager = QueueManager(address=(server_addr, int(self.ps_host.split(":")[1])), authkey=b'abc')
            manager.start()

            task = manager.get_task_queue()
            result = manager.get_result_queue()
            end_flag = manager.get_flag()  # flag for whether the whole process is over

            round = 1
            while (len(NETWORK_POOL) > 1):
                scores, best_score = self.__game(pros, spl, eva, NETWORK_POOL, task, result, round, best_score)
                opt.update_model(pros, scores)
                pros = opt.sample()
                round += 1

                # Step 4: Eliminate half structures and increase dataset size
                self.__eliminate(NETWORK_POOL, scores)
                # self.__datasize_ctrl("same", epic)

            print("NAS: We got a WINNER!")
            # Step 5: Global optimize the best network
            best_nn, best_cell_i = self.__train_winner(pros, spl, eva, NETWORK_POOL)

            end_flag.put(1)
            manager.shutdown()
            # self.__save_log("", opt, spl, enu, eva)

            return best_nn, best_cell_i
        else:
            result_list = []
            time.sleep(10)  # To ensure worker hosts are activated after PS host

            QueueManager.register('get_task_queue')
            QueueManager.register('get_result_queue')
            QueueManager.register('get_flag')

            # TODO there might be other ways to get the IP address
            server_addr = socket.gethostbyname(self.ps_host.split(".")[0])
            manager = QueueManager(address=(server_addr, int(self.ps_host.split(":")[1])), authkey=b'abc')
            manager.connect()

            task = manager.get_task_queue()
            result = manager.get_result_queue()
            end_flag = manager.get_flag()  # flag for whether the whole process is over

            for gpu in range(self.num_gpu):
                gpu_list.put(gpu)

            pool = Pool(processes=self.num_gpu)

            last_round = -1
            while end_flag.empty():
                # as long as there's still task available
                while not task.empty():
                    gpu = gpu_list.get()  # get a gpu resource
                    try:
                        nn, network_index, round = task.get(timeout=1)
                    except:  # if failed, give back gpu resource and break out the loop, this might indicates this round is over
                        gpu_list.put(gpu)
                        break
                    # Data control for new round
                    if round!=last_round:
                        eva.add_data(1280)
                    eva_result = pool.apply(call_eva, args=(eva, nn, network_index, gpu, time.time()))
                    result_list.append([eva_result, network_index, nn])

                for _ in range(len(result_list)):
                    res=result_list.pop(0)
                    tmp = res[0].get()
                    # The order of result is evaluation score,network index, network configuration,duration time
                    result.put(tmp[0], res[1], res[2], tmp[1])
            pool.close()
            pool.join()


if __name__ == '__main__':
    nas = Nas()
    print(nas.run())
