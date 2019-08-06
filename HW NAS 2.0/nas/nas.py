import random
import time
import tensorflow as tf
from multiprocessing import Process,Pool

from .enumerater import Enumerater
from .evaluater import Evaluater
from .optimizer import Optimizer
from .sampler import Sampler

NETWORK_POOL = []


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


class Nas:
    def __init__(self, m_best=1, opt_best_k=5, randseed=-1, depth=6, width=3, max_branch_depth=6):
        self.__m_best = m_best
        self.__m_pool = []
        self.__opt_best_k = opt_best_k
        self.__depth = depth
        self.__width = width
        self.__max_bdepth = max_branch_depth

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


    def __game(self, pros, spl, opt, eva, NETWORK_POOL):
        print("NAS: Now we have {0} networks. Start game!".format(len(NETWORK_POOL)))
        scores = []
        spl.renewp(pros)
        eva.add_data(800)
        i = 0

        while i < len(NETWORK_POOL):
            print(i)
            with Pool(1) as p:
                key=p.apply(run_proc, args=(NETWORK_POOL[i:], spl, eva, scores,))
                i+=key
            # try:
            #     p=Pool(1)
            #     key=p.apply(run_proc, args=(NETWORK_POOL[i:], spl, eva, scores,))
            #     # p = Process(target=run_proc, args=(NETWORK_POOL[i:], spl, eva, scores,))
            #     # key = p.start()
            # except Exception:
            #     print(key)
            #     i += key-1
            #     p.close()


        # for nn in NETWORK_POOL:
        #     spl_list = spl.sample(len(nn.graph_part))
        #     nn.cell_list.append(spl_list)
        #     # with Pool(1) as p:
        #     #     score=p.apply(eva.evaluate,args=(nn,))
        #     #     scores.append(score)
        #     #
        #     score = eva.evaluate(nn)
        #     scores.append(score)

        return scores

    def __train_winner(self, pros, spl, opt, eva, NETWORK_POOL):
        best_nn = NETWORK_POOL[0]
        best_opt_score = 0
        best_cell_i = 0
        spl.renewp(pros)
        eva.add_data(-1)
        for i in range(self.__opt_best_k):
            best_nn.cell_list.append(spl.sample(len(best_nn.graph_part)))
            opt_score = eva.evaluate(best_nn)
            if opt_score > best_opt_score:
                best_opt_score = opt_score
                best_cell_i = i
        print(best_opt_score)
        return best_nn, best_cell_i

    def run(self):
        """
        Algorithm Main Function
        """
        # Step 0: Initialize
        print("NAS start running...")
        enu, eva, spl, opt = self.__run_init()

        # Step 1: Brute Enumerate all possible network structures
        NETWORK_POOL = enu.enumerate()

        print("NAS: Enumerated all possible networks!")
        # Step 2: Search best structure
        pros = opt.sample()
        while (len(NETWORK_POOL) > 1):
            # Step 3: Sample, train and evaluate every network
            scores = self.__game(pros, spl, opt, eva, NETWORK_POOL)
            opt.update_model(pros, scores)
            pros = opt.sample()

            # Step 4: Eliminate half structures and increase dataset size
            self.__eliminate(NETWORK_POOL, scores)
            # self.__datasize_ctrl("same", epic)

        print("NAS: We got a WINNER!")
        # Step 5: Global optimize the best network
        best_nn, best_cell_i = self.__train_winner(pros, spl, opt, eva, NETWORK_POOL)

        # self.__save_log("", opt, spl, enu, eva)

        return best_nn, best_cell_i


if __name__ == '__main__':
    nas = Nas()
    print(nas.run())
