import copy
from queue import Queue
import time

import pickle
from base import Network
import matplotlib.pyplot as plt
from info_str import NAS_CONFIG

# from .base import NetworkUnit, NETWORK_POOL

# TODO Please fix your variables and functions naming.
# Ref: Google Code Style (Python) - Python 風格規範 “命名”

def _read_pool(path):
    pool = None
    try:
        f = open(path, 'rb')
    except IOError as e:
        print('Starting enumeration...')
    else:
        print('Loading successfully!')
        pool = pickle.load(f)
        f.close()
    return pool

def _save_pool(path, pool):
    with open(path, 'wb') as f:
        pickle.dump(pool, f)
    print('Saved in %s' % path)

class Enumerater:
    """Summary of class here.
        Generate adjacency of network topology.
        Attributes:
            parameters of enumerater module
                are given by folder 'parameters'.
    """

    def __init__(self):
        self.depth = NAS_CONFIG['enum']['depth']
        self.width = NAS_CONFIG['enum']['width']
        self.max_branch_depth = NAS_CONFIG['enum']['max_depth']
        self._info_dict = {}
        self._info_group = []
        self._log = ""
        self._pickle_name = 'pcache\\enum_%d-%d-%d.pickle' % (self.depth, self.width, self.max_branch_depth)

    def enumerate(self):
        """
        The main function of generating network topology.
        No Args.
        Retruns:
            1. pool (1d Network list)
        """
        pool = _read_pool(self._pickle_name)

        if pool and NAS_CONFIG['enum_debug']:
            return pool  # for debug

        self._filldict()  # Generate chain dictionary

        self._fillgroup()  # Generate topology number

        pool = self._encode2adjaceny()  # Restore network topology

        _save_pool(self._pickle_name, pool)
        return pool  # return the list of Network [Net,Net,...]

    def _filldict(self):
        """
        The starting node i, the ending node J, the number of chain nodes k,
        Judge legal and add to dictionary structure.
        """
        cnt = 0
        for i in range(self.depth - 2):
            for j in range(self.depth):
                if j <= i + 1:
                    continue
                for k in range(j - i):
                    if k < self.max_branch_depth:
                        # print(i,j,k)
                        self._info_dict[cnt] = [i, j, k]
                        cnt += 1
        return


    def _fillgroup(self):
        """
        Search for non-incrementing topology numbers by breadth-first search.
        """
        q = Queue()
        q.put([[], 0])
        while not q.empty():
            t = q.get()
            #   print(t[0])
            self._info_group.append(t[0])
            if t[1] == self.width:
                continue
            m = -1
            for i in t[0]:
                m = max(m, i)
            for i in range(len(self._info_dict)):
                if i >= m:
                    tmp = copy.deepcopy(t)
                    tmp[0].append(i)
                    tmp[1] += 1
                    q.put(tmp)
        return

    def _encode2adjaceny(self):
        """
        Use the dictionary of the chain, the topology number
        to restore the network topology adjacency list.
        """
        pool = []
        tmp_init = []
        id_tmp = 0
        for i in range(self.depth):
            if i != self.depth - 1:
                tmp_init.append([i + 1])
            else:
                tmp_init.append([])

        for g in self._info_group:
            tmp = copy.deepcopy(tmp_init)
            for i in g:
                info = self._info_dict[i]
                s = info[0]
                e = info[1]
                l = info[2]
                for o in range(l):
                    p = len(tmp)
                    tmp.append([])
                    tmp[s].append(p)
                    s = p
                tmp[s].append(e)
            if self._judgemultiple(tmp) == 1:
                tmp_net = Network(id_tmp, tmp)
                id_tmp += 1

                pool.append(tmp_net)
        return pool

    def _judgemultiple(self, adja):
        """
        Judging the repetition when restore the network topology adjacency list.
        """
        for i in adja:
            for j in i:
                cnt = 0
                for k in i:
                    if j == k:
                        cnt += 1
                    if cnt >= 2:
                        return 0
        return 1

    # def adjaceny2visualzation(self, adja):
    #     """
    #     Enter a network adjacency table to display its topology by pyplot.
    #     """
    #     import networkx as nx
    #     nodelist = []
    #     edgelist = []
    #     for i in range(len(adja)):
    #         nodelist.append(i)
    #         for j in adja[i]:
    #             edgelist.append((i, j))
    #     G = nx.DiGraph()
    #     G.add_nodes_from(nodelist)
    #     G.add_edges_from(edgelist)
    #     nx.draw(G, pos=None, with_labels=True)
    #     plt.show()

    def save_adj_log(self, POOL, PATH, date):
        for i, j in enumerate(POOL):
            s = 'nn_graph_'
            s = s + str(i) + '_'
            s = s + str(date)
            fp = open(PATH + s, "wb")
            # print(s)
            pickle.dump(j.graph_part, fp)


if __name__ == '__main__':
    time1 = time.time()

    obj = Enumerater()

    res = obj.enumerate()

    # for i in res:
    #     print(i.get_graph_part())  #
    # print(len(res))
    # obj.adjaceny2visualzation(res[1])

    time2 = time.time()
    print('Running time:', time2 - time1)

    NETWORK_POOL = res
    print(len(NETWORK_POOL))
    for i in NETWORK_POOL:
        print(i.graph_template)
        print(i.id)
    # 取 9，4 不约束支链节点数量产生 6980275 运行时间8分左右