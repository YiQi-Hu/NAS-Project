import copy
from queue import Queue
import time
import networkx as nx
import pickle
from base import NetworkUnit
import matplotlib.pyplot as plt
from info_str import NAS_CONFIG

# from .base import NetworkUnit, NETWORK_POOL

def read_pool(path):
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

def save_pool(path, pool):
    with open(path, 'wb') as f:
        pickle.dump(pool, f)
    print('Saved in %s' % path)

class Enumerater:
    """Summary of class here.
        Generate adjacency of network topology.
        Attributes:
            depth: Maximum length of the network topology.
            width: Number of branches in the network topology.
            max_branch_depth: The maximum number of nodes in the number of branches of the network topology
    """

    def __init__(self, depth=6, width=1, max_branch_depth=6):
        self.depth = depth
        self.width = width
        self.max_branch_depth = max_branch_depth
        self.info_dict = {}
        self.info_group = []
        self.log = ""
        self.pickle_name = 'pcache\\enum_%d-%d-%d.pickle' % (depth, width, max_branch_depth)

    def enumerate(self):
        """
        The main function of generating network topology.
        """
        pool = read_pool(self.pickle_name)

        if pool and NAS_CONFIG['enum_debug']:
            return pool  # for debug

        self.filldict()  # Generate chain dictionary

        self.fillgroup()  # Generate topology number

        pool = self.encode2adjaceny()  # Restore network topology

        save_pool(self.pickle_name, pool)
        return pool  # return the list of NetworkUnit [Net,Net,...]

    def filldict(self):
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
                        self.info_dict[cnt] = [i, j, k]
                        cnt += 1
        return


    def fillgroup(self):
        """
        Search for non-incrementing topology numbers by breadth-first search.
        """
        q = Queue()
        q.put([[], 0])
        while not q.empty():
            t = q.get()
            #   print(t[0])
            self.info_group.append(t[0])
            if t[1] == self.width:
                continue
            m = -1
            for i in t[0]:
                m = max(m, i)
            for i in range(len(self.info_dict)):
                if i >= m:
                    tmp = copy.deepcopy(t)
                    tmp[0].append(i)
                    tmp[1] += 1
                    q.put(tmp)
        return

    def encode2adjaceny(self):
        """
        Use the dictionary of the chain, the topology number
        to restore the network topology adjacency list.
        """
        pool = []
        tmp_init = []
        for i in range(self.depth):
            if i != self.depth - 1:
                tmp_init.append([i + 1])
            else:
                tmp_init.append([])

        for g in self.info_group:
            tmp = copy.deepcopy(tmp_init)
            for i in g:
                info = self.info_dict[i]
                s = info[0]
                e = info[1]
                l = info[2]
                for o in range(l):
                    p = len(tmp)
                    tmp.append([])
                    tmp[s].append(p)
                    s = p
                tmp[s].append(e)
            if self.judgemultiple(tmp) == 1:
                tmp_net = NetworkUnit(tmp, [])
                # print(tmp)
                pool.append(tmp_net)
        return pool

    def judgemultiple(self, adja):
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

    def adjaceny2visualzation(self, adja):
        """
        Enter a network adjacency table to display its topology by pyplot.
        """
        nodelist = []
        edgelist = []
        for i in range(len(adja)):
            nodelist.append(i)
            for j in adja[i]:
                edgelist.append((i, j))
        G = nx.DiGraph()
        G.add_nodes_from(nodelist)
        G.add_edges_from(edgelist)
        nx.draw(G, pos=None, with_labels=True)
        plt.show()

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
    D = 10
    W = 2
    W_number = 30
    obj = Enumerater(D, W, W_number)

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
        print(i.graph_part)
    # 取 9，4 不约束支链节点数量产生 6980275 运行时间8分左右