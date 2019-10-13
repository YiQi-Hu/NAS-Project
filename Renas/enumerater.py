import copy
from queue import Queue
import time
import networkx as nx
import matplotlib.pyplot as plt
import pickle
from base import NetworkUnit

# from .base import NetworkUnit, NETWORK_POOL


class Enumerater:

    def __init__(self, depth=6, width=1, max_branch_depth=6):
        self.depth = depth
        self.width = width
        self.max_branch_depth = max_branch_depth
        self.info_dict = {}
        self.info_group = []
        self.log = ""
        self.pickle_name = 'depth=' + str(depth) + '_' + 'width=' + str(width) + '_' + "max_branch_depth=" + str(
            max_branch_depth) + '.pickle'

    # 生成Adjacney 填充全局变量NETWORK_POOL
    def enumerate(self):
        try:
            op = open(self.pickle_name, 'rb')
        except IOError:
            print('Starting enumeration...')
        else:
            print('Loading successfully!')
            pool = pickle.load(op)
            op.close()
            return pool

        # 生成链字典
        self.filldict()
        # print(len(self.info_dict))

        # 生成拓扑结构编号
        self.fillgroup()
        # print(len(self.info_group))

        # 还原拓扑结构
        pool = self.encode2adjaceny()

        op = open(self.pickle_name, 'wb')
        pickle.dump(pool, op)
        op.close()
        print('Saved ' + self.pickle_name)
        return pool
        # 填满全局变量；NetworkUnit类的序列[Net,Net,Net,...]

    # 遍历以起点i,终点j,链节点个数k，判断合法并加入字典结构种
    # max_branch_depth 规定支链节点个数不超过max_branch_depth个
    def filldict(self):
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

    # 以广搜的方法搜索 非递增拓扑结构编号
    def fillgroup(self):

        q = Queue()
        q.put([[], 0])
        while q.empty() != True:
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

    # 利用链的字典，拓扑结构编号，还原拓扑结构邻接表
    def encode2adjaceny(self):
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

    # 还原邻接表种的判重部分
    def judgemultiple(self, adja):
        for i in adja:
            for j in i:
                cnt = 0
                for k in i:
                    if j == k:
                        cnt += 1
                    if cnt >= 2:
                        return 0
        return 1

    # 一个比较烂的可视化
    def adjaceny2visualzation(self, adja):
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
