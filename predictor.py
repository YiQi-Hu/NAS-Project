import numpy as np
from enumerater import Enumerater
from predict_op.label_encoding import decoder, encoder, getClassNum
from keras.utils.np_utils import to_categorical
from keras.models import model_from_json
import time
MAX_NETWORK_LENGTH = 71
model_json_path = './predict_op/model.json'
model_weights_path = './predict_op/model.json.h5'


class Feature:
    def __init__(self, graph):
        self.graph = graph

    def feature_links(self):
        # 从邻接矩阵中提取所有的支链，每一条支链有五个特征，编号，起点，终点，长度，节点编号
        g = self.graph
        endpoint = np.zeros((len(g), 1),dtype=int)
        endpoint[0] = 1
        link_set = []
        endpoint_link_num_set = []
        node_link_num_set = []
        for i in range(1, len(g)):
            out_link_num = 0
            in_link_num = 0
            for j in range(len(g)):
                if g[i][j] == 1:
                    out_link_num += 1
                if g[j][i] == 1:
                    in_link_num += 1
                if out_link_num > 1 and in_link_num > 1:
                    break
            if out_link_num != 1:
                endpoint[i] = 1
            if in_link_num > 1:
                endpoint[i] = 1
        link_id = 0
        for i in range(len(endpoint)):
            if endpoint[i] == 1:
                for j in range(len(g)):
                    if g[i][j] == 1:

                        link = [link_id, i, 0, 0, []]
                        link = self._find_link(endpoint, j, link, g, node_link_num_set)
                        link_id += 1
                        link_set.append(link)

        for i in range(len(endpoint)):
            if endpoint[i] == 1:
                i_links_num_set = []
                i_links_num_set.append(i)
                for j in range(len(link_set)):
                    if link_set[j][1] == i or link_set[j][2] == i:
                        i_links_num_set.append(j)
                endpoint_link_num_set.append(i_links_num_set)

        return link_set, endpoint_link_num_set, node_link_num_set

    def feature_nodes(self):
        # 对每一个节点提取特征
        link_set, endpointLinkNumSet, nodeLinkNumSet = self.feature_links()

        node_num = len(self.graph)
        feature_num = 25
        node_feature = np.zeros((node_num, feature_num), dtype=float)

        node_feature[:, 0] = node_num
        node_feature[:, 1] = len(link_set)

        max_length, max_link_index = self._find_max_link(link_set)
        min_length, min_link_index = self._find_min_link(link_set)
        node_feature[:, 2] = max_length
        node_feature[:, 3] = max_link_index
        node_feature[:, 4] = min_length
        node_feature[:, 5] = min_link_index

        mean = len(link_set)/node_num
        link_len = []
        for i in range(len(link_set)):
            link_len.append(link_set[i][3])
        var = np.var(link_len)
        node_feature[:, 6] = mean
        node_feature[:, 7] = var

        # 全局特征，端点个数
        endpoint_num = len(endpointLinkNumSet)
        node_feature[:, 8] = endpoint_num

        global_num = 9

        # 局部特征
        for i in range(node_num):

            node_feature[i][global_num] = self._is_endpoint(i, endpointLinkNumSet)
            node_feature[i][global_num+1] = i

            if node_feature[i][global_num] == 1:
                node_feature[i][global_num+2] = self._link_num(i, endpointLinkNumSet)
                links = self._find_endpoint_link_set(i, endpointLinkNumSet, link_set)
                node_feature[i][global_num+3] = self._mean_link(links)
                node_feature[i][global_num+4] = self._var_link(links)
                _, max_length = self._find_max_link(links)
                _, min_length = self._find_min_link(links)
                node_feature[i][global_num+5] = max_length
                node_feature[i][global_num+6] = min_length

            else:

                link = self._find_node_link(i, nodeLinkNumSet, link_set)
                node_feature[i][global_num+7] = self._relative_Loc(i, link)
                node_feature[i][global_num+8] = link[3]
                node_feature[i][global_num+9] = link[1]
                node_feature[i][global_num+10] = link[2]

                links = self._find_node_links(i, nodeLinkNumSet, link_set)
                node_feature[i][global_num+11] = self._mean_link(links)
                node_feature[i][global_num+12] = self._var_link(links)

        return node_feature

    @staticmethod
    def _find_endpoint_link_set(id, endpoint_link_num_set, link_set):
        # 寻找端点的支链集
        links = []
        for i in range(len(endpoint_link_num_set)):
            if endpoint_link_num_set[i][0] == id:
                for e in range(1, len(endpoint_link_num_set[i])):
                    links.append(link_set[endpoint_link_num_set[i][e]])
                break
        return links

    @staticmethod
    def _find_node_links(id, node_link_num_set, link_set):
        # 寻找与节点所在支链有相同端点的支链集
        links = []
        link_num = 0
        for i in range(len(node_link_num_set)):
            if node_link_num_set[i][0] == id:
                link_num = node_link_num_set[i][1]
                break
        for i in range(len(link_set)):
            if link_set[i][1] == link_set[link_num][1] and link_set[i][2] == link_set[link_num][2]:
                links.append(link_set[i])
        return links

    @staticmethod
    def _find_node_link(id, node_link_num_set, link_set):
        # 寻找节点所在的支链
        for i in range(len(node_link_num_set)):
            if node_link_num_set[i][0] == id:
                link_num = link_set[node_link_num_set[i][1]]
                return link_num

    @staticmethod
    def _is_endpoint(node_num, endpoint_link_num_set):
        # 判断是否为端点
        for i in range(len(endpoint_link_num_set)):
            if endpoint_link_num_set[i][0] == node_num:
                return 1
        return 0

    @staticmethod
    def _link_num(node_id, endpoint_link_num_set):
        # 支链的个数
        for i in range(len(endpoint_link_num_set)):
            if endpoint_link_num_set[i][0] == node_id:
                e = endpoint_link_num_set[i]
                return len(e)-1
        return 0

    @staticmethod
    def _mean_link(link_set):
        # 支链长度的期望
        links_len = []
        for e in link_set:
            links_len.append(e[3])
        return np.mean(links_len)

    @staticmethod
    def _var_link(link_set):
        # 支链的方差
        links_len = []
        for e in link_set:
            links_len.append(e[3])
        return np.var(links_len)

    @staticmethod
    def _relative_Loc(id, link):
        # 节点在支链中的相对位置
        for i in range(len(link[4])):
            if link[4][i] == id:
                return i+1

    @staticmethod
    def _find_max_link(link_set):
        # 寻找最长支链
        max_length = 0
        index = 0
        for i in range(len(link_set)):
            if link_set[i][3] > max_length:
                index = i
                max_length = link_set[i][3]
        return index, max_length

    @staticmethod
    def _find_min_link(link_set):
        # 寻找最短支链
        min_length = 0
        index = 0
        for i in range(len(link_set)):
            if link_set[i][3] < min_length:
                index = i
                min_length = link_set[i][3]
        return index, min_length

    def _find_link(self, endpoint, id, link, G, node_link_num_set):
        # 递归搜索链上的所有节点
        if endpoint[id] == 1:
            link[2] = id
            return link
        else:
            link[3] += 1
            link[4].append(id)
            node_link_num_set.append([id, link[0]])
            for i in range(len(G)):
                if G[id][i] == 1:
                    link = self._find_link(endpoint, i, link, G, node_link_num_set)
                    break
        return link


class Predictor:
    def __init__(self):
        self.cell_list = []
        with open(model_json_path, 'r') as file:
            model_json = file.read()
        self.model = model_from_json(model_json)
        self.model.load_weights(model_weights_path)

    @staticmethod
    def _list2mat(G):
        # 将领接表转换成邻接矩阵
        graph = np.zeros((len(G), len(G)), dtype=int)
        for i in range(len(G)):
            e = G[i]
            if e:
                for k in e:
                    graph[i][k] = 1
        return graph

    @staticmethod
    def _graph_concat(graphs):
        if len(graphs) == 1:
            return graphs[0]
        elif len(graphs) > 1:
            new_graph_length = 0
            for g in graphs:
                new_graph_length += len(g)
            new_graph = np.zeros((new_graph_length, new_graph_length), dtype=int)
            x_index = 0  # the staring connection position of next graph
            y_index = 0
            for g in graphs:
                new_graph[x_index:x_index + len(g), y_index:y_index + len(g)] = g
                if y_index + len(g) < new_graph_length:
                    new_graph[x_index + len(g) - 1][y_index + len(g)] = 1
                x_index = x_index + len(g)
                y_index = y_index + len(g)

            return new_graph

    @staticmethod
    def _get_new_order(links, graph_size):
        # 获得节点在新的编码方式下的顺序
        new_order = np.zeros((2, graph_size), dtype=int)
        for i in range(graph_size):
            new_order[0][i] = new_order[1][i] = i
        for l in links:
            nodes = l[4]
            if nodes:
                if nodes[0] > l[2]:
                    for i in range(len(nodes)):
                        new_order[1][nodes[i]] = l[1] + i + 1
        new_order = np.argsort(new_order[1, :])
        return new_order

    @staticmethod
    def _get_new_mat(new_order, mat):
        # 获得在新的编码方式下网络结构的邻接矩阵
        size = len(mat)
        graph = np.zeros((size, size), dtype=int)
        for i in range(size):
            e = mat[i]
            if e:
                for k in e:
                    pre = int(np.argwhere(new_order == i))
                    after = int(np.argwhere(new_order == k))
                    graph[pre][after] = 1
        return graph

    @staticmethod
    def _padding(node_feature, length):
        # 对输入数据做填充，保证输入数据的一致性
        if len(node_feature) < length:
            add = np.ones((length - len(node_feature), len(node_feature[0])))
            add = -add
            node_feature = np.append(node_feature, add, axis=0)

        return node_feature

    def _trans(self, graphs):
        # 对输入的邻接表重新编码并转换成矩阵的形式提取特征
        graphs_mat = []
        graphs_orders = []
        for g in graphs:
            g_mat = self._list2mat(g)
            links, _, _ = Feature(g_mat).feature_links()
            order = self._get_new_order(links, len(g_mat))
            graph_mat = self._get_new_mat(order, g)
            graphs_mat.append(graph_mat)
            graphs_orders.append(order)
        return graphs_mat, graphs_orders

    @staticmethod
    def _class_id_2_parameter(order, class_list):
        # 将最后输出的类别转换成需要预测的操作详细参数
        parameters = decoder(class_list)
        parameters_cp = parameters.copy()
        for i in range(len(order)):
            parameters[order[i]] = parameters_cp[i]
        return parameters[:len(order)]

    @staticmethod
    def _save_model(model, json_path, weights_path):
        model_json = model.to_json()
        with open(json_path, 'w') as file:
            file.write(model_json)
            model.save_weights(weights_path)

    @staticmethod
    def _my_param_style(cell_list):
        filter_size = [16, 32, 48, 64, 96, 128, 192, 256, 512, 1024]
        pool_size = [2, 3, 4, 5, 7]
        labels = []
        for cell in cell_list:
            label = []
            if cell[0] == 'conv':
                for f_size in filter_size:
                    if cell[1] <= f_size:
                        label = [1, [str(cell[2]), str(f_size), 'relu', '0', '0']]
            else:
                if cell[1] == 'max' or cell[1] == 'avg':
                    for p_size in pool_size:
                        if cell[2] <= p_size:
                            label = [0, ['pool ' + cell[1], str(p_size)]]

                elif cell[1] == 'global':
                    label = [0, ['pool avg', 'global']]
            labels.append(label)
        return labels

    def _predict(self, inputs):
        # 根据输入特征预测操作
        inputs = np.array(inputs)
        inputs = np.reshape(inputs, (1, inputs.shape[0], inputs.shape[1]))
        # model = load_model(model_json_path, model_weights_path)
        predict_y = self.model.predict(inputs)
        predict_y = np.reshape(predict_y, (predict_y.shape[1], predict_y.shape[2]))
        output = []
        for i in range(len(inputs[0])):
            output.append(np.argmax(predict_y[i]))
        return output

    # 模块接口
    def predictor(self, blocks, graph_part):
        graph_list = []
        if blocks:
            for block in blocks:
                graph_list.append(block)
        graph_list.append(graph_part)
        graphs_mat, graphs_orders = self._trans(graph_list)
        new_graph = self._graph_concat(graphs_mat)
        inputs = Feature(new_graph).feature_nodes()
        inputs = self._padding(inputs, MAX_NETWORK_LENGTH)
        class_list = self._predict(inputs)
        self.cell_list = self._class_id_2_parameter(graphs_orders[-1],
                                                    class_list[len(new_graph) - len(graph_part):len(new_graph)])
        return self.cell_list

    def train(self, graph_part_group, cell_list_group, batch_size=32, epochs=50):
        x_train = []
        y_train = []
        graphs_mat,_ = self._trans(graph_part_group)
        for graph in graphs_mat:
            x = Feature(graph).feature_nodes()
            x = self._padding(x, MAX_NETWORK_LENGTH)
            x_train.append(x)
            x_train = np.array(x_train)
        for cell_list in cell_list_group:
            cell_list = self._my_param_style(cell_list)
            y = encoder(cell_list)
            y = to_categorical(y, getClassNum())
            y = self._padding(y, MAX_NETWORK_LENGTH)
            y_train.append(y)
            y_train = np.array(y_train)

        self._save_model(model=self.model,
                         json_path='./predict_op/outdated_model.json',
                         weights_path='./predict_op/outdated_model.json.h5')

        self.model.compile(loss='categorical_crossentropy',
                           optimizer='rmsprop',
                           metrics=['accuracy'])
        self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

        self._save_model(model=self.model,
                         json_path=model_json_path,
                         weights_path=model_weights_path)


if __name__ == '__main__':
    graph = [[[1],[2],[3],[4],[5],[]]]
    cell_list = [[('conv', 256, 3, 'relu'), ('conv', 192, 3, 'relu'), ('conv', 512, 1, 'relu'), ('pooling','max',4)
    , ('conv', 128, 1, 'relu'),('conv', 512, 5, 'relu')]]
    pred = Predictor()
    Blocks = []
    pred.train(graph, cell_list)

    # enu = Enumerater(depth=6, width=3)
    # network_pool = enu.enumerate()
    # print(len(network_pool))
    # start = time.time()
    # i = 0
    # pred = Predictor()
    # for ind in range(2, len(network_pool)):
    #     gra = network_pool[ind].graph_part
    #
    #     #Blocks = [network_pool[ind - 2].graph_part, network_pool[ind - 1].graph_part]
    #     Blocks = []
    #     cell_list = pred.predictor(Blocks, gra)
    #     if i%100 == 0:
    #         print("iterator:", i)
    #     i += 1
    #     print(gra)
    #     print(cell_list)
    # end = time.time()
    # print(end-start)