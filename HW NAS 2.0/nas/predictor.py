import numpy as np
from enumerater import Enumerater
from predict_op.label_encoding import decoder
from keras.models import model_from_json
MAX_NETWORK_LENGTH = 71
model_json_path = './predict_op/model.json'
model_weights_path = './predict_op/model.json.h5'


class Feature:
    def __init__(self, graph):
        self.graph = graph

    # 从邻接矩阵中提取所有的支链，每一条支链有五个特征，编号，起点，终点，长度，节点编号
    def feature_links(self):
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
                        link = self.find_link(endpoint, j, link, g, node_link_num_set)
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

    # 对每一个节点提取特征
    def feature_nodes(self):
        link_set, endpointLinkNumSet, nodeLinkNumSet = self.feature_links()

        node_num = len(self.graph)                  # 单个网络结构的节点个数
        feature_num = 25                            # 手动提取的特征数
        node_feature = np.zeros((node_num, feature_num), dtype=float)

        # 全局特征，节点数，链数
        node_feature[:, 0] = node_num
        node_feature[:, 1] = len(link_set)

        # 全局特征，最长支链长度和编号，最短支链长度和编号
        max_length, max_link_index = self.find_max_link(link_set)
        min_length, min_link_index = self.find_min_link(link_set)
        node_feature[:, 2] = max_length
        node_feature[:, 3] = max_link_index
        node_feature[:, 4] = min_length
        node_feature[:, 5] = min_link_index

        # 全局特征，期望和方差
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

            node_feature[i][global_num] = self.isendpoint(i, endpointLinkNumSet)
            node_feature[i][global_num+1] = i

            if node_feature[i][global_num] == 1:
                node_feature[i][global_num+2] = self.link_num(i, endpointLinkNumSet)
                links = self.findendpointLinkSet(i,endpointLinkNumSet, link_set)
                node_feature[i][global_num+3] = self.mean_link(links)
                node_feature[i][global_num+4] = self.var_link(links)
                _, max_length = self.find_max_link(links)
                _, min_length = self.find_min_link(links)
                node_feature[i][global_num+5] = max_length
                node_feature[i][global_num+6] = min_length

            else:

                link = self.findnodelink(i, nodeLinkNumSet, link_set)
                node_feature[i][global_num+7] = self.relativeLoc(i, link)
                node_feature[i][global_num+8] = link[3]
                node_feature[i][global_num+9] = link[1]
                node_feature[i][global_num+10] = link[2]

                links = self.findnodelinks(i, nodeLinkNumSet, link_set)
                node_feature[i][global_num+11] = self.mean_link(links)
                node_feature[i][global_num+12] = self.var_link(links)

        return node_feature

    # 寻找端点的支链集
    def findendpointLinkSet(self, id, endpoint_link_num_set, link_set):
        links = []
        for i in range(len(endpoint_link_num_set)):
            if endpoint_link_num_set[i][0] == id:
                for e in range(1, len(endpoint_link_num_set[i])):
                    links.append(link_set[endpoint_link_num_set[i][e]])
                break
        return links

    # 寻找与节点所在支链有相同端点的支链集
    def findnodelinks(self, id, node_link_num_set, link_set):
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

    # 寻找节点所在的支链
    def findnodelink(self, id, node_link_num_set, link_set):
        for i in range(len(node_link_num_set)):
            if node_link_num_set[i][0] == id:
                link_num = link_set[node_link_num_set[i][1]]
                return link_num

    # 判断是否为端点
    def isendpoint(self, node_num, endpoint_link_num_set):
        for i in range(len(endpoint_link_num_set)):
            if endpoint_link_num_set[i][0] == node_num:
                return 1
        return 0

    # 支链的个数
    def link_num(self, node_id, endpoint_link_num_set):
        for i in range(len(endpoint_link_num_set)):
            if endpoint_link_num_set[i][0] == node_id:
                e = endpoint_link_num_set[i]
                return len(e)-1
        return 0

    # 支链长度的期望
    def mean_link(self, link_set):
        links_len = []
        for e in link_set:
            links_len.append(e[3])
        return np.mean(links_len)

    # 支链的方差
    def var_link(self, link_set):
        links_len = []
        for e in link_set:
            links_len.append(e[3])
        return np.var(links_len)

    # 节点在支链中的相对位置
    def relativeLoc(self, id, link):
        for i in range(len(link[4])):
            if link[4][i] == id:
                return i+1

    # 寻找最长支链
    def find_max_link(self, link_set):
        max_length = 0
        index = 0
        for i in range(len(link_set)):
            if link_set[i][3] > max_length:
                index = i
                max_length = link_set[i][3]
        return index, max_length

    # 寻找最短支链
    def find_min_link(self, link_set):
        min_length = 0
        index = 0
        for i in range(len(link_set)):
            if link_set[i][3] < min_length:
                index = i
                min_length = link_set[i][3]
        return index, min_length

    # 递归搜索链上的所有节点
    def find_link(self, endpoint, id, link, G, node_link_num_set):
        if endpoint[id] == 1:
            link[2] = id
            return link
        else:
            link[3] += 1
            link[4].append(id)
            node_link_num_set.append([id, link[0]])
            for i in range(len(G)):
                if G[id][i] == 1:
                    link = self.find_link(endpoint, i, link, G, node_link_num_set)
                    break
        return link


# 将领接表转换成邻接矩阵
def list2mat(G):
    graph = np.zeros((len(G), len(G)), dtype=int)
    for i in range(len(G)):
        e = G[i]
        if e:
            for k in e:
                graph[i][k] = 1
    return graph


# concat the all graph
def graph_concat(graphs):
    if len(graphs) == 1:
        return graphs[0]
    elif len(graphs) > 1:
        new_graph_length = 0
        for g in graphs:
            new_graph_length += len(g)
        new_graph = np.zeros((new_graph_length, new_graph_length),dtype=int)
        x_index = 0             # the staring connection position of next graph
        y_index = 0
        for g in graphs:
            new_graph[x_index:x_index+len(g), y_index:y_index+len(g)] = g
            if y_index + len(g) < new_graph_length:
                new_graph[x_index+len(g)-1][y_index+len(g)] = 1
            x_index = x_index+len(g)
            y_index = y_index+len(g)

        return new_graph


# 获得节点在新的编码方式下的顺序
def get_new_order(links, graph_size):
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


# 获得在新的编码方式下网络结构的邻接矩阵
def get_new_mat(new_order, mat):
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


# 对输入数据做填充，保证输入数据的一致性
def padding(node_feature, length):
    if len(node_feature) < length:
        add = np.ones((length-len(node_feature), len(node_feature[0])))
        add = -add
        node_feature = np.append(node_feature,add,axis=0)

    return node_feature


# 加载训练好的模型
def load_model(model_json, weights):
    with open(model_json, 'r') as file:
        model_json1 = file.read()
    new_model = model_from_json(model_json1)
    new_model.load_weights(weights)
    return new_model


Model = load_model(model_json_path,model_weights_path)


# 根据输入特征预测操作
def predict(inputs):
    inputs = np.array(inputs)
    inputs = np.reshape(inputs,(1, inputs.shape[0], inputs.shape[1]))
    #model = load_model(model_json_path, model_weights_path)
    predict_y = Model.predict(inputs)
    predict_y = np.reshape(predict_y, (predict_y.shape[1], predict_y.shape[2]))
    output = []
    for i in range(len(inputs[0])):
        output.append(np.argmax(predict_y[i]))
    return output


class Predictor:
    def __init__(self):
        self.graph = []
        self.cell_list = []

    # 对输入的邻接表重新编码并转换成矩阵的形式提取特征
    def _trans(self, graphs):
        graphs_mat = []
        graphs_orders = []
        for g in graphs:
            g_mat = list2mat(g)
            links, _, _ = Feature(g_mat).feature_links()
            order = get_new_order(links, len(g_mat))
            graph_mat = get_new_mat(order, g)
            graphs_mat.append(graph_mat)
            graphs_orders.append(order)
        return graphs_mat, graphs_orders

    # 将最后输出的类别转换成需要预测的操作详细参数
    def class_id_2_parameter(self, order, class_list):
        parameters = decoder(class_list)
        parameters_cp = parameters.copy()
        for i in range(len(order)):
            parameters[order[i]] = parameters_cp[i]
        return parameters[:len(order)]

    # 模块接口
    def predictor(self, blocks, graph_part):
        graph_list = []
        if blocks:
            for block in blocks:
                graph_list.append(block.graph_part)
        graph_list.append(graph_part)
        graphs_mat, graphs_orders = self._trans(graph_list)
        new_graph = graph_concat(graphs_mat)
        inputs = Feature(new_graph).feature_nodes()
        inputs = padding(inputs,MAX_NETWORK_LENGTH)
        class_list = predict(inputs)
        self.cell_list = self.class_id_2_parameter(graphs_orders[-1], class_list[len(new_graph)-len(graph_part):len(new_graph)])
        return self.cell_list


if __name__ == '__main__':
    gra = [[1, 10], [2, 14], [3], [4], [5], [6], [7], [8], [9], [], [11], [12], [13], [6], [7]]
    enu = Enumerater(depth=6,width=3)
    network_pool = enu.enumerate()
    for ind in range(2, len(network_pool)):
        gra = network_pool[ind].graph_part
        pred = Predictor()
        Blocks = [network_pool[ind - 2], network_pool[ind - 1]]
        cell_list = pred.predictor(Blocks, gra)
        print(cell_list)
