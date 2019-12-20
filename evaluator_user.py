import os
import tensorflow as tf

from base import Cell, NetworkItem
from info_str import NAS_CONFIG
from utils import NAS_LOG


class DataSet:
    # TODO for dataset changing please rewrite this class's "inputs" function and "process" function

    def __init__(self):
        self.data_path = "./data"
        return

    def inputs(self):
        '''
        Method for load data
                Returns:
                  train_data, train_label, valid_data, valid_label, test_data, test_label
        '''
        return train_data, train_label, test_data, test_label

    def process(self, x):
        return x


class Evaluator:
    def __init__(self):
        # don't change the parameters below
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        self.train_num = 0
        self.block_num = 0
        self.log = ''

        # change the value of parameters below
        self.batch_size = 64
        self.input_shape = []
        self.output_shape = []
        self.model_path = "./model"
        self.train_data, self.train_label, self.test_data, self.test_label = DataSet().inputs()
        return


    def set_epoch(self, e):
        self.epoch = e
        return

    def _inference(self, images, graph_part, cell_list, train_flag):
        '''Method for recovering the network model provided by graph_part and cellist.
        Args:
          images: Images returned from Dataset() or inputs().
          graph_part: The topology structure of th network given by adjacency table
          cellist:
        Returns:
          Logits.'''
        topo_order = self._toposort(graph_part)
        nodelen = len(graph_part)
        inputs = [images for _ in range(nodelen)]
        getinput = [False for _ in range(nodelen)]
        getinput[0] = True

        for node in topo_order:
            layer = self._make_layer(inputs[node], cell_list[node], node, train_flag)
            for j in graph_part[node]:
                if getinput[j]:
                    inputs[j] = self._pad(inputs[j], layer)
                else:
                    inputs[j] = layer
                    getinput[j] = True
        # give last layer a name
        last_layer = tf.identity(layer, name="last_layer" + str(self.block_num))
        return last_layer

    def _toposort(self, graph):
        node_len = len(graph)
        in_degrees = dict((u, 0) for u in range(node_len))
        for u in range(node_len):
            for v in graph[u]:
                in_degrees[v] += 1
        queue = [u for u in range(node_len) if in_degrees[u] == 0]
        result = []
        while queue:
            u = queue.pop()
            result.append(u)
            for v in graph[u]:
                in_degrees[v] -= 1
                if in_degrees[v] == 0:
                    queue.append(v)
        return result

    def _make_layer(self, inputs, cell, node, train_flag):
        '''Method for constructing and calculating cell in tensorflow
        Args:
                inputs: the input tensor of this operation
                cell: Class Cell(), hyper parameters for building this layer
                node: int, the index of this operation
                train_flag: boolean, indicating whether this is a training process or not
        Returns:
                layer: tensor.'''
        if cell.type == 'conv':
            layer = self._makeconv(inputs, cell, node, train_flag)
        elif cell.type == 'pooling':
            layer = self._makepool(inputs, cell)
        elif cell.type == 'sep_conv':
            layer = self._makesep_conv(inputs, cell, node, train_flag)
        # TODO add any other new operations here
        #  use the form as shown above
        #  '''elif cell.type == 'operation_name':
        #         layer = self._name_your_function_here(inputs, cell, node)'''
        #  The "_name_your_function_here" is a function take (inputs, cell, node) or any other needed parameter as
        #  input, and output the corresponding tensor calculated use tensorflow, see self._makeconv as an example.
        #  The "inputs" is the input tensor, and "cell" is the hyper parameters for building this layer, given by
        #  class Cell(). The "node" is the index of this layer, mainly for the nomination of the output tensor.
        else:
            assert False, "Wrong cell type!"

        return layer

    def _name_your_function_here(self, inputs, cell, node):
        """
        the operation defined by user,
                Args:
                    inputs: the input tensor of this operation
                    cell: Class Cell(), hyper parameters for building this layer
                    node: int, the index of this operation
                Returns:
                    layer: the output tensor
                """
        # TODO add your function here if any new operation was added, see _makeconv as an example
        return layer

    def _makeconv(self, x, hplist, node, train_flag):
        """Generates a convolutional layer according to information in hplist
        Args:
            x: inputing data.
            hplist: hyperparameters for building this layer
            node: int, the index of this operation
        Returns:
            conv_layer: the output tensor
        """
        with tf.variable_scope('conv' + str(node) + 'block' + str(self.block_num)) as scope:
            inputdim = x.shape[3]
            kernel = self._get_variable('weights',
                                        shape=[hplist.kernel_size, hplist.kernel_size, inputdim, hplist.filter_size])
            x = self._activation_layer(hplist.activation, x, scope)
            x = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], padding='SAME')
            biases = self._get_variable('biases', hplist.filter_size)
            x = self._batch_norm(tf.nn.bias_add(x, biases), train_flag)
        return x

    def _makesep_conv(self, inputs, hplist, node, train_flag):
        with tf.variable_scope('conv' + str(node) + 'block' + str(self.block_num)) as scope:
            inputdim = inputs.shape[3]
            dfilter = self._get_variable('weights', shape=[hplist.kernel_size, hplist.kernel_size, inputdim, 1])
            pfilter = self._get_variable('pointwise_filter', [1, 1, inputdim, hplist.filter_size])
            conv = tf.nn.separable_conv2d(inputs, dfilter, pfilter, strides=[1, 1, 1, 1], padding='SAME')
            biases = self._get_variable('biases', hplist.filter_size)
            bn = self._batch_norm(tf.nn.bias_add(conv, biases), train_flag)
            conv_layer = self._activation_layer(hplist.activation, bn, scope)
        return conv_layer

    def _batch_norm(self, input, train_flag):
        return tf.contrib.layers.batch_norm(input, decay=0.9, center=True, scale=True, epsilon=1e-3,
                                            updates_collections=None, is_training=train_flag)

    def _get_variable(self, name, shape):
        if name == "weights":
            ini = tf.contrib.keras.initializers.he_normal()
        else:
            ini = tf.constant_initializer(0.0)
        return tf.get_variable(name, shape, initializer=ini)

    def _activation_layer(self, type, inputs, scope):
        if type == 'relu':
            layer = tf.nn.relu(inputs, name=scope.name)
        elif type == 'relu6':
            layer = tf.nn.relu6(inputs, name=scope.name)
        elif type == 'tanh':
            layer = tf.tanh(inputs, name=scope.name)
        elif type == 'sigmoid':
            layer = tf.sigmoid(inputs, name=scope.name)
        elif type == 'leakyrelu':
            layer = tf.nn.leaky_relu(inputs, name=scope.name)
        else:
            layer = tf.identity(inputs, name=scope.name)

        return layer

    def _makepool(self, inputs, hplist):
        """Generates a pooling layer according to information in hplist
        Args:
            inputs: inputing data.
            hplist: hyperparameters for building this layer
        Returns:
            tensor.
        """
        if hplist.ptype == 'avg':
            return tf.nn.avg_pool(inputs, ksize=[1, hplist.kernel_size, hplist.kernel_size, 1],
                                  strides=[1, hplist.kernel_size, hplist.kernel_size, 1], padding='SAME')
        elif hplist.ptype == 'max':
            return tf.nn.max_pool(inputs, ksize=[1, hplist.kernel_size, hplist.kernel_size, 1],
                                  strides=[1, hplist.kernel_size, hplist.kernel_size, 1], padding='SAME')
        elif hplist.ptype == 'global':
            return tf.reduce_mean(inputs, [1, 2], keep_dims=True)

    def _makedense(self, inputs, hplist):
        """Generates dense layers according to information in hplist
        Args:
                   inputs: inputing data.
                   hplist: hyperparameters for building layers
                   node: number of this cell
        Returns:
                   tensor.
        """
        inputs = tf.reshape(inputs, [self.batch_size, -1])

        for i, neural_num in enumerate(hplist[1]):
            with tf.variable_scope('dense' + str(i) + 'block' + str(self.block_num)) as scope:
                weights = self._get_variable('weights', shape=[inputs.shape[-1], neural_num])
                biases = self._get_variable('biases', [neural_num])
                mul = tf.matmul(inputs, weights) + biases
                # TODO ????
                if neural_num == DataSet().NUM_CLASSES:
                    local3 = self._activation_layer('', mul, scope)
                else:
                    local3 = self._activation_layer(hplist[2], mul, scope)
            inputs = local3
        return inputs

    def _pad(self, inputs, layer):
        # padding
        a = int(layer.shape[1])
        b = int(inputs.shape[1])
        pad = abs(a - b)
        if layer.shape[1] > inputs.shape[1]:
            tmp = tf.pad(inputs, [[0, 0], [0, pad], [0, pad], [0, 0]])
            inputs = tf.concat([tmp, layer], 3)
        elif layer.shape[1] < inputs.shape[1]:
            tmp = tf.pad(layer, [[0, 0], [0, pad], [0, pad], [0, 0]])
            inputs = tf.concat([inputs, tmp], 3)
        else:
            inputs = tf.concat([inputs, layer], 3)

        return inputs

    def evaluate(self, network, pre_block=[], is_bestNN=False, update_pre_weight=False):
        '''Method for evaluate the given network.
        Args:
            network: NetworkItem()
            pre_block: The pre-block structure, every block has two parts: graph_part and cell_list of this block.
            is_bestNN: Symbol for indicating whether the evaluating network is the best network of this round, default False.
            update_pre_weight: Symbol for indicating whether to update previous blocks' weight, default by False.
        Returns:
            Accuracy'''
        assert self.train_num >= self.batch_size
        tf.reset_default_graph()
        self.block_num = len(pre_block)

        self.log = "-" * 20 + str(network.id) + "-" * 20 + '\n'
        for block in pre_block:
            self.log = self.log + str(block.graph) + str(block.cell_list) + '\n'
        self.log = self.log + str(network.graph) + str(network.cell_list) + '\n'

        with tf.Session() as sess:
            data_x, data_y, block_input, train_flag = self._get_input(sess, pre_block, update_pre_weight)

            graph_full, cell_list = self._recode(network.graph, network.cell_list,
                                                 NAS_CONFIG['nas_main']['repeat_search'])
            # a pooling layer for last repeat block
            graph_full = graph_full + [[]]
            cell_list = cell_list + [Cell('pooling', 'max', 2)]
            logits = self._inference(block_input, graph_full, cell_list, train_flag)

            logits = tf.nn.dropout(logits, keep_prob=1.0)
            # TODO dense layer?
            logits = self._makedense(logits, ('', [self.output_shape[-1]], ''))

            precision, saver, log = self._eval(sess, data_x, data_y, logits, train_flag)
            self.log += log

            if is_bestNN:  # save model
                saver.save(sess, os.path.join(
                    self.model_path, 'model' + str(network.id)))

        NAS_LOG << ('eva', self.log)
        return precision

    def _get_input(self, sess, pre_block, update_pre_weight=False):
        '''Get input for _inference'''
        # if it got previous blocks
        if len(pre_block) > 0:
            new_saver = tf.train.import_meta_graph(
                os.path.join(self.model_path, 'model' + str(pre_block[-1].id) + '.meta'))
            new_saver.restore(sess, os.path.join(
                self.model_path, 'model' + str(pre_block[-1].id)))
            graph = tf.get_default_graph()
            x = graph.get_tensor_by_name("input:0")
            labels = graph.get_tensor_by_name("label:0")
            train_flag = graph.get_tensor_by_name("train_flag:0")
            input = graph.get_tensor_by_name("last_layer" + str(self.block_num - 1) + ":0")
            # only when there's not so many network in the pool will we update the previous blocks' weight
            if not update_pre_weight:
                input = tf.stop_gradient(input, name="stop_gradient")
        # if it's the first block
        else:
            x = tf.placeholder(tf.float32, self.input_shape, name='input')
            labels = tf.placeholder(tf.int32, self.output_shape, name="label")
            train_flag = tf.placeholder(tf.bool, name='train_flag')
            input = tf.identity(x)
        return x, labels, input, train_flag

    def _recode(self, graph_full, cell_list, repeat_num):
        new_graph = [] + graph_full
        new_cell_list = [] + cell_list
        add = 0
        for i in range(repeat_num - 1):
            new_cell_list += cell_list
            add += len(graph_full)
            for sub_list in graph_full:
                new_graph.append([x + add for x in sub_list])
        return new_graph, new_cell_list

    def _eval(self, sess, logits, data_x, data_y, *args, **kwargs):
        # TODO change here to run training step and evaluation step
        """
        The actual training process, including the definination of loss and train optimizer
        Args:
            sess: tensorflow session
            logits: output tensor of the model, 2-D tensor of shape [self.batch_size, self.NUM_CLASS]
            data_x: input image
            data_y: input label, 2-D tensor of shape [self.batch_size, self.NUM_CLASS]
        Returns:
            targets: float, the optimization target, could be the accuracy or the combination of both time and accuracy, etc
            saver: Tensorflow Saver class
            log: string, log to be write and saved
        """
        global_step = tf.Variable(0, trainable=False, name='global_step' + str(self.block_num))
        accuracy = self._cal_accuracy(logits, data_y)
        loss = self._loss(logits, data_y)
        train_op = self._train_op(global_step, loss)

        saver = tf.train.Saver(tf.global_variables())
        sess.run(tf.global_variables_initializer())

        log = ""

        return target, saver, log

    def _cal_accuracy(self, logits, labels):
        """
        calculate the target of this task
            Args:
                logits: Logits from softmax.
                labels: Labels from distorted_inputs or inputs(). 2-D tensor of shape [self.batch_size, self.NUM_CLASS]
            Returns:
                Target tensor of type float.
        """
        # TODO change here for the way of calculating target
        return accuracy

    def _loss(self, logits, labels):
        """
          Args:
            logits: Logits from softmax.
            labels: Labels from distorted_inputs or inputs(). 2-D tensor of shape [self.batch_size, self.NUM_CLASS]
          Returns:
            Loss tensor of type float.
          """
        # TODO change here for the way of calculating loss
        return loss

    def _train_op(self, global_step, loss):
        # TODO change here for learning rate and optimizer
        return train_op

    def _cal_multi_target(self, precision, time):
        # TODO change here for target calculating
        return target

    def set_data_size(self, num):
        if num > len(list(self.train_label)) or num < 0:
            num = len(list(self.train_label))
            print('Warning! Data size has been changed to', num, ', all data is loaded.')
        self.train_num = num
        return


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    eval = Evaluator()
    eval.set_data_size(50000)
    eval.set_epoch(10)

    graph_full = [[1, 3], [2, 3], [3], [4]]
    cell_list = [Cell('conv', 24, 3, 'relu'), Cell('conv', 32, 3, 'relu'), Cell('conv', 24, 3, 'relu'),
                 Cell('conv', 32, 3, 'relu')]
    network1 = NetworkItem(0, graph_full, cell_list, "")
    network2 = NetworkItem(1, graph_full, cell_list, "")
    e = eval.evaluate(network1, is_bestNN=True)
    eval.set_data_size(500)
    e = eval.evaluate(network2, [network1], is_bestNN=True)
