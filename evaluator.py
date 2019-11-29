import os
import pickle
import random
import sys
import time

import numpy as np
import tensorflow as tf

from base import Cell, NetworkItem
from info_str import NAS_CONFIG
from utils import NAS_LOG


class DataSet:

    def __init__(self):
        self.IMAGE_SIZE = 32
        self.NUM_CLASSES = NAS_CONFIG['eva']['num_classes']
        self.NUM_EXAMPLES_FOR_TRAIN = NAS_CONFIG['eva']['num_examples_for_train']
        self.NUM_EXAMPLES_FOR_EVAL = NAS_CONFIG['eva']['num_examples_per_epoch_for_eval']
        self.task = NAS_CONFIG['eva']['task_name']
        self.data_path = NAS_CONFIG['eva']['dataset_path']
        return

    def inputs(self):
        print("======Loading data======")
        if self.task == 'cifar-10':
            test_files = ['test_batch']
            train_files = ['data_batch_%d' % d for d in range(1, 6)]
        else:
            train_files = ['train']
            test_files = ['test']
        train_data, train_label = self._load(train_files)
        train_data, train_label, valid_data, valid_label = self._split(
            train_data, train_label)
        test_data, test_label = self._load(test_files)
        print("======Data Process Done======")
        return train_data, train_label, valid_data, valid_label, test_data, test_label

    def _load_one(self, file):
        with open(file, 'rb') as fo:
            batch = pickle.load(fo, encoding='bytes')
        data = batch[b'data']
        label = batch[b'labels'] if self.task == 'cifar-10' else batch[b'fine_labels']
        return data, label

    def _load(self, files):
        file_name = 'cifar-10-batches-py' if self.task == 'cifar-10' else 'cifar-100-python'
        data_dir = os.path.join(self.data_path, file_name)
        data, label = self._load_one(os.path.join(data_dir, files[0]))
        for f in files[1:]:
            batch_data, batch_label = self._load_one(os.path.join(data_dir, f))
            data = np.append(data, batch_data, axis=0)
            label = np.append(label, batch_label, axis=0)
        label = np.array([[float(i == label)
                           for i in range(self.NUM_CLASSES)] for label in label])
        data = data.reshape([-1, 3, self.IMAGE_SIZE, self.IMAGE_SIZE])
        data = data.transpose([0, 2, 3, 1])
        # pre-process
        data = self._normalize(data)

        return data, label

    def _split(self, data, label):
        # shuffle
        index = [i for i in range(len(data))]
        random.shuffle(index)
        data = data[index]
        label = label[index]
        return data[:self.NUM_EXAMPLES_FOR_TRAIN], label[:self.NUM_EXAMPLES_FOR_TRAIN], \
               data[self.NUM_EXAMPLES_FOR_TRAIN:self.NUM_EXAMPLES_FOR_TRAIN + self.NUM_EXAMPLES_FOR_EVAL], \
               label[self.NUM_EXAMPLES_FOR_TRAIN:self.NUM_EXAMPLES_FOR_TRAIN +
                                                 self.NUM_EXAMPLES_FOR_EVAL]

    def _normalize(self, x_train):
        x_train = x_train.astype('float32')

        x_train[:, :, :, 0] = (
                                      x_train[:, :, :, 0] - np.mean(x_train[:, :, :, 0])) / np.std(x_train[:, :, :, 0])
        x_train[:, :, :, 1] = (
                                      x_train[:, :, :, 1] - np.mean(x_train[:, :, :, 1])) / np.std(x_train[:, :, :, 1])
        x_train[:, :, :, 2] = (
                                      x_train[:, :, :, 2] - np.mean(x_train[:, :, :, 2])) / np.std(x_train[:, :, :, 2])

        return x_train

    def process(self, x):
        x = self._random_flip_leftright(x)
        x = self._random_crop(x, [32, 32], 4)
        x = self._cutout(x)
        return x

    def _random_crop(self, batch, crop_shape, padding=None):
        oshape = np.shape(batch[0])
        if padding:
            oshape = (oshape[0] + 2 * padding, oshape[1] + 2 * padding)
        new_batch = []
        npad = ((padding, padding), (padding, padding), (0, 0))
        for i in range(len(batch)):
            new_batch.append(batch[i])
            if padding:
                new_batch[i] = np.lib.pad(batch[i], pad_width=npad,
                                          mode='constant', constant_values=0)
            nh = random.randint(0, oshape[0] - crop_shape[0])
            nw = random.randint(0, oshape[1] - crop_shape[1])
            new_batch[i] = new_batch[i][nh:nh + crop_shape[0],
                           nw:nw + crop_shape[1]]
        return np.array(new_batch)

    def _random_flip_leftright(self, batch):
        for i in range(len(batch)):
            if bool(random.getrandbits(1)):
                batch[i] = np.fliplr(batch[i])
        return batch

    def _cutout(self, x):
        for i in range(len(x)):
            cut_size = random.randint(0, self.IMAGE_SIZE // 2)
            s = random.randint(0, self.IMAGE_SIZE - cut_size)
            x[i, s:s + cut_size, s:s + cut_size, :] = 0
        return x


class Evaluator:
    def __init__(self):
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        # Global constants describing the CIFAR-10 data set.
        self.IMAGE_SIZE = 32
        self.NUM_CLASSES = NAS_CONFIG['eva']['num_classes']
        self.NUM_EXAMPLES_FOR_TRAIN = NAS_CONFIG['eva']['num_examples_for_train']
        self.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = NAS_CONFIG['eva']['num_examples_per_epoch_for_eval']
        # Constants describing the training process.
        # Initial learning rate.
        self.INITIAL_LEARNING_RATE = NAS_CONFIG['eva']['initial_learning_rate']
        # Epochs after which learning rate decays
        self.NUM_EPOCHS_PER_DECAY = NAS_CONFIG['eva']['num_epochs_per_decay']
        # Learning rate decay factor.
        self.LEARNING_RATE_DECAY_FACTOR = NAS_CONFIG['eva']['learning_rate_decay_factor']
        self.MOVING_AVERAGE_DECAY = NAS_CONFIG['eva']['moving_average_decay']
        self.batch_size = NAS_CONFIG['eva']['batch_size']
        self.weight_decay = NAS_CONFIG['eva']['weight_decay']
        self.momentum_rate = NAS_CONFIG['eva']['momentum_rate']
        self.model_path = NAS_CONFIG['eva']['model_path']
        self.train_num = 0
        self.max_steps = 0
        self.block_num = 0
        self.log = ''
        self.train_data, self.train_label, self.valid_data, self.valid_label, \
        self.test_data, self.test_label = DataSet().inputs()

    def set_epoch(self, e):
        self.epoch = e
        return

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

    def _batch_norm(self, input, train_flag):
        return tf.contrib.layers.batch_norm(input, decay=0.9, center=True, scale=True, epsilon=1e-3,
                                            updates_collections=None, is_training=train_flag)

    def _makeconv(self, inputs, hplist, node, train_flag, sep=False):
        """Generates a convolutional layer according to information in hplist
        Args:
        inputs: inputing data.
        hplist: hyperparameters for building this layer
        node: number of this cell
        Returns:
        tensor.
        """
        # print('Evaluater:right now we are making conv layer, its node is %d, and the inputs is'%node,inputs,'and the node before it is ',cellist[node-1])
        with tf.variable_scope('conv' + str(node) + 'block' + str(self.block_num)) as scope:
            inputdim = inputs.shape[3]
            if sep:
                kernel = tf.get_variable('weights', shape=[hplist.kernel_size, hplist.kernel_size, inputdim, 1],
                                         initializer=tf.contrib.keras.initializers.he_normal())
                pfilter = tf.get_variable(
                    'pointwise_filter', [1, 1, inputdim, hplist.filter_size])
                conv = tf.nn.separable_conv2d(inputs, kernel, pfilter, strides=[
                    1, 1, 1, 1], padding='SAME')
            else:
                kernel = tf.get_variable('weights',
                                         shape=[
                                             hplist.kernel_size, hplist.kernel_size, inputdim, hplist.filter_size],
                                         initializer=tf.contrib.keras.initializers.he_normal())
                conv = tf.nn.conv2d(
                    inputs, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(
                'biases', hplist.filter_size, initializer=tf.constant_initializer(0.0))
            bias = self._batch_norm(tf.nn.bias_add(conv, biases), train_flag)
            if hplist.activation == 'relu':
                conv1 = tf.nn.relu(bias, name=scope.name)
            elif hplist.activation == 'relu6':
                conv1 = tf.nn.relu6(bias, name=scope.name)
            elif hplist.activation == 'tanh':
                conv1 = tf.tanh(bias, name=scope.name)
            elif hplist.activation == 'sigmoid':
                conv1 = tf.sigmoid(bias, name=scope.name)
            elif hplist.activation == 'leakyrelu':
                conv1 = tf.nn.leaky_relu(bias, name=scope.name)
            else:
                conv1 = tf.identity(bias, name=scope.name)

        return conv1

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

    def _makedense(self, inputs, hplist, train_flag):
        """Generates dense layers according to information in hplist
        Args:
                   inputs: inputing data.
                   hplist: hyperparameters for building layers
                   node: number of this cell
        Returns:
                   tensor.
        """
        i = 0
        inputs = tf.reshape(inputs, [self.batch_size, -1])

        for i, neural_num in enumerate(hplist[1]):
            with tf.variable_scope('dense' + str(i) + 'block' + str(self.block_num)) as scope:
                weights = tf.get_variable('weights', shape=[inputs.shape[-1], neural_num],
                                          initializer=tf.contrib.keras.initializers.he_normal())
                biases = tf.get_variable(
                    'biases', [neural_num], initializer=tf.constant_initializer(0.0))
                if neural_num == self.NUM_CLASSES:
                    local3 = tf.identity(
                        tf.matmul(inputs, weights) + biases, name=scope.name)
                else:
                    if hplist[2] == 'relu':
                        local3 = tf.nn.relu(self._batch_norm(tf.matmul(inputs, weights) + biases, train_flag),
                                            name=scope.name)
                    elif hplist[2] == 'tanh':
                        local3 = tf.tanh(
                            tf.matmul(inputs, weights) + biases, name=scope.name)
                    elif hplist[2] == 'sigmoid':
                        local3 = tf.sigmoid(
                            tf.matmul(inputs, weights) + biases, name=scope.name)
                    else:
                        local3 = tf.identity(
                            tf.matmul(inputs, weights) + biases, name=scope.name)
            inputs = local3
        return inputs

    def _inference(self, images, graph_part, cellist, train_flag):
        '''Method for recovering the network model provided by graph_part and cellist.
        Args:
          images: Images returned from Dataset() or inputs().
          graph_part: The topology structure of th network given by adjacency table
          cellist:
        Returns:
          Logits.'''
        # print('Evaluater:starting to reconstruct the network')
        topo_order = self._toposort(graph_part)
        nodelen = len(graph_part)
        # input list for every cell in network
        inputs = [images for _ in range(nodelen)]
        # bool list for whether this cell has already got input or not
        getinput = [False for _ in range(nodelen)]
        getinput[0] = True

        for node in topo_order:
            # print('Evaluater:right now we are processing node %d'%node,', ',cellist[node])
            if cellist[node].type == 'conv':
                layer = self._makeconv(
                    inputs[node], cellist[node], node, train_flag)
            elif cellist[node].type == 'pooling':
                layer = self._makepool(inputs[node], cellist[node])
            elif cellist[node].type == 'sep_conv':
                layer = self._makeconv(
                    inputs[node], cellist[node], node, train_flag, sep=True)

            # update inputs information of the cells below this cell
            for j in graph_part[node]:
                if getinput[j]:  # if this cell already got inputs from other cells precedes it
                    inputs[j] = self._pad(inputs[j], layer)
                else:
                    inputs[j] = layer
                    getinput[j] = True

        # give last layer a name
        last_layer = tf.identity(
            layer, name="last_layer" + str(self.block_num))
        return last_layer

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

    def _loss(self, labels, logits):
        """
          Args:
            logits: Logits from softmax.
            labels: Labels from distorted_inputs or inputs(). 1-D tensor of shape [self.batch_size]
          Returns:
            Loss tensor of type float.
          """
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
        loss = cross_entropy + l2 * self.weight_decay
        return loss, cross_entropy

    def _train_op(self, global_step, loss):
        # Variables that affect learning rate.
        lr_type = NAS_CONFIG['eva']['learning_rate_type']
        num_batches_per_epoch = self.train_num / self.batch_size
        decay_steps = int(num_batches_per_epoch * self.NUM_EPOCHS_PER_DECAY)

        if lr_type == 'const':
            lr = tf.train.piecewise_constant(global_step, boundaries=NAS_CONFIG['eva']['boundaries'],
                                             values=NAS_CONFIG['eva']['learing_rate'])
        elif lr_type == 'cos':
            lr = tf.train.cosine_decay(
                self.INITIAL_LEARNING_RATE, global_step, decay_steps)
        else:
            # Decay the learning rate exponentially based on the number of steps.
            lr = tf.train.exponential_decay(self.INITIAL_LEARNING_RATE,
                                            global_step,
                                            decay_steps,
                                            self.LEARNING_RATE_DECAY_FACTOR,
                                            staircase=True, )

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.

        opt = tf.train.MomentumOptimizer(lr, self.momentum_rate, name='Momentum' + str(self.block_num),
                                         use_nesterov=True)
        grad = opt.compute_gradients(loss)
        train_op = opt.minimize(loss, global_step=global_step)
        return train_op, grad

    def evaluate(self, network, pre_block=[], is_bestNN=False, update_pre_weight=False):
        '''Method for evaluate the given network.
        Args:
            network: NetworkItem()
            pre_block: The pre-block structure, every block has two parts: graph_part and cell_list of this block.
            is_bestNN: Symbol for indicating whether the evaluating network is the best network of this round, default False.
            update_pre_weight: Symbol for indicating whether to update previous blocks' weight, default by False.
        Returns:
            Accuracy'''
        self.log = "-" * 20 + str(network.id) + "-" * 20 + '\n'
        tf.reset_default_graph()
        # print("-" * 20, network.id, "-" * 20)
        # print(network.graph, network.cell_list, Network.pre_block)
        for block in pre_block:
            self.log = self.log + str(block.graph) + str(block.cell_list)
        self.log = self.log + str(network.graph) + str(network.cell_list) + '\n'
        network.graph.append([])
        assert self.train_num >= self.batch_size, "Wrong! The data added in train dataset is smaller than batch size!"
        self.block_num = len(pre_block) * NAS_CONFIG['eva']['repeat_search']
        # a pooling later for every block
        if self.block_num == NAS_CONFIG['nas_main']['block_num']:
            network.cell_list.append(Cell('pooling', 'global', 1))
        else:
            network.cell_list.append(Cell('pooling', 'max', 2))

        with tf.Session() as sess:
            global_step = tf.Variable(
                0, trainable=False, name='global_step' + str(self.block_num))
            x, labels, input, train_flag = self._get_input(
                sess, pre_block, update_pre_weight)

            logits = self._inference(
                input, network.graph, network.cell_list, train_flag)
            for _ in range(NAS_CONFIG['eva']['repeat_search'] - 1):
                self.block_num += 1
                logits = self._inference(
                    logits, network.graph, network.cell_list, train_flag)

            logits = tf.nn.dropout(logits, keep_prob=1.0)
            # softmax
            logits = self._makedense(

                logits, ('', [256, self.NUM_CLASSES], 'relu'), train_flag)

            correct_prediction = tf.equal(
                tf.argmax(logits, 1), tf.argmax(labels, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            loss, cross_entropy = self._loss(labels, logits)
            train_op, lr = self._train_op(global_step, loss)
            # Create a saver.
            saver = tf.train.Saver(tf.global_variables())
            # Start running operations on the Graph.

            sess.run(tf.global_variables_initializer())

            precision = self._eval(
                sess, train_op, cross_entropy, accuracy, x, labels, train_flag, lr)

            if is_bestNN:  # save model
                saver.save(sess, os.path.join(
                    self.model_path, 'model' + str(network.id)))

        network.graph.pop()
        network.cell_list.pop()
        return float(precision[-1])

    def _get_input(self, sess, pre_block, update_pre_weight=False):
        '''Get input for _inference'''
        # if it got previous blocks
        if len(pre_block) > 0:
            # TODO check whether there is a model file exit
            new_saver = tf.train.import_meta_graph(
                os.path.join(self.model_path, 'model' + str(pre_block[-1].id) + '.meta'))
            new_saver.restore(sess, os.path.join(
                self.model_path, 'model' + str(pre_block[-1].id)))
            graph = tf.get_default_graph()
            x = graph.get_tensor_by_name("input:0")
            labels = graph.get_tensor_by_name("label:0")
            train_flag = graph.get_tensor_by_name("train_flag:0")
            input = graph.get_tensor_by_name(
                "last_layer" + str(self.block_num - 1) + ":0")
            # only when there's not so many network in the pool will we update the previous blocks' weight
            if not update_pre_weight:
                input = tf.stop_gradient(input, name="stop_gradient")
        # if it's the first block
        else:
            x = tf.placeholder(
                tf.float32, [self.batch_size, self.IMAGE_SIZE, self.IMAGE_SIZE, 3], name='input')
            labels = tf.placeholder(
                tf.int32, [self.batch_size, self.NUM_CLASSES], name="label")
            train_flag = tf.placeholder(tf.bool, name='train_flag')
            input = tf.identity(x)
        return x, labels, input, train_flag

    def retrain(self, pre_block):
        tf.reset_default_graph()
        retrain_log = "-" * 20 + "retrain" + "-" * 20 + '\n'
        for block in pre_block:
            retrain_log = retrain_log + str(block.graph) + str(block.cell_list) + '\n'
        assert self.train_num >= self.batch_size
        self.block_num = len(pre_block) * NAS_CONFIG['eva']['repeat_search'] + 1
        with tf.Session() as sess:
            global_step = tf.Variable(
                0, trainable=False, name='global_step' + str(self.block_num))
            x, labels, logits, train_flag = self._get_input(sess, [])
            for i, block in enumerate(pre_block):
                graph = block.graph + [[]]
                cell_list = []
                for cell in block.cell_list:
                    if cell.type == 'conv':
                        cell_list.append(Cell(cell.type, cell.filter_size * 2, cell.kernel_size, cell.activation))
                    else:
                        cell_list.append(cell)
                if i == NAS_CONFIG['nas_main']['block_num']:
                    cell_list.append(Cell('pooling', 'global', 1))
                else:
                    cell_list.append(Cell('pooling', 'max', 2))
                logits = self._inference(logits, graph, cell_list, train_flag)
                self.block_num += 1
            logits = tf.nn.dropout(logits, keep_prob=1.0)
            # softmax
            logits = self._makedense(
                logits, ('', [256, self.NUM_CLASSES], 'relu'), train_flag)
            correct_prediction = tf.equal(
                tf.argmax(logits, 1), tf.argmax(labels, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            loss, cross_entropy = self._loss(labels, logits)
            train_op, lr = self._train_op(global_step, loss)

            sess.run(tf.global_variables_initializer())
            train_data = np.concatenate((np.array(self.train_data), np.array(self.valid_data)), axis=0).tolist()
            train_label = np.concatenate((np.array(self.train_label), np.array(self.valid_label)), axis=0).tolist()
            max_steps = (self.NUM_EXAMPLES_FOR_TRAIN + self.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL) // self.batch_size
            for ep in range(self.epoch):
                start_time = time.time()
                for step in range(max_steps):
                    batch_x = train_data[step *
                                         self.batch_size:(step + 1) * self.batch_size]
                    batch_y = train_label[step *
                                          self.batch_size:(step + 1) * self.batch_size]
                    batch_x = DataSet().process(batch_x)
                    _, loss_value, acc = sess.run([train_op, cross_entropy, accuracy],
                                                  feed_dict={x: batch_x, labels: batch_y, train_flag: True})

                precision = 0
                num_iter = len(self.test_label) // self.batch_size
                for step in range(num_iter):
                    batch_x = self.test_data[step *
                                             self.batch_size:(step + 1) * self.batch_size]
                    batch_y = self.test_label[step *
                                              self.batch_size:(step + 1) * self.batch_size]
                    l, acc_ = sess.run([cross_entropy, accuracy],
                                       feed_dict={x: batch_x, labels: batch_y, train_flag: False})
                    precision += acc_ / num_iter
                retrain_log += 'retrain epoch %d: precision = %.3f, cost time %.3f\n' % (
                    ep, precision, float(time.time() - start_time))
        NAS_LOG << ('eva', retrain_log)
        return float(precision)

    def _eval(self, sess, train_op, cross_entropy, accuracy, x, labels, train_flag, lr):
        precision = np.zeros([self.epoch])
        for ep in range(self.epoch):
            start_time = time.time()
            # print("epoch", ep, ":")
            # train step
            for step in range(self.max_steps):
                batch_x = self.train_data[step *
                                          self.batch_size:(step + 1) * self.batch_size]
                batch_y = self.train_label[step *
                                           self.batch_size:(step + 1) * self.batch_size]
                batch_x = DataSet().process(batch_x)
                _, loss_value, acc, ll = sess.run([train_op, cross_entropy, accuracy, lr],
                                                  feed_dict={x: batch_x, labels: batch_y, train_flag: True})
                if np.isnan(loss_value):
                    NAS_LOG << ('eva', self.log)
                    print("loss NAN")
                    return [-1]
                # sys.stdout.write("\r>> train %d/%d loss %.4f acc %.4f" %
                #                  (step, self.max_steps, loss_value, acc))
            # sys.stdout.write("\n")
            # evaluation step
            num_iter = self.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL // self.batch_size
            for step in range(num_iter):
                batch_x = self.valid_data[step *
                                          self.batch_size:(step + 1) * self.batch_size]
                batch_y = self.valid_label[step *
                                           self.batch_size:(step + 1) * self.batch_size]
                l, acc_ = sess.run([cross_entropy, accuracy],
                                   feed_dict={x: batch_x, labels: batch_y, train_flag: False})
                precision[ep] += acc_ / num_iter
                # sys.stdout.write(
                #     "\r>> valid %d/%d loss %.4f acc %.4f" % (step, num_iter, l, acc_))
            # early stop
            if ep > 5:
                if precision[ep] < 1.2 / self.NUM_CLASSES:
                    NAS_LOG << ('eva', self.log)
                    print("early stop")
                    return [-1]
                if 2 * precision[ep] - precision[ep - 5] - precision[ep - 1] < 0.001 / self.NUM_CLASSES:
                    precision = precision[:ep]
                    # print('early stop at %d epoch' % ep)
                    self.log += 'early stop at %d epoch\n' % ep
                    break

            self.log += 'epoch %d: precision = %.3f, cost time %.3f\n' % (
                ep, precision[ep], float(time.time() - start_time))
            # print('precision = %.3f, cost time %.3f' %
            #       (precision[ep], float(time.time() - start_time)))
        NAS_LOG << ('eva', self.log)

        return precision

    def set_data_size(self, num):
        if num > self.NUM_EXAMPLES_FOR_TRAIN or num < 0:
            num = self.NUM_EXAMPLES_FOR_TRAIN
            self.train_num = self.NUM_EXAMPLES_FOR_TRAIN
            print('Warning! Data size has been changed to',
                  num, ', all data is loaded.')
        else:
            self.train_num = num
        # print('************A NEW ROUND************')
        self.max_steps = self.train_num // self.batch_size
        return


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    eval = Evaluator()
    eval.set_data_size(50000)
    eval.set_epoch(10)
    # graph_full = [[1], [2], [3], []]
    # cell_list = [Cell('conv', 64, 5, 'relu'), Cell('pooling', 'max', 3), Cell('conv', 64, 5, 'relu'),
    #              Cell('pooling', 'max', 3)]
    # lenet = NetworkItem(0, graph_full, cell_list, "")
    # e = eval.evaluate(lenet, [], is_bestNN=True)
    # Network.pre_block.append(lenet)

    graph_full = [[1, 4, 3, 5], [2, 4, 3], [3, 5], [5], [3, 5]]
    cell_list = [Cell('conv', 24, 1, 'relu'), Cell('conv', 16, 3, 'relu'), Cell('conv', 24, 3, 'relu'),
                 Cell('conv', 24, 1, 'relu'), Cell('conv', 32, 1, 'relu')]
    network1 = NetworkItem(0, graph_full, cell_list, "")
    # network2 = NetworkItem(1, graph_full, cell_list, "")
    e = eval.evaluate(network1, [], is_bestNN=True)
    # eval.set_data_size(500)
    # e = eval.evaluate(network2, [network1], is_bestNN=True)
    # eval.set_epoch(2)
    # eval.retrain([network1, network2])
    # eval.add_data(5000)
    # print(eval._toposort([[1, 3, 6, 7], [2, 3, 4], [3, 5, 7, 8], [
    #       4, 5, 6, 8], [5, 7], [6, 7, 9, 10], [7, 9], [8], [9, 10], [10]]))
    # graph_full = [[1], [2], [3], []]
    # cell_list = [Cell('conv', 64, 5, 'relu'), Cell('pooling', 'max', 3), Cell('conv', 64, 5, 'relu'),
    #              Cell('pooling', 'max', 3)]
    # cell_list = [cell_list]
    # e=eval.evaluate(graph_full,cell_list[-1])#,is_bestNN=True)
    # print(e)
    # cellist=[('conv', 128, 1, 'relu'), ('conv', 32, 1, 'relu'), ('conv', 256, 1, 'relu'), ('pooling', 'max', 2), ('pooling', 'global', 3), ('conv', 32, 1, 'relu')]
    # cellist=[('pooling', 'global', 2), ('pooling', 'max', 3), ('conv', 21, 32, 'leakyrelu'), ('conv', 16, 32, 'leakyrelu'), ('pooling', 'max', 3), ('conv', 16, 32, 'leakyrelu')]
    # graph_part = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15], [16], [17], []]
    # cell_list = [('conv', 64, 3, 'relu'), ('conv', 64, 3, 'relu'), ('pooling', 'max', 2), ('conv', 128, 3, 'relu'),
    #              ('conv', 128, 3, 'relu'), ('pooling', 'max', 2), ('conv', 256, 3, 'relu'),
    #              ('conv', 256, 3, 'relu'), ('conv', 256, 3, 'relu'), ('pooling', 'max', 2),
    #              ('conv', 512, 3, 'relu'), ('conv', 512, 3, 'relu'), ('conv', 512, 3, 'relu'),
    #              ('pooling', 'max', 2), ('conv', 512, 3, 'relu'), ('conv', 512, 3, 'relu'),
    #              ('conv', 512, 3, 'relu'), ('dense', [4096, 4096, 1000], 'relu')]
    # pre_block = [network]
    # Network.pre_block.append(network1)
    # network2 = NetworkItem(1, graph_full, cell_list, "")
    # e = eval.evaluate(network2, is_bestNN=True)
    # Network.pre_block.append(network2)
    # network3 = NetworkItem(2, graph_full, cell_list, "")
    # e = eval.evaluate(network3, is_bestNN=True)
    # e=eval.train(network.graph_full,cellist)
    # print(e)
    eval.retrain()
