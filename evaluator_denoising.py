import os
import tensorflow as tf
import numpy as np

from base import Cell, NetworkItem
from info_str import NAS_CONFIG
from utils import NAS_LOG

from tqdm import tqdm
from glob import glob
import cv2
import random
import sys

DATA_AUG_TIMES = 3


class DataSet:
    # TODO for dataset changing please rewrite this class's "inputs" function and "process" function

    def __init__(self):
        self.data_path = "/home/amax/PycharmProjects/ViDeNN/Spartial-CNN/data/"
        return

    def add_noise(self):
        imgs_path = glob(self.data_path + "pristine_images/*.bmp")
        num_of_samples = len(imgs_path)
        imgs_path_train = imgs_path[:int(num_of_samples * 0.7)]
        imgs_path_test = imgs_path[int(num_of_samples * 0.7):]

        sigma_train = np.linspace(0, 50, int(num_of_samples * 0.7) + 1)
        for i in tqdm(range(int(num_of_samples * 0.7)), desc="[*] Creating original-noisy train set..."):
            img_path = imgs_path_train[i]
            img_file = os.path.basename(img_path).split('.bmp')[0]
            sigma = sigma_train[i]
            img_original = cv2.imread(img_path)
            img_noisy = self.gaussian_noise(sigma, img_original)

            cv2.imwrite(self.data_path + "train/noisy/" + img_file + ".png", img_noisy)
            cv2.imwrite(self.data_path + "train/original/" + img_file + ".png", img_original)

        for i in tqdm(range(int(num_of_samples * 0.3)), desc="[*] Creating original-noisy test set..."):
            img_path = imgs_path_test[i]
            img_file = os.path.basename(img_path).split('.bmp')[0]
            sigma = np.random.randint(0, 50)

            img_original = cv2.imread(img_path)
            img_noisy = self.gaussian_noise(sigma, img_original)

            cv2.imwrite(self.data_path + "test/noisy/" + img_file + ".png", img_noisy)
            cv2.imwrite(self.data_path + "test/original/" + img_file + ".png", img_original)

    def gaussian_noise(self, sigma, image):
        gaussian = np.random.normal(0, sigma, image.shape)
        noisy_image = image + gaussian
        noisy_image = np.clip(noisy_image, 0, 255)
        noisy_image = noisy_image.astype(np.uint8)
        return noisy_image

    def inputs(self, pat_size=50, stride=100, batch_size=64):
        '''
        Method for load data

        :return: train_data, train_label, valid_data, valid_label, test_data, test_label
        '''
        noisy_eval_files = glob(self.data_path + 'test/noisy/*.png')
        noisy_eval_files = sorted(noisy_eval_files)
        test_data = np.array([cv2.imread(img) for img in noisy_eval_files])

        eval_files = glob(self.data_path + 'test/original/*.png')
        eval_files = sorted(eval_files)
        test_label = np.array([cv2.imread(img) for img in eval_files])
        if os.path.exists(self.data_path + "train/img_noisy_pats.npy"):
            train_data = np.load(self.data_path + "train/img_noisy_pats.npy")
            train_label = np.load(self.data_path + "train/img_clean_pats.npy")
            train_data = train_data.astype(np.float32)
            train_label = train_label.astype(np.float32)
            return train_data, train_label, test_data, test_label
        if not os.path.exists(self.data_path + "train/noisy/") or not os.listdir(self.data_path + "train/noisy/"):
            self.add_noise()

        global DATA_AUG_TIMES
        count = 0
        filepaths = glob(
            self.data_path + "train/original/" + '/*.png')  # takes all the paths of the png files in the train folder
        filepaths.sort(key=lambda x: int(os.path.basename(x)[:-4]))  # order the file list
        filepaths_noisy = glob(self.data_path + "train/noisy/" + '/*.png')
        filepaths_noisy.sort(key=lambda x: int(os.path.basename(x)[:-4]))
        print("[*] Number of training samples: %d" % len(filepaths))
        scales = [1, 0.8]

        # calculate the number of patches
        for i in range(len(filepaths)):
            img = cv2.imread(filepaths[i])
            for s in range(len(scales)):
                newsize = (int(img.shape[0] * scales[s]), int(img.shape[1] * scales[s]))
                img_s = cv2.resize(img, newsize, interpolation=cv2.INTER_CUBIC)
                im_h = img_s.shape[0]
                im_w = img_s.shape[1]
                for x in range(0, (im_h - pat_size), stride):
                    for y in range(0, (im_w - pat_size), stride):
                        count += 1

        origin_patch_num = count * DATA_AUG_TIMES

        if origin_patch_num % batch_size != 0:
            numPatches = (origin_patch_num // batch_size + 1) * batch_size  # round
        else:
            numPatches = origin_patch_num
        print("[*] Number of patches = %d, batch size = %d, total batches = %d" % \
              (numPatches, batch_size, numPatches / batch_size))

        # data matrix 4-D
        train_label = np.zeros((numPatches, pat_size, pat_size, 3), dtype="uint8")  # clean patches
        train_data = np.zeros((numPatches, pat_size, pat_size, 3), dtype="uint8")  # noisy patches

        count = 0
        # generate patches
        for i in range(len(filepaths)):
            img = cv2.imread(filepaths[i])
            img_noisy = cv2.imread(filepaths_noisy[i])
            for s in range(len(scales)):
                newsize = (int(img.shape[0] * scales[s]), int(img.shape[1] * scales[s]))
                img_s = cv2.resize(img, newsize, interpolation=cv2.INTER_CUBIC)
                img_s_noisy = cv2.resize(img_noisy, newsize, interpolation=cv2.INTER_CUBIC)
                img_s = np.reshape(np.array(img_s, dtype="uint8"),
                                   (img_s.shape[0], img_s.shape[1], 3))  # extend one dimension
                img_s_noisy = np.reshape(np.array(img_s_noisy, dtype="uint8"),
                                         (img_s_noisy.shape[0], img_s_noisy.shape[1], 3))  # extend one dimension

                for j in range(DATA_AUG_TIMES):
                    im_h = img_s.shape[0]
                    im_w = img_s.shape[1]
                    for x in range(0, im_h - pat_size, stride):
                        for y in range(0, im_w - pat_size, stride):
                            a = random.randint(0, 7)
                            train_label[count, :, :, :] = self.process(
                                img_s[x:x + pat_size, y:y + pat_size, :], a)
                            train_data[count, :, :, :] = self.process(
                                img_s_noisy[x:x + pat_size, y:y + pat_size, :], a)
                            count += 1
        # pad the batch
        if count < numPatches:
            to_pad = numPatches - count
            train_label[-to_pad:, :, :, :] = train_label[:to_pad, :, :, :]
            train_data[-to_pad:, :, :, :] = train_data[:to_pad, :, :, :]

        train_data = train_data.astype(np.float32)
        return train_data.astype(np.float32), train_label.astype(np.float32), test_data, test_label

    def process(self, image, mode):
        if mode == 0:
            # original
            return image
        elif mode == 1:
            # flip up and down
            return np.flipud(image)
        elif mode == 2:
            # rotate counterwise 90 degree
            return np.rot90(image)
        elif mode == 3:
            # rotate 90 degree and flip up and down
            image = np.rot90(image)
            return np.flipud(image)
        elif mode == 4:
            # rotate 180 degree
            return np.rot90(image, k=2)
        elif mode == 5:
            # rotate 180 degree and flip
            image = np.rot90(image, k=2)
            return np.flipud(image)
        elif mode == 6:
            # rotate 270 degree
            return np.rot90(image, k=3)
        elif mode == 7:
            # rotate 270 degree and flip
            image = np.rot90(image, k=3)
            return np.flipud(image)


class Evaluator:
    def __init__(self):
        # don't change the parameters below
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        self.block_num = 0
        self.log = ''
        self.model_path = "./model"

        # change the value of parameters below
        self.input_shape = [None, None, None, 3]
        self.output_shape = [None, None, None, 3]
        self.batch_size = 64
        self.train_data, self.train_label, self.test_data, self.test_label = DataSet().inputs()

        self.INITIAL_LEARNING_RATE = 0.025
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
        elif cell.type == 'id':
            layer = tf.identity(inputs)
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
        return

    def _makeconv(self, x, hplist, node, train_flag):
        """Generates a convolutional layer according to information in hplist
        Args:
            x: inputing data.
            hplist: hyperparameters for building this layer
            node: int, the index of this operation
        Returns:
            conv_layer: the output tensor
        """
        with tf.variable_scope('block' + str(self.block_num) + 'conv' + str(node)) as scope:
            inputdim = x.shape[3]
            kernel = self._get_variable('weights',
                                        shape=[hplist.kernel_size, hplist.kernel_size, inputdim, hplist.filter_size])
            x = self._activation_layer(hplist.activation, x, scope)
            x = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], padding='SAME')
            biases = self._get_variable('biases', hplist.filter_size)
            x = self._batch_norm(tf.nn.bias_add(x, biases), train_flag)
        return x

    def _makesep_conv(self, inputs, hplist, node, train_flag):
        with tf.variable_scope('block' + str(self.block_num) + 'conv' + str(node)) as scope:
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
        if hplist.pooling_type == 'avg':
            return tf.nn.avg_pool(inputs, ksize=[1, hplist.kernel_size, hplist.kernel_size, 1],
                                  strides=[1, hplist.kernel_size, hplist.kernel_size, 1], padding='SAME')
        elif hplist.pooling_type == 'max':
            return tf.nn.max_pool(inputs, ksize=[1, hplist.kernel_size, hplist.kernel_size, 1],
                                  strides=[1, hplist.kernel_size, hplist.kernel_size, 1], padding='SAME')
        elif hplist.pooling_type == 'global':
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
            with tf.variable_scope('block' + str(self.block_num) + 'dense' + str(i)) as scope:
                weights = self._get_variable('weights', shape=[inputs.shape[-1], neural_num])
                biases = self._get_variable('biases', [neural_num])
                mul = tf.matmul(inputs, weights) + biases
                if neural_num == self.output_shape[-1]:
                    local3 = self._activation_layer('', mul, scope)
                else:
                    local3 = self._activation_layer(hplist[2], mul, scope)
            inputs = local3
        return inputs

    def _pad(self, inputs, layer):
        # padding
        a = tf.shape(layer)[1]
        b = tf.shape(inputs)[1]
        pad = tf.abs(tf.subtract(a, b))
        output = tf.where(tf.greater(a, b), tf.concat([tf.pad(inputs, [[0, 0], [0, pad], [0, pad], [0, 0]]), layer], 3),
                          tf.concat([inputs, tf.pad(layer, [[0, 0], [0, pad], [0, pad], [0, 0]])], 3))
        return output

    def evaluate(self, network, pre_block=[], is_bestNN=False, update_pre_weight=False):
        '''Method for evaluate the given network.
        
        :param network: NetworkItem
        :param pre_block: The pre-block structure, every block has two parts: graph_part and cell_list of this block.
        :param is_bestNN: Symbol for indicating whether the evaluating network is the best network of this round.
        :param update_pre_weight: Symbol for indicating whether to update previous blocks' weight.
        :return: accuracy, float
        '''
        assert self.train_num >= self.batch_size
        tf.reset_default_graph()
        self.block_num = len(pre_block)

        self.log = "-" * 20 + str(network.id) + "-" * 20 + '\n'
        for block in pre_block:
            self.log = self.log + str(block.graph) + str(block.cell_list) + '\n'
        self.log = self.log + str(network.graph) + str(network.cell_list) + '\n'

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            data_x, data_y, block_input, train_flag = self._get_input(sess, pre_block, update_pre_weight)

            graph_full, cell_list = self._recode(network.graph, network.cell_list,
                                                 NAS_CONFIG['nas_main']['repeat_num'])
            graph_full = graph_full + [[]]
            if NAS_CONFIG['nas_main']['link_node']:
                # a pooling layer for last repeat block
                cell_list = cell_list + [Cell('pooling', 'max', 2)]
            else:
                cell_list = cell_list + [Cell('id', 'max', 1)]
            logits = self._inference(block_input, graph_full, cell_list, train_flag)

            precision, log = self._eval(sess, logits, data_x, data_y, train_flag)
            self.log += log

            saver = tf.train.Saver(tf.global_variables())

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
            data_x = graph.get_tensor_by_name("input:0")
            data_y = graph.get_tensor_by_name("label:0")
            train_flag = graph.get_tensor_by_name("train_flag:0")
            block_input = graph.get_tensor_by_name("last_layer" + str(self.block_num - 1) + ":0")
            # only when there's not so many network in the pool will we update the previous blocks' weight
            if not update_pre_weight:
                block_input = tf.stop_gradient(block_input, name="stop_gradient")
        # if it's the first block
        else:
            data_x = tf.placeholder(np.array(self.train_data).dtype, self.input_shape, name='input')
            data_y = tf.placeholder(np.array(self.train_label).dtype, self.output_shape, name="label")
            train_flag = tf.placeholder(tf.bool, name='train_flag')
            block_input = tf.identity(data_x)
        return data_x, data_y, block_input, train_flag

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

    def _eval(self, sess, logits, data_x, data_y, train_flag):
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
        # TODO shuffle
        # logits = tf.nn.dropout(logits2, keep_prob=1.0)
        noise = tf.layers.conv2d(logits, 3, 3, padding='same', name="l", use_bias=False)
        pred = data_x - noise
        global_step = tf.Variable(0, trainable=False, name='global_step' + str(self.block_num))
        accuracy = self._cal_accuracy(pred, data_y)
        loss = self._loss(pred, data_y)
        train_op = self._train_op(global_step, loss)

        sess.run(tf.global_variables_initializer())

        log = ""
        psnr_sum = 0
        max_steps = self.train_num // self.batch_size
        for _ in range(self.epoch):
            for step in range(max_steps):
                batch_x = self.train_data[step * self.batch_size:(step + 1) * self.batch_size].astype(
                    np.float32) / 255.0
                batch_y = self.train_label[step * self.batch_size:(step + 1) * self.batch_size].astype(
                    np.float32) / 255.0
                _, loss_value, acc, ans_show = sess.run([train_op, loss, accuracy, pred],
                                                        feed_dict={data_x: batch_x, data_y: batch_y, train_flag: True})
                if np.isnan(loss_value):
                    return -1, log
                sys.stdout.write("\r>> train %d/%d loss %.4f acc %.4f" % (step, max_steps, loss_value, acc))
            sys.stdout.write("\n")

            # evaluation step
            for i in range(20):
                batch_x = self.test_data[i].astype(np.float32) / 255.0
                batch_x = batch_x[np.newaxis, ...]
                batch_y = self.test_label[i].astype(np.float32) / 255.0
                batch_y = batch_y[np.newaxis, ...]
                l, psnr = sess.run([loss, accuracy],
                                   feed_dict={data_x: batch_x, data_y: batch_y, train_flag: False})
                psnr_sum += psnr / len(self.test_label)
                print("test %d/%d loss %.4f acc %.4f" % (i, len(self.test_label), l, psnr))

        target = self._cal_multi_target(psnr_sum)
        return target, log

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
        mse = tf.losses.mean_squared_error(labels=labels * 255.0, predictions=logits * 255.0)
        accuracy = 10.0 * (tf.log(255.0 ** 2 / mse) / tf.log(10.0))
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
        loss = (1.0 / self.batch_size) * tf.nn.l2_loss(logits - labels)
        return loss

    def _train_op(self, global_step, loss):
        # TODO change here for learning rate and optimizer
        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        opt = tf.train.AdamOptimizer(0.001, name='Momentum' + str(self.block_num))
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = opt.minimize(loss)
        return train_op

    def _cal_multi_target(self, precision):
        # TODO change here for target calculating
        target = precision
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
    eval.set_data_size(-1)
    eval.set_epoch(50)

    graph_full = [[1]]
    cell_list = [Cell('conv', 128, 3, 'relu')]
    for i in range(2, 19):
        graph_full.append([i])
        cell_list.append(Cell('conv', 64, 3, 'relu'))
    graph_full.append([])
    cell_list.append(Cell('conv', 64, 3, 'relu'))

    # graph_full = [[1, 3], [2, 3], [3], [4]]
    # cell_list = [Cell('conv', 128, 3, 'relu'), Cell('conv', 32, 3, 'relu'), Cell('conv', 24, 3, 'relu'),
    #              Cell('conv', 32, 3, 'relu')]
    network1 = NetworkItem(0, graph_full, cell_list, "")
    network2 = NetworkItem(1, graph_full, cell_list, "")
    e = eval.evaluate(network1, is_bestNN=True)
    eval.set_data_size(500)
    e = eval.evaluate(network2, [network1], is_bestNN=True)
