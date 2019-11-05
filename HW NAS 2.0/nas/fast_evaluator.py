import tensorflow as tf
import numpy as np
import time
import math
import datetime
import os

# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = 32
NUM_CLASSES = 10
global NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000
# Constants describing the training process.
INITIAL_LEARNING_RATE = 0.001  # Initial learning rate.
MOVING_AVERAGE_DECAY = 0.99
batch_size = 128
epoch = 164
weight_decay = 0.0003
momentum_rate = 0.9


def read_cifar10(filename_queue):
    """Reads and parses examples from CIFAR10 data files.

    Recommendation: if you want N-way read parallelism, call this function
    N times.  This will give you N independent Readers reading different
    files & positions within those files, which will give better mixing of
    examples.

    Args:
      filename_queue: A queue of strings with the filenames to read from.

    Returns:
      An object representing a single example, with the following fields:
        height: number of rows in the result (32)
        width: number of columns in the result (32)
        depth: number of color channels in the result (3)
        key: a scalar string Tensor describing the filename & record number
          for this example.
        label: an int32 Tensor with the label in the range 0..9.
        uint8image: a [height, width, depth] uint8 Tensor with the image data
    """

    class CIFAR10Record(object):
        pass

    result = CIFAR10Record()

    # Dimensions of the images in the CIFAR-10 dataset.
    # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
    # input format.
    label_bytes = 1  # 2 for CIFAR-100
    result.height = 32
    result.width = 32
    result.depth = 3
    image_bytes = result.height * result.width * result.depth
    # Every record consists of a label followed by the image, with a
    # fixed number of bytes for each.
    record_bytes = label_bytes + image_bytes

    # Read a record, getting filenames from the filename_queue.  No
    # header or footer in the CIFAR-10 format, so we leave header_bytes
    # and footer_bytes at their default of 0.
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(filename_queue)

    # Convert from a string to a vector of uint8 that is record_bytes long.
    record_bytes = tf.decode_raw(value, tf.uint8)

    # The first bytes represent the label, which we convert from uint8->int32.
    result.label = tf.cast(
        tf.slice(record_bytes, [0], [label_bytes]), tf.int32)

    # The remaining bytes after the label represent the image, which we reshape
    # from [depth * height * width] to [depth, height, width].
    depth_major = tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]),
                             [result.depth, result.height, result.width])
    # Convert from [depth, height, width] to [height, width, depth].
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])

    return result


def inputs(eval_data, data_dir, batch_size):
    """Construct input for CIFAR evaluation using the Reader ops.

    Args:
      eval_data: bool, indicating if one should use the train or eval data set.
      data_dir: Path to the CIFAR-10 data directory.
      batch_size: Number of images per batch.

    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    """
    if not eval_data:
        filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
                     for i in range(1, 6)]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    else:
        filenames = [os.path.join(data_dir, 'test_batch.bin')]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

    for f in filenames:
        if not gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames)

    # Read examples from files in the filename queue.
    read_input = read_cifar10(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    if eval_data:
        # Image processing for evaluation.
        # Crop the central [height, width] of the image.
        distorted_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                                 width, height)
    else:
        # Image processing for training the network. Note the many random
        # distortions applied to the image.

        # Randomly crop a [height, width] section of the image.
        distorted_image = tf.image.random_crop(reshaped_image, [height, width])

        # Randomly flip the image horizontally.
        distorted_image = tf.image.random_flip_left_right(distorted_image)

        # Because these operations are not commutative, consider randomizing
        # randomize the order their operation.
        distorted_image = tf.image.random_brightness(distorted_image,
                                                     max_delta=63)
        distorted_image = tf.image.random_contrast(distorted_image,
                                                   lower=0.2, upper=1.8)

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_whitening(distorted_image)

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)

    # Generate a batch of images and labels by building up a queue of examples.
    num_preprocess_threads = 16
    images, label_batch = tf.train.shuffle_batch(
        [float_image, read_input.label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)

    # Display the training images in the visualizer.
    tf.image_summary('images', images)

    return images, tf.reshape(label_batch, [batch_size])


def _makeconv(self, inputs, hplist, node):
    """Generates a convolutional layer according to information in hplist

    Args:
    inputs: inputing data.
    hplist: hyperparameters for building this layer
    node: number of this cell
    Returns:
    tensor.
    """
    # print('Evaluater:right now we are making conv layer, its node is %d, and the inputs is'%node,inputs,'and the node before it is ',cellist[node-1])
    with tf.variable_scope('conv' + str(node)) as scope:
        inputdim = inputs.shape[3]
        kernel = tf.get_variable('weights', shape=[hplist[2], hplist[2], inputdim, hplist[1]],
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv = tf.nn.conv2d(inputs, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', hplist[1], initializer=tf.constant_initializer(0.0))
        bias = batch_norm(tf.nn.bias_add(conv, biases))
        if hplist[3] == 'relu':
            conv1 = tf.nn.relu(bias, name=scope.name)
        elif hplist[3] == 'tenh' or hplist[3] == 'tanh':
            conv1 = tf.tanh(bias, name=scope.name)
        elif hplist[3] == 'sigmoid':
            conv1 = tf.sigmoid(bias, name=scope.name)
        elif hplist[3] == 'identity':
            conv1 = tf.identity(bias, name=scope.name)
        elif hplist[3] == 'leakyrelu':
            conv1 = tf.nn.leaky_relu(bias, name=scope.name)
    return conv1


def _makepool(self, inputs, hplist):
    """Generates a pooling layer according to information in hplist

    Args:
        inputs: inputing data.
        hplist: hyperparameters for building this layer
    Returns:
        tensor.
    """
    if hplist[1] == 'avg':
        return tf.nn.avg_pool(inputs, ksize=[1, hplist[2], hplist[2], 1],
                              strides=[1, hplist[2], hplist[2], 1], padding='SAME')
    elif hplist[1] == 'max':
        return tf.nn.max_pool(inputs, ksize=[1, hplist[2], hplist[2], 1],
                              strides=[1, hplist[2], hplist[2], 1], padding='SAME')
    elif hplist[1] == 'global':
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
    i = 0
    print(inputs.shape)
    inputs = tf.reshape(inputs, [-1, 2 * 2 * 512])

    for neural_num in hplist[1]:
        with tf.variable_scope('dense' + str(i)) as scope:
            weights = tf.get_variable('weights', shape=[inputs.shape[-1], neural_num],
                                      initializer=tf.contrib.keras.initializers.he_normal())
            # weight = tf.multiply(tf.nn.l2_loss(weights), 0.004, name='weight_loss')
            # tf.add_to_collection('losses', weight)
            biases = tf.get_variable('biases', [neural_num], initializer=tf.constant_initializer(0.0))
            if hplist[2] == 'relu':
                local3 = tf.nn.relu(batch_norm(tf.matmul(inputs, weights) + biases), name=scope.name)
            elif hplist[2] == 'tanh':
                local3 = tf.tanh(tf.matmul(inputs, weights) + biases, name=scope.name)
            elif hplist[2] == 'sigmoid':
                local3 = tf.sigmoid(tf.matmul(inputs, weights) + biases, name=scope.name)
            elif hplist[2] == 'identity':
                local3 = tf.identity(tf.matmul(inputs, weights) + biases, name=scope.name)
        inputs = local3
        i += 1
    return inputs


def _inference(self, images, graph_part, cellist):  # ,regularizer):
    '''Method for recovering the network model provided by graph_part and cellist.
    Args:
      images: Images returned from Dataset() or inputs().
      graph_part: The topology structure of th network given by adjacency table
      cellist:

    Returns:
      Logits.'''
    # print('Evaluater:starting to reconstruct the network')
    nodelen = len(graph_part)
    inputs = [0 for i in range(nodelen)]  # input list for every cell in network
    inputs[0] = images
    getinput = [False for i in range(nodelen)]  # bool list for whether this cell has already got input or not
    getinput[0] = True

    for node in range(nodelen):
        # print('Evaluater:right now we are processing node %d'%node,', ',cellist[node])
        if cellist[node][0] == 'conv':
            layer = self._makeconv(inputs[node], cellist[node], node)
            layer = tf.nn.lrn(layer, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
        elif cellist[node][0] == 'pooling':
            layer = self._makepool(inputs[node], cellist[node])
        elif cellist[node][0] == 'dense':
            layer = self._makedense(inputs[node], cellist[node])
        else:
            print('WRONG!!!!! Notice that you got a layer type we cant process!', cellist[node][0])
            layer = []

        # update inputs information of the cells below this cell
        for j in graph_part[node]:
            if getinput[j]:  # if this cell already got inputs from other cells precedes it
                # padding
                a = int(layer.shape[1])
                b = int(inputs[j].shape[1])
                pad = abs(a - b)
                if layer.shape[1] > inputs[j].shape[1]:
                    inputs[j] = tf.pad(inputs[j], [[0, 0], [0, pad], [0, pad], [0, 0]])
                if layer.shape[1] < inputs[j].shape[1]:
                    layer = tf.pad(layer, [[0, 0], [0, pad], [0, pad], [0, 0]])
                inputs[j] = tf.concat([inputs[j], layer], 3)
            else:
                inputs[j] = layer
                getinput[j] = True

    # softmax
    # layer = tf.reshape(layer, [batch_size, -1])
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.get_variable('weights', shape=[layer.shape[-1], NUM_CLASSES],
                                  initializer=tf.truncated_normal_initializer(stddev=0.04))  # 1 / float(dim)))
        biases = tf.get_variable('biases', shape=[NUM_CLASSES], initializer=tf.constant_initializer(0.0))
        # softmax_linear = tf.nn.softmax(tf.matmul(layer, weights)+ biases, name=scope.name)
        softmax_linear = tf.add(tf.matmul(layer, weights), biases, name=scope.name)
        # tf.add_to_collection('losses', regularizer(weights))
    return softmax_linear


def train():
    """Train CIFAR-10 for a number of steps."""
    with tf.Graph().as_default():
        train_flag = tf.placeholder(tf.bool)
        global_step = tf.Variable(0, trainable=False)

        # Get images and labels for CIFAR-10.
        images, labels = inputs(train_flag, data_dir=data_path, batch_size=batch_size)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = _inference(images)

        # Calculate predictions.
        top_k_op = tf.nn.in_top_k(logits, labels, 1)

        # Calculate loss.
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
        loss = cross_entropy + l2 * weight_decay

        # Variables that affect learning rate.
        num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / batch_size
        decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

        # Decay the learning rate exponentially based on the number of steps.
        lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                        global_step,
                                        decay_steps,
                                        LEARNING_RATE_DECAY_FACTOR,
                                        staircase=True)
        tf.scalar_summary('learning_rate', lr)

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = tf.train.MomentumOptimizer(lr, momentum_rate, use_nesterov=True).minimize(loss)

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables())

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()

        # Build an initialization operation to run below.
        init = tf.initialize_all_variables()

        # Start running operations on the Graph.
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement))
        sess.run(init)

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, graph_def=sess.graph_def)

        for step in range(max_steps):
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss], feed_dict={train_flag: True})
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 10 == 0:
                num_examples_per_step = batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print(format_str % (datetime.now(), step, loss_value,
                                    examples_per_sec, sec_per_batch))

            if step % 100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % 1000 == 0 or (step + 1) == max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

        # Start the queue runners.
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                 start=True))

            num_iter = int(math.ceil(FLAGS.num_examples / batch_size))
            true_count = 0  # Counts the number of correct predictions.
            total_sample_count = num_iter * batch_size
            step = 0
            while step < num_iter and not coord.should_stop():
                predictions = sess.run([top_k_op], feed_dict={train_flag: True})
                true_count += np.sum(predictions)
                step += 1

            # Compute precision @ 1.
            precision = true_count / total_sample_count
            print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='Precision @ 1', simple_value=precision)
            summary_writer.add_summary(summary, global_step)
        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def add_data(add_num=0):
    if NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN + add_num > 50000 or add_num < 0:
        add_num = 50000 - NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
        NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
        print('Warning! Add number has been changed to ', add_num, ', all data is loaded.')
    else:
        NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN += add_num
    # print('************A NEW ROUND************')
    self.network_num = 0
    max_steps = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / batch_size)

    return 0


if