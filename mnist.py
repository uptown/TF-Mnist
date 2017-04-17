import tensorflow as tf
import random
import os
import numpy as np
from common.layer import *
import tensorlayer as tfl
from common.utils import *
from tensorflow.examples.tutorials.mnist import input_data

# 재현을 위해 rand seed 설정
tf.set_random_seed(777)

# load mnist data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


class Network:
    def __init__(self, input_shape=(None, 28, 28, 1), output_shape=(None, 10), learning_rate=0.0001):
        self.data = tf.placeholder(tf.float32, list(input_shape))
        self.dropout_rate = tf.placeholder(tf.float32)
        self.true_labels = tf.placeholder(tf.float32, list(output_shape))
        self.layers = []

        # cnn
        conv1 = ConvolutionLayer("conv1", 1, 128, c_size=3, padding="SAME").connect(self.data)  # 28
        conv2 = ConvolutionLayer("conv2", 128, 128, c_size=3, padding="VALID").connect(conv1)  # 26
        conv3 = ConvolutionLayer("conv3", 128, 128, c_size=3, padding="VALID").connect(conv2)  # 24
        pool1 = MaxPoolLayer("pool1").connect(conv3)  # 12
        # do1 = DropOutLayer("do1", self.dropout_rate).connect(conv3)
        conv4 = ConvolutionLayer("conv4", 128, 256, c_size=5, padding="VALID").connect(pool1)  # 10
        # do2 = DropOutLayer("do2", self.dropout_rate).connect(conv4)
        conv5 = ConvolutionLayer("conv5", 256, 512, c_size=5, padding="VALID").connect(conv4)  # 8
        # do3 = DropOutLayer("do3", self.dropout_rate).connect(conv5)
        conv6 = ConvolutionLayer("conv6", 512, 512, c_size=5, padding="VALID").connect(conv5)  # 6
        # do4 = DropOutLayer("do4", self.dropout_rate).connect(conv6)
        conv7 = ConvolutionLayer("conv7", 512, 512, c_size=7, padding="VALID").connect(conv6)  # 4
        # do5 = DropOutLayer("do5", self.dropout_rate).connect(conv7)
        conv8 = ConvolutionLayer("conv8", 512, 1024, c_size=7, padding="VALID").connect(conv7)  # 2
        conv9 = ConvolutionLayer("conv9", 1024, 1024, c_size=7, padding="SAME").connect(conv8)  # 2

        fc1_1 = FullConnectedLayer("fc1_1", [28 * 28], 2048).connect(self.data)

        fc1 = DropConnectedLayer("fc1", [2 * 2 * 1024, 12 * 12 * 128], 1024, 0.3).connect(conv9, pool1)
        fc2 = FullConnectedLayer("fc2", [1024, 2048], 4096).connect(fc1, fc1_1)
        do10 = DropOutLayer("do10", self.dropout_rate).connect(fc2)

        fc3 = FullConnectedLayer("fc3", 4096, 2048).connect(do10)
        do11 = DropOutLayer("do11", self.dropout_rate).connect(fc3)
        fc4 = FullConnectedLayer("fc4", 2048, 512).connect(do11)
        do12 = DropOutLayer("do12", self.dropout_rate).connect(fc4)
        eq = FullConnectedLayer("fc5", 512, 10).connect(do12)

        self.cost_eq = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=eq, labels=self.true_labels))

        pred_val = tf.nn.softmax(eq)
        # rnn
        n_hidden = 256
        w = tf.Variable(tf.truncated_normal([n_hidden, output_shape[-1]], Layer.mean, Layer.stddev),
                        name="rnn.w")
        b = tf.Variable(tf.truncated_normal([output_shape[-1]], Layer.mean, Layer.stddev), name="rnn.b")

        x = tf.unstack(tf.reshape(tf.transpose(pool1, [0, 3, 1, 2]), [-1, 128 * 12, 12]), 128 * 12, 1)  # 128, 2, 2
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)
        outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
        eq2 = tf.matmul(outputs[-1], w) + b

        self.cost_eq = self.cost_eq + tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=eq2, labels=self.true_labels))
        pred_val += tf.nn.softmax(eq2)

        # self.true_labels, None x 10
        # Feature들을 각 feature region에 대응되게 학습시켜본다

        # region specific fc
        # for region in range(output_shape[-1]):
        #    mask = tf.constant([1.0 if tt == region else 0 for tt in range(output_shape[-1])])

        #    t_s1 = RegionalSelectionLayer("t_s1", output_shape[-1], 2 * 2 * 1024).connect(conv9, region)

        #    t_fc1 = DropConnectedLayer("t_fc1", 2 * 2 * 1024, 512, self.dropout_rate).connect(t_s1)
        #    do11 = DropOutLayer("do11", self.dropout_rate).connect(t_fc1)
        #    t_fc2 = DropConnectedLayer("t_fc2", 512, 512, self.dropout_rate).connect(do11)
        #    t_s2 = RegionalSelectionLayer("t_s2", output_shape[-1], 512).connect(t_fc2, region)
        #    eq3 = tf.multiply(mask, FullConnectedLayer("t_fc3", 512, 1).connect(t_s2))

        # eq = tf.add(eq, t_eq)

        #    self.cost_eq = self.cost_eq + tf.reduce_mean(
        #        tf.nn.softmax_cross_entropy_with_logits(logits=eq3, labels=self.true_labels))
        #    pred_val += tf.nn.softmax(eq3)

        # for layer in self.layers:
        #     print(eq.get_shape())
        #     eq = layer.connect(eq)
        # print(self.true_labels.get_shape(), self.region_params.get_shape())

        self.opt = tf.train.AdamOptimizer(name="adam", learning_rate=learning_rate).minimize(self.cost_eq)

        prediction = tf.equal(tf.argmax(pred_val, 1), tf.argmax(self.true_labels, 1))
        self.acc = tf.reduce_sum(tf.cast(prediction, tf.float32))

    def train(self, session, data, true_out, dropout=0.5):
        return session.run(self.opt,
                           feed_dict={self.data: data, self.true_labels: true_out, self.dropout_rate: dropout})

    def cost(self, session, data, true_out):
        return session.run(self.cost_eq, feed_dict={self.data: data, self.true_labels: true_out, self.dropout_rate: 0})

    def accuracy(self, session, data, true_out):
        return session.run(self.acc, feed_dict={self.data: data, self.true_labels: true_out, self.dropout_rate: 0})


Layer.mean = 0.0
Layer.stddev = 0.01
batch_size = 250
norm = 0.15

net = Network(learning_rate=0.0001, input_shape=(batch_size, 28, 28, 1), output_shape=(batch_size, 10))

saver = tf.train.Saver()

with tf.device('/gpu:0'):
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    cost_sum = 0
    max_acc = 0
    max_iter = 0
    epoch = 100
    for i in range(180 * epoch):
        # test classification again, should have a higher probability about tiger
        if i % 100 == 0 and i != 0:
            print(i)
            print("cost: ", cost_sum / 100.0)
            cost_sum = 0
        if i % epoch == 0 and i != 0:
            j = 0
            acc = 0
            xs, ys = mnist.test.images, mnist.test.labels
            while len(xs) > j * batch_size:
                test_xs, test_ys = xs[j * batch_size:j * batch_size + batch_size], ys[
                                                                                   j * batch_size:j * batch_size + batch_size]
                acc += net.accuracy(sess, test_xs.reshape(len(test_ys), 28, 28, 1) - norm, test_ys)
                j += 1
            if acc > max_acc:
                max_iter = i
                save_path = saver.save(sess, "model/model.ckpt")
            max_acc = max(max_acc, acc)
            print("Acc: ", int(acc), "/" + str(len(xs)))
            print("Max Acc: ", int(max_acc), "/" + str(len(xs)), max_iter)

        # tk 설치
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        if i % 4 == 0:
            batch_xs = tfl.prepro.elastic_transform_multi(batch_xs.reshape([batch_size, 28, 28]), alpha=1,
                                                          sigma=0.04)

        if i % 4 == 1:
            batch_xs = tfl.prepro.rotation_multi(batch_xs.reshape([batch_size, 28, 28, 1]), rg=40, is_random=True)

        if i % 4 == 2:
            batch_xs = tfl.prepro.rotation_multi(batch_xs.reshape([batch_size, 28, 28, 1]), rg=40, is_random=True)
            batch_xs = tfl.prepro.elastic_transform_multi(batch_xs.reshape([batch_size, 28, 28]), alpha=1,
                                                          sigma=0.04)

        net.train(sess, np.reshape(batch_xs, (batch_size, 28, 28, 1)) - norm, batch_ys)

        cost_sum += net.cost(sess, np.reshape(batch_xs, (batch_size, 28, 28, 1)) - norm, batch_ys)

    xs, ys = mnist.test.images, mnist.test.labels
    j = 0
    acc = 0
    while len(xs) > j * batch_size:
        test_xs, test_ys = xs[j * batch_size:j * batch_size + batch_size], ys[
                                                                           j * batch_size:j * batch_size + batch_size]
        acc += net.accuracy(sess, test_xs.reshape(len(test_ys), 28, 28, 1) - norm, test_ys)
        j += 1
    print(acc / (j * 100))
