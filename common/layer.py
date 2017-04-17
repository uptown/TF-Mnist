import tensorflow as tf
import numpy as np


class Layer:
    mean = 0.0
    stddev = 0.01

    def __init__(self, name):
        self.name = name

    def connect(self, *args, **kwargs):
        raise NotImplemented


class ConvolutionLayer(Layer):
    def __init__(self, name, in_ch, out_ch, c_size=3, strides=None, padding="SAME", activation=tf.nn.relu):
        super().__init__(name)
        if strides is None:
            strides = [1, 1, 1, 1]
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.c_size = 3
        self.strides = strides
        self.padding = padding
        self.activation = activation
        with tf.variable_scope(self.name):
            self.f = tf.Variable(
                tf.truncated_normal([self.c_size, self.c_size, self.in_ch, self.out_ch], self.mean, self.stddev),
                name=self.name + ".f")
            self.b = tf.Variable(tf.truncated_normal([self.out_ch], self.mean, self.stddev), name=self.name + ".b")

    def connect(self, data):
        t = tf.nn.conv2d(data, self.f, strides=self.strides, padding=self.padding)
        t = tf.nn.bias_add(t, self.b)
        return self.activation(t)


class MaxPoolLayer(Layer):
    def __init__(self, name, k_size=None, strides=None, padding='SAME'):
        super().__init__(name)

        if strides is None:
            strides = [1, 2, 2, 1]
        if k_size is None:
            k_size = [1, 2, 2, 1]
        self.k_size = k_size
        self.strides = strides
        self.padding = padding

    def connect(self, data):
        return tf.nn.max_pool(data, ksize=self.k_size, strides=self.strides, padding=self.padding, name=self.name)


class RegionalSelectionLayer(Layer):
    def __init__(self, name, region_count, size):
        super().__init__(name)
        self.region_count = region_count
        self.size = size
        self.s = tf.range(0, size, 1)
        self.region_map = []
        for i in range(2, region_count + 2):
            region = []
            for j in range(size):
                region.append((j // i) % 2)
            self.region_map.append(region)
        self.region_map = tf.constant(self.region_map)

    def connect(self, data, selected_param):  # [20, features], [20]
        return tf.multiply(tf.to_float(tf.reshape(self.region_map[selected_param], [1, -1])),
                           tf.reshape(data, [-1, self.size]))


class AvgPoolLayer(Layer):
    def __init__(self, name, k_size=None, strides=None, padding='SAME'):
        super().__init__(name)

        if strides is None:
            strides = [1, 2, 2, 1]
        if k_size is None:
            k_size = [1, 2, 2, 1]
        self.k_size = k_size
        self.strides = strides
        self.padding = padding

    def connect(self, data):
        return tf.nn.avg_pool(data, ksize=self.k_size, strides=self.strides, padding=self.padding, name=self.name)


class FullConnectedLayer(Layer):
    def __init__(self, name, in_ch, out_ch, activation=tf.nn.relu):
        super().__init__(name)

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.activation = activation
        with tf.variable_scope(self.name):

            if type(in_ch) in (list, tuple):
                self.w = tf.Variable(tf.truncated_normal([sum(self.in_ch), self.out_ch], self.mean, self.stddev),
                                     name=self.name + ".w")
            else:
                self.w = tf.Variable(tf.truncated_normal([self.in_ch, self.out_ch], self.mean, self.stddev),
                                     name=self.name + ".w")
            self.b = tf.Variable(tf.truncated_normal([self.out_ch], self.mean, self.stddev), name=self.name + ".b")

    def connect(self, *args):
        if len(args) > 1:
            # data_min = tf.reduce_mean(data)
            # print(data_min)
            data = tf.reshape(args[0], [-1, self.in_ch[0]])
            i = 1
            in_ch = self.in_ch[0]
            for each in args[1:]:
                data = tf.concat([data, tf.reshape(each, [-1, self.in_ch[i]])], 1)
                in_ch += self.in_ch[i]
                i += 1
            x = tf.reshape(data, [-1, in_ch])
        else:
            data = args[0]  # tf.reshape(args[0], [-1])

            if type(self.in_ch) in (list, tuple):
                x = tf.reshape(data, [-1, self.in_ch[0]])
            else:
                x = tf.reshape(data, [-1, self.in_ch])
        fc = tf.nn.bias_add(tf.matmul(x, self.w), self.b)
        return self.activation(fc)


class DropConnectedLayer(FullConnectedLayer):

    def __init__(self, name, in_ch, out_ch, rate, activation=tf.nn.relu):
        super().__init__(name, in_ch, out_ch, activation)
        self.rate = rate
        self.w = tf.nn.dropout(self.w, 1 - self.rate)


class DropOutLayer(Layer):
    def __init__(self, name, rate):
        super().__init__(name)

        self.rate = rate

    def connect(self, *args):
        data = args[0]
        return tf.nn.dropout(data, 1 - self.rate, name=self.name)

