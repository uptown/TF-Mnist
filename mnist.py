import tensorflow as tf
import tensorlayer as tfl
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from model import build

# 재현을 위해 rand seed 설정
tf.set_random_seed(777)

# load mnist data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
batch_size = 250
norm = 0.15

dropout_rate = tf.placeholder(tf.float32)
true_labels = tf.placeholder(tf.float32, [None, 10])
data = tf.placeholder(tf.float32, [None, 28, 28, 1])

train, cost, accuracy = build(data, true_labels, dropout_rate)

saver = tf.train.Saver()

with tf.device('/gpu:0'):
    sess = tf.Session()

    sess.run(tf.global_variables_initializer())
    cost_sum = 0
    max_acc = 0
    max_iter = 0
    epoch = 100
    for i in range(172 * epoch):
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
                acc += accuracy(sess, test_xs.reshape(len(test_ys), 28, 28, 1) - norm, test_ys)
                j += 1
            if acc > max_acc:
                max_iter = i
                save_path = saver.save(sess, "model_/model.ckpt")
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

        # 가로 수축, 세로 수축 .... 시간이 없어서 못함ㅠㅠㅠ
        #

        batch_xs4 = np.concatenate((np.zeros([batch_size, 28, 1]), np.resize(batch_xs, [batch_size, 28, 26]),
                                    np.zeros([batch_size, 28, 1])), axis=2)

        train(sess, np.reshape(batch_xs4, (batch_size, 28, 28, 1)) - norm, batch_ys)

        train(sess, np.reshape(batch_xs, (batch_size, 28, 28, 1)) - norm, batch_ys)

        cost_sum += cost(sess, np.reshape(batch_xs, (batch_size, 28, 28, 1)) - norm, batch_ys)
