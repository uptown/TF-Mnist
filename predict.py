import tensorflow as tf
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
    saver.restore(sess, "model_9962/model.ckpt")

    xs, ys = mnist.test.images, mnist.test.labels
    j = 0
    acc = 0
    while len(xs) > j * batch_size:
        test_xs, test_ys = xs[j * batch_size:j * batch_size + batch_size], ys[
                                                                           j * batch_size:j * batch_size + batch_size]
        acc += accuracy(sess, test_xs.reshape(len(test_ys), 28, 28, 1) - norm, test_ys)
        j += 1
    print("Accurancy: ", acc / (j * batch_size))
