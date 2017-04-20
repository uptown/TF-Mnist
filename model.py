from common.layer import *
from common.network import *


def build(data, true_labels, dropout_rate, learning_rate=0.0001):
    cnn = ClassificationNetwork(data)

    # cnn
    cnn.add_layer_and_connect(ConvolutionLayer("conv1", 1, 128, c_size=3, padding="SAME"))
    cnn.add_layer_and_connect(ConvolutionLayer("conv2", 128, 128, c_size=3, padding="VALID"))
    cnn.add_layer_and_connect(ConvolutionLayer("conv3", 128, 128, c_size=3, padding="VALID"))
    _, pool1 = cnn.add_layer_and_connect(MaxPoolLayer("pool1"))

    cnn.add_layer_and_connect(ConvolutionLayer("conv4", 128, 256, c_size=5, padding="VALID"))
    cnn.add_layer_and_connect(ConvolutionLayer("conv5", 256, 512, c_size=5, padding="VALID"))
    cnn.add_layer_and_connect(ConvolutionLayer("conv6", 512, 512, c_size=5, padding="VALID"))
    cnn.add_layer_and_connect(ConvolutionLayer("conv7", 512, 512, c_size=5, padding="VALID"))

    cnn.add_layer_and_connect(ConvolutionLayer("conv8", 512, 1024, c_size=7, padding="VALID"))
    cnn.add_layer_and_connect(ConvolutionLayer("conv9", 1024, 1024, c_size=7, padding="SAME"))

    cnn.add_layer_and_connect(DropConnectedLayer("fc1", [2 * 2 * 1024, 12 * 12 * 128], 1024, dropout_rate), ["conv9", "pool1"])

    cnn.add_layer_and_connect(FullConnectedLayer("fc1_1", [28 * 28], 2048), ["data"])

    cnn.add_layer_and_connect(FullConnectedLayer("fc2", [1024, 2048], 4096), ["fc1", "fc1_1"])

    cnn.add_layer_and_connect(DropOutLayer("do10", dropout_rate))
    cnn.add_layer_and_connect(FullConnectedLayer("fc3", 4096, 2048))
    cnn.add_layer_and_connect(DropOutLayer("do11", dropout_rate))
    cnn.add_layer_and_connect(FullConnectedLayer("fc4", 2048, 512))
    cnn.add_layer_and_connect(DropOutLayer("do12", dropout_rate))
    cnn.add_layer_and_connect(FullConnectedLayer("fc5", 512, 10))
    cnn.build(true_labels)

    x = tf.unstack(tf.reshape(tf.transpose(pool1, [0, 3, 1, 2]), [-1, 128 * 12, 12]), 128 * 12, 1)  # 128, 12, 12

    # rnn
    rnn = ClassificationNetwork(x)
    rnn.add_layer_and_connect(LSTMLayer("rnn", 256, 10))
    rnn.build(true_labels)

    cost_eq = cnn.cost_eq + rnn.cost_eq
    pred_val = cnn.pred_val + rnn.pred_val

    opt = tf.train.AdamOptimizer(name="adam", learning_rate=learning_rate).minimize(cost_eq)
    prediction = tf.equal(tf.argmax(pred_val, 1), tf.argmax(true_labels, 1))
    acc = tf.reduce_sum(tf.cast(prediction, tf.float32))

    def train(session, d, true_out, dropout=0.5):
        return session.run(opt,
                           feed_dict={data: d, true_labels: true_out, dropout_rate: dropout})

    def cost(session, d, true_out):
        return session.run(cost_eq,
                           feed_dict={data: d, true_labels: true_out, dropout_rate: 0})

    def accuracy(session, d, true_out):
        return session.run(acc, feed_dict={data: d, true_labels: true_out, dropout_rate: 0})

    return train, cost, accuracy
