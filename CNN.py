"""
Created by Haochuan Lu on 10/23/17.
"""
import tensorflow as tf
import pandas as pd
import numpy as np


def weight_variable(shape,name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial,name=name)


def bias_variable(shape,name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial,name=name)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 1, 2, 1],strides=[1, 1, 2, 1], padding='SAME')


class CNN:
    def __init__(self, input_dim, output_dim):
        self.__input_dim = input_dim
        self.__output_dim = output_dim


    def train(self, X, y):
        x = tf.placeholder("float", shape=[None, self.__input_dim])
        y_ = tf.placeholder("float", shape=[None, self.__output_dim])
        W_conv1 = weight_variable([1, 25, 1, 32],'W_conv1')
        b_conv1 = bias_variable([32],'b_conv1')
        x_image = tf.reshape(x, [-1, 1, self.__input_dim, 1])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)
        W_conv2 = weight_variable([1, 12, 32, 32], 'W_conv2')
        b_conv2 = bias_variable([32], 'b_conv2')
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)
        W_fc1 = weight_variable([int(self.__input_dim/4) *32, 256],'W_fc1')
        b_fc1 = bias_variable([256],'b_fc1')
        h_pool2_flat = tf.reshape(h_pool2, [-1, int(self.__input_dim / 4)*32])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        W_fc2 = weight_variable([256, self.__output_dim],'W_fc2')
        keep_prob = tf.placeholder("float")
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        b_fc2 = bias_variable([self.__output_dim],'b_fc2')
        y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
        cross_entropy = -tf.reduce_sum(y * tf.log(y_conv + 1e-10), name='cross_entropy')#-tf.reduce_sum(y_*tf.log(y_conv),name='cross_entropy')
        train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"),name='accuracy')
        all_vars = tf.trainable_variables()
        for v in all_vars:
            print(v.name)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            for i in range(20000):
                if i%100 == 0:
                    train_cross_entropy = sess.run(cross_entropy,feed_dict={x:X, y_: y, keep_prob: 1.0})
                    train_accuracy = accuracy.eval(feed_dict={x:X, y_: y, keep_prob: 1.0})
                    result = sess.run(y_conv,feed_dict={x:X, y_:y, keep_prob: 1.0})
                    print(result)
                    print ("step %d, training accuracy %g"%(i, train_accuracy))
                    print (train_cross_entropy)
                sess.run(train_step,feed_dict={x: X, y_: y, keep_prob: 1.0})
            saver_path = saver.save(sess, "model/model_4.ckpt")
            print ("Model saved in file: ", saver_path)


if __name__ == "__main__":
    X = pd.read_csv("data/single_mall_generated_feature/m_6167_2.csv", index_col=0)
    y = pd.read_csv("data/second/m_6167_l.csv")
    X = np.array(X)
    y = np.array(pd.get_dummies(y, columns=["shop_num"]))
    print(y)
    input_dim = X.shape[1]
    output_dim = y.shape[1]
    cnn = CNN(input_dim, output_dim)
    cnn.train(X,y)
