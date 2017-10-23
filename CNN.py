"""
Created by Haochuan Lu on 10/23/17.
"""
import tensorflow as tf


def weight_variable(shape,name):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial,name=name)


def bias_variable(shape,name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial,name=name)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 0, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 1, 2, 1],strides=[1, 0, 2, 1], padding='SAME')


class CNN:
    def __init__(self, input_dim, output_dim):
        self.__input_dim = input_dim
        self.__output_dim = output_dim


    def train(self, X, y):
        x = tf.placeholder("float", shape=[None, self.__input_dim])
        y_ = tf.placeholder("float", shape=[None, self.__output_dim])
        W_conv1 = weight_variable([1, 5, 1, 32],'W_conv1')
        b_conv1 = bias_variable([32],'b_conv1')
        x_image = tf.reshape(x, [-1, 1, self.__input_dim, 1])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)
        W_conv2 = weight_variable([1, 5, 128, 128], 'W_conv2')
        b_conv2 = bias_variable([128], 'b_conv2')
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)
        W_fc1 = weight_variable([7 * 7 * 128, 1024],'W_fc1')
        b_fc1 = bias_variable([1024],'b_fc1')
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*128])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        W_fc2 = weight_variable([1024, 10],'W_fc2')
        keep_prob = tf.placeholder("float")
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        b_fc2 = bias_variable([10],'b_fc2')
        y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv),name='cross_entropy')
train_step = tf.train.AdamOptimizer(3e-5).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"),name='accuracy')
all_vars = tf.trainable_variables()
for v in all_vars:
    print(v.name)
saver = tf.train.Saver()
print(mnist.train.images.shape)
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for i in range(20000):
        batch = mnist.train.next_batch(100)
        if i%100 == 0:
            train_cross_entropy = sess.run(cross_entropy,feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
            train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
            print ("step %d, training accuracy %g"%(i, train_accuracy))
            print (train_cross_entropy)
        sess.run(train_step,feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    saver_path = saver.save(sess, "model/model_4.ckpt")
    print ("Model saved in file: ", saver_path)
