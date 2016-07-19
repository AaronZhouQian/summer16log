
#Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.
See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import data
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import timeit

BATCH_SIZE=100
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', 'data', 'Directory for storing data')

mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
batch_xs, batch_ys = mnist.train.next_batch(55000)

sess = tf.Session()

# Create the model
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

inputs=tf.get_variable("input",trainable=False, shape=[BATCH_SIZE, 784])
labels = tf.get_variable("labels",trainable=False, shape=[BATCH_SIZE,10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

softmax_probabilities = tf.nn.softmax(tf.matmul(inputs, W) + b)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(labels * tf.log(softmax_probabilities), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

start_time = timeit.default_timer()
# Train


with sess.as_default():
	tf.initialize_all_variables().run(feed_dict={x: batch_xs.astype('float32'), y_: batch_ys})
	for j in range(50):
		for i in range(0,50000,BATCH_SIZE):
			# Test trained model
			sess.run(inputs.assign(tf.slice(x, [i,0],[BATCH_SIZE,-1])))
			sess.run(train_step)
			correct_prediction = tf.equal(tf.argmax(softmax_probabilities, 1), tf.argmax(labels, 1))
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	end_time = timeit.default_timer()
	print("final accuracy: %f" % accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))
	print("total time: %.1fs" % (end_time - start_time))












