
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


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', 'data', 'Directory for storing data')

mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)


# Create the model

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Define loss and optimizer

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

start_time = timeit.default_timer()
# Train
sess = tf.Session()
sess.run(tf.initialize_all_variables())
run_metadata = tf.RunMetadata()
for i in range(25000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	sess.run(train_step(
		feed_dict={x: batch_xs, y_: batch_ys},
		options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
		run_metadata=run_metadata
	))

	if i % 100 == 0:
	# Test trained model
		print("%d iterations, accuracy: %f" % (i, accuracy.eval({x: mnist.test.images, y_: mnist.test.labels})))

end_time = timeit.default_timer()
print("final accuracy: %f" % accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))
print("total time: %.1fs" % (end_time - start_time))


