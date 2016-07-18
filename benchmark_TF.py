import tensorflow as tf
import os
import numpy as np
import time
np.random.seed(23)

matrix_dim=15000
matrix_initializer=np.random.rand(matrix_dim, matrix_dim).astype('float32')
with tf.variable_scope("test"):
    bigmatrix=tf.get_variable(name="bigmatrix", shape=[matrix_dim,matrix_dim], initializer=tf.constant_initializer(matrix_initializer))

product = tf.matmul(bigmatrix, bigmatrix)
init=tf.initialize_all_variables()

sess=tf.Session()
sess.run(init)
start = time.time()
sess.run(product)
end_time=time.time()
print("total time: ",end_time-start)

