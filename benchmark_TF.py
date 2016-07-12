import tensorflow as tf
import os
import numpy as np
np.random.seed(23)
matrix_dim=10000
a=np.random.rand(matrix_dim, matrix_dim).astype('float32')
y=tf.placeholder(dtype='float32',shape=[matrix_dim,matrix_dim],name='y')



print (a)
import time
X=tf.matmul(y,y)
init=tf.initialize_all_variables()
sess=tf.Session()
sess.run(init)
start = time.time()
result=sess.run(X,{y:a})
end_time=time.time()
print(result)
print("total time: ",end_time-start)

