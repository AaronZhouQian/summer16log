import numpy as np
import theano
import theano.tensor as T
import time

np.random.seed(23)
matrix_dim = 1000
def matrix():
    matrix= np.random.randn(matrix_dim, matrix_dim)
    x = T.matrix(name="x",dtype='float32')
    y = T.matrix(name="y",dtype='float32')
    start_time = time.time()
    for i in range(100):
        y=T.dot(y,x)
    f = theano.function([x,y],y)
    print(f(matrix,np.eye(matrix_dim,dtype='float32')))
    end_time = time.time()
    total_time = end_time - start_time
    print("total time is: ", total_time)
    print("time per multiplication: ", total_time/100)


def main():
    matrix()

if __name__=="__main__":
    main()

