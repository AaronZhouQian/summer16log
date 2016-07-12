import numpy as np
import theano
import theano.tensor as T
import time

np.random.seed(23)
matrix_dim = 10000
def matrix():
    matrix= np.random.rand(matrix_dim, matrix_dim).astype('float32')
    x = T.matrix(name="x",dtype='float32')
    y = T.dot(x,x)
    f = theano.function([x],y)
    start_time = time.time()
    end_time = time.time()
    print(f(matrix))
    total_time = end_time - start_time
    print("total time is: %.5f" %total_time)


def main():
    matrix()

if __name__=="__main__":
    main()

