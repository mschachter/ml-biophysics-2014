import numpy as np

def example1():
    A = np.array([ [2, -1], [1, 1] ])
    b = np.array([1, 5])
    v = np.array([-1, -1])

    u = np.dot(A, v)
    print 'actual right hand side b: ',v
    print 'Av: ',u

def example2():

    A = np.array([ [2, -1], [1, 1] ])
    b = np.array([1, 5])
    v = np.linalg.solve(A, b)

    print 'Solution to Av=b is v=',v

def example3():

    A = np.array([ [2, -1], [1, 1] ])
    B = np.array([ [3, 2], [-1, -0.5] ])

    print 'AB ='
    print np.dot(A, B)
    print 'BA ='
    print np.dot(B, A)

def example4():

    A = np.array([ [-3, -1, 4], [0, 4, 0], [5, -9, -2] ])
    print 'A ='
    print A
    print 'A.T ='
    print A.T

def example5():
    A = np.array([ [1, 2], [4, 8] ])
    b = np.array([3, 6])
    v = np.linalg.solve(A, b)

    print 'Solution to Av=b is v=',v

def example6():
    A = np.array([ [2, -1], [1, 1] ])
    print 'A='
    print A
    print 'rank of A=',np.linalg.matrix_rank(A)

    B = np.array([ [1, 2], [4, 8] ])
    print 'B='
    print B
    print 'rank of B=',np.linalg.matrix_rank(B)

def example7():

    N = 5
    M = N
    #create a random data matrix
    X = np.random.randn(N, M)
    #create a random weight vector
    w = np.random.randn(M)

    #generate output samples
    y = np.dot(X, w)

    #solve the linear system and see if the weights match
    w_est = np.linalg.solve(X, y)

    print 'actual weight vector: ', w
    print 'linear system solution weight vector: ', w_est
    

def least_squares_fit(X, y):
    
    #compute the autocorrelation matrix, which is X.T*X
    autocorr = np.dot(X.T, X)

    #invert the autocorrelation matrix
    autocorr_inv = np.linalg.inv(autocorr)
    
    #compute the matrix-vector multiplication X.T*y
    crosscorr = np.dot(X.T, y)
    
    #compute the optimal weights w
    w = np.dot(autocorr_inv, crosscorr)

    return w

def create_fake_data(num_samples, num_features, noise_std=0.0):
    """ Create a matrix of fake data, some random weights, and produce an output data vector y. """
    
    #create a random data matrix
    X = np.random.randn(num_samples, num_features)

    #create a random weight vector
    w = np.random.randn(num_features)

    #generate output samples
    y = np.dot(X, w)

    #add some random noise to y
    y += np.random.randn(num_samples)*noise_std

    return X,y,w

X,y,w = create_fake_data(100, 5, noise_std=0)
w_est = least_squares_fit(X, y)
print 'True value for w: ',w
print 'Least squares estimate for w: ', w_est



