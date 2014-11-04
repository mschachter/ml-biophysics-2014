import numpy as np
import matplotlib.pyplot as plt

def generate_guassmix(num_samples_per_cluster=100, plot=True):
    """ Generate data from a mixture of 2D Gaussians. """

    #the centers of each Gaussian distribution
    centers = [[1.0, 1.0], [-2.5, -2], [-1.5, 2]]

    #specify the covariance matrix for each Gaussian
    cmats = ([[1.0, 0.3], [0.3, 1.0]],
             [[1.0, 2.5], [2.5, 1.0]],
             [[1.0, 0.0], [0.0, 1.0]])

    #generate random samples for each distribution
    X = list()
    y = list()
    for k,(mean,cov_mat) in enumerate(zip(centers, cmats)):
        X.extend(np.random.multivariate_normal(mean, cov_mat, size=num_samples_per_cluster))
        y.extend([k]*num_samples_per_cluster)

    X = np.array(X)
    y = np.array(y)

    if plot:
        clusters = np.unique(y)
        plt.figure()
        for k in clusters:
            plt.plot(X[y == k, 0], X[y == k, 1], 'o')

    return X,y


if __name__ == '__main__':

    generate_guassmix(num_samples_per_cluster=100)
    plt.show()







