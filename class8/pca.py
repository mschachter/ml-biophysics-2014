from matplotlib import cm
import numpy as np

import matplotlib.pyplot as plt

from sklearn.datasets import *
from sklearn.decomposition import PCA


def pca_example(N=1000):
    """ In this example we'll illustrate PCA in two dimensions using data
        generated from a multivariate Gaussian.
    """

    #construct a 2x2 covariance matrix
    C = np.array([[1.0, 0.5], [0.5, 1.0]])

    #generate samples from a 2D multivariate Gaussian
    X = np.random.multivariate_normal(np.array([0.0, 0.0]), C, size=N)

    #fit PCA on the data
    pca = PCA()
    pca.fit(X)

    #project the data onto the principal components
    Xproj = pca.transform(X)

    #print the covariance matrices of raw and projected data
    Craw = np.cov(X.T)
    Cproj = np.cov(Xproj.T)
    print 'Raw data covariance matrix:'
    print Craw
    print 'Projected data covariance matrix:'
    print Cproj

    plt.figure()

    #plot the raw data
    plt.subplot(2, 1, 1)
    plt.axhline(0.0, c='k')
    plt.axvline(0.0, c='k')
    plt.plot(X[:, 0], X[:, 1], 'ko')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Raw Data')

    #plot the PCs over the raw data
    pc1 = pca.components_[0, :]*3
    pc2 = pca.components_[1, :]*3

    plt.plot([0.0, pc1[0]], [0.0, pc1[1]], 'r-', linewidth=3.0)
    plt.plot([0.0, pc2[0]], [0.0, pc2[1]], 'g-', linewidth=3.0)
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)

    plt.subplot(2, 1, 2)
    plt.axhline(0.0, c='k')
    plt.axvline(0.0, c='k')
    plt.plot(Xproj[:, 0], Xproj[:, 1], 'ro')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)

    plt.title('Projected Data')


def plot_eigenfaces(num_pcs=36):
    import glob
    #get the 64x64 grayscale image names from the current directory
    image_names = glob.glob("images/*.jpg")

    #construct a data matrix of flattened images
    X = list()
    for iname in image_names:
        #read the image from the file
        img = plt.imread(iname)
        X.append(img.ravel())
    X = np.array(X)

    assert num_pcs <= X.shape[0], "There are only {} data points, can't have more PCs than that!".format(X.shape[0])

    #do PCA on the matrix of images
    pca = PCA(n_components=num_pcs)
    pca.fit(X)

    #project the data into a lower dimensional subspace
    Xproj = pca.transform(X)

    #plot some of the actual faces
    plt.figure()
    plt.suptitle('Actual Faces')
    nrows = 6
    ncols = 6
    num_plots = nrows*ncols
    for k,flat_img in enumerate(X[:num_plots, :]):
        #reshape the flattened image into a matrix
        img = flat_img.reshape([64, 64])
        plt.subplot(nrows, ncols, k)
        plt.imshow(img, interpolation='nearest', aspect='auto')

    #plot some of the eigenfaces and the variance they capture
    plt.figure()
    plt.suptitle('Eigenfaces')
    nrows = int(np.ceil(np.sqrt(num_pcs)))
    ncols = nrows
    num_plots = min(nrows*ncols, num_pcs)
    for k,explained_variance in enumerate(pca.explained_variance_ratio_[:num_plots]):
        #get the principal component
        pc = pca.components_[k, :]

        #reshape it into an image
        pc_img = pc.reshape([64, 64])

        #plot the principle component
        plt.subplot(nrows, ncols, k)
        plt.imshow(pc_img, interpolation='nearest', aspect='auto')

        #show the variance captured by this component
        plt.title('EV: {:.3f}'.format(explained_variance))

    #reconstruct the compressed images
    nrows = 6
    ncols = 6
    num_plots = nrows*ncols
    plt.figure()
    plt.suptitle('Reconstructed Images')
    for k,compressed_img in enumerate(Xproj[:num_plots, :]):
        #uncompress the image
        flat_img = pca.inverse_transform(compressed_img)
        img = flat_img.reshape([64, 64])
        plt.subplot(nrows, ncols, k)
        plt.imshow(img, interpolation='nearest', aspect='auto')


def plot_digit_covariance(digit=0):

    #load handwritten digits for 0
    data_dict = load_digits()

    #get the target values
    y = data_dict['target']

    #get the feature matrix to regress on
    X = data_dict['data']

    #select out the digits of interest
    index = y == digit

    #compute the covariance matrix of the zeros
    C = np.cov(X[index, :].T)

    #compute the covariance of a random matrix that is the same size
    R = np.random.randn(64, index.sum())
    Crand = np.cov(R)

    #plot the covariance matrix
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.imshow(C, interpolation='nearest', aspect='auto', cmap=cm.afmhot, vmin=0)
    plt.colorbar()
    plt.title('Covariance matrix for {}'.format(digit))

    plt.subplot(2, 1, 2)
    plt.imshow(Crand, interpolation='nearest', aspect='auto', cmap=cm.afmhot, vmin=0)
    plt.colorbar()
    plt.title('Covariance for random matrix')

def eigenvectors_example(N):

    #construct a Gaussian random matrix
    A = np.random.randn(N, N)
    #rescale it so it's maximum value is one
    absmax = np.abs(A).max()
    A /= absmax

    #compute the eigenvalues and eigenvectors of A
    eigenvalues,eigenvectors = np.linalg.eig(A)

    #we want to plot the complex-valued eigenvalues.  we'll
    #consider their real part the x coordinate and the imaginary
    #part the y coordinate.

    plt.figure()
    #first plot the random matrix
    plt.subplot(2, 1, 1)
    plt.imshow(A, interpolation='nearest', aspect='auto', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Random Matrix')

    #then plot the eigenvalue spectrum
    plt.subplot(2, 1, 2)
    #plot the unit circle
    phase = np.linspace(-np.pi, np.pi, 200)
    xcirc = np.cos(phase)
    ycirc = np.sin(phase)
    plt.axhline(0.0, c='k')
    plt.axvline(0.0, c='k')
    plt.plot(xcirc, ycirc, 'k-')
    plt.plot(eigenvalues.real, eigenvalues.imag, 'ro')
    plt.axis('tight')
    plt.title('Eigenvalues')

#gaussian_example(100)
plot_eigenfaces(num_pcs=36)
#plot_digit_covariance(0)
#pca_example()

plt.show()

