import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plot_eigenfaces():
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

    #do PCA on the matrix of images
    pca = PCA()
    pca.fit(X)

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
    num_plots = nrows*ncols
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

def gaussian_example(N):

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
plot_eigenfaces()
plt.show()

