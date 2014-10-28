import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.datasets import *
from sklearn.cross_validation import KFold

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.svm import SVC


def load_binary_digits(noise_std=5.00):
    """
        Loads up some handwritten digit data from scikits and plots it.

        noise_std: The standard deviation of the guassian noise
            added to the image to corrupt it.
    """
    
    #load handwritten digits for 0 and 1
    data_dict = load_digits(n_class=2)  

    #get the binary output targets
    y = data_dict['target']

    #get the feature matrix to regress on
    X = data_dict['data']

    #each pixel takes on a gray level value from 1-16 that indicates
    #intensity. we're going to treat the data as if it was continuous
    #and z-score it.

    #subtract off the mean for each feature
    X -= X.mean(axis=0)
    #compute the standard deviation for each feature    
    Xstd = X.std(axis=0, ddof=1)    
    #divide the feature matrix by the nonzero stds of features
    nz = Xstd > 0.0
    X[:, nz] /= Xstd[nz]

    #add gaussian noise to the images
    X += np.random.randn(X.shape[0], X.shape[1])*noise_std

    #plot some of the images
    nrows = 10
    ncols = 10
    nplots = nrows*ncols
    plt.figure()
    for k in range(nplots):
        #reshape the features into an image
        img = X[k].reshape([8, 8])
        plt.subplot(nrows, ncols, k+1)
        plt.imshow(img, interpolation='nearest', aspect='auto', cmap=cm.gist_yarg)
        plt.xticks([])
        plt.yticks([])
    
    return X,y


def plot_sigmoid():
    x = np.linspace(-7, 7, 100)
    plt.axhline(0.5, color='b')
    plt.axvline(0.0, color='r')
    plt.plot(x, 1.0 / (1.0 + np.exp(-x)), 'k-', linewidth=2.0)
    plt.axis('tight')


def fit_binary_data(X, y, C_to_try=[1e-3, 1e-2, 1e-1, 1.0], nfolds=10):
    """
        Use logistic regression to fit the image data using cross validation.
    """

    best_C = None
    best_pc = 0
    best_weights = None
    best_intercept = None
    
    for C in C_to_try:

        aucs = list()
        pcs = list()
        weights = list()
        intercepts = list()
        cmats = list()

        for train_indices,test_indices in KFold(len(y), n_folds=nfolds):
            assert len(np.intersect1d(train_indices, test_indices)) == 0
            #break the data matrix up into training and test sets
            Xtrain, Xtest, ytrain, ytest = X[train_indices], X[test_indices], y[train_indices], y[test_indices]
            
            #construct a logistic regression object
            lc = LogisticRegression(C=C)
            lc.fit(Xtrain, ytrain)

            #predict the identity of images on the test set
            ypred = lc.predict(Xtest)

            #compute confusion matrix
            cmat = confusion_matrix(ytest, ypred, labels=[0, 1]).astype('float')

            #normalize each row of the confusion matrix so they represent probabilities
            cmat = (cmat.T / cmat.sum(axis=1)).T

            #compute the percent correct
            pcs.append((cmat[0, 0] + cmat[1, 1]) / 2.0)

            #record the confusion matrix for this fold
            cmats.append(cmat)

            #predict the probability of a 1 for each test sample
            ytest_prob = lc.predict_proba(Xtest)[:, 1]

            #compute and record the area under the curve for the predictions
            auc = roc_auc_score(ytest, ytest_prob)
            aucs.append(auc)

            #record the weights and intercept
            weights.append(lc.coef_)
            intercepts.append(lc.intercept_)

        #compute the mean confusion matrix
        cmats = np.array(cmats)
        Cmean = cmats.mean(axis=0)

        #compute the mean AUC and PC
        mean_auc = np.mean(auc)
        std_auc = np.std(auc, ddof=1)
        mean_pc = np.mean(pcs)
        std_pc = np.std(pcs, ddof=1)

        #compute the mean weights
        weights = np.array(weights)
        mean_weights = weights.mean(axis=0)        
        mean_intercept = np.mean(intercepts)

        print 'C={:.4f}'.format(C)
        print '\tPercent Correct: {:.3f} +/- {:.3f}'.format(mean_pc, std_pc)
        print '\tAUC: {:.3f} +/- {:.3f}'.format(mean_auc, std_auc)
        print '\tConfusion Matrix:'
        print Cmean

        #determine if we've found the best model thus far
        if mean_pc > best_pc:
            best_pc = mean_pc
            best_C = C
            best_weights = mean_weights
            best_intercept = mean_intercept

    #reshape the weights into an image
    weights_img = best_weights.reshape([8, 8])

    #make a plot of the weights
    weights_absmax = np.abs(best_weights).max()
    plt.figure()
    plt.imshow(weights_img, interpolation='nearest', aspect='auto', cmap=cm.seismic, vmin=-weights_absmax, vmax=weights_absmax)
    plt.colorbar()
    plt.title('Model Weights C: {:.4f}, PC: {:.2f}, Intercept: {:.3f}'.format(best_C, best_pc, best_intercept))


def generate_nonlinear_data(plot=False):

    #the decision function is just some wacky arbitrary nonlinear function
    decision_function = lambda xp,yp: yp**3*np.cos(xp) + xp**3*np.sin(yp)

    npts = 25
    xvals = np.linspace(-2, 2, npts)
    yvals = np.linspace(-2, 2, npts)
    Xcoords,Ycoords = np.meshgrid(xvals, yvals)

    D = decision_function(Xcoords, Ycoords)

    #construct feature matrix and target vector from data
    X = list()
    y = list()
    for xval,yval,dval in zip(Xcoords.ravel(), Ycoords.ravel(), D.ravel()):
        #decide class based on sign of decision function
        clz = int(dval) > 0
        X.append( (xval, yval))
        y.append(clz)
    X = np.array(X)
    y = np.array(y)

    if plot:
        plt.figure()
        zi = y == 0
        plt.plot(X[zi, 0], X[zi, 1], 'ro')
        plt.plot(X[~zi, 0], X[~zi, 1], 'k^')
        plt.legend(['0', '1'])
        plt.axis('tight')

    return X,y


def plot_2d_predictions(classifier, X, y):

    #construct a 2D grid of points and then turn it into a feature matrix called Xgrid
    npts = 25
    xvals = np.linspace(-2, 2, npts)
    yvals = np.linspace(-2, 2, npts)
    Xcoords,Ycoords = np.meshgrid(xvals, yvals)
    Xgrid = np.array(zip(Xcoords.ravel(), Ycoords.ravel()))

    #compute the predictions for each location in the 2D grid
    ypred = classifier.predict(Xgrid)

    #plot the actual data
    plt.figure()
    plt.subplot(2, 1, 1)
    zi = y == 0
    plt.plot(X[zi, 0], X[zi, 1], 'ro')
    plt.plot(X[~zi, 0], X[~zi, 1], 'k^')
    plt.legend(['0', '1'])
    plt.axis('tight')
    plt.title('Actual Data')

    #plot the predictions
    plt.subplot(2, 1, 2)
    zi = ypred == 0
    plt.plot(Xgrid[zi, 0], Xgrid[zi, 1], 'ro')
    plt.plot(Xgrid[~zi, 0], Xgrid[~zi, 1], 'k^')
    plt.legend(['0', '1'])
    plt.title('Predictions')
    plt.suptitle(classifier.__class__.__name__)


if __name__ == '__main__':

    #X,y = load_binary_digits(noise_std=5.0)
    #fit_binary_data(X, y)
    #plot_sigmoid()

    X,y = generate_nonlinear_data()
    lc = LogisticRegression(C=1e-3)
    lc.fit(X, y)
    plot_2d_predictions(lc, X, y)

    svmc = SVC()
    svmc.fit(X, y)
    plot_2d_predictions(svmc, X, y)

    plt.show()
    
      
    

