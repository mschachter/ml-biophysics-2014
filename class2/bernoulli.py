import numpy as np
from scipy.stats import *
import matplotlib.pyplot as plt


def plot_bernoulli(p, num_samples=1000):
    """
        Plot a histogram of a Bernoulli distribution.

        p: The probability of emitting a 1.
    """

    #p is the probability of emitting a one
    rv = bernoulli(p)

    #generate random bernoulli numbers and compute the mean
    sample_data = rv.rvs(num_samples)
    sample_mean = sample_data.mean()
    print 'Sample mean: {:.2f}'.format(sample_mean)
    print 'True mean: {:0.2f}'.format(p)

    #make a histogram of the sample data
    plt.figure()
    plt.hist(sample_data, color='g')
    plt.xlim(-0.5, 1.5)
    plt.title('The Bernoulli Distribution (p={:.2f}, N={})'.format(p, num_samples))
    plt.xticks([0, 1], [0, 1])
    plt.xlabel('Value')
    plt.ylabel('Count')
    plt.show()

plot_bernoulli(0.25, num_samples=1000)

