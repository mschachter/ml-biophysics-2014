import numpy as np
import matplotlib.pyplot as plt

#the scipy.stats package contains many different
#types of random variables that we can choose from
from scipy.stats import *


def plot_uniform(num_samples=10000):
    """ Create a uniform random variable and make a histogram.

        num_samples: The number of random numbers to generate.        
    """

    #create a uniform random variable
    rv = uniform()

    #generate a bunch of random numbers from rv
    sample_data = rv.rvs(num_samples)

    #create a histogram
    plt.figure()
    plt.hist(sample_data, bins=30, color='r')
    plt.title('The Uniform Distribution (N={0})'.format(num_samples))
    plt.xlabel('Value')
    plt.ylabel('Count')

    plt.show()

def compute_probability_of_range(start, end):
    """ Compute the probability of a uniform random variable
        falling between start and end.
    """
    #create a uniform random variable
    rv = uniform()

    #compute the difference in cdf between the two points
    p = rv.cdf(end) - rv.cdf(start)

    return p

#compute_probability_of_range(0.25, 0.40)
plot_uniform(num_samples=1000)

       


