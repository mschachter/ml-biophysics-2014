import numpy as np
from scipy.stats import *
import matplotlib.pyplot as plt

def plot_gaussian(mean, std, num_samples=10000):

   #create a gaussian random variable
   rv = norm(loc=mean, scale=std)

   #generate random Gaussian numbers and compute the mean
   sample_data = rv.rvs(num_samples)
   sample_mean = sample_data.mean()
   print 'Sample mean: {:.2f}'.format(sample_mean)
   print 'True mean: {:0.2f}'.format(mean)
   print 'Sample std: {:.2f}'.format(sample_std)
   print 'True std: {:0.2f}'.format(std)

   #make a histogram of the sample data
   plt.figure()
   plt.hist(sample_data, color='c', bins=30)
   plt.title('The Gaussian Distribution ($\mu$={:.2f}, $\sigma$={:.2f}, N={})'.format(mean, std, num_samples))
   plt.xlabel('Value')
   plt.ylabel('Count')
   plt.show()

plot_gaussian(0.0, 1.0, num_samples=10000)

