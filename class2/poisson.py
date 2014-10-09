import numpy as np
import matplotlib.pyplot as plt

def plot_poisson(rate, num_samples=10000):

   #generate random Poisson numbers and compute the mean
   sample_data = np.random.poisson(rate, size=num_samples)
   sample_mean = sample_data.mean()
   print 'Sample rate: {:.2f}'.format(sample_mean)
   print 'Sample variance: {:.2f}'.format(sample_data.var())
   print 'True rate: {:0.2f}'.format(rate)

   sample_max = sample_data.max()

   #make a histogram of the sample data
   plt.figure()
   plt.hist(sample_data, color='c', bins=30)
   plt.title('The Poisson Distribution ($\lambda$={:.2f}, N={})'.format(rate, num_samples))
   plt.xlabel('Value')
   plt.ylabel('Count')
   plt.xticks(np.arange(sample_max+1), np.arange(sample_max+1))
   plt.axis('tight')
   plt.show()

plot_poisson(20.0, num_samples=10000)

