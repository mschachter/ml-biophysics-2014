import numpy as np
from scipy.stats import *
import pandas as pd
import matplotlib.pyplot as plt

def generate_cell_data(num_samples=100):
    """
        Create fake data that mimics a drug trial on various cell types.
    """

    #probabiity that a cell was treated
    p_treated = 0.5

    #probability mass function for cell type
    type_pmf = np.array([0.20, 0.40, 0.15, 0.25])
    #cdf for cell type
    type_cdf = np.cumsum(type_pmf)

    #mean and std of cell diameter in microns
    diameter_mean = 40.0
    diameter_std = 5.0

    #mean calcium concentration as a function of cell diameter and treatment
    def ca_conc(dia, treatment):
        base_ca = (np.random.randn() + dia) / 6.0
        return base_ca + treatment*np.random.poisson()

    #mean lifetime of cell in days as a function of cell type
    mean_lifetime_by_type = np.array([1.0, 1.5, 2.0, 4.0])

    #create a dictionary of data columns to be turned into a DataFrame
    data = {'treated':list(), 'type':list(), 'diameter':list(), 'calcium':list(), 'lifetime':list()}
    for n in range(num_samples):
        #sample whether or not the cell was treated
        treated = np.random.rand() < p_treated

        #sample the cell type
        cell_type = np.where(type_cdf >= np.random.rand())[0].min()

        #sample the diameter
        dia = np.random.randn()*diameter_std + diameter_mean

        #sample the calcium concentration
        ca = ca_conc(dia, treated)

        #sample cell lifetime
        w1 = 1.0
        w2 = 0.16
        lifetime = np.random.exponential(w1*mean_lifetime_by_type[cell_type] + w2*ca)

        #append the data
        data['treated'].append(treated)
        data['type'].append(cell_type)
        data['diameter'].append(dia)
        data['calcium'].append(ca)
        data['lifetime'].append(lifetime)

    return pd.DataFrame(data)


def hist2d_plots(df, var1="diameter", var2="calcium"):
    
    plt.figure()
    plt.hist2d(df[var1], df[var2], bins=[20, 20], normed=True)
    plt.xlabel(var1)
    plt.ylabel(var2)
    plt.colorbar(label="Joint Probability")


def hist_by_discrete(df):
    #select the indices for treated cells
    treated_indices = df['treated'] == True
    plt.figure()
    #create a histogram for untreated cells
    plt.hist(df['lifetime'][treated_indices].values, bins=30, color='b')
    #overlay a histogram for treated cells
    plt.hist(df['lifetime'][~treated_indices].values, bins=30, color='r', alpha=0.75)
    plt.legend(['Treated', 'Untreated'])
    plt.title('Lifetime')


def lifetime_likelihood(df, rate):
    """ Computes the likelihood of the lifetime column given a rate. """

    #get an array of data points
    x = df['lifetime'].values
    #compute the likelihood
    likelihood = np.log(rate*np.exp(-rate*x)).sum()
    #compute the maximum likelihood estimate of the rate
    best_rate = 1.0 / x.mean()
    max_likelihood = np.log(best_rate*np.exp(-best_rate*x)).sum()
    print 'Log Likehood for rate={:.4f}: {}'.format(rate, likelihood)
    print 'Max Log Likehood for rate={:.4f}: {}'.format(best_rate, max_likelihood)

    plt.figure()
    #plot the data distribution    
    plt.hist(x, bins=30, color='k', normed=True)
    plt.xlabel('Lifetime')

    #plot the pdf of an exponential distribution fit with max likelihood
    rv = expon(best_rate)
    xrng = np.linspace(0, x.max(), 200)
    plt.plot(xrng, rv.pdf(xrng), 'r-', linewidth=2.0)

    #plot the pdf of an exponential distribution with given rate
    rv = expon(rate)
    plt.plot(xrng, rv.pdf(xrng), 'b-', linewidth=2.0)
    plt.legend(['Max likelihood', '$\lambda$={:.4}'.format(rate), 'Data'])
   
df = generate_cell_data(num_samples=1000)
#g = df.groupby(['treated', 'type'])
#print g.size()
#hist2d_plots(df, var1="diameter", var2="calcium")
#hist2d_plots(df, var1="type", var2="treated")
#df.boxplot('lifetime', by='treated')
#hist_by_discrete(df)
lifetime_likelihood(df, 5.0)
plt.show()



