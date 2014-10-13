import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_cell_data(num_samples=100):

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
        
    #mean lifetime of cell in minutes as a function of cell type
    mean_lifetime_by_type = np.array([1.0, 1.5, 2.0, 4.0])

    #create a dictionary of data columns to be turned into a DataFrame
    data = {'treated':list(), 'type':list(), 'diameter':list(), 'calcium':list(), 'lifetime':list()}
    for n in range(num_samples):
        #sample whether or not the cell was treated
        treated = np.random.rand() > p_treated
        
        #sample the cell type
        cell_type = np.where(type_cdf >= np.random.rand())[0].min()
        
        #sample the diameter
        dia = np.random.randn()*diameter_std + diameter_mean
        
        #sample the calcium concentration
        ca = ca_conc(dia, treated)

        #normalize calcium concentration by diameter
        ca_norm = (ca*6.0) / dia

        #sample cell lifetime
        lifetime = np.random.exponential(mean_lifetime_by_type[cell_type] / ca_norm)

        #append the data
        data['treated'].append(treated)
        data['type'].append(cell_type)    
        data['diameter'].append(dia)
        data['calcium'].append(ca)
        data['lifetime'].append(lifetime)
    
    return pd.DataFrame(data)


