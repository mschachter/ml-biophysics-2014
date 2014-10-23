import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_solar_data(num_samples=100):

    #sunlight possible in a day is gaussian distributed
    sunlight_mean = 12.5
    sunlight_std = 0.9

    #probability mass function for weather (0 = rain, 1 = cloudy, 2 = sun)
    weather_pmf = np.array([0.25, 0.25, 0.50])

    #cdf for weather
    weather_cdf = np.cumsum(weather_pmf)

    #string names of weather
    weather_names = ['rainy', 'cloudy', 'sunny']

    #effect of weather on temperature
    weather_effect_on_temp = np.array([-5.0, -2.0, 2.0])

    #temperature is gaussian distributed by affected by weather and sun
    temp_mean = 10.0
    temp_std = 3.0

    #the sunlight possible in a day has an linear effect on temp    
    temp_sun_slope = 1 / 3.0

    #probability that panel is clean
    clean_p = 0.75

    #poisson rate of rabbits
    rabbits_rate = 10

    #mean and std of energy in Watts
    energy_mean = 200.0
    energy_std = 10.0

    #weight for clean
    clean_w = 20

    #weight for weather
    weather_w = np.array([-30, -20, 30]) 
    
    #synthesize the data
    data = {'sun':list(), 'weather':list(), 'temp':list(),
            'clean':list(), 'rabbits':list(), 'energy':list()}    

    for k in range(num_samples):
        
        #generate a sample for the possible hours of sunlight for this day
        sun = np.random.randn()*sunlight_std + sunlight_mean
        
        #generate a sample for the weather for this day
        weather = np.where(weather_cdf >= np.random.rand())[0].min()
        weather_name = weather_names[weather]
        
        #generate the temperature in celcius, dependent on sun and weather
        temp = np.random.randn()*temp_std + temp_mean + temp_sun_slope*sun + weather_effect_on_temp[weather]

        #generate sample for clean or not clean
        clean = 1 - int(np.random.rand() < clean_p)
        
        #generate number of rabbits
        rabbits = np.random.poisson(rabbits_rate)

        #the effect of temperature on energy is nonlinear
        temp_effect = np.tanh(-0.1*temp) * 100.0

        #the effect of # of hours of sun on energy
        sun_effect = sun*3

        #the effect of weather on energy
        weather_effect = weather_w[weather]

        #the effect of a clean panel on energy
        clean_effect = clean_w*clean

        #generate a sample for the energy
        energy = np.random.randn()*energy_std + energy_mean + sun_effect + weather_effect + clean_effect + temp_effect

        data['sun'].append(sun)
        data['weather'].append(weather_name)
        data['temp'].append(temp)
        data['clean'].append(clean)
        data['rabbits'].append(rabbits)
        data['energy'].append(energy)

    return pd.DataFrame(data)


def example1():

    #generate the data
    df = generate_solar_data(num_samples=1000)

    #boxplot for weather vs energy
    df.boxplot('energy', by='weather')

    #boxplot for clean vs energy
    df.boxplot('energy', by='clean')

def example2():

    #generate the data
    df = generate_solar_data(num_samples=1000)

    plt.figure()
    #plot sun vs energy
    plt.subplot(1, 3, 1)
    plt.plot(df['sun'], df['energy'], 'go')
    plt.xlabel('Sun (hours)')
    plt.ylabel('Energy (Watts)')

    #plot temp vs energy
    plt.subplot(1, 3, 2)
    plt.plot(df['temp'], df['energy'], 'ro')
    plt.xlabel('Temp (degrees C)')
    plt.ylabel('Energy (Watts)')

    #plot rabbits vs energy
    plt.subplot(1, 3, 3)
    plt.plot(df['rabbits'], df['energy'], 'bo')
    plt.xlabel('# of rabbits')
    plt.ylabel('Energy (Watts)')

def one_of_k(features, feature_names=None):
    """ Transforms a numpy array of strings into one-of-k coding, where each
        string is represented by a binary vector.
    """

    if feature_names is None:
        feature_names = list(np.unique(features))
    
    encoded_features = list()
    for fname in features:
        #get the index of the feature
        findex = feature_names.index(fname)
        #create an empty binary vector  
        v = [0.0]*len(feature_names)
        #set the bit for the feature
        v[findex] = 1.0
        #append to the list
        encoded_features.append(v)

    return np.array(encoded_features),feature_names

def data_frame_to_matrix(df, dependent_column, categorical_columns=[]):
    """ Convert a pandas DataFrame to a feature matrix and target vector, for
        easy use within scikits.learn models.

        df: The pandas dataframe
        dependent_column: the name of the column that is the dependent variable
        categorical_columns: a list of column names that are categorical, the
            values of that column are re-encoded into one-of-k binary vectors.

        Returns X,y,col_names: X is a matrix of features, the number of rows
            equals the number of samples, and the number of columns is the number
            of features. y is a vector of dependent variable values. col_names is
            a string name that describes each feature.    
    """
    
    #make a list of continuous valued columns
    cont_cols = [key for key in df.keys() if key not in categorical_columns and key != dependent_column]
   
    #keep track of feature column names
    col_names = list()
    col_names.extend(cont_cols)
    
    #convert those columns to a matrix
    X = df.as_matrix(cont_cols)
    
    #convert the categorical columns
    for ccol in categorical_columns:
        #convert the values to one-of-k binary vectors
        ook,feature_names = one_of_k(df[ccol].values)
        #append the feature names
        col_names.extend(['%s_%s' % (ccol, fname) for fname in feature_names])
        #create a new extended feature matrix
        Xext = np.zeros([X.shape[0], X.shape[1]+len(feature_names)])
        Xext[:, :X.shape[1]] = X
        Xext[:, X.shape[1]:] = ook   
        X = Xext

    #create the target vector
    y = df[dependent_column].values

    return X,y,col_names

#example1()
#example2()

plt.show()

    
