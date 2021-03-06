import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.cross_validation import train_test_split

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
        clean = int(np.random.rand() < clean_p)
        
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

def example3():

    #generate the dataset 
    df = generate_solar_data(num_samples=1000)

    #convert Pandas DataFrame to a feature matrix
    X,y,col_names = data_frame_to_matrix(df, 'energy', ['weather'])

    #split into training and test sets
    Xtrain,Xtest,ytrain,ytest = train_test_split(X, y, test_size=0.5)


def example4():
    #generate the dataset 
    df = generate_solar_data(num_samples=1000)

    #convert Pandas DataFrame to a feature matrix
    X,y,col_names = data_frame_to_matrix(df, 'energy', ['weather'])

    #split into training and test sets
    Xtrain,Xtest,ytrain,ytest = train_test_split(X, y, test_size=0.5)

    #create a Ridge object
    rr = Ridge()

    #fit the training data
    rr.fit(Xtrain, ytrain)

    #print out the weights and their names
    for weight,cname in zip(rr.coef_, col_names):
        print "{}: {:.6f}".format(cname, weight)
    print "Intercept: {:.6f}".format(rr.intercept_)

    #print out the R-squared on the test set
    r2 = rr.score(Xtest, ytest)
    print "R-squared: {:.2f}".format(r2)


def run_full_example(df, ridge_alpha=1.0, test_set_fraction=0.5):
    
    #convert Pandas DataFrame to a feature matrix
    X,y,col_names = data_frame_to_matrix(df, 'energy', ['weather'])

    #split into training and test sets
    Xtrain,Xtest,ytrain,ytest = train_test_split(X, y, test_size=test_set_fraction)
    print '# of training samples: {}'.format(len(ytrain))
    print '# of test samples: {}'.format(len(ytest))
    print 'alpha: {:.2f}'.format(ridge_alpha)    
    print ''

    #create a Ridge object
    rr = Ridge(alpha=ridge_alpha)

    #fit the training data
    rr.fit(Xtrain, ytrain)

    #print out the weights and their names
    for weight,cname in zip(rr.coef_, col_names):
        print "{}: {:.6f}".format(cname, weight)
    print "Intercept: {:.6f}".format(rr.intercept_)
    print ''

    #compute the prediction on the test set
    ypred = rr.predict(Xtest)

    #compute the sum-of-squares error on the test set, which is
    #proportional to the log likelihood
    sqerr = np.sum((ytest - ypred)**2) / len(ytest)
    print 'Normalized Sum-of-squares Error: {:.3f}'.format(sqerr)

    #compute the sum-of-squares error for a model that is just
    #comprised of the mean on the training set
    sqerr_mean_only = np.sum((ytest - ytrain.mean())**2) / len(ytest)
    print 'Normalized Sum-of-squares Error for mean-only: {:.3f}'.format(sqerr_mean_only)

    #print out the R-squared on the test set
    r2 = rr.score(Xtest, ytest)
    print "R-squared: {:.2f}".format(r2)
    print ''    


def run_cross_validation(df, alphas_to_try=[0.1, 0.5, 1.0, 5.0, 25.0], nfolds=10):
    """ Use k-fold cross validation to fit the weights of the solar panel data, and
        also to determine the optimal ridge parameter.        
    """

    #import KFold from scikits
    from sklearn.cross_validation import KFold

    #keep track of the mean and std of the error for each ridge parameter
    mean_error_per_ridge_param = list()
    std_error_per_ridge_param = list()

    #keep track of the mean weights and intercept for each ridge parameter
    weights_per_ridge_param = list()
    intercept_per_ridge_param = list()

    #convert Pandas DataFrame to a feature matrix
    X,y,col_names = data_frame_to_matrix(df, 'energy', ['weather'])
    
    #run k-fold cross validation for each ridge parameter
    for ridge_alpha in alphas_to_try:
        
        #keep track of the weights and intercept computed on each fold
        weights = list()
        intercepts = list()

        #keep track of the errors on each fold
        errs = list()

        for train_indices,test_indices in KFold(len(df), n_folds=nfolds):
            #break the data matrix up into training and test sets
            Xtrain, Xtest, ytrain, ytest = X[train_indices], X[test_indices], y[train_indices], y[test_indices]
    
            #create a Ridge object
            rr = Ridge(alpha=ridge_alpha)

            #fit the training data
            rr.fit(Xtrain, ytrain)

            #record the weights and intercept
            weights.append(rr.coef_)
            intercepts.append(rr.intercept_)

            #compute the prediction on the test set
            ypred = rr.predict(Xtest)

            #compute and record the sum-of-squares error on the test set
            sqerr = np.sum((ytest - ypred)**2) / len(ytest)
            errs.append(sqerr)
        
        #compute the mean weight and intercept
        weights = np.array(weights)
        mean_weights = weights.mean(axis=0)
        std_weights = weights.std(axis=0, ddof=1)
        intercepts = np.array(intercepts)
        mean_intercept = intercepts.mean()
        std_intercept = intercepts.std(ddof=1)

        #compute the mean and std of the test error
        errs = np.array(errs)
        mean_err = errs.mean()
        std_err = errs.std(ddof=1)

        #print out some information
        print 'ridge_alpha={:.2f}'.format(ridge_alpha)
        print '\t Test error: {:.3f} +/- {:.3f}'.format(mean_err, std_err)
        print '\t Weights:'    
        for mean_weight,std_weight,cname in zip(mean_weights, std_weights, col_names):
            print "\t\t{}: {:.3f} +/- {:.3f}".format(cname, mean_weight, std_weight)
        print "\tIntercept: {:.3f} +/- {:.3f}".format(mean_intercept, std_intercept)
        print ''
        
        #record the mean weight and intercept
        weights_per_ridge_param.append(mean_weights)
        intercept_per_ridge_param.append(mean_intercept)
        
        #record the errors
        mean_error_per_ridge_param.append(mean_err)
        std_error_per_ridge_param.append(std_err)

    #identify the best ridge param
    best_index = np.argmin(mean_error_per_ridge_param)
    best_alpha = alphas_to_try[best_index]
    best_err = mean_error_per_ridge_param[best_index]
    best_weights = weights_per_ridge_param[best_index]
    best_intercept = intercept_per_ridge_param[best_index]
      

df = generate_solar_data(num_samples=1000) 
run_cross_validation(df)

#example1()
#example2()
#example3()
#example4()
#run_full_example()

plt.show()

    
