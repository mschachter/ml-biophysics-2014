import numpy as np
import matplotlib.pyplot as plt

def example1():
    #create an array of points to evaluate the line
    x = np.arange(-5, 5, 1e-2)

    #set the value of the slope and intercept
    slopes = [-1.0, 0.0, 1.0, 2.0]
    intercepts = [0.25, -0.5, 1.25, 0.6]

    #open up a blank figure
    plt.figure()
    #plot a line for each slope and intercept value
    for m,b in zip(slopes, intercepts):
        #compute the value of the line at each point
        y = m*x + b
        #make a plot of the line
        plt.plot(x, y, linewidth=2.0)    
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.axis('tight')
    plt.show()

def example2():

    #create an array of points to evaluate the line
    x = np.arange(-5, 6, 1.0)

    #get the number of samples and print it
    num_samples = len(x)
    print("# of samples: {0}".format(num_samples))

    #set the value of the slope and intercept
    m = 2.0
    b = -0.9

    #create some Gaussian random noise that 
    noise = np.random.randn(num_samples)

    #create a noiseless line
    y = m*x + b
    #create a noisy line
    ynoisy = m*x + b + noise

    #open up a blank figure
    plt.figure()
    #plot the non-noisy line
    plt.plot(x, y, 'k-', linewidth=2.0)
    #plot the noisy data
    plt.plot(x, ynoisy, 'go')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.axis('tight')

    #now fit the noisy line using polyfit
    fitted_slope, fitted_intercept = np.polyfit(x, ynoisy, deg=1)

    print("Fitted slope={0:.2f}, Predicted slope={1:.2f}".format(fitted_slope, m))
    print("Fitted intercept={0:.2f}, Predicted intercept={1:.2f}".format(fitted_intercept, b))

    #compute the line predicted by polyfit
    ypredicted = fitted_slope*x + fitted_intercept

    #plot the actual line, the predicted line, and the data points
    plt.figure()
    #plot the non-noisy line
    plt.plot(x, y, 'k-', linewidth=2.0)
    #plot the predicted line
    plt.plot(x, ypredicted, 'r-')
    #plot the noisy data
    plt.plot(x, ynoisy, 'go', linewidth=2.0, alpha=0.75)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.axis('tight')
    plt.show()

example2()

