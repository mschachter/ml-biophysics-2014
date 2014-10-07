import numpy as np
import matplotlib.pyplot as plt

def evaluate_polynomial(x, coefficients):
    """ Evaluate a polynomial.

        x: a numpy.array of points to evaluate the polynomial at.
        
        coefficients: a numpy.array of coefficients. Don't forget
            the coefficient for the bias term! The degree of the
            polynomial is len(coefficients)-1.

        Returns: y, a numpy.array of values for the polynomial.
    """

    #initialize the output points to zero
    num_samples = len(x)
    y = np.zeros([num_samples])
    
    deg = len(coefficients)-1
    for k in range(deg+1):
        y += coefficients[deg-k] * x**k

    return y

def generate_polynomial(deg=3, noise_std=1e-1, num_samples=100):
    """ Generate the outputs of a noisy polynomial.

        x: A numpy.array of points on the x-axis to
            evaluate the polynomial at.

        deg: The degree of the polynomial, defaults to 1.

        noise_std: The standard deviation of the Gaussian noise
            that is added to each sample. If zero, no noise will
            be added. The higher the standard deviation, the more
            the polynomial will be drown out in noise.

        returns: coefficients,y - An array of coefficients (from
            the highest degree to lowest), and an array of points
            where the polynomial was evaluated.            
    """

    #generate the points to evaluate the polynomial
    x = np.linspace(-1, 1, num_samples)

    #generate random coefficients from a Gaussian
    #distribution that has zero mean and a standard
    #deviation of 1.
    coefficients = np.random.randn(deg+1)

    y = evaluate_polynomial(x, coefficients)

    #create and add the noise
    noise = np.random.randn(len(x))*noise_std
    y += noise

    return x,y,coefficients

def fit_and_plot(degree_of_actual=3, degree_of_fit=3,
                 num_samples=10, noise_std=1e-1):
    """ Generates a random noisy polynomial and fits it using polyfit.

        degree_of_actual: The degree of the generated polynomial.
        
        degree_of_fit: The degree passed to polyfit used to fit the
            generated polynomial.

        num_samples: The number of samples generated for the polynomial.

        noise_std: The standard deviation of the noise added to the
            generated polynomial.    
    """

    #generate a polynomial    
    x,y,coefficients = generate_polynomial(deg=degree_of_actual,
                                           noise_std=noise_std,
                                           num_samples=num_samples)
       
    #fit the polynomial
    fit_coefficients = np.polyfit(x, y, deg=degree_of_fit)

    #evaluate the fit polynomial
    fit_y = evaluate_polynomial(x, fit_coefficients)

    #create a dense set of points on which both polynomials
    #will be plotted and evaluated
    dense_x = np.linspace(-1, 1, 100)
    
    plt.figure()
    #plot the generated polynomial
    plt.plot(dense_x, evaluate_polynomial(dense_x, coefficients), 'k-')
    #plot the fit polynomial
    plt.plot(dense_x, evaluate_polynomial(dense_x, fit_coefficients), 'r-')
    #plot the data points
    plt.plot(x, y, 'bo')
    plt.axis('tight')
    #make a legend
    plt.legend(['Actual', 'Fit'])
    plt.show()

#generate a 3rd degree polynomial and plot it
#x,y,coef = generate_polynomial(deg=3, noise_std=1e-1, num_samples=100)
#plt.figure()
#plt.plot(x, y, 'k-')
#plt.axis('tight')
#plt.show()

fit_and_plot(degree_of_actual=3, degree_of_fit=3, num_samples=10, noise_std=1e-1)

