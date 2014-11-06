import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge


def generate_sine_wave(freqs=[1.0, 2.0, 5.0], duration=5.0, sample_rate=1e3, plot=True):

    #generate vector that represents time
    num_samps = int(duration*sample_rate)
    t = np.arange(num_samps) / sample_rate

    #generate sine wave
    y = np.zeros([len(t)])
    for freq in freqs:
        y += np.sin(2*np.pi*t*freq)

    if plot:
        #plot the sine wave
        plt.figure()
        plt.plot(t, y, 'c-', linewidth=2.0)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')

    return t,y


def generate_ar_process(weights=[-0.3, 0.5], duration=0.050, sample_rate=1e3, plot=True):

    #generate vector that represents time
    num_samps = int(duration*sample_rate)
    t = np.arange(num_samps) / sample_rate

    #generate the series
    y = np.zeros([num_samps])
    #generate a random starting point
    y[0] = np.random.randn()
    for k in range(1, num_samps):
        #determine the number of previous time points available
        nw = min(k+1, len(weights))
        #multiply each weight by the previous point in the series
        for j in range(nw):
            y[k] += y[k-(j+1)]*weights[j]

    if plot:
        plt.figure()
        plt.plot(t, y, 'g-', linewidth=2.0)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')

    return t,y


def run(transition_function, initial_value, nsteps=100):
    """ Simulate a system using a difference equation.

        transition_function: The right hand side of the difference equation.
        initial_value: The starting value for the state.
        nsteps: The number of steps to run the system for.
    """

    x = np.zeros([nsteps])
    x[0] = initial_value
    for k in range(1, nsteps):
        x[k] = transition_function(x[k-1])

    return x


def difference_equation_examples(logistic_map_r=3.86):
    """ Show some examples of difference equations.

        logistic_map_r: The value of r to give the logistic
            map in the third plot. Defaults to 3.86, which
            creates chaotic dynamics.
    """

    plt.figure()

    plt.subplot(3, 1, 1)
    x = run(np.cos, initial_value=1.0, nsteps=20)
    plt.plot(x, 'k-')
    plt.axis('tight')
    plt.title('$x_{t+1} = cos(x_t)$')

    plt.subplot(3, 1, 2)
    r = 3.57
    logistic_map = lambda x: r*x*(1.0 - x)
    x = run(logistic_map, initial_value=0.5, nsteps=75)
    plt.plot(x, 'k-')
    plt.axis('tight')
    plt.title('$x_{t+1} = %0.6f x_t (1 - x_t)$' % r)

    plt.subplot(3, 1, 3)
    r = logistic_map_r
    logistic_map = lambda x: r*x*(1.0 - x)
    x = run(logistic_map, initial_value=0.5, nsteps=175)
    plt.plot(x, 'k-')
    plt.axis('tight')
    plt.title('$x_{t+1} = %0.6f x_t (1 - x_t)$' % r)


def plot_power_spectrum(x, sample_rate=1.0):

    #take the fourier transform of the time series x
    xft = np.fft.fft(x)
    freq = np.fft.fftfreq(len(x), d=1.0/sample_rate)
    findex = freq > 0.0

    #square the magnitude of the fourier transform to get
    #the power spectrum
    ps = np.abs(xft)**2

    #make a plot
    plt.figure()
    plt.plot(freq[findex], ps[findex], 'g-')
    plt.ylabel('Power')
    plt.xlabel('Frequency')


def autocorrelation_function(y, lags=range(20)):
    """ Compute the autocorrelation function for the time series y
        at the given lags.
    """

    acf = np.zeros([len(lags)])
    for k,lag in enumerate(lags):
        #compute the correlation coefficient between y and lagged y
        C = np.corrcoef(y[:len(y)-lag], y[lag:])
        acf[k] = C[0, 1]

    return acf,lags


def get_system_data(nsteps=500):
    """ Generate data from linearly filtered smoothed Gaussian noise input. """

    #generate random noise input
    input = np.random.randn(nsteps)

    #smooth the noise with a hanning window
    h = np.hanning(30)
    input_smooth = np.convolve(input, h, mode='same')

    #normalize the input so it's between -1 and 1
    input_smooth /= np.abs(input_smooth).max()

    ##generate the output by convolving with some sort of oscillating filter
    the_filter = [1.0, 0.5, 0.0, -0.5, -1.0, -0.25, 0.0, 0.25, 0.0, -0.05, 0.0, 0.05]
    intercept = 0.7
    y = np.convolve(input_smooth, the_filter, mode='same') + intercept

    return input_smooth,y,the_filter,intercept


def fit_linear_filter(input, output, nlags):
    """ Fit the weights of a linear filter with nlags between
        the input time series and the output time series,
    """

    #generate data matrix
    X = list()
    for k in range(nlags, len(output)):
        X.append(input[k-nlags:k])
    X = np.array(X)

    #generate target vector
    y = output[nlags:]

    #do a ridge regression
    rr = Ridge(alpha=1)
    rr.fit(X, y)

    #return the filter weights and the bias
    return rr.coef_[::-1],rr.intercept_


if __name__ == '__main__':

    #t,y = generate_sine_wave()
    #t,y = generate_ar_process(sample_rate=5e3)
    #difference_equation_examples()

    #sr = 1e3
    #t,y = generate_sine_wave(freqs=[2.0, 57.0, 143.0], duration=5.0, sample_rate=1e3)
    #plot_power_spectrum(y, sample_rate=1e3)

    """
    plt.figure()
    #plot the ACF of random noise
    y = np.random.randn(1000)
    plt.subplot(2, 2, 1)
    acf,lags = autocorrelation_function(y, lags=range(50))
    plt.plot(lags, acf, 'k-', linewidth=2.0)
    plt.title('ACF of Random Noise')
    plt.axis('tight')

    #plot the ACF of a sine wave
    t,y = generate_sine_wave(freqs=[2.0, 57.0, 143.0], duration=5.0, sample_rate=1e3, plot=False)
    plt.subplot(2, 2, 2)
    acf,lags = autocorrelation_function(y, lags=range(50))
    plt.plot(lags, acf, 'k-', linewidth=2.0)
    plt.title('ACF of Sum of Sine Wave')
    plt.axis('tight')

    #plot the ACF of the logistic map
    logistic_map = lambda x: 3.86*x*(1.0 - x)
    y = run(logistic_map, initial_value=0.5, nsteps=500)
    acf,lags = autocorrelation_function(y, lags=range(100))
    plt.subplot(2, 2, 3)
    plt.plot(lags, acf, 'k-', linewidth=2.0)
    plt.title('ACF of Logistic Map (r=3.86)')
    plt.axis('tight')

    #plot the ACF of another logistic map
    logistic_map = lambda x: 3.7*x*(1.0 - x)
    y = run(logistic_map, initial_value=0.5, nsteps=500)
    acf,lags = autocorrelation_function(y, lags=range(100))
    plt.subplot(2, 2, 4)
    plt.plot(lags, acf, 'k-', linewidth=2.0)
    plt.title('ACF of Logistic Map (r=3.7)')
    plt.axis('tight')
    """

    #r = 3.86
    #logistic_map = lambda x: r*x*(1.0 - x)
    #y = run(logistic_map, initial_value=0.5, nsteps=2000)
    #plot_power_spectrum(y, sample_rate=1.0)


    #generate the input/output data
    input,output,the_filter,the_intercept = get_system_data()

    #fit a linear filter to the input/output data
    pred_filter,pred_intercept = fit_linear_filter(input, output, 12)

    #generate a predicted output from the input using a convolution
    pred_output = np.convolve(input, pred_filter, mode='same') + pred_intercept

    #compute the correlation coefficient between the predicted and actual output
    C = np.corrcoef(output, pred_output)
    cc = C[0, 1]

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(input, 'k-', linewidth=2.0)
    plt.plot(output, 'r-', linewidth=2.0)
    plt.xlabel('Time')
    plt.legend(['Input', 'Output'])
    plt.axis('tight')

    plt.subplot(2, 2, 2)
    plt.plot(the_filter, 'bo-', linewidth=2.0)
    plt.plot(pred_filter, 'co-', linewidth=2.0)
    plt.xlabel('Lag')
    plt.legend(['Actual', 'Predicted'])
    plt.title('Filters')
    plt.axis('tight')

    plt.subplot(2, 2, 3)
    plt.plot(output, 'r-', linewidth=2.0)
    plt.plot(pred_output, 'g-', linewidth=2.0)
    plt.legend(['Actual Output', 'Predicted Output'])
    plt.xlabel('Time')
    plt.axis('tight')
    plt.title('cc=%0.2f' % cc)

    plt.show()






