import numpy as np
import matplotlib.pyplot as plt


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

if __name__ == '__main__':

    #generate_sine_wave()
    generate_ar_process()
    plt.show()




