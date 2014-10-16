import numpy as np
import matplotlib.pyplot as plt


def create_blank_figure():
    plt.figure()
    plt.axhline(0.0, c='k')
    plt.axvline(0.0, c='k')
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)

def plot_linear_combination_2d(v1, v2, coef_min=-1.0, coef_max=1.0, num_points=30, marker_size=15.0):

    

    #create a figure and plot the x and y axis
    plt.figure()
    plt.axhline(0.0, c='k')
    plt.axvline(0.0, c='k')
    plt.xlim(coef_min, coef_max)
    plt.ylim(coef_min, coef_max)

    coefs = np.linspace(coef_min, coef_max, 30)

    #plot all linear combinations of vectors
    for c1 in coefs:
        for c2 in coefs:
            plt.plot(c1*v1[0] + c2*v2[0], c1*v1[1] + c2*v2[1], 'k.', ms=marker_size)

    #plot the line defined by scaling each vector   
    clrs = ['b', 'r']
    for c in coefs:
        plt.plot(c*v1[0], c*v1[1], '.', c=clrs[0], ms=marker_size)
        plt.plot(c*v2[0], c*v2[1], '.', c=clrs[1], ms=marker_size)


v1 = np.array([0.5, -0.3])
v2 = np.array([0.5, -0.2])
plot_linear_combination_2d(v1, v2, coef_min=-5.0, coef_max=5.0)
plt.show()

