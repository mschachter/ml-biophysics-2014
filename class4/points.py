import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def create_blank_figure2d():
    plt.figure()
    plt.axhline(0.0, c='k')
    plt.axvline(0.0, c='k')
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)

def plot_point2d(x):
    assert len(x) == 2, "plot_point only works with 2D points"
    plot(x[0], x[1], 'o')

def create_blank_figure3d():    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_ylabel('Z')

def plot_point3d(x):
    assert len(x) == 3, "plot_point3d only works with 3d points"
    ax = plt.gca()
    ax.scatter(x[0], x[1], x[2], 'o')
    plt.draw()

def plot_linear_combination2d(v1, v2, coef_min=-5.0, coef_max=5.0, num_points=30, marker_size=15.0):
    #create a blank figure
    create_blank_figure2d()

    #compute the dot product
    dp = np.dot(v1, v2)
    plt.title('dot product={:.6f}'.format(dp))

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

def plot_linear_combination3d(v1, v2, v3, coef_min=-5.0, coef_max=5.0, num_points=15, marker_size=15.0):

    #create a blank figure
    create_blank_figure3d()

    coefs = np.linspace(coef_min, coef_max, num_points)

    #compute all linear combinations of vectors
    all_points = list()
    for c1 in coefs:
        for c2 in coefs:
            for c3 in coefs:
                all_points.append([c1*v1[0] + c2*v2[0] + c3*v3[0],
                                   c1*v1[1] + c2*v2[1] + c3*v2[1],
                                   c1*v1[2] + c2*v2[2] + c3*v3[2]])
    all_points = np.array(all_points)

    #plot all the points
    ax = plt.gca()
    ax.scatter(all_points[:, 0], all_points[:, 1], all_points[:, 2], c='k', s=marker_size, alpha=0.5)

    #plot the line defined by scaling each vector   
    clrs = ['b', 'r', 'g']
    coefs_dense = np.linspace(coef_min, coef_max, 100)
    for c in coefs_dense:
        ax.scatter(c*v1[0], c*v1[1], c*v1[2], '.', c=clrs[0], s=marker_size)
        ax.scatter(c*v2[0], c*v2[1], c*v2[2], '.', c=clrs[1], s=marker_size)
        ax.scatter(c*v3[0], c*v3[1], c*v3[2], '.', c=clrs[2], s=marker_size)


#v1 = np.array([1.0, 0.5])
#v2 = np.array([0.0, 1.0])
#plot_linear_combination2d(v1, v2)

v1 = np.array([1, 0, 0])
v2 = np.array([0, 1, 0])
v3 = np.array([0, 0.5, 0.25])
plot_linear_combination3d(v1, v2, v3)
plt.show()

