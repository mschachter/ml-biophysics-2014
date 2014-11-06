from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.mixture import GMM

import networkx as nx


def generate_guassmix(num_samples_per_cluster=100, plot=True, cluster_probs=(0.33, 0.33, 0.33)):
    """ Generate data from a mixture of 2D Gaussians. """

    #the centers of each Gaussian distribution
    centers = [[1.0, 1.0], [-2.5, -2], [-1.5, 2]]

    #specify the covariance matrix for each Gaussian
    cmats = ([[1.0, 0.3], [0.3, 1.0]],
             [[1.0, 2.5], [2.5, 1.0]],
             [[1.0, 0.0], [0.0, 1.0]])

    #generate random samples for each distribution
    X = list()
    y = list()
    for k,(mean,cov_mat) in enumerate(zip(centers, cmats)):
        nsamps = int(num_samples_per_cluster*cluster_probs[k])
        X.extend(np.random.multivariate_normal(mean, cov_mat, size=nsamps))
        y.extend([k]*nsamps)

    X = np.array(X)
    y = np.array(y)

    if plot:
        clusters = np.unique(y)
        plt.figure()
        for k in clusters:
            plt.plot(X[y == k, 0], X[y == k, 1], 'o')
        plt.title('Data From 3 MV Gaussians')

    return X,y


def run_kmeans(X, num_clusters=3, plot=True):

    km = KMeans(n_clusters=num_clusters)
    km.fit(X)
    ypred = km.predict(X)

    if plot:
        plt.figure()
        for k in range(num_clusters):
            plt.plot(X[ypred == k, 0], X[ypred == k, 1], 'o')
        plt.title('KMeans Result, num_clusters=%d' % num_clusters)

    return ypred


def run_gmm(X, num_clusters=3, plot=True):

    gmm = GMM(n_components=num_clusters, covariance_type='full')
    gmm.fit(X)
    ypred = gmm.predict(X)

    #print out the information for each fit cluster
    for k in range(num_clusters):
        the_mean = gmm.means_[k]
        the_cov_mat = gmm.covars_[k]
        the_cluster_prob = gmm.weights_[k]
        print 'Cluster %d' % k
        print '\tProbability: %0.2f' % the_cluster_prob
        print '\tMean: ',the_mean
        print '\tCovariance:'
        print the_cov_mat

    if plot:
        plt.figure()
        for k in range(num_clusters):
            plt.plot(X[ypred == k, 0], X[ypred == k, 1], 'o')
        plt.title('GMM Clustering, num_clusters=%d' % num_clusters)

    return ypred


def generate_social_graph(num_nodes=100, num_clusters=4, plot=True):

    g = nx.Graph()

    #create all the nodes first
    for k in range(num_nodes):
        #randomly select a cluster
        c = np.random.randint(num_clusters)

        #construct nodes
        g.add_node(k, cluster=c)

    #now connect the nodes with undirected edges, nodes
    #in the same cluster will have a stronger weight
    #than those outside of their cluster
    for n1 in range(num_nodes):
        #get the cluster for node 1
        c1 = g.node[n1]['cluster']

        for n2 in range(n1):
            #get the cluster for node 2
            c2 = g.node[n2]['cluster']

            #determine the edge weight
            if c1 == c2:
                w = np.random.rand()*3
            else:
                w = np.random.rand()
                if w < 0.9:
                    w = 0.0
            if w > 0:
                g.add_edge(n1, n2, weight=w)

    if plot:
        plt.figure()
        cluster_colors = ['r', 'g', 'b', 'y']
        node_clrs = [cluster_colors[g.node[n]['cluster']] for n in g.nodes()]
        pos = nx.spectral_layout(g, scale=100)
        weights = [g[n1][n2]['weight'] for n1,n2 in g.edges()]

        nx.draw_networkx(g, pos=pos, node_color=node_clrs, edge_cmap=cm.Greys, edge_vmin=0.0, edge_vmax=3.0, edge_color=weights)

    return g


if __name__ == '__main__':

    #X,y = generate_guassmix(num_samples_per_cluster=100, plot=True, cluster_probs=[0.15, 0.35, 0.5])
    #ypred = run_kmeans(X, num_clusters=3)
    #ypred = run_gmm(X, num_clusters=5)
    g = generate_social_graph(num_nodes=100, num_clusters=4, plot=True)



    plt.show()


