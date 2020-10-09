__author__ = "shekkizh"

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from utils.plotting import Graph


def create_distance_matrix(X, p=2):
    """
    Create distance matrix
    :param X:
    :param metric:
    :return:
    """
    n = X.shape[0]
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            W[i, j] = lp_distance(X[i, :], X[j, :], p)
    W = W + W.T
    return W


def create_distance_to_Y_matrix(X, Y=None, p=2):
    """
    Create distance matrix
    :param X:
    :param Y:
    :param metric:
    :return:
    """
    if Y is None:
        Y = X
    W = np.zeros((X.shape[0], Y.shape[0]))
    for i in range(X.shape[0]):
        for j in range(0, Y.shape[0]):
            W[i, j] = lp_distance(X[i, :], Y[j, :], p)
    return W


def create_directed_KNN_mask(D, knn_param=10, D_type='distance'):
    if D_type == 'similarity':
        np.fill_diagonal(D, - np.inf)
        directed_KNN_mask = np.argpartition(D, -knn_param, axis=1)[:, -knn_param:]
    else:
        np.fill_diagonal(D, np.inf)
        directed_KNN_mask = np.argpartition(D, knn_param + 1, axis=1)[:, 0:knn_param]
    return directed_KNN_mask


def plot_graph(W, X, filename=None, vertex_color=(0.12, 0.47, 0.71, 0.5), vertex_size=None, colorbar=True, ax=None):
    """
    Plot graph.
    :param W: Adjacency matrix
    :param X: Coordinate location for each node
    :param filename: filename to save graph fig in
    :param vertex_size: size of the nodes (can be used to highlight certain nodes)
    :param vertex_color: color of the nodes (another modifiable parameter to highlight nodes)
    :return: No return value
    """
    g = Graph(W, coords=X)
    if ax is None:
        f = plt.figure()
        if X.shape[1] == 3:
            ax = f.gca(projection='3d')
        else:
            ax = f.gca()

    ax.grid(False)
    ax.axis('off')
    ax.axis('equal')
    ax.set_title('')
    _, _, weights = g.get_edge_list()
    g.plot(vertex_color=vertex_color, vertex_size=vertex_size, edge_width=weights,
           ax=ax, title='', colorbar=colorbar)  # , show_edges=False
    plt.tight_layout()
    # plt.show()
    if filename is not None:
        plt.savefig(filename + '_graph.pdf', bbox_inches='tight')


def lp_distance(pointA, pointB, p):
    """
    Function to calculate the lp distance between two points
    :param p: the norm type to  calculate
    :return: distance
    """
    dist = (np.sum(np.abs(pointA - pointB) ** p)) ** (1.0 / p)

    return dist


def to_categorical(y):
    """Converts a class vector (integers) to binary class matrix.
    Code modified from tensorflow keras
    """
    y = np.array(y, dtype=np.int)
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()

    num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=np.float)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


class Create_Data:
    """
        Create blobs of data in a given space.
        :param n_samples: no. of samples to generate
        :param n_dim: dimension of each sample
        :param n_clusters: no. of clusters present in dataset
        :return: data points X, labels y
    """

    def __init__(self, n_samples=100, n_dim=20, n_clusters=2):
        self.n_samples = n_samples
        self.n_dim = n_dim
        self.n_clusters = n_clusters
        self.random_state = 4629
        self.dataset_func = {"gauss_mixture": self.make_blobs,
                             "classification": self.make_classification,
                             "circles": self.make_circles,
                             "two_moons": self.make_two_moons}

    def get_dataset(self, dataset):
        if dataset not in self.dataset_func.keys():
            raise Exception("Unknown dataset: " + dataset)
        return self.dataset_func[dataset]()

    def make_classification(self):
        """
        creates dataset from hyper cube
        :return:
        """
        return datasets.make_classification(n_samples=self.n_samples, n_features=self.n_dim, n_classes=self.n_clusters,
                                            n_informative=2,
                                            n_redundant=0,
                                            class_sep=5,
                                            shuffle=True,
                                            random_state=self.random_state)

    def make_circles(self):
        """
        create concentric circle data
        :return:
        """
        return datasets.make_circles(n_samples=self.n_samples, shuffle=True, noise=0.1, random_state=self.random_state,
                                     factor=0.3)

    def make_blobs(self):
        """
        creates data from a Gaussian mixture
        :return:
        """
        return datasets.make_blobs(n_samples=self.n_samples, n_features=self.n_dim,
                                   centers=np.random.uniform(-5, 5, size=(self.n_clusters, self.n_dim)),
                                   center_box=(-1, 1), shuffle=True, random_state=self.random_state)

    def make_two_moons(self):
        return datasets.make_moons(n_samples=self.n_samples, noise=0.1, random_state=self.random_state)
