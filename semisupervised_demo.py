# Demonstration of semi-supervised learning
import numpy as np
import graphlearning as gl
import matplotlib.pyplot as plt
import sklearn.datasets as datasets
import utils.graph_utils as utils
from utils.graph_construction import knn_graph, nnk_graph

# data
n = 500
# X, L = datasets.make_moons(n_samples=n, noise=0.1)
X, L = datasets.make_circles(n_samples=n, noise=0.075, factor=0.5)
# X,L = datasets.make_blobs(n_samples=n, cluster_std=[1,1.5,0.5])

# Build graph
k = 10
# I, J, D = gl.knnsearch(X, k)
# W = gl.weight_matrix(I, J, D, k)
D = utils.create_distance_matrix(X=X, p=2)
knn_mask = utils.create_directed_KNN_mask(D=D.copy(), knn_param=k, D_type='distance')
sigma = np.mean(D[:, knn_mask[:, -1]]) / 2
G = np.exp(-(D ** 2) / (sigma ** 2))

W_knn = knn_graph(G, knn_mask, k, 1e-10, directed=True)
W_nnk = nnk_graph(G, knn_mask, k, 1e-10, directed=True)

# Randomly choose labels
m = 2  # 5 labels per class
ind = gl.randomize_labels(L, m)  # indices of labeled points

# Semi-supervised learning
l_knn = gl.graph_ssl(W_knn, ind, L[ind], method='poissonmbo', symmetrize=False)
knn_acc = gl.accuracy(l_knn, L, m)
l_knn[ind] = 2

l_nnk = gl.graph_ssl(W_nnk, ind, L[ind], method='poissonmbo', symmetrize=False)
nnk_acc = gl.accuracy(l_nnk, L, m)
l_nnk[ind] = 2

# Plot result (red points are labels)
fig = plt.figure(figsize=(12, 6))
ax = plt.subplot(1, 2, 1)
utils.plot_graph(W_knn, X, vertex_color=l_knn, ax=ax)
ax.set_title(f"KNN graph, Acc:{knn_acc}")

ax = plt.subplot(1, 2, 2)
utils.plot_graph(W_nnk, X, vertex_color=l_nnk, ax=ax)
ax.set_title(f"NNK graph, Acc:{nnk_acc}")

plt.show()
