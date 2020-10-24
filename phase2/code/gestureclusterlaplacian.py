import sys
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import numpy as np
import networkx
import random
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigs

# folder = sys.argv[1]
# option = sys.argv[2]  # dot PCA SVD NMF LDA edit DTW
#
# if option == '1':
#     print()
# elif option == 2:
#     print()
# elif option == 3:
#     print()
# elif option == 4:
#     print()
# elif option == 5:
#     print()
# elif option == 6:
#     print()
# elif option == 7:
#     print()
# else:
#     print('wrong clustering option')
#
testmatrix = np.zeros((10, 10))
for i in range(len(testmatrix) // 2):
    for j in range(len(testmatrix) // 2):
        testmatrix[i, j] = 1
        testmatrix[j, i] = 1
for i in range(len(testmatrix) // 2, len(testmatrix)):
    for j in range(len(testmatrix) // 2, len(testmatrix)):
        testmatrix[i, j] = 1
        testmatrix[j, i] = 1


# print(testmatrix)


def gesturecluster(matrix, k, knear=3):
    if not matrix.shape[0] == matrix.shape[1]:
        print('matrix is not a square')
    G = networkx.Graph()
    for i in range(len(matrix)):
        G.add_node(str(i))
    for i in range(len(matrix)):
        subvector = matrix[i, :]
        #print(subvector)
        nearestranking = np.argsort(subvector, )[::-1]
        knearest = nearestranking[:knear]  # k nearest
        #print(knearest)
        for j in knearest:
            if subvector[j] > 0:  # graph weight need to be positive
                G.add_edge(str(i), str(j), weight=subvector[j])  # build k-nearest graph
    # pos = networkx.spring_layout(G)
    # networkx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif")
    # networkx.draw_networkx_nodes(G, pos, node_size=400)
    # networkx.draw_networkx_edges(G, pos, edgelist=G.edges, width=4)
    # plt.show()

    normalaplacian = networkx.normalized_laplacian_matrix(G)
    kvals, kvecs = eigs(normalaplacian, k)
    V = np.mat(kvecs).real  # k*n matrix for cluster
    #print(V)
    U = np.zeros(shape=V.shape)
    for n in range(0, U.shape[0]):
        temp = np.sum(np.power(V[n, :], 2)) ** (1 / 2)
        for k1 in range(0, U.shape[1]):
            U[n, k1] = V[n, k1] / temp
    for i in range(0, U.shape[0]):
        if U[i, 0] < 0:
            U[i, :] = -U[i, :]
    print(U)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(U)
    clusterresult = kmeans.labels_
    print(clusterresult)
    return clusterresult


gesturecluster(testmatrix, 2)
