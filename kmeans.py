import numpy as np

def kmeans(X, K):
    """ Cluster data X into K converged clusters.

        X: an N-by-M numpy ndarray, where we want to assign each
            of the N data points to a cluster.

        K: an integer denoting the number of clusters.

        Returns a tuple of length two containing (C, z):
            C: a numpy ndarray with shape (K,M), where each row is a cluster center
            z: a numpy ndarray with shape (N,) where the i-th entry is an int from {0..K-1}
                representing the cluster index for the i-th point in X
    """
    N = X.shape[0]

    # Initialize cluster centers to the first K points of X
    C = np.copy(X[:K])

    # Initialize z temporarily to all -1 values
    z = -1*np.ones(N, dtype=np.int32)

    t = -2*np.ones(N, dtype=np.int32)

    while True:
      for i in range(N):
        distances = np.linalg.norm(C - X[i], axis=1)
        t[i] = np.argmin(distances)
      if np.array_equal(t, z):
        break
      z = np.copy(t)
      for i in range(K):
        C[i] = np.mean(X[z == i], axis=0)

    return (C, z)