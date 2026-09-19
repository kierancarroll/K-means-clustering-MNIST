import numpy as np


def load_mnist_data():
    """ Load MNIST dataset

        Returns an 30000-by-784 numpy ndarray X_train
    """

    X = np.load('mnist.npy').astype(np.float32)
    X_train = np.reshape(X[:30000][:][:], (30000, 784))
    return X_train


def kmeans_loss(X, C, z):
    """ Compute the K-means loss.

        Input:
        X: a numpy ndarray with shape (N,M), where each row is a data point
        C: a numpy ndarray with shape (K,M), where each row is a cluster center
        z: a numpy ndarray with shape (N,) where the i-th entry is an int from {0..K-1}
            representing the cluster index for the i-th point in X

        Returns mean squared distance from each point to the center for its assigned cluster
    """
    N = X.shape[0]
    loss = 0
    for i in range(N):
        diff = X[i] - C[z[i]]
        loss += diff @ diff / N

    return loss

