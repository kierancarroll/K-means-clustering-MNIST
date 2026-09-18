from utils import kmeans_loss, load_mnist_data
from kmeans import kmeans
import numpy as np

def test_kmeans(arguments, expected, full=False):
    X = arguments["X"]
    K = arguments["K"]
    exp_loss = expected["loss"]
    threshold = 5 if full else 1e-3

    C, z = kmeans(X, K)
    loss = kmeans_loss(X, C, z)

    assert abs(loss - exp_loss) < threshold, 'Incorrect loss found. Expected {}, found {}'.format(exp_loss, loss)

mnist = load_mnist_data()

#Test 1
X = np.array([[1, 1], [11, 1], [1, 11], [11,11],
              [3, 3], [13, 3], [3, 13], [13,13]], dtype=np.float32)
K = 4
arguments = {"X": X, "K": K}
expected = {"loss": 2.0}
test_kmeans(arguments, expected)

#Test 2
X = np.array([[1, 1], [3, 1], [1, 3], [3,3],
              [11, 1], [1, 11], [11, 11], [13,13]], dtype=np.float32)
K = 4
arguments = {"X": X, "K": K}
expected = {"loss": 1.5}
test_kmeans(arguments, expected)

#Test 3
X = mnist[:10]
K = 10
arguments = {"X": X, "K": K}
expected = {"loss": 0}
test_kmeans(arguments, expected)

#Test 4
X = mnist[:1000,:]
K = 2
arguments = {"X": X, "K": K}
expected = {"loss": 3222354.07050}
test_kmeans(arguments, expected, full=True)

#Test 5
X = mnist[:5000,:]
K = 5
arguments = {"X": X, "K": K}
expected = {"loss": 2824418.06335}
test_kmeans(arguments, expected, full=True)