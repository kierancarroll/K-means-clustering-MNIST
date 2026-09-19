from utils import rec_load_tiny_data, rec_load_data, \
    visualize_embeddings, load_mnist_data, kmeans_loss, plot_data_encodings, display_samples, \
    plot_losses, SpriteTensorDataset, visualize_latent_grid, plot_image_points, \
    set_all_parameters, round_to_list

import matplotlib.pyplot as plt
from kmeans import kmeans
from utils import load_mnist_data


def show_mnist_kmeans_centers(C):
    K = C.shape[0]
    if K == 2:
        r, c = 1, 2
    elif K == 5:
        r, c = 1, 5
    elif K == 10:
        r, c = 2, 5
    fig, axes = plt.subplots(r, c)
    for k in range(K):
        if len(axes.shape) == 2:
            axis = axes[k//5,k%5]
        else:
            axis = axes[k]
        axis.imshow(C[k].reshape((28,28)))
    plt.show()

mnist = load_mnist_data()

#Visualization 1
X = mnist
K = 2
C, z = kmeans(X, K)
show_mnist_kmeans_centers(C)

#Visualization 2
X = mnist
K = 5
C, z = kmeans(X, K)
show_mnist_kmeans_centers(C)

#Visualization 3
X = mnist
K = 10
C, z = kmeans(X, K)
show_mnist_kmeans_centers(C)
