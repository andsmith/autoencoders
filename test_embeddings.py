from posixpath import sep
from embeddings import PassThroughEmbedding, PCAEmbedding, UMAPEmbedding, TSNEEmbedding
import logging
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')

from util import make_test_data


class TestDataset(object):
    def __init__(self, d=3, n_points=20000, n_clusters=10, separability=3.0):
        self.d = d
        self.n_points = n_points
        self.n_clusters = n_clusters
        self.separability = separability
        self.points, self.labels = make_test_data(d, n_points, n_clusters, separability)
        logging.info(f"Created test dataset with {n_points} points in {d}-dimensional space, "
                     f"{n_clusters} clusters, separability {separability}")


def test_embedding(embedding_class, dataset,ax=None):
    embedding = embedding_class()
    points_2d  = embedding.fit_embed(dataset.points)
    if ax is not None:
    # Plot the embedded points
    # dark mode
        colors = plt.cm.gist_ncar(np.linspace(0, 1, dataset.n_clusters))
        for i in range(dataset.n_clusters):
            cluster_points = points_2d[dataset.labels == i]
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[i], alpha=0.2, label=f'Cluster {i}')
        ax.set_title(f"{embedding_class.__name__}" ,fontsize=12)
        ax.set_xlabel("Embedded X")
        ax.set_ylabel("Embedded Y")
        ax.axis('equal')

    logging.info(f"Test passed for {embedding_class.__name__} with {dataset.n_points} points in {dataset.d}-dimensional space.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    dataset = TestDataset(d=10, n_points=500, n_clusters=10, separability=1.0)
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    test_embedding(PassThroughEmbedding, dataset, ax=ax[0,0])
    test_embedding(PCAEmbedding, dataset, ax=ax[0,1])
    test_embedding(UMAPEmbedding, dataset, ax=ax[1,0])
    test_embedding(TSNEEmbedding, dataset, ax=ax[1,1])
    plt.suptitle("Embedding Test for random dataset \n"
                 f"Dataset: {dataset.n_points} points, {dataset.d}-D, {dataset.n_clusters} clusters, "
                 f"separability {dataset.separability}", fontsize=14)
    plt.show()
    # Add tests for other embedding classes as needed
    # test_embedding(UMAPEmbedding, n_points=20000, d=42, n_clusters=10)
    # test_embedding(TSNEEmbedding, n_points=20000, d=42, n_clusters=10)
    # test_embedding(MDSEmbedding, n_points=20000, d=42, n_clusters=10)
    # test_embedding(PCAEmbedding, n_points=20000, d=42, n_clusters=10)
