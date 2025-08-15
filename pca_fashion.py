from tests import load_fashion_mnist
import logging
from pca_digits import draw_pca_maps, show_components

def make_figs():
    _, (images, labels) = load_fashion_mnist()

    draw_pca_maps(images, labels,dataset='Fashion')
    show_components(images, labels,dataset='Fashion')


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    make_figs()
