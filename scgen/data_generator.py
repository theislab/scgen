from copy import deepcopy

from scipy.ndimage import imread
import numpy as np
import pandas as pd
import anndata
import os

# data_name = "horse2zebra"
# data_path = "../data/" + data_name
#
# train_images = []
# for image in os.listdir(os.path.join(data_path, "trainA")):
#     if image.endswith(".jpg"):
#         image = imread(fname=os.path.join(data_path, "trainA", image))
#         image = np.reshape(image, newshape=(256 * 256 * 3, ))
#         image = image.tolist()
#         train_images.append(image)
#
# n_horses = len(train_images)
#
# for image in os.listdir(os.path.join(data_path, "trainB")):
#     if image.endswith(".jpg"):
#         image = imread(fname=os.path.join(data_path, "trainB", image))
#         if len(image.shape) == 3:
#             image = np.reshape(image, newshape=(256 * 256 * 3, ))
#             image = image.tolist()
#             train_images.append(image)
#
# train_images = np.array(train_images)
# print(train_images.shape)
# n_zebras = train_images.shape[0] - n_horses
#
# conditions = ["horse"] * n_horses
# conditions += ["zebra"] * n_zebras
# train_adata = anndata.AnnData(train_images, obs={"condition": conditions})
# train_adata.write_h5ad(filename="../data/h2z.h5ad")
#
#
#

import scanpy as sc
# mnist_data = sc.read("../data/normal_thick.h5ad")
# mnist_data = mnist_data.copy()[mnist_data.obs["condition"] == "normal"]
# mnist_data.obs["condition"] = mnist_data.copy().obs["labels"].values.astype(dtype=np.str)
# # mnist_data.obs["labels"] = np.reshape(mnist_data.obs["labels"].values, (-1, 1))
# mnist_data.write_h5ad("../data/mnist.h5ad")

# mnist_data = sc.read("../data/mnist.h5ad")
# mnist_data.obs["condition"] = mnist_data.obs["condition"].astype(np.str)
# print(mnist_data)
# print(mnist_data.obs)
# print(mnist_data[mnist_data.obs["labels"] == 2])

def load_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels

x_train, y_train = load_mnist(path="../data/fashion/")

fashion_adata = anndata.AnnData(X=x_train, obs={"condition": y_train, "labels": y_train})
print(fashion_adata)
fashion_adata.write_h5ad("../data/fashion.h5ad")