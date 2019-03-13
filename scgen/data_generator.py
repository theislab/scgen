from scipy.ndimage import imread
import numpy as np
import pandas as pd
import anndata
import os

data_name = "horse2zebra"
data_path = "../data/" + data_name

train_images = []
for image in os.listdir(os.path.join(data_path, "trainA")):
    if image.endswith(".jpg"):
        image = imread(fname=os.path.join(data_path, "trainA", image))
        image = np.reshape(image, newshape=(256 * 256 * 3, ))
        image = image.tolist()
        train_images.append(image)

n_horses = len(train_images)

for image in os.listdir(os.path.join(data_path, "trainB")):
    if image.endswith(".jpg"):
        image = imread(fname=os.path.join(data_path, "trainB", image))
        if len(image.shape) == 3:
            image = np.reshape(image, newshape=(256 * 256 * 3, ))
            image = image.tolist()
            train_images.append(image)

train_images = np.array(train_images)
print(train_images.shape)
n_zebras = train_images.shape[0] - n_horses

conditions = ["horse"] * n_horses
conditions += ["zebra"] * n_zebras
train_adata = anndata.AnnData(train_images, obs={"condition": conditions})
train_adata.write_h5ad(filename="../data/h2z.h5ad")



