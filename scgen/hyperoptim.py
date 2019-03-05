from __future__ import print_function

import os

import numpy as np
import scanpy as sc
from hyperas import optim
from hyperas.distributions import choice
from hyperopt import Trials, STATUS_OK, tpe

import scgen


def data():
    """
    Data providing function:

    This function is separated from create_model() so that hyperopt
    won't reload data for each evaluation run.
    """
    x_train = sc.read("../data/train.h5ad")
    return x_train


def create_model(x_train):
    """
    Model providing function:

    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    """
    network = scgen.MMDCVAE(x_dimension=x_train.X.shape[1], z_dimension={{choice([10, 20, 50, 75, 100])}},
                            alpha={{choice([0.1, 0.01, 0.001])}}, beta={{choice([1, 10, 100, 1000])}},
                            batch_mmd=True, kernel={{choice(["multi-scale-rbf", "rbf"])}}, train_with_fake_labels=False,
                            model_path=f"./")
    result = network.train(x_train,
                           n_epochs={{choice([500, 1000, 1500, 2000])}},
                           batch_size={{choice([256, 512, 768, 1024, 1280, 1536, 1792, 2048])}},
                           verbose=2,
                           shuffle=True,
                           save=False)[0]
    best_mmd_loss = np.amax(result.history['mmd_loss'])
    print('Best validation acc of epoch:', best_mmd_loss)
    return {'loss': best_mmd_loss, 'status': STATUS_OK, 'model': network.cvae_model}


if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials())
    x_train = data()
    true_labels, _ = scgen.label_encoder(x_train)
    fake_labels = np.ones(shape=true_labels.shape)
    print("Evalutation of best performing model:")
    print(best_model.evaluate([x_train.X, true_labels, fake_labels]))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
