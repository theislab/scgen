from __future__ import print_function

import numpy as np
import scanpy as sc
from hyperas import optim
from hyperas.distributions import choice
from hyperopt import Trials, STATUS_OK, tpe

import scgen


def data():
    x_train = sc.read("./data/train.h5ad")
    return x_train


def create_model(x_train):
    network = scgen.VAEArith(x_dimension=x_train.X.shape[1],
                             z_dimension={{choice([10, 20, 50, 75, 100])}},
                             learning_rate={{choice([0.1, 0.01, 0.001, 0.0001])}},
                             alpha={{choice([0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001])}},
                             dropout_rate={{choice([0.2, 0.25, 0.5, 0.75, 0.8])}},
                             model_path=f"./")

    result = network.train(x_train,
                           n_epochs={{choice([100, 150, 200, 250])}},
                           batch_size={{choice([32, 64, 128, 256])}},
                           verbose=2,
                           shuffle=True,
                           save=False)
    best_loss = np.amin(result.history['loss'])
    print('Best Loss of model:', best_loss)
    return {'loss': best_loss, 'status': STATUS_OK, 'model': network.vae_model}


if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=100,
                                          trials=Trials())
    x_train = data()
    # print("Evalutation of best performing model:")
    # print(best_model.evaluate([x_train.X]))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)


"""
    best run for VAE Arithmetic:
    alpha = .01
    batch_size = 256 
    dropout_rate = 0.75
    learning_rate = 0.1
    n_epochs = 100
    z_dimension = 20
"""