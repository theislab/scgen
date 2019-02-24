import keras
import numpy as np
from keras.layers import Dense


class DeepTaylor(object):
    def __init__(self, model):
        self.model = model

    def _relevance_propagate(self, R):
        for layer in self.model.layers[::-1]:
            R = layer.relprop(R)
        return R


class DenseLRP(Dense):
    def __init__(self, units, **kwargs):
        super().__init__(units, **kwargs)

    def relevance_propagate(self, R):
        if self.activation == keras.activations.relu:
            return R


def z_plos(X, W, eps=1e-9):
    R = 1
    V = np.maximum(0, W)
    Z = np.dot(V, X) + eps
    S = np.divide(R, Z)
    C = np.dot(S, V.T)
    R = np.dot(X, C)
    return R