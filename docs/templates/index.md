# scGen 
[![PyPI version](https://badge.fury.io/py/scgen.svg)](https://badge.fury.io/py/scgen) [![Build Status](https://travis-ci.com/theislab/scGen.svg?branch=master)](https://travis-ci.com/theislab/scGen) [![Documentation Status](https://readthedocs.org/projects/scgen/badge/?version=latest)](https://scgen.readthedocs.io/en/latest/?badge=latest)

![](./sketch/sketch.png)

## scGen is Single-cell Generator!
scGen is a generative model to predict single-cell perturbation response accross cell types, studies and species.
scGen is a high-level API, written in Python and capable of running different deep learning architectures such as VAE, C-VAE and etc. It also provides some visualizations in order to analyze latent space mappings from gene expression space.

scGen is compatible with: Python 3.6-3.7.

## Main Principles
scGen has some main principles:
  
- __User Friendly__: scGen is an API designed for human beings, not machines. scGen offers consistent & simple APIs, it minimizes the number of user actions required for a common use case, and it provides clear feedback upon user error.

- __Modularity__: A model is understood as a sequence or a graph of standalone modules that can be plugged together with as few restrictions as possible. In particular, embedding methods, semi-supervised algorithms schemes are all standalone modules that you can combine to create your own new model.

- __extensibility__: It's very simple to add new modules, and existing modules provide examples. To be able to easily create new modules allows scGen suitable for advanced research.

- __Python Implementation__: All models are described in Python code, which is compact, easier to debug, and allows for ease of extensibility.

## Getting Started: A Simple Example
Here is a simple example to train a VAE with training data 

```python
import scgen as scg
train_data = scg.read_load.load_file("./data/train_data.h5ad") # Read and load the training dataset (AnnData object)
valid_data = scg.read_load.load_file("./data/valid_data.h5ad") # Read and load the validation dataset (AnnData object)
network = scg.VAEArith(x_dimension=train_data.shape[1], z_dimension=100) # Construct a VAE object to train on train_data
network.train(train_data, use_validation=True, valid_data=valid_data, n_epochs=25, batch_size=32, shuffle=True) # Train VAE on train_data and validate on valid_data
```

## Support
Please feel free to ask questions:

- [Mohammad Lotfollahi](mailto:mohammad.lotfollahi@helmholtz-muenchen.de)

- [Mohsen Naghipourfar](mailto:mn7697np@gmail.com)

You can also post bug reports and feature requests in [GitHub issues](https://github.com/M0hammadL/scGen/issues). Please Make sure read our guidelines first.

