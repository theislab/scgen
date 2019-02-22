# -*- coding: utf-8 -*-
'''
General documentation architecture:
Home
Index
- Getting started
    Getting started with the sequential model
    Getting started with the functional api
    FAQ
- Models
    About Keras models
        explain when one should use Sequential or functional API
        explain compilation step
        explain weight saving, weight loading
        explain serialization, deserialization
    Sequential
    Model (functional API)
- Layers
    About Keras layers
        explain common layer functions: get_weights, set_weights, get_config
        explain input_shape
        explain usage on non-Keras tensors
    Core Layers
    Convolutional Layers
    Pooling Layers
    Locally-connected Layers
    Recurrent Layers
    Embedding Layers
    Merge Layers
    Advanced Activations Layers
    Normalization Layers
    Noise Layers
    Layer Wrappers
    Writing your own Keras layers
- Preprocessing
    Sequence Preprocessing
    Text Preprocessing
    Image Preprocessing
Losses
Metrics
Optimizers
Activations
Callbacks
Datasets
Applications
Backend
Initializers
Regularizers
Constraints
Visualization
Scikit-learn API
Utils
Contributing
'''

from scgen import models
from scgen import read_load
from scgen import plotting
from scgen.models import util

PAGES = [
    {
        'page': 'models/vae.md',
        'classes': [models.VAEArith],
        'methods': [models.VAEArith.linear_interpolation,
                    models.VAEArith.predict,
                    models.VAEArith.reconstruct,
                    models.VAEArith.restore_model,
                    models.VAEArith.to_latent,
                    models.VAEArith.train]
    },
    {
        'page': 'read_load.md',
        'all_module_functions': [read_load],
    },
    {
        'page': 'models/utils.md',
        'all_module_functions': [util],
    },
    {
        'page': 'plotting.md',
        'all_module_functions': [plotting],
    },
]

