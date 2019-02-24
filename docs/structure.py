# -*- coding: utf-8 -*-
"""
General documentation architecture:
Home
- Getting started
    Simple Example
    FAQ
- API
    - Models
        About scGen models
        VAE
    - read_load
    - plotting
"""

from scgen import models
from scgen import plotting
from scgen import read_load
from scgen.models import util

PAGES = [
    # {
    #     'page': 'models/index.md',
    #     'classes': [models.VAEArith],
    # },
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
