"""ScGen - Predicting single cell perturbations"""

from .models import *
from .read_load import load_file
from . import plotting


__author__ = ', '.join([
    'Mohammad  Lotfollahi',
])

__email__ = ', '.join([
    'Mohammad.lotfollahi@helmholtz-muenchen.de',
])

from get_version import get_version
__version__ = get_version(__file__)
del get_version




