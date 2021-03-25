"""ScGen - Predicting single cell perturbations"""

from scvi.data import setup_anndata

from ._scgen import SCGEN
from ._scgenvae import SCGENVAE

__author__ = ", ".join(["Mohammad  Lotfollahi", "Mohsen Naghipourfar"])

__email__ = ", ".join(
    ["Mohammad.lotfollahi@helmholtz-muenchen.de", "mohsen.naghipourfar@gmail.com"]
)

from get_version import get_version

__version__ = get_version(__file__)
del get_version
