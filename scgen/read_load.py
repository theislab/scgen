import os
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import anndata


def load_file(filename, backup_url=None, **kwargs):#TODO : what if several fileS provided as csv or h5 e.g. x, label1, label2

    """
        Loads file in any of pandas, numpy or AnnData's extension.

        # Parameters
            filename: basestring
                name of the file which is going to be loaded.
            backup_url: basestring
                backup url for downloading data if the file with the specified `filename`
                does not exists.
            kwargs: dict
                dictionary of additional arguments for loading data with each package.

        # Returns
            The annotated matrix of loaded data.

        # Example
        ```python
        import scgen
        train_data_filename = "./data/train.h5ad"
        train_data = scgen.load_file(train_data_filename)
        ```

    """
    numpy_ext = {'npy', 'npz'}
    pandas_ext = {'csv', 'h5'}
    adata_ext = {"h5ad"}

    if not os.path.exists(filename) and backup_url is None:
        raise FileNotFoundError('Did not find file {}.'.format(filename))

    elif not os.path.exists(filename):
        d = os.path.dirname(filename)
        if not os.path.exists(d): os.makedirs(d)
        urlretrieve(backup_url, filename)

    ext = Path(filename).suffixes[-1][1:]

    if ext in numpy_ext:
        return np.load(filename, **kwargs)
    elif ext in pandas_ext:
        return pd.read_csv(filename, **kwargs)
    elif ext in adata_ext:
        return anndata.read(filename, **kwargs)
    else:
        raise ValueError('"{}" does not end on a valid extension.\n'
                         'Please, provide one of the available extensions.\n{}\n'
                         .format(filename, numpy_ext | pandas_ext))
