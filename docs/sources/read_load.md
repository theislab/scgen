### load_file


```python
scgen.read_load.load_file(filename, backup_url=None)
```



Loads file in any of pandas, numpy or AnnData's extension.

__Parameters__

- __filename__: basestring
    name of the file which is going to be loaded.
- __backup_url__: basestring
    backup url for downloading data if the file with the specified `filename`
    does not exists.
- __kwargs__: dict
    dictionary of additional arguments for loading data with each package.

__Returns__

The annotated matrix of loaded data.

__Example__

```python
import scgen
train_data_filename = "./data/train.h5ad"
train_data = scgen.load_file(train_data_filename)
```

