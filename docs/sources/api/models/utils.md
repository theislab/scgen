### data_remover


```python
scgen.models.util.data_remover(adata, remain_list, remove_list)
```



Removes specific cell type in stimulated condition form `adata`.

Parameters
----------
adata: `~anndata.AnnData`
Annotated data matrix

remain_list: list
list of cell types which are going to be remained in `adata`.

remove_list: list
list of cell types which are going to be removed from `adata`.

Returns
-------
merged_data: list
returns array of specified cell types in stimulated condition
Example
--------
>>> import scgen
>>> import anndata
>>> train_data = anndata.read("./data/train_kang.h5ad")
>>> remove_list = ["CD14+Mono", "CD8T"]
>>> remain_list = ["CD4T", "Dendritic"]
>>> filtered_data = data_remover(train_data, remain_list, remove_list)

----

### extractor


```python
scgen.models.util.extractor(data, cell_type, conditions)
```



Returns a list of `data` files while filtering for a specific `cell_type`.

Parameters
----------
data: `~anndata.AnnData`
Annotated data matrix

cell_type: basestring
specific cell type to be extracted from `data`.

conditions: dict
dictionary of stimulated/control of `data`.

Returns
-------
list of `data` files while filtering for a specific `cell_type`.

Example
--------
>>> import scgen
>>> import anndata
>>> train_data = anndata.read("./data/train.h5ad")
>>> test_data = anndata.read("./data/test.h5ad")
>>> train_data_extracted_list = extractor(train_data, "CD4T", conditions={"ctrl": "control", "stim": "stimulated"})


----

### balancer


```python
scgen.models.util.balancer(adata)
```



Makes cell type population equal.

Parameters
----------
adata: `~anndata.AnnData`
Annotated data matrix.

Returns
-------
balanced_data: `~anndata.AnnData`
Equal cell type population Annotated data matrix.

Example
--------
>>> import scgen
>>> import anndata
>>> train_data = anndata.read("./train_kang.h5ad")
>>> train_ctrl = train_data[train_data.obs["condition"] == "control", :]
>>> train_ctrl = balancer(train_ctrl)

----

### shuffle_data


```python
scgen.models.util.shuffle_data(adata, labels=None)
```



Shuffles the `adata`.

Parameters
----------
adata: `~anndata.AnnData`
Annotated data matrix.

labels: numpy nd-array
list of encoded labels

Returns
-------
adata: `~anndata.AnnData`
Shuffled annotated data matrix.

labels: numpy nd-array
Array of shuffled labels if `labels` is not None.

Example
--------
>>> import scgen
>>> import anndata
>>> import pandas as pd
>>> train_data = anndata.read("./data/train.h5ad")
>>>train_labels = pd.read_csv("./data/train_labels.csv", header=None)
>>> train_data, train_labels = shuffle_data(train_data, train_labels)

----

### batch_removal


```python
scgen.models.util.batch_removal(network, adata)
```



Removes batch effect of adata

Parameters
----------
network: `scgen VAE`
Variational Auto-encoder class object after training the network.

adata: `~anndata.AnnData`
Annotated data matrix. adata must have `batch` and `cell_type` column in its obs.

Returns
-------
corrected: `~anndata.AnnData`
Annotated matrix of corrected data consisting of all cell types whether they have batch effect or not.

Example
--------
>>> import scgen
>>> import anndata
>>> train = anndata.read("data/pancreas.h5ad")
>>> train.obs["cell_type"] = train.obs["celltype"].tolist()
>>> network = scgen.VAEArith(x_dimension=train.shape[1], model_path="./models/batch")
>>> network.train(train_data=train, n_epochs=20)
>>> corrected_adata = scgen.batch_removal(network, train)

----

### training_data_provider


```python
scgen.models.util.training_data_provider(train_s, train_t)
```



Concatenates two lists containing adata files

Parameters
----------
train_s: `~anndata.AnnData`
Annotated data matrix.

train_t: `~anndata.AnnData`
Annotated data matrix.

Returns
-------
Concatenated Annotated data matrix.

Example
--------
>>> import scgen
>>> import anndata
>>> train_data = anndata.read("./data/train_kang.h5ad")
>>> test_data = anndata.read("./data/test.h5ad")
>>> whole_data = training_data_provider(train_data, test_data)
