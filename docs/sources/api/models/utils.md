### data_remover


```python
scgen.models.util.data_remover(adata, remain_list, remove_list)
```



Removes specific cell type in stimulated condition form `adata`.

__Parameters__

- __adata__: `~anndata.AnnData`
    Annotated data matrix
- __remain_list__: list
    list of cell types which are going to be remained in `adata`.
- __remove_list__: list
    list of cell types which are going to be removed from `adata`.

__Returns__

- __merged_data__: list
    returns array of specified cell types in stimulated condition

__Example__

```python
import scgen
import anndata
train_data = anndata.read("./data/train_kang.h5ad")
remove_list = ["CD14+Mono", "CD8T"]
remain_list = ["CD4T", "Dendritic"]
filtered_data = data_remover(train_data, remain_list, remove_list)
```
    
----

### extractor


```python
scgen.models.util.extractor(data, cell_type, conditions)
```



Returns a list of `data` files while filtering for a specific `cell_type`.

__Parameters__

data: `~anndata.AnnData`
Annotated data matrix
cell_type: basestring
specific cell type to be extracted from `data`.
conditions: dict
dictionary of stimulated/control of `data`.

__Returns__

list of `data` files while filtering for a specific `cell_type`.

__Example__

```python
import scgen
import anndata
train_data = anndata.read("./data/train.h5ad")
test_data = anndata.read("./data/test.h5ad")
train_data_extracted_list = extractor(train_data, "CD4T", conditions={"ctrl": "control", "stim": "stimulated"})
```


----

### training_data_provider


```python
scgen.models.util.training_data_provider(train_s, train_t)
```



Concatenates two lists containing adata files

__Parameters__

train_s: `~anndata.AnnData`
Annotated data matrix.
train_t: `~anndata.AnnData`
Annotated data matrix.

__Returns__

Concatenated Annotated data matrix.

__Example__

```python
import scgen
import anndata
train_data = anndata.read("./data/train_kang.h5ad")
test_data = anndata.read("./data/test.h5ad")
whole_data = training_data_provider(train_data, test_data)
```
    
----

### balancer


```python
scgen.models.util.balancer(adata)
```



Makes cell type population equal.

__Parameters__

adata: `~anndata.AnnData`
Annotated data matrix.

__Returns__

- __balanced_data__: `~anndata.AnnData`
    Equal cell type population Annotated data matrix.

__Example__

```python
import scgen
import anndata
train_data = anndata.read("./train_kang.h5ad")
train_ctrl = train_data[train_data.obs["condition"] == "control", :]
train_ctrl = balancer(train_ctrl)
```
    
----

### shuffle_data


```python
scgen.models.util.shuffle_data(adata, labels=None)
```



Shuffles the `adata`.

__Parameters__

adata: `~anndata.AnnData`
Annotated data matrix.
labels: numpy nd-array
list of encoded labels

__Returns__

- __adata__: `~anndata.AnnData`
    Shuffled annotated data matrix.
- __labels__: numpy nd-array
    Array of shuffled labels if `labels` is not None.

__Example__

```python
import scgen
import anndata
import pandas as pd
train_data = anndata.read("./data/train.h5ad")
train_labels = pd.read_csv("./data/train_labels.csv", header=None)
train_data, train_labels = shuffle_data(train_data, train_labels)
```
    
----

### batch_removal


```python
scgen.models.util.batch_removal(network, adata)
```



Removes batch effect of adata

__Parameters__

network: `scgen VAE`
Variational Auto-encoder class object after training the network.
adata: `~anndata.AnnData`
Annotated data matrix. adata must have `batch` and `cell_type` column in its obs.

__Returns__

- __corrected__: `~anndata.AnnData`
    Annotated matrix of corrected data consisting of all cell types whether they have batch effect or not.

__Example__

```python
import scgen
import anndata
train = anndata.read("data/pancreas.h5ad")
train.obs["cell_type"] = train.obs["celltype"].tolist()
network = scgen.VAEArith(x_dimension=train.shape[1], model_path="./models/batch")
network.train(train_data=train, n_epochs=20)
corrected_adata = scgen.batch_removal(network, train)
```
 