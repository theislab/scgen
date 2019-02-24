### reg_mean_plot


```python
scgen.plotting.reg_mean_plot(adata, condition_key, axis_keys, path_to_save='./reg_mean.pdf', gene_list=None, show=False)
```



Plots mean matching figure for a set of specific genes.

__Parameters__

- __adata__: `~anndata.AnnData`
    Annotated Data Matrix.
- __condition_key__: basestring
    Condition state to be used.
- __axis_keys__: dict
    dictionary of axes labels.
- __path_to_save__: basestring
    path to save the plot.
- __gene_list__: list
    list of gene names to be plotted.
- __show__: bool
    if `True`: will show to the plot after saving it.

__Example__

```python
import anndata
import scgen
import scanpy as sc
train = sc.read("./tests/data/train.h5ad", backup_url="https://goo.gl/33HtVh")
network = scgen.VAEArith(x_dimension=train.shape[1], model_path="../models/test")
network.train(train_data=train, n_epochs=0)
unperturbed_data = train[((train.obs["cell_type"] == "CD4T") & (train.obs["condition"] == "control"))]
condition = {"ctrl": "control", "stim": "stimulated"}
pred, delta = network.predict(adata=train, adata_to_predict=unperturbed_data, conditions=condition)
pred_adata = anndata.AnnData(pred, obs={"condition": ["pred"] * len(pred)}, var={"var_names": train.var_names})
CD4T = train[train.obs["cell_type"] == "CD4T"]
all_adata = CD4T.concatenate(pred_adata)
scgen.plotting.reg_mean_plot(all_adata, condition_key="condition", axis_keys={"x": "control", "y": "pred", "y1": "stimulated"},
                             gene_list=["ISG15", "CD3D"], path_to_save="tests/reg_mean.pdf", show=False)
network.sess.close()
```


----

### reg_var_plot


```python
scgen.plotting.reg_var_plot(adata, condition_key, axis_keys, path_to_save='./reg_var.pdf', gene_list=None, show=False)
```



Plots variance matching figure for a set of specific genes.

__Parameters__

- __adata__: `~anndata.AnnData`
    Annotated Data Matrix.
- __condition_key__: basestring
    Condition state to be used.
- __axis_keys__: dict
    dictionary of axes labels.
- __path_to_save__: basestring
    path to save the plot.
- __gene_list__: list
    list of gene names to be plotted.
- __show__: bool
    if `True`: will show to the plot after saving it.

__Example__

```python
import anndata
import scgen
import scanpy as sc
train = sc.read("./tests/data/train.h5ad", backup_url="https://goo.gl/33HtVh")
network = scgen.VAEArith(x_dimension=train.shape[1], model_path="../models/test")
network.train(train_data=train, n_epochs=0)
unperturbed_data = train[((train.obs["cell_type"] == "CD4T") & (train.obs["condition"] == "control"))]
condition = {"ctrl": "control", "stim": "stimulated"}
pred, delta = network.predict(adata=train, adata_to_predict=unperturbed_data, conditions=condition)
pred_adata = anndata.AnnData(pred, obs={"condition": ["pred"] * len(pred)}, var={"var_names": train.var_names})
CD4T = train[train.obs["cell_type"] == "CD4T"]
all_adata = CD4T.concatenate(pred_adata)
scgen.plotting.reg_var_plot(all_adata, condition_key="condition", axis_keys={"x": "control", "y": "pred", "y1": "stimulated"},
                            gene_list=["ISG15", "CD3D"], path_to_save="tests/reg_var4.pdf", show=False)
network.sess.close()
```


----

### binary_classifier


```python
scgen.plotting.binary_classifier(scg_object, adata, delta, condition_key, conditions, path_to_save)
```



Builds a linear classifier based on the dot product between
the difference vector and the latent representation of each
cell and plots the dot product results between delta and latent
representation.

__Parameters__

- __scg_object__: `~scgen.models.VAEArith`
    one of scGen models object.
- __adata__: `~anndata.AnnData`
    Annotated Data Matrix.
- __delta__: float
    Difference between stimulated and control cells in latent space
- __condition_key__: basestring
    Condition state to be used.
- __conditions__: dict
    dictionary of conditions.
- __path_to_save__: basestring
    path to save the plot.

__Example__

```python
import anndata
import scgen
import scanpy as sc
train = sc.read("./tests/data/train.h5ad", backup_url="https://goo.gl/33HtVh")
network = scgen.VAEArith(x_dimension=train.shape[1], model_path="../models/test")
network.train(train_data=train, n_epochs=0)
unperturbed_data = train[((train.obs["cell_type"] == "CD4T") & (train.obs["condition"] == "control"))]
condition = {"ctrl": "control", "stim": "stimulated"}
pred, delta = network.predict(adata=train, adata_to_predict=unperturbed_data, conditions=condition)
scgen.plotting.binary_classifier(network, train, delta, condtion_key="condition",
                                 conditions={"ctrl": "control", "stim": "stimulated"},
                                 path_to_save="tests/binary_classifier.pdf")
network.sess.close()
```

