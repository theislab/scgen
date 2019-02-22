<span style="float:right;">[[source]](https://github.com/M0hammadL/scgen/blob/master/scgen/models/_vae.py#L12)</span>
### VAEArith

```python
scgen.models._vae.VAEArith(x_dimension, z_dimension=100)
```


VAE with Arithmetic vector Network class. This class contains the implementation of Variational
Auto-encoder network with Vector Arithmetics.

Parameters
----------
kwargs:
:key `validation_data` : AnnData
must be fed if `use_validation` is true.
:key dropout_rate: float
dropout rate
:key learning_rate: float
learning rate of optimization algorithm
:key model_path: basestring
path to save the model after training

x_dimension: integer
number of gene expression space dimensions.

z_dimension: integer
number of latent space dimensions.

See also
--------
CVAE from scgen.models._cvae : Conditional VAE implementation.


----

### linear_interpolation


```python
linear_interpolation(source_adata, dest_adata, n_steps)
```



Maps `source_adata` and `dest_adata` into latent space and linearly interpolate
`n_steps` points between them.

Parameters
----------
source_adata: `~anndata.AnnData`
Annotated data matrix of source cells in gene expression space (`x.X` must be in shape [n_obs, n_vars])
dest_adata: `~anndata.AnnData`
Annotated data matrix of destinations cells in gene expression space (`y.X` must be in shape [n_obs, n_vars])
n_steps: int
Number of steps to interpolate points between `source_adata`, `dest_adata`.

Returns
-------
interpolation: numpy nd-array
Returns the `numpy nd-array` of interpolated points in gene expression space.

Example
--------
>>> import anndata
>>> import scgen
>>> train_data = anndata.read("./data/train.h5ad")
>>> validation_data = anndata.read("./data/validation.h5ad")
>>> network = scgen.VAEArith(x_dimension= train_data.shape[1], model_path="./models/test" )
>>> network.train(train_data=train_data, use_validation=True, valid_data=validation_data, shuffle=True, n_epochs=2)
>>> souece = train_data[((train_data.obs["cell_type"] == "CD8T") & (train_data.obs["condition"] == "control"))]
>>> destination = train_data[((train_data.obs["cell_type"] == "CD8T") & (train_data.obs["condition"] == "stimulated"))]
>>> interpolation = network.linear_interpolation(souece, destination, n_steps=25)

----

### predict


```python
predict(adata, conditions, adata_to_predict=None, celltype_to_predict=None, obs_key='all')
```



Predicts the cell type provided by the user in stimulated condition.

Parameters
----------
celltype_to_predict: basestring
The cell type you want to be predicted.

obs_key: basestring or dict
Dictionary of celltypes you want to be observed for prediction.

adata_to_predict: `~anndata.AnnData`
Adata for unpertubed cells you want to be predicted.

Returns
-------
predicted_cells: numpy nd-array
`numpy nd-array` of predicted cells in primary space.
delta: float
Difference between stimulated and control cells in latent space

Example
--------
>>> import anndata
>>> import scgen
>>> train_data = anndata.read("./data/train.h5ad"
>>> validation_data = anndata.read("./data/validation.h5ad")
>>> network = scgen.VAEArith(x_dimension= train_data.shape[1], model_path="./models/test" )
>>> network.train(train_data=train_data, use_validation=True, valid_data=validation_data, shuffle=True, n_epochs=2)
>>> prediction, delta = network.predict(adata= train_data, celltype_to_predict= "CD4T", conditions={"ctrl": "control", "stim": "stimulated"})

----

### reconstruct


```python
reconstruct(data, use_data=False)
```



Map back the latent space encoding via the decoder.

Parameters
----------
data: `~anndata.AnnData`
Annotated data matrix whether in latent space or gene expression space.

use_data: bool
This flag determines whether the `data` is already in latent space or not.
if `True`: The `data` is in latent space (`data.X` is in shape [n_obs, z_dim]).
if `False`: The `data` is not in latent space (`data.X` is in shape [n_obs, n_vars]).

Returns
-------
rec_data: 'numpy nd-array'
Returns 'numpy nd-array` containing reconstructed 'data' in shape [n_obs, n_vars].

----

### restore_model


```python
restore_model()
```



restores model weights from `model_to_use`.

Parameters
----------
No parameters are needed.

Returns
-------
Nothing will be returned.

Example
--------
>>> import anndata
>>> import scgen
>>> train_data = anndata.read("./data/train.h5ad")
>>> validation_data = anndata.read("./data/validation.h5ad")
>>> network = scgen.VAEArith(x_dimension= train_data.shape[1], model_path="./models/test" )
>>> network.restore_model()

----

### to_latent


```python
to_latent(data)
```



Map `data` in to the latent space. This function will feed data
in encoder part of VAE and compute the latent space coordinates
for each sample in data.

Parameters
----------
data:  numpy nd-array
Numpy nd-array to be mapped to latent space. `data.X` has to be in shape [n_obs, n_vars].

Returns
-------
latent: numpy nd-array
Returns array containing latent space encoding of 'data'

----

### train


```python
train(train_data, use_validation=False, valid_data=None, n_epochs=25, batch_size=32, early_stop_limit=20, threshold=0.0025, initial_run=True, shuffle=True)
```



Trains the network `n_epochs` times with given `train_data`
and validates the model using validation_data if it was given
in the constructor function. This function is using `early stopping`
technique to prevent over-fitting.

Parameters
----------
train_data: scanpy AnnData
Annotated Data Matrix for training VAE network.

use_validation: bool
if `True`: must feed a valid AnnData object to `valid_data` argument.

valid_data: scanpy AnnData
Annotated Data Matrix for validating VAE network after each epoch.

n_epochs: int
Number of epochs to iterate and optimize network weights

batch_size: integer
size of each batch of training dataset to be fed to network while training.

early_stop_limit: int
Number of consecutive epochs in which network loss is not going lower.
After this limit, the network will stop training.

threshold: float
Threshold for difference between consecutive validation loss values
if the difference is upper than this `threshold`, this epoch will not
considered as an epoch in early stopping.

initial_run: bool
if `True`: The network will initiate training and log some useful initial messages.
if `False`: Network will resume the training using `restore_model` function in order
to restore last model which has been trained with some training dataset.

shuffle: bool
if `True`: shuffles the training dataset

Returns
-------
Nothing will be returned

Example
--------
>>> import anndata
>>> import scgen
>>> train_data = anndata.read("./data/train.h5ad"
>>> validation_data = anndata.read("./data/validation.h5ad"
>>> network = scgen.VAEArith(x_dimension= train_data.shape[1], model_path="./models/test")
>>> network.train(train_data=train_data, use_validation=True, valid_data=validation_data, shuffle=True, n_epochs=2)
