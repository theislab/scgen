# scGen [![PyPI version](https://badge.fury.io/py/scgen.svg)](https://badge.fury.io/py/scgen) [![Build Status](https://travis-ci.com/theislab/scGen.svg?branch=master)](https://travis-ci.com/theislab/scGen) [![Documentation Status](https://readthedocs.org/projects/scgen/badge/?version=latest)](https://scgen.readthedocs.io/en/latest/?badge=latest)



<img align="center" src="./sketch/sketch.png?raw=true">

## Introduction
A tensorflow implementation of scGen. scGen is a generative model to predict single-cell perturbation response across cell types, studies and species
  [(bioRxiv, 2018)](https://www.biorxiv.org/content/10.1101/478503v2) .
<div float="left">
  <img src="https://www.tensorflow.org/images/tf_logo_transp.png" height="80" >
  <img src="https://s3.amazonaws.com/keras.io/img/keras-logo-2018-large-1200.png" height="80">
</div>
<div float="right">
</div>

## Getting Started
*What you can do with scGen:

* Train on a dataset wih multiple cell types and conditions and predict the the perturbation effect on the cell type
which you only have in one condition. This scenario can be extended to multiple species where you want to predict
the effect of a specific species using another or all the species.

* Train on a dataset where you have two conditions (e.g. control and perturbed) and predict on second dataset
with similar genes.

* Remove batch effect on labeled data. In this scenario you need to provide cell_type and batch labels to
the method. Note that `batch_removal` does not require all cell types to be present in all datasets (batches). If
you have dataset specific cell type it will preserved as before.

* We assume there exist two conditions in you dataset (e.g. control and perturbed). You can train the model and with
your data and predict the perturbation for the cell type/species of interest. 

* We recommend to use normalized data for the training. A simple example for normalization pipeline using scanpy:

``` python
import scanpy as sc
adata = sc.read(data)
sc.pp.normalize_per_cell(adata)
sc.pp.log1p(adata)
```
* We further recommend to use highly variable genes (HVG). For the most examples in the paper we used top ~7000
HVG. However, this is optional and highly depend on your application and computational power.




## Installation

### Installation with pip
To install the latest version from PyPI, simply use the following bash script:
```bash
pip install scgen
```
or install the development version via pip: 
```bash
pip install git+https://github.com/theislab/scgen.git
```

or you can first install flit and clone this repository:
```bash
pip install flit
git clone https://github.com/theislab/scGen
cd scgen
flit install
```

On Windows machines you may need to download a C++ compiler if you wish to build from source yourself.



## Examples

* For perturbation prediction check this [example](https://nbviewer.jupyter.org/github/M0hammadL/scGen_notebooks/blob/master/notebooks/scgen_kang.ipynb)
 for interferon (IFN)-β stimulation from [Kang et al.](https://www.nature.com/articles/nbt.4042).

* For batch removal check our [example](https://nbviewer.jupyter.org/github/M0hammadL/scGen_notebooks/blob/master/notebooks/scgen_batch_removal.ipynb) on integrating four pancreas datasets.


###### Reproducing paper results:
In order to reproduce paper results visit [here](https://github.com/M0hammadL/scGen_reproducibility).
