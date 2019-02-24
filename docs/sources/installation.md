## Installation
- __Install scGen from PyPI (recommended)__:

The easiest way to get scGen is through pip using the following command:
```bash
sudo pip install scgen
```

- __Alternatively: install scGen from the GitHub source:__

You can also get scGen from Github using the following steps:
First, clone scGen using `git`:

```bash
git clone https://github.com/theislab/scGen
```

Then, `cd` to the scGen folder and run the install command:
```bash
cd scgen
pip install flit
flit install
```

On Windows machines you may need to download a C++ compiler if you wish to build from source yourself. 

## Dependencies
The requirements for scGen can be found in the requirements.txt file in the repository, and include numpy, pandas, tensorflow, keras, gensim, pomegranate, and matplotlib.

- [__numpy__](http://numpy.org): The fundamental package for scientific computing.

- [__pandas__](https://pandas.pydata.org): The library which provides high-performance, easy-to-use data structures and data analysis tools for the Python.

- [__tensorflow__](https://www.tensorflow.org): The library for high performance numerical computation.

- [__matplotlib__](https://matplotlib.org): a Python 2D plotting library which produces publication quality figures

- [__scipy__](https://scipy.org): SciPy is a Python-based open-source software for mathematics, science, and engineering.

- [__anndata__](https://anndata.readthedocs.io/en/latest/): AnnData provides a scalable way of keeping track of data together with learned annotations.

- [__scanpy__](https://scanpy.readthedocs.io): Scanpy is a scalable toolkit for analyzing single-cell gene expression data.

- [__keras__](https://keras.io): Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano.

- [__get_version__](https://github.com/flying-sheep/get_version): get_version is a package which Automatically uses the latest “vX.X.X” Git tag as version in your Python package.

- [__six__](https://github.com/benjaminp/six):Six is a Python 2 and 3 compatibility library.
