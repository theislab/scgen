## Installation
- __Install scGen from PyPI (recommended)__:

The easiest way to get scGen is through pip using the following command:
```bash
sudo pip install scgen
```
If you are using a virtualenv, you may want to avoid using sudo:
```bash
pip install -r requirements.txt 
```
This should install all the dependencies in addition to the package.

- __Alternatively: install scGen from the GitHub source:__

You can also get scGen from Github using the following steps:
First, clone scGen using `git`:

```bash
git clone https://github.com/M0hammadL/scgen
```

Then, `cd` to the scGen folder and run the install command:
```bash
cd scgen
python setup.py install
```

On Windows machines you may need to download a C++ compiler if you wish to build from source yourself. 

## Dependencies
The requirements for scGen can be found in the requirements.txt file in the repository, and include numpy, pandas, tensorflow, keras, gensim, pomegranate, and matplotlib.

- [__numpy__](http://numpy.org): The fundamental package for scientific computing.

- [__pandas__](https://pandas.pydata.org): The library which provides high-performance, easy-to-use data structures and data analysis tools for the Python.

- [__tensorflow__](https://www.tensorflow.org): The library for high performance numerical computation.

- [__matplotlib__](https://matplotlib.org): a Python 2D plotting library which produces publication quality figures

