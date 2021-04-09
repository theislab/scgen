===================
scGen documentation
===================
.. image:: ./sketch.png
  :width: 400
  :alt: scGen Architecture

scGen is Single-cell Generator!
~~~~~~~~~~~~~
scGen is a generative model to predict single-cell perturbation
response across cell types, studies and species. ``scgen`` is a high-level API, written in
Python and capable of running different deep learning architectures such as VAE,
C-VAE and etc. It also provides some visualizations in order to analyze latent space
mappings from gene expression space.

scGen is compatible with: Python 3.6-3.7.

Main Principles
~~~~~~~~~~~~~
scGen has some main principles:

* User Friendly: scGen is an API designed for human beings, not machines. scGen offers consistent & simple APIs, it minimizes the number of user actions required for a common use case, and it provides clear feedback upon user error.

* Modularity: A model is understood as a sequence or a graph of standalone modules that can be plugged together with as few restrictions as possible. In particular, embedding methods, semi-supervised algorithms schemes are all standalone modules that you can combine to create your own new model.

* extensibility: It's very simple to add new modules, and existing modules provide examples. To be able to easily create new modules allows scGen suitable for advanced research.

* Python Implementation: All models are described in Python code, which is compact, easier to debug, and allows for ease of extensibility.

Getting Started: A Simple Example
~~~~~~~~~~~~~
Here is a simple example to train a VAE with training data

.. code-block:: html
    :linenos:
    import anndata
    import scgen
    import scanpy as sc
    train = sc.read("./tests/data/train.h5ad", backup_url="https://goo.gl/33HtVh")
    scgen.setup_anndata(train)
    network = scgen.SCGEN(train)
    network.train()

Support
~~~~~~~~~~~~~
Please feel free to ask questions:

`Mohammad Lotfollahi
<mailto:mohammad.lotfollahi@helmholtz-muenchen.de>`_

`Mohsen Naghipourfar
<mailto:mn7697np@gmail.com>`_

You can also post bug reports and feature requests in
`GitHub issues
<https://github.com/M0hammadL/scGen/issues>`_. Please Make sure read our
guidelines first.



.. raw:: html

    <div class="container">
        <div class="row">
            <div class="col-lg-6 col-md-6 col-sm-6 col-xs-12 d-flex">
                <div class="card text-center intro-card shadow">
                <img src="_static/computer-24px.svg" class="card-img-top" alt="installation with scvi action icon" height="52">
                <div class="card-body flex-fill">
                    <h5 class="card-title">Installation</h5>
                    <p class="card-text">New to <em>scvi-tools</em>? Check out the installation guide.
                    </p>
