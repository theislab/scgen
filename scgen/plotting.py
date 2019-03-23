import numpy
from matplotlib import pyplot
from scipy import stats, sparse


def reg_mean_plot(adata, condition_key, axis_keys, path_to_save="./reg_mean.pdf", gene_list=None, show=False,
                  legend=True, title=None):
    """
        Plots mean matching figure for a set of specific genes.

        # Parameters
            adata: `~anndata.AnnData`
                Annotated Data Matrix.
            condition_key: basestring
                Condition state to be used.
            axis_keys: dict
                dictionary of axes labels.
            path_to_save: basestring
                path to save the plot.
            gene_list: list
                list of gene names to be plotted.
            show: bool
                if `True`: will show to the plot after saving it.

        # Example
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

        """
    if sparse.issparse(adata.X):
        adata.X = adata.X.A
    stim = adata[adata.obs[condition_key] == axis_keys["y"]]
    ctrl = adata[adata.obs[condition_key] == axis_keys["x"]]
    if "y1" in axis_keys.keys():
        real_stim = adata[adata.obs[condition_key] == axis_keys["y1"]]
    x = numpy.average(ctrl.X, axis=0)
    y = numpy.average(stim.X, axis=0)
    m, b, r_value, p_value, std_err = stats.linregress(x, y)
    _p1 = pyplot.scatter(x, y, marker=".", label=f"{axis_keys['x']}-{axis_keys['y']}")
    pyplot.plot(x, m * x + b, "-", color="green")
    pyplot.xlabel(axis_keys["x"], fontsize=12)
    pyplot.ylabel(axis_keys["y"], fontsize=12)
    if "y1" in axis_keys.keys():
        y1 = numpy.average(real_stim.X, axis=0)
        _p2 = pyplot.scatter(x, y1, marker="*", c="red", alpha=.5, label=f"{axis_keys['x']}-{axis_keys['y1']}")
    if gene_list is not None:
        for i in gene_list:
            j = adata.var_names.tolist().index(i)
            x_bar = x[j]
            y_bar = y[j]
            pyplot.text(x_bar, y_bar, i, fontsize=11, color="grey")
            if "y1" in axis_keys.keys():
                y1_bar = y1[j]
                pyplot.text(x_bar, y1_bar, i, fontsize=11, color="grey")
    if legend:
        pyplot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    if title is None:
        pyplot.title(f"", fontsize=12)
    else:
        pyplot.title(title, fontsize=12)
    pyplot.text(max(x) - max(x) * .25, max(y) - .8 * max(y), r'$R^2$=' + f"{r_value ** 2:.2f}")
    pyplot.savefig(f"{path_to_save}", bbox_inches='tight', dpi=100)
    if show:
        pyplot.show()
    pyplot.close()


def reg_var_plot(adata, condition_key, axis_keys, path_to_save="./reg_var.pdf", gene_list=None, show=False,
                 legend=True, title=None):
    """
        Plots variance matching figure for a set of specific genes.

        # Parameters
            adata: `~anndata.AnnData`
                Annotated Data Matrix.
            condition_key: basestring
                Condition state to be used.
            axis_keys: dict
                dictionary of axes labels.
            path_to_save: basestring
                path to save the plot.
            gene_list: list
                list of gene names to be plotted.
            show: bool
                if `True`: will show to the plot after saving it.

        # Example
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

        """
    if sparse.issparse(adata.X):
        adata.X = adata.X.A
    stim = adata[adata.obs[condition_key] == axis_keys["y"]]
    ctrl = adata[adata.obs[condition_key] == axis_keys["x"]]
    if "y1" in axis_keys.keys():
        real_stim = adata[adata.obs[condition_key] == axis_keys["y1"]]
    x = numpy.var(ctrl.X, axis=0)
    y = numpy.var(stim.X, axis=0)
    m, b, r_value, p_value, std_err = stats.linregress(x, y)
    _p1 = pyplot.scatter(x, y, marker=".", label=f"{axis_keys['x']}-{axis_keys['y']}")
    pyplot.plot(x, m * x + b, "-", color="green")
    pyplot.xlabel(axis_keys["x"], fontsize=12)
    pyplot.ylabel(axis_keys["y"], fontsize=12)
    if "y1" in axis_keys.keys():
        y1 = numpy.var(real_stim.X, axis=0)
        _p2 = pyplot.scatter(x, y1, marker="*", c="grey", alpha=.5, label=f"{axis_keys['x']}-{axis_keys['y1']}")
    if gene_list is not None:
        for i in gene_list:
            j = adata.var_names.tolist().index(i)
            x_bar = x[j]
            y_bar = y[j]
            pyplot.text(x_bar, y_bar, i, fontsize=11, color="grey")
            if "y1" in axis_keys.keys():
                y1_bar = y1[j]
                pyplot.text(x_bar, y1_bar, '*', color="blue", alpha=.5)
    if legend:
        pyplot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    if title is None:
        pyplot.title(f"", fontsize=12)
    else:
        pyplot.title(title, fontsize=12)
    pyplot.text(max(x) - .2 * max(x), max(y) - .8 * max(y), r'$R^2$=' + f"{r_value ** 2:.2f}")
    pyplot.savefig(f"{path_to_save}", bbox_inches='tight', dpi=100)
    if show:
        pyplot.show()
    pyplot.close()


def binary_classifier(scg_object, adata, delta, condition_key, conditions, path_to_save):
    """
        Builds a linear classifier based on the dot product between
        the difference vector and the latent representation of each
        cell and plots the dot product results between delta and latent
        representation.

        # Parameters
            scg_object: `~scgen.models.VAEArith`
                one of scGen models object.
            adata: `~anndata.AnnData`
                Annotated Data Matrix.
            delta: float
                Difference between stimulated and control cells in latent space
            condition_key: basestring
                Condition state to be used.
            conditions: dict
                dictionary of conditions.
            path_to_save: basestring
                path to save the plot.

        # Example
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

        """
    if sparse.issparse(adata.X):
        adata.X = adata.X.A
    cd = adata[adata.obs[condition_key] == conditions["ctrl"], :]
    stim = adata[adata.obs[condition_key] == conditions["stim"], :]
    all_latent_cd = scg_object.to_latent(cd.X)
    all_latent_stim = scg_object.to_latent(stim.X)
    dot_cd = numpy.zeros((len(all_latent_cd)))
    dot_sal = numpy.zeros((len(all_latent_stim)))
    for ind, vec in enumerate(all_latent_cd):
        dot_cd[ind] = numpy.dot(delta, vec)
    for ind, vec in enumerate(all_latent_stim):
        dot_sal[ind] = numpy.dot(delta, vec)
    pyplot.hist(dot_cd, label=conditions["ctrl"], bins=50, )
    pyplot.hist(dot_sal, label=conditions["stim"], bins=50)
    pyplot.legend(loc=1, prop={'size': 7})
    pyplot.axvline(0, color='k', linestyle='dashed', linewidth=1)
    pyplot.title("  ", fontsize=10)
    pyplot.xlabel("  ", fontsize=10)
    pyplot.ylabel("  ", fontsize=10)
    pyplot.xticks(fontsize=8)
    pyplot.yticks(fontsize=8)
    ax = pyplot.gca()
    ax.grid(False)
    pyplot.savefig(f"{path_to_save}", bbox_inches='tight', dpi=100)
