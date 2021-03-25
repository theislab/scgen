import numpy
from matplotlib import pyplot
from scipy import stats
from adjustText import adjust_text
import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from scipy import sparse
import scanpy as sc

from scvi._compat import Literal
from scvi.model.base import ArchesMixin, BaseModelClass, RNASeqMixin, VAEMixin, UnsupervisedTrainingMixin

from ._scgenvae import SCGENVAE
from ._utils import (
    extractor,
    balancer
)

font = {
    'family': 'Arial',
    'size': 14
}


class SCGEN(RNASeqMixin, VAEMixin, ArchesMixin, UnsupervisedTrainingMixin, BaseModelClass):
    """
    Parameters
    ----------
    adata
        AnnData object that has been registered via :func:`~scvi.data.setup_anndata`.
    n_hidden
        Number of nodes per hidden layer.
    n_latent
        Dimensionality of the latent space.
    n_layers
        Number of hidden layers used for encoder and decoder NNs.
    dropout_rate
        Dropout rate for neural networks.
    latent_distribution
        One of:
        * ``'normal'`` - Normal distribution
        * ``'ln'`` - Logistic normal distribution (Normal(0, I) transformed by softmax)
    **model_kwargs
        Keyword args for :class:`~scvi.modules.VAE`

    Examples
    --------
    >>> vae = scvi.model.SCGEN(adata)
    >>> vae.train()
    >>> adata.obsm["X_scVI"] = vae.get_latent_representation()
    >>> adata.obsm["X_normalized_scVI"] = vae.get_normalized_expression()
    """

    def __init__(
        self,
        adata: AnnData,
        n_hidden: int = 800,
        n_latent: int = 100,
        n_layers: int = 2,
        dropout_rate: float = 0.2,
        latent_distribution: Literal["normal", "ln"] = "normal",
        **model_kwargs,
    ):
        super(SCGEN, self).__init__(adata)
        self.adata = adata

        self.module = SCGENVAE(
            n_input=self.summary_stats["n_vars"],
            n_batch=self.summary_stats["n_batch"],
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            latent_distribution=latent_distribution,
            **model_kwargs,
        )
        self._model_summary_string = (
            "SCGEN Model with the following params: \nn_hidden: {}, n_latent: {}, n_layers: {}, dropout_rate: "
            "{}, latent_distribution: {}"
        ).format(
            n_hidden,
            n_latent,
            n_layers,
            dropout_rate,
            latent_distribution,
        )
        self.init_params_ = self._get_init_params(locals())

    def predict(self, 
                cell_type_key, 
                condition_key, 
                conditions = None, 
                adata_to_predict=None, 
                celltype_to_predict=None, 
                obs_key="all"):
        """
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
        """
        if obs_key == "all":
            ctrl_x = self.adata[self.adata.obs["condition"] == conditions["ctrl"], :]
            stim_x = self.adata[self.adata.obs["condition"] == conditions["stim"], :]
            ctrl_x = balancer(ctrl_x, cell_type_key=cell_type_key, condition_key=condition_key)
            stim_x = balancer(stim_x, cell_type_key=cell_type_key, condition_key=condition_key)
        else:
            key = list(obs_key.keys())[0]
            values = obs_key[key]
            subset = self.adata[self.adata.obs[key].isin(values)]
            ctrl_x = subset[subset.obs["condition"] == conditions["ctrl"], :]
            stim_x = subset[subset.obs["condition"] == conditions["stim"], :]
            if len(values) > 1:
                ctrl_x = balancer(ctrl_x, cell_type_key=cell_type_key, condition_key=condition_key)
                stim_x = balancer(stim_x, cell_type_key=cell_type_key, condition_key=condition_key)
        if celltype_to_predict is not None and adata_to_predict is not None:
            raise Exception("Please provide either a cell type or adata not both!")
        if celltype_to_predict is None and adata_to_predict is None:
            raise Exception("Please provide a cell type name or adata for your unperturbed cells")
        if celltype_to_predict is not None:
            ctrl_pred = extractor(self.adata, celltype_to_predict, conditions, cell_type_key, condition_key)[1]
        else:
            ctrl_pred = adata_to_predict
            
        eq = min(ctrl_x.X.shape[0], stim_x.X.shape[0])
        cd_ind = np.random.choice(range(ctrl_x.shape[0]), size=eq, replace=False)
        stim_ind = np.random.choice(range(stim_x.shape[0]), size=eq, replace=False)

        ctrl_adata = ctrl_x[cd_ind, :]
        stim_adata = stim_x[stim_ind, :]
        
        if sparse.issparse(ctrl_adata.X) and sparse.issparse(stim_adata.X):
            ctrl_adata.X = ctrl_adata.X.A
            stim_adata.X = stim_adata.X.A
            
        latent_ctrl = self._avg_vector(ctrl_adata)
        latent_stim = self._avg_vector(stim_adata)

        delta = latent_stim - latent_ctrl
        if sparse.issparse(ctrl_pred.X):
            ctrl_pred.X = ctrl_pred.X.A
        
        latent_cd = self.get_latent_representation(ctrl_pred)
        
        stim_pred = delta + latent_cd
        predicted_cells = self.module.generative(torch.Tensor(stim_pred), None)['px'].cpu().detach().numpy()

        predicted_adata = AnnData(X=predicted_cells, obs=ctrl_pred.obs.copy(), var=ctrl_pred.var.copy())
        return predicted_adata, delta

    def _avg_vector(self, adata):
        return np.mean(self.get_latent_representation(adata), axis=0)

    def batch_removal(self, adata, batch_key="condition", cell_label_key="cell_type"):
        """
            Removes batch effect of adata
            #Parameters
            adata: `~anndata.AnnData`
                Annotated data matrix. adata must have `batch_key` and `cell_label_key` which you pass to the function
                in its obs.
            batch_key: `str` batch label key in  adata.obs
            cell_label_key: `str` cell type label key in adata.obs
            return_latent: `bool` if `True` the returns corrected latent representation
            # Returns
                corrected: `~anndata.AnnData`
                    adata of corrected gene expression in adata.X and corrected latent space in adata.obsm["latent"].
        """
        if sparse.issparse(adata.X):
            adata.X = adata.X.A
        
        latent_all = self.get_latent_representation(adata)

        adata_latent = AnnData(latent_all)
        adata_latent.obs = adata.obs.copy(deep=True)
        unique_cell_types = np.unique(adata_latent.obs[cell_label_key])
        shared_ct = []
        not_shared_ct = []
        for cell_type in unique_cell_types:
            temp_cell = adata_latent[adata_latent.obs[cell_label_key] == cell_type]
            if len(np.unique(temp_cell.obs[batch_key])) < 2:
                cell_type_ann = adata_latent[adata_latent.obs[cell_label_key] == cell_type]
                not_shared_ct.append(cell_type_ann)
                continue
            temp_cell = adata_latent[adata_latent.obs[cell_label_key] == cell_type]
            batch_list = {}
            batch_ind = {}
            max_batch = 0
            max_batch_ind = ""
            batches = np.unique(temp_cell.obs[batch_key])
            for i in batches:
                temp = temp_cell[temp_cell.obs[batch_key] == i]
                temp_ind = temp_cell.obs[batch_key] == i
                if max_batch < len(temp):
                    max_batch = len(temp)
                    max_batch_ind = i
                batch_list[i] = temp
                batch_ind[i] = temp_ind
            max_batch_ann = batch_list[max_batch_ind]
            for study in batch_list:
                delta = np.average(max_batch_ann.X, axis=0) - np.average(batch_list[study].X, axis=0)
                batch_list[study].X = delta + batch_list[study].X
                temp_cell[batch_ind[study]].X = batch_list[study].X
            shared_ct.append(temp_cell)
        all_shared_ann = AnnData.concatenate(*shared_ct, batch_key="concat_batch", index_unique=None)
        if "concat_batch" in all_shared_ann.obs.columns:
            del all_shared_ann.obs["concat_batch"]
        if len(not_shared_ct) < 1:
            corrected = AnnData(self.module.generative(torch.Tensor(all_shared_ann.X), None),obs=all_shared_ann.obs)
            corrected.var_names = adata.var_names.tolist()
            corrected = corrected[adata.obs_names]
            if adata.raw is not None:
                adata_raw = AnnData(X=adata.raw.X, var=adata.raw.var)
                adata_raw.obs_names = adata.obs_names
                corrected.raw = adata_raw
            corrected.obsm["latent"] = all_shared_ann.X
            return corrected
        else:
            all_not_shared_ann = AnnData.concatenate(*not_shared_ct, batch_key="concat_batch", index_unique=None)
            all_corrected_data = AnnData.concatenate(all_shared_ann, all_not_shared_ann, batch_key="concat_batch", index_unique=None)
            if "concat_batch" in all_shared_ann.obs.columns:
                del all_corrected_data.obs["concat_batch"]
            corrected = AnnData(self.module.generative(torch.Tensor(all_corrected_data.X), None)["px"].detach().cpu().numpy(), all_corrected_data.obs)
            corrected.var_names = adata.var_names.tolist()
            corrected = corrected[adata.obs_names]
            if adata.raw is not None:
                adata_raw = AnnData(X=adata.raw.X, var=adata.raw.var)
                adata_raw.obs_names = adata.obs_names
                corrected.raw = adata_raw
            corrected.obsm["latent"] = all_corrected_data.X
            return corrected

    def reg_mean_plot(
            self,
            adata,
            condition_key,
            axis_keys,
            labels,
            path_to_save="./reg_mean.pdf",
            gene_list=None,
            top_100_genes=None,
            show=False,
            verbose=False,
            legend=True,
            title=None,
            x_coeff=0.30,
            y_coeff=0.8,
            fontsize=14,
            **kwargs
    ):
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
        import seaborn as sns
        sns.set()
        sns.set(color_codes=True)
        if sparse.issparse(adata.X):
            adata.X = adata.X.A
        diff_genes = top_100_genes
        stim = adata[adata.obs[condition_key] == axis_keys["y"]]
        ctrl = adata[adata.obs[condition_key] == axis_keys["x"]]
        if diff_genes is not None:
            if hasattr(diff_genes, "tolist"):
                diff_genes = diff_genes.tolist()
            adata_diff = adata[:, diff_genes]
            stim_diff = adata_diff[adata_diff.obs[condition_key] == axis_keys["y"]]
            ctrl_diff = adata_diff[adata_diff.obs[condition_key] == axis_keys["x"]]
            x_diff = numpy.average(ctrl_diff.X, axis=0)
            y_diff = numpy.average(stim_diff.X, axis=0)
            m, b, r_value_diff, p_value_diff, std_err_diff = stats.linregress(x_diff, y_diff)
            if verbose:
                print('top_100 DEGs mean: ', r_value_diff ** 2)
        if "y1" in axis_keys.keys():
            real_stim = adata[adata.obs[condition_key] == axis_keys["y1"]]
        x = numpy.average(ctrl.X, axis=0)
        y = numpy.average(stim.X, axis=0)
        m, b, r_value, p_value, std_err = stats.linregress(x, y)
        if verbose:
            print('All genes mean: ', r_value ** 2)
        df = pd.DataFrame({axis_keys["x"]: x, axis_keys["y"]: y})
        ax = sns.regplot(x=axis_keys["x"], y=axis_keys["y"], data=df)
        ax.tick_params(labelsize=fontsize)
        if "range" in kwargs:
            start, stop, step = kwargs.get("range")
            ax.set_xticks(numpy.arange(start, stop, step))
            ax.set_yticks(numpy.arange(start, stop, step))
        # _p1 = pyplot.scatter(x, y, marker=".", label=f"{axis_keys['x']}-{axis_keys['y']}")
        # pyplot.plot(x, m * x + b, "-", color="green")
        ax.set_xlabel(labels["x"], fontsize=fontsize)
        ax.set_ylabel(labels["y"], fontsize=fontsize)
        # if "y1" in axis_keys.keys():
            # y1 = numpy.average(real_stim.X, axis=0)
            # _p2 = pyplot.scatter(x, y1, marker="*", c="red", alpha=.5, label=f"{axis_keys['x']}-{axis_keys['y1']}")
        if gene_list is not None:
            texts = []
            for i in gene_list:
                j = adata.var_names.tolist().index(i)
                x_bar = x[j]
                y_bar = y[j]
                texts.append(pyplot.text(x_bar, y_bar , i, fontsize=11, color ="black"))
                pyplot.plot(x_bar, y_bar, 'o', color="red", markersize=5)
                # if "y1" in axis_keys.keys():
                    # y1_bar = y1[j]
                    # pyplot.text(x_bar, y1_bar, i, fontsize=11, color="black")
        if gene_list is not None:
            adjust_text(texts, x=x, y=y, arrowprops=dict(arrowstyle="->", color='grey', lw=0.5), force_points=(0.0, 0.0))
        if legend:
            pyplot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        if title is None:
            pyplot.title(f"", fontsize=fontsize)
        else:
            pyplot.title(title, fontsize=fontsize)
        ax.text(max(x) - max(x) * x_coeff, max(y) - y_coeff * max(y), r'$\mathrm{R^2_{\mathrm{\mathsf{all\ genes}}}}$= ' + f"{r_value ** 2:.2f}", fontsize=kwargs.get("textsize", fontsize))
        if diff_genes is not None:
            ax.text(max(x) - max(x) * x_coeff, max(y) - (y_coeff+0.15) * max(y), r'$\mathrm{R^2_{\mathrm{\mathsf{top\ 100\ DEGs}}}}$= ' + f"{r_value_diff ** 2:.2f}", fontsize=kwargs.get("textsize", fontsize))
        pyplot.savefig(f"{path_to_save}", bbox_inches='tight', dpi=100)
        if show:
            pyplot.show()
        pyplot.close()
        if diff_genes is not None:
            return r_value ** 2, r_value_diff ** 2
        else:
            return r_value ** 2

    def reg_var_plot(
            self,
            adata,
            condition_key,
            axis_keys,
            labels,
            path_to_save="./reg_var.pdf",
            gene_list=None,
            top_100_genes=None,
            show=False,
            legend=True,
            title=None,
            verbose=False,
            x_coeff=0.30,
            y_coeff=0.8,
            fontsize=14,
            **kwargs
    ):
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
        import seaborn as sns
        sns.set()
        sns.set(color_codes=True)
        if sparse.issparse(adata.X):
            adata.X = adata.X.A
        sc.tl.rank_genes_groups(adata, groupby=condition_key, n_genes=100, method="wilcoxon")
        diff_genes = top_100_genes
        stim = adata[adata.obs[condition_key] == axis_keys["y"]]
        ctrl = adata[adata.obs[condition_key] == axis_keys["x"]]
        if diff_genes is not None:
            if hasattr(diff_genes, "tolist"):
                diff_genes = diff_genes.tolist()
            adata_diff = adata[:, diff_genes]
            stim_diff = adata_diff[adata_diff.obs[condition_key] == axis_keys["y"]]
            ctrl_diff = adata_diff[adata_diff.obs[condition_key] == axis_keys["x"]]
            x_diff = numpy.var(ctrl_diff.X, axis=0)
            y_diff = numpy.var(stim_diff.X, axis=0)
            m, b, r_value_diff, p_value_diff, std_err_diff = stats.linregress(x_diff, y_diff)
            if verbose:
                print('Top 100 DEGs var: ', r_value_diff ** 2)
        if "y1" in axis_keys.keys():
            real_stim = adata[adata.obs[condition_key] == axis_keys["y1"]]
        x = numpy.var(ctrl.X, axis=0)
        y = numpy.var(stim.X, axis=0)
        m, b, r_value, p_value, std_err = stats.linregress(x, y)
        if verbose:
            print('All genes var: ', r_value ** 2)
        df = pd.DataFrame({axis_keys["x"]: x, axis_keys["y"]: y})
        ax = sns.regplot(x=axis_keys["x"], y=axis_keys["y"], data=df)
        ax.tick_params(labelsize=fontsize)
        if "range" in kwargs:
            start, stop, step = kwargs.get("range")
            ax.set_xticks(numpy.arange(start, stop, step))
            ax.set_yticks(numpy.arange(start, stop, step))
        # _p1 = pyplot.scatter(x, y, marker=".", label=f"{axis_keys['x']}-{axis_keys['y']}")
        # pyplot.plot(x, m * x + b, "-", color="green")
        ax.set_xlabel(labels['x'], fontsize=fontsize)
        ax.set_ylabel(labels['y'], fontsize=fontsize)
        if "y1" in axis_keys.keys():
            y1 = numpy.var(real_stim.X, axis=0)
            _p2 = pyplot.scatter(x, y1, marker="*", c="grey", alpha=.5, label=f"{axis_keys['x']}-{axis_keys['y1']}")
        if gene_list is not None:
            for i in gene_list:
                j = adata.var_names.tolist().index(i)
                x_bar = x[j]
                y_bar = y[j]
                pyplot.text(x_bar, y_bar, i, fontsize=11, color="black")
                pyplot.plot(x_bar, y_bar, 'o', color="red", markersize=5)
                if "y1" in axis_keys.keys():
                    y1_bar = y1[j]
                    pyplot.text(x_bar, y1_bar, '*', color="black", alpha=.5)
        if legend:
            pyplot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        if title is None:
            pyplot.title(f"", fontsize=12)
        else:
            pyplot.title(title, fontsize=12)
        ax.text(max(x) - max(x) * x_coeff, max(y) - y_coeff * max(y), r'$\mathrm{R^2_{\mathrm{\mathsf{all\ genes}}}}$= ' + f"{r_value ** 2:.2f}", fontsize=kwargs.get("textsize", fontsize))
        if diff_genes is not None:
            ax.text(max(x) - max(x) * x_coeff, max(y) - (y_coeff + 0.15) * max(y), r'$\mathrm{R^2_{\mathrm{\mathsf{top\ 100\ DEGs}}}}$= ' + f"{r_value_diff ** 2:.2f}", fontsize=kwargs.get("textsize", fontsize))
        pyplot.savefig(f"{path_to_save}", bbox_inches='tight', dpi=100)
        if show:
            pyplot.show()
        pyplot.close()
        if diff_genes is not None:
            return r_value ** 2, r_value_diff ** 2
        else:
            return r_value ** 2

    def binary_classifier(
            self,
            scg_object,
            adata,
            delta,
            condition_key,
            conditions,
            path_to_save,
            fontsize=14
    ):
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
        # matplotlib.rcParams.update(matplotlib.rcParamsDefault)
        pyplot.close("all")
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
        # pyplot.legend(loc=1, prop={'size': 7})
        pyplot.axvline(0, color='k', linestyle='dashed', linewidth=1)
        pyplot.title("  ", fontsize=fontsize)
        pyplot.xlabel("  ", fontsize=fontsize)
        pyplot.ylabel("  ", fontsize=fontsize)
        pyplot.xticks(fontsize=fontsize)
        pyplot.yticks(fontsize=fontsize)
        ax = pyplot.gca()
        ax.grid(False)
        pyplot.savefig(f"{path_to_save}", bbox_inches='tight', dpi=100)
        pyplot.show()