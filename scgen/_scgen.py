from typing import Optional, Sequence

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from adjustText import adjust_text
from anndata import AnnData, concat
from matplotlib import pyplot
from scipy import stats
from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data.fields import CategoricalObsField, LayerField
from scvi.model.base import BaseModelClass, UnsupervisedTrainingMixin, VAEMixin
from scvi.utils import setup_anndata_dsp

from ._scgenvae import SCGENVAE
from ._utils import balancer, extractor

font = {"family": "Arial", "size": 14}


class SCGEN(VAEMixin, UnsupervisedTrainingMixin, BaseModelClass):
    """
    Implementation of scGen model for batch removal and perturbation prediction.

    Parameters
    ----------
    adata
        AnnData object that has been registered via :meth:`~scgen.SCGEN.setup_anndata`.
    n_hidden
        Number of nodes per hidden layer.
    n_latent
        Dimensionality of the latent space.
    n_layers
        Number of hidden layers used for encoder and decoder NNs.
    dropout_rate
        Dropout rate for neural networks.
    **model_kwargs
        Keyword args for :class:`~scgen.SCGENVAE`

    Examples
    --------
    >>> vae = scgen.SCGEN(adata)
    >>> vae.train()
    >>> adata.obsm["X_scgen"] = vae.get_latent_representation()
    """

    def __init__(
        self,
        adata: AnnData,
        n_hidden: int = 800,
        n_latent: int = 100,
        n_layers: int = 2,
        dropout_rate: float = 0.2,
        **model_kwargs,
    ):
        super(SCGEN, self).__init__(adata)

        self.module = SCGENVAE(
            n_input=self.summary_stats.n_vars,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            **model_kwargs,
        )
        self._model_summary_string = (
            "SCGEN Model with the following params: \nn_hidden: {}, n_latent: {}, n_layers: {}, dropout_rate: "
            "{}"
        ).format(
            n_hidden,
            n_latent,
            n_layers,
            dropout_rate,
        )
        self.init_params_ = self._get_init_params(locals())

    def predict(
        self,
        ctrl_key=None,
        stim_key=None,
        adata_to_predict=None,
        celltype_to_predict=None,
        restrict_arithmetic_to="all",
    ) -> AnnData:
        """
        Predicts the cell type provided by the user in stimulated condition.

        Parameters
        ----------
        ctrl_key: basestring
            key for `control` part of the `data` found in `condition_key`.
        stim_key: basestring
            key for `stimulated` part of the `data` found in `condition_key`.
        adata_to_predict: `~anndata.AnnData`
            Adata for unperturbed cells you want to be predicted.
        celltype_to_predict: basestring
            The cell type you want to be predicted.
        restrict_arithmetic_to: basestring or dict
            Dictionary of celltypes you want to be observed for prediction.
        Returns
        -------
        predicted_cells: np nd-array
            `np nd-array` of predicted cells in primary space.
        delta: float
            Difference between stimulated and control cells in latent space
        """
        # use keys registered from `setup_anndata()`
        cell_type_key = self.adata_manager.get_state_registry(
            REGISTRY_KEYS.LABELS_KEY
        ).original_key
        condition_key = self.adata_manager.get_state_registry(
            REGISTRY_KEYS.BATCH_KEY
        ).original_key

        if restrict_arithmetic_to == "all":
            ctrl_x = self.adata[self.adata.obs[condition_key] == ctrl_key, :]
            stim_x = self.adata[self.adata.obs[condition_key] == stim_key, :]
            ctrl_x = balancer(ctrl_x, cell_type_key)
            stim_x = balancer(stim_x, cell_type_key)
        else:
            key = list(restrict_arithmetic_to.keys())[0]
            values = restrict_arithmetic_to[key]
            subset = self.adata[self.adata.obs[key].isin(values)]
            ctrl_x = subset[subset.obs[condition_key] == ctrl_key, :]
            stim_x = subset[subset.obs[condition_key] == stim_key, :]
            if len(values) > 1:
                ctrl_x = balancer(ctrl_x, cell_type_key)
                stim_x = balancer(stim_x, cell_type_key)
        if celltype_to_predict is not None and adata_to_predict is not None:
            raise Exception("Please provide either a cell type or adata not both!")
        if celltype_to_predict is None and adata_to_predict is None:
            raise Exception(
                "Please provide a cell type name or adata for your unperturbed cells"
            )
        if celltype_to_predict is not None:
            ctrl_pred = extractor(
                self.adata,
                celltype_to_predict,
                condition_key,
                cell_type_key,
                ctrl_key,
                stim_key,
            )[1]
        else:
            ctrl_pred = adata_to_predict

        eq = min(ctrl_x.X.shape[0], stim_x.X.shape[0])
        cd_ind = np.random.choice(range(ctrl_x.shape[0]), size=eq, replace=False)
        stim_ind = np.random.choice(range(stim_x.shape[0]), size=eq, replace=False)
        ctrl_adata = ctrl_x[cd_ind, :]
        stim_adata = stim_x[stim_ind, :]

        latent_ctrl = self._avg_vector(ctrl_adata)
        latent_stim = self._avg_vector(stim_adata)

        delta = latent_stim - latent_ctrl

        latent_cd = self.get_latent_representation(ctrl_pred)

        stim_pred = delta + latent_cd
        predicted_cells = (
            self.module.generative(torch.Tensor(stim_pred))["px"].cpu().detach().numpy()
        )

        predicted_adata = AnnData(
            X=predicted_cells,
            obs=ctrl_pred.obs.copy(),
            var=ctrl_pred.var.copy(),
            obsm=ctrl_pred.obsm.copy(),
        )
        return predicted_adata, delta

    def _avg_vector(self, adata):
        return np.mean(self.get_latent_representation(adata), axis=0)

    @torch.no_grad()
    def get_decoded_expression(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """Get decoded expression."""
        if self.is_trained_ is False:
            raise RuntimeError("Please train the model first.")

        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )
        decoded = []
        for tensors in scdl:
            _, generative_outputs = self.module(tensors, compute_loss=False)
            px = generative_outputs["px"].cpu()
            decoded.append(px)

        return torch.cat(decoded).numpy()

    @torch.no_grad()
    def batch_removal(self, adata: Optional[AnnData] = None) -> AnnData:
        """
        Removes batch effects.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model. Must have been setup with `batch_key` and `labels_key`,
            corresponding to batch and cell type metadata, respectively.

        Returns
        -------
        corrected: `~anndata.AnnData`
            AnnData of corrected gene expression in adata.X and corrected latent space in adata.obsm["latent"].
            A reference to the original AnnData is in `corrected.raw` if the input adata had no `raw` attribute.
        """
        adata = self._validate_anndata(adata)
        latent_all = self.get_latent_representation(adata)
        # use keys registered from `setup_anndata()`
        cell_label_key = self.adata_manager.get_state_registry(
            REGISTRY_KEYS.LABELS_KEY
        ).original_key
        batch_key = self.adata_manager.get_state_registry(
            REGISTRY_KEYS.BATCH_KEY
        ).original_key

        adata_latent = AnnData(latent_all)
        adata_latent.obs = adata.obs.copy(deep=True)
        unique_cell_types = np.unique(adata_latent.obs[cell_label_key])
        shared_ct = []
        not_shared_ct = []
        for cell_type in unique_cell_types:
            temp_cell = adata_latent[
                adata_latent.obs[cell_label_key] == cell_type
            ].copy()
            if len(np.unique(temp_cell.obs[batch_key])) < 2:
                cell_type_ann = adata_latent[
                    adata_latent.obs[cell_label_key] == cell_type
                ]
                not_shared_ct.append(cell_type_ann)
                continue
            temp_cell = adata_latent[
                adata_latent.obs[cell_label_key] == cell_type
            ].copy()
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
                delta = np.average(max_batch_ann.X, axis=0) - np.average(
                    batch_list[study].X, axis=0
                )
                batch_list[study].X = delta + batch_list[study].X
                temp_cell[batch_ind[study]].X = batch_list[study].X
            shared_ct.append(temp_cell)
        all_shared_ann = concat(
            shared_ct, label="concat_batch", index_unique=None
        )
        if "concat_batch" in all_shared_ann.obs.columns:
            del all_shared_ann.obs["concat_batch"]
        if len(not_shared_ct) < 1:
            corrected = AnnData(
                self.module.generative(torch.Tensor(all_shared_ann.X))["px"]
                .cpu()
                .numpy(),
                obs=all_shared_ann.obs,
            )
            corrected.var_names = adata.var_names.tolist()
            corrected = corrected[adata.obs_names]
            if adata.raw is not None:
                adata_raw = AnnData(X=adata.raw.X, var=adata.raw.var)
                adata_raw.obs_names = adata.obs_names
                corrected.raw = adata_raw
            corrected.obsm["latent"] = all_shared_ann.X
            corrected.obsm["corrected_latent"] = self.get_latent_representation(
                corrected
            )
            return corrected
        else:
            all_not_shared_ann = AnnData.concatenate(
                *not_shared_ct, batch_key="concat_batch", index_unique=None
            )
            all_corrected_data = AnnData.concatenate(
                all_shared_ann,
                all_not_shared_ann,
                batch_key="concat_batch",
                index_unique=None,
            )
            if "concat_batch" in all_shared_ann.obs.columns:
                del all_corrected_data.obs["concat_batch"]
            corrected = AnnData(
                self.module.generative(torch.Tensor(all_corrected_data.X))["px"]
                .cpu()
                .numpy(),
                obs=all_corrected_data.obs,
            )
            corrected.var_names = adata.var_names.tolist()
            corrected = corrected[adata.obs_names]
            if adata.raw is not None:
                adata_raw = AnnData(X=adata.raw.X, var=adata.raw.var)
                adata_raw.obs_names = adata.obs_names
                corrected.raw = adata_raw
            corrected.obsm["latent"] = all_corrected_data.X
            corrected.obsm["corrected_latent"] = self.get_latent_representation(
                corrected
            )
            return corrected

    def reg_mean_plot(
        self,
        adata,
        axis_keys,
        labels,
        path_to_save="./reg_mean.pdf",
        save=True,
        gene_list=None,
        show=False,
        top_100_genes=None,
        verbose=False,
        legend=True,
        title=None,
        x_coeff=0.30,
        y_coeff=0.8,
        fontsize=14,
        **kwargs,
    ):
        """
        Plots mean matching figure for a set of specific genes.

        Parameters
        ----------
        adata: `~anndata.AnnData`
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model. Must have been setup with `batch_key` and `labels_key`,
            corresponding to batch and cell type metadata, respectively.
        axis_keys: dict
            Dictionary of `adata.obs` keys that are used by the axes of the plot. Has to be in the following form:
             `{"x": "Key for x-axis", "y": "Key for y-axis"}`.
        labels: dict
            Dictionary of axes labels of the form `{"x": "x-axis-name", "y": "y-axis name"}`.
        path_to_save: basestring
            path to save the plot.
        save: boolean
            Specify if the plot should be saved or not.
        gene_list: list
            list of gene names to be plotted.
        show: bool
            if `True`: will show to the plot after saving it.
        Examples
        --------
        >>> import anndata
        >>> import scgen
        >>> import scanpy as sc
        >>> train = sc.read("./tests/data/train.h5ad", backup_url="https://goo.gl/33HtVh")
        >>> scgen.SCGEN.setup_anndata(train)
        >>> network = scgen.SCGEN(train)
        >>> network.train()
        >>> unperturbed_data = train[((train.obs["cell_type"] == "CD4T") & (train.obs["condition"] == "control"))]
        >>> pred, delta = network.predict(
        >>>     adata=train,
        >>>     adata_to_predict=unperturbed_data,
        >>>     ctrl_key="control",
        >>>     stim_key="stimulated"
        >>>)
        >>> pred_adata = anndata.AnnData(
        >>>     pred,
        >>>     obs={"condition": ["pred"] * len(pred)},
        >>>     var={"var_names": train.var_names},
        >>>)
        >>> CD4T = train[train.obs["cell_type"] == "CD4T"]
        >>> all_adata = CD4T.concatenate(pred_adata)
        >>> network.reg_mean_plot(
        >>>     all_adata,
        >>>     axis_keys={"x": "control", "y": "pred", "y1": "stimulated"},
        >>>     gene_list=["ISG15", "CD3D"],
        >>>     path_to_save="tests/reg_mean.pdf",
        >>>     show=False
        >>> )
        """
        import seaborn as sns

        sns.set()
        sns.set(color_codes=True)

        condition_key = self.adata_manager.get_state_registry(
            REGISTRY_KEYS.BATCH_KEY
        ).original_key

        diff_genes = top_100_genes
        stim = adata[adata.obs[condition_key] == axis_keys["y"]]
        ctrl = adata[adata.obs[condition_key] == axis_keys["x"]]
        if diff_genes is not None:
            if hasattr(diff_genes, "tolist"):
                diff_genes = diff_genes.tolist()
            adata_diff = adata[:, diff_genes]
            stim_diff = adata_diff[adata_diff.obs[condition_key] == axis_keys["y"]]
            ctrl_diff = adata_diff[adata_diff.obs[condition_key] == axis_keys["x"]]
            x_diff = np.asarray(np.mean(ctrl_diff.X, axis=0)).ravel()
            y_diff = np.asarray(np.mean(stim_diff.X, axis=0)).ravel()
            m, b, r_value_diff, p_value_diff, std_err_diff = stats.linregress(
                x_diff, y_diff
            )
            if verbose:
                print("top_100 DEGs mean: ", r_value_diff**2)
        x = np.asarray(np.mean(ctrl.X, axis=0)).ravel()
        y = np.asarray(np.mean(stim.X, axis=0)).ravel()
        m, b, r_value, p_value, std_err = stats.linregress(x, y)
        if verbose:
            print("All genes mean: ", r_value**2)
        df = pd.DataFrame({axis_keys["x"]: x, axis_keys["y"]: y})
        ax = sns.regplot(x=axis_keys["x"], y=axis_keys["y"], data=df)
        ax.tick_params(labelsize=fontsize)
        if "range" in kwargs:
            start, stop, step = kwargs.get("range")
            ax.set_xticks(np.arange(start, stop, step))
            ax.set_yticks(np.arange(start, stop, step))
        ax.set_xlabel(labels["x"], fontsize=fontsize)
        ax.set_ylabel(labels["y"], fontsize=fontsize)
        if gene_list is not None:
            texts = []
            for i in gene_list:
                j = adata.var_names.tolist().index(i)
                x_bar = x[j]
                y_bar = y[j]
                texts.append(pyplot.text(x_bar, y_bar, i, fontsize=11, color="black"))
                pyplot.plot(x_bar, y_bar, "o", color="red", markersize=5)
                # if "y1" in axis_keys.keys():
                # y1_bar = y1[j]
                # pyplot.text(x_bar, y1_bar, i, fontsize=11, color="black")
        if gene_list is not None:
            adjust_text(
                texts,
                x=x,
                y=y,
                arrowprops=dict(arrowstyle="->", color="grey", lw=0.5),
                force_points=(0.0, 0.0),
            )
        if legend:
            pyplot.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        if title is None:
            pyplot.title("", fontsize=fontsize)
        else:
            pyplot.title(title, fontsize=fontsize)
        ax.text(
            max(x) - max(x) * x_coeff,
            max(y) - y_coeff * max(y),
            r"$\mathrm{R^2_{\mathrm{\mathsf{all\ genes}}}}$= " + f"{r_value ** 2:.2f}",
            fontsize=kwargs.get("textsize", fontsize),
        )
        if diff_genes is not None:
            ax.text(
                max(x) - max(x) * x_coeff,
                max(y) - (y_coeff + 0.15) * max(y),
                r"$\mathrm{R^2_{\mathrm{\mathsf{top\ 100\ DEGs}}}}$= "
                + f"{r_value_diff ** 2:.2f}",
                fontsize=kwargs.get("textsize", fontsize),
            )
        if save:
            pyplot.savefig(f"{path_to_save}", bbox_inches="tight", dpi=100)
        if show:
            pyplot.show()
        pyplot.close()
        if diff_genes is not None:
            return r_value**2, r_value_diff**2
        else:
            return r_value**2

    def reg_var_plot(
        self,
        adata,
        axis_keys,
        labels,
        path_to_save="./reg_var.pdf",
        save=True,
        gene_list=None,
        top_100_genes=None,
        show=False,
        legend=True,
        title=None,
        verbose=False,
        x_coeff=0.30,
        y_coeff=0.8,
        fontsize=14,
        **kwargs,
    ):
        """
        Plots variance matching figure for a set of specific genes.

        Parameters
        ----------
        adata: `~anndata.AnnData`
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model. Must have been setup with `batch_key` and `labels_key`,
            corresponding to batch and cell type metadata, respectively.
        axis_keys: dict
            Dictionary of `adata.obs` keys that are used by the axes of the plot. Has to be in the following form:
             `{"x": "Key for x-axis", "y": "Key for y-axis"}`.
        labels: dict
            Dictionary of axes labels of the form `{"x": "x-axis-name", "y": "y-axis name"}`.
        path_to_save: basestring
            path to save the plot.
        save: boolean
            Specify if the plot should be saved or not.
        gene_list: list
            list of gene names to be plotted.
        show: bool
            if `True`: will show to the plot after saving it.

        Examples
        --------
        >>> import anndata
        >>> import scgen
        >>> import scanpy as sc
        >>> train = sc.read("./tests/data/train.h5ad", backup_url="https://goo.gl/33HtVh")
        >>> scgen.SCGEN.setup_anndata(train)
        >>> network = scgen.SCGEN(train)
        >>> network.train()
        >>> unperturbed_data = train[((train.obs["cell_type"] == "CD4T") & (train.obs["condition"] == "control"))]
        >>> pred, delta = network.predict(
        >>>     adata=train,
        >>>     adata_to_predict=unperturbed_data,
        >>>     ctrl_key="control",
        >>>     stim_key="stimulated"
        >>>)
        >>> pred_adata = anndata.AnnData(
        >>>     pred,
        >>>     obs={"condition": ["pred"] * len(pred)},
        >>>     var={"var_names": train.var_names},
        >>>)
        >>> CD4T = train[train.obs["cell_type"] == "CD4T"]
        >>> all_adata = CD4T.concatenate(pred_adata)
        >>> network.reg_var_plot(
        >>>     all_adata,
        >>>     axis_keys={"x": "control", "y": "pred", "y1": "stimulated"},
        >>>     gene_list=["ISG15", "CD3D"],
        >>>     path_to_save="tests/reg_var4.pdf",
        >>>     show=False
        >>>)
        """
        import seaborn as sns

        sns.set()
        sns.set(color_codes=True)

        condition_key = self.adata_manager.get_state_registry(
            REGISTRY_KEYS.BATCH_KEY
        ).original_key

        sc.tl.rank_genes_groups(
            adata, groupby=condition_key, n_genes=100, method="wilcoxon"
        )
        diff_genes = top_100_genes
        stim = adata[adata.obs[condition_key] == axis_keys["y"]]
        ctrl = adata[adata.obs[condition_key] == axis_keys["x"]]
        if diff_genes is not None:
            if hasattr(diff_genes, "tolist"):
                diff_genes = diff_genes.tolist()
            adata_diff = adata[:, diff_genes]
            stim_diff = adata_diff[adata_diff.obs[condition_key] == axis_keys["y"]]
            ctrl_diff = adata_diff[adata_diff.obs[condition_key] == axis_keys["x"]]
            x_diff = np.asarray(np.var(ctrl_diff.X, axis=0)).ravel()
            y_diff = np.asarray(np.var(stim_diff.X, axis=0)).ravel()
            m, b, r_value_diff, p_value_diff, std_err_diff = stats.linregress(
                x_diff, y_diff
            )
            if verbose:
                print("Top 100 DEGs var: ", r_value_diff**2)
        if "y1" in axis_keys.keys():
            real_stim = adata[adata.obs[condition_key] == axis_keys["y1"]]
        x = np.asarray(np.var(ctrl.X, axis=0)).ravel()
        y = np.asarray(np.var(stim.X, axis=0)).ravel()
        m, b, r_value, p_value, std_err = stats.linregress(x, y)
        if verbose:
            print("All genes var: ", r_value**2)
        df = pd.DataFrame({axis_keys["x"]: x, axis_keys["y"]: y})
        ax = sns.regplot(x=axis_keys["x"], y=axis_keys["y"], data=df)
        ax.tick_params(labelsize=fontsize)
        if "range" in kwargs:
            start, stop, step = kwargs.get("range")
            ax.set_xticks(np.arange(start, stop, step))
            ax.set_yticks(np.arange(start, stop, step))
        # _p1 = pyplot.scatter(x, y, marker=".", label=f"{axis_keys['x']}-{axis_keys['y']}")
        # pyplot.plot(x, m * x + b, "-", color="green")
        ax.set_xlabel(labels["x"], fontsize=fontsize)
        ax.set_ylabel(labels["y"], fontsize=fontsize)
        if "y1" in axis_keys.keys():
            y1 = np.asarray(np.var(real_stim.X, axis=0)).ravel()
            _ = pyplot.scatter(
                x,
                y1,
                marker="*",
                c="grey",
                alpha=0.5,
                label=f"{axis_keys['x']}-{axis_keys['y1']}",
            )
        if gene_list is not None:
            for i in gene_list:
                j = adata.var_names.tolist().index(i)
                x_bar = x[j]
                y_bar = y[j]
                pyplot.text(x_bar, y_bar, i, fontsize=11, color="black")
                pyplot.plot(x_bar, y_bar, "o", color="red", markersize=5)
                if "y1" in axis_keys.keys():
                    y1_bar = y1[j]
                    pyplot.text(x_bar, y1_bar, "*", color="black", alpha=0.5)
        if legend:
            pyplot.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        if title is None:
            pyplot.title("", fontsize=12)
        else:
            pyplot.title(title, fontsize=12)
        ax.text(
            max(x) - max(x) * x_coeff,
            max(y) - y_coeff * max(y),
            r"$\mathrm{R^2_{\mathrm{\mathsf{all\ genes}}}}$= " + f"{r_value ** 2:.2f}",
            fontsize=kwargs.get("textsize", fontsize),
        )
        if diff_genes is not None:
            ax.text(
                max(x) - max(x) * x_coeff,
                max(y) - (y_coeff + 0.15) * max(y),
                r"$\mathrm{R^2_{\mathrm{\mathsf{top\ 100\ DEGs}}}}$= "
                + f"{r_value_diff ** 2:.2f}",
                fontsize=kwargs.get("textsize", fontsize),
            )

        if save:
            pyplot.savefig(f"{path_to_save}", bbox_inches="tight", dpi=100)
        if show:
            pyplot.show()
        pyplot.close()
        if diff_genes is not None:
            return r_value**2, r_value_diff**2
        else:
            return r_value**2

    def binary_classifier(
        self,
        adata,
        delta,
        ctrl_key,
        stim_key,
        path_to_save,
        save=True,
        fontsize=14,
    ):
        """
        Latent space classifier.

        Builds a linear classifier based on the dot product between
        the difference vector and the latent representation of each
        cell and plots the dot product results between delta and latent
        representation.


        Parameters
        ----------
        adata: `~anndata.AnnData`
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model. Must have been setup with `batch_key` and `labels_key`,
            corresponding to batch and cell type metadata, respectively.
        delta: float
            Difference between stimulated and control cells in latent space
        ctrl_key: basestring
            key for `control` part of the `data` found in `condition_key`.
        stim_key: basestring
            key for `stimulated` part of the `data` found in `condition_key`.
        path_to_save: basestring
            path to save the plot.
        save: boolean
            Specify if the plot should be saved or not.
        fontsize: integer
            Set the font size of the plot.

        Examples
        --------
        >>> import anndata
        >>> import scgen
        >>> import scanpy as sc
        >>> train = sc.read("./tests/data/train.h5ad", backup_url="https://goo.gl/33HtVh")
        >>> scgen.SCGEN.setup_anndata(train)
        >>> network = scgen.SCGEN(train)
        >>> network.train()
        >>> unperturbed_data = train[((train.obs["cell_type"] == "CD4T") & (train.obs["condition"] == "control"))]
        >>> pred, delta = network.predict(
        >>>     adata=train,
        >>>     adata_to_predict=unperturbed_data,
        >>>     ctrl_key="control",
        >>>     stim_key="stimulated"
        >>>)
        >>> network.binary_classifier(
        >>>     network,
        >>>     train,
        >>>     delta,
        >>>     ctrl_key="control",
        >>>     stim_key="stimulated",
        >>>     path_to_save="tests/binary_classifier.pdf"
        >>>     )
        """
        # matplotlib.rcParams.update(matplotlib.rcParamsDefault)
        pyplot.close("all")
        adata = self._validate_anndata(adata)
        condition_key = self.adata_manager.get_state_registry(
            REGISTRY_KEYS.BATCH_KEY
        ).original_key
        cd = adata[adata.obs[condition_key] == ctrl_key, :]
        stim = adata[adata.obs[condition_key] == stim_key, :]
        all_latent_cd = self.get_latent_representation(cd.X)
        all_latent_stim = self.get_latent_representation(stim.X)
        dot_cd = np.zeros((len(all_latent_cd)))
        dot_sal = np.zeros((len(all_latent_stim)))
        for ind, vec in enumerate(all_latent_cd):
            dot_cd[ind] = np.dot(delta, vec)
        for ind, vec in enumerate(all_latent_stim):
            dot_sal[ind] = np.dot(delta, vec)
        pyplot.hist(
            dot_cd,
            label=ctrl_key,
            bins=50,
        )
        pyplot.hist(dot_sal, label=stim_key, bins=50)
        # pyplot.legend(loc=1, prop={'size': 7})
        pyplot.axvline(0, color="k", linestyle="dashed", linewidth=1)
        pyplot.title("  ", fontsize=fontsize)
        pyplot.xlabel("  ", fontsize=fontsize)
        pyplot.ylabel("  ", fontsize=fontsize)
        pyplot.xticks(fontsize=fontsize)
        pyplot.yticks(fontsize=fontsize)
        ax = pyplot.gca()
        ax.grid(False)
        if save:
            pyplot.savefig(f"{path_to_save}", bbox_inches="tight", dpi=100)
        pyplot.show()

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        batch_key: Optional[str] = None,
        labels_key: Optional[str] = None,
        **kwargs,
    ):
        """
        %(summary)s.

        Parameters
        ----------
        %(param_batch_key)s
        %(param_labels_key)s

        Notes
        -----
        scGen expects the expression data to come from `adata.X`
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, None, is_count_data=False),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
        ]
        adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)
