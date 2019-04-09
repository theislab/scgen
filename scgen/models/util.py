import os
from random import shuffle

import anndata
import numpy as np
import scanpy as sc
from matplotlib import pyplot as plt
from scipy import sparse
from sklearn import preprocessing

import scgen


def data_remover(adata, remain_list, remove_list, cell_type_key, condition_key):
    """
        Removes specific cell type in stimulated condition form `adata`.

        # Parameters
            adata: `~anndata.AnnData`
                Annotated data matrix
            remain_list: list
                list of cell types which are going to be remained in `adata`.
            remove_list: list
                list of cell types which are going to be removed from `adata`.

        # Returns
            merged_data: list
                returns array of specified cell types in stimulated condition

        # Example
        ```python
        import scgen
        import anndata
        train_data = anndata.read("./data/train_kang.h5ad")
        remove_list = ["CD14+Mono", "CD8T"]
        remain_list = ["CD4T", "Dendritic"]
        filtered_data = data_remover(train_data, remain_list, remove_list)
        ```
    """
    source_data = []
    for i in remain_list:
        source_data.append(extractor(adata, i, conditions={"ctrl": "control", "stim": "stimulated"},
                                     cell_type_key=cell_type_key, condition_key=condition_key)[3])
    target_data = []
    for i in remove_list:
        target_data.append(extractor(adata, i, conditions={"ctrl": "control", "stim": "stimulated"},
                                     cell_type_key=cell_type_key, condition_key=condition_key)[1])
    merged_data = training_data_provider(source_data, target_data)
    merged_data.var_names = adata.var_names
    return merged_data


def extractor(data, cell_type, conditions, cell_type_key="cell_type", condition_key="condition"):
    """
        Returns a list of `data` files while filtering for a specific `cell_type`.

        # Parameters
        data: `~anndata.AnnData`
            Annotated data matrix
        cell_type: basestring
            specific cell type to be extracted from `data`.
        conditions: dict
            dictionary of stimulated/control of `data`.

        # Returns
            list of `data` files while filtering for a specific `cell_type`.

        # Example
        ```python
        import scgen
        import anndata
        train_data = anndata.read("./data/train.h5ad")
        test_data = anndata.read("./data/test.h5ad")
        train_data_extracted_list = extractor(train_data, "CD4T", conditions={"ctrl": "control", "stim": "stimulated"})
        ```

    """
    cell_with_both_condition = data[data.obs[cell_type_key] == cell_type]
    condtion_1 = data[(data.obs[cell_type_key] == cell_type) & (data.obs[condition_key] == conditions["ctrl"])]
    condtion_2 = data[(data.obs[cell_type_key] == cell_type) & (data.obs[condition_key] == conditions["stim"])]
    training = data[~((data.obs[cell_type_key] == cell_type) & (data.obs[condition_key] == conditions["stim"]))]
    return [training, condtion_1, condtion_2, cell_with_both_condition]


def training_data_provider(train_s, train_t):
    """
        Concatenates two lists containing adata files

        # Parameters
        train_s: `~anndata.AnnData`
            Annotated data matrix.
        train_t: `~anndata.AnnData`
            Annotated data matrix.

        # Returns
            Concatenated Annotated data matrix.

        # Example
        ```python
        import scgen
        import anndata
        train_data = anndata.read("./data/train_kang.h5ad")
        test_data = anndata.read("./data/test.h5ad")
        whole_data = training_data_provider(train_data, test_data)
        ```
    """
    train_s_X = []
    train_s_diet = []
    train_s_groups = []
    for i in train_s:
        train_s_X.append(i.X.A)
        train_s_diet.append(i.obs["condition"].tolist())
        train_s_groups.append(i.obs["cell_type"].tolist())
    train_s_X = np.concatenate(train_s_X)
    temp = []
    for i in train_s_diet:
        temp = temp + i
    train_s_diet = temp
    temp = []
    for i in train_s_groups:
        temp = temp + i
    train_s_groups = temp
    train_t_X = []
    train_t_diet = []
    train_t_groups = []
    for i in train_t:
        train_t_X.append(i.X.A)
        train_t_diet.append(i.obs["condition"].tolist())
        train_t_groups.append(i.obs["cell_type"].tolist())
    temp = []
    for i in train_t_diet:
        temp = temp + i
    train_t_diet = temp
    temp = []
    for i in train_t_groups:
        temp = temp + i
    train_t_groups = temp
    train_t_X = np.concatenate(train_t_X)
    train_real = np.concatenate([train_s_X, train_t_X])  # concat all
    train_real = anndata.AnnData(train_real)
    train_real.obs["condition"] = train_s_diet + train_t_diet
    train_real.obs["cell_type"] = train_s_groups + train_t_groups
    return train_real


def balancer(adata, cell_type_key="cell_type", condition_key="condition"):
    """
        Makes cell type population equal.

        # Parameters
        adata: `~anndata.AnnData`
            Annotated data matrix.

        # Returns
            balanced_data: `~anndata.AnnData`
                Equal cell type population Annotated data matrix.

        # Example
        ```python
        import scgen
        import anndata
        train_data = anndata.read("./train_kang.h5ad")
        train_ctrl = train_data[train_data.obs["condition"] == "control", :]
        train_ctrl = balancer(train_ctrl)
        ```
    """
    class_names = np.unique(adata.obs[cell_type_key])
    class_pop = {}
    for cls in class_names:
        class_pop[cls] = adata.copy()[adata.obs[cell_type_key] == cls].shape[0]
    max_number = np.max(list(class_pop.values()))
    all_data_x = []
    all_data_label = []
    all_data_condition = []
    for cls in class_names:
        temp = adata.copy()[adata.obs[cell_type_key] == cls]
        index = np.random.choice(range(len(temp)), max_number)
        if sparse.issparse(temp.X):
            temp_x = temp.X.A[index]
        else:
            temp_x = temp.X[index]
        all_data_x.append(temp_x)
        temp_ct = np.repeat(cls, max_number)
        all_data_label.append(temp_ct)
        temp_cc = np.repeat(np.unique(temp.obs[condition_key]), max_number)
        all_data_condition.append(temp_cc)
    balanced_data = anndata.AnnData(np.concatenate(all_data_x))
    balanced_data.obs[cell_type_key] = np.concatenate(all_data_label)
    balanced_data.obs[condition_key] = np.concatenate(all_data_label)
    class_names = np.unique(balanced_data.obs[cell_type_key])
    class_pop = {}
    for cls in class_names:
        class_pop[cls] = len(balanced_data[balanced_data.obs[cell_type_key] == cls])
    return balanced_data


def shuffle_data(adata, labels=None):
    """
        Shuffles the `adata`.

        # Parameters
        adata: `~anndata.AnnData`
            Annotated data matrix.
        labels: numpy nd-array
            list of encoded labels

        # Returns
            adata: `~anndata.AnnData`
                Shuffled annotated data matrix.
            labels: numpy nd-array
                Array of shuffled labels if `labels` is not None.

        # Example
        ```python
        import scgen
        import anndata
        import pandas as pd
        train_data = anndata.read("./data/train.h5ad")
        train_labels = pd.read_csv("./data/train_labels.csv", header=None)
        train_data, train_labels = shuffle_data(train_data, train_labels)
        ```
    """
    ind_list = [i for i in range(adata.shape[0])]
    shuffle(ind_list)
    if sparse.issparse(adata.X):
        x = adata.X.A[ind_list, :]
    else:
        x = adata.X[ind_list, :]
    if labels is not None:
        labels = labels[ind_list]
        adata = anndata.AnnData(x, obs={"labels": list(labels)})
        return adata, labels
    else:
        return anndata.AnnData(x, obs=adata.obs)


def batch_removal(network, adata):
    """
        Removes batch effect of adata

        # Parameters
        network: `scgen VAE`
            Variational Auto-encoder class object after training the network.
        adata: `~anndata.AnnData`
            Annotated data matrix. adata must have `batch` and `cell_type` column in its obs.

        # Returns
            corrected: `~anndata.AnnData`
                Annotated matrix of corrected data consisting of all cell types whether they have batch effect or not.

        # Example
        ```python
        import scgen
        import anndata
        train = anndata.read("data/pancreas.h5ad")
        train.obs["cell_type"] = train.obs["celltype"].tolist()
        network = scgen.VAEArith(x_dimension=train.shape[1], model_path="./models/batch")
        network.train(train_data=train, n_epochs=20)
        corrected_adata = scgen.batch_removal(network, train)
        ```
     """
    if sparse.issparse(adata.X):
        latent_all = network.to_latent(adata.X.A)
    else:
        latent_all = network.to_latent(adata.X)
    adata_latent = anndata.AnnData(latent_all)
    adata_latent.obs["cell_type"] = adata.obs["cell_type"].tolist()
    adata_latent.obs["batch"] = adata.obs["batch"].tolist()
    unique_cell_types = np.unique(adata_latent.obs["cell_type"])
    shared_ct = []
    not_shared_ct = []
    for cell_type in unique_cell_types:
        temp_cell = adata_latent[adata_latent.obs["cell_type"] == cell_type]
        if len(np.unique(temp_cell.obs["batch"])) < 2:
            cell_type_ann = adata_latent[adata_latent.obs["cell_type"] == cell_type]
            not_shared_ct.append(cell_type_ann)
            continue
        temp_cell = adata_latent[adata_latent.obs["cell_type"] == cell_type]
        batch_list = {}
        batch_ind = {}
        max_batch = 0
        max_batch_ind = ""
        batches = np.unique(temp_cell.obs["batch"])
        for i in batches:
            temp = temp_cell[temp_cell.obs["batch"] == i]
            temp_ind = temp_cell.obs["batch"] == i
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
    all_shared_ann = anndata.AnnData.concatenate(*shared_ct, batch_key="concat_batch")
    del all_shared_ann.obs["concat_batch"]
    if len(not_shared_ct) < 1:
        corrected = anndata.AnnData(network.reconstruct(all_shared_ann.X, use_data=True))
        corrected.obs["cell_type"] = all_shared_ann.obs["cell_type"].tolist()
        corrected.obs["batch"] = all_shared_ann.obs["batch"].tolist()
        corrected.var_names = adata.var_names.tolist()
        return corrected
    else:
        all_not_shared_ann = anndata.AnnData.concatenate(*not_shared_ct, batch_key="concat_batch")
        all_corrected_data = anndata.AnnData.concatenate(all_shared_ann, all_not_shared_ann, batch_key="concat_batch")
        del all_corrected_data.obs["concat_batch"]
        corrected = anndata.AnnData(network.reconstruct(all_corrected_data.X, use_data=True), )
        corrected.obs["cell_type"] = all_shared_ann.obs["cell_type"].tolist() + all_not_shared_ann.obs[
            "cell_type"].tolist()
        corrected.obs["batch"] = all_shared_ann.obs["batch"].tolist() + all_not_shared_ann.obs["batch"].tolist()
        corrected.var_names = adata.var_names.tolist()
        return corrected


def label_encoder(adata):
    """
        Encode labels of Annotated `adata` matrix using sklearn.preprocessing.LabelEncoder class.
        Parameters
        ----------
        adata: `~anndata.AnnData`
            Annotated data matrix.
        Returns
        -------
        labels: numpy nd-array
            Array of encoded labels
        Example
        --------
        >>> import scgen
        >>> import scanpy as sc
        >>> train_data = sc.read("./data/train.h5ad")
        >>> train_labels, label_encoder = label_encoder(train_data)
    """
    le = preprocessing.LabelEncoder()
    labels = le.fit_transform(adata.obs["condition"].tolist())
    return labels.reshape(-1, 1), le


def visualize_trained_network_results(network, train, cell_type,
                                      conditions={"ctrl": "control", "stim": "stimulated"},
                                      condition_key="condition",
                                      cell_type_key="cell_type",
                                      path_to_save="./figures/",
                                      plot_umap=True,
                                      plot_reg=True):
    plt.close("all")
    os.makedirs(path_to_save, exist_ok=True)
    sc.settings.figdir = os.path.abspath(path_to_save)
    if isinstance(network, scgen.VAEArithKeras):
        if sparse.issparse(train.X):
            latent = network.to_latent(train.X.A)
        else:
            latent = network.to_latent(train.X)
        latent = sc.AnnData(X=latent,
                            obs={condition_key: train.obs[condition_key].tolist(),
                                 cell_type_key: train.obs[cell_type_key].tolist()})
        if plot_umap:
            sc.pp.neighbors(latent)
            sc.tl.umap(latent)
            sc.pl.umap(latent, color=[condition_key, cell_type_key],
                       save=f"_latent",
                       show=False)

        cell_type_data = train[train.obs[cell_type_key] == cell_type]

        pred, delta = network.predict(adata=cell_type_data,
                                      conditions=conditions,
                                      cell_type_key=cell_type_key,
                                      condition_key=condition_key,
                                      celltype_to_predict=cell_type)

        pred_adata = anndata.AnnData(pred, obs={condition_key: ["pred"] * len(pred)},
                                     var={"var_names": cell_type_data.var_names})
        all_adata = cell_type_data.concatenate(pred_adata)
        sc.tl.rank_genes_groups(cell_type_data, groupby=condition_key, n_genes=100)
        diff_genes = cell_type_data.uns["rank_genes_groups"]["names"][conditions["stim"]]
        if plot_reg:
            scgen.plotting.reg_mean_plot(all_adata, condition_key=condition_key,
                                         axis_keys={"x": "pred", "y": conditions["stim"]},
                                         gene_list=diff_genes[:5],
                                         path_to_save=os.path.join(path_to_save, f"reg_mean_all_genes.pdf"))

            scgen.plotting.reg_var_plot(all_adata, condition_key=condition_key,
                                        axis_keys={"x": "pred", "y": conditions["stim"]},
                                        gene_list=diff_genes[:5],
                                        path_to_save=os.path.join(path_to_save, f"reg_var_all_genes.pdf"))

            all_adata_top_100_genes = all_adata.copy()[:, diff_genes.tolist()]

            scgen.plotting.reg_mean_plot(all_adata_top_100_genes, condition_key=condition_key,
                                         axis_keys={"x": "pred", "y": conditions["stim"]},
                                         gene_list=diff_genes[:5],
                                         path_to_save=os.path.join(path_to_save, f"reg_mean_top_100_genes.pdf"))

            scgen.plotting.reg_var_plot(all_adata_top_100_genes, condition_key=condition_key,
                                        axis_keys={"x": "pred", "y": conditions["stim"]},
                                        gene_list=diff_genes[:5],
                                        path_to_save=os.path.join(path_to_save, f"reg_var_top_100_genes.pdf"))

            all_adata_top_50_genes = all_adata.copy()[:, diff_genes.tolist()[:50]]

            scgen.plotting.reg_mean_plot(all_adata_top_50_genes, condition_key=condition_key,
                                         axis_keys={"x": "pred", "y": conditions["stim"]},
                                         gene_list=diff_genes[:5],
                                         path_to_save=os.path.join(path_to_save, f"reg_mean_top_50_genes.pdf"))

            scgen.plotting.reg_var_plot(all_adata_top_50_genes, condition_key=condition_key,
                                        axis_keys={"x": "pred", "y": conditions["stim"]},
                                        gene_list=diff_genes[:5],
                                        path_to_save=os.path.join(path_to_save, f"reg_var_top_50_genes.pdf"))

            if plot_umap:
                sc.pp.neighbors(all_adata)
                sc.tl.umap(all_adata)
                sc.pl.umap(all_adata, color=condition_key,
                           save="pred_all_genes",
                           show=False)

                sc.pp.neighbors(all_adata_top_100_genes)
                sc.tl.umap(all_adata_top_100_genes)
                sc.pl.umap(all_adata_top_100_genes, color=condition_key,
                           save="pred_top_100_genes",
                           show=False)

                sc.pp.neighbors(all_adata_top_50_genes)
                sc.tl.umap(all_adata_top_50_genes)
                sc.pl.umap(all_adata_top_50_genes, color=condition_key,
                           save="pred_top_50_genes",
                           show=False)

        sc.pl.violin(all_adata, keys=diff_genes.tolist()[0], groupby=condition_key,
                     save=f"_{diff_genes.tolist()[0]}",
                     show=False)

        plt.close("all")

    elif isinstance(network, scgen.VAEArith):
        if sparse.issparse(train.X):
            latent = network.to_latent(train.X.A)
        else:
            latent = network.to_latent(train.X)
        latent = sc.AnnData(X=latent,
                            obs={condition_key: train.obs[condition_key].tolist(),
                                 cell_type_key: train.obs[cell_type_key].tolist()})
        if plot_umap:
            sc.pp.neighbors(latent)
            sc.tl.umap(latent)
            sc.pl.umap(latent, color=[condition_key, cell_type_key],
                       save=f"_latent",
                       show=False)

        cell_type_data = train[train.obs[cell_type_key] == cell_type]

        pred, delta = network.predict(adata=cell_type_data,
                                      conditions=conditions,
                                      cell_type_key=cell_type_key,
                                      condition_key=condition_key,
                                      celltype_to_predict=cell_type)

        pred_adata = anndata.AnnData(pred, obs={condition_key: ["pred"] * len(pred)},
                                     var={"var_names": cell_type_data.var_names})
        all_adata = cell_type_data.concatenate(pred_adata)
        sc.tl.rank_genes_groups(cell_type_data, groupby=condition_key, n_genes=100)
        diff_genes = cell_type_data.uns["rank_genes_groups"]["names"][conditions["stim"]]
        if plot_reg:
            scgen.plotting.reg_mean_plot(all_adata, condition_key=condition_key,
                                         axis_keys={"x": "pred", "y": conditions["stim"]},
                                         gene_list=diff_genes[:5],
                                         path_to_save=os.path.join(path_to_save, f"reg_mean_all_genes.pdf"))

            scgen.plotting.reg_var_plot(all_adata, condition_key=condition_key,
                                        axis_keys={"x": "pred", "y": conditions["stim"]},
                                        gene_list=diff_genes[:5],
                                        path_to_save=os.path.join(path_to_save, f"reg_var_all_genes.pdf"))

            all_adata_top_100_genes = all_adata.copy()[:, diff_genes.tolist()]

            scgen.plotting.reg_mean_plot(all_adata_top_100_genes, condition_key=condition_key,
                                         axis_keys={"x": "pred", "y": conditions["stim"]},
                                         gene_list=diff_genes[:5],
                                         path_to_save=os.path.join(path_to_save, f"reg_mean_top_100_genes.pdf"))

            scgen.plotting.reg_var_plot(all_adata_top_100_genes, condition_key=condition_key,
                                        axis_keys={"x": "pred", "y": conditions["stim"]},
                                        gene_list=diff_genes[:5],
                                        path_to_save=os.path.join(path_to_save, f"reg_var_top_100_genes.pdf"))

            all_adata_top_50_genes = all_adata.copy()[:, diff_genes.tolist()[:50]]

            scgen.plotting.reg_mean_plot(all_adata_top_50_genes, condition_key=condition_key,
                                         axis_keys={"x": "pred", "y": conditions["stim"]},
                                         gene_list=diff_genes[:5],
                                         path_to_save=os.path.join(path_to_save, f"reg_mean_top_50_genes.pdf"))

            scgen.plotting.reg_var_plot(all_adata_top_50_genes, condition_key=condition_key,
                                        axis_keys={"x": "pred", "y": conditions["stim"]},
                                        gene_list=diff_genes[:5],
                                        path_to_save=os.path.join(path_to_save, f"reg_var_top_50_genes.pdf"))

            if plot_umap:
                sc.pp.neighbors(all_adata)
                sc.tl.umap(all_adata)
                sc.pl.umap(all_adata, color=condition_key,
                           save="pred_all_genes",
                           show=False)

                sc.pp.neighbors(all_adata_top_100_genes)
                sc.tl.umap(all_adata_top_100_genes)
                sc.pl.umap(all_adata_top_100_genes, color=condition_key,
                           save="pred_top_100_genes",
                           show=False)

                sc.pp.neighbors(all_adata_top_50_genes)
                sc.tl.umap(all_adata_top_50_genes)
                sc.pl.umap(all_adata_top_50_genes, color=condition_key,
                           save="pred_top_50_genes",
                           show=False)

        sc.pl.violin(all_adata, keys=diff_genes.tolist()[0], groupby=condition_key,
                     save=f"_{diff_genes.tolist()[0]}",
                     show=False)

        plt.close("all")

    elif isinstance(network, scgen.CVAE):
        true_labels, _ = scgen.label_encoder(train)


        if sparse.issparse(train.X):
            latent = network.to_latent(train.X.A, labels=true_labels)
        else:
            latent = network.to_latent(train.X, labels=true_labels)
        latent = sc.AnnData(X=latent,
                            obs={condition_key: train.obs[condition_key].tolist(),
                                 cell_type_key: train.obs[cell_type_key].tolist()})
        if plot_umap:
            sc.pp.neighbors(latent)
            sc.tl.umap(latent)
            sc.pl.umap(latent, color=[condition_key, cell_type_key],
                       save=f"_latent",
                       show=False)

        cell_type_data = train[train.obs[cell_type_key] == cell_type]
        fake_labels = np.ones(shape=(cell_type_data.shape[0], 1))

        pred = network.predict(data=cell_type_data, labels=fake_labels)

        pred_adata = anndata.AnnData(pred, obs={condition_key: ["pred"] * len(pred)},
                                     var={"var_names": cell_type_data.var_names})

        all_adata = cell_type_data.concatenate(pred_adata)
        sc.tl.rank_genes_groups(cell_type_data, groupby=condition_key, n_genes=100)
        diff_genes = cell_type_data.uns["rank_genes_groups"]["names"][conditions["stim"]]
        if plot_reg:
            scgen.plotting.reg_mean_plot(all_adata, condition_key=condition_key,
                                         axis_keys={"x": "pred", "y": conditions["stim"]},
                                         gene_list=diff_genes[:5],
                                         path_to_save=os.path.join(path_to_save, f"reg_mean_all_genes.pdf"))

            scgen.plotting.reg_var_plot(all_adata, condition_key=condition_key,
                                        axis_keys={"x": "pred", "y": conditions["stim"]},
                                        gene_list=diff_genes[:5],
                                        path_to_save=os.path.join(path_to_save, f"reg_var_all_genes.pdf"))

            all_adata_top_100_genes = all_adata.copy()[:, diff_genes.tolist()]

            scgen.plotting.reg_mean_plot(all_adata_top_100_genes, condition_key=condition_key,
                                         axis_keys={"x": "pred", "y": conditions["stim"]},
                                         gene_list=diff_genes[:5],
                                         path_to_save=os.path.join(path_to_save, f"reg_mean_top_100_genes.pdf"))

            scgen.plotting.reg_var_plot(all_adata_top_100_genes, condition_key=condition_key,
                                        axis_keys={"x": "pred", "y": conditions["stim"]},
                                        gene_list=diff_genes[:5],
                                        path_to_save=os.path.join(path_to_save, f"reg_var_top_100_genes.pdf"))

            all_adata_top_50_genes = all_adata.copy()[:, diff_genes.tolist()[:50]]

            scgen.plotting.reg_mean_plot(all_adata_top_50_genes, condition_key=condition_key,
                                         axis_keys={"x": "pred", "y": conditions["stim"]},
                                         gene_list=diff_genes[:5],
                                         path_to_save=os.path.join(path_to_save, f"reg_mean_top_50_genes.pdf"))

            scgen.plotting.reg_var_plot(all_adata_top_50_genes, condition_key=condition_key,
                                        axis_keys={"x": "pred", "y": conditions["stim"]},
                                        gene_list=diff_genes[:5],
                                        path_to_save=os.path.join(path_to_save, f"reg_var_top_50_genes.pdf"))

            if plot_umap:
                sc.pp.neighbors(all_adata)
                sc.tl.umap(all_adata)
                sc.pl.umap(all_adata, color=condition_key,
                           save="pred_all_genes",
                           show=False)

                sc.pp.neighbors(all_adata_top_100_genes)
                sc.tl.umap(all_adata_top_100_genes)
                sc.pl.umap(all_adata_top_100_genes, color=condition_key,
                           save="pred_top_100_genes",
                           show=False)

                sc.pp.neighbors(all_adata_top_50_genes)
                sc.tl.umap(all_adata_top_50_genes)
                sc.pl.umap(all_adata_top_50_genes, color=condition_key,
                           save="pred_top_50_genes",
                           show=False)

        sc.pl.violin(all_adata, keys=diff_genes.tolist()[0], groupby=condition_key,
                     save=f"_{diff_genes.tolist()[0]}",
                     show=False)

        plt.close("all")