import os

import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

import scgen

if not os.getcwd().endswith("tests"):
    os.chdir("./tests")


def test_train_whole_data_one_celltype_out(data_name="pbmc",
                                           z_dim=50,
                                           alpha=0.1,
                                           n_epochs=1000,
                                           batch_size=32,
                                           dropout_rate=0.25,
                                           learning_rate=0.001,
                                           condition_key="condition"):
    if data_name == "pbmc":
        cell_type_to_monitor = "CD4T"
        stim_key = "stimulated"
        ctrl_key = "control"
        cell_type_key = "cell_type"
        train = sc.read("../data/train.h5ad")
    elif data_name == "hpoly":
        cell_type_to_monitor = None
        stim_key = "Hpoly.Day10"
        ctrl_key = "Control"
        cell_type_key = "cell_label"
        train = sc.read("../data/ch10_train_7000.h5ad")
    elif data_name == "salmonella":
        cell_type_to_monitor = None
        stim_key = "Salmonella"
        ctrl_key = "Control"
        cell_type_key = "cell_label"
        train = sc.read("../data/chsal_train_7000.h5ad")
    elif data_name == "species":
        cell_type_to_monitor = "rat"
        stim_key = "LPS6"
        ctrl_key = "unst"
        cell_type_key = "species"
        train = sc.read("../data/train_all_lps6.h5ad")

    for cell_type in train.obs[cell_type_key].unique().tolist():
        os.makedirs(f"./vae_results/{data_name}/{cell_type}/", exist_ok=True)
        os.chdir(f"./vae_results/{data_name}/{cell_type}")
        net_train_data = train[~((train.obs[cell_type_key] == cell_type) & (train.obs[condition_key] == stim_key))]
        network = scgen.VAEArith(x_dimension=net_train_data.X.shape[1],
                                 z_dimension=z_dim,
                                 alpha=alpha,
                                 dropout_rate=dropout_rate,
                                 learning_rate=learning_rate)

        # network.restore_model()
        network.train(net_train_data, n_epochs=n_epochs, batch_size=batch_size)
        print(f"network_{cell_type} has been trained!")

        scgen.visualize_trained_network_results(network, train, cell_type,
                                                conditions={"ctrl": ctrl_key, "stim": stim_key},
                                                condition_key="condition", cell_type_key=cell_type_key,
                                                path_to_save="./figures/tensorflow/")
        os.chdir("../../../")


def reconstruct_whole_data(data_name="pbmc", condition_key="condition"):
    if data_name == "pbmc":
        stim_key = "stimulated"
        ctrl_key = "control"
        cell_type_key = "cell_type"
        train = sc.read("../data/train.h5ad")
    elif data_name == "hpoly":
        stim_key = "Hpoly.Day10"
        ctrl_key = "Control"
        cell_type_key = "cell_label"
        train = sc.read("../data/ch10_train_7000.h5ad")
    elif data_name == "salmonella":
        stim_key = "Salmonella"
        ctrl_key = "Control"
        cell_type_key = "cell_label"
        train = sc.read("../data/chsal_train_7000.h5ad")
    elif data_name == "species":
        stim_key = "LPS6"
        ctrl_key = "unst"
        cell_type_key = "species"
        train = sc.read("../data/train_all_lps6.h5ad")
    all_data = anndata.AnnData()
    for idx, cell_type in enumerate(train.obs[cell_type_key].unique().tolist()):
        print(f"Reconstructing for {cell_type}")
        os.chdir(f"./vae_results/{data_name}/{cell_type}")
        network = scgen.VAEArith(x_dimension=train.X.shape[1],
                                 z_dimension=100,
                                 alpha=0.00005,
                                 dropout_rate=0.2,
                                 learning_rate=0.001)
        network.restore_model()

        cell_type_data = train[train.obs[cell_type_key] == cell_type]
        cell_type_ctrl_data = train[((train.obs[cell_type_key] == cell_type) & (train.obs[condition_key] == ctrl_key))]
        pred, delta = network.predict(adata=cell_type_data,
                                      conditions={"ctrl": ctrl_key, "stim": stim_key},
                                      cell_type_key=cell_type_key,
                                      condition_key=condition_key,
                                      celltype_to_predict=cell_type)

        pred_adata = anndata.AnnData(pred, obs={condition_key: [f"{cell_type}_pred_stim"] * len(pred),
                                                cell_type_key: [cell_type] * len(pred)},
                                     var={"var_names": cell_type_data.var_names})
        ctrl_adata = anndata.AnnData(cell_type_ctrl_data.X,
                                     obs={condition_key: [f"{cell_type}_ctrl"] * len(cell_type_ctrl_data),
                                          cell_type_key: [cell_type] * len(cell_type_ctrl_data)},
                                     var={"var_names": cell_type_ctrl_data.var_names})
        if sparse.issparse(cell_type_data.X):
            real_stim = cell_type_data[cell_type_data.obs[condition_key] == stim_key].X.A
        else:
            real_stim = cell_type_data[cell_type_data.obs[condition_key] == stim_key].X
        real_stim_adata = anndata.AnnData(real_stim,
                                          obs={condition_key: [f"{cell_type}_real_stim"] * len(real_stim),
                                               cell_type_key: [cell_type] * len(real_stim)},
                                          var={"var_names": cell_type_data.var_names})
        if idx == 0:
            all_data = ctrl_adata.concatenate(pred_adata, real_stim_adata)
        else:
            all_data = all_data.concatenate(ctrl_adata, pred_adata, real_stim_adata)

        os.chdir("../../../")
        print(f"Finish Reconstructing for {cell_type}")
    all_data.write_h5ad(f"./vae_results/{data_name}/reconstructed.h5ad")


def score(adata, n_deg=10, n_genes=1000, condition_key="condition",
          conditions={"stim": "stimulated", "ctrl": "control"},
          sortby="median_score"):
    import scanpy as sc
    import numpy as np
    from scipy.stats import entropy
    import pandas as pd
    sc.tl.rank_genes_groups(adata, groupby=condition_key, method="wilcoxon", n_genes=2 * n_genes)
    gene_names = adata.uns["rank_genes_groups"]['names'][conditions['stim']]
    gene_lfcs = adata.uns["rank_genes_groups"]['logfoldchanges'][conditions['stim']]
    diff_genes_df = pd.DataFrame({"names": gene_names, "lfc": gene_lfcs})
    diff_genes = diff_genes_df[diff_genes_df['lfc'] > 0.3]["names"].tolist()
    diff_genes = diff_genes[:n_genes]
    # gene_list = ["CCDC50", "MYBL2", "RBMS1", "CHI3L2", "AZI2", "CSRP2", "PGAP1", "PIM2", "DNPEP", "EDEM1"]
    # print(diff_genes_df[(diff_genes_df['lfc'] > 0.3) & (diff_genes_df["names"].isin(gene_list))])
    # exit()

    adata_deg = adata[:, diff_genes].copy()
    cell_types = adata_deg.obs["cell_type"].cat.categories.tolist()
    lfc_temp = np.zeros((len(cell_types), n_genes))
    for j, ct in enumerate(cell_types):
        stim = adata_deg[(adata_deg.obs["cell_type"] == ct) &
                         (adata_deg.obs[condition_key] == conditions["stim"])].X.mean(0).A1
        ctrl = adata_deg[(adata_deg.obs["cell_type"] == ct) &
                         (adata_deg.obs[condition_key] == conditions["ctrl"])].X.mean(0).A1
        lfc_temp[j] = np.abs((stim - ctrl)[None, :])
    norm_lfc = lfc_temp / lfc_temp.sum(0).reshape((1, n_genes))
    ent_scores = entropy(norm_lfc)
    median = np.median(lfc_temp, axis=0)
    med_scores = np.max(np.abs((lfc_temp - median)), axis=0)
    df_score = pd.DataFrame({"genes": adata_deg.var_names.tolist(), "median_score": med_scores,
                             "entropy_score": ent_scores})
    if sortby == "median_score":
        return df_score.sort_values(by=['median_score'], ascending=False).iloc[:n_deg, :]
    else:
        return df_score.sort_values(by=['entropy_score']).iloc[:n_deg, :]


def rank_genes(data, diff_names, conditions={}, n_genes=100, start=0):
    data = data[:, diff_names]

    ctrl_data = data.copy()[data.obs["condition"] == conditions["ctrl"]]
    stim_data = data.copy()[data.obs["condition"] == conditions["stim"]]
    if sparse.issparse(ctrl_data.X):
        ctrl_avg = np.average(ctrl_data.X.A, axis=0)
        stim_avg = np.average(stim_data.X.A, axis=0)
    else:
        ctrl_avg = np.average(ctrl_data.X, axis=0)
        stim_avg = np.average(stim_data.X, axis=0)
    gene_lfcs = np.array(stim_avg - ctrl_avg)
    gene_scores = gene_lfcs - np.median(gene_lfcs)
    gene_df = pd.DataFrame({"score": gene_scores}, index=diff_names)
    gene_df.sort_values(by=["score"], ascending=False)
    return gene_df.index.tolist()[start: start + n_genes]


def calc_score(data_name="pbmc", n_genes=100, restore=True):
    if data_name == "pbmc":
        stim_key = "stimulated"
        ctrl_key = "control"
        cell_type_key = "cell_type"
        train = sc.read("../data/train.h5ad")
    elif data_name == "hpoly":
        stim_key = "Hpoly.Day10"
        ctrl_key = "Control"
        cell_type_key = "cell_label"
        train = sc.read("../data/ch10_train_7000.h5ad")
    elif data_name == "salmonella":
        stim_key = "Salmonella"
        ctrl_key = "Control"
        cell_type_key = "cell_label"
        train = sc.read("../data/chsal_train_7000.h5ad")
    elif data_name == "species":
        stim_key = "LPS6"
        ctrl_key = "unst"
        cell_type_key = "species"
        train = sc.read("../data/train_all_lps6.h5ad")
    conditions = {"ctrl": ctrl_key, "stim": stim_key}
    recon_data = sc.read(f"./vae_results/{data_name}/reconstructed.h5ad")
    diff_genes = score(train, n_deg=10 * n_genes, n_genes=1000, conditions=conditions, sortby="median_score")[
        "genes"].tolist()
    top_50 = diff_genes[-50:]
    sc.pl.stacked_violin(recon_data,
                         var_names=top_50[:10],
                         groupby="condition",
                         save=f"_Top_{10}_Median_genes_out_of_50",
                         swap_axes=True,
                         show=False)
    exit()
    # plot_reg_mean_with_genes("pbmc", gene_list=top_50[:10])
    if not restore:
        all_scores = np.zeros(shape=(7 * 10 * n_genes, 1))
        for bin_idx in range(10):
            for idx, cell_type in enumerate(train.obs[cell_type_key].unique().tolist()):
                print(f"Running for {cell_type} ...")
                real_stim = recon_data[(recon_data.obs[cell_type_key] == cell_type) & (
                        recon_data.obs["condition"] == f"{cell_type}_real_stim")]
                pred_stim = recon_data[(recon_data.obs[cell_type_key] == cell_type) & (
                        recon_data.obs["condition"] == f"{cell_type}_pred_stim")]

                real_stim = real_stim[:, diff_genes[bin_idx * n_genes:(bin_idx + 1) * n_genes]]
                pred_stim = pred_stim[:, diff_genes[bin_idx * n_genes:(bin_idx + 1) * n_genes]]
                if sparse.issparse(real_stim.X):
                    real_stim_avg = np.average(real_stim.X.A, axis=0)
                    pred_stim_avg = np.average(pred_stim.X.A, axis=0)
                else:
                    real_stim_avg = np.average(real_stim.X, axis=0)
                    pred_stim_avg = np.average(pred_stim.X, axis=0)

                absolute_values = np.abs(np.square(real_stim_avg - pred_stim_avg))
                absolute_values = np.reshape(absolute_values, (-1, 1))
                normalized_abs_values = np.reshape(absolute_values, (-1,))

                start = 7 * n_genes * bin_idx
                all_scores[start + n_genes * idx:start + n_genes * (idx + 1), 0] = normalized_abs_values
        np.savetxt(X=all_scores.T, fname=f"./vae_results/all_scores_{n_genes}.txt", delimiter=",")
        all_scores = np.reshape(all_scores, (-1,))
    else:
        all_scores = np.loadtxt(fname=f"./vae_results/all_scores_{n_genes}.txt", delimiter=",")
    import seaborn as sns
    conditions = [f"Bin-{i // (7 * n_genes) + 1}" for i in range(7 * 10 * n_genes)]
    print(all_scores.shape)
    all_scores_df = pd.DataFrame({"scores": all_scores})
    all_scores_df["conditions"] = conditions
    ax = sns.boxplot(data=all_scores_df, x="conditions", y="scores", whis=np.inf)
    # ax = sns.swarmplot(x="conditions", y="scores", data=all_scores_df, color=".01")
    # plt.show()
    os.chdir(f"./vae_results/{data_name}/")
    sc.pl.stacked_violin(recon_data,
                         var_names=diff_genes[:10],
                         groupby="condition",
                         save=f"_Top_{10}_Median_genes_From_Top_{10 * n_genes}_DEGs",
                         swap_axes=True,
                         show=False)
    plt.savefig(f"./{data_name}_boxplot_scores_scaled_{n_genes}.pdf")
    plt.close()
    sc.settings.figdir = os.getcwd()
    all_scores = np.reshape(all_scores, (-1, 1))
    adata = anndata.AnnData(X=all_scores, obs={"condition": conditions}, var={"var_names": ["0"]})
    # sc.pl.violin(adata, keys="0", groupby="condition", save=f"_scores_scaled_{n_genes}.pdf", show=False)
    os.chdir("../../")


def violin_plot(data_name="pbmc"):
    if data_name == "pbmc":
        stim_key = "stimulated"
        ctrl_key = "control"
        cell_type_key = "cell_type"
        train = sc.read("../data/train.h5ad")
    elif data_name == "hpoly":
        stim_key = "Hpoly.Day10"
        ctrl_key = "Control"
        cell_type_key = "cell_label"
        train = sc.read("../data/ch10_train_7000.h5ad")
    elif data_name == "salmonella":
        stim_key = "Salmonella"
        ctrl_key = "Control"
        cell_type_key = "cell_label"
        train = sc.read("../data/chsal_train_7000.h5ad")
    elif data_name == "species":
        stim_key = "LPS6"
        ctrl_key = "unst"
        cell_type_key = "species"
        train = sc.read("../data/train_all_lps6.h5ad")
    conditions = {"ctrl": ctrl_key, "stim": stim_key}
    recon_data = sc.read(f"./vae_results/{data_name}/reconstructed.h5ad")

    for idx, cell_type in enumerate(train.obs[cell_type_key].unique().tolist()):
        os.makedirs(f"./vae_results/{data_name}/violin_plots/", exist_ok=True)
        cell_type_data = train[train.obs[cell_type_key] == cell_type]
        sc.tl.rank_genes_groups(cell_type_data, groupby="condition", n_genes=10, method="wilcoxon")
        gene_names = cell_type_data.uns["rank_genes_groups"]['names'][conditions['stim']]
        gene_lfcs = cell_type_data.uns["rank_genes_groups"]['logfoldchanges'][conditions['stim']]
        diff_genes_df = pd.DataFrame({"names": gene_names, "lfc": gene_lfcs})
        diff_genes = diff_genes_df[diff_genes_df['lfc'] > 0.2]["names"]

        recon_cell_type = recon_data[(recon_data.obs[cell_type_key] == cell_type)]
        sc.settings.figdir = f"./vae_results/{data_name}/violin_plots/"
        # for i in range(10):
        #         #     sc.pl.violin(recon_cell_type, keys=diff_genes.tolist()[i], groupby="condition",
        #                  save=f"_{cell_type}_{diff_genes.tolist()[i]}",
        #                  show=False)
        sc.pl.stacked_violin(recon_cell_type, var_names=diff_genes.tolist(), groupby="condition",
                             save=f"_{cell_type}_Top_DE_genes", swap_axes=True,
                             show=False)
    # sc.pl.rank_genes_groups_stacked_violin(recon_data, n_genes=10, groupby="condition", save="test.pdf", show=False)


def plot_reg_mean_with_genes(data_name="pbmc", gene_list=None):
    if data_name == "pbmc":
        stim_key = "stimulated"
        ctrl_key = "control"
        cell_type_key = "cell_type"
        train = sc.read("../data/train.h5ad")
    elif data_name == "hpoly":
        stim_key = "Hpoly.Day10"
        ctrl_key = "Control"
        cell_type_key = "cell_label"
        train = sc.read("../data/ch10_train_7000.h5ad")
    elif data_name == "salmonella":
        stim_key = "Salmonella"
        ctrl_key = "Control"
        cell_type_key = "cell_label"
        train = sc.read("../data/chsal_train_7000.h5ad")
    elif data_name == "species":
        stim_key = "LPS6"
        ctrl_key = "unst"
        cell_type_key = "species"
        train = sc.read("../data/train_all_lps6.h5ad")
    recon_data = sc.read(f"./vae_results/{data_name}/reconstructed.h5ad")
    import scgen
    for cell_type in train.obs[cell_type_key].unique().tolist():
        adata = recon_data[:, gene_list]
        scgen.plotting.reg_mean_plot(adata, condition_key="condition",
                                     axis_keys={"x": f"{cell_type}_pred_stim", "y": f"{cell_type}_real_stim"},
                                     gene_list=gene_list[:5],
                                     path_to_save=f"./vae_results/{data_name}/{cell_type}/reg_mean.pdf")
    exit()


if __name__ == '__main__':
    # test_train_whole_data_one_celltype_out(data_name="hpoly",
    #                                        z_dim=100,
    #                                        alpha=0.00005,
    #                                        n_epochs=300,
    #                                        batch_size=32,
    #                                        dropout_rate=0.2,
    #                                        learning_rate=0.001,
    #                                        condition_key="condition")
    # reconstruct_whole_data(data_name="pbmc", condition_key="condition")
    # reconstruct_whole_data(data_name="hpoly", condition_key="condition")
    # reconstruct_whole_data(data_name="salmonella", condition_key="condition")
    # calc_score(data_name="pbmc", n_genes=20, restore=False)
    # calc_score(data_name="pbmc", n_genes=25, restore=False)
    calc_score(data_name="pbmc", n_genes=30, restore=False)
    # calc_score(data_name="pbmc", n_genes=50, restore=False)
    # calc_score(data_name="pbmc", n_genes=75, restore=False)
    # calc_score(data_name="pbmc", n_genes=100, restore=False)
    # calc_score(data_name="pbmc", n_genes=200, restore=False)
    # violin_plot(data_name="pbmc")
    # violin_plot(data_name="hpoly")
    # violin_plot(data_name="salmonella")
