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
    elif data_name == "study":
        stim_key = "stimulated"
        ctrl_key = "control"
        cell_type_key = "cell_type"
        train = sc.read("../data/kang_cross_train.h5ad")

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
    elif data_name == "study":
        stim_key = "stimulated"
        ctrl_key = "control"
        cell_type_key = "cell_type"
        train = sc.read("../data/kang_cross_train.h5ad")

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


def score(adata, n_deg=10, n_genes=1000, condition_key="condition", cell_type_key="cell_type",
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
    cell_types = adata_deg.obs[cell_type_key].cat.categories.tolist()
    lfc_temp = np.zeros((len(cell_types), n_genes))
    for j, ct in enumerate(cell_types):
        if cell_type_key == "cell_type":  # if data is pbmc
            stim = adata_deg[(adata_deg.obs[cell_type_key] == ct) &
                             (adata_deg.obs[condition_key] == conditions["stim"])].X.mean(0).A1
            ctrl = adata_deg[(adata_deg.obs[cell_type_key] == ct) &
                             (adata_deg.obs[condition_key] == conditions["ctrl"])].X.mean(0).A1
        else:
            stim = adata_deg[(adata_deg.obs[cell_type_key] == ct) &
                             (adata_deg.obs[condition_key] == conditions["stim"])].X.mean(0)
            ctrl = adata_deg[(adata_deg.obs[cell_type_key] == ct) &
                             (adata_deg.obs[condition_key] == conditions["ctrl"])].X.mean(0)
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
        return df_score.sort_values(by=['entropy_score'], ascending=False).iloc[:n_deg, :]


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


def plot_boxplot(data_name="pbmc", n_genes=100, restore=True, score_type="median_score", y_measure="SE"):
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

    os.chdir(f"./vae_results/{data_name}/")
    sc.settings.figdir = os.getcwd()

    recon_data = sc.read(f"./reconstructed.h5ad")
    diff_genes = score(train, n_deg=10 * n_genes, n_genes=1000, cell_type_key=cell_type_key, conditions=conditions,
                       sortby=score_type)
    diff_genes = diff_genes["genes"].tolist()
    # epsilon = 1e-7
    os.makedirs(f"./boxplots/Top_{10 * n_genes}/{y_measure}/", exist_ok=True)
    if not restore:
        n_cell_types = len(train.obs[cell_type_key].unique().tolist())
        all_scores = np.zeros(shape=(n_cell_types * 10 * n_genes, 1))
        for bin_idx in range(10):
            for cell_type_idx, cell_type in enumerate(train.obs[cell_type_key].unique().tolist()):
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
                if y_measure == "SE":  # (x - xhat) ^ 2
                    y_measures = np.abs(np.square(real_stim_avg - pred_stim_avg))
                    y_measures_reshaped = np.reshape(y_measures, (-1,))
                elif y_measure == "AE":  # x - xhat
                    y_measures = np.abs(real_stim_avg - pred_stim_avg)
                    y_measures_reshaped = np.reshape(y_measures, (-1,))
                elif y_measure == "AE:x":  # (x - xhat) / x
                    y_measures = np.abs(real_stim_avg - pred_stim_avg)
                    y_measures = np.divide(y_measures, real_stim_avg)
                    y_measures_reshaped = np.reshape(y_measures, (-1,))
                elif y_measure == "SE:x^2":  # (x - xhat) / x^2
                    y_measures = np.abs(np.square(real_stim_avg - pred_stim_avg))
                    y_measures = np.divide(y_measures, np.power(real_stim_avg, 2))
                    y_measures_reshaped = np.reshape(y_measures, (-1,))
                elif y_measure == "AE:max(x, 1)":  # (x - xhat) / max(x, 1)
                    y_measures = np.abs(real_stim_avg - pred_stim_avg)
                    y_measures = np.divide(y_measures, np.maximum(real_stim_avg, 1.0))
                    y_measures_reshaped = np.reshape(y_measures, (-1,))
                elif y_measure == "SE:max(x, 1)^2":  # (x - xhat)^2 / max(x, 1)^2
                    y_measures = np.abs(np.square(real_stim_avg - pred_stim_avg))
                    y_measures = np.divide(y_measures, np.power(np.maximum(real_stim_avg, 1.0), 2))
                    y_measures_reshaped = np.reshape(y_measures, (-1,))
                elif y_measure == "1 - AE:x":  # 1 - ((x - xhat) / x)
                    y_measures = np.abs(real_stim_avg - pred_stim_avg)
                    y_measures = np.divide(y_measures, real_stim_avg)
                    y_measures = np.abs(1.0 - y_measures)
                    y_measures_reshaped = np.reshape(y_measures, (-1,))
                elif y_measure == "1 - SE:x^2":  # 1 - ((x - xhat) / x)^2
                    y_measures = np.abs(np.square(real_stim_avg - pred_stim_avg))
                    y_measures = np.divide(y_measures, np.power(real_stim_avg, 2))
                    y_measures = np.abs(1.0 - y_measures)
                    y_measures_reshaped = np.reshape(y_measures, (-1,))
                elif y_measure == "1 - AE:max(x, 1)":  # 1 - ((x - xhat) / max(x, 1.0))
                    y_measures = np.abs(real_stim_avg - pred_stim_avg)
                    y_measures = np.true_divide(y_measures, np.maximum(real_stim_avg, 1.0))
                    y_measures = np.abs(1.0 - y_measures)
                    y_measures_reshaped = np.reshape(y_measures, (-1,))
                elif y_measure == "1 - SE:max(x, 1)^2":  # 1 - ((x - xhat) / max(x, 1.0))
                    y_measures = np.abs(np.square(real_stim_avg - pred_stim_avg))
                    y_measures = np.true_divide(y_measures, np.power(np.maximum(real_stim_avg, 1.0), 2))
                    y_measures = np.abs(1.0 - y_measures)
                    y_measures_reshaped = np.reshape(y_measures, (-1,))

                start = n_cell_types * n_genes * bin_idx
                all_scores[start + n_genes * cell_type_idx:start + n_genes * (cell_type_idx + 1),
                0] = y_measures_reshaped
        np.savetxt(X=all_scores.T,
                   fname=f"./boxplots/Top_{10 * n_genes}/{y_measure}/y_measures_{score_type}_{n_genes}_({y_measure}).txt",
                   delimiter=",")
        all_scores = np.reshape(all_scores, (-1,))
    else:
        all_scores = np.loadtxt(
            fname=f"./boxplots/Top_{10 * n_genes}/{y_measure}/y_measures_{score_type}_{n_genes}_({y_measure}).txt",
            delimiter=",")
    import seaborn as sns
    conditions = [f"Bin-{i // (n_cell_types * n_genes) + 1}" for i in range(n_cell_types * 10 * n_genes)]
    all_scores_df = pd.DataFrame({"scores": all_scores})
    all_scores_df["conditions"] = conditions
    ax = sns.boxplot(data=all_scores_df, x="conditions", y="scores", whis=np.inf)
    if y_measure == "SE":
        plt.ylabel("(x - xhat) ^ 2")
    elif y_measure == "AE":
        plt.ylabel("x - xhat")
    elif y_measure == "AE:x":
        plt.ylabel("(x - xhat) / x")
    elif y_measure == "SE:x^2":
        plt.ylabel("((x - xhat) ^ 2) / (x ^ 2)")
    elif y_measure == "AE:max(x, 1)":
        plt.ylabel("(x - xhat) / max(x, 1)")
    elif y_measure == "SE:max(x, 1)^2":
        plt.ylabel("(x - xhat)^2 / max(x, 1)^2")
    elif y_measure == "1 - AE:x":
        plt.ylabel("1 - ((x - xhat) / x)")
    elif y_measure == "1 - SE:x^2":
        plt.ylabel("1 - ((x - xhat)^2 / x^2)")
    elif y_measure == "1 - AE:max(x, 1)":
        plt.ylabel("1 - ((x - xhat) / max(x, 1))")
    elif y_measure == "1 - SE:max(x, 1)^2":
        plt.ylabel("1 - ((x - xhat)^2 / max(x, 1)^2)")
    os.makedirs(f"./boxplots/Top_{10 * n_genes}/{y_measure}/", exist_ok=True)
    plt.savefig(f"./boxplots/Top_{10 * n_genes}/{y_measure}/boxplot_{score_type}_{n_genes}_({y_measure}).pdf")
    plt.close()

    all_scores = np.reshape(all_scores, (-1, 1))
    # adata = anndata.AnnData(X=all_scores, obs={"condition": conditions}, var={"var_names": ["0"]})
    # sc.pl.violin(adata,
    #              keys="0",
    #              groupby="condition",
    #              save=f"_scores_{y_measure}_{n_genes}.pdf",
    #              show=False)
    os.chdir("../../")


def stacked_violin_plot(data_name="pbmc", score_type="median_score"):
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

    elif data_name == "pca":
        stim_key = "stimulated"
        ctrl_key = "control"
        cell_type_key = "cell_type"
        train = sc.read("../data/train.h5ad")
    conditions = {"ctrl": ctrl_key, "stim": stim_key}

    # os.chdir(f"./vae_results/{data_name}/")
    # sc.settings.figdir = os.getcwd()

    recon_data = sc.read(f"../data/PCA.h5ad")
    diff_genes = score(train, n_deg=500, n_genes=1000, cell_type_key=cell_type_key, conditions=conditions,
                       sortby=score_type)
    diff_genes = diff_genes["genes"].tolist()
    sc.pl.stacked_violin(recon_data,
                         var_names=diff_genes[:10],
                         groupby="condition",
                         save=f"_Top_{10}_{score_type}_genes_out_of_500_{data_name}",
                         swap_axes=True,
                         show=False)
    os.chdir("../../")


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


def train(data_name="study",
          z_dim=100,
          alpha=0.00005,
          n_epochs=300,
          batch_size=32,
          dropout_rate=0.2,
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
    elif data_name == "study":
        stim_key = "stimulated"
        ctrl_key = "control"
        cell_type_key = "cell_type"
        train = sc.read("../data/kang_cross_train.h5ad")

    os.makedirs(f"./vae_results/{data_name}/whole/", exist_ok=True)
    os.chdir(f"./vae_results/{data_name}/whole/")
    net_train_data = train
    network = scgen.VAEArith(x_dimension=net_train_data.X.shape[1],
                             z_dimension=z_dim,
                             alpha=alpha,
                             dropout_rate=dropout_rate,
                             learning_rate=learning_rate)

    # network.restore_model()
    network.train(net_train_data, n_epochs=n_epochs, batch_size=batch_size)
    print(f"network has been trained!")

    # scgen.visualize_trained_network_results(network, train, cell_type,
    #                                         conditions={"ctrl": ctrl_key, "stim": stim_key},
    #                                         condition_key="condition", cell_type_key=cell_type_key,
    #                                         path_to_save="./figures/tensorflow/")
    os.chdir("../../../")


def test_train_whole_data_some_celltypes_out(data_name="study",
                                             z_dim=100,
                                             alpha=0.00005,
                                             n_epochs=300,
                                             batch_size=32,
                                             dropout_rate=0.2,
                                             learning_rate=0.001,
                                             condition_key="condition",
                                             c_out=None,
                                             c_in=None):
    if data_name == "pbmc":
        stim_key = "stimulated"
        ctrl_key = "control"
        cell_type_key = "cell_type"
        train = sc.read("../data/train.h5ad")

    os.makedirs(f"./vae_results/{data_name}/heldout/{len(c_out)}/", exist_ok=True)
    os.chdir(f"./vae_results/{data_name}/heldout/{len(c_out)}/")

    net_train_data = scgen.data_remover(train, remain_list=c_in, remove_list=c_out,
                                        cell_type_key=cell_type_key, condition_key=condition_key)

    print(net_train_data)

    network = scgen.VAEArith(x_dimension=net_train_data.X.shape[1],
                             z_dimension=z_dim,
                             alpha=alpha,
                             dropout_rate=dropout_rate,
                             learning_rate=learning_rate)

    # network.restore_model()
    network.train(net_train_data, n_epochs=n_epochs, batch_size=batch_size)
    print(f"network has been trained!")
    os.chdir("../../../../")


def train_cross_study(data_name="study",
                      z_dim=100,
                      alpha=0.00005,
                      n_epochs=300,
                      batch_size=32,
                      dropout_rate=0.2,
                      learning_rate=0.001,
                      condition_key="condition"):
    stim_key = "stimulated"
    ctrl_key = "control"
    cell_type_key = "cell_type"
    train = sc.read("../data/kang_cross_train.h5ad")

    os.makedirs(f"./vae_results/{data_name}/all/", exist_ok=True)
    os.chdir(f"./vae_results/{data_name}/all/")
    net_train_data = train
    network = scgen.VAEArith(x_dimension=net_train_data.X.shape[1],
                             z_dimension=z_dim,
                             alpha=alpha,
                             dropout_rate=dropout_rate,
                             learning_rate=learning_rate)

    # network.restore_model()
    network.train(net_train_data, n_epochs=n_epochs, batch_size=batch_size)
    print(f"network_{data_name} has been trained!")

    os.chdir("../../../")


def train_batch_removal(data_name="study",
                        z_dim=100,
                        alpha=0.00005,
                        n_epochs=300,
                        batch_size=32,
                        dropout_rate=0.2,
                        learning_rate=0.001,
                        condition_key="condition"):
    stim_key = "stimulated"
    ctrl_key = "control"
    cell_type_key = "cell_type"
    train = sc.read("../data/kang_cross_train.h5ad")

    os.makedirs(f"./vae_results/{data_name}/all/", exist_ok=True)
    os.chdir(f"./vae_results/{data_name}/all/")
    net_train_data = train
    network = scgen.VAEArith(x_dimension=net_train_data.X.shape[1],
                             z_dimension=z_dim,
                             alpha=alpha,
                             dropout_rate=dropout_rate,
                             learning_rate=learning_rate)

    # network.restore_model()
    network.train(net_train_data, n_epochs=n_epochs, batch_size=batch_size)
    print(f"network_{data_name} has been trained!")

    os.chdir("../../../")


if __name__ == '__main__':
    # c_in = ['NK', 'B', 'CD14+Mono']
    # c_out = ['CD4T', 'FCGR3A+Mono', 'CD8T', 'Dendritic']
    # test_train_whole_data_some_celltypes_out(data_name="pbmc",
    #                                          z_dim=100,
    #                                          alpha=0.00005,
    #                                          n_epochs=300,
    #                                          batch_size=32,
    #                                          dropout_rate=0.2,
    #                                          learning_rate=0.001,
    #                                          condition_key="condition",
    #                                          c_out=c_out,
    #                                          c_in=c_in)
    # c_in = ['CD14+Mono']
    # c_out = ['CD4T', 'FCGR3A+Mono', 'CD8T', 'NK', 'B', 'Dendritic']
    # test_train_whole_data_some_celltypes_out(data_name="pbmc",
    #                                          z_dim=100,
    #                                          alpha=0.00005,
    #                                          n_epochs=300,
    #                                          batch_size=32,
    #                                          dropout_rate=0.2,
    #                                          learning_rate=0.001,
    #                                          condition_key="condition",
    #                                          c_out=c_out,
    #                                          c_in=c_in)
    # c_in = ['CD8T', 'NK', 'B', 'Dendritic', 'CD14+Mono']
    # c_out = ['CD4T', 'FCGR3A+Mono']
    # test_train_whole_data_some_celltypes_out(data_name="pbmc",
    #                                          z_dim=100,
    #                                          alpha=0.00005,
    #                                          n_epochs=300,
    #                                          batch_size=32,
    #                                          dropout_rate=0.2,
    #                                          learning_rate=0.001,
    #                                          condition_key="condition",
    #                                          c_out=c_out,
    #                                          c_in=c_in)
    # train(data_name="study",
    #       z_dim=100,
    #       alpha=0.00005,
    #       n_epochs=300,
    #       batch_size=32,
    #       dropout_rate=0.2,
    #       learning_rate=0.001,
    #       condition_key="condition")
    # test_train_whole_data_one_celltype_out(data_name="study",
    #                                        z_dim=100,
    #                                        alpha=0.00005,
    #                                        n_epochs=300,
    #                                        batch_size=32,
    #                                        dropout_rate=0.2,
    #                                        learning_rate=0.001,
    #                                        condition_key="condition")
    # reconstruct_whole_data(data_name="study", condition_key="condition")
    # reconstruct_whole_data(data_name="hpoly", condition_key="condition")
    # reconstruct_whole_data(data_name="salmonella", condition_key="condition")
    # for data_name in ["pbmc", "hpoly", "salmonella"]:
    #     for score_type in ["median_score", "entropy_score"]:
    #         stacked_violin_plot(data_name, score_type)
    #         for n_genes in [20, 30, 40, 50, 70, 100]:
    #             for y_measure in ["SE", "SE:x^2", "SE:max(x, 1)^2", "1 - SE:x^2", "1 - SE:max(x, 1)^2",
    #                               "AE", "AE:x",   "AE:max(x, 1)",   "1 - AE:x",   "1 - AE:max(x, 1)"]:
    #                 print(data_name, n_genes, y_measure)
    #                 plot_boxplot(data_name=data_name, n_genes=n_genes, restore=False, score_type=score_type,
    #                              y_measure=y_measure)
    # stacked_violin_plot(data_name="pca")
    test_train_whole_data_one_celltype_out(data_name="study",
                                           z_dim=100,
                                           alpha=0.00005,
                                           n_epochs=300,
                                           batch_size=32,
                                           dropout_rate=0.2,
                                           learning_rate=0.001,
                                           condition_key="condition")
    train_batch_removal(data_name="pancreas",
                        z_dim=100,
                        alpha=0.00005,
                        n_epochs=300,
                        batch_size=32,
                        dropout_rate=0.2,
                        learning_rate=0.001,
                        condition_key="condition")
