import os

import anndata
import numpy as np
import scanpy as sc

import scgen

if not os.getcwd().endswith("tests"):
    os.chdir("./tests")


# from datetime import datetime, timezone

# current_time = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H:%M:%S")
# os.makedirs(current_time, exist_ok=True)
# os.chdir("./" + current_time)


def test_train_whole_data_one_celltype_out(z_dim=50, alpha=0.001, beta=100, kernel="multi-scale-rbf", n_epochs=1000,
                                           batch_size=1024):
    train = sc.read("../data/ch10_train_7000.h5ad", backup_url="https://goo.gl/33HtVh")
    for cell_type in train.obs["cell_label"].unique().tolist():
        os.makedirs(f"./results/{cell_type}/", exist_ok=True)
        os.chdir(f"./results/{cell_type}")
        net_train_data = train[~((train.obs["cell_label"] == cell_type) & (train.obs["condition"] == "stimulated"))]

        network = scgen.MMDCVAE(x_dimension=net_train_data.X.shape[1], z_dimension=z_dim, alpha=alpha, beta=beta,
                                batch_mmd=True, kernel=kernel, train_with_fake_labels=False,
                                model_path=f"./")

        # network.restore_model()
        network.train(net_train_data, n_epochs=n_epochs, batch_size=batch_size, verbose=2)
        print(f"network_{cell_type} has been trained!")

        true_labels, _ = scgen.label_encoder(net_train_data)
        fake_labels = np.ones(shape=(net_train_data.shape[0], 1))

        latent_with_true_labels = network.to_latent(net_train_data.X, labels=true_labels)
        latent_with_true_labels = sc.AnnData(X=latent_with_true_labels,
                                             obs={"condition": net_train_data.obs["condition"].tolist(),
                                                  "cell_label": net_train_data.obs["cell_label"].tolist()})
        sc.pp.neighbors(latent_with_true_labels)
        sc.tl.umap(latent_with_true_labels)
        sc.pl.umap(latent_with_true_labels, color=["condition", "cell_label"],
                   save=f"_latent_true_labels_{z_dim}",
                   show=False)

        latent_with_fake_labels = network.to_latent(net_train_data.X, fake_labels)
        latent_with_fake_labels = sc.AnnData(X=latent_with_fake_labels,
                                             obs={"condition": net_train_data.obs["condition"].tolist(),
                                                  "cell_label": net_train_data.obs["cell_label"].tolist()})
        sc.pp.neighbors(latent_with_fake_labels)
        sc.tl.umap(latent_with_fake_labels)
        sc.pl.umap(latent_with_fake_labels, color=["condition", "cell_label"],
                   save=f"_latent_fake_labels_{z_dim}",
                   show=False)

        mmd_with_true_labels = network.to_mmd_layer(network.cvae_model, net_train_data.X,
                                                    encoder_labels=true_labels, feed_fake=False)
        mmd_with_true_labels = sc.AnnData(X=mmd_with_true_labels,
                                          obs={"condition": net_train_data.obs["condition"].tolist(),
                                               "cell_label": net_train_data.obs["cell_label"].tolist()})
        sc.pp.neighbors(mmd_with_true_labels)
        sc.tl.umap(mmd_with_true_labels)
        sc.pl.umap(mmd_with_true_labels, color=["condition", "cell_label"],
                   save=f"_mmd_true_labels_{z_dim}",
                   show=False)

        mmd_with_fake_labels = network.to_mmd_layer(network.cvae_model, net_train_data.X,
                                                    encoder_labels=true_labels, feed_fake=True)
        mmd_with_fake_labels = sc.AnnData(X=mmd_with_fake_labels,
                                          obs={"condition": net_train_data.obs["condition"].tolist(),
                                               "cell_label": net_train_data.obs["cell_label"].tolist()})
        sc.pp.neighbors(mmd_with_fake_labels)
        sc.tl.umap(mmd_with_fake_labels)
        sc.pl.umap(mmd_with_fake_labels, color=["condition", "cell_label"],
                   save=f"_mmd_fake_labels_{z_dim}",
                   show=False)

        decoded_latent_with_true_labels = network.predict(data=latent_with_true_labels, labels=true_labels,
                                                          data_space='latent')

        cell_type_data = train[train.obs["cell_label"] == cell_type]
        unperturbed_data = train[((train.obs["cell_label"] == cell_type) & (train.obs["condition"] == "control"))]
        true_labels = np.zeros((len(unperturbed_data), 1))
        fake_labels = np.ones((len(unperturbed_data), 1))

        sc.tl.rank_genes_groups(cell_type_data, groupby="condition", n_genes=100)
        diff_genes = cell_type_data.uns["rank_genes_groups"]["names"]["stimulated"]
        # cell_type_data = cell_type_data.copy()[:, diff_genes.tolist()]

        pred = network.predict(data=unperturbed_data, encoder_labels=true_labels, decoder_labels=fake_labels)
        pred_adata = anndata.AnnData(pred, obs={"condition": ["pred"] * len(pred)},
                                     var={"var_names": cell_type_data.var_names})
        all_adata = cell_type_data.concatenate(pred_adata)

        scgen.plotting.reg_mean_plot(all_adata, condition_key="condition",
                                     axis_keys={"x": "pred", "y": "stimulated", "y1": "stimulated"},
                                     gene_list=diff_genes,
                                     path_to_save=f"./figures/reg_mean_{z_dim}.pdf")
        scgen.plotting.reg_var_plot(all_adata, condition_key="condition",
                                    axis_keys={"x": "pred", "y": "stimulated", 'y1': "stimulated"},
                                    gene_list=diff_genes,
                                    path_to_save=f"./figures/reg_var_{z_dim}.pdf")

        sc.pp.neighbors(all_adata)
        sc.tl.umap(all_adata)
        sc.pl.umap(all_adata, color="condition", save="pred")

        sc.pl.violin(all_adata, keys=diff_genes.tolist()[0], groupby="condition",
                     save=f"_{z_dim}_{diff_genes.tolist()[0]}")

        os.chdir("../../../")


def reconstruct_whole_data():
    train = sc.read("../data/ch10_train_7000.h5ad", backup_url="https://goo.gl/33HtVh")
    all_data = anndata.AnnData()
    for cell_type in train.obs["cell_label"].unique().tolist():
        os.chdir(f"./results/new/{cell_type}")
        net_train_data = train[~((train.obs["cell_label"] == cell_type) & (train.obs["condition"] == "stimulated"))]
        network = scgen.MMDCVAE(x_dimension=net_train_data.X.shape[1], z_dimension=50, alpha=0.001, beta=100,
                                batch_mmd=True, kernel="multi-scale-rbf", train_with_fake_labels=False,
                                model_path=f"./")
        network.restore_model()

        cell_type_data = train[train.obs["cell_label"] == cell_type]
        cell_type_ctrl_data = train[(train.obs["cell_label"] == cell_type_data) & (train.obs["condition"] == "control")]
        unperturbed_data = train[((train.obs["cell_label"] == cell_type) & (train.obs["condition"] == "control"))]
        true_labels = np.zeros((len(unperturbed_data), 1))
        fake_labels = np.ones((len(unperturbed_data), 1))
        pred = network.predict(data=unperturbed_data, encoder_labels=true_labels, decoder_labels=fake_labels)
        ctrl_reconstructed = network.predict(data=cell_type_ctrl_data,
                                                       encoder_labels=np.zeros(shape=(len(cell_type_ctrl_data), 1)),
                                                       decoder_labels=np.zeros(shape=(len(cell_type_ctrl_data), 1)))
        pred_adata = anndata.AnnData(pred, obs={"condition": [f"{cell_type}_pred_stim"] * len(pred)},
                                     var={"var_names": cell_type_data.var_names})
        ctrl_adata = anndata.AnnData(ctrl_reconstructed, obs={"condition": [f"{cell_type}_pred_stim"] * len(ctrl_reconstructed)},
                                     var={"var_names": cell_type_data.var_names})
        all_data = all_data.concatenate(cell_type_data)
        all_data = all_data.concatenate(pred_adata)

        os.chdir("../../../")


if __name__ == '__main__':
    test_train_whole_data_one_celltype_out(z_dim=100,
                                           alpha=0.01,
                                           beta=1,
                                           kernel="multi-scale-rbf",
                                           n_epochs=1500,
                                           batch_size=768)
    # reconstruct_whole_data()
