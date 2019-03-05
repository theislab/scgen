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
    train = sc.read("../data/train.h5ad", backup_url="https://goo.gl/33HtVh")
    for cell_type in train.obs["cell_type"].unique().tolist():
        os.makedirs(f"./results/{cell_type}/", exist_ok=True)
        os.chdir(f"./results/{cell_type}")
        net_train_data = train[~((train.obs["cell_type"] == cell_type) & (train.obs["condition"] == "stimulated"))]

        network = scgen.MMDCVAE(x_dimension=net_train_data.X.shape[1], z_dimension=z_dim, alpha=alpha, beta=beta,
                                batch_mmd=True, kernel=kernel, train_with_fake_labels=False)

        # network.restore_model()
        network.train(net_train_data, n_epochs=n_epochs, batch_size=batch_size, verbose=2)
        print(f"network_{cell_type} has been trained!")

        true_labels, _ = scgen.label_encoder(net_train_data)
        fake_labels = np.ones(shape=(net_train_data.shape[0], 1))

        latent_with_true_labels = network.to_latent(net_train_data.X, labels=true_labels)
        latent_with_true_labels = sc.AnnData(X=latent_with_true_labels,
                                             obs={"condition": net_train_data.obs["condition"].tolist(),
                                                  "cell_type": net_train_data.obs["cell_type"].tolist()})
        sc.pp.neighbors(latent_with_true_labels)
        sc.tl.umap(latent_with_true_labels)
        sc.pl.umap(latent_with_true_labels, color=["condition", "cell_type"],
                   save=f"_latent_true_labels_{z_dim}",
                   show=False)

        latent_with_fake_labels = network.to_latent(net_train_data.X, fake_labels)
        latent_with_fake_labels = sc.AnnData(X=latent_with_fake_labels,
                                             obs={"condition": net_train_data.obs["condition"].tolist(),
                                                  "cell_type": net_train_data.obs["cell_type"].tolist()})
        sc.pp.neighbors(latent_with_fake_labels)
        sc.tl.umap(latent_with_fake_labels)
        sc.pl.umap(latent_with_fake_labels, color=["condition", "cell_type"],
                   save=f"_latent_fake_labels_{z_dim}",
                   show=False)

        mmd_with_true_labels = network.to_mmd_layer(network.cvae_model, net_train_data.X,
                                                    encoder_labels=true_labels, feed_fake=False)
        mmd_with_true_labels = sc.AnnData(X=mmd_with_true_labels,
                                          obs={"condition": net_train_data.obs["condition"].tolist(),
                                               "cell_type": net_train_data.obs["cell_type"].tolist()})
        sc.pp.neighbors(mmd_with_true_labels)
        sc.tl.umap(mmd_with_true_labels)
        sc.pl.umap(mmd_with_true_labels, color=["condition", "cell_type"],
                   save=f"_mmd_true_labels_{z_dim}",
                   show=False)

        mmd_with_fake_labels = network.to_mmd_layer(network.cvae_model, net_train_data.X,
                                                    encoder_labels=true_labels, feed_fake=True)
        mmd_with_fake_labels = sc.AnnData(X=mmd_with_fake_labels,
                                          obs={"condition": net_train_data.obs["condition"].tolist(),
                                               "cell_type": net_train_data.obs["cell_type"].tolist()})
        sc.pp.neighbors(mmd_with_fake_labels)
        sc.tl.umap(mmd_with_fake_labels)
        sc.pl.umap(mmd_with_fake_labels, color=["condition", "cell_type"],
                   save=f"_mmd_fake_labels_{z_dim}",
                   show=False)

        # decoded_latent_with_true_labels = network.predict(data=latent_with_true_labels, labels=true_labels,
        #                                                   data_space='latent')

        cell_type_data = train[train.obs["cell_type"] == cell_type]
        unperturbed_data = train[((train.obs["cell_type"] == cell_type) & (train.obs["condition"] == "control"))]
        fake_labels = np.ones((len(unperturbed_data), 1))

        pred = network.predict(data=unperturbed_data, labels=fake_labels)
        pred_adata = anndata.AnnData(pred, obs={"condition": ["pred"] * len(pred)},
                                     var={"var_names": cell_type_data.var_names})
        all_adata = cell_type_data.concatenate(pred_adata)
        top_100_genes = sc.tl.rank_genes_groups(cell_type_data, groupby="condition", n_genes=100)
        scgen.plotting.reg_mean_plot(all_adata, condition_key="condition",
                                     axis_keys={"x": "pred", "y": "stimulated", "y1": "stimulated"},
                                     gene_list=top_100_genes,
                                     path_to_save=f"./figures/reg_mean_{z_dim}.pdf")
        scgen.plotting.reg_var_plot(all_adata, condition_key="condition",
                                    axis_keys={"x": "pred", "y": "stimulated", 'y1': "stimulated"},
                                    gene_list=top_100_genes,
                                    path_to_save=f"./figures/reg_var_{z_dim}.pdf")

        sc.pp.neighbors(all_adata)
        sc.tl.umap(all_adata)
        sc.pl.umap(all_adata, color="condition", save="pred")

        sc.pl.violin(all_adata, keys="ISG15", groupby="condition", save=f"_{z_dim}")


if __name__ == '__main__':
    test_train_whole_data_one_celltype_out(z_dim=50,
                                           alpha=0.001,
                                           beta=100,
                                           kernel="multi-scale-rbf",
                                           n_epochs=1000,
                                           batch_size=1024)
