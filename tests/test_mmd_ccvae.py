import os

import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scgen

if not os.getcwd().endswith("tests"):
    os.chdir("./tests")


def test_train_whole_data_one_celltype_out(data_name="pbmc",
                                           z_dim=50,
                                           alpha=0.001,
                                           beta=100,
                                           kernel="multi-scale-rbf",
                                           n_epochs=1000,
                                           batch_size=1024,
                                           condition_key="condition",
                                           arch_style=1):
    if data_name == "normal_thin":
        stim_key = "thin"
        ctrl_key = "normal"
        cell_type_key = "labels"
        train = sc.read(f"../data/{data_name}.h5ad")
        train = train[((train.obs["labels"] == 1) |
                       (train.obs["labels"] == 2) |
                       (train.obs["labels"] == 7))]
        train.X /= 255.0
    elif data_name == "normal_thick":
        stim_key = "thick"
        ctrl_key = "normal"
        cell_type_key = "labels"
        train = sc.read(f"../data/{data_name}.h5ad")
        train = train[((train.obs["labels"] == 8) |
                       (train.obs["labels"] == 3) |
                       (train.obs["labels"] == 5) |
                       (train.obs["labels"] == 2))]
        train.X /= 255.0
    elif data_name == "h2z":
        stim_key = "zebra"
        ctrl_key = "horse"
        cell_type_key = None
        train = sc.read(f"../data/{data_name}.h5ad")
        train.X /= 255.0
    elif data_name == "mnist":
        stim_key = '7'
        ctrl_key = '1'
        cell_type_key = None
        train = sc.read(f"../data/{data_name}.h5ad")
        train.obs["condition"] = train.obs["condition"].astype(np.str)
        train = train[((train.obs["labels"] == 1) |
                       (train.obs["labels"] == 7))]
        train.X /= 255.0
    elif data_name == "fashion":
        stim_key = 5
        ctrl_key = 9
        cell_type_key = None
        train = sc.read(f"../data/{data_name}.h5ad")
        train = train[((train.obs["labels"] == 5) |
                       (train.obs["labels"] == 9))]
        train.X /= 255.0
    if cell_type_key is None:
        os.makedirs(f"./results/{data_name}/", exist_ok=True)
        os.chdir(f"./results/{data_name}/")
        network = scgen.MMDCCVAE(x_dimension=(784,), z_dimension=z_dim, alpha=alpha, beta=beta,
                                 batch_mmd=True, kernel=kernel, train_with_fake_labels=False,
                                 model_path=f"./", arch_style=arch_style)
        net_train_data = train
        network.train(net_train_data, n_epochs=n_epochs, batch_size=batch_size, verbose=2)
        print(f"network has been trained!")
        true_labels, _ = scgen.label_encoder(net_train_data)
        fake_labels = np.ones(shape=(net_train_data.shape[0], 1))
    else:
        for cell_type in train.obs[cell_type_key].unique().tolist():
            if cell_type != 3:
                continue
            os.makedirs(f"./results/{data_name}/{cell_type}/", exist_ok=True)
            os.chdir(f"./results/{data_name}/{cell_type}")
            # net_train_data = train[~((train.obs[cell_type_key] == cell_type) & (train.obs[condition_key] == stim_key))]
            net_train_data = train
            network = scgen.MMDCCVAE(x_dimension=(256, 256, 3,), z_dimension=z_dim, alpha=alpha, beta=beta,
                                     batch_mmd=True, kernel=kernel, train_with_fake_labels=False,
                                     model_path=f"./", arch_style=arch_style)

            # network.restore_model()
            network.train(net_train_data, n_epochs=n_epochs, batch_size=batch_size, verbose=1)
            print(f"network_{cell_type} has been trained!")

            true_labels, _ = scgen.label_encoder(net_train_data)
            fake_labels = np.ones(shape=(net_train_data.shape[0], 1))

            latent_with_true_labels = network.to_latent(net_train_data.X, labels=true_labels)
            latent_with_true_labels = sc.AnnData(X=latent_with_true_labels,
                                                 obs={condition_key: net_train_data.obs[condition_key].tolist(),
                                                      cell_type_key: pd.Categorical(net_train_data.obs[cell_type_key])})
            sc.pp.neighbors(latent_with_true_labels)
            sc.tl.umap(latent_with_true_labels)
            sc.pl.umap(latent_with_true_labels, color=[condition_key, cell_type_key],
                       save=f"_latent_true_labels_{z_dim}",
                       show=False)

            latent_with_fake_labels = network.to_latent(net_train_data.X, fake_labels)
            latent_with_fake_labels = sc.AnnData(X=latent_with_fake_labels,
                                                 obs={condition_key: net_train_data.obs[condition_key].tolist(),
                                                      cell_type_key: pd.Categorical(net_train_data.obs[cell_type_key])})
            sc.pp.neighbors(latent_with_fake_labels)
            sc.tl.umap(latent_with_fake_labels)
            sc.pl.umap(latent_with_fake_labels, color=[condition_key, cell_type_key],
                       save=f"_latent_fake_labels_{z_dim}",
                       show=False)

            mmd_with_true_labels = network.to_mmd_layer(network.cvae_model, net_train_data.X,
                                                        encoder_labels=true_labels, feed_fake=False)
            mmd_with_true_labels = sc.AnnData(X=mmd_with_true_labels,
                                              obs={condition_key: net_train_data.obs[condition_key].tolist(),
                                                   cell_type_key: pd.Categorical(net_train_data.obs[cell_type_key])})
            sc.pp.neighbors(mmd_with_true_labels)
            sc.tl.umap(mmd_with_true_labels)
            sc.pl.umap(mmd_with_true_labels, color=[condition_key, cell_type_key],
                       save=f"_mmd_true_labels_{z_dim}",
                       show=False)

            mmd_with_fake_labels = network.to_mmd_layer(network.cvae_model, net_train_data.X,
                                                        encoder_labels=true_labels, feed_fake=True)
            mmd_with_fake_labels = sc.AnnData(X=mmd_with_fake_labels,
                                              obs={condition_key: net_train_data.obs[condition_key].tolist(),
                                                   cell_type_key: pd.Categorical(net_train_data.obs[cell_type_key])})
            sc.pp.neighbors(mmd_with_fake_labels)
            sc.tl.umap(mmd_with_fake_labels)
            sc.pl.umap(mmd_with_fake_labels, color=[condition_key, cell_type_key],
                       save=f"_mmd_fake_labels_{z_dim}",
                       show=False)

            cell_type_data = train[train.obs[cell_type_key] == cell_type]
            unperturbed_data = train[((train.obs[cell_type_key] == cell_type) & (train.obs[condition_key] == ctrl_key))]
            true_labels = np.zeros((len(unperturbed_data), 1))
            fake_labels = np.ones((len(unperturbed_data), 1))

            pred = network.predict(data=unperturbed_data, encoder_labels=true_labels, decoder_labels=fake_labels)
            pred_adata = anndata.AnnData(pred, obs={condition_key: ["pred"] * len(pred)},
                                         var={"var_names": cell_type_data.var_names})
            all_adata = cell_type_data.copy().concatenate(pred_adata.copy())

            scgen.plotting.reg_mean_plot(all_adata, condition_key=condition_key,
                                         axis_keys={"x": ctrl_key, "y": "pred", "y1": stim_key},
                                         path_to_save=f"./figures/reg_mean_{z_dim}.pdf")
            scgen.plotting.reg_var_plot(all_adata, condition_key=condition_key,
                                        axis_keys={"x": ctrl_key, "y": "pred", 'y1': stim_key},
                                        path_to_save=f"./figures/reg_var_{z_dim}.pdf")

            sc.pp.neighbors(all_adata)
            sc.tl.umap(all_adata)
            sc.pl.umap(all_adata, color=condition_key,
                       save="pred")

            # sc.pl.violin(all_adata, keys=diff_genes.tolist()[0], groupby=condition_key,
            #              save=f"_{z_dim}_{diff_genes.tolist()[0]}")

            os.chdir("../../../")


def feed_normal_sample(data_name="normal_thin", digit=1):
    if data_name == "normal_thin":
        data = sc.read(f"../data/{data_name}.h5ad")
        data = data[((data.obs["labels"] == 1) |
                     (data.obs["labels"] == 2) |
                     (data.obs["labels"] == 7))]
        normal_data = data[data.obs["condition"] == "normal"]
        image_shape = (28, 28, 1)
    elif data_name == "normal_thick":
        data = sc.read(f"../data/{data_name}.h5ad")
        data = data[((data.obs["labels"] == 8) |
                     (data.obs["labels"] == 3) |
                     (data.obs["labels"] == 5) |
                     (data.obs["labels"] == 2))]
        normal_data = data[data.obs["condition"] == "normal"]
        image_shape = (28, 28, 1)
    elif data_name == "h2z":
        data = sc.read(f"../data/{data_name}.h5ad")
        normal_data = data[data.obs["condition"] == "horse"]
        normal_data.X /= 255.
        image_shape = (256 * 256 * 3, )
    elif data_name == "mnist":
        data = sc.read(f"../data/{data_name}.h5ad")
        data.obs["condition"] = data.obs["condition"].astype(np.str)
        normal_data = data[data.obs["condition"] == str(digit)]
        normal_data.X /= 255.
        image_shape = (784,)
    elif data_name == "fashion":
        data = sc.read(f"../data/{data_name}.h5ad")
        data.obs["condition"] = data.obs["condition"].astype(np.str)
        normal_data = data[data.obs["condition"] == '5']
        normal_data.X /= 255.
        image_shape = (784,)
    print(data.shape)
    if data_name == "mnist":
        os.makedirs(f"./results/{data_name}/{digit} to 7/")
        os.chdir(f"./results/{data_name}/{digit} to 7/")
    else:
        os.chdir(f"./results/{data_name}/")
    network = scgen.MMDCCVAE(x_dimension=image_shape, z_dimension=100, alpha=0.001, beta=100,
                             batch_mmd=True, kernel="multi-scale-rbf", train_with_fake_labels=False,
                             model_path=f"./", arch_style=2)

    network.restore_model()
    print("model has been restored!")

    for j in range(5):
        k = 5
        random_samples = np.random.choice(normal_data.shape[0], k, replace=False)
        sample_normal = normal_data.X[random_samples]
        sample_normal_reshaped = np.reshape(sample_normal, (-1, 28, 28))
        sample_normal = np.reshape(sample_normal, (-1, *image_shape))
        sample_normal = anndata.AnnData(X=sample_normal)
        sample_normal.X = np.reshape(sample_normal.X, (-1, *image_shape))

        sample_thick = network.predict(data=sample_normal,
                                       encoder_labels=np.zeros((len(sample_normal), 1)),
                                       decoder_labels=np.ones((len(sample_normal), 1)))
        sample_thick = np.reshape(sample_thick, newshape=(-1, 28, 28))
        print(sample_thick.shape)
        print(sample_normal_reshaped.shape)
        plt.close("all")
        fig, ax = plt.subplots(k, 2, figsize=(k * 1, 6))
        for i in range(k):
            ax[i, 0].axis('off')
            ax[i, 0].imshow(sample_normal_reshaped[i], cmap='Greys', vmin=0, vmax=1)
            ax[i, 1].axis('off')
            if i == 0:
                if data_name.endswith("h2z"):
                    ax[i, 0].set_title("Horse")
                else:
                    ax[i, 0].set_title("Normal")
                if data_name.endswith("thick"):
                    ax[i, 1].set_title("Thick")
                elif data_name.endswith("thin"):
                    ax[i, 1].set_title("Thin")
                else:
                    ax[i, 1].set_title("Zebra")
            ax[i, 1].imshow(sample_thick[i], cmap='Greys', vmin=0, vmax=1)
        plt.savefig(f"./sample_images_{data_name}_{j}.pdf")
    if data_name == "mnist":
        os.chdir("../../../")


if __name__ == '__main__':
    test_train_whole_data_one_celltype_out(data_name="mnist",
                                           z_dim=100,
                                           alpha=0.01,
                                           beta=100,
                                           kernel="multi-scale-rbf",
                                           n_epochs=1500,
                                           batch_size=1024,
                                           condition_key="condition",
                                           arch_style=1)
    # feed_normal_sample("normal_thin")
    # feed_normal_sample("normal_thick")
    # feed_normal_sample("h2z")
    # for i in range(10):
    #     feed_normal_sample("mnist", digit=i)
    # feed_normal_sample("fashion")
