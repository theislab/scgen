import os

import anndata
import numpy as np
import scanpy as sc

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
        network.train(net_train_data, train, n_epochs=n_epochs, batch_size=batch_size, verbose=2,
                      ctrl_key=ctrl_key, stim_key=stim_key,
                      condition_key=condition_key, cell_type_key=cell_type_key,
                      cell_type=cell_type)
        print(f"network_{cell_type} has been trained!")

        scgen.visualize_trained_network_results(network, train, cell_type,
                                                ctrl_key, stim_key,
                                                condition_key, cell_type_key)
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
        os.chdir(f"./results/{data_name}/{cell_type}")
        net_train_data = train[~((train.obs[cell_type_key] == cell_type) & (train.obs[condition_key] == stim_key))]
        network = scgen.MMDCVAE(x_dimension=net_train_data.X.shape[1], z_dimension=50, alpha=0.001, beta=100,
                                batch_mmd=True, kernel="multi-scale-rbf", train_with_fake_labels=False,
                                model_path=f"./")
        network.restore_model()

        cell_type_data = train[train.obs[cell_type_key] == cell_type]
        cell_type_ctrl_data = train[((train.obs[cell_type_key] == cell_type) & (train.obs[condition_key] == ctrl_key))]
        unperturbed_data = train[((train.obs[cell_type_key] == cell_type) & (train.obs[condition_key] == ctrl_key))]
        true_labels = np.zeros((len(unperturbed_data), 1))
        fake_labels = np.ones((len(unperturbed_data), 1))
        pred = network.predict(data=unperturbed_data, encoder_labels=true_labels, decoder_labels=fake_labels)
        ctrl_reconstructed = network.predict(data=cell_type_ctrl_data,
                                             encoder_labels=np.zeros(shape=(len(cell_type_ctrl_data), 1)),
                                             decoder_labels=np.zeros(shape=(len(cell_type_ctrl_data), 1)))
        pred_adata = anndata.AnnData(pred, obs={condition_key: [f"{cell_type}_pred_stim"] * len(pred)},
                                     var={"var_names": cell_type_data.var_names})
        ctrl_adata = anndata.AnnData(ctrl_reconstructed,
                                     obs={condition_key: [f"{cell_type}_ctrl"] * len(ctrl_reconstructed)},
                                     var={"var_names": cell_type_data.var_names})
        if data_name == "pbmc":
            real_stim = cell_type_data[cell_type_data.obs[condition_key] == stim_key].X.A
        else:
            real_stim = cell_type_data[cell_type_data.obs[condition_key] == stim_key].X
        real_stim_adata = anndata.AnnData(real_stim,
                                          obs={condition_key: [f"{cell_type}_real_stim"] * len(real_stim)},
                                          var={"var_names": cell_type_data.var_names})
        if idx == 0:
            all_data = ctrl_adata.concatenate(pred_adata, real_stim_adata)
        else:
            all_data = all_data.concatenate(ctrl_adata, pred_adata, real_stim_adata)

        os.chdir("../../../")
        print(f"Finish Reconstructing for {cell_type}")
    all_data.write_h5ad(f"./results/{data_name}/reconstructed.h5ad")


if __name__ == '__main__':
    test_train_whole_data_one_celltype_out(data_name="pbmc",
                                           z_dim=100,
                                           alpha=0.00005,
                                           n_epochs=300,
                                           batch_size=32,
                                           dropout_rate=0.2,
                                           learning_rate=0.001,
                                           condition_key="condition")
    # test_train_whole_data_one_celltype_out(data_name="hpoly",
    #                                        z_dim=100,
    #                                        alpha=0.001,
    #                                        n_epochs=250,
    #                                        batch_size=64,
    #                                        condition_key="condition")
    # test_train_whole_data_one_celltype_out(data_name="salmonella",
    #                                        z_dim=100,
    #                                        alpha=0.001,
    #                                        n_epochs=250,
    #                                        batch_size=64,
    #                                        condition_key="condition")
    # reconstruct_whole_data(data_name="species")
    # reconstruct_whole_data(data_name="salmonella")
