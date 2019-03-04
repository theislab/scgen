import os

import anndata
import numpy as np
import scanpy as sc

import scgen

if not os.getcwd().endswith("tests"):
    os.chdir("./tests")
# def get_layer_output_grad(model, inputs, outputs, layer=-1):
#     from keras import backend as K
#     grads = model.optimizer.get_gradients(model.loss_fnctions[1], model.layers[layer].output)
#     symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
#     f = K.function(symb_inputs, grads)
#     x, y, sample_weight = model._standardize_user_data(inputs, outputs)
#     output_grad = f(x + y + sample_weight)
#     return output_grad


train = sc.read("../data/train.h5ad", backup_url="https://goo.gl/33HtVh")

CD4T = train[train.obs["cell_type"] == "CD4T"]
CD4T_labels, _ = scgen.label_encoder(CD4T)
z_dim = 50
net_train_data = train[~((train.obs["cell_type"] == "CD4T") & (train.obs["condition"] == "stimulated"))]
network = scgen.MMDCVAE(x_dimension=CD4T.X.shape[1], z_dimension=z_dim, alpha=0.001, beta=100, batch_mmd=True,
                        kernel="multi-scale-rbf", train_with_fake_labels=False)
# network.restore_model()
network.train(net_train_data, n_epochs=500, batch_size=1024, verbose=2)
print("network has been restored/trained!")
# model = network.cvae_model
# print(get_layer_output_grad(model, CD4T.X, CD4T_labels, layer=0))

true_labels, _ = scgen.label_encoder(net_train_data)
fake_labels = np.zeros(shape=true_labels.shape)

latent_with_true_labels = network.to_latent(net_train_data.X, labels=true_labels)
latent_with_true_labels = sc.AnnData(X=latent_with_true_labels,
                                     obs={"condition": CD4T.obs["condition"].tolist(),
                                          "cell_type": CD4T.obs["cell_type"].tolist()})
sc.pp.neighbors(latent_with_true_labels)
sc.tl.umap(latent_with_true_labels)
sc.pl.umap(latent_with_true_labels, color=["condition", "cell_type"], save=f"latent_true_labels_{z_dim}")

latent_with_fake_labels = network.to_latent(net_train_data.X, np.ones(shape=(CD4T.shape[0], 1)))
latent_with_fake_labels = sc.AnnData(X=latent_with_fake_labels,
                                     obs={"condition": net_train_data.obs["condition"].tolist(),
                                          "cell_type": net_train_data.obs["cell_type"].tolist()})
sc.pp.neighbors(latent_with_fake_labels)
sc.tl.umap(latent_with_fake_labels)
sc.pl.umap(latent_with_fake_labels, color=["condition", "cell_type"], save=f"latent_fake_labels_{z_dim}")

mmd_with_true_labels = network.to_mmd_layer(network.cvae_model, net_train_data.X,
                                            encoder_labels=true_labels,
                                            decoder_labels=true_labels)
mmd_with_true_labels = sc.AnnData(X=mmd_with_true_labels,
                                  obs={"condition": net_train_data.obs["condition"].tolist(),
                                       "cell_type": net_train_data.obs["cell_type"].tolist()})
sc.pp.neighbors(mmd_with_true_labels)
sc.tl.umap(mmd_with_true_labels)
sc.pl.umap(mmd_with_true_labels, color=["condition", "cell_type"], save=f"mmd_true_labels_{z_dim}")

mmd_with_fake_labels = network.to_mmd_layer(network.cvae_model, net_train_data.X,
                                            encoder_labels=true_labels,
                                            decoder_labels=fake_labels)
mmd_with_fake_labels = sc.AnnData(X=mmd_with_fake_labels,
                                  obs={"condition": net_train_data.obs["condition"].tolist(),
                                       "cell_type": net_train_data.obs["cell_type"].tolist()})
sc.pp.neighbors(mmd_with_fake_labels)
sc.tl.umap(mmd_with_fake_labels)
sc.pl.umap(mmd_with_fake_labels, color=["condition", "cell_type"], save=f"mmd_fake_labels_{z_dim}")

decoded_latent_with_true_labels = network.predict(data=latent_with_true_labels, labels=true_labels, data_space='latent')

unperturbed_data = train[((train.obs["cell_type"] == "CD4T") & (train.obs["condition"] == "control"))]
fake_labels = np.ones((len(unperturbed_data), 1))

pred = network.predict(data=unperturbed_data, labels=fake_labels)
pred_adata = anndata.AnnData(pred, obs={"condition": ["pred"] * len(pred)}, var={"var_names": CD4T.var_names})
all_adata = CD4T.concatenate(pred_adata)
scgen.plotting.reg_mean_plot(all_adata, condition_key="condition",
                             axis_keys={"x": "pred", "y": "stimulated", "y1": "stimulated"},
                             gene_list=["ISG15", "CD3D"],
                             path_to_save=f"./figures/reg_mean_{z_dim}.pdf")
scgen.plotting.reg_var_plot(all_adata, condition_key="condition",
                            axis_keys={"x": "pred", "y": "stimulated", 'y1': "stimulated"},
                            gene_list=["ISG15", "CD3D"],
                            path_to_save=f"./figures/reg_var_{z_dim}.pdf")

sc.pp.neighbors(all_adata)
sc.tl.umap(all_adata)
sc.pl.umap(all_adata, color="condition", save="pred")

sc.pl.violin(all_adata, keys="ISG15", groupby="condition", save=f"violin_{z_dim}")
