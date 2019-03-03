import numpy as np
import scanpy as sc

import scgen

# def get_layer_output_grad(model, inputs, outputs, layer=-1):
#     from keras import backend as K
#     grads = model.optimizer.get_gradients(model.loss_fnctions[1], model.layers[layer].output)
#     symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
#     f = K.function(symb_inputs, grads)
#     x, y, sample_weight = model._standardize_user_data(inputs, outputs)
#     output_grad = f(x + y + sample_weight)
#     return output_grad


train = sc.read("../data/train.h5ad")
train = train[train.obs["cell_type"] == "CD4T"]
train_labels = scgen.label_encoder(train)
z_dim = 20
network = scgen.MMDCVAE(x_dimension=train.X.shape[1], z_dimension=z_dim, alpha=0.001, beta=100, batch_mmd=True,
                        kernel="multi-scale-rbf", train_with_fake_labels=False)
# network.restore_model()
network.train(train, n_epochs=1000, verbose=2)
# model = network.cvae_model
# print(get_layer_output_grad(model, train.X, train_labels, layer=0))

true_labels, _ = scgen.label_encoder(train)
fake_labels = np.zeros(shape=true_labels.shape)

latent_with_true_labels = network.to_latent(train.X, labels=true_labels)
adata = sc.AnnData(X=latent_with_true_labels,
                   obs={"condition": train.obs["condition"].tolist(), "cell_type": train.obs["cell_type"].tolist()})
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.pl.umap(adata, color=["condition", "cell_type"], save=f"latent_true_labels_{z_dim}")

latent_with_fake_labels = network.to_latent(train.X, np.ones(shape=(train.shape[0], 1)))
adata = sc.AnnData(X=latent_with_fake_labels,
                   obs={"condition": train.obs["condition"].tolist(), "cell_type": train.obs["cell_type"].tolist()})
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.pl.umap(adata, color=["condition", "cell_type"], save=f"latent_fake_labels_{z_dim}")

mmd_with_true_labels = network.to_mmd_layer(network.cvae_model, train.X,
                                            encoder_labels=true_labels,
                                            decoder_labels=true_labels)
adata_mmd = sc.AnnData(X=mmd_with_true_labels,
                       obs={"condition": train.obs["condition"].tolist(), "cell_type": train.obs["cell_type"].tolist()})
sc.pp.neighbors(adata_mmd)
sc.tl.umap(adata_mmd)
sc.pl.umap(adata_mmd, color=["condition", "cell_type"], save=f"mmd_true_labels_{z_dim}")

mmd_with_fake_labels = network.to_mmd_layer(network.cvae_model, train.X,
                                            encoder_labels=true_labels,
                                            decoder_labels=fake_labels)
adata_mmd = sc.AnnData(X=mmd_with_fake_labels,
                       obs={"condition": train.obs["condition"].tolist(), "cell_type": train.obs["cell_type"].tolist()})
sc.pp.neighbors(adata_mmd)
sc.tl.umap(adata_mmd)
sc.pl.umap(adata_mmd, color=["condition", "cell_type"], save=f"mmd_true_labels_{z_dim}")
