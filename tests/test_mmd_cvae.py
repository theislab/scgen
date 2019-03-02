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
# train = train[~((train.obs["cell_type"] == "CD4T") & (train.obs["condition"] == "stimulated"))]
z_dim = 20
network = scgen.MMDCVAE(x_dimension=train.X.shape[1], z_dimension=z_dim, alpha=0.1, batch_mmd=True, kernel="multi-scale-rbf")
# network.restore_model()
network.train(train, n_epochs=100, verbose=1)
# model = network.cvae_model
# print(get_layer_output_grad(model, train.X, train_labels, layer=0))

labels, _ = scgen.label_encoder(train)
latent = network.to_latent(train.X, labels=labels)
adata = sc.AnnData(X=latent,
                   obs={"condition": train.obs["condition"].tolist(), "cell_type": train.obs["cell_type"].tolist()})
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.pl.umap(adata, color=["condition", "cell_type"], save=f"train_{z_dim}")
mmd = network.to_mmd_layer(network.cvae_model, train.X, labels=labels)
adata_mmd = sc.AnnData(X=mmd,
                       obs={"condition": train.obs["condition"].tolist(), "cell_type": train.obs["cell_type"].tolist()})
sc.pp.neighbors(adata_mmd)
sc.tl.umap(adata_mmd)
sc.pl.umap(adata_mmd, color=["condition", "cell_type"], save=f"true_labels_{z_dim}")