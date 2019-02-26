import scgen
import scanpy as sc
import numpy as np

train = sc.read("./data/train.h5ad")
# train = train[train.obs["cell_type"] == "CD4T"]
network = scgen.CVAE(x_dimension=train.X.shape[1], z_dimension=50, alpha=0.01)
# network.train(train, n_epochs=20)
network.restore_model()

labels, _ = scgen.label_encoder(train)
latent = network._to_latent(train.X.A, labels=labels)
adata = sc.AnnData(X=latent, obs={"condition": train.obs["condition"].tolist(), "cell_type": train.obs["cell_type"].tolist()})
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.pl.umap(adata, color =["condition", "cell_type"], save="train")

mmd = network.to_mmd_layer(train.X.A, labels=labels)
adata_mmd = sc.AnnData(X=mmd, obs={"condition": train.obs["condition"].tolist(), "cell_type": train.obs["cell_type"].tolist()})
sc.pp.neighbors(adata_mmd)
sc.tl.umap(adata_mmd)
sc.pl.umap(adata_mmd, color =["condition", "cell_type"], save="mmd")

CD4T = train[train.obs["cell_type"] == "CD4T"]
unperturbed_data = train[((train.obs["cell_type"] == "CD4T") & (train.obs["condition"] == "control"))]
fake_labels = np.ones((len(unperturbed_data), 1))
predicted_cells = network.predict(unperturbed_data, fake_labels)
adata = sc.AnnData(predicted_cells, obs={"condition": ["pred"]*len(fake_labels)})
adata.var_names = CD4T.var_names
all_adata = CD4T.concatenate(adata)
scgen.plotting.reg_mean_plot(all_adata, condition_key="condition",
                             axis_keys={"x": "pred", "y": "stimulated"},
                             gene_list= ["ISG15", "CD3D"],
                            path_to_save="figures/reg_mean.pdf")

scgen.plotting.reg_var_plot(all_adata, condition_key="condition",
                             axis_keys={"x": "pred", "y": "stimulated"},
                             gene_list= ["ISG15", "CD3D"],
                            path_to_save="figures/reg_var.pdf")

sc.pp.neighbors(all_adata)
sc.tl.umap(all_adata)
sc.pl.umap(all_adata, color ="condition", save="pred")

sc.pl.violin(all_adata, keys="ISG15", groupby="condition", save="violin")