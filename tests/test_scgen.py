import scanpy as sc
import scvi
from scgen_refactor.scgen import SCGEN

adata = sc.read("./tests/data/train_kang.h5ad",
                backup_url="https://goo.gl/33HtVh")

# train_adata = adata
train_adata = adata[~((adata.obs['cell_type'] == 'CD4T') & (adata.obs['condition'] == 'stimulated'))]
train_adata = scvi.data.setup_anndata(train_adata, copy=True)
# train_adata.obs["cell_type"] = train_adata.obs["celltype"].tolist()

model = SCGEN(train_adata)
model.save("../saved_models/model2.pt", overwrite=True)

model.train(
    max_epochs=1,
    batch_size=32,
    early_stopping=True,
    early_stopping_patience=25
)

#batch Removal 
# corrected_adata = model.batch_removal(train_adata, batch_key="batch", cell_label_key="cell_type")

# sc.pp.neighbors(corrected_adata)
# sc.tl.umap(corrected_adata)
# sc.pl.umap(corrected_adata, color=['batch', 'cell_type'], wspace=0.4, frameon=False,
#            save='batch_corrected_b32_klw000005_z100__200e.pdf'
# )

# # test mapping to Latent Space
# latent_X = model.get_latent_representation()
# latent_adata = sc.AnnData(X=latent_X, obs=train_adata.obs.copy())

# sc.pp.neighbors(latent_adata)
# sc.tl.umap(latent_adata)
# sc.pl.umap(latent_adata, color=['condition', 'cell_type'], wspace=0.4, frameon=False,
#            save='latentspace_b32_klw000005_z100__200e.pdf'
# )

# test prediction
pred_adata, _ = model.predict('cell_type', 'condition', conditions={'ctrl': 'control', 'stim': 'stimulated'}, celltype_to_predict='CD4T')
pred_adata.obs['condition'] = 'pred_stimulated'

ctrl_adata = adata[((adata.obs['cell_type'] == 'CD4T') & (adata.obs['condition'] == 'control'))]
stim_adata = adata[((adata.obs['cell_type'] == 'CD4T') & (adata.obs['condition'] == 'stimulated'))]

eval_adata = ctrl_adata.concatenate(stim_adata, pred_adata)

# sc.pp.neighbors(eval_adata)
# sc.tl.umap(eval_adata)
# sc.pl.umap(eval_adata, color=['condition'], wspace=0.4, frameon=False,
#            save='pred_stim_b32_klw000005_z100__200e.pdf'
# )
# CD4T = train_adata[train_adata.obs["cell_type"] =="CD4T"]
# all_adata = CD4T.concatenate(pred_adata)
# print(">> >>> >>> >> \n", train_adata.obs["cell_type"] =="CD4T")
# print(">> >>> >>> >> \n", train_adata.var.copy())
# print(train_adata[train_adata.obs["cell_type"] =="CD4T"].obs)
# CD4T = AnnData(CD4T, obs=train_adata[train_adata.obs["cell_type"] =="CD4T"].obs, var=train_adata.var.copy())
# sc.tl.rank_genes_groups(CD4T.X.A, groupby="condition", method="wilcoxon")
# diff_genes = CD4T.uns["rank_genes_groups"]["names"]["stimulated"]
# print(">>>> \n",diff_genes)
# r2_value = model.reg_mean_plot(all_adata, condition_key="condition",
#                                         axis_keys={
#                                             "x": "pred", "y": "stimulated"},
#                                         gene_list=diff_genes[:10],
#                                         labels={"x": "predicted",
#                                                 "y": "ground truth"},
#                                         path_to_save="./reg_mean1.pdf",
#                                         show=True,
#                                         legend=False)
