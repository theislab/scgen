import scanpy as sc


def score(adata, n_deg=10, condition_key="condition",
                 conditions={"stim": "stimulated", "ctrl": "control"},
         sortby="median_score"):

    import scanpy as sc
    import numpy as np
    from scipy.stats import entropy
    import pandas as pd
    sc.tl.rank_genes_groups(adata, groupby=condition_key, method="wilcoxon", n_genes=n_deg)
    adata_deg = adata[:, adata.uns["rank_genes_groups"]["names"][conditions["stim"]].tolist()].copy()
    cell_types = adata_deg.obs["cell_type"].cat.categories.tolist()
    lfc_temp = np.zeros((len(cell_types), n_deg))
    for j , ct in enumerate(cell_types):
        stim = adata_deg[(adata_deg.obs["cell_type"] == ct) &
                         (adata_deg.obs[condition_key] == conditions["stim"])].X.mean(0).A1
        ctrl = adata_deg[(adata_deg.obs["cell_type"] == ct) &
                         (adata_deg.obs[condition_key] == conditions["ctrl"])].X.mean(0).A1
        lfc_temp[j] = np.abs((stim - ctrl)[None, :])
    norm_lfc = lfc_temp/lfc_temp.sum(0).reshape((1, n_deg))
    ent_scores = entropy(norm_lfc)
    median = np.median(lfc_temp, axis=0)
    med_scores = np.max(np.abs((lfc_temp - median)), axis=0)
    df_score = pd.DataFrame({"genes": adata_deg.var_names.tolist(), "median_score": med_scores,
                             "entropy_score": ent_scores })
    if (sortby == "median_score"):
        return df_score.sort_values(by=['median_score'], ascending=False)
    else:
        return df_score.sort_values(by=['entropy_score'])



if __name__ == "main":
    adata_file = 'tests/data/train_kang.h5ad'
    adata = sc.read(adata_file)
    df_score = score(adata)
    df_score = score(adata, sortby="entropy_score")