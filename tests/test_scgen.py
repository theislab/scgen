import scanpy as sc
import scvi

from scgen import SCGEN


def test_scgen():
    adata = scvi.data.synthetic_iid()
    SCGEN.setup_anndata(
        adata,
        batch_key="batch",
        labels_key="labels",
    )

    model = SCGEN(adata)
    model.train(
        max_epochs=1, batch_size=32, early_stopping=True, early_stopping_patience=25
    )

    # batch Removal
    model.batch_removal()

    # predict
    pred, delta = model.predict(
        ctrl_key="batch_0", stim_key="batch_1", celltype_to_predict="label_0"
    )
    pred.obs["batch"] = "pred"

    # reg mean and reg var
    ctrl_adata = adata[
        ((adata.obs["labels"] == "label_0") & (adata.obs["batch"] == "batch_0"))
    ]
    stim_adata = adata[
        ((adata.obs["labels"] == "label_0") & (adata.obs["batch"] == "batch_1"))
    ]
    eval_adata = ctrl_adata.concatenate(stim_adata, pred, batch_key="concat_batches")
    label_0 = adata[adata.obs["labels"] == "label_0"]
    sc.tl.rank_genes_groups(label_0, groupby="batch", method="wilcoxon")
    diff_genes = label_0.uns["rank_genes_groups"]["names"]["batch_1"]

    model.reg_mean_plot(
        eval_adata,
        axis_keys={"x": "pred", "y": "batch_1"},
        gene_list=diff_genes[:10],
        labels={"x": "predicted", "y": "ground truth"},
        save=False,
        show=False,
        legend=False,
    )
