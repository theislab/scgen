import scanpy as sc
import scvi
from scgen import SCGEN


def test_scgen():

    adata = scvi.data.synthetic_iid()
    model = SCGEN(adata)
    model.train(
        max_epochs=1, batch_size=32, early_stopping=True, early_stopping_patience=25
    )

    # batch Removal
    corrected_adata = model.batch_removal()

    # # test prediction
    # pred_adata, _ = model.predict(
    #     "cell_type",
    #     "condition",
    #     conditions={"ctrl": "control", "stim": "stimulated"},
    #     celltype_to_predict="CD4T",
    # )
    # pred_adata.obs["condition"] = "pred_stimulated"

    # ctrl_adata = adata[
    #     ((adata.obs["cell_type"] == "CD4T") & (adata.obs["condition"] == "control"))
    # ]
    # stim_adata = adata[
    #     ((adata.obs["cell_type"] == "CD4T") & (adata.obs["condition"] == "stimulated"))
    # ]

    # eval_adata = ctrl_adata.concatenate(stim_adata, pred_adata)
