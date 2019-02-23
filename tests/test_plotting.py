import scgen
import scanpy as sc
import anndata


def test_reg_mean_plot():
    train = sc.read("./tests/data/train.h5ad", backup_url="https://goo.gl/33HtVh")
    network = scgen.VAEArith(x_dimension=train.shape[1], model_path="../models/test")
    network.train(train_data=train, n_epochs=0)
    unperturbed_data = train[((train.obs["cell_type"] == "CD4T") & (train.obs["condition"] == "control"))]
    condition = {"ctrl": "control", "stim": "stimulated"}
    pred, delta = network.predict(adata=train, adata_to_predict=unperturbed_data, conditions=condition)
    pred_adata = anndata.AnnData(pred, obs={"condition": ["pred"] * len(pred)}, var={"var_names": train.var_names})
    CD4T = train[train.obs["cell_type"] == "CD4T"]
    all_adata = CD4T.concatenate(pred_adata)
    scgen.plotting.reg_mean_plot(all_adata, condition_key="condition", axis_keys={"x": "control", "y": "pred"},
                                 path_to_save="tests/reg_mean1.pdf")
    scgen.plotting.reg_mean_plot(all_adata, condition_key="condition", axis_keys={"x": "control", "y": "pred"},
                                 path_to_save="tests/reg_mean2.pdf",  gene_list=["ISG15", "CD3D"])
    scgen.plotting.reg_mean_plot(all_adata,condition_key="condition", axis_keys={"x": "control", "y": "pred", "y1": "stimulated"},
                                 path_to_save="tests/reg_mean3.pdf")
    scgen.plotting.reg_mean_plot(all_adata, condition_key="condition", axis_keys={"x": "control", "y": "pred", "y1": "stimulated"},
                                 gene_list=["ISG15", "CD3D"], path_to_save="tests/reg_mean.pdf",)
    network.sess.close()


def test_reg_var_plot():
    train = sc.read("./tests/data/train.h5ad", backup_url="https://goo.gl/33HtVh")
    network = scgen.VAEArith(x_dimension=train.shape[1], model_path="../models/test")
    network.train(train_data=train, n_epochs=0)
    unperturbed_data = train[((train.obs["cell_type"] == "CD4T") & (train.obs["condition"] == "control"))]
    condition = {"ctrl": "control", "stim": "stimulated"}
    pred, delta = network.predict(adata=train, adata_to_predict=unperturbed_data, conditions=condition)
    pred_adata = anndata.AnnData(pred, obs={"condition": ["pred"] * len(pred)}, var={"var_names": train.var_names})
    CD4T = train[train.obs["cell_type"] == "CD4T"]
    all_adata = CD4T.concatenate(pred_adata)
    scgen.plotting.reg_var_plot(all_adata, condition_key="condition", axis_keys={"x": "control", "y": "pred"},
                                path_to_save="tests/reg_var1.pdf")
    scgen.plotting.reg_var_plot(all_adata, condition_key="condition", axis_keys={"x": "control", "y": "pred"},
                                path_to_save="tests/reg_var2.pdf",  gene_list=["ISG15", "CD3D"])
    scgen.plotting.reg_var_plot(all_adata, condition_key="condition", axis_keys={"x": "control", "y": "pred", "y1": "stimulated"},
                                path_to_save="tests/reg_var3.pdf")
    scgen.plotting.reg_var_plot(all_adata, condition_key="condition", axis_keys={"x": "control", "y": "pred", "y1": "stimulated"},
                                gene_list=["ISG15", "CD3D"], path_to_save="tests/reg_var4.pdf")
    network.sess.close()



def test_binary_classifier():
    train = sc.read("./tests/data/train.h5ad", backup_url="https://goo.gl/33HtVh")
    network = scgen.VAEArith(x_dimension=train.shape[1], model_path="../models/test")
    network.train(train_data=train, n_epochs=0)
    unperturbed_data = train[((train.obs["cell_type"] == "CD4T") & (train.obs["condition"] == "control"))]
    condition = {"ctrl": "control", "stim": "stimulated"}
    pred, delta = network.predict(adata=train, adata_to_predict=unperturbed_data, conditions=condition)
    scgen.plotting.binary_classifier(network, train, delta, condition_key="condition",
                                     conditions={"ctrl": "control", "stim": "stimulated"},
                                     path_to_save="tests/binary_classifier.pdf")
    network.sess.close()


