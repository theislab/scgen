import scgen
import scanpy as sc


def test_vae_arith_validation():
    train_data = sc.read("./tests/data/train.h5ad", backup_url="https://goo.gl/33HtVh")
    validation_data = sc.read("./tests/data/validation.h5ad", backup_url="https://goo.gl/8pdXiQ")
    network = scgen.VAEArith(x_dimension=train_data.shape[1], model_path="./tests/models")
    network.train(train_data=train_data, use_validation=True, valid_data=validation_data, n_epochs=0)
    unperturbed_data = train_data[
        ((train_data.obs["cell_type"] == "CD4T") & (train_data.obs["condition"] == "control"))]
    condition = {"ctrl": "control", "stim": "stimulated"}
    pred, delta = network.predict(adata=train_data, celltype_to_predict="CD4T",
                                  obs_key={"cell_type": ["CD8T", "NK"]}, conditions=condition)
    pred, delta = network.predict(adata=train_data, celltype_to_predict="CD4T",
                                  obs_key={"cell_type": ["CD8T"]}, conditions=condition)
    pred, delta = network.predict(adata=train_data,
                                  celltype_to_predict="CD4T", conditions=condition)
    pred, delta = network.predict(adata=train_data, adata_to_predict=unperturbed_data, conditions=condition)
    network.sess.close()



