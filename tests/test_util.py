import scgen
import scanpy as sc


def test_batch_removal():
    train = sc.read("./data/pancreas.h5ad", backup_url="https://goo.gl/V29FNk")
    train.obs["cell_type"] = train.obs["celltype"].tolist()
    network = scgen.VAEArith(x_dimension=train.shape[1], model_path="./models/batch")
    network.train(train_data=train, n_epochs=1, verbose=1)
    corrected_adata = scgen.batch_removal(network, train)
    print(corrected_adata.obs)
    network.sess.close()


test_batch_removal()