import tensorflow as tf
import numpy as np
import scanpy.api as sc
from random import  shuffle
import wget
import os
import sklearn as sk


train_path = "../data/pancreas.h5ad"

if os.path.isfile(train_path):
    data = sc.read(train_path)
else:
    train_url = "https://www.dropbox.com/s/zvmt8oxhfksumw2/pancreas.h5ad?dl=1"
    t_dl = wget.download(train_url, train_path)
    data = sc.read(train_path)




sc.settings.figdir = "../results"
model_to_use = "../models/pancreas"
batch_size = 32
train_real = data
input_matrix = data.X
ind_list = [i for i in range(input_matrix.shape[0])]
shuffle(ind_list)
train_data = input_matrix[ind_list, :]
gex_size = input_matrix.shape[1]
X_dim = gex_size
z_dim = 100
lr = 0.001
dr_rate = .2
X = tf.placeholder(tf.float32, shape=[None, X_dim],name="data")
z = tf.placeholder(tf.float32, shape=[None, z_dim],name="noise")
data_max_value = np.amax(input_matrix)
time_step = tf.placeholder(tf.int32)
size  = tf.placeholder(tf.int32)
is_training = tf.placeholder(tf.bool)
init_w =  tf.contrib.layers.xavier_initializer()
regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)

def give_me_latent(data):
    latent = sess.run(z_mean,feed_dict = {X : data,size:len(data),is_training:False})
    return  latent
def avg_vector(data):

    latent =  give_me_latent(data)
    arithmatic =  np.average(latent,axis=0)
    return  arithmatic
def reconstruct(data,use_data = False):

    if(use_data):
        latent = data
    else:
        latent = give_me_latent(data)

    reconstruct = sess.run(X_hat,feed_dict = {z_mean : latent ,is_training:False})
    return  reconstruct
# =============================== Q(z|X) ======================================

def Q(X, reuse=False):
    with tf.variable_scope("gq", reuse=reuse):
        h = tf.layers.dense(inputs=X, units=800, kernel_initializer=init_w,use_bias=False,
                            kernel_regularizer=regularizer)
        h = tf.layers.batch_normalization(h,axis= 1,training=is_training)
        h = tf.nn.leaky_relu(h)
        h = tf.layers.dropout(h,dr_rate, training= is_training)
        h = tf.layers.dense(inputs=h, units=800, kernel_initializer=init_w, use_bias=False,
                            kernel_regularizer=regularizer)
        h = tf.layers.batch_normalization(h, axis=1, training=is_training)
        h = tf.nn.leaky_relu(h)
        h = tf.layers.dropout(h,dr_rate, training= is_training)
        mean =  tf.layers.dense(inputs=h, units=z_dim, kernel_initializer=init_w)
        variance =  tf.layers.dense(inputs=h, units=z_dim, kernel_initializer=init_w)
        return mean, variance

# =============================== P(Z) ======================================
def sample_z(mu, log_var,size):
    eps = tf.random_normal(shape=[size,z_dim])
    return mu + tf.exp(log_var / 2) * eps

def sample(n_sample):
    noise = np.random.normal(0.0, 1, size=(n_sample, z_dim))
    gen_cells = sess.run(X_hat, feed_dict={z_mean: noise,is_training:False})
    return  gen_cells
# =============================== P(X|z) ======================================
def P(z,reuse=False):
    with tf.variable_scope("gp", reuse=reuse):
        h = tf.layers.dense(inputs=z,units= 800,kernel_initializer=init_w,use_bias=False,
                            kernel_regularizer=regularizer)
        h = tf.layers.batch_normalization(h,axis= 1,training=is_training)
        h = tf.nn.leaky_relu(h)
        h = tf.layers.dropout(h,dr_rate, training= is_training)

        h = tf.layers.dense(inputs=h, units=800, kernel_initializer=init_w,use_bias=False,
                            kernel_regularizer=regularizer)
        tf.layers.batch_normalization(h,axis= 1,training=is_training)
        h = tf.nn.leaky_relu(h)
        h = tf.layers.dropout(h,dr_rate, training= is_training)
        h = tf.layers.dense(inputs=h, units=X_dim, kernel_initializer=init_w, use_bias=True)
        h = tf.nn.relu(h)
        return h
mean, variance = Q(X)
z_mean = sample_z(mean,variance,size)
X_hat = P(z_mean)
# =============================== loss ====================================
kl_loss = 0.5 * tf.reduce_sum(tf.exp(variance) + variance**2 - 1. - variance, 1)
recon_loss = 0.5*tf.reduce_sum(tf.square((X-X_hat)), 1)
vae_loss = tf.reduce_mean(recon_loss + kl_loss)
recon_loss_summary = tf.summary.scalar('REC', tf.reduce_mean(recon_loss))
vae_loss_summary = tf.summary.scalar('VAE', vae_loss)
t_vars = tf.trainable_variables()
g_lrate = tf.placeholder(tf.float32, shape=[])
global_step = tf.Variable(0, name='global_step', trainable=False, dtype=tf.int32)
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    Solver = tf.train.AdamOptimizer(learning_rate=lr).minimize(vae_loss)
sess=tf.InteractiveSession()
saver = tf.train.Saver(max_to_keep=1)
init = tf.global_variables_initializer().run()

def train(n_epochs, full_training=True, initial_run=True):
    if initial_run:
        print("Initial run")
        assign_step_zero = tf.assign(global_step, 0)
        init_step = sess.run(assign_step_zero)
    if not initial_run:
        saver.restore(sess, model_to_use)
    for it in range(n_epochs):
        increment_global_step_op = tf.assign(global_step, global_step + 1)
        step = sess.run(increment_global_step_op)
        current_step = sess.run(global_step)
        train_loss = 0
        if (full_training):
            input_matrix = train_data[0:train_data.shape[0] // batch_size * batch_size, :]
            for lower in range(0, input_matrix.shape[0], batch_size):
                upper = min(lower + batch_size, input_matrix.shape[0])

                X_mb = input_matrix[lower:upper, :]
                _, D_loss_curr = sess.run(
                    [Solver, vae_loss], feed_dict={X: X_mb, time_step: current_step,
                                                           size: batch_size, is_training: True})
                train_loss += D_loss_curr
    save_path = saver.save(sess, model_to_use)
    print("Model saved in file: %s" % save_path)
    print(f"total number of trained epochs is {current_step}")

def vector_batch_removal():
    #projecting data to latent space
    latent_all = give_me_latent(data.X)
    latent_ann = sc.AnnData(latent_all)
    latent_ann.obs["cell_type"] = data.obs["cell_type"].tolist()
    latent_ann.obs["batch"] = data.obs["batch"].tolist()
    latent_ann.obs["sample"] = data.obs["sample"].tolist()
    unique_cell_types = np.unique(latent_ann.obs["cell_type"])
    shared_anns = []
    not_shared_ann = []
    for cell_type in unique_cell_types :
        temp_cell = latent_ann[latent_ann.obs["cell_type"] ==  cell_type]
        if (len(np.unique(temp_cell.obs["batch"])) <2) :
            cell_type_ann = latent_ann[latent_ann.obs["cell_type"] == cell_type]
            not_shared_ann.append(cell_type_ann)
            continue
        print(cell_type)
        temp_cell = latent_ann[latent_ann.obs["cell_type"] ==  cell_type]
        batch_list = {}
        max_batch = 0
        max_batch_ind = ""
        batchs = np.unique(temp_cell.obs["batch"])
        for i in batchs:
            temp = temp_cell[temp_cell.obs["batch"] == i]
            if max_batch <len(temp):
                max_batch = len(temp)
                max_batch_ind = i
            batch_list[i] = temp
        max_batch_ann = batch_list[max_batch_ind]
        for study in batch_list :
            delta = np.average(max_batch_ann.X,axis=0) - np.average(batch_list[study].X, axis=0)
            batch_list[study].X = delta +   batch_list[study].X
        corrected = sc.AnnData.concatenate(*list(batch_list.values()))
        shared_anns.append(corrected)
    all_shared_ann = sc.AnnData.concatenate(*shared_anns)
    all_not_shared_ann = sc.AnnData.concatenate(*not_shared_ann)
    all_corrected_data = sc.AnnData.concatenate(all_shared_ann, all_not_shared_ann)
    #reconstructing data to gene epxression space
    corrected = sc.AnnData(reconstruct(all_corrected_data.X, use_data = True))
    corrected.obs["cell_type"] = all_shared_ann.obs["cell_type"].tolist() + all_not_shared_ann.obs["cell_type"].tolist()
    corrected.obs["study"] = all_shared_ann.obs["sample"].tolist() + all_not_shared_ann.obs["sample"].tolist()
    corrected.var_names = data.var_names.tolist()
    #shared cell_types
    corrected_shared = sc.AnnData(reconstruct(all_shared_ann.X, use_data = True))
    corrected_shared.obs["cell_type"] = all_shared_ann.obs["cell_type"].tolist()
    corrected_shared.obs["study"] = all_shared_ann.obs["sample"].tolist()
    corrected_shared.var_names = data.var_names.tolist()
    return corrected, corrected_shared

if __name__ == "__main__":
    data.obs["cell_type"] = data.obs["celltype"]
    data.obs["study"] = data.obs["sample"]
    sc.pl.umap(data, color=["cell_type"], save="pancreas_cell_before.pdf", show=False)
    sc.pl.umap(data, color=["study"], save="study_pancreas_before.pdf", show=False)
    sc.tl.pca(data, svd_solver='arpack')
    X_pca = data.obsm["X_pca"]
    labels = data.obs["batch"].tolist()
    print(f" average silhouette_score for original data  :{sk.metrics.silhouette_score(X_pca, labels)}")
    train(300)
    all_data, shared = vector_batch_removal()
    sc.pp.neighbors(all_data)
    sc.tl.umap(all_data)
    sc.pl.umap(all_data, color=["cell_type"], save="pancreas_cell_batched.pdf", frameon=False, show=False)
    sc.pl.umap(all_data, color=["study"], save="study_pancreas_batched.pdf", frameon=False, show=False)
    sc.tl.pca(all_data, svd_solver='arpack')
    X_pca = all_data.obsm["X_pca"]
    labels = all_data.obs["study"].tolist()
    print(f" average silhouette_score for scGen  :{sk.metrics.silhouette_score(X_pca,labels)}")


