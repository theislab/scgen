import tensorflow as tf
import numpy as np
import os
import scanpy.api as sc
from sklearn import  preprocessing
from random import shuffle
import wget
from data_reader import data_reader




# =============================== downloading training and validation files ====================================

train_path = "../data/train_kang.h5ad"
valid_path = "../data/valid_kang.h5ad"

if os.path.isfile(train_path):
    data = sc.read(train_path)
else:
    train_url = "https://drive.google.com/uc?export=download&id=1-RpxbXwXEJLYZDFSHnWYenojZ8TxRZsP"
    t_dl = wget.download(train_url, train_path)
    data = sc.read(train_path)
    
if os.path.isfile(valid_path):
    validation = sc.read(valid_path)
else:
    train_url = "https://drive.google.com/uc?export=download&id=1-RpxbXwXEJLYZDFSHnWYenojZ8TxRZsP"
    t_dl = wget.download(train_url,valid_path)
    validation = sc.read(valid_path)

# =============================== data gathering ====================================
#training cells
t_in = ['CD8T','NK','B','Dendritic', 'FCGR3A+Mono','CD14+Mono']
#heldout cells
t_out = [ 'CD4T']
dr = data_reader(data, validation,{"ctrl":"control", "stim":"stimulated"}, t_in, t_out)
train_real = dr.train_real_adata
valid_real = dr.valid_real_adata
le = preprocessing.LabelEncoder()
labels = le.fit_transform(train_real.obs["condition"].tolist())
input_matrix = train_real.X
ind_list = [i for i in range(input_matrix.shape[0])]
shuffle(ind_list)
train_data = input_matrix[ind_list, :]
labels = labels[ind_list]
validation_labels = le.transform(valid_real.obs["condition"].tolist()).reshape((len(valid_real),1))
#=============================== parameters ====================================
model_to_use = "../models/CVAE"
batch_size = 32
X_dim = input_matrix.shape[1]
z_dim = 100
h_dim = 200
lr = 0.001
inflate_to_size = 100
Y = tf.placeholder(tf.float32, shape=[None, 1],name="label")
X = tf.placeholder(tf.float32, shape=[None, X_dim],name="data")
z = tf.placeholder(tf.float32, shape=[None, z_dim],name="noise")
time_step = tf.placeholder(tf.int32)
size  = tf.placeholder(tf.int32)
is_training = tf.placeholder(tf.bool)
init_w =  tf.contrib.layers.xavier_initializer()
dr_rate = 0.5

#==================helper functions to map to latent space and reconstruct it back===========
def give_me_latent(data, labels):
    latent = sess.run(z_mean,feed_dict = {X : data, Y : labels, size:len(data),is_training:False})
    return  latent
def reconstruct(data, labels, use_data = False):

    if(use_data):
        latent = data
    else:
        latent = give_me_latent(data,labels)

    reconstruct = sess.run(X_hat,feed_dict = {z_mean : latent , Y : labels,is_training:False})
    return  reconstruct
# =============================== Q(z|X) ======================================
def Q(X,Y, reuse=False):
    with tf.variable_scope("gq", reuse=reuse):
        concat = tf.concat([X,Y],axis=1)
        h = tf.layers.dense(inputs=concat, units=700, kernel_initializer=init_w,use_bias=False )
        h = tf.layers.batch_normalization(h,axis= 1,training=is_training)
        h = tf.nn.leaky_relu(h)
        h = tf.layers.dropout(h,dr_rate, training= is_training)
        h = tf.layers.dense(inputs=h, units=400, kernel_initializer=init_w, use_bias=False,)
        h = tf.layers.batch_normalization(h, axis=1, training=is_training)
        h = tf.nn.leaky_relu(h)
        h = tf.layers.dropout(h,dr_rate, training= is_training)
        mean =  tf.layers.dense(inputs=h, units=z_dim, kernel_initializer=init_w)
        log_var =  tf.layers.dense(inputs=h, units=z_dim, kernel_initializer=init_w)
        return mean, log_var
# =============================== P(Z) ======================================
def sample_z(mu, log_var,size):
    eps = tf.random_normal(shape=[size,z_dim])
    return mu + tf.exp(log_var / 2) * eps
# =============================== P(X|z) ======================================

def P(z,Y,reuse=False):
    with tf.variable_scope("gp", reuse=reuse):
        concat = tf.concat([z,Y],axis=1)
        h = tf.layers.dense(inputs=concat,units= 400,kernel_initializer=init_w,use_bias=False)
        h = tf.layers.batch_normalization(h,axis= 1,training=is_training, )
        h = tf.nn.leaky_relu(h)
        h = tf.layers.dropout(h,dr_rate, training= is_training)
        h = tf.layers.dense(inputs=h, units=700, kernel_initializer=init_w,use_bias=False)
        tf.layers.batch_normalization(h,axis= 1,training=is_training)
        h = tf.nn.leaky_relu(h)
        h = tf.layers.dropout(h,dr_rate, training= is_training)
        h = tf.layers.dense(inputs=h, units=X_dim, kernel_initializer=init_w, use_bias=True)
        h = tf.nn.relu(h)
        return h


mean, log_var = Q(X,Y)
z_mean = sample_z(mean,log_var,size)
X_hat = P(z_mean,Y)
# =============================== loss ====================================
kl_loss = 0.5 * tf.reduce_sum(tf.exp(log_var) + log_var**2 - 1. - log_var, 1)
recon_loss = 0.5*tf.reduce_sum(tf.square((X-X_hat)), 1)
vae_loss = tf.reduce_mean(recon_loss + kl_loss)
global_step = tf.Variable(0, name='global_step', trainable=False, dtype=tf.int32)
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    Solver = tf.train.AdamOptimizer(learning_rate=lr).minimize(vae_loss)
sess=tf.InteractiveSession()
saver = tf.train.Saver(max_to_keep=1)
init = tf.global_variables_initializer().run()
# =============================== training ====================================
def train(n_epochs, early_stop_limit = 10, threshold = .1,  full_training = False, initial_run = True):
    if initial_run:
        print("Initial run")
        assign_step_zero = tf.assign(global_step, 0)
        init_step = sess.run(assign_step_zero)
    if not initial_run:
        saver.restore(sess, model_to_use)
    loss_hist = []
    patience = early_stop_limit
    min_delta = threshold
    patience_cnt = 0
    print("Training started")
    for it in range(n_epochs):
        increment_global_step_op = tf.assign(global_step, global_step + 1)
        step = sess.run(increment_global_step_op)
        current_step = sess.run(global_step)
        train_loss = 0
        if (full_training):
            input_matrix = train_data[0:train_data.shape[0] // batch_size * batch_size, :]
            X_valid = valid_real.X
            X_valid = X_valid[0:X_valid.shape[0] // batch_size * batch_size, :]
            for lower in range(0, input_matrix.shape[0], batch_size):
                upper = min(lower + batch_size, input_matrix.shape[0])
                X_mb = input_matrix[lower:upper, :]
                Y_mb = np.reshape(labels[lower:upper], (batch_size,1))
                _, D_loss_curr = sess.run(
                    [Solver, vae_loss], feed_dict={X: X_mb, Y:Y_mb, time_step: current_step,size:batch_size,is_training:True}
                )
                train_loss += D_loss_curr
        ###early stoping
        valid_loss = 0
        for lower in range(0, X_valid.shape[0], batch_size):
            upper = min(lower + batch_size, X_valid.shape[0])
            X_mb = X_valid[lower:upper, :]
            Y_mb = np.reshape(validation_labels[lower:upper], (batch_size, 1))
            valid_loss_epoch = sess.run(
                vae_loss,
                feed_dict={X: X_mb, Y: Y_mb, time_step: current_step, size: batch_size, is_training: False})
            valid_loss += valid_loss_epoch
        loss_hist.append(valid_loss / X_valid.shape[0])
        if it > 0 and loss_hist[it - 1] - loss_hist[it] > min_delta:
            patience_cnt = 0
        else:
            patience_cnt += 1
        if patience_cnt > patience:
            print("early stopping")
            print(f" current  loss : {loss_hist[it - 1]}, last loss :  {loss_hist[it]}, patience : {patience_cnt}")
            save_path = saver.save(sess, model_to_use)
            print("Training finished")
            break
    save_path = saver.save(sess, model_to_use)
    print("Model saved in file: %s" % save_path)
    print("Training finished")



if __name__ == "__main__":
    sc.settings.figdir = "../results"
    train(150, initial_run=True, full_training=True, early_stop_limit=20, threshold =0.0025)
    adata_list =  dr.extractor(data,"CD4T")
    ctrl_CD4T = adata_list[1]
    #here we add fake stimulated labels (1) to control cells with label 0
    fake_labels = np.ones((len(ctrl_CD4T), 1))
    predicted_cells = reconstruct(ctrl_CD4T.X.A, fake_labels)
    all_Data = sc.AnnData(np.concatenate([adata_list[1].X.A, adata_list[2].X.A, predicted_cells]))
    all_Data.obs["condition"] = ["ctrl"] * len(adata_list[1].X.A) + ["real_stim"] * len(adata_list[2].X.A) +\
                                ["pred_stim"] * len(predicted_cells)
    all_Data.var_names = adata_list[3].var_names
    dr.reg_mean_plot(all_Data, "../results/", "CVAE ")
    dr.reg_var_plot(all_Data, "../results/", "CVAE ")
    sc.pl.violin(all_Data, groupby="condition", keys="ISG15", save = "_ISG15_CVAE.pdf", show=False)
    sc.pp.neighbors(all_Data)
    sc.tl.umap(all_Data)
    sc.pl.umap(all_Data, color=["condition"], save="_CVAE.pdf", show=False)
    lbs = le.transform(train_real.obs["condition"].tolist()).reshape((len(train_real), 1))
    low_dim = give_me_latent(train_real.X, lbs)
    dt = sc.AnnData(low_dim)
    sc.pp.neighbors(dt)
    dt.obs["cell_type"] = train_real.obs["cell_type"]
    dt.obs["condition"] = train_real.obs["condition"]
    sc.pp.neighbors(train_real)
    sc.tl.umap(dt)
    sc.pl.umap(dt, color=["cell_type", "condition"], show=False, save="_CVAE_latent.pdf")


