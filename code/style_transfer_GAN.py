import tensorflow as tf
import numpy as np
import os
import scanpy.api as sc
import wget
from data_reader import data_reader
from random import shuffle


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
    t_dl = wget.download(train_url, valid_path)
    validation = sc.read(valid_path)
# =============================== data gathering ====================================
#training cells
t_in = ['CD8T','NK','B','Dendritic', 'FCGR3A+Mono','CD14+Mono']
#heldout cells
t_out = [ 'CD4T']
dr = data_reader(data, validation,{"ctrl":"control", "stim":"stimulated"}, t_in, t_out)
train_real = dr.train_real_adata
train_real_stim = train_real[train_real.obs["condition"] == "stimulated"]
train_real_ctrl = train_real[train_real.obs["condition"] == "control"]
train_real_stim = train_real_stim.X
ind_list = [i for i in range(train_real_stim.shape[0])]
shuffle(ind_list)
train_real_stim = train_real_stim[ind_list, :]
gex_size = train_real_stim.shape[1]
train_real_ctrl = train_real_ctrl.X
ind_list = [i for i in range(train_real_ctrl.shape[0])]
shuffle(ind_list)
train_real_ctrl = train_real_ctrl[ind_list, :]
eq = min(len(train_real_ctrl), len(train_real_stim))
stim_ind = np.random.choice(range(len(train_real_stim)), size=eq, replace=False)
ctrl_ind = np.random.choice(range(len(train_real_ctrl)), size=eq, replace=False)
##selecting equal size for both stimulated and control cells
train_real_ctrl = train_real_ctrl[ctrl_ind,:]
train_real_stim = train_real_stim[stim_ind, :]
# =============================== parameters ====================================
model_to_use = "../models/style_transfer"
X_dim = gex_size
z_dim = 100
h_dim = 200
batch_size = 512
inflate_to_size = 100
lambda_l2 = .8
arch = {"noise_input_size": z_dim, "inflate_to_size":inflate_to_size,
 "epochs": 0 , "bsize":batch_size,"disc_internal_size ": h_dim, "#disc_train":1}
X_stim = tf.placeholder(tf.float32, shape=[None, X_dim],name="data_stim")
X_ctrl =   tf.placeholder(tf.float32, shape=[None, X_dim],name="data_ctrl")
time_step = tf.placeholder(tf.int32)
size  = tf.placeholder(tf.int32)
learning_rate = 0.001
initializer = tf.truncated_normal_initializer(stddev=0.02)
is_training = tf.placeholder(tf.bool)
dr_rate = .5
const = 5
### helper function


def predict(ctrl):
   pred  =   sess.run(gen_stim_fake,feed_dict={X_ctrl :ctrl , is_training: False} )
   return  pred


def low_embed(all):
   pred  =   sess.run(disc_c,feed_dict={X_ctrl :all , is_training: False} )
   return  pred


def low_embed_stim(all):
   pred  =   sess.run(disc_s,feed_dict={X_stim :all , is_training: False} )
   return  pred



# network

def discriminator_stimulated(tensor, reuse=False,):
    with tf.variable_scope("discriminator_s",reuse = reuse):
        h = tf.layers.dense(inputs=tensor, units=700, kernel_initializer=initializer, use_bias=False)
        h = tf.layers.batch_normalization(h, axis=1, training=is_training)
        h = tf.nn.leaky_relu(h)
        h = tf.layers.dropout(h, dr_rate, training=is_training)
        h = tf.layers.dense(inputs=h, units=100, kernel_initializer=initializer, use_bias=False, )
        h = tf.layers.batch_normalization(h, axis=1, training=is_training)
        disc = tf.nn.leaky_relu(h)
        h = tf.layers.dropout(disc, dr_rate, training=is_training)

        h = tf.layers.dense(inputs=h, units=1, kernel_initializer=initializer, use_bias=False)
        h = tf.nn.sigmoid(h)

        return h, disc


def discriminator_control(tensor, reuse=False,):

    with tf.variable_scope("discriminator_b",reuse = reuse):
        h = tf.layers.dense(inputs=tensor, units=700, kernel_initializer=initializer, use_bias=False)
        h = tf.layers.batch_normalization(h, axis=1, training=is_training)
        h = tf.nn.leaky_relu(h)
        h = tf.layers.dropout(h, dr_rate, training=is_training)

        h = tf.layers.dense(inputs=h, units=100, kernel_initializer=initializer, use_bias=False, )
        h = tf.layers.batch_normalization(h, axis=1, training=is_training)
        disc = tf.nn.leaky_relu(h)
        h = tf.layers.dropout(disc, dr_rate, training=is_training)

        h = tf.layers.dense(inputs=h, units=1, kernel_initializer=initializer, use_bias=False)
        h = tf.nn.sigmoid(h)
        return h,disc


def generator_stim_ctrl(image,  reuse=False):
    with tf.variable_scope("generator_sb",reuse = reuse ):
        h = tf.layers.dense(inputs=image, units=700, kernel_initializer=initializer, use_bias=False)
        h = tf.layers.batch_normalization(h, axis=1, training=is_training)
        h = tf.nn.leaky_relu(h)
        h = tf.layers.dropout(h, dr_rate, training=is_training)

        h = tf.layers.dense(inputs=h, units=100, kernel_initializer=initializer, use_bias=False)
        h = tf.layers.batch_normalization(h, axis=1, training=is_training)
        h = tf.nn.leaky_relu(h)
        h = tf.layers.dropout(h, dr_rate, training=is_training)

        h = tf.layers.dense(inputs=h, units=50, kernel_initializer=initializer, use_bias=False, )
        h = tf.layers.batch_normalization(h, axis=1, training=is_training)
        h = tf.nn.leaky_relu(h)
        h = tf.layers.dropout(h, dr_rate, training=is_training)

        h = tf.layers.dense(inputs=h, units=100, kernel_initializer=initializer, use_bias=False)
        h = tf.layers.batch_normalization(h, axis=1, training=is_training)
        h = tf.nn.leaky_relu(h)
        h = tf.layers.dropout(h, dr_rate, training=is_training)

        h = tf.layers.dense(inputs=h, units=700, kernel_initializer=initializer, use_bias=False)
        h = tf.layers.batch_normalization(h, axis=1, training=is_training)
        h = tf.nn.leaky_relu(h)
        h = tf.layers.dropout(h, dr_rate, training=is_training)

        h = tf.layers.dense(inputs=h, units=X_dim, kernel_initializer=initializer, use_bias=False, )
        h = tf.layers.batch_normalization(h, axis=1, training=is_training)
        h = tf.nn.relu(h)
        return h


def generator_ctrl_stim(image, reuse=False,):
    with tf.variable_scope("generator_bs",reuse = reuse):
        h = tf.layers.dense(inputs=image, units=700, kernel_initializer=initializer, use_bias=False)
        h = tf.layers.batch_normalization(h, axis=1, training=is_training)
        h = tf.nn.leaky_relu(h)
        h = tf.layers.dropout(h, dr_rate, training=is_training)

        h = tf.layers.dense(inputs=h, units=100, kernel_initializer=initializer, use_bias=False)
        h = tf.layers.batch_normalization(h, axis=1, training=is_training)
        h = tf.nn.leaky_relu(h)
        h = tf.layers.dropout(h, dr_rate, training=is_training)

        h = tf.layers.dense(inputs=h, units=50, kernel_initializer=initializer, use_bias=False, )
        h = tf.layers.batch_normalization(h, axis=1, training=is_training)
        h = tf.nn.leaky_relu(h)
        h = tf.layers.dropout(h, dr_rate, training=is_training)

        h = tf.layers.dense(inputs=h, units=100, kernel_initializer=initializer, use_bias=False)
        h = tf.layers.batch_normalization(h, axis=1, training=is_training)
        h = tf.nn.leaky_relu(h)
        h = tf.layers.dropout(h, dr_rate, training=is_training)

        h = tf.layers.dense(inputs=h, units=700, kernel_initializer=initializer, use_bias=False)
        h = tf.layers.batch_normalization(h, axis=1, training=is_training)
        h = tf.nn.leaky_relu(h)
        h = tf.layers.dropout(h, dr_rate, training=is_training)

        h = tf.layers.dense(inputs=h, units=X_dim, kernel_initializer=initializer, use_bias=False, )
        h = tf.layers.batch_normalization(h, axis=1, training=is_training)
        h = tf.nn.relu(h)

        return h


# generator and discriminator

gen_stim_fake = generator_ctrl_stim(X_ctrl)
gen_ctrl_fake = generator_stim_ctrl(X_stim)

recon_ctrl = generator_stim_ctrl(gen_stim_fake, reuse=True)
recon_stim = generator_ctrl_stim(gen_ctrl_fake, reuse=True)

disc_ctrl_fake, _ = discriminator_control(gen_ctrl_fake)
disc_stim_fake, _ =  disc_s = discriminator_stimulated(gen_stim_fake)

disc_ctrl_real, disc_c = discriminator_control(X_ctrl, reuse=True)
disc_stim_real, disc_s= discriminator_stimulated(X_stim, reuse=True)

# computing loss

const_loss_s = tf.reduce_sum(tf.losses.mean_squared_error(recon_ctrl, X_ctrl))
const_loss_b = tf.reduce_sum(tf.losses.mean_squared_error(recon_stim, X_stim))

gen_ctrl_loss = tf.reduce_sum(tf.square(disc_ctrl_fake - 1)) / 2
gen_stim_loss = tf.reduce_sum(tf.square(disc_stim_fake - 1)) / 2

disc_ctrl_loss = tf.reduce_sum(tf.square(disc_ctrl_real - 1) + tf.square(disc_ctrl_fake)) / 2
disc_stim_loss = tf.reduce_sum(tf.square(disc_stim_real - 1) + tf.square(disc_stim_fake)) / 2

gen_loss = const * (const_loss_s + const_loss_b) + gen_ctrl_loss + gen_stim_loss
disc_loss = disc_ctrl_loss + disc_stim_loss

# applying gradients

gen_sb_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator_sb")
gen_bs_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator_bs")
dis_s_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator_s")
dis_b_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator_b")



with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    update_D = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(disc_loss,
                                                                               var_list=dis_s_variables + dis_b_variables,
                                                                            )
    update_G = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(gen_loss,
                                                                               var_list=gen_sb_variables + gen_bs_variables)

global_step = tf.Variable(0, name='global_step', trainable=False, dtype=tf.int32)


sess=tf.InteractiveSession()
saver = tf.train.Saver(max_to_keep=1)
init = tf.global_variables_initializer().run()


def train(n_epochs, initial_run=True):
    if initial_run:
        print("Initial run")
        print("Training started")
        assign_step_zero = tf.assign(global_step, 0)
        init_step = sess.run(assign_step_zero)
    if not initial_run:
        saver.restore(sess, model_to_use)
        current_step = sess.run(global_step)
    for it in range(n_epochs):
        increment_global_step_op = tf.assign(global_step, global_step + 1)
        step = sess.run(increment_global_step_op)
        current_step = sess.run(global_step)
        batch_ind1 = np.random.choice(range(len(train_real_stim)), size=eq, replace=False)
        mb_ctrl = train_real_ctrl[batch_ind1, :]
        mb_stim = train_real_stim[batch_ind1, :]
        for gen_it in range(2):
            _, g_loss, d_loss = sess.run([update_G, gen_loss, disc_loss],
                                         feed_dict={X_ctrl: mb_ctrl, X_stim: mb_stim, is_training: True})
        _ = sess.run(update_D,feed_dict={X_ctrl :mb_ctrl, X_stim : mb_stim, is_training:True})
    save_path = saver.save(sess, model_to_use)
    print("Model saved in file: %s" % save_path)
    print(f"Training finished")


if __name__ == "__main__":
    sc.settings.figdir = "../results"
    train(1000, initial_run=True)
    adata_list =  dr.extractor(data,"CD4T")
    ctrl_CD4T = adata_list[1]
    predicted_cells = predict(ctrl_CD4T.X.A)
    all_Data = sc.AnnData(np.concatenate([adata_list[1].X.A, adata_list[2].X.A, predicted_cells]))
    all_Data.obs["condition"] = ["ctrl"] * len(adata_list[1].X.A) + ["real_stim"] * len(adata_list[2].X.A) +\
                                ["pred_stim"] * len(predicted_cells)
    all_Data.var_names = adata_list[3].var_names
    dr.reg_mean_plot(all_Data, "../results/", "style_trasnfer")
    dr.reg_var_plot(all_Data, "../results/", "style_trasnfer ")
    sc.pl.violin(all_Data, groupby="condition", keys="ISG15", save = "_ISG15_style_trasnfer.pdf", show=False)
    sc.pp.neighbors(all_Data)
    sc.tl.umap(all_Data)
    sc.pl.umap(all_Data, color=["condition"], save="style_trasnfer.pdf", show=False)
    low_dim = low_embed_stim(train_real.X)
    dt = sc.AnnData(low_dim)
    sc.pp.neighbors(dt)
    sc.tl.umap(dt)
    dt.obs["cell_type"] = train_real.obs["cell_type"]
    dt.obs["condition"] = train_real.obs["condition"]
    sc.pl.umap(dt, color=["cell_type", "condition"], show=False, save="_style_transfer_latent.pdf")

