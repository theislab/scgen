import tensorflow as tf
import numpy as np
from data_reader import data_reader


class scGen(object):

    def __init__(self, train_data, valid_data,conditions,tr_ct_list =None,ho_ct_list=None,
                 dr_rate=.2, learning_rate=0.001, batch_size=32, z_dimension=100, use_validation=True,
                 model_path="./models/scGen"):

        self.dr = data_reader(train_data, valid_data,conditions, tr_ct_list, ho_ct_list)
        self.train_real = self.dr.train_real
        self.use_validation = use_validation
        if  self.use_validation:
            self.valid_real = self.dr.valid_real
        self.X_dim =  self.train_real.shape[1]
        self.is_training = tf.placeholder(tf.bool, name='training_flag')
        self.z_dim = z_dimension
        self.lr = learning_rate
        self.conditions = conditions
        self.dr_rate = dr_rate
        self.model_to_use = model_path
        self.batch_size = batch_size
        self.global_step = tf.Variable(0, name='global_step', trainable=False, dtype=tf.int32)
        self.X = tf.placeholder(tf.float32, shape=[None, self.X_dim ], name="data")
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim], name="noise")
        self.time_step = tf.placeholder(tf.int32)
        self.size = tf.placeholder(tf.int32)
        self.init_w = tf.contrib.layers.xavier_initializer()
        self._create_network()
        self._loss_function(self.X_hat,self.X,self.mu, self.log_var)
        self.sess = tf.InteractiveSession()
        self.saver = tf.train.Saver(max_to_keep=1)
        self.init = tf.global_variables_initializer().run()

    def _Q(self, reuse=False):
        with tf.variable_scope("gq", reuse=reuse):
            h = tf.layers.dense(inputs=self.X, units=800, kernel_initializer=self.init_w, use_bias=False)
            h = tf.layers.batch_normalization(h, axis=1, training=self.is_training)
            h = tf.nn.leaky_relu(h)
            h = tf.layers.dropout(h, self.dr_rate, training=self.is_training)
            h = tf.layers.dense(inputs=h, units=800, kernel_initializer=self.init_w, use_bias=False)
            h = tf.layers.batch_normalization(h, axis=1, training=self.is_training)
            h = tf.nn.leaky_relu(h)
            h = tf.layers.dropout(h,self.dr_rate, training=self.is_training)
            mean = tf.layers.dense(inputs=h, units=self.z_dim, kernel_initializer=self.init_w)
            log_var = tf.layers.dense(inputs=h, units=self.z_dim, kernel_initializer=self.init_w)
            return mean, log_var

    def _P(self, z, reuse=False):
        with tf.variable_scope("gp", reuse=reuse):
            h = tf.layers.dense(inputs=z, units=800, kernel_initializer=self.init_w, use_bias=False)
            h = tf.layers.batch_normalization(h, axis=1, training=self.is_training)
            h = tf.nn.leaky_relu(h)
            h = tf.layers.dropout(h, self.dr_rate, training=self.is_training)

            h = tf.layers.dense(inputs=h, units=800, kernel_initializer=self.init_w, use_bias=False)
            tf.layers.batch_normalization(h, axis=1, training=self.is_training)
            h = tf.nn.leaky_relu(h)
            h = tf.layers.dropout(h, self.dr_rate, training=self.is_training)
            h = tf.layers.dense(inputs=h, units=self.X_dim, kernel_initializer=self.init_w, use_bias=True)
            h = tf.nn.relu(h)
            return h

    def _sample_z(self, mu, log_var, size):
        eps = tf.random_normal(shape=[size, self.z_dim])
        return mu + tf.exp(log_var / 2) * eps

    def _create_network(self):
        self.mu, self.log_var = self._Q()
        self.z_mean = self._sample_z(self.mu, self.log_var, self.size)
        self.X_hat = self._P(self.z_mean)

    def _loss_function(self, x_hat, x, mu, log_var):
        kl_loss = 0.5 * tf.reduce_sum(tf.exp(log_var) + tf.square(log_var) - 1. - log_var, 1)
        recon_loss = 0.5 * tf.reduce_sum(tf.square((x - x_hat)), 1)
        self.vae_loss = tf.reduce_mean(recon_loss + .01 * kl_loss)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.solver = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.vae_loss)

    def _to_latent(self, data):
        latent = self.sess.run(self.z_mean, feed_dict={self.X: data, self.size: len(data), self.is_training: False})
        return latent

    def _avg_vector(self,data):
        latent =  self._to_latent(data)
        latent_avg =  np.average(latent,axis=0)
        return  latent_avg

    def _reconstruct(self, data,use_data=False):
        if(use_data):
            latent = data
        else:
            latent = self._to_latent(data)
        rec_data = self.sess.run(self.X_hat,feed_dict = {self.z_mean:latent, self.is_training:False})
        return  rec_data

    def linear_interploation(self, x, y, nb_steps):
        start = self._to_latent(np.average(x, axis=0).reshape((1, x.shape[1])))
        end = self._to_latent(np.average(y, axis=0).reshape((1, x.shape[1])))
        vectors = np.zeros((nb_steps, start.shape[1]))
        alphaValues = np.linspace(0, 1, nb_steps)
        for i, alpha in enumerate(alphaValues):
            vector = start * (1 - alpha) + end * alpha
            vectors[i, :] = vector
        vectors = np.array(vectors)
        interpolation = self._reconstruct(vectors, use_data=True)
        return interpolation

    def predict(self, celltype_to_predict):
        cd_x = self.dr.train_real_adata[self.dr.train_real_adata.obs["condition"] == self.conditions["ctrl"], :]
        cd_x =  self.dr.balancer(cd_x)
        stim_x = self.dr.train_real_adata[self.dr.train_real_adata.obs["condition"] == self.conditions["stim"], :]
        stim_x = self.dr.balancer(stim_x)
        cd_y = self.dr.extractor(self.dr.train_real_adata,celltype_to_predict)[1]
        eq = min(cd_x.X.shape[0], stim_x.X.shape[0])
        cd_ind = np.random.choice(range(cd_x.shape[0]), size=eq, replace=False)
        stim_ind = np.random.choice(range(stim_x.shape[0]), size=eq, replace=False)
        lat_cd = self._avg_vector(cd_x.X[cd_ind, :])
        lat_stim = self._avg_vector(stim_x.X[stim_ind, :])
        delta = lat_stim - lat_cd
        latent_cd = self._to_latent(cd_y.X)
        stim_pred = delta + latent_cd
        predicted_cells = self._reconstruct(stim_pred, use_data=True)
        return predicted_cells, delta


    def predict_cross(self, data):
        cd_x = self.dr.train_real_adata[self.dr.train_real_adata.obs["condition"] == self.conditions["ctrl"], :]
        cd_x =  self.dr.balancer(cd_x)
        stim_x = self.dr.train_real_adata[self.dr.train_real_adata.obs["condition"] == self.conditions["stim"], :]
        stim_x = self.dr.balancer(stim_x)
        cd_y = data
        eq = min(cd_x.X.shape[0], stim_x.X.shape[0])
        cd_ind = np.random.choice(range(cd_x.shape[0]), size=eq, replace=False)
        stim_ind = np.random.choice(range(stim_x.shape[0]), size=eq, replace=False)
        lat_cd = self._avg_vector(cd_x.X[cd_ind, :])
        lat_stim = self._avg_vector(stim_x.X[stim_ind, :])
        delta = lat_stim - lat_cd
        latent_cd = self._to_latent(cd_y)
        stim_pred = delta + latent_cd
        predicted_cells = self._reconstruct(stim_pred, use_data=True)
        return predicted_cells, delta

    def restore_model(self):
        self.saver.restore(self.sess, self.model_to_use)

    def linear_interploation(self, x, y, nb_steps):
        start = self._to_latent(np.average(x, axis=0).reshape((1, x.shape[1])))
        end = self._to_latent(np.average(y, axis=0).reshape((1, x.shape[1])))
        vectors = np.zeros((nb_steps, start.shape[1]))
        alphaValues = np.linspace(0, 1, nb_steps)
        for i, alpha in enumerate(alphaValues):
            vector = start * (1 - alpha) + end * alpha
            vectors[i, :] = vector
        vectors = np.array(vectors)
        interpolation = self._reconstruct(vectors, use_data=True)
        return interpolation




    def train(self, n_epochs, early_stop_limit=20, threshold=0.0025, full_training=True, initial_run=True):
        if initial_run:
            print("----Training----")
            assign_step_zero = tf.assign(self.global_step, 0)
            init_step = self.sess.run(assign_step_zero)
        if not initial_run:
            self.saver.restore(self.sess, self.model_to_use)
        loss_hist = []
        patience = early_stop_limit
        min_delta = threshold
        patience_cnt = 0
        for it in range(n_epochs):
            increment_global_step_op = tf.assign(self.global_step,  self.global_step + 1)
            step = self.sess.run(increment_global_step_op)
            current_step = self.sess.run(self.global_step)
            train_loss = 0
            if (full_training):
                input_ltpm_matrix = self.train_real[0:self.train_real.shape[0] // self.batch_size * self.batch_size, :]
                if self.use_validation:
                    X_valid = self.valid_real[0:self.valid_real.shape[0] // self.batch_size * self.batch_size, :]
                for lower in range(0, input_ltpm_matrix.shape[0], self.batch_size):
                    upper = min(lower + self.batch_size, input_ltpm_matrix.shape[0])
                    X_mb = input_ltpm_matrix[lower:upper, :]
                    _, D_loss_curr = self.sess.run(
                        [self.solver, self.vae_loss], feed_dict={self.X: X_mb, self.time_step: current_step,
                                                               self.size: self.batch_size, self.is_training: True})
                    train_loss += D_loss_curr

            if self.use_validation:
                valid_loss = 0
                for lower in range(0, X_valid.shape[0], self.batch_size):
                    upper = min(lower + self.batch_size, X_valid.shape[0])

                    X_mb = X_valid[lower:upper, :]
                    valid_loss_epoch = self.sess.run(self.vae_loss, feed_dict={self.X: X_mb, self.time_step: current_step,
                                                                     self.size: self.batch_size, self.is_training: False})
                    valid_loss += valid_loss_epoch
                loss_hist.append(valid_loss / X_valid.shape[0])
                if it > 0 and loss_hist[it - 1] - loss_hist[it] > min_delta:
                    patience_cnt = 0
                else:
                    patience_cnt += 1
                if patience_cnt > patience:
                    save_path = self.saver.save(self.sess, self.model_to_use)
                    break
        save_path = self.saver.save(self.sess, self.model_to_use)
        print(f"Model saved in file: {save_path}")
        print(f"Training finished")





