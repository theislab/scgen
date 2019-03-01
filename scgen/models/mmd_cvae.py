import logging
import os

import keras
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.callbacks import EarlyStopping, CSVLogger
from keras.layers import Dense, BatchNormalization, Dropout, Input, concatenate, LeakyReLU, ReLU, Lambda
from keras.models import Model, load_model
from scipy import sparse

from scgen.models.util import shuffle_data, label_encoder

log = logging.getLogger(__file__)


class MMDCVAE:
    """
        MMD C-VAE vector Network class. This class contains the implementation of Conditional
        Variational Auto-encoder network.

        # Parameters
            kwargs:
                key: `dropout_rate`: float
                        dropout rate
                key: `learning_rate`: float
                    learning rate of optimization algorithm
                key: `model_path`: basestring
                    path to save the model after training
                key: `alpha`: float
                    alpha coefficient for loss.
                key: `beta`: float
                    beta coefficient for loss.
            x_dimension: integer
                number of gene expression space dimensions.
            z_dimension: integer
                number of latent space dimensions.
    """

    def __init__(self, x_dimension, z_dimension=100, **kwargs):
        self.x_dim = x_dimension
        self.z_dim = z_dimension
        self.lr = kwargs.get("learning_rate", 0.001)
        self.alpha = kwargs.get("alpha", 0.1)
        self.beta = kwargs.get("beta", 1)
        self.conditions = kwargs.get("condition_list")
        self.dr_rate = kwargs.get("dropout_rate", 0.2)
        self.model_to_use = kwargs.get("model_path", "./")
        self.batch_mmd = kwargs.get("batch_mmd", True)
        self.x = Input(shape=(self.x_dim,), name="data")
        self.y = Input(shape=(1,), name="labels")
        self.z = Input(shape=(self.z_dim,), name="latent_data")
        self.source_x = Input(shape=(self.x_dim,), name="source_x")
        self.source_y = Input(shape=(1,), name="source_y")
        self.source_z = Input(shape=(self.z_dim,), name="source_z")
        self.dest_x = Input(shape=(self.x_dim,), name="dest_x")
        self.dest_y = Input(shape=(1,), name="dest_y")
        self.dest_z = Input(shape=(self.z_dim,), name="dest_z")
        self.init_w = keras.initializers.glorot_normal()
        self._create_network()
        self._loss_function()
        self.cvae_model.summary()

    def _encoder(self, x, y, name="encoder"):
        """
            Constructs the encoder sub-network of C-VAE. This function implements the
            encoder part of Variational Auto-encoder. It will transform primary
            data in the `n_vars` dimension-space to the `z_dimension` latent space.

            # Parameters
                No parameters are needed.

            # Returns
                mean: Tensor
                    A dense layer consists of means of gaussian distributions of latent space dimensions.
                log_var: Tensor
                    A dense layer consists of log transformed variances of gaussian distributions of latent space dimensions.
        """
        xy = concatenate([x, y], axis=1)
        h = Dense(256, kernel_initializer=self.init_w, use_bias=False)(xy)
        h = BatchNormalization(axis=1)(h)
        h = LeakyReLU()(h)
        h = Dropout(self.dr_rate)(h)
        h = Dense(128, kernel_initializer=self.init_w, use_bias=False)(h)
        h = BatchNormalization(axis=1)(h)
        h = LeakyReLU()(h)
        h = Dropout(self.dr_rate)(h)
        mean = Dense(self.z_dim, kernel_initializer=self.init_w)(h)
        log_var = Dense(self.z_dim, kernel_initializer=self.init_w)(h)
        z = Lambda(self._sample_z, output_shape=(self.z_dim,))([mean, log_var])
        # source and dest data are not connected to encoder's 1st dense but will used for mmd batch computation
        model = Model(inputs=[x, y], outputs=[mean, log_var, z], name=name)
        return mean, log_var, model

    def _mmd_decoder(self, x, y, name="decoder"):
        """
            Constructs the decoder sub-network of C-VAE. This function implements the
            decoder part of Variational Auto-encoder. It will transform constructed
            latent space to the previous space of data with n_dimensions = n_vars.

            # Parameters
                No parameters are needed.

            # Returns
                h: Tensor
                    A Tensor for last dense layer with the shape of [n_vars, ] to reconstruct data.

        """
        xy = concatenate([x, y], axis=1)
        h = Dense(128, kernel_initializer=self.init_w, use_bias=False)(xy)
        h = BatchNormalization(axis=1)(h)
        h_mmd = LeakyReLU(name="mmd")(h)
        h = Dense(256, kernel_initializer=self.init_w, use_bias=False)(h_mmd)
        h = BatchNormalization(axis=1)(h)
        h = LeakyReLU()(h)
        h = Dropout(self.dr_rate)(h)
        h = Dense(self.x_dim, kernel_initializer=self.init_w, use_bias=True)(h)
        h = ReLU()(h)
        model = Model(inputs=[x, y], outputs=[h, h_mmd], name=name)
        return h, h_mmd, model

    @staticmethod
    def _sample_z(args):
        """
            Samples from standard Normal distribution with shape [size, z_dim] and
            applies re-parametrization trick. It is actually sampling from latent
            space distributions with N(mu, var) computed in `_encoder` function.

            # Parameters
                No parameters are needed.

            # Returns
                The computed Tensor of samples with shape [size, z_dim].
        """
        mu, log_var = args
        batch_size = K.shape(mu)[0]
        z_dim = K.int_shape(mu)[1]
        eps = K.random_normal(shape=[batch_size, z_dim])
        return mu + K.exp(log_var / 2) * eps

    def _create_network(self):
        """
            Constructs the whole C-VAE network. It is step-by-step constructing the C-VAE
            network. First, It will construct the encoder part and get mu, log_var of
            latent space. Second, It will sample from the latent space to feed the
            decoder part in next step. Finally, It will reconstruct the data by
            constructing decoder part of C-VAE.

            # Parameters
                No parameters are needed.

            # Returns
                Nothing will be returned.
        """

        # normal flow
        self.pseudo_labels = Input(shape=(1,), name="pseudo_labels")
        if self.batch_mmd:
            inputs = [self.x, self.y, self.pseudo_labels]
        else:
            inputs = [self.x, self.y, self.pseudo_labels]
        self.mu, self.log_var, self.encoder_model = self._encoder(*inputs[:2], name="encoder")
        self.x_hat, self.mmd_hl, self.decoder_model = self._mmd_decoder(self.z, self.pseudo_labels,
                                                                        name="decoder")
        self.cvae_model = Model(inputs=inputs,
                                outputs=self.decoder_model([self.encoder_model(inputs[:2])[2], self.pseudo_labels]),
                                name="cvae")

        # source flow
        # source_inputs = [self.source_x, self.source_y]
        # self.mu_s, self.log_var_s, self.source_encoder_model = self._encoder(*source_inputs, name="source_encoder")
        # self.x_hat_s, self.mmd_s, self.source_decoder_model = self._mmd_decoder(
        #     self.source_z,
        #     self.source_y, name="source_decoder")
        # self.source_cvae_model = Model(inputs=source_inputs,
        #                                outputs=self.source_decoder_model(
        #                                    [self.source_encoder_model(source_inputs)[2], self.source_y]),
        #                                name="source_cvae")
        #
        # # destination flow
        # dest_inputs = [self.dest_x, self.dest_y]
        # self.mu_d, self.log_var_d, self.dest_encoder_model = self._encoder(*dest_inputs, name="dest_encoder")
        # self.x_hat_d, self.mmd_d, self.dest_decoder_model = self._mmd_decoder(self.dest_z,
        #                                                                       self.dest_y, name="dest_decoder")
        # self.dest_cvae_model = Model(inputs=dest_inputs,
        #                              outputs=self.dest_decoder_model(
        #                                  [self.dest_encoder_model(dest_inputs)[2], self.dest_y]),
        #                              name="dest_cvae")

    @staticmethod
    def compute_kernel(x, y):
        """
            Computes RBF kernel between x and y.

            # Parameters
                x: Tensor
                    Tensor with shape [batch_size, z_dim]
                y: Tensor
                    Tensor with shape [batch_size, z_dim]

            # Returns
                returns the computed RBF kernel between x and y
        """
        x_size = K.shape(x)[0]
        y_size = K.shape(y)[0]
        dim = K.shape(x)[1]
        tiled_x = K.tile(K.reshape(x, K.stack([x_size, 1, dim])), K.stack([1, y_size, 1]))
        tiled_y = K.tile(K.reshape(y, K.stack([1, y_size, dim])), K.stack([x_size, 1, 1]))
        return K.exp(-K.mean(K.square(tiled_x - tiled_y), axis=2) / K.cast(dim, tf.float32))

    @staticmethod
    def compute_mmd(x, y):  # [batch_size, z_dim] [batch_size, z_dim]
        """
            Computes Maximum Mean Discrepancy(MMD) between x and y.

            # Parameters
                x: Tensor
                    Tensor with shape [batch_size, z_dim]
                y: Tensor
                    Tensor with shape [batch_size, z_dim]

            # Returns
                returns the computed MMD between x and y
        """
        x_kernel = MMDCVAE.compute_kernel(x, x)
        y_kernel = MMDCVAE.compute_kernel(y, y)
        xy_kernel = MMDCVAE.compute_kernel(x, y)
        return K.mean(x_kernel) + K.mean(y_kernel) - 2 * K.mean(xy_kernel)

    def _loss_function(self, data=np.zeros((10, 6998)), labels=np.zeros((10, 1))):
        """
            Defines the loss function of C-VAE network after constructing the whole
            network. This will define the KL Divergence and Reconstruction loss for
            C-VAE and also defines the Optimization algorithm for network. The C-VAE Loss
            will be weighted sum of reconstruction loss and KL Divergence loss.

            # Parameters
                No parameters are needed.

            # Returns
                Nothing will be returned.
        """

        def batch_loss():
            def cvae_kl_re_loss(mu, log_var):
                def loss(y_true, y_pred):
                    kl_loss = 0.5 * K.mean(K.exp(log_var) + K.square(mu) - 1. - log_var, 1)
                    recon_loss = 0.5 * K.sum(K.square((y_true - y_pred)), axis=1)
                    return recon_loss + self.alpha * kl_loss

                return loss

            def cvae_mmd_loss(y_true, y_pred):
                data = self.cvae_model.inputs[0]
                labels = self.cvae_model.inputs[1]
                bool_mask = K.equal(labels, K.zeros(shape=K.shape(labels)))
                bool_mask = K.reshape(bool_mask, shape=(-1,))
                source_x = tf.boolean_mask(data, bool_mask)
                source_x = K.reshape(source_x, shape=(-1, K.get_variable_shape(source_x)[1]))
                source_y = K.zeros(shape=(K.shape(source_x)[0], 1))
                bool_mask = K.equal(labels, K.ones(shape=K.shape(labels)))
                bool_mask = K.reshape(bool_mask, shape=(-1,))
                dest_x = tf.boolean_mask(data, bool_mask)
                dest_x = K.reshape(dest_x, shape=(-1, K.get_variable_shape(dest_x)[1]))
                dest_y = K.ones(shape=(K.shape(dest_x)[0], 1))

                mmd_s = self._to_mmd_layer(model=self.cvae_model,
                                           data=source_x,  # source_x
                                           labels=source_y)  # source_y
                mmd_d = self._to_mmd_layer(model=self.cvae_model,
                                           data=dest_x,  # dest_x
                                           labels=dest_y)  # dest_y
                mmd_loss = self.compute_mmd(mmd_s, mmd_d)
                return self.beta * mmd_loss

            self.cvae_optimizer = keras.optimizers.Adam(lr=self.lr)
            self.cvae_model.compile(optimizer=self.cvae_optimizer,
                                    loss=[cvae_kl_re_loss(self.mu, self.log_var), cvae_mmd_loss])

        def non_batch_loss(data, labels):
            def cvae_kl_re_loss(mu, log_var):
                def loss(y_true, y_pred):
                    kl_loss = 0.5 * K.mean(K.exp(log_var) + K.square(mu) - 1. - log_var, 1)
                    recon_loss = 0.5 * K.sum(K.square((y_true - y_pred)), axis=1)
                    return recon_loss + self.alpha * kl_loss
                return loss

            def cvae_mmd_loss(data, labels, sample_size=1000):
                def loss(y_true, y_pred):
                    train_data = K.variable(value=data)
                    train_labels = K.variable(value=labels)
                    n_samples = K.eval(K.shape(train_data)[0])

                    sample_indices = K.cast(K.round(K.random_uniform_variable(shape=tuple([sample_size]), low=0,
                                                                              high=n_samples - 1)), 'int32')

                    selected_samples = K.gather(train_data, sample_indices)
                    selected_labels = K.gather(train_labels, sample_indices)

                    bool_mask = K.equal(selected_labels, K.zeros(shape=K.shape(selected_labels)))
                    bool_mask = K.reshape(bool_mask, shape=(-1,))
                    source_x = tf.boolean_mask(selected_samples, bool_mask)
                    source_x = K.reshape(source_x, shape=(-1, K.get_variable_shape(source_x)[1]))
                    source_y = K.zeros(shape=(K.shape(source_x)[0], 1))

                    bool_mask = K.equal(selected_labels, K.ones(shape=K.shape(selected_labels)))
                    bool_mask = K.reshape(bool_mask, shape=(-1,))
                    dest_x = tf.boolean_mask(selected_samples, bool_mask)
                    dest_x = K.reshape(dest_x, shape=(-1, K.get_variable_shape(dest_x)[1]))
                    dest_y = K.ones(shape=(K.shape(dest_x)[0], 1))

                    mmd_s = self._to_mmd_layer(model=self.cvae_model,
                                               data=source_x,  # source_x
                                               labels=source_y)  # source_y
                    mmd_d = self._to_mmd_layer(model=self.cvae_model,
                                               data=dest_x,  # dest_x
                                               labels=dest_y)  # dest_y

                    mmd_loss = self.compute_mmd(mmd_s, mmd_d)
                    return self.beta * mmd_loss
                return loss

            self.cvae_optimizer = keras.optimizers.Adam(lr=self.lr)
            self.cvae_model.compile(optimizer=self.cvae_optimizer,
                                    loss=[cvae_kl_re_loss(self.mu, self.log_var),
                                          cvae_mmd_loss(data, labels)])

        if self.batch_mmd:
            batch_loss()
        else:
            non_batch_loss(data, labels)

    def to_latent(self, data, labels):
        """
            Map `data` in to the latent space. This function will feed data
            in encoder part of C-VAE and compute the latent space coordinates
            for each sample in data.

            # Parameters
                data: `~anndata.AnnData`
                    Annotated data matrix to be mapped to latent space. `data.X` has to be in shape [n_obs, n_vars].
                labels: numpy nd-array
                    `numpy nd-array` of labels to be fed as CVAE's condition array.

            # Returns
                latent: numpy nd-array
                    returns array containing latent space encoding of 'data'
        """
        latent = self.encoder_model.predict(x=[data, labels])[2]
        return latent

    def _to_mmd_layer(self, model, data, labels):
        """
            Map `data` in to the pn layer after latent layer. This function will feed data
            in encoder part of C-VAE and compute the latent space coordinates
            for each sample in data.

            # Parameters
                data: `~anndata.AnnData`
                    Annotated data matrix to be mapped to latent space. `data.X` has to be in shape [n_obs, n_vars].
                labels: numpy nd-array
                    `numpy nd-array` of labels to be fed as CVAE's condition array.

            # Returns
                latent: numpy nd-array
                    returns array containing latent space encoding of 'data'
        """

        mmd_model = Model(inputs=model.inputs, outputs=model.output[1])
        latent = mmd_model([data, labels, K.ones(shape=K.shape(labels))])
        return latent

    def to_mmd_layer(self, model, data, labels):
        """
            Map `data` in to the pn layer after latent layer. This function will feed data
            in encoder part of C-VAE and compute the latent space coordinates
            for each sample in data.

            # Parameters
                data: `~anndata.AnnData`
                    Annotated data matrix to be mapped to latent space. `data.X` has to be in shape [n_obs, n_vars].
                labels: numpy nd-array
                    `numpy nd-array` of labels to be fed as CVAE's condition array.

            # Returns
                latent: numpy nd-array
                    returns array containing latent space encoding of 'data'
        """

        mmd_model = Model(inputs=model.inputs, outputs=model.output[1])
        mmd_latent = mmd_model.predict([data, labels, np.ones(shape=labels.shape)])
        return mmd_latent

    def _reconstruct(self, data, labels, use_data=False):
        """
            Map back the latent space encoding via the decoder.

            # Parameters
                data: `~anndata.AnnData`
                    Annotated data matrix whether in latent space or primary space.

                labels: numpy nd-array
                    `numpy nd-array` of labels to be fed as CVAE's condition array.

                use_data: bool
                    this flag determines whether the `data` is already in latent space or not.
                    if `True`: The `data` is in latent space (`data.X` is in shape [n_obs, z_dim]).
                    if `False`: The `data` is not in latent space (`data.X` is in shape [n_obs, n_vars]).

            # Returns
                rec_data: 'numpy nd-array'
                    returns 'numpy nd-array` containing reconstructed 'data' in shape [n_obs, n_vars].
        """
        if use_data:
            latent = data
        else:
            latent = self.to_latent(data, labels)
        rec_data = self.decoder_model.predict([latent, labels])
        return rec_data

    def predict(self, data, labels):
        """
            Predicts the cell type provided by the user in stimulated condition.

            # Parameters
                data: `~anndata.AnnData`
                    Annotated data matrix whether in primary space.

                labels: numpy nd-array
                    `numpy nd-array` of labels to be fed as CVAE's condition array.

            # Returns
                stim_pred: numpy nd-array
                    `numpy nd-array` of predicted cells in primary space.

            # Example
            ```python
            import scanpy as sc
            import scgen
            train_data = sc.read("train_kang.h5ad")
            validation_data = sc.read("./data/validation.h5ad")
            network = scgen.CVAE(train_data=train_data, use_validation=True, validation_data=validation_data, model_path="./saved_models/", conditions={"ctrl": "control", "stim": "stimulated"})
            network.train(n_epochs=20)
            prediction = network.predict('CD4T', obs_key={"cell_type": ["CD8T", "NK"]})
            ```
        """
        if sparse.issparse(data.X):
            stim_pred = self._reconstruct(data.X.A, labels)
        else:
            stim_pred = self._reconstruct(data.X, labels)
        return stim_pred

    def restore_model(self):
        """
            restores model weights from `model_to_use`.

            # Parameters
                No parameters are needed.

            # Returns
                Nothing will be returned.

            # Example
            ```python
            import scanpy as sc
            import scgen
            train_data = sc.read("./data/train_kang.h5ad")
            validation_data = sc.read("./data/valiation.h5ad")
            network = scgen.CVAE(train_data=train_data, use_validation=True, validation_data=validation_data, model_path="./saved_models/", conditions={"ctrl": "control", "stim": "stimulated"})
            network.restore_model()
            ```
        """
        self.cvae_model = load_model(os.path.join(self.model_to_use, 'cvae.h5'), compile=False)
        self.encoder_model = load_model(os.path.join(self.model_to_use, 'encoder.h5'), compile=False)
        self.decoder_model = load_model(os.path.join(self.model_to_use, 'decoder.h5'), compile=False)
        self._loss_function()

    def train(self, train_data, use_validation=False, valid_data=None, n_epochs=25, batch_size=32, early_stop_limit=20,
              threshold=0.0025, initial_run=True,
              shuffle=True, verbose=2):  # TODO: Write minibatches for each source and destination
        """
            Trains the network `n_epochs` times with given `train_data`
            and validates the model using validation_data if it was given
            in the constructor function. This function is using `early stopping`
            technique to prevent overfitting.

            # Parameters
                n_epochs: int
                    number of epochs to iterate and optimize network weights
                early_stop_limit: int
                    number of consecutive epochs in which network loss is not going lower.
                    After this limit, the network will stop training.
                threshold: float
                    Threshold for difference between consecutive validation loss values
                    if the difference is upper than this `threshold`, this epoch will not
                    considered as an epoch in early stopping.
                full_training: bool
                    if `True`: Network will be trained with all batches of data in each epoch.
                    if `False`: Network will be trained with a random batch of data in each epoch.
                initial_run: bool
                    if `True`: The network will initiate training and log some useful initial messages.
                    if `False`: Network will resume the training using `restore_model` function in order
                        to restore last model which has been trained with some training dataset.


            # Returns
                Nothing will be returned

            # Example
            ```python
            import scanpy as sc
            import scgen
            train_data = sc.read(train_katrain_kang.h5ad           >>> validation_data = sc.read(valid_kang.h5ad)
            network = scgen.CVAE(train_data=train_data, use_validation=True, validation_data=validation_data, model_path="./saved_models/", conditions={"ctrl": "control", "stim": "stimulated"})
            network.train(n_epochs=20)
            ```
        """
        if initial_run:
            log.info("----Training----")
        train_labels, le = label_encoder(train_data)
        train_source = train_data[train_data.obs["condition"] == "control"]
        train_dest = train_data[train_data.obs["condition"] == "stimulated"]
        pseudo_labels = np.ones(shape=train_labels.shape)
        if not self.batch_mmd:
            self._loss_function(train_data.X.A, train_labels)

        if shuffle:
            train_data, train_labels = shuffle_data(train_data, train_labels)
        if use_validation and valid_data is None:
            raise Exception("valid_data is None but use_validation is True.")
        if use_validation:
            valid_labels, _ = label_encoder(valid_data)
        callbacks = [
            EarlyStopping(patience=early_stop_limit, monitor='loss', min_delta=threshold),
            CSVLogger(filename="./results/csv_logger.log")
        ]
        if use_validation:
            self.cvae_model.fit(
                x=[train_data.X, train_labels, pseudo_labels],
                y=[train_data.X, train_data.X],
                epochs=n_epochs,
                batch_size=batch_size,
                validation_data=(valid_data.X, valid_data.X),
                shuffle=shuffle,
                callbacks=callbacks,
                verbose=verbose)
        else:
            self.cvae_model.fit(
                x=[train_data.X, train_labels, pseudo_labels],
                y=[train_data.X, train_data.X],  # 2nd element does not matter! :)
                epochs=n_epochs,
                batch_size=batch_size,
                shuffle=shuffle,
                callbacks=callbacks,
                verbose=verbose)
        self.cvae_model.save(os.path.join("cvae.h5"), overwrite=True)
        self.encoder_model.save(os.path.join("encoder.h5"), overwrite=True)
        self.decoder_model.save(os.path.join("decoder.h5"), overwrite=True)
        log.info(f"Model saved in file: {self.model_to_use}. Training finished")
