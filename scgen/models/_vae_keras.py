import logging
import os

import anndata
import keras
import numpy
from keras import backend as K, Model
from keras.callbacks import CSVLogger, EarlyStopping
from keras.layers import Input, Dense, BatchNormalization, LeakyReLU, Dropout, Lambda
from keras.models import load_model
from scipy import sparse

from reptrvae.utils import balancer, extractor, shuffle_data, remove_sparsity

log = logging.getLogger(__file__)


class scGen:
    """
        VAE with Arithmetic vector Network class. This class contains the implementation of Variational
        Auto-encoder network with Vector Arithmetics.

        Parameters
        ----------
        kwargs:
            :key `validation_data` : AnnData
                must be fed if `use_validation` is true.
            :key dropout_rate: float
                    dropout rate
            :key learning_rate: float
                learning rate of optimization algorithm
            :key model_path: basestring
                path to save the model after training

        x_dimension: integer
            number of gene expression space dimensions.

        z_dimension: integer
            number of latent space dimensions.

        See also
        --------
        CVAE from scgen.models._cvae : Conditional VAE implementation.

    """

    def __init__(self, x_dimension, z_dimension=100, **kwargs):
        self.x_dim = x_dimension
        self.z_dim = z_dimension

        self.learning_rate = kwargs.get("learning_rate", 0.001)
        self.dropout_rate = kwargs.get("dropout_rate", 0.2)
        self.model_to_use = kwargs.get("model_path", "./models/")
        self.alpha = kwargs.get("alpha", 0.00005)

        self.x = Input(shape=(x_dimension,), name="input")
        self.z = Input(shape=(z_dimension,), name="latent")

        self.init_w = keras.initializers.glorot_normal()
        self.aux_models = {}

        self._create_network()
        self._loss_function()
        self.vae_model.summary()

    def _encoder(self):
        """
            Constructs the encoder sub-network of VAE. This function implements the
            encoder part of Variational Auto-encoder. It will transform primary
            data in the `n_vars` dimension-space to the `z_dimension` latent space.

            Parameters
            ----------
            No parameters are needed.

            Returns
            -------
            mean: Tensor
                A dense layer consists of means of gaussian distributions of latent space dimensions.
            log_var: Tensor
                A dense layer consists of log transformed variances of gaussian distributions of latent space dimensions.
        """
        h = Dense(800, kernel_initializer=self.init_w, use_bias=False)(self.x)
        h = BatchNormalization(axis=1)(h)
        h = LeakyReLU()(h)
        h = Dropout(self.dropout_rate)(h)
        h = Dense(800, kernel_initializer=self.init_w, use_bias=False)(h)
        h = BatchNormalization(axis=1)(h)
        h = LeakyReLU()(h)
        h = Dropout(self.dropout_rate)(h)

        mean = Dense(self.z_dim, kernel_initializer=self.init_w)(h)
        log_var = Dense(self.z_dim, kernel_initializer=self.init_w)(h)
        z = Lambda(self._sample_z, output_shape=(self.z_dim,), name="Z")([mean, log_var])

        self.encoder_model = Model(inputs=self.x, outputs=z, name="encoder")
        return mean, log_var

    def _decoder(self):
        """
            Constructs the decoder sub-network of VAE. This function implements the
            decoder part of Variational Auto-encoder. It will transform constructed
            latent space to the previous space of data with n_dimensions = n_vars.

            Parameters
            ----------
            No parameters are needed.

            Returns
            -------
            h: Tensor
                A Tensor for last dense layer with the shape of [n_vars, ] to reconstruct data.

        """
        h = Dense(800, kernel_initializer=self.init_w, use_bias=False)(self.z)
        h = BatchNormalization(axis=1)(h)
        h_mmd = LeakyReLU()(h)
        h = Dropout(self.dropout_rate)(h_mmd)
        h = Dense(800, kernel_initializer=self.init_w, use_bias=False)(h)
        h = BatchNormalization(axis=1)(h)
        h = LeakyReLU()(h)
        h = Dropout(self.dropout_rate)(h)
        h = Dense(self.x_dim, kernel_initializer=self.init_w, use_bias=True)(h)

        self.decoder_model = Model(inputs=self.z, outputs=h, name="decoder")
        self.decoder_mmd_model = Model(self.z, h_mmd, name="decoder_mmd")
        return h

    @staticmethod
    def _sample_z(args):
        """
            Samples from standard Normal distribution with shape [size, z_dim] and
            applies re-parametrization trick. It is actually sampling from latent
            space distributions with N(mu, var) computed in `_encoder` function.

            Parameters
            ----------
            No parameters are needed.

            Returns
            -------
            The computed Tensor of samples with shape [size, z_dim].
        """
        mu, log_var = args
        batch_size = K.shape(mu)[0]
        z_dim = K.shape(mu)[1]
        eps = K.random_normal(shape=[batch_size, z_dim])
        return mu + K.exp(log_var / 2) * eps

    def _create_network(self):
        """
            Constructs the whole VAE network. It is step-by-step constructing the VAE
            network. First, It will construct the encoder part and get mu, log_var of
            latent space. Second, It will sample from the latent space to feed the
            decoder part in next step. Finally, It will reconstruct the data by
            constructing decoder part of VAE.

            Parameters
            ----------
            No parameters are needed.

            Returns
            -------
            Nothing will be returned.
        """
        self.mu, self.log_var = self._encoder()

        self.x_hat = self._decoder()
        self.vae_model = Model(inputs=self.x, outputs=self.decoder_model(self.encoder_model(self.x)), name="VAE")
        self.aux_models['mmd'] = Model(self.x, self.decoder_mmd_model(self.encoder_model(self.x)), name="MMD")

    def _loss_function(self):
        """
            Defines the loss function of VAE network after constructing the whole
            network. This will define the KL Divergence and Reconstruction loss for
            VAE and also defines the Optimization algorithm for network. The VAE Loss
            will be weighted sum of reconstruction loss and KL Divergence loss.

            Parameters
            ----------
            No parameters are needed.

            Returns
            -------
            Nothing will be returned.
        """

        def vae_loss(y_true, y_pred):
            return K.mean(recon_loss(y_true, y_pred) + kl_loss(y_true, y_pred))

        def kl_loss(y_true, y_pred):
            return self.alpha * (0.5 * K.sum(K.exp(self.log_var) + K.square(self.mu) - 1. - self.log_var, axis=1))

        def recon_loss(y_true, y_pred):
            return 0.5 * K.sum(K.square((y_true - y_pred)), axis=1)

        self.vae_optimizer = keras.optimizers.Adam(lr=self.learning_rate)
        self.vae_model.compile(optimizer=self.vae_optimizer, loss=vae_loss, metrics=[kl_loss, recon_loss])

    def to_latent(self, adata):
        adata = remove_sparsity(adata)

        latent = self.encoder_model.predict(adata.X)
        latent_adata = anndata.AnnData(X=latent)
        latent_adata.obs = adata.obs.copy(deep=True)

        return latent_adata

    def to_mmd_layer(self, adata):
        adata = remove_sparsity(adata)

        mmd_latent = self.aux_models['mmd'].predict(adata.X)
        mmd_adata = anndata.AnnData(X=mmd_latent)
        mmd_adata.obs = adata.obs.copy(deep=True)

        return mmd_adata

    def _avg_vector(self, data):
        """
            Computes the average of points which computed from mapping `data`
            to encoder part of VAE.

            Parameters
            ----------
            data:  numpy nd-array
                Numpy nd-array matrix to be mapped to latent space. Note that `data.X` has to be in shape [n_obs, n_vars].

            Returns
            -------
                The average of latent space mapping in numpy nd-array.

        """
        latent = self.to_latent(data)
        latent_avg = numpy.average(latent, axis=0)
        return latent_avg

    def reconstruct(self, data):
        """
            Map back the latent space encoding via the decoder.

            Parameters
            ----------
            data: `~anndata.AnnData`
                Annotated data matrix whether in latent space or gene expression space.

            use_data: bool
                This flag determines whether the `data` is already in latent space or not.
                if `True`: The `data` is in latent space (`data.X` is in shape [n_obs, z_dim]).
                if `False`: The `data` is not in latent space (`data.X` is in shape [n_obs, n_vars]).

            Returns
            -------
            rec_data: 'numpy nd-array'
                Returns 'numpy nd-array` containing reconstructed 'data' in shape [n_obs, n_vars].
        """
        rec_data = self.decoder_model.predict(x=data)
        return rec_data

    def predict(self, adata, conditions, cell_type_key, condition_key, adata_to_predict=None, celltype_to_predict=None,
                obs_key="all"):
        if obs_key == "all":
            ctrl_x = adata[adata.obs["condition"] == conditions["ctrl"], :]
            stim_x = adata[adata.obs["condition"] == conditions["stim"], :]
            ctrl_x = balancer(ctrl_x, cell_type_key=cell_type_key, condition_key=condition_key)
            stim_x = balancer(stim_x, cell_type_key=cell_type_key, condition_key=condition_key)
        else:
            key = list(obs_key.keys())[0]
            values = obs_key[key]
            subset = adata[adata.obs[key].isin(values)]
            ctrl_x = subset[subset.obs["condition"] == conditions["ctrl"], :]
            stim_x = subset[subset.obs["condition"] == conditions["stim"], :]
            if len(values) > 1:
                ctrl_x = balancer(ctrl_x, cell_type_key=cell_type_key, condition_key=condition_key)
                stim_x = balancer(stim_x, cell_type_key=cell_type_key, condition_key=condition_key)
        if celltype_to_predict is not None and adata_to_predict is not None:
            raise Exception("Please provide either a cell type or adata not both!")
        if celltype_to_predict is None and adata_to_predict is None:
            raise Exception("Please provide a cell type name or adata for your unperturbed cells")
        if celltype_to_predict is not None:
            ctrl_pred = extractor(adata, celltype_to_predict, conditions, cell_type_key, condition_key)[1]
        else:
            ctrl_pred = adata_to_predict
        eq = min(ctrl_x.X.shape[0], stim_x.X.shape[0])
        cd_ind = numpy.random.choice(range(ctrl_x.shape[0]), size=eq, replace=False)
        stim_ind = numpy.random.choice(range(stim_x.shape[0]), size=eq, replace=False)
        if sparse.issparse(ctrl_x.X) and sparse.issparse(stim_x.X):
            latent_ctrl = self._avg_vector(ctrl_x.X.A[cd_ind, :])
            latent_sim = self._avg_vector(stim_x.X.A[stim_ind, :])
        else:
            latent_ctrl = self._avg_vector(ctrl_x.X[cd_ind, :])
            latent_sim = self._avg_vector(stim_x.X[stim_ind, :])
        delta = latent_sim - latent_ctrl
        if sparse.issparse(ctrl_pred.X):
            latent_cd = self.to_latent(ctrl_pred.X.A)
        else:
            latent_cd = self.to_latent(ctrl_pred.X)
        stim_pred = delta + latent_cd
        predicted_cells = self.reconstruct(stim_pred)
        return predicted_cells, delta

    def restore_model(self):
        self.vae_model = load_model(os.path.join(self.model_to_use, 'vae.h5'), compile=False)
        self.encoder_model = load_model(os.path.join(self.model_to_use, 'encoder.h5'), compile=False)
        self.decoder_model = load_model(os.path.join(self.model_to_use, 'decoder.h5'), compile=False)
        self._loss_function()

    def save_model(self):
        os.makedirs(self.model_to_use, exist_ok=True)
        self.vae_model.save(os.path.join(self.model_to_use, "vae.h5"), overwrite=True)
        self.encoder_model.save(os.path.join(self.model_to_use, "encoder.h5"), overwrite=True)
        self.decoder_model.save(os.path.join(self.model_to_use, "decoder.h5"), overwrite=True)
        log.info(f"Models are saved in file: {self.model_to_use}. Training finished")

    def train(self, train_data, validation_data=None,
              n_epochs=25,
              batch_size=32,
              early_stop_limit=20,
              threshold=0.0025,
              initial_run=True,
              shuffle=True,
              verbose=1,
              save=True,
              checkpoint=50,
              **kwargs):
        if initial_run:
            log.info("----Training----")
        if shuffle:
            train_data = shuffle_data(train_data)

        train_data = remove_sparsity(train_data)

        callbacks = [
            EarlyStopping(patience=early_stop_limit, monitor='val_loss', min_delta=threshold),
            CSVLogger(filename="./csv_logger.log")
        ]
        if validation_data is not None:
            result = self.vae_model.fit(x=train_data.X,
                                        y=train_data.X,
                                        epochs=n_epochs,
                                        batch_size=batch_size,
                                        validation_data=(validation_data.X, validation_data.X),
                                        shuffle=shuffle,
                                        callbacks=callbacks,
                                        verbose=verbose)
        else:
            result = self.vae_model.fit(x=train_data.X,
                                        y=train_data.X,
                                        validation_split=0.2,
                                        epochs=n_epochs,
                                        batch_size=batch_size,
                                        shuffle=shuffle,
                                        callbacks=callbacks,
                                        verbose=verbose)

        if save is True:
            self.save_model()
        return result
