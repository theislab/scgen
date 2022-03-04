import numpy as np
import torch
from scvi import REGISTRY_KEYS
from scvi._compat import Literal
from scvi.module.base import BaseModuleClass, LossRecorder, auto_move_data
from scvi.nn import Encoder
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl

from ._base_components import DecoderSCGEN


class SCGENVAE(BaseModuleClass):
    """
    Variational auto-encoder model.

    Parameters
    ----------
    n_input
        Number of input genes
    n_hidden
        Number of nodes per hidden layer
    n_latent
        Dimensionality of the latent space
    n_layers
        Number of hidden layers used for encoder and decoder NNs
    dropout_rate
        Dropout rate for neural networks
    use_layer_norm
        Whether to use layer norm in layers
    kl_weight
        Weight for kl divergence
    """

    def __init__(
        self,
        n_input: int,
        n_hidden: int = 800,
        n_latent: int = 10,
        n_layers: int = 2,
        dropout_rate: float = 0.1,
        log_variational: bool = False,
        latent_distribution: str = "normal",
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        kl_weight: float = 0.00005,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.latent_distribution = "normal"
        self.kl_weight = kl_weight

        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"

        self.z_encoder = Encoder(
            n_input,
            n_latent,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=latent_distribution,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            activation_fn=torch.nn.LeakyReLU,
        )

        n_input_decoder = n_latent
        self.decoder = DecoderSCGEN(
            n_input_decoder,
            n_input,
            n_layers=n_layers,
            n_hidden=n_hidden,
            activation_fn=torch.nn.LeakyReLU,
            dropout_rate=dropout_rate,
        )

    def _get_inference_input(self, tensors):
        x = tensors[REGISTRY_KEYS.X_KEY]
        input_dict = dict(
            x=x,
        )
        return input_dict

    def _get_generative_input(self, tensors, inference_outputs):
        z = inference_outputs["z"]
        input_dict = {
            "z": z,
        }
        return input_dict

    @auto_move_data
    def inference(self, x):
        """
        High level inference method.

        Runs the inference (encoder) model.
        """
        qz_m, qz_v, z = self.z_encoder(x)

        outputs = dict(z=z, qz_m=qz_m, qz_v=qz_v)
        return outputs

    @auto_move_data
    def generative(self, z):
        """Runs the generative model."""
        px = self.decoder(z)

        return dict(px=px)

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
    ):

        x = tensors[REGISTRY_KEYS.X_KEY]
        qz_m = inference_outputs["qz_m"]
        qz_v = inference_outputs["qz_v"]
        p = generative_outputs["px"]

        kld = kl(
            Normal(qz_m, torch.sqrt(qz_v)),
            Normal(0, 1),
        ).sum(dim=1)
        rl = self.get_reconstruction_loss(p, x)
        loss = (0.5 * rl + 0.5 * (kld * self.kl_weight)).mean()
        return LossRecorder(loss, rl, kld)

    @torch.no_grad()
    def sample(
        self,
        tensors,
        n_samples=1,
    ) -> np.ndarray:
        r"""
        Generate observation samples from the posterior predictive distribution.

        The posterior predictive distribution is written as :math:`p(\hat{x} \mid x)`.

        Parameters
        ----------
        tensors
            Tensors dict
        n_samples
            Number of required samples for each cell
        library_size
            Library size to scale scamples to

        Returns
        -------
        x_new : :py:class:`torch.Tensor`
            tensor with shape (n_cells, n_genes, n_samples)
        """
        inference_kwargs = dict(n_samples=n_samples)
        inference_outputs, generative_outputs, = self.forward(
            tensors,
            inference_kwargs=inference_kwargs,
            compute_loss=False,
        )
        px = Normal(generative_outputs["px"], 1).sample()
        return px.cpu().numpy()

    def get_reconstruction_loss(self, x, px) -> torch.Tensor:
        loss = ((x - px) ** 2).sum(dim=1)
        return loss
