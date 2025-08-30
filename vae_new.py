import torch
import torch.nn as nn
from typing import List

from pythae.models.nn import BaseEncoder, BaseDecoder
from pythae.models.base.base_utils import ModelOutput
from pythae.models.vae import VAE, VAEConfig
from pythae.trainers import BaseTrainerConfig
from pythae.pipelines.training import TrainingPipeline


from experiment import AutoencoderExperiment

class MLPEncoder(BaseEncoder):
    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int):
        super().__init__()
        self.input_dim = input_dim
        dims = [input_dim] + hidden_dims
        layers = []
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_d, out_d))
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        h = self.net(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return ModelOutput(embedding=mu, log_covariance=logvar)

class MLPDecoder(BaseDecoder):
    def __init__(self, latent_dim: int, hidden_dims: List[int], output_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        dims = [latent_dim] + hidden_dims
        layers = []
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_d, out_d))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> ModelOutput:
        recon = self.net(z)
        return ModelOutput(reconstruction=recon)
    
    
class VAEExperiment(AutoencoderExperiment):


    def __init__(self, enc_layer_desc=(64,), pca_dims=0, whiten_input=True, 
                 n_train_samples=0, binary_input=False, use_pca_cache=True,
                 dropout_info=None, **kwargs):
        self.enc_layer_desc = enc_layer_desc
        self.pca_dims = pca_dims
        self.whiten_input = whiten_input
        self.n_train_samples = n_train_samples
        self.binary_input = binary_input
        self.use_pca_cache = use_pca_cache
        self.dropout_info = dropout_info
        super().__init__(**kwargs)

def train_custom_vae(
    x_train: torch.Tensor,
    encoder_dims: List[int],
    latent_dim: int,
    epochs: int = 10,
    lr: float = 1e-3,
    batch_size: int = 64
):
    input_dim = x_train.shape[-1]
    encoder = MLPEncoder(input_dim, encoder_dims, latent_dim)
    decoder = MLPDecoder(latent_dim, list(reversed(encoder_dims)), input_dim)

    config = VAEConfig(input_dim=(input_dim,), latent_dim=latent_dim)
    model = VAE(model_config=config, encoder=encoder, decoder=decoder)

    trainer_config = BaseTrainerConfig(
        num_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size
    )
    pipeline = TrainingPipeline(model=model, training_config=trainer_config)
    pipeline(train_data=x_train, eval_data=x_train)

    return model

def example_mnist():
    from tests import load_mnist

# Example usage
if __name__ == "__main__":
    data = torch.randn(1000, 784)  # Example: flattened MNIST
    model = train_custom_vae(
        x_train=data,
        encoder_dims=[512, 256],
        latent_dim=20,
        epochs=5,
        batch_size=128
    )
