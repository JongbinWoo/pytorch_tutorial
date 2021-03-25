#%%
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.batchnorm import BatchNorm2d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
def enc_block(in_channels, out_channels, *args, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, *args, **kwargs),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU()
    )
# %%
class Encoder(nn.Module):
    def __init__(self, hidden_dims, latent_dim, image_channel):
        super(Encoder, self).__init__()
        self.hidden_dims = [image_channel] + hidden_dims
        self.enc_blocks = nn.Sequential(
            *[enc_block(in_channels, out_channels, kernel_size=3, stride=2, padding=1) 
              for in_channels, out_channels in zip(self.hidden_dims, self.hidden_dims[1:])])

        self.fc_mu = nn.Linear(self.hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(self.hidden_dims[-1]*4, latent_dim)
    
    def forward(self, x):
        x = self.enc_blocks(x)
        x = torch.flatten(x, start_dim=1) ##
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return [mu, log_var]
# %%
def dec_block(in_channels, out_channels, *args, **kwargs):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, *args, **kwargs),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU()
    )
# %%
class Decoder(nn.Module):
    def __init__(self, hidden_dims, latent_dim, image_channel):
        super(Decoder, self).__init__()
        self.hidden_dims = hidden_dims + [image_channel]
        self.dec_blocks = nn.Sequential(
            *[dec_block(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
              for in_channels, out_channels in zip(self.hidden_dims, self.hidden_dims[1:])]
        )
        self.decoder_input = nn.Linear(latent_dim, self.hidden_dims[0]*4)
        self.final_layer = nn.Sequential(
            # dec_block(self.hidden_dims[-1], self.hidden_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Conv2d(self.hidden_dims[-1], out_channels=1, kernel_size=5, padding=0),
            nn.Tanh())

    def forward(self, x):
        x = self.decoder_input(x)
        x = x.view(-1, 256, 2, 2)
        x = self.dec_blocks(x)
        x = self.final_layer(x)
        return x
# %%
# from torchsummary import summary 
# encoder = Encoder([1, 32, 64, 128, 256], 128).to(device)
# summary(encoder, (1, 28, 28))

# a = torch.randn(1, 1, 28, 28, device=device)
# out = encoder.forward(a)
# decoder = Decoder([1, 32, 64, 128, 256][::-1], 128).to(device)
# result = decoder(out[0])
# %%
class VAE(nn.Module):
    def __init__(self, hidden_dims, latent_dim, image_channel):
        super(VAE, self).__init__()
        self.encoder = Encoder(hidden_dims, latent_dim, image_channel)
        self.decoder = Decoder(hidden_dims[::-1], latent_dim, image_channel)
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x = self.decoder(z)
        return x, mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
# %%
vae = VAE([32, 64, 128, 256], 128, 1).to(device)
result = vae(a)
# %%
