'''Credit: https://github.com/ermongroup/ddim'''
'''Undergoing work'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from a2c_ppo_acktr.utils import init
import os
import numpy as np
from torch.utils.data import RandomSampler, BatchSampler
from .trainer import Trainer
from .utils import EarlyStopping
import math

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Unflatten(nn.Module):
    def __init__(self, new_shape):
        super().__init__()
        self.new_shape = new_shape

    def forward(self, x):
        x_uf = x.view(-1, *self.new_shape)
        return x_uf

class NatureCNND(nn.Module):

    def __init__(self, input_channels, args):
        super().__init__()
        self.feature_size = args.feature_size
        self.hidden_size = self.feature_size
        self.downsample = not args.no_downsample
        self.input_channels = input_channels
        self.end_with_relu = args.end_with_relu
        self.args = args
        self.device = torch.device("cuda:" + str(args.cuda_id) if torch.cuda.is_available() else "cpu")
        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               nn.init.calculate_gain('relu'))
        self.flatten = Flatten()

        if self.downsample:
            self.final_conv_size = 32 * 7 * 7
            self.final_conv_shape = (32, 7, 7)
            self.main = nn.Sequential(
                init_(nn.Conv2d(input_channels, 32, 8, stride=4)),
                nn.ReLU(),
                init_(nn.Conv2d(32, 64, 4, stride=2)),
                nn.ReLU(),
                init_(nn.Conv2d(64, 32, 3, stride=1)),
                nn.ReLU(),
                Flatten(),
                init_(nn.Linear(self.final_conv_size, self.feature_size)),
                #nn.ReLU()
            )
        else:

            self.temb = nn.Module()
            self.temb.dense = nn.ModuleList([
                torch.nn.Linear(32,
                                128),
                torch.nn.Linear(128,
                                128),
            ])
            self.temb_proj1 = torch.nn.Linear(128,
                                         64)
            self.temb_proj2 = torch.nn.Linear(128,
                                         128)
            self.temb_proj3 = torch.nn.Linear(128,
                                         64)                                       
            self.final_conv_size = 64 * 9 * 6
            self.final_conv_shape = (64, 9, 6)
            self.main = nn.Sequential(
                init_(nn.Conv2d(input_channels, 32, 8, stride=4)),
                nn.ReLU(),
                init_(nn.Conv2d(32, 64, 4, stride=2)),
                nn.ReLU(),
                init_(nn.Conv2d(64, 128, 4, stride=2)),
                nn.ReLU(),
                init_(nn.Conv2d(128, 64, 3, stride=1)),
                nn.ReLU(),
                Flatten(),
                init_(nn.Linear(self.final_conv_size, self.feature_size)),
                #nn.ReLU()
            )
        self.train()

    @property
    def local_layer_depth(self):
        return self.main[4].out_channels

    def forward(self, inputs, t=None, fmaps=False):
        if t == None:
            # temb=torch.zeros(inputs.size(0), 128).to(self.device)
            t = torch.arange(0, 1000, 1000/inputs.size(0)).to(torch.long).to(self.device)
            
        temb = get_timestep_embedding(t, 32)
        temb = self.temb.dense[0](temb)
        temb = F.relu(temb)
        temb = self.temb.dense[1](temb)

        hs = [self.main[:2](inputs)]
        hs.append(self.main[2:4](hs[-1])+ self.temb_proj1(F.relu(temb))[:, :, None, None])
        hs.append(self.main[4:6](hs[-1])+ self.temb_proj2(F.relu(temb))[:, :, None, None])
        hs.append(self.main[6:8](hs[-1])+ self.temb_proj3(F.relu(temb))[:, :, None, None])

        out = self.main[8:](hs[-1])
        if self.end_with_relu:
            assert self.args.method != "ddim", "can't end with relu and use ddim!"
            out = F.relu(out)
        if fmaps:
            return {
                'f5': hs,
                'temb': temb,
                'out': out
            }
        return out

class Decoder(nn.Module):
    def __init__(self, feature_size, final_conv_size, final_conv_shape, num_input_channels, encoder_type="Nature"):
        super().__init__()
        self.feature_size = feature_size
        self.final_conv_size = final_conv_size
        self.final_conv_shape = final_conv_shape
        self.num_input_channels = num_input_channels
        # self.fc =
        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               nn.init.calculate_gain('relu'))
        if encoder_type == "Nature":

            self.temb_proj1 = torch.nn.Linear(128,
                                         128)
            self.temb_proj2 = torch.nn.Linear(128,
                                         256)
            self.temb_proj3 = torch.nn.Linear(128,
                                         128)

            self.main = nn.Sequential(
                nn.Linear(in_features=self.feature_size,
                          out_features=self.final_conv_size),
                nn.ReLU(),
                Unflatten(self.final_conv_shape),

                init_(nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0)),
                nn.ReLU(),
                init_(nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=4, stride=2, padding=0)),
                nn.ReLU(),
                init_(nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=4, stride=2, padding=0,
                                         output_padding=1)),
                nn.ReLU(),
                init_(nn.ConvTranspose2d(in_channels=32, out_channels=num_input_channels,
                                         kernel_size=8, stride=4, output_padding=(2, 0))),
                nn.Sigmoid()
            )

    def forward(self, f, temb, hs):
        im = self.main[:3](f)
        im = self.main[3:5](torch.cat([im, hs.pop()], dim=1)+ self.temb_proj1(F.relu(temb))[:, :, None, None])
        im = self.main[5:7](torch.cat([im, hs.pop()], dim=1)+ self.temb_proj2(F.relu(temb))[:, :, None, None])
        im = self.main[7:9](torch.cat([im, hs.pop()], dim=1)+ self.temb_proj3(F.relu(temb))[:, :, None, None])
        im = self.main[9:](im)
        return im

def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

def logit_transform(image, lam=1e-6):
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)

def data_transform(config, X):
    X = 2 * X - 1.0
    return X

def inverse_data_transform(config, X):
    X = (X + 1.0) / 2.0
    return torch.clamp(X, 0.0, 1.0)

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def noise_estimation_loss(model,
                          x0: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor, keepdim=False):
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(x, t.float())
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)


class DDIM(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.feature_size = self.encoder.feature_size
        self.final_conv_size = self.encoder.final_conv_size
        self.final_conv_shape = self.encoder.final_conv_shape
        self.input_channels = self.encoder.input_channels
#         self.mu_fc = nn.Linear(in_features=self.feature_size,
#                                    out_features=self.feature_size)

        self.decoder = Decoder(feature_size=self.feature_size,
                               final_conv_size=self.final_conv_size,
                               final_conv_shape=self.final_conv_shape,
                               num_input_channels=self.input_channels)

    def forward(self, x, t):
        x = self.encoder(x, t, fmaps=True)
        x = self.decoder(x['out'], x['temb'], x['f5'])
        return x


class DDIMLoss(object):
    def __init__(self, beta=1.0):
        self.beta = beta

    def __call__(self, output, e, keepdim=False):
        if keepdim:
            return (e - output).square().sum(dim=(1, 2, 3))
        else:
            return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)


class DDIMTrainer(Trainer):
    # TODO: Make it work for all modes, right now only it defaults to pcl.
    def __init__(self, encoder, config, device=torch.device('cpu'), wandb=None):
        super().__init__(encoder, wandb, device)
        self.config = config
        self.patience = self.config["patience"]
        self.DDIM = DDIM(encoder).to(device)
        self.epochs = config['epochs']
        self.batch_size = config['batch_size']
        self.device = device
        self.optimizer = torch.optim.Adam(list(self.DDIM.parameters()),
                                          lr=config['lr'], eps=1e-5)
        self.loss_fn = DDIMLoss(beta=self.config["beta"])
        self.early_stopper = EarlyStopping(patience=self.patience, verbose=False, wandb=self.wandb, name="encoder")

        self.model_var_type = 'fixedlarge'
        betas = get_beta_schedule(
            beta_schedule='linear',
            beta_start=0.0001,
            beta_end=0.02,
            num_diffusion_timesteps=1000,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def generate_batch(self, episodes):
        total_steps = sum([len(e) for e in episodes])
        print('Total Steps: {}'.format(total_steps))
        # Episode sampler
        # Sample `num_samples` episodes then batchify them with `self.batch_size` episodes per batch
        sampler = BatchSampler(RandomSampler(range(len(episodes)),
                                             replacement=True, num_samples=total_steps),
                               self.batch_size, drop_last=True)
        for indices in sampler:
            episodes_batch = [episodes[x] for x in indices]
            x_t, x_tprev, x_that, ts, thats = [], [], [], [], []
            for episode in episodes_batch:
                # Get one sample from this episode
                t, t_hat = 0, 0
                t, t_hat = np.random.randint(0, len(episode)), np.random.randint(0, len(episode))
                x_t.append(episode[t])
            yield torch.stack(x_t).float().to(self.device) / 255.

    def do_one_epoch(self, epoch, episodes):
        mode = "train" if self.DDIM.training else "val"
        epoch_loss, accuracy, steps = 0., 0., 0
        data_generator = self.generate_batch(episodes)
        for x_t in data_generator:
            with torch.set_grad_enabled(mode == 'train'):
                x_0 = data_transform(self.config, x_t)
                n = x_t.size(0)
                e = torch.randn_like(x_0)
                b = self.betas
                t = torch.randint(
                    low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                ).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
                x = x_0 * a.sqrt() + e * (1.0 - a).sqrt()
                output = self.DDIM(x, t)

                loss = self.loss_fn(output, e)

            if mode == "train":
                self.optimizer.zero_grad()
                try:
                    torch.nn.utils.clip_grad_norm_(
                        DDIM.parameters(), 1.0
                    )
                except Exception:
                    pass
                loss.backward()
                self.optimizer.step()

            epoch_loss += loss.detach().item()
            steps += 1
        self.log_results(epoch, epoch_loss / steps, prefix=mode)
        if mode == "val":
            self.early_stopper(-epoch_loss / steps, self.encoder)

    #             xim = x_hat.detach().cpu().numpy()[0].transpose(1,2,0)
    #             self.wandb.log({"example_reconstruction": [self.wandb.Image(xim, caption="")]})

    def train(self, tr_eps, val_eps):
        for e in range(self.epochs):
            self.DDIM.train()
            self.do_one_epoch(e, tr_eps)

            self.DDIM.eval()
            self.do_one_epoch(e, val_eps)

            if self.early_stopper.early_stop:
                break
        torch.save(self.encoder.state_dict(), os.path.join(self.wandb.run.dir, self.config['env_name'] + '.pt'))

    def log_results(self, epoch_idx, epoch_loss, prefix=""):
        print("{} Epoch: {}, Epoch Loss: {}".format(prefix.capitalize(), epoch_idx, epoch_loss))
        self.wandb.log({prefix + '_loss': epoch_loss})