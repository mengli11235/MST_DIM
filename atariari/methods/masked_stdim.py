import random
from re import M

import torch
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.layers import trunc_normal_
from torch.utils.data import RandomSampler, BatchSampler
from .utils import calculate_accuracy, Cutout
from .trainer import Trainer
from .utils import EarlyStopping
from torchvision import transforms
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from a2c_ppo_acktr.utils import init
import matplotlib.pyplot as plt
import cv2

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

class NatureCNNMask(nn.Module):

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
            if self.args.use_mask_parameters:
                self.mask_token = nn.Parameter(torch.zeros(1, 1, 210, 160))
                trunc_normal_(self.mask_token, mean=0., std=.02)
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

    def forward(self, x, fmaps=False, mask=None):
        if mask is not None:
            if self.args.use_mask_parameters:
                B, L, H, W = x.shape

                mask_token = self.mask_token.expand(B, L, H, W)
                w = mask.unsqueeze(1).type_as(mask_token)
                x = x * (1 - w) + mask_token * w
            else:
                x = x * (1-mask.unsqueeze(1))
        
        x = self.main[:2](x)
        f5 = self.main[2:6](x)
        f7 = self.main[6:8](f5)

        out = self.main[8:](f7)
        if self.end_with_relu:
            assert self.args.method != "vae", "can't end with relu and use vae!"
            out = F.relu(out)
        if fmaps:
            return {
                'f5': f5.permute(0, 2, 3, 1),
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
            self.main = nn.Sequential(
                nn.Linear(in_features=self.feature_size,
                          out_features=self.final_conv_size),
                nn.ReLU(),
                Unflatten(self.final_conv_shape),

                init_(nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0)),
                nn.ReLU(),
                init_(nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=0)),
                nn.ReLU(),
                init_(nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=0,
                                         output_padding=1)),
                nn.ReLU(),
                init_(nn.ConvTranspose2d(in_channels=32, out_channels=num_input_channels,
                                         kernel_size=8, stride=4, output_padding=(2, 0))),
                nn.Sigmoid()
            )

    def forward(self, f):
        im = self.main(f)
        return im

class MaskGenerator:
    def __init__(self, input_size=(210, 160), mask_ratio=0.4, mask_type=['', ''], num_mask_patch=10):
        self.input_size = input_size

        self.mask_ratio = mask_ratio
        self.mask_type = mask_type
        self.num_mask_patch = num_mask_patch
        
        self.token_count = self.input_size[0]*self.input_size[1]
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

        if 'local' in self.mask_type:
            self.input_size = (self.input_size[0]//self.num_mask_patch, self.input_size[1]//self.num_mask_patch)
            self.token_count = self.token_count//(self.num_mask_patch**2)
            self.mask_count = self.mask_count//(self.num_mask_patch**2)

    def __call__(self):
        mask = np.zeros(self.token_count, dtype=int)

        if 'square' in self.mask_type:
            start_index = (self.token_count - self.mask_count)//2
            mask[start_index:start_index + self.mask_count] = 1

        else:
            mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
            mask[mask_idx] = 1
        
        mask = mask.reshape((self.input_size[0], self.input_size[1]))

        if 'local' in self.mask_type:
            mask = mask.repeat(self.num_mask_patch, axis=0).repeat(self.num_mask_patch, axis=1)
        return mask

class Classifier(nn.Module):
    def __init__(self, num_inputs1, num_inputs2):
        super().__init__()
        self.network = nn.Bilinear(num_inputs1, num_inputs2, 1)

    def forward(self, x1, x2):
        return self.network(x1, x2)


class MaskedTrainer(Trainer):
    def __init__(self, encoder, config, device=torch.device('cpu'), wandb=None):
        super().__init__(encoder, wandb, device)
        self.config = config
        self.patience = self.config["patience"]
        self.classifier1 = nn.Linear(self.encoder.hidden_size, self.encoder.local_layer_depth).to(device)  # x1 = global, x2=patch, n_channels = 32
        self.classifier2 = nn.Linear(self.encoder.local_layer_depth, self.encoder.local_layer_depth).to(device)
        self.epochs = config['epochs']
        self.batch_size = config['batch_size']
        self.device = device
        self.early_stopper = EarlyStopping(patience=self.patience, verbose=False, wandb=self.wandb, name="encoder")
        self.transform = transforms.Compose([Cutout(n_holes=1, length=80)])
        mask_type = []
        if config['pretrain_local']:
            mask_type.append('local')
        self.mask_generator = MaskGenerator(mask_ratio=config['mask_ratio'], mask_type=mask_type)

        self.optimizer = torch.optim.Adam(list(self.classifier1.parameters()) + list(self.encoder.parameters()) +
                                          list(self.classifier2.parameters()),
                                          lr=config['lr'], eps=1e-5)

        self.resize_cropper = T.RandomResizedCrop(size=(210, 160), scale=(0.35, 1.0))
        self.rotater = T.RandomRotation(degrees=(0, 180))
        self.hflipper = T.RandomHorizontalFlip(p=0.5)

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
            x_t, x_tprev, x_that, ts, thats, masks = [], [], [], [], [], []
            for episode in episodes_batch:
                # Get one sample from this episode
                t, t_hat = 0, 0
                t, t_hat = np.random.randint(0, len(episode)), np.random.randint(0, len(episode))
                mask = self.mask_generator()
                masks.append(mask)

                x_t.append(episode[t])
                x_tprev.append(episode[t-1])
                ts.append([t])
            yield torch.stack(x_t).float().to(self.device) / 255., torch.stack(x_tprev).float().to(self.device) / 255., torch.Tensor(np.array(masks)).float().to(self.device)

    def do_one_epoch(self, epoch, episodes):
        mode = "train" if self.encoder.training and self.classifier1.training else "val"
        epoch_loss, accuracy, steps = 0., 0., 0
        accuracy1, accuracy2 = 0., 0.
        epoch_loss1, epoch_loss2 = 0., 0.
        data_generator = self.generate_batch(episodes)

        for x_t, x_tprev, masks in data_generator:

            if not self.config["pretrain_masks"]:
                masks = None

            f_t_maps, f_t_prev_maps = self.encoder(x_t, fmaps=True, mask=masks), self.encoder(x_tprev, fmaps=True)

            # Loss 1: Global at time t, f5 patches at time t-1
            f_t, f_t_prev = f_t_maps['out'], f_t_prev_maps['f5']
            sy = f_t_prev.size(1)
            sx = f_t_prev.size(2)
            loss = 0.
            N = f_t.size(0)
            loss1 = 0.
            for y in range(sy):
                for x in range(sx):
                    predictions = self.classifier1(f_t)
                    positive = f_t_prev[:, y, x, :]
                    logits = torch.matmul(predictions, positive.t())
                    step_loss = F.cross_entropy(logits, torch.arange(N).to(self.device))
                    loss1 += step_loss
            loss1 = loss1 / (sx * sy)

            ## Loss 2: f5 patches at time t, with f5 patches at time t-1
            f_t = f_t_maps['f5']
            loss2 = 0.
            for y in range(sy):
                for x in range(sx):
                    predictions = self.classifier2(f_t[:, y, x, :])
                    positive = f_t_prev[:, y, x, :]
                    logits = torch.matmul(predictions, positive.t())
                    step_loss = F.cross_entropy(logits, torch.arange(N).to(self.device),)
                    loss2 += step_loss
            loss2 = loss2 / (sx * sy)
            loss = loss + loss1 + loss2

            if mode == "train":
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            epoch_loss += loss.detach().item()
            epoch_loss1 += loss1.detach().item()
            epoch_loss2 += loss2.detach().item()
            steps += 1
        self.log_results(epoch, epoch_loss1 / steps, epoch_loss2 / steps, epoch_loss / steps, prefix=mode)
        if mode == "val":
            self.early_stopper(-epoch_loss / steps, self.encoder)

    def train(self, tr_eps, val_eps):
        # TODO: Make it work for all modes, right now only it defaults to pcl.
        for e in range(self.epochs):
            self.encoder.train(),self.classifier1.train(), self.classifier2.train()
            self.do_one_epoch(e, tr_eps)

            self.encoder.eval(),self.classifier1.eval(), self.classifier2.eval()
            self.do_one_epoch(e, val_eps)

            if self.early_stopper.early_stop:
                break
        torch.save(self.encoder.state_dict(), os.path.join(self.wandb.run.dir, self.config['env_name'] + '.pt'))

    def log_results(self, epoch_idx, epoch_loss1, epoch_loss2, epoch_loss, prefix=""):
        print("{} Epoch: {}, Epoch Loss: {}, {}".format(prefix.capitalize(), epoch_idx, epoch_loss,
                                                                     prefix.capitalize()))
        self.wandb.log({prefix + '_loss': epoch_loss,
                        prefix + '_loss1': epoch_loss1,
                        prefix + '_loss2': epoch_loss2}, step=epoch_idx, commit=False)
