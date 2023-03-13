import time
from collections import deque
from itertools import chain

import numpy as np
import torch

from atariari.methods.dim_baseline import DIMTrainer
from atariari.methods.swin_mlp import SwinMLP
from atariari.methods.global_infonce_stdim import GlobalInfoNCESpatioTemporalTrainer
from atariari.methods.global_local_infonce import GlobalLocalInfoNCESpatioTemporalTrainer
from atariari.methods.jsd_stdim import SpatioTemporalTrainer
from atariari.methods.utils import get_argparser
from atariari.methods.encoders import NatureCNN, ImpalaCNN
from atariari.methods.ddim import NatureCNND

from atariari.methods.cpc import CPCTrainer
from atariari.methods.vae import VAETrainer
from atariari.methods.ddim import DDIMTrainer
from atariari.methods.masked_stdim import MaskedTrainer, NatureCNNMask
from atariari.methods.no_action_feedforward_predictor import NaFFPredictorTrainer
from atariari.methods.stdim import InfoNCESpatioTemporalTrainer
import wandb
from atariari.benchmark.episodes import get_episodes


def train_encoder(args):
    device = torch.device("cuda:" + str(args.cuda_id) if torch.cuda.is_available() else "cpu")
    tr_eps, val_eps = get_episodes(steps=args.pretraining_steps,
                                 env_name=args.env_name,
                                 seed=args.seed,
                                 num_processes=args.num_processes,
                                 num_frame_stack=args.num_frame_stack,
                                 downsample=not args.no_downsample,
                                 color=args.color,
                                 entropy_threshold=args.entropy_threshold,
                                 collect_mode=args.probe_collect_mode,
                                 train_mode="train_encoder",
                                 checkpoint_index=args.checkpoint_index,
                                 min_episode_length=args.batch_size)

    observation_shape = tr_eps[0][0].shape
    if args.encoder_type == "Nature":
        if args.method == "ddim":
            print('Using ddim')
            encoder = NatureCNND(observation_shape[0], args)
        elif args.method == "mask":
            print('Use masks')
            encoder = NatureCNNMask(observation_shape[0], args)
        else:
            encoder = NatureCNN(observation_shape[0], args)
    elif args.encoder_type == "Impala":
        encoder = ImpalaCNN(observation_shape[0], args)
    elif args.encoder_type == "Swin":
                encoder = SwinMLP(args=args, in_chans=observation_shape[0], num_classes=args.feature_size, depths=[2, 3, 2], num_heads=[3, 3, 6])
    encoder.to(device)
    torch.set_num_threads(1)

    config = {}
    config.update(vars(args))
    config['obs_space'] = observation_shape  # weird hack
    if args.method == 'cpc':
        trainer = CPCTrainer(encoder, config, device=device, wandb=wandb)
    elif args.method == 'jsd-stdim':
        trainer = SpatioTemporalTrainer(encoder, config, device=device, wandb=wandb)
    elif args.method == 'vae':
        trainer = VAETrainer(encoder, config, device=device, wandb=wandb)
    elif args.method == 'ddim':
        trainer = DDIMTrainer(encoder, config, device=device, wandb=wandb)
    elif args.method == 'mask':
        trainer = MaskedTrainer(encoder, config, device=device, wandb=wandb)
    elif args.method == "naff":
        trainer = NaFFPredictorTrainer(encoder, config, device=device, wandb=wandb)
    elif args.method == "infonce-stdim":
        trainer = InfoNCESpatioTemporalTrainer(encoder, config, device=device, wandb=wandb)
    elif args.method == "global-infonce-stdim":
        trainer = GlobalInfoNCESpatioTemporalTrainer(encoder, config, device=device, wandb=wandb)
    elif args.method == "global-local-infonce-stdim":
        trainer = GlobalLocalInfoNCESpatioTemporalTrainer(encoder, config, device=device, wandb=wandb)
    elif args.method == "dim":
        trainer = DIMTrainer(encoder, config, device=device, wandb=wandb)
    else:
        assert False, "method {} has no trainer".format(args.method)

    trainer.train(tr_eps, val_eps)

    return encoder


if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args()
    tags = ['pretraining-only']
    wandb.init(project=args.wandb_proj, entity="curl-atari", tags=tags)
    config = {}
    config.update(vars(args))
    wandb.config.update(config)
    train_encoder(args)
