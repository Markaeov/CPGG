import argparse
import datetime
import numpy as np
import os
import time

import torch
import torch.backends.cudnn as cudnn


import util.misc as misc

from engine_cpgg import cache_latents


from dataset import UKBCMR2DTrain_build_cache, UKBCMR2DValidation_build_cache
from models import vae3d


def get_args_parser():
    parser = argparse.ArgumentParser('caching CMR latent', add_help=False)
    parser.add_argument('--batch_size', default=4, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * # gpus')
    parser.add_argument('--epochs', default=400, type=int)

    # Model parameters
    parser.add_argument('--model', default='vae3d', type=str, metavar='MODEL',
                        help='Name of model to train')


    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.02,
                        help='weight decay (default: 0.02)')

    parser.add_argument('--grad_checkpointing', action='store_true')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--lr_schedule', type=str, default='constant',
                        help='learning rate schedule')
    parser.add_argument('--warmup_epochs', type=int, default=0, metavar='N',
                        help='epochs to warmup LR')

    parser.add_argument('--grad_clip', type=float, default=3.0,
                        help='Gradient clip')

    # Dataset parameters

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # caching latents

    parser.add_argument('--cfg', default='./config/vae_kl.yaml', help='config path')

    parser.add_argument('--disc_loss_scale', default=0.1, type=float, help='disc_loss_scale')
    parser.add_argument('--save_last_freq', default=1, type=int, help='save_last_freq')
    parser.add_argument('--eval_freq', default=1, type=int, help='eval_freq')
    parser.add_argument('--limit_num', default=100, type=int, help='')
    parser.add_argument('--online_eval', action='store_true', help='online_eval')
    parser.add_argument('--debug', action='store_true', help='debug mode')

    parser.add_argument('--cached_path', default='', help='cached save path')

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True


    dataset_train = UKBCMR2DTrain_build_cache()
    dataset_val = UKBCMR2DValidation_build_cache()
    print(dataset_train)


    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False, 
        shuffle=False
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False, 
        shuffle=False
    )


    from omegaconf import OmegaConf
    model_config = OmegaConf.load(args.cfg)
    
    vae = vae3d.__dict__[args.model](
        lossconfig=model_config.model.params.lossconfig, 
        ddconfig=model_config.model.params.ddconfig, 
        ddconfig_2d=model_config.model.params.ddconfig_2d,
        embed_dim=model_config.model.params.embed_dim,
    ).cuda()

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        vae.load_state_dict(checkpoint)
        print("Resume checkpoint %s" % args.resume)
        del checkpoint
    vae = vae.eval()
    # training
    print(f"Start caching VAE latents")
    start_time = time.time()
    print('Starting caching training latents')
    cache_latents(
        vae,
        data_loader_train,
        device,
        args=args
    )
    print('Starting caching validation latents')
    cache_latents(
        vae,
        data_loader_val,
        device,
        args=args
    )
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Caching time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
