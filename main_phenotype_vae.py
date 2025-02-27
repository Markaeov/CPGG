import argparse
import datetime
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import util.misc as misc

from models.phenotypeVAE import PhenotypeVAE
from engine_phenotype_vae import train_one_epoch, evaluate
from dataset import UKBCMR_MetricsTrain, UKBCMR_MetricsValidation

import warnings
warnings.filterwarnings("ignore")

def parse_tuple(value):
    return tuple(map(int, value.split(',')))

def get_args_parser():
    parser = argparse.ArgumentParser('phenotype vae training', add_help=False)
    parser.add_argument('--batch_size', default=4, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * # gpus')
    parser.add_argument('--epochs', default=400, type=int)


    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.02,
                        help='weight decay (default: 0.02)')

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

    parser.add_argument('--save_last_freq', default=1, type=int, help='save_last_freq')
    parser.add_argument('--eval_freq', default=1, type=int, help='eval_freq')
    parser.add_argument('--online_eval', action='store_true', help='online_eval')
    parser.add_argument('--gen_num', default=30000, type=int, help='debug mode')
    parser.add_argument('--temperature', default=1.0, type=float, help='generate temperature')
    parser.add_argument('--kld_weight', default=1.0, type=float, help='kld_weight')
    parser.add_argument('--latent_dim', default=32, type=int, help='z dim')
    parser.add_argument('--hidden_dims', default=(128, 128, 128), type=parse_tuple, help='hidden_dims')

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

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs(args.log_dir + '/reconstruct', exist_ok=True)
        os.makedirs(args.log_dir + '/config', exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None


    dataset_train = UKBCMR_MetricsTrain()
    dataset_val = UKBCMR_MetricsValidation()

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        shuffle=True,
    )
    from omegaconf import OmegaConf
    # Save args to a yaml file
    args_dict = vars(args)
    args_yaml_path = os.path.join(args.output_dir, 'config/args.yaml')
    with open(args_yaml_path, 'w') as f:
        OmegaConf.save(config=OmegaConf.create(args_dict), f=f)

    model = PhenotypeVAE(in_channels=82, latent_dim=args.latent_dim, hidden_dims=args.hidden_dims)

    print("Model = %s" % str(model))
    # following timm: set wd as 0 for bias and norm layers
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: {}M".format(n_params / 1e6))

    model.to(device)
    # model_without_ddp = model

    eff_batch_size = args.batch_size * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("effective batch size: %d" % eff_batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)

    print("Training from scratch")

    # training
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(
            model,
            data_loader_train,
            optimizer,
            device, epoch,
            log_writer=log_writer,
            args=args,
        )

        # save checkpoint
        if epoch % args.save_last_freq == 0 or epoch + 1 == args.epochs:
            torch.save(model.state_dict(), os.path.join(args.output_dir, f"checkpoint-{epoch:04d}.pth"))

        # online evaluation
        if args.online_eval and (epoch % args.eval_freq == 0 or epoch + 1 == args.epochs):
            torch.cuda.empty_cache()
            # self defined fid for evaluating phenotype vae
            evaluate(model, dataset_train, dataset_val, args, epoch, log_writer=log_writer, device=device)
            torch.cuda.empty_cache()

        if misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    args.log_dir = args.output_dir
    main(args)
