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
from util.misc import NativeScalerWithGradNormCount as NativeScaler

from models import vae3d
from engine_vae import train_one_epoch, evaluate
import copy
from dataset import UKBCMR2DTrain, UKBCMR2DValidation

import warnings
warnings.filterwarnings("ignore")

def get_args_parser():
    parser = argparse.ArgumentParser('VAE-3D', add_help=False)
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


    dataset_train = UKBCMR2DTrain()
    print(dataset_train)
    dataset_val = UKBCMR2DValidation()

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        shuffle=True,
    )
    data_loader_validation = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        shuffle=False,
    )
    from omegaconf import OmegaConf
    model_config = OmegaConf.load(args.cfg)
    # Save args to a yaml file
    args_dict = vars(args)
    args_yaml_path = os.path.join(args.output_dir, 'config/args.yaml')
    with open(args_yaml_path, 'w') as f:
        OmegaConf.save(config=OmegaConf.create(args_dict), f=f)

    # Save model_config to a yaml file
    model_config_yaml_path = os.path.join(args.output_dir, 'config/model_config.yaml')
    with open(model_config_yaml_path, 'w') as f:
        OmegaConf.save(config=model_config, f=f)
    model = vae3d.__dict__[args.model](
        lossconfig=model_config.model.params.lossconfig, 
        ddconfig=model_config.model.params.ddconfig, 
        ddconfig_2d=model_config.model.params.ddconfig_2d,
        embed_dim=model_config.model.params.embed_dim,
    )

    print("Model = %s" % str(model))
    # following timm: set wd as 0 for bias and norm layers
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: {}M".format(n_params / 1e6))

    model.to(device)

    eff_batch_size = args.batch_size * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("effective batch size: %d" % eff_batch_size)

    params = list(model.encoder_3d.parameters())+ list(model.decoder.parameters())+ \
                        list(model.quant_conv.parameters())+ list(model.post_quant_conv.parameters())
        
    if model.enable_2d:
        params += list(model.encoder_2d.parameters())
            
    if model.loss.logvar.requires_grad:
        params.append(model.loss.logvar)

    optimizer = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)

    disc_params = []
    if model.loss.enable_2d:
        disc_params  +=  list(model.loss.discriminator_2d.parameters())
    if model.loss.enable_3d:
        disc_params  +=  list(model.loss.discriminator.parameters())
    optimizer_d = torch.optim.AdamW(disc_params, lr=float(args.disc_loss_scale) * float(args.lr), betas=(0.9, 0.95), weight_decay=args.weight_decay)
    print(optimizer)
    print(optimizer_d)
    loss_scaler = NativeScaler()

    # resume training
    if args.resume and os.path.exists(os.path.join(args.resume, "checkpoint-last.pth")):
        checkpoint = torch.load(os.path.join(args.resume, "checkpoint-last.pth"), map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        model_params = list(model.parameters())
        ema_state_dict = checkpoint['model_ema']
        ema_params = [ema_state_dict[name].cuda() for name, _ in model.named_parameters()]
        print("Resume checkpoint %s" % args.resume)

        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            print("With optim & sched!")
        del checkpoint
    else:
        model_params = list(model.parameters())
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
            optimizer, optimizer_d,
            device, epoch,
            log_writer=log_writer,
            args=args,
            loss_scaler=loss_scaler,
        )

        # save checkpoint
        if epoch % args.save_last_freq == 0 or epoch + 1 == args.epochs:
            torch.save(model.state_dict(), os.path.join(args.output_dir, f"checkpoint-{epoch:04d}.pth"))

        # online evaluation
        if args.online_eval and (epoch % args.eval_freq == 0 or epoch + 1 == args.epochs):
            torch.cuda.empty_cache()
            if args.debug:
                args.limit_num = 1
            evaluate(model, data_loader_validation, args, epoch, log_writer=log_writer, limit_num=args.limit_num, device=device)
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
