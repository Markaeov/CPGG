import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched
import numpy as np
import os
from ldm.util import compute_ssim, compute_psnr
from einops import rearrange

def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def train_one_epoch(model,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    optimizer_d: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if args.debug:
            print('debug_mode is on, only run 10 iterations, current iteration: {}'.format(data_iter_step))
            if data_iter_step > 10:
                break
        # we use a per iteration (instead of per epoch) lr scheduler
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        # labels = labels.to(device, non_blocking=True)

        inputs = model.get_input(samples)
        start_frame = 0

        with torch.cuda.amp.autocast():
            reconstructions, posterior, motion = model(inputs)
            aeloss, log_dict_ae = model.loss(inputs[:, :, start_frame:], reconstructions, posterior, 0, model.global_step, last_layer=model.get_last_layer(), split="train")

        params = list(model.encoder_3d.parameters())+ list(model.decoder.parameters())+ \
                        list(model.quant_conv.parameters())+ list(model.post_quant_conv.parameters())
        if model.enable_2d:
            params += list(model.encoder_2d.parameters())  
        if model.loss.logvar.requires_grad:
            params.append(model.loss.logvar)
        loss_scaler(aeloss, optimizer, clip_grad=args.grad_clip, parameters=params, update_grad=True)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            discloss, log_dict_disc = model.loss(inputs[:, :, start_frame:], reconstructions, posterior, 1, model.global_step,last_layer=model.get_last_layer(), split="train")

        disc_params = []
        if model.loss.enable_2d:
            disc_params  +=  list(model.loss.discriminator_2d.parameters())
        if model.loss.enable_3d:
            disc_params  +=  list(model.loss.discriminator.parameters())
        loss_scaler(discloss, optimizer_d, clip_grad=args.grad_clip, parameters=disc_params, update_grad=True)
        optimizer_d.zero_grad()

        model.global_step += 1
        loss_value = aeloss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        metric_logger.update(loss=loss_value)
        metric_logger.update(total_loss_ae=log_dict_ae['train/total_loss_ae'])
        metric_logger.update(p_loss=log_dict_ae['train/p_loss'])
        metric_logger.update(nll_loss=log_dict_ae['train/nll_loss'])
        metric_logger.update(rec_loss=log_dict_ae['train/rec_loss'])
        metric_logger.update(kl_loss=log_dict_ae['train/kl_loss'])
        metric_logger.update(weighted_kl_loss=log_dict_ae['train/weighted_kl_loss'])
        metric_logger.update(g_loss=log_dict_ae['train/g_loss'])
        metric_logger.update(weighted_g_loss=log_dict_ae['train/weighted_g_loss'])
        metric_logger.update(penalty_loss=log_dict_ae['train/penalty_loss'])

        # discriminator loss
        metric_logger.update(disc_loss=log_dict_disc['train/disc_loss'])

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        if log_writer is not None:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train/total_aeloss', log_dict_ae['train/total_loss_ae'], epoch_1000x)
            log_writer.add_scalar('train/p_loss', log_dict_ae['train/p_loss'], epoch_1000x)
            log_writer.add_scalar('train/nll_loss', log_dict_ae['train/nll_loss'], epoch_1000x)
            log_writer.add_scalar('train/rec_loss', log_dict_ae['train/rec_loss'], epoch_1000x)
            log_writer.add_scalar('train/kl_loss', log_dict_ae['train/kl_loss'], epoch_1000x)
            log_writer.add_scalar('train/weighted_kl_loss', log_dict_ae['train/weighted_kl_loss'], epoch_1000x)
            log_writer.add_scalar('train/g_loss', log_dict_ae['train/g_loss'], epoch_1000x)
            log_writer.add_scalar('train/weighted_g_loss', log_dict_ae['train/weighted_g_loss'], epoch_1000x)
            log_writer.add_scalar('train/penalty_loss', log_dict_ae['train/penalty_loss'], epoch_1000x)
            log_writer.add_scalar('train/disc_loss', log_dict_disc['train/disc_loss'], epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def evaluate(model, dataloader, args, epoch, log_writer=None, limit_num=-1, device=None):
    model.eval()
    val_psnr = 0
    val_ssim = 0
    val_lpips = 0
    val_l1 = 0

    for idx, (samples, labels) in enumerate(dataloader):
        if limit_num != -1 and idx >= limit_num:
            break
        with torch.no_grad():
            samples = samples.to(device, non_blocking=True)
            x = model.get_input(samples)
            batch_size = x.shape[0]
            with torch.cuda.amp.autocast():
                xrec, posterior, _ = model(x)
            x = rearrange(x[:, :, 0:], "b c t h w -> (b t) c h w").clamp(-1.0, 1.0)
            xrec = rearrange(xrec, "b c t h w -> (b t) c h w").clamp(-1.0, 1.0)
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            xrec = (xrec + 1.0) / 2.0   # -1,1 -> 0,1; c,h,w
            xrec = xrec.float()
            psnr = compute_psnr(xrec, x)
            ssim = compute_ssim(xrec, x)
            lpips = model.loss.perceptual_loss(x, xrec).mean()
            l1 = torch.nn.functional.l1_loss(x, xrec)
            val_psnr += psnr
            val_ssim += ssim
            val_lpips += lpips
            val_l1 += l1

            if idx == 0:
                save_num = 3 if x.shape[0] > 3 else x.shape[0]
                x = x.cpu().numpy()
                xrec = xrec.cpu().numpy()
                x = x * 255
                xrec = xrec * 255
                x = x.astype(np.uint8)
                xrec = xrec.astype(np.uint8)
                x = rearrange(x, "(b t) c h w -> b t c h w", b=batch_size)
                xrec = rearrange(xrec, "(b t) c h w -> b t c h w", b=batch_size)
                gif = np.concatenate(x[:save_num], axis=3)
                gif_rec = np.concatenate(xrec[:save_num], axis=3)
                save_gif = np.concatenate([gif, gif_rec], axis=2)
                save_numpy_as_gif(save_gif, os.path.join(args.output_dir, "reconstruct/epoch_{}.gif".format(epoch)), duration=0.05)
                
                

    if limit_num == -1: limit_num = len(dataloader)
    val_psnr /= limit_num
    val_ssim /= limit_num
    val_lpips /= limit_num
    val_l1 /= limit_num

    print("Validation PSNR: {:.4f} SSIM: {:.4f} LPIPS: {:.4f} L1: {:.4f}".format(val_psnr, val_ssim, val_lpips, val_l1))

    if log_writer is not None:
        """ We use epoch_1000x as the x-axis in tensorboard.
        This calibrates different curves when batch size changes.
        """
        log_writer.add_scalar('val/psnr', val_psnr, epoch)
        log_writer.add_scalar('val/ssim', val_ssim, epoch)
        log_writer.add_scalar('val/lpips', val_lpips, epoch)
        log_writer.add_scalar('val/l1', val_l1, epoch)

import imageio
def save_numpy_as_gif(frames, path, duration=None):
    """
    save numpy array as gif file
    frames: frame_num, c, h, w
    """
    image_list = []
    for frame in frames:
        image = frame.transpose(1, 2, 0)
        image_list.append(image)
    if duration:
        imageio.mimsave(path, image_list, format="GIF", duration=duration, loop=0)
    else:
        imageio.mimsave(path, image_list, format="GIF", loop=0)
