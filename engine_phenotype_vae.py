import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched
import numpy as np

def train_one_epoch(model,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int,
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
        # we use a per iteration (instead of per epoch) lr scheduler
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        # labels = labels.to(device, non_blocking=True)

        rec, inputs, mu, log_var = model(samples)
        loss_dict = model.loss_function(rec, inputs, mu, log_var, args.kld_weight)
        loss = loss_dict['loss']
        loss_value = loss.detach().item()
        rec_loss = loss_dict['Reconstruction_Loss'].item()
        kld = loss_dict['KLD'].item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        metric_logger.update(loss=loss_value)
        metric_logger.update(rec_loss=rec_loss)
        metric_logger.update(kld=kld)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        # loss_value_reduce = misc.all_reduce_mean(loss_value) # 这个是获取各个显卡的平均值
        if log_writer is not None:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train/loss', loss_value, epoch_1000x)
            log_writer.add_scalar('train/rec_loss', rec_loss, epoch_1000x)
            log_writer.add_scalar('train/kld', kld, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def evaluate(model, dataset_t, dataset_v, args, epoch, log_writer=None, device=None):
    model.eval()

    with torch.no_grad():
        samples = model.sample(args.gen_num, device, temperature=args.temperature)
            
    samples = samples.cpu().detach().numpy()
    samples_mean_values = np.mean(samples, axis=0)
    samples_variance_values = np.cov(samples, rowvar=False)

    orig_data = dataset_t.data
    orig_data = dataset_t.scaler.transform(orig_data)
    mean_values = np.mean(orig_data, axis=0)
    variance_values = np.cov(orig_data, rowvar=False)
    # variance_values = np.diag(variance_values)
    
    fid_t = calculate_fid(mean_values, variance_values, samples_mean_values, samples_variance_values)

    orig_data = dataset_v.data
    orig_data = dataset_v.scaler.transform(orig_data)
    mean_values = np.mean(orig_data, axis=0)
    variance_values = np.cov(orig_data, rowvar=False)
    # variance_values = np.diag(variance_values)
    fid_v = calculate_fid(mean_values, variance_values, samples_mean_values, samples_variance_values)


    print("Train fid: {:.4f} Validation fid: {:.4f}".format(fid_t, fid_v))

    if log_writer is not None:
        """ We use epoch_1000x as the x-axis in tensorboard.
        This calibrates different curves when batch size changes.
        """
        log_writer.add_scalar('val/train_fid', fid_t, epoch)
        log_writer.add_scalar('val/val_fid', fid_v, epoch)


from scipy.linalg import sqrtm
def calculate_fid(mu1, sigma1, mu2, sigma2):
    """Calculate FID score given means and covariances."""
    # Compute the difference between means
    mean_diff = np.sum((mu1 - mu2)**2)
    
    # Compute the sqrt of product of covariance matrices
    covmean = sqrtm(sigma1.dot(sigma2))
    
    # Numerical stability: ensure real values
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    # FID formula
    fid = mean_diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid