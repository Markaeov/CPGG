import os
import os
import torch
import numpy as np
import argparse
from models import cpgg, vae3d
from pathlib import Path
from omegaconf import OmegaConf
from models.phenotypeVAE import PhenotypeVAE
from torch.utils.data import DataLoader, TensorDataset
torch.set_grad_enabled(False)

def parse_tuple(value):
    return tuple(map(int, value.split(',')))

def get_args_parser():
    parser = argparse.ArgumentParser('parser', add_help=False)
    parser.add_argument('--batch_size', default=4, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * # gpus')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=1, type=int)

    parser.add_argument('--gen_num', default=30000, type=int, help='debug mode')
    parser.add_argument('--diffloss_d', default=3, type=int, help='d')
    parser.add_argument('--diffloss_w', default=1024, type=int, help='w')
    parser.add_argument('--img_size', default=(50, 96, 96), type=parse_tuple, help='img_size')
    parser.add_argument('--vae_stride', default=(2, 8, 8), type=parse_tuple, help='vae_stride')
    parser.add_argument('--patch_size', default=(5, 2, 2), type=parse_tuple, help='patch_size')

    # phenotype vae config
    parser.add_argument('--latent_dim', default=32, type=int, help='z dim')
    parser.add_argument('--hidden_dims', default=(128, 128, 128), type=parse_tuple, help='hidden_dims')

    parser.add_argument('--cpgg_ckpt', default="", type=str, help='cpgg model ckpt')
    parser.add_argument('--vae3d_config', default="", type=str, help='vae3d model config yaml')
    parser.add_argument('--vae3d_ckpt', default="", type=str, help='vae3d model ckpt')
    parser.add_argument('--phenotype_vae_ckpt', default="", type=str, help='phenotype vae model ckpt')
    
    # generate config
    parser.add_argument('--num_ar_steps', default=16, type=int, help='generate ar step')
    parser.add_argument('--diff_temperature', default=1.0, type=float, help='temperature')
    parser.add_argument('--phenotype_vae_temperature', default=1.0, type=float, help='temperature')
    parser.add_argument('--cfg_scale', default=3.0, type=float, help='cfg scale')

    parser.add_argument('--use_rep_cond', action='store_true', help='use_rep_cond')
    # parser.set_defaults(use_rep_cond=True)
    parser.add_argument('--use_mae_loss', action='store_true', help='use_mae_loss')
    # parser.set_defaults(use_mae_loss=True)
    parser.add_argument('--coef_mae_loss', default=1.0, type=float, help='use_mae_loss')
    parser.add_argument('--diffloss_on_rep', action='store_true', help='')

    parser.add_argument('--phenotype_path', default='None', type=str, help='')
    return parser

args = get_args_parser()
args = args.parse_args()
Path(args.output_dir).mkdir(parents=True, exist_ok=True)
args.log_dir = args.output_dir

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("GPU not found. Using CPU instead.")
if args.log_dir is not None:
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.log_dir + '/sample', exist_ok=True)
    os.makedirs(args.log_dir + '/config', exist_ok=True)

seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)

model_type = "cpgg"
num_sampling_steps_diffloss = 100

model = cpgg.__dict__[model_type](
    buffer_size=64,
    diffloss_d=args.diffloss_d,
    diffloss_w=args.diffloss_w,
    num_sampling_steps=str(num_sampling_steps_diffloss),
    img_size=args.img_size,
    vae_stride=args.vae_stride,
    patch_size=args.patch_size,
    use_rep_cond=args.use_rep_cond,
    use_mae_loss=args.use_mae_loss,
    coef_mae_loss=args.coef_mae_loss,
    diffloss_on_rep=args.diffloss_on_rep,
).to(device)

state_dict = torch.load(args.cpgg_ckpt)["model_ema"]
print(model)
model.load_state_dict(state_dict, strict=False)
model.eval()
del state_dict


model_config = OmegaConf.load(args.vae3d_config)

vae = vae3d.__dict__['vae3d'](
    lossconfig=model_config.model.params.lossconfig, 
    ddconfig=model_config.model.params.ddconfig, 
    ddconfig_2d=model_config.model.params.ddconfig_2d,
    embed_dim=model_config.model.params.embed_dim,
).to(device)
checkpoint = torch.load(args.vae3d_ckpt)
vae.load_state_dict(checkpoint)
vae.eval()
del checkpoint


#phenotypes sampling
current_gen = os.listdir(os.path.join(args.output_dir, 'sample'))
current_len = len(current_gen)
if current_len > 0 and args.phenotype_path == 'None':
    phenotype_data = np.load(os.path.join(args.output_dir, 'phenotype_data.npy'))
else:
    if args.phenotype_path == 'None':
        PhenotypeVAE_model = PhenotypeVAE(82, args.latent_dim, args.hidden_dims)
        state_dict = torch.load(args.phenotype_vae_ckpt) #['model']
        PhenotypeVAE_model.load_state_dict(state_dict, strict=True)
        PhenotypeVAE_model.to('cuda:0')
        PhenotypeVAE_model.eval()
        phenotype_data = PhenotypeVAE_model.sample(num_samples=args.gen_num, temperature=args.phenotype_vae_temperature, current_device=0).detach().cpu().numpy()
        del state_dict
        # Save phenotype_data to the output directory
        output_path = os.path.join(args.output_dir, 'phenotype_data.npy')
        np.save(output_path, phenotype_data)
        print(f"Phenotype data saved to {output_path}")
    else:
        phenotype_data = torch.load(args.phenotype_path)
# Convert phenotype_data to a PyTorch tensor

phenotype_tensor = torch.tensor(phenotype_data[current_len:], dtype=torch.float32)
# Create a TensorDataset and DataLoader
phenotype_dataset = TensorDataset(phenotype_tensor)
phenotype_dataloader = DataLoader(phenotype_dataset, batch_size=args.batch_size, shuffle=False)
################################################################################

import numpy as np
import nibabel as nib


num_ar_steps = args.num_ar_steps
cfg_scale = args.cfg_scale
cfg_schedule = "constant"
temperature = args.diff_temperature

from tqdm import tqdm
# generate

iter = 0
for data in tqdm(phenotype_dataloader):
    class_labels = data[0] # TensorDataset
    # print(class_labels.shape)
    with torch.cuda.amp.autocast():
        sampled_tokens = model.sample_tokens(
            bsz=len(class_labels), num_iter=num_ar_steps,
            cfg=cfg_scale, cfg_schedule=cfg_schedule,
            labels=torch.Tensor(class_labels).float().cuda(),
            temperature=temperature, progress=False)
        sampled_images = vae.decode(sampled_tokens, None)
        # Convert the tensor to a numpy array and save it as a NIfTI file

    sampled_images_np = sampled_images.cpu().float().numpy()
    for i in range(sampled_images_np.shape[0]):
        nii_image = nib.Nifti1Image(sampled_images_np[i, 0], np.eye(4))
        nib.save(nii_image, args.output_dir + '/sample/{}.nii'.format(iter*args.batch_size + i + current_len))
    iter += 1