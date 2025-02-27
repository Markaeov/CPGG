OUT_DIR='outdir'
CPGG_CKPT='cpgg_ckpt'
VAE3D_CONFIG='./config/vae_kl_ft2_fs8_z16.yaml'
VAE3D_CKPT='vae3d_ckpt'
PHENOTYPE_VAE_CKPT='phenotype_vae_ckpt'
mkdir -p ${OUT_DIR}
CUDA_VISIBLE_DEVICES=0 python main_cpgg_sample.py --batch_size 16 --gen_num 2048 --diffloss_d 3 --diffloss_w 1024 \
                                        --img_size 50,96,96 --vae_stride 2,8,8 --patch_size 5,2,2 \
                                        --latent_dim 32 --hidden_dims 128,128,128\
                                        --cpgg_ckpt ${CPGG_CKPT} --vae3d_config ${VAE3D_CONFIG} --vae3d_ckpt ${VAE3D_CKPT} --phenotype_vae_ckpt ${PHENOTYPE_VAE_CKPT} \
                                        --num_ar_steps 16 --cfg_scale 3.0 --diff_temperature 1.0 --phenotype_vae_temperature 1.0 \
                                        --output_dir ${OUT_DIR} --log_dir ${OUT_DIR} --seed 0