OUT_DIR='outdir'
CACHED_PATH='cached_path'
mkdir -p ${OUT_DIR}
CUDA_VISIBLE_DEVICES=0 python main_cpgg.py --batch_size 64 --epochs 400 --warmup_epochs 100 --lr 8e-4 --weight_decay 0.01 --vae_embed_dim 16 --img_size 50,96,96 --vae_stride 2,8,8 --patch_size 5,2,2 \
                                        --model cpgg --diffloss_d 3 --diffloss_w 1024 --diffusion_batch_mul 4 \
                                        --output_dir ${OUT_DIR} --log_dir ${OUT_DIR} --cached_path ${CACHED_PATH} --use_cached