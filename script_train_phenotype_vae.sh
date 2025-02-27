OUT_DIR='outdir'
mkdir -p ${OUT_DIR}
CUDA_VISIBLE_DEVICES=0 python main_phenotype_vae.py --batch_size 64 --epochs 400 --warmup_epochs 10 --lr 5e-4 --weight_decay 0.01 \
                                        --save_last_freq 1 --eval_freq 1 --online_eval --gen_num 30000 --temperature 1.0 \
                                        --latent_dim 32 --hidden_dims 128,128,128 --kld_weight 0.01\
                                        --output_dir ${OUT_DIR} --log_dir ${OUT_DIR}
