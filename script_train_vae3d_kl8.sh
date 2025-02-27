
OUT_DIR='outdir'
CFG_FILE='config/vae_kl_ft2_fs8_z16.yaml'
mkdir -p ${OUT_DIR}
CUDA_VISIBLE_DEVICES=0 python main_vae.py --batch_size 10 --epochs 100 --lr 8e-4 --weight_decay 0.01 --disc_loss_scale 0.1 --limit_num 100 --save_last_freq 1 --eval_freq 1 --online_eval --output_dir ${OUT_DIR} --log_dir ${OUT_DIR} --cfg ${CFG_FILE}