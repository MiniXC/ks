# sleep 15m
rm -r exp
CUDA_VISIBLE_DEVICES="3" pdm run python ks.py \
--train_set last_one \
--run_name last_one \
--log_stages tdnn \
--clean_stages all \
--wandb_mode online \
--verbose True \
--use_cmvn True \
--use_cnn False
