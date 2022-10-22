CUDA_VISIBLE_DEVICES="3" pdm run python ks.py \
--train_set early_stop_mae_phone_full \
--run_name early_stop_mae_phone_full \
--log_stages tdnn \
--clean_stages all \
--wandb_mode online \
--verbose True \
--use_cmvn True \
--use_cnn False
