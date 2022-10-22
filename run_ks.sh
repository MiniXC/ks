# Achelous
pdm run python ks.py --train_set achelous/baseline --run_name achelous_baseline_tdnn --log_stages tdnn --clean_stages all --wandb_mode online --verbose True --use_cmvn True --use_cnn False
pdm run python ks.py --train_set achelous/baseline --run_name achelous_baseline_cnn --log_stages tdnn --clean_stages all --wandb_mode online --verbose True --use_cmvn True --use_cnn True

# DOLOS
#pdm run python ks.py --train_set dolos/baseline --run_name dolos_baseline_tdnn --log_stages tdnn --clean_stages all --wandb_mode online --verbose True --use_cmvn True --use_cnn False
#pdm run python ks.py --train_set dolos/baseline --run_name dolos_baseline_cnn --log_stages tdnn --clean_stages all --wandb_mode online --verbose True --use_cmvn True --use_cnn True

# HOMADOS
#pdm run python ks.py --train_set homados/topline_duration --run_name homados_topline_tdnn_dur --log_stages tdnn --clean_stages all --wandb_mode online --verbose True --use_cmvn True --use_cnn False
#pdm run python ks.py --train_set homados/baseline_duration --run_name homados_baseline_tdnn_dur --log_stages tdnn --clean_stages all --wandb_mode online --verbose True --use_cmvn True --use_cnn False
#pdm run python ks.py --train_set homados/topline_duration --run_name homados_topline_cnn_dur --log_stages tdnn --clean_stages all --wandb_mode online --verbose True --use_cmvn True --use_cnn True
#pdm run python ks.py --train_set homados/baseline_duration --run_name homados_baseline_cnn_dur --log_stages tdnn --clean_stages all --wandb_mode online --verbose True --use_cmvn True --use_cnn True

# CRONUS
#pdm run python ks.py --train_set cronus/convert     --run_name cronus_convert_cnn   --log_stages tdnn --clean_stages all --wandb_mode online --verbose True --use_cmvn True --use_cnn True
#pdm run python ks.py --train_set cronus/sample      --run_name cronus_sample_cnn    --log_stages tdnn --clean_stages all --wandb_mode online --verbose True --use_cmvn True --use_cnn True
#pdm run python ks.py --train_set cronus/baseline    --run_name cronus_baseline_cnn  --log_stages tdnn --clean_stages all --wandb_mode online --verbose True --use_cmvn True --use_cnn True
#pdm run python ks.py --train_set cronus/topline     --run_name cronus_topline_cnn_new   --log_stages tdnn --clean_stages all --wandb_mode online --verbose True --use_cmvn True --use_cnn True


#pdm run python ks.py --train_set v3_synth_1 --run_name v3_synth_1 --log_stages tdnn --clean_stages all --wandb_mode online --verbose True --use_cmvn True --use_cnn True
# pdm run python ks.py --train_set v3_synth_10 --run_name v3_synth_10 --log_stages tdnn --clean_stages all --wandb_mode online --verbose True --use_cmvn True --use_cnn True
# pdm run python ks.py --train_set v3_synth_50 --run_name v3_synth_50 --log_stages tdnn --clean_stages all --wandb_mode online --verbose True --use_cmvn True --use_cnn True
# pdm run python ks.py --train_set v3_synth_100 --run_name v3_synth_100 --log_stages tdnn --clean_stages all --wandb_mode online --verbose True --use_cmvn True --use_cnn True
#pdm run python ks.py --train_set v3_synth_250 --run_name v3_synth_250 --log_stages tdnn --clean_stages all --wandb_mode online --verbose True --use_cmvn True --use_cnn True
#pdm run python ks.py --train_set v3_synth_500 --run_name v3_synth_500 --log_stages tdnn --clean_stages all --wandb_mode online --verbose True --use_cmvn True --use_cnn True
#pdm run python ks.py --train_set v3_synth_900 --run_name v3_synth_900 --log_stages tdnn --clean_stages all --wandb_mode online --verbose True --use_cmvn True --use_cnn True



#pdm run python ks.py --train_set v2_synth --run_name v2-synth --log_stages tdnn --clean_stages all --wandb_mode online --verbose True --use_cmvn False
# pdm run python ks.py --train_set v2_real --run_name v2_real_cnn --log_stages tdnn --clean_stages tdnn --wandb_mode online --verbose True --use_cmvn True --use_cnn True

#pdm run python ks.py --train_set v2_synth_duration --run_name v2-synth-duration --log_stages tdnn --clean_stages all --wandb_mode online --verbose True --use_cmvn False
#pdm run python ks.py --train_set v2_synth_duration --run_name v2-synth-duration-cmvn --log_stages tdnn --clean_stages all --wandb_mode online --verbose True --use_cmvn True

#pdm run python ks.py --train_set v2_real --run_name v2-real --log_stages tdnn --clean_stages all --wandb_mode online --verbose True --use_cmvn False
#pdm run python ks.py --train_set v2_real --run_name v2-real-cmvn --log_stages tdnn --clean_stages all --wandb_mode online --verbose True --use_cmvn True

#pdm run python ks.py --train_set v2_synth_augment --run_name v2-synth-augment --log_stages tdnn --clean_stages all --wandb_mode online --verbose True --use_cmvn False
#pdm run python ks.py --train_set v2_synth_augment --run_name v2-synth-augment-cmvn --log_stages tdnn --clean_stages all --wandb_mode online --verbose True --use_cmvn True

#pdm run python ks.py --train_set v2_real_augment --run_name v2-real-augment --log_stages tdnn --clean_stages all --wandb_mode online --verbose True --use_cmvn False
#pdm run python ks.py --train_set v2_real_augment --run_name v2-real-augment-cmvn --log_stages tdnn --clean_stages all --wandb_mode online --verbose True --use_cmvn True
