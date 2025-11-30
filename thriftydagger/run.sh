#! /bin/bash
# Run ThriftyDagger experiment
# Usage: bash run.sh 

conda activate rl-final

nohup python3 scripts/run_thriftydagger.py \
    --seed 0 \
    --device 0 \
    --iters 20 \
    --targetrate 0.01 \
    --expert_policy_file models/model_epoch_2000_low_dim_v15_success_0.5.pth \
    --recovery_policy_file models/model_epoch_1000.pth \
    --demonstration_set_file models/model_epoch_2000_low_dim_v15_success_0.5-1000.pkl \
    --max_expert_query 2000 \
    --environment NutAssembly \
    meeting_exp_1130 \
