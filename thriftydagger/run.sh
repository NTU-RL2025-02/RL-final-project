#! /bin/bash
# Run ThriftyDagger experiment in tmux
# Usage: bash run.sh

EXP_NAME="thrifty_1130"
SESSION_NAME="thriftydagger_$EXP_NAME"

tmux new-session -d -s "$SESSION_NAME" "
source ~/.zshrc
conda activate rl-final

python3 scripts/run_thriftydagger.py \
  --seed 0 \
  --device 0 \
  --iters 20 \
  --targetrate 0.01 \
  --expert_policy_file models/model_epoch_2000_low_dim_v15_success_0.5.pth \
  --recovery_policy_file models/model_epoch_1000.pth \
  --demonstration_set_file models/model_epoch_2000_low_dim_v15_success_0.5-1000.pkl \
  --max_expert_query 2000 \
  --environment SquareNutAssembly \
  $EXP_NAME > output_$EXP_NAME.txt
"

echo "Started tmux session: $SESSION_NAME"
echo "Attach with: tmux attach -t $SESSION_NAME"
