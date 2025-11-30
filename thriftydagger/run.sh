#! /bin/zsh
# Run ThriftyDagger experiment in tmux
# Usage: ./run.sh

EXP_NAME="thrifty_1130-target_rate=0.01"
SESSION_NAME="thriftydagger_$EXP_NAME"

# 檢查 session 是否已存在
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "Session '$SESSION_NAME' already exists. Abort."
  exit 1
fi

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
  --max_expert_query 160000 \
  --environment SquareNutAssembly \
  $EXP_NAME > output_$EXP_NAME.txt 2>&1
"

echo "Started tmux session: $SESSION_NAME"
echo "Attach with: tmux attach -t $SESSION_NAME"
