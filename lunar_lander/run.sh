#! /bin/zsh
# Run ThriftyDagger experiment in tmux
# Usage: ./run.sh

EXP_NAME="add_nn_modulelist"
SESSION_NAME="lunarlander_$EXP_NAME"

# 檢查 session 是否已存在
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "Session '$SESSION_NAME' already exists. Abort."
  exit 1
fi

tmux new-session -d -s "$SESSION_NAME" "
source ~/.zshrc
conda activate rl-final

python3 scripts/run_thriftydagger.py \
  --seed 42 \
  --device 0 \
  --iters 100 \
  --targetrate 0.01 \
  --expert_policy_file models/best_model.zip \
  --recovery_policy_file models/best_model.zip \
  --demonstration_set_file models/offline_dataset.pkl \
  --max_expert_query 2000 \
  --environment LunarLander-v3 \
  --no_render \
  $EXP_NAME > output_$EXP_NAME.txt 2>&1
"

echo "Started tmux session: $SESSION_NAME"
echo "Attach with: tmux attach -t $SESSION_NAME"

