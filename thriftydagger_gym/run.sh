#! /bin/zsh
# Run ThriftyDagger experiment in tmux
# Usage: ./run.sh

EXP_NAME="exp1"
# Recovery policy type: "five_q" (default) or "q"
RECOVERY_TYPE="five_q"
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
  --seed 365365365 \
  --device 0 \
  --iters 100 \
  --targetrate 0.01 \
  --expert_policy_file models/lunar_lander_best_model \
  --recovery_policy_file models/lunar_lander_best_model \
  --demonstration_set_file models/offline_dataset.pkl \
  --max_expert_query 2000 \
  --environment LunarLander-v3 \
  --no_render \
  --recovery_type $RECOVERY_TYPE \
  $EXP_NAME > output_$EXP_NAME.txt 2>&1
"

echo "Started tmux session: $SESSION_NAME"
echo "Attach with: tmux attach -t $SESSION_NAME"

