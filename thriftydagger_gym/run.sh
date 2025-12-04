#! /bin/bash
# Run ThriftyDagger experiment in tmux
# Usage: ./run.sh

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXP_NAME="exp_pointmaze_medium_$TIMESTAMP"
SESSION_NAME="pointmaze_$EXP_NAME"
RECOVERY_TYPE="five_q"

# 檢查 session 是否已存在
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "Session '$SESSION_NAME' already exists. Abort."
  exit 1
fi

tmux new-session -d -s "$SESSION_NAME" "
source ~/.bashrc
conda activate rl-final

python3 scripts/run_thriftydagger.py \
  --seed 365365365 \
  --device 0 \
  --iters 100 \
  --targetrate 0.01 \
  --expert_policy_file models/best_model_medium \
  --recovery_policy_file models/best_model_medium \
  --demonstration_set_file models/offline_dataset_mazeMedium_1000.pkl \
  --max_expert_query 2000 \
  --environment 'PointMaze_Medium-v3' \
  --recovery_type $RECOVERY_TYPE \
  --num_test_episodes 100 \
  $EXP_NAME > output_$EXP_NAME.txt 2>&1
"

echo "Started tmux session: $SESSION_NAME"
echo "Attach with: tmux attach -t $SESSION_NAME"

