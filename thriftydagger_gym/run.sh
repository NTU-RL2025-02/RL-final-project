#! /bin/bash
# Run ThriftyDagger experiment in tmux
# Usage: ./run.sh

EXP_NAME="exp_q_four_room_expert"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BASENAME="${EXP_NAME}_${TIMESTAMP}"
SESSION_NAME="pointmaze_$BASENAME"
RECOVERY_TYPE="q"
BC_CKPT="models/bc_policy_medium.pt"   # 跟剛剛存的一樣路徑

# 檢查 session 是否已存在
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "Session '$SESSION_NAME' already exists. Abort."
  exit 1
fi

tmux new-session -d -s "$SESSION_NAME" "
source ~/.bashrc
conda activate rl-final

python3 scripts/run_thriftydagger.py \
  --seed 48763 \
  --device 0 \
  --iters 100 \
  --targetrate 0.01 \
  --expert_policy_file models/best_model_4rooms \
  --recovery_policy_file models/best_model_4rooms \
  --demonstration_set_file models/offline_dataset_4rooms_392.pkl \
  --max_expert_query 100000 \
  --environment 'PointMaze_4rooms-v3' \
  --recovery_type $RECOVERY_TYPE \
  --num_test_episodes 100 \
  $BASENAME > output_$BASENAME.txt 2>&1
"

echo "Started tmux session: $SESSION_NAME"
echo "Attach with: tmux attach -t $SESSION_NAME"
