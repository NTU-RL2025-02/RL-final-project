#! /bin/bash
# Run ThriftyDagger experiment in tmux
# Usage: ./run.sh

EXP_NAME="exp_expert_four_room_noisy=0.2"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BASENAME="${EXP_NAME}_${TIMESTAMP}"
SESSION_NAME="pointmaze_$BASENAME"
RECOVERY_TYPE="expert"
BC_CKPT="models/bc_policy_aawmaze_10.pt"   # 跟剛剛存的一樣路徑

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
  --expert_policy_file models/best_model \
  --demonstration_set_file models/offline_dataset_aawmaze_10.pkl \
  --max_expert_query 100000 \
  --environment 'PointMaze_4rooms-v3' \
  --recovery_type $RECOVERY_TYPE \
  --num_test_episodes 100 \
  --fix_thresholds \
  --noisy_scale 0.2 \
  --bc_checkpoint $BC_CKPT \
  $BASENAME > output_$BASENAME.txt 2>&1
"

echo "Started tmux session: $SESSION_NAME"
echo "Attach with: tmux attach -t $SESSION_NAME"

# Add --skip_bc_pretrain \ after bc_checkpoint... when you have generated the bc model
