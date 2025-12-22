#! /bin/bash
# Run ThriftyDagger experiment in tmux
# Usage: ./run.sh

ENVIRONMENT_NAME="PointMaze_4rooms-v3"

FALSE=0
TRUE=1
USE_RULE_BASE_EXPERT=$TRUE
DEMONSTRATION_PATH="models/demonstrations/offline_data_100.pkl"
USE_BC_CHECKPOINT=$TRUE
BC_CHECKPOINT_PATH="models/bc_models/4room_rule_base_100_noise_0.pt"

RECOVERY_TYPE="q"
NOISY_SCALE="1.0"
MAX_EXPERT_QUERY="50000"
TEST_EPISODE_AMOUNT="100"

EXP_NAME="4room_no_risk_expert"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BASENAME="${TIMESTAMP}_${EXP_NAME}"
SESSION_NAME="pointmaze_$BASENAME"

# 檢查 session 是否已存在
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "Session '$SESSION_NAME' already exists. Abort."
  exit 1
fi


if (( $USE_BC_CHECKPOINT )); then 
  tmux new-session -d -s "$SESSION_NAME" "
  source ~/.bashrc
  conda activate rl-final
    
  python3 scripts/run_thriftydagger.py \
    --seed 48763 \
    --device 0 \
    --iters 100 \
    --targetrate 0.01 \
    --demonstration_set_file $DEMONSTRATION_PATH \
    --max_expert_query $MAX_EXPERT_QUERY \
    --environment $ENVIRONMENT_NAME \
    --recovery_type $RECOVERY_TYPE \
    --num_test_episodes $TEST_EPISODE_AMOUNT \
    --fix_thresholds \
    --noisy_scale $NOISY_SCALE \
    --rule_expert $USE_RULE_BASE_EXPERT \
    --bc_checkpoint $BC_CHECKPOINT_PATH \
    --skip_bc_pretrain \
    $BASENAME > output_$BASENAME.txt 2>&1
  "
else
  tmux new-session -d -s "$SESSION_NAME" "
  source ~/.bashrc
  conda activate rl-final
    
  python3 scripts/run_thriftydagger.py \
    --seed 48763 \
    --device 0 \
    --iters 100 \
    --targetrate 0.01 \
    --demonstration_set_file $DEMONSTRATION_PATH \
    --max_expert_query $MAX_EXPERT_QUERY \
    --environment $ENVIRONMENT_NAME \
    --recovery_type $RECOVERY_TYPE \
    --num_test_episodes $TEST_EPISODE_AMOUNT \
    --fix_thresholds \
    --noisy_scale $NOISY_SCALE \
    --rule_expert $USE_RULE_BASE_EXPERT\
    --save_bc_checkpoint $BC_CHECKPOINT_PATH \
    $BASENAME > output_$BASENAME.txt 2>&1
  "
fi

echo "Started tmux session: $SESSION_NAME"
echo "Attach with: tmux attach -t $SESSION_NAME"
