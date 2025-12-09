#! /bin/bash
# Run ThriftyDagger experiment in tmux
# Usage: ./run.sh

ENVIRONMENT_NAME="PointMaze_4rooms-v3"

FALSE=0
TRUE=1
USE_RULE_BASE_EXPERT=$TRUE
# if USE_RULE_BASE_EXPERT is false, then use following expert policy
EXPERT_POLICY_PATH="models/experts/best_model_4rooms.zip"
DEMONSTRATION_PATH="models/demonstrations/4room_rule_base_5_noise_0.2.pkl"
USE_BC_CHECKPOINT=$FALSE
BC_CHECKPOINT_PATH="models/bc_models/4room_rule_base_5_noise_0.2.pt"

RECOVERY_TYPE="expert"
NOISY_SCALE="0.2"
MAX_EXPERT_QUERY="5000"
TEST_EPISODE_AMOUNT="100"

EXP_NAME="exp_4room_rule_based_pkl_ep5_noise0.2_noisy=0.2"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BASENAME="${EXP_NAME}_${TIMESTAMP}"
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
    --expert_policy_file $EXPERT_POLICY_PATH \
    --demonstration_set_file $DEMONSTRATION_PATH \
    --max_expert_query $MAX_EXPERT_QUERY \
    --environment $ENVIRONMENT_NAME \
    --recovery_type $RECOVERY_TYPE \
    --num_test_episodes $TEST_EPISODE_AMOUNT \
    --fix_thresholds \
    --noisy_scale $NOISY_SCALE \
    --rule_expert $USE_RULE_BASE_EXPERT\
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
    --expert_policy_file $EXPERT_POLICY_PATH \
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

