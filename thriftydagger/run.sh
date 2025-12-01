#! /bin/zsh
# Run ThriftyDagger experiment in tmux
# Usage: ./run.sh

EXP_NAME="modifying thriftydagger script"
SESSION_NAME="thriftydagger_$EXP_NAME"

# 檢查 session 是否已存在
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "Session '$SESSION_NAME' already exists. Abort."
  exit 1
fi

tmux new-session -d -s "$SESSION_NAME" "
source ~/.zshrc
conda activate rl-final

# Ensure Python imports local workspace packages first (avoid /home/amber/Desktop/codes/...)
PROJECT_ROOT="$(cd \"$(dirname \"$0\")/..\" && pwd)"
export PYTHONPATH="$PROJECT_ROOT":$PYTHONPATH

python3 scripts/run_thriftydagger.py \
  --seed 42 \
  --device 0 \
  --iters 100 \
  --targetrate 0.01 \
  --expert_policy_file models/model_epoch_1150_low_dim_v15_success_0.74.pth \
  --recovery_policy_file models/model_epoch_1150_low_dim_v15_success_0.74.pth \
  --demonstration_set_file models/model_epoch_1150_low_dim_v15_success_0.74-10000.pkl \
  --max_expert_query 2000 \
  --environment SquareNutAssembly \
  --no_render \
  $EXP_NAME > output_$EXP_NAME.txt 2>&1
"

echo "Started tmux session: $SESSION_NAME"
echo "Attach with: tmux attach -t $SESSION_NAME"

# What I've changed:
# Changed the place of a_expert and a_recovery.
# Add nn.modulelist in core/Ensemble