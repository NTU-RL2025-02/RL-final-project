#! /bin/zsh
# Run ThriftyDagger experiment in tmux
# Usage: ./run.sh

EXP_NAME="hard_code_20251201T1330"
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
  --iters 100 \
  --targetrate 0.01 \
  --expert_policy_file models/model_epoch_1150_low_dim_v15_success_0.74.pth \
  --recovery_policy_file models/model_epoch_1150_low_dim_v15_success_0.74.pth \
  --demonstration_set_file models/model_epoch_1150_low_dim_v15_success_0.74-10000.pkl \
  --max_expert_query 2000 \
  --environment SquareNutAssembly \
  --algo_sup \
  $EXP_NAME > output_$EXP_NAME.txt 2>&1
"

echo "Started tmux session: $SESSION_NAME"
echo "Attach with: tmux attach -t $SESSION_NAME"

# Experiment note:
# max_expert_query 更改成 計算 切換到 expert + recovery policy 的次數
# 更改 pth 和 pkl 為 成功率 0.74 的 model
# Iter 改成 100
# targetrate 為 0.01
# Expected Result: 成功率至少 0.5 以上
