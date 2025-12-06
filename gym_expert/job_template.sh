#!/bin/bash
#SBATCH --job-name=rl_final # 作業名稱
#SBATCH --output=pointmaz_4room.txt # 輸出檔名
#SBATCH --ntasks=16          # 使用核心數量
#SBATCH --time=12:00:00     # 最大執行時間 (HH:MM:SS)
#SBATCH --mem=128G            # 記憶體需求
#SBATCH --account=b13901088 # 指定自己的帳戶名
#SBATCH --partition=long   # 根據時間選擇 short、standard、long

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/storage/undergrad/b13901088/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/storage/undergrad/b13901088/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/storage/undergrad/b13901088/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/storage/undergrad/b13901088/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate rl-final
cd "${HOME}/RL-final-project/gym_expert" || exit
python point_maze_sac_4room.py