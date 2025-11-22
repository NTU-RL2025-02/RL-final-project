# RL-final-project
Repository for RL final project - create for blank now
  
# If you want to use new robosuite+mujoco, pleases
```sh
git clone https://github.com/NTU-RL2025-02/RL-final-project
cd RL-final-project
git clone https://github.com/NTU-RL2025-02/thriftydagger
cd thriftydagger
conda create -n thrifty-dagger py=3.10
conda activate thrifty-dagger
pip install -e .
```
to run (still fixing)
```sh
python scripts/run_thriftydagger.py test --no_render
```
and you will see numpy broadcast mismatch (51, )->(61, ) error

to record (still fixing)
```sh
python scripts/run_thriftydagger.py test --gen_data
```