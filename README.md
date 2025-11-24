# RL-final-project
Repository for RL final project - create for blank now
  
## If you want to use new robosuite+mujoco, pleases
```sh
git clone https://github.com/NTU-RL2025-02/RL-final-project
cd RL-final-project

git submodule update --init --recursive

cd robomimic
conda create -n robomimic python=3.10
conda activate robomimic
pip install -e .
pip install robosuite==1.5.1

# in RL-final-project
cd thriftydagger
conda create -n thrifty-dagger python=3.10
conda activate thrifty-dagger
pip install -e .
# actually I work in robomimic env
```

## Robomimic
to run robomimic
```sh
#in robomimic/robomimic/scripts
python download_datasets.py \ --tasks square \ --dataset_types ph \ --hdf5_types low_dim
```
to train robomimic model
```sh
#in robomimic
python robomimic/scripts/train.py --config robomimic/exps/paper/core/square/ph/low_dim/bc_rnn.json
#if having an GPU: add --device cuda:0 at the back
```


### Thrifty DAgger
to run (still fixing)
```sh
# move the model trained from robomimic to scripts/expert_model first
python scripts/run_thriftydagger.py square_data --environment Square --no_render --gen_data
python scripts/run_thriftydagger.py test --no_render
# or
python scripts/run_thriftydagger.py square_thrifty --environment Square --no_render
```
and you will see numpy broadcast mismatch (51, )->(61, ) error

to record (still fixing)
```sh
python scripts/run_thriftydagger.py test --gen_data
```


## Flow
1. Train a good model with robomimic
