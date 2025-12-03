# RL-final-project

Main repository for the RL final project. It is a thin wrapper around two submodules:

- `robomimic/`: train imitation / offline RL policies on robosuite datasets.
- `thriftydagger/`: run ThriftyDAgger with a robomimic policy as the expert.

## Clone with submodules

```sh
git clone --recurse-submodules https://github.com/NTU-RL2025-02/RL-final-project
cd RL-final-project
```

If you already cloned without `--recurse-submodules`, run `git submodule update --init --recursive`.

To update submodules to their tracked branches: `git submodule update --init --remote --recursive`.

---

## One conda environment for everything

The two submodules are compatible in a single environment. ThriftyDAgger’s dependencies are a superset of robomimic’s, and both work with `robosuite==1.5.1` + `mujoco>=2.3` (tested on Python 3.10).

1. Create and activate the env

```sh
conda create -n rl-final python=3.10
conda activate rl-final
```

2. Install PyTorch for your platform (pick the right command from https://pytorch.org)

```sh
# Example: CPU / Apple Silicon
pip install torch torchvision
# Example: CUDA 12.1
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

3. Install project packages (from repo root)

```sh
pip install -e robomimic
pip install -e thriftydagger "robosuite==1.5.1" "mujoco>=2.3" "gymnasium>=0.29"
pip install -e thriftydagger_gym
```

4. (Optional) For headless rendering set `export MUJOCO_GL=egl` (Linux) or `export MUJOCO_GL=glfw` (Mac).

You can now import everything from the same shell: `python -c "import robomimic, thrifty, robosuite; print('ok')"` should print `ok`.

---

## Robomimic (training policies)

Download datasets:

```sh
cd robomimic/robomimic/scripts
python download_datasets.py --tasks square --dataset_types ph --hdf5_types low_dim
```

Train (add `--device cuda:0` if you have a GPU):

```sh
# in robomimic
python robomimic/scripts/train.py --config robomimic/exps/paper/core/square/ph/low_dim/bc_rnn.json
```

---

## ThriftyDAgger (using the same env)

1. Put a trained robomimic checkpoint under `thriftydagger/scripts/expert_model/` (see the default path in `thriftydagger/scripts/run_thriftydagger.py`).
2. Run from inside `thriftydagger`:

```sh
python scripts/run_thriftydagger.py square_data --environment Square --no_render --gen_data
python scripts/run_thriftydagger.py test --no_render
# or
python scripts/run_thriftydagger.py square_thrifty --environment Square --no_render
```

And you will see numpy broadcast mismatch `(51,) -> (61,)` error.

To record (still fixing):

```sh
python scripts/run_thriftydagger.py test --gen_data
```

---

## Flow

1. Train a good model with robomimic.

## Branch workflow

We follow a simple branch naming convention:

- `main`: stable branch for releases / final submissions.
- `dev/<who>-<topic>`: development branches for normal coding work.
  - e.g., `dev/aaron-env-wrapper`, `dev/mia-fix-robomimic-seed`
- `exp/<date>-<env>-<algo>-<who>-<tag>`: experiment branches for RL experiments.
  - e.g., `exp/20251123-square-bc-aaron-baseline`, `exp/20251124-square-thriftydagger-aaron-cbf-v1`
- `hotfix/<who>-<topic>`: quick fixes on top of `main` (urgent bugfixes, broken configs, etc.).
  - e.g., `hotfix/aaron-fix-square-config`, `hotfix/mia-robomimic-path-bug`

Rules:

- Only lowercase letters, digits, `-`, `_`, `.`, and `/` are allowed.
- Branch name must be one of:
  - `main`
  - `dev/<something>`
  - `exp/<something>`
  - `hotfix/<something>`

A GitHub Actions workflow (`.github/workflows/enforce-branch-name.yml`) will reject pushes / PRs from branches that do not follow this convention.
