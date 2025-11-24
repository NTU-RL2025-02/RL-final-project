# RL-final-project

Repository for RL final project – currently a blank template.

## Clone with submodules

This repository uses Git submodules for `robomimic` and `thriftydagger`.

Recommended way (clone + init submodules in one step):

```sh
git clone --recurse-submodules https://github.com/NTU-RL2025-02/RL-final-project
cd RL-final-project
```

If you already cloned without `--recurse-submodules`:

```sh
git clone https://github.com/NTU-RL2025-02/RL-final-project
cd RL-final-project
git submodule update --init --recursive
```

To update submodules to the latest `master` (of each submodule):

```sh
git submodule update --init --remote --recursive
```

---

## If you want to use new robosuite + mujoco, please

```sh
cd robomimic
conda create -n robomimic python=3.10
conda activate robomimic
pip install -e .
pip install robosuite==1.5.1
```

```sh
# in RL-final-project
cd thriftydagger
pip install -e .
```

---

## Robomimic

To run robomimic:

```sh
# in robomimic/robomimic/scripts
python download_datasets.py \
  --tasks square \
  --dataset_types ph \
  --hdf5_types low_dim
```

To train robomimic model:

```sh
# in robomimic
python robomimic/scripts/train.py --config robomimic/exps/paper/core/square/ph/low_dim/bc_rnn.json
# if having a GPU: add --device cuda:0 at the back
```

---

## Thrifty DAgger

To run (still fixing):

```sh
# move the model trained from robomimic to scripts/expert_model first
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

--

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
