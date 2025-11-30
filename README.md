# We Still Don’t Understand High-Dimensional Bayesian Optimization

This repository contains the code to reproduce the linear model (after spherical mapping) proposed in the paper _We Still Don’t Understand High-Dimensional Bayesian Optimization_. 

### Table of Contents
*   [Getting Started](#getting-started)
*   [Running Experiments](#running-experiments)
*   [Advanced Usage](#advanced-usage)
*   [Citation](#citation)

## Getting Started

We recommend using [uv](https://docs.astral.sh/uv/) for dependency management. It automatically handles Python versions and virtual environment creation.

To get started, clone the repository and sync the environment:
```
git clone https://github.com/colmont/linear-bo.git
cd linear-bo
uv sync
source .venv/bin/activate
```
Alternatively, for other dependency managers, please see the `requirements.txt` file.

## Running Experiments

Experiments can be run using the `main.py` script. You must specify a benchmark to run.

**Basic Command**
```
python main.py benchmark=<benchmark_name>
```

*   To see a list of available benchmarks, run `python main.py`.
*   Adding `seed=<number>` is recommended for reproducibility.

**Configuration Overrides**
This project uses [Hydra](https://hydra.cc/) for configuration management. You can override any option defined in `configs/default.yaml` directly from the command line.

```
# Example: override the total number of iterations for the MOPTA08 benchmark
python main.py benchmark=mopta08 seed=0 benchmark.n_tot=200
```

**MuJoCo Benchmarks**
Running MuJoCo benchmarks requires a specific `sif` file from [this repository](https://github.com/DonneyF/mujoco-v2-for-global-optimization). After obtaining the file, you must specify its path in `configs/default.yaml`.

## Citation
If you use this code in your research, please cite the following paper:

```
@article{
  doumont2025linear,
  title={We Still Don’t Understand High-Dimensional Bayesian Optimization},
  author={Doumont, Colin and Fan, Donney and Maus, Natalie and Gardner, Jacob R. and Moss, Henry and Pleiss, Geoff},
  journal={arXiv preprint},
  year={2025}
}
```