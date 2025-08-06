# ActiveRLHF

[![python](https://img.shields.io/badge/Python-~3.10-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![pytorch](https://img.shields.io/badge/PyTorch-~2.5-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![GitHub last commit (branch)](https://img.shields.io/github/last-commit/bkuen/active-rlhf/main)](https://github.com/bkuen/active-rlhf/pulse)
[![GitHub Issues or Pull Requests](https://img.shields.io/github/issues-closed/bkuen/active-rlhf)](https://github.com/bkuen/active-rlhf/pulls)

A framework to benchmark Active Learning methods inside Reinforcement Learning from Human Feedback.

This is the official repository for the thesis: 

> **On Leveraging Active Learning Techniques to Improve the Efficiency of RLHF**
>
> Reinforcement learning from human feedback (RLHF) augments many machine learning applications with human preferences, but is limited by high labeling costs and annotator biases. By relying on active learning (AL) techniques, one could develop sample selection strategies to improve data efficiency and mitigate such biases. The goal of this thesis is to reduce the number of preference queries in RLHF while maintaining or improving policy performance. We want to investigate whether AL techniques, such as uncertainty, diversity, or hybrid strategies, can enhance RLHF sample efficiency. Therefore, we extend a CleanRL-based PrefPPO pipeline with interchangeable query selectors. Specifically, we implement uncertainty and diversity-based clustering using both the state and the reward difference space, and present a novel hybrid selector combining multiple aspects using determinental point processes. We evaluate on four MuJoCo continuous control tasks under a fixed low query budget using a synthetic oracle. We found that AL methods outperform random sampling in 3/4 environments in terms of mean final reward, although we didn't achieve statistical significance. Our hybrid approach achieves top performance in HalfCheetah, but shows environment-dependent variability. Notably, diversity clustering using reward differences achieves the best trade-off between performance gains and computational overhead. Our results show that AL can substantially reduce feedback requirements for RLHF. However, gains vary across tasks due to varying state representation quality, hyperparameter sensitivity, and unmodeled dynamics discontinuities of the MuJoCo agent.

Cite this repository:

```bibtex
@masterthesis{kuen2025leveraging,
  author       = {Benjamin Kuen},
  title        = {On Leveraging Active Learning Techniques to Improve the Efficiency of RLHF},
  type         = {Bachelor’s Thesis},
  school       = {LMU Munich, Department of Mathematics, Informatics and Statistics, Institute of Informatics, Artificial Intelligence and Machine Learning (AIML)},
  address      = {Akademiestraße\,7, 80799 Munich, Germany},
  month        = aug,
  year         = {2025},
  date         = {2025-08-07},
}
```

## 💡 Getting Started

### Prerequisites

Ensure the following are installed on your system:

* [poetry](https://python-poetry.org/docs/#installation)
* Python 3.10+
* xvfb (`sudo apt install xvfb` on Debian-based systems)
* ffmpeg (`sudo apt install ffmpeg` on Debian-based systems)

### Installation

1. Install dependencies using poetry

```bash
poetry install
```

2. Run one experiment

```bash
python3 scripts/pref_ppo.sh
```

## 📈 Reproduce the results

### 🔋 Run experiments on LMU SLURM

1. Login on [CIP Pool via SSH](https://www.rz.ifi.lmu.de/infos/ssh_de.html)
2. Clone the repository into ``~/work`` directory

```bash
git clone git@github.com:bkuen/active-rlhf.git
```

3. Install a virtual environment. This will install miniconda into your home directory and all necessary dependencies on the machine (Python3, xvfbwrapper, ffmpeg, pipx, poetry)

```
sh ./active-rlhf/slurm/prepare_lmu_slurm.sh
```

4. Create a bash script ``run_experiment.sh`` inside ``~/work`` similar to those defined in ``./scripts``. You could adjust the hyperparameters as needed.

5. Create a slurm configuration file ``~/work/experiment.sbatch``. Make sure to adjust all absolute paths using the correct path to your home directory, e.g. ``/home/k/kuen``. Replace ``<EMAIL>`` with your email to receive notifications

```bash
#!/bin/bash
#SBATCH --array=1-10
#SBATCH --job-name=ActiveRLHF
#SBATCH --comment="Running benchmarks for RLHF+AL thesis"
#SBATCH --partition=NvidiaAll
#SBATCH --mail-user=<EMAIL>
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/c/<CIP_USER>/work/active-rlhf
#SBATCH --output=/home/c/<CIP_USER>/work/slurm/%A_%a.out
#SBATCH --error=/home/c/<CIP_USER>/work/slurm/%A_%a.err

# This script will run an active rlhf experiment on one environment for 10 seeds.

# This script assumes that you set up active-rlhf in your home directory and
# run the setup script once

echo 'activating virtual environment'
source /home/c/<CIP_USER>/miniconda3/bin/activate
conda activate py310activerlhf
which python3.10

source /home/c/<CIP_USER>/work/run_experiment.sh
```

6. Login into wandb. You could skip this if you solely rely on Tensorboard to track metrics. Make sure that your entity and project matches the ones defined as parameters in your ``run_experiment.sh`` script.

```bash
source ~/miniconda3/bin/activate
conda activate py310activerlhf
wandb login
```

7. Start SLURM job

```bash
sbatch ./experiment.sbatch
```

8. You could check the status of you job using

```bash
squeue -u <CIP_USER>
```

9. You could check the performance metrics either on [Weights & Biases](https://https://wandb.ai) or use tensorboard 

```bash
tensorboard --logdir ./active-rlhf/runs
```

### 🔎 Analyze results

We offer multiple scripts to export and analyze the experiments regarding descriptive statistics and significance tests.

> Note: Some of these scripts may require adjustment to support different experiment names and to map experiment names to human readable labels.

#### Export raw metrics from Wandb

```bash
python3 src/benchmark/export_metrics.py
```

#### Generate performance curves plot using Matplotlib

```bash
 python3 src/benchmark/generate_learning_curves.py \
  --input ./output/runs_walker_400.csv \
  --ooutput ./output/performance_walker_400.png \
  --smooth 20
```

#### Calculate descriptive statistics

```bash
 python3 src/benchmark/summarize_rl.py \
  --input ./output/runs_hopper_400.csv \
  --output ./output/summary_hopper_400.csv \
  --endpoint tail_mean \
  --tail-k 10
```

#### Run pairwise wilcoxon tests on both mean final reward (tail_mean) and AUC (area under curve)

```bash
 python3 src/benchmark/pairwise_wilcoxon_both.py \
  --input ./output/runs_walker_400.csv \
  --metric charts/episodic_return \
  --tail-k 10 \
  --endpoint-mode tail_mean \
  --ci both \
  --n-boot 10000 \
  --alpha 0.05 \
  --output ./output/statistical_significance_walker_400.csv
```

#### Export overhead statistics from Wandb

```bash
python3 src/benchmark/export_overhead.py \
  --project bkuen-ludwig-maximilianuniversity-of-munich/thesis-benchmark-halfcheetah \
  --output output/overhead_halfcheetah.csv \
  --include prefppo_random__ \
  --include prefppo_variquery__ \
  --include prefppo_duo_prio_v2__ \
  --include prefppo_hybrid_prio_u_v6__
```

> Note: Those statistics are only available if you track metrics on [Weights & Biases](https://https://wandb.ai) 

#### Generate latex tables

```bash
python3 src/benchmark/generate_summary_latex_table.py \
 --input ./output/summary_halfcheetah_400.csv \
 --label tab:halfcheetah-400 \
 --caption "HalfCheetah (400 Feedback)" \
 --method-order=Random,VARIQuery,DUO,Hybrid
```

```bash
 python3 src/benchmark/generate_significance_latex_table.py \
  --input output/statistical_significance_walker_400.csv \
  --output output/statistical_significance_table_walker_400.tex
```

```bash
 python3 src/benchmark/combine_overhead.py \
  --files output/overhead_walker_aggregated.csv output/overhead_halfcheetah_aggregated.csv output/overhead_ant_aggregated.csv output/overhead_hopper_aggregated.csv \
  --output output/overhead_combined_table.tex
```

### 🛠️ Perform Hyperparameter Tuning (Sweeps)

1. Init a new sweep using one of the sweep configurations inside ``./sweep``:

```bash
wandb sweep sweep/ppo_rewardnet_baseline_sweep_<ENVIRONMENT>.yaml
```

This command returns a sweep id, like ``7t2hhqxd``

2. Run 10 agents to perform sweep runs

```bash
wandb agent <WANDB_PROJECT>/<WANDB_ENTITY>/<SWEEP_ID> \
  --count 10
```

3. Wait for all agents to finish

3. Find best hyperparameters using

```bash
python3 src/benchmark/select_hparams_with_args.py \
  --entity bkuen-ludwig-maximilianuniversity-of-munich \
  --project prefppo-variquery-tuning \
  --sweep-id 8lxiqj6e \
  --top-pct 0.1 \
  --sweep-config sweep/variquery_vae_sweep_hopper.yaml \
  --output-dir output/sweep/8lxiqj6e
```

> Note: Make sure to replace all placeholder <PLACE_HOLDER> and adjust the commands accordingly.

