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