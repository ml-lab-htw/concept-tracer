#!/bin/bash

#SBATCH --job-name=concept_run
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=10-00:00:00
#SBATCH --cpus-per-gpu=32
#SBATCH --mem-per-gpu=32G
#SBATCH --qos=normal

source $HOME/.bashrc
conda activate concept_env

export TABPFN_DISABLE_TELEMETRY=1

srun python -m concept_tracer.calculations
