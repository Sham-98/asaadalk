#!/bin/bash
#SBATCH --job-name=pytorch_gpu
#SBATCH --account=project_2001382
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

module --force purge
#module purge

# Choose exactly one Python module at a time:
# module load python-data    # generic Python (CPU-only)
module load pytorch/2.6     # GPU calculations

source /projappl/project_2001382/asaadalk/venv_sham/bin/activate
export HF_TOKEN=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
srun /projappl/project_2001382/asaadalk/venv_sham/bin/python -u testSAM.py

