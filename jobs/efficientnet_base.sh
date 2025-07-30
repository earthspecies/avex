#!/usr/bin/env bash

#SBATCH --partition=a100-80
#SBATCH --ntasks-per-node=3
#SBATCH --gpus-per-node=3
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --job-name="rl-efffnet"
#SBATCH --cpus-per-gpu=12

cd ~/rep5
# export UV_PROJECT_ENVIRONMENT=/scratch/$USER/venvs/
# export UV_CACHE_DIR=/scratch/$USER/uv_cache/
export CLOUDPATHLIB_FORCE_OVERWRITE_FROM_CLOUD=1


uv sync
rm /scratch/david_earthspecies_org/job_tmpdir_11838/esp-ml-datasets/animalspeak/v0.1.0/raw/16KHz/animalspeak2_train.csv
srun uv run repr-learn train --config  configs/run_configs/aaai_train/sl_efficientnet_animalspeak_no_whales.yml