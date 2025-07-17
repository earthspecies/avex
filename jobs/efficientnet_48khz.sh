#!/usr/bin/env bash

#SBATCH --partition=a100-40
#SBATCH --gpus=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --job-name="rl-effnet-48khz"
#SBATCH --cpus-per-gpu=12

# uv tool install keyring --with keyrings.google-artifactregistry-auth
# Fix keyring installation - install ALL needed packages
uv tool install keyring --with keyrings.google-artifactregistry-auth --with keyrings.alt --force

# Test if the Google AR backend is available
# echo "=== Testing keyring backends ==="
# uv tool run keyring --list-backends

# Test authentication with correct command (use 'creds' mode, no username needed)
# echo "=== Testing Google AR authentication ==="
# uv tool run keyring --mode creds get https://us-central1-python.pkg.dev/okapi-274503/esp-pypi/simple/

cd ~/representation-learning
uv sync
srun uv run representation_learning/run_train.py --config configs/run_configs/efficientnet_48khz.yml
