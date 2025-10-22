#!/usr/bin/env bash

#SBATCH --partition=a100-40
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --job-name="sed-detector-classifier-eval"
#SBATCH --cpus-per-gpu=12
#SBATCH --output="/home/%u/logs/%A.log"

uv sync --extra eval
cd ~/representation-learning/representation_learning/evaluation
srun uv run evaluate_detector_classifier.py