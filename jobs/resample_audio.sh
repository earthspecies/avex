#!/usr/bin/env bash

#SBATCH --partition=cpu
#SBATCH --ntasks-per-node=1
#SBATCH --output="/home/%u/logs/%A.log"
#SBATCH --job-name="rl-resample-16khz"
#SBATCH --cpus-per-task=24

export GOOGLE_APPLICATION_CREDENTIALS=/home/david_earthspecies_org/.config/gcloud/application_default_credentials.json
export CLOUDPATHLIB_FORCE_OVERWRITE_FROM_CLOUD=1

cd ~/representation-learning
uv tool install keyring --with keyrings.google-artifactregistry-auth
export UV_PROJECT_ENVIRONMENT=/scratch/$USER/venvs/
uv sync

INPUT_FOLDER="/scratch-representation-learning/beans"
OUTPUT_FOLDER="/scratch-representation-learning/beans_16khz"  # Optional second argument
NUM_WORKERS=16 # Optional third argument, defaults to 16

echo "Starting audio resampling job..."
echo "Input folder: $INPUT_FOLDER"
echo "Output folder: $OUTPUT_FOLDER"
echo "Number of workers: $NUM_WORKERS"

# Build the command
# CMD="uv run scripts/resample_folder_16khz.py --input_folder ../data/voxaboxen --num_workers 30 --output_folder ../data/voxaboxen_16khz"
# CMD="uv run scripts/resample_folder_16khz.py --input_folder /scratch-representation-learning/beans --num_workers 30 --output_folder /scratch-representation-learning/beans_16khz"
# CMD="uv run scripts/resample_folder_16khz.py --input_folder ../data/birdset-test/ --num_workers 23 --output_folder ../data/birdset-test_16khz"
# CMD="uv run scripts/resample_folder_16khz.py --input_folder ../data/beans --num_workers 23 --output_folder ../data/beans_16khz_v2"
# CMD="uv run scripts/resample_folder_16khz.py --input_folder ../esp-data/v1 --num_workers 23 --output_folder ../esp-data/wabad_16khz"
# CMD="uv run scripts/resample_folder_16khz.py --input_folder ../data/beans/v0.1.0/raw/audio/hainan-gibbons-detection/ --num_workers 16 --output_folder ../data/beans_16khz_v2/v0.1.0/raw/audio/hainan-gibbons-detection/"

CMD="uv run scripts/resample_folder_16khz.py --input_folder /home/marius_miron_earthspecies_org/data/BirdSet/audio/birdset-test/SSW --num_workers 23 --output_folder ../data/SSW"

srun $CMD

echo "Audio resampling job completed!"