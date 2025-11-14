#!/usr/bin/env bash

#SBATCH --partition=a100-40
#SBATCH --gpus=0
#SBATCH --ntasks=1
#SBATCH --output="/home/%u/logs/explore_scratch_%A.log"
#SBATCH --error="/home/%u/logs/explore_scratch_%A.err"
#SBATCH --job-name="explore-scratch"
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --nodelist=slurm-8x-a100-40gb-2

set -e

SCRATCH_DIR="/scratch-representation-learning"

echo "=========================================="
echo "Exploring: $SCRATCH_DIR"
echo "Date: $(date)"
echo "=========================================="
echo ""

# Check if directory exists
if [ ! -d "$SCRATCH_DIR" ]; then
    echo "ERROR: Directory $SCRATCH_DIR does not exist!"
    exit 1
fi

echo "Directory exists: ✓"
echo ""

# List top-level contents
echo "Top-level contents:"
echo "----------------------------------------"
ls -lah "$SCRATCH_DIR" | head -30
echo ""

# Check for beans directory
if [ -d "$SCRATCH_DIR/beans" ]; then
    echo "Found beans directory:"
    echo "----------------------------------------"
    ls -lah "$SCRATCH_DIR/beans" | head -20
    echo ""

    # Check for version directories
    if [ -d "$SCRATCH_DIR/beans/v0.1.0" ]; then
        echo "Found beans/v0.1.0 directory:"
        echo "----------------------------------------"
        ls -lah "$SCRATCH_DIR/beans/v0.1.0" | head -20
        echo ""

        # Check for raw directory
        if [ -d "$SCRATCH_DIR/beans/v0.1.0/raw" ]; then
            echo "Found beans/v0.1.0/raw directory:"
            echo "----------------------------------------"
            ls -lah "$SCRATCH_DIR/beans/v0.1.0/raw" | head -20
            echo ""

            # Check for audio directory
            if [ -d "$SCRATCH_DIR/beans/v0.1.0/raw/audio" ]; then
                echo "Found beans/v0.1.0/raw/audio directory:"
                echo "----------------------------------------"
                ls -lah "$SCRATCH_DIR/beans/v0.1.0/raw/audio" | head -20
                echo ""

                # Check for esc50 directory
                if [ -d "$SCRATCH_DIR/beans/v0.1.0/raw/audio/esc50" ]; then
                    echo "Found esc50 directory:"
                    echo "----------------------------------------"
                    ls -lah "$SCRATCH_DIR/beans/v0.1.0/raw/audio/esc50" | head -20
                    echo ""
                    echo "Total files in esc50: $(find "$SCRATCH_DIR/beans/v0.1.0/raw/audio/esc50" -type f | wc -l)"
                fi

                # Check for esc50-all directory
                if [ -d "$SCRATCH_DIR/beans/v0.1.0/raw/audio/esc50-all" ]; then
                    echo "Found esc50-all directory:"
                    echo "----------------------------------------"
                    ls -lah "$SCRATCH_DIR/beans/v0.1.0/raw/audio/esc50-all" | head -20
                    echo ""
                    echo "Total files in esc50-all: $(find "$SCRATCH_DIR/beans/v0.1.0/raw/audio/esc50-all" -type f | wc -l)"
                fi
            fi
        fi
    fi
fi

# Check for beans_16khz_v3 directory
if [ -d "$SCRATCH_DIR/beans_16khz_v3" ]; then
    echo "Found beans_16khz_v3 directory:"
    echo "----------------------------------------"
    ls -lah "$SCRATCH_DIR/beans_16khz_v3" | head -20
    echo ""

    if [ -d "$SCRATCH_DIR/beans_16khz_v3/v0.1.0" ]; then
        echo "Found beans_16khz_v3/v0.1.0 directory:"
        echo "----------------------------------------"
        ls -lah "$SCRATCH_DIR/beans_16khz_v3/v0.1.0" | head -20
        echo ""

        if [ -d "$SCRATCH_DIR/beans_16khz_v3/v0.1.0/raw" ]; then
            echo "Found beans_16khz_v3/v0.1.0/raw directory:"
            echo "----------------------------------------"
            ls -lah "$SCRATCH_DIR/beans_16khz_v3/v0.1.0/raw" | head -20
            echo ""

            if [ -d "$SCRATCH_DIR/beans_16khz_v3/v0.1.0/raw/audio" ]; then
                echo "Found beans_16khz_v3/v0.1.0/raw/audio directory:"
                echo "----------------------------------------"
                ls -lah "$SCRATCH_DIR/beans_16khz_v3/v0.1.0/raw/audio" | head -20
                echo ""
            fi
        fi
    fi
fi

# Summary
echo "=========================================="
echo "Summary:"
echo "=========================================="
echo "Directory structure checked: $SCRATCH_DIR"
echo "Exploration complete at: $(date)"

