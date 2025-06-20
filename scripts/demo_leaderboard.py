#!/usr/bin/env python3
"""
Demonstration script for the enhanced experiment leaderboard with parameter extraction.
"""

import os
import sys

# Add the scripts directory to the path so we can import from experiment_leaderboard
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def demo_leaderboard_usage() -> None:
    """Demonstrate different ways to use the enhanced leaderboard."""

    print("🚀 Enhanced Experiment Leaderboard with Parameter Extraction")
    print("=" * 60)

    print("\n📋 Available Usage Examples:")

    print("\n1️⃣ Basic usage (no parameter extraction):")
    print("python3 scripts/experiment_leaderboard.py \\")
    print("  --csv_file evaluation_results/debug/metadata/evaluation_metadata.csv")

    print("\n2️⃣ Extract training parameters:")
    print("python3 scripts/experiment_leaderboard.py \\")
    print("  --csv_file evaluation_results/debug/metadata/evaluation_metadata.csv \\")
    print("  --training_params 'train_epochs,lr,batch_size,optimizer,weight_decay'")

    print("\n3️⃣ Extract evaluation config parameters:")
    print("python3 scripts/experiment_leaderboard.py \\")
    print("  --csv_file evaluation_results/debug/metadata/evaluation_metadata.csv \\")
    print("  --eval_config 'device,seed,num_workers,eval_modes'")

    print("\n4️⃣ Extract run config parameters:")
    print("python3 scripts/experiment_leaderboard.py \\")
    print("  --csv_file evaluation_results/debug/metadata/evaluation_metadata.csv \\")
    print(
        "  --run_config_params "
        "'run_name,wandb_project,logging,label_type,loss_function'"
    )

    print("\n5️⃣ Extract parameters from all config types:")
    print("python3 scripts/experiment_leaderboard.py \\")
    print("  --csv_file evaluation_results/debug/metadata/evaluation_metadata.csv \\")
    print("  --eval_config 'device,seed' \\")
    print("  --training_params 'train_epochs,lr,batch_size' \\")
    print("  --run_config_params 'run_name,wandb_project'")

    print("\n6️⃣ With custom server settings:")
    print("python3 scripts/experiment_leaderboard.py \\")
    print("  --csv_file evaluation_results/debug/metadata/evaluation_metadata.csv \\")
    print("  --training_params 'train_epochs,lr' \\")
    print("  --host 0.0.0.0 \\")
    print("  --port 8080 \\")
    print("  --share")

    print("\n" + "=" * 60)
    print("✨ Key Features:")
    print("• JSON configuration columns are automatically hidden from display")
    print("• Extracted parameters are added as new columns for easy filtering/sorting")
    print("• Robust error handling for missing or malformed JSON")
    print("• Flexible field selection for each config type")
    print("• Maintains backward compatibility with existing CSV files")

    print("\n🔧 Available Parameters by Config Type:")

    print("\n📊 Evaluation Config (--eval_config):")
    print("  device, seed, num_workers, eval_modes, overwrite_embeddings")

    print("\n🏋️ Training Parameters (--training_params):")
    print("  train_epochs, lr, batch_size, optimizer, weight_decay,")
    print("  amp, amp_dtype, log_steps, gradient_checkpointing")

    print("\n⚙️ Run Config Parameters (--run_config_params):")
    print("  run_name, wandb_project, logging, label_type, multilabel,")
    print("  loss_function, device, seed, num_workers, distributed, debug_mode")

    print("\n" + "=" * 60)
    print("💡 Tips:")
    print("• Use comma-separated field names without spaces")
    print("• Fields are case-sensitive and must match the JSON structure")
    print("• Missing fields will be populated with None values")
    print(
        "• The leaderboard will automatically filter and sort by extracted parameters"
    )


if __name__ == "__main__":
    demo_leaderboard_usage()
