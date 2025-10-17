#!/usr/bin/env python3
"""
Integration test script for the new flexible probing system.
This script runs a subset of the flexible probing tests to verify functionality.
"""

import sys
import tempfile
import time
from pathlib import Path


def test_flexible_probing_integration() -> None:
    """Test the flexible probing system with a minimal configuration.

    Returns:
        None
    """

    print("🧪 Testing Flexible Probing Integration")
    print("=" * 50)

    # Create temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_output_dir = Path(temp_dir)
        print(f"📁 Using temporary output directory: {temp_output_dir}")

        # Path to our test configuration
        config_path = Path(
            "configs/evaluation_configs/flexible_probing_minimal_test.yml"
        )

        if not config_path.exists():
            print(f"❌ Test configuration not found: {config_path}")
            return False

        print(f"📋 Using test configuration: {config_path}")

        # Prepare patches for the test
        patches = (
            f"save_dir={temp_output_dir}",
            "device=cpu",
            "seed=42",
            "training_params.train_epochs=1",
            "training_params.batch_size=2",
        )

        print(f"🔧 Applying patches: {patches}")

        try:
            print("\n🚀 Starting flexible probing evaluation...")
            start_time = time.time()

            # Import and run the main function
            from representation_learning.run_evaluate import main

            # Run the evaluation
            main(config_path, patches)

            end_time = time.time()
            duration = end_time - start_time

            print(f"✅ Evaluation completed in {duration:.2f} seconds")

            # Check for output files
            print("\n📊 Checking output files...")

            # Look for summary CSV files
            summary_csvs = list(temp_output_dir.rglob("*summary*.csv"))
            if summary_csvs:
                print(f"✅ Found {len(summary_csvs)} summary CSV files:")
                for csv_file in summary_csvs:
                    print(f"   📄 {csv_file.relative_to(temp_output_dir)}")
            else:
                print("⚠️  No summary CSV files found")

            # Look for evaluation metadata
            metadata_files = list(temp_output_dir.rglob("*metadata*.json"))
            if metadata_files:
                print(f"✅ Found {len(metadata_files)} metadata files:")
                for meta_file in metadata_files:
                    print(f"   📄 {meta_file.relative_to(temp_output_dir)}")
            else:
                print("⚠️  No metadata files found")

            # Look for experiment directories
            experiment_dirs = [d for d in temp_output_dir.iterdir() if d.is_dir()]
            if experiment_dirs:
                print(f"✅ Found {len(experiment_dirs)} experiment directories:")
                for exp_dir in experiment_dirs:
                    print(f"   📁 {exp_dir.name}")

                    # Check for embedding files
                    embedding_files = list(exp_dir.rglob("*.h5"))
                    if embedding_files:
                        print(f"      📊 {len(embedding_files)} embedding files")

                    # Check for log files
                    log_files = list(exp_dir.rglob("*.log"))
                    if log_files:
                        print(f"      📝 {len(log_files)} log files")

            print("\n🎉 Flexible probing integration test completed successfully!")
            return True

        except Exception as e:
            print(f"\n❌ Integration test failed: {e}")
            import traceback

            traceback.print_exc()
            return False


if __name__ == "__main__":
    success = test_flexible_probing_integration()
    sys.exit(0 if success else 1)
