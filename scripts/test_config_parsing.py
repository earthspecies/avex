#!/usr/bin/env python3
"""
Test script to demonstrate the config parsing functionality.
"""

import os
import sys
import tempfile

import pandas as pd
from experiment_leaderboard import (
    extract_config_parameters,
    load_data,
    parse_config_fields,
    prepare_data_for_leaderboard,
)

# Add the scripts directory to the path so we can import the function
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_parse_config_fields() -> bool:
    """Test the parse_config_fields function with example data.

    Returns:
        bool: True if all tests pass, False otherwise
    """
    # Example training_params JSON string from the user's data
    training_params = (
        '{"model_spec": {"name": "efficientnet", "pretrained": true, '
        '"device": "cuda", "audio_config": {"sample_rate": 16000, "n_fft": 800, '
        '"hop_length": 160, "win_length": 800, "window": "hann", "n_mels": 128, '
        '"representation": "mel_spectrogram", "normalize": false, '
        '"target_length_seconds": 10, "window_selection": "start", "center": true}, '
        '"text_model_name": null, "projection_dim": null, "temperature": null, '
        '"eat_cfg": null, "pretraining_mode": null, "handle_padding": null, '
        '"efficientnet_variant": "b0"}, "training_params": {"train_epochs": 30, '
        '"lr": 0.0003, "batch_size": 64, "optimizer": "adamw", "weight_decay": 0.01, '
        '"adam_betas": null, "amp": false, "amp_dtype": "bf16", "log_steps": 100, '
        '"gradient_checkpointing": false}, "dataset_config": '
        '"configs/data_configs/data_base.yml", "output_dir": '
        '"./runs/efficientnet_single_26_05", "preprocessing": null, "sr": 16000, '
        '"logging": "mlflow", "label_type": "supervised", '
        '"resume_from_checkpoint": null, "distributed": false, '
        '"distributed_backend": "nccl", "distributed_port": 29500, '
        '"augmentations": [], "loss_function": "cross_entropy", "multilabel": false, '
        '"device": "cuda", "seed": 42, "num_workers": 10, '
        '"run_name": "efficientnet_base", "wandb_project": "representation_learning", '
        '"scheduler": {"name": "cosine", "warmup_steps": 4000, "min_lr": 1e-06}, '
        '"debug_mode": false}'
    )

    # Example eval_config (empty for this test)
    eval_config = ""

    # Example run_config_params (empty for this test)
    run_config_params = ""

    # Fields to extract from training_params
    training_param_fields = ["train_epochs", "lr", "batch_size", "optimizer"]

    # Test the function
    result = parse_config_fields(
        eval_config=eval_config,
        training_params=training_params,
        run_config_params=run_config_params,
        eval_config_fields=None,
        training_param_fields=training_param_fields,
        run_config_fields=None,
    )

    print("Extracted parameters:")
    for key, value in result.items():
        print(f"  {key}: {value}")

    # Verify expected results
    expected = {
        "training_train_epochs": 30,
        "training_lr": 0.0003,
        "training_batch_size": 64,
        "training_optimizer": "adamw",
    }

    print("\nExpected results:")
    for key, value in expected.items():
        print(f"  {key}: {value}")

    # Check if results match expected
    success = True
    for key, expected_value in expected.items():
        if key not in result:
            print(f"❌ Missing key: {key}")
            success = False
        elif result[key] != expected_value:
            print(
                f"❌ Value mismatch for {key}: expected {expected_value}, "
                f"got {result[key]}"
            )
            success = False
        else:
            print(f"✅ {key}: {result[key]}")

    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n❌ Some tests failed!")

    return success


def test_extract_config_parameters() -> None:
    """Test the extract_config_parameters function with sample DataFrame."""
    # Create sample DataFrame
    data = {
        "timestamp": ["2025-06-20T11:21:35.975258", "2025-06-20T11:21:48.674987"],
        "dataset_name": ["egyptian_fruit_bats", "egyptian_fruit_bats"],
        "experiment_name": ["efficientnet_pretrained_test", "efficientnet_frozen_test"],
        "eval_config": [
            '{"device": "cuda", "seed": 42, "num_workers": 8}',
            '{"device": "cuda", "seed": 42, "num_workers": 8}',
        ],
        "training_params": [
            '{"train_epochs": 30, "lr": 0.0003, "batch_size": 64}',
            '{"train_epochs": 30, "lr": 0.0003, "batch_size": 64}',
        ],
        "run_config_params": [
            '{"run_name": "efficientnet_base", '
            '"wandb_project": "representation_learning"}',
            '{"run_name": "efficientnet_base", '
            '"wandb_project": "representation_learning"}',
        ],
    }

    df = pd.DataFrame(data)

    print("Original DataFrame:")
    print(df)
    print(f"\nOriginal columns: {list(df.columns)}")

    # Extract parameters
    result_df = extract_config_parameters(
        df,
        eval_config_fields=["device", "seed"],
        training_param_fields=["train_epochs", "lr"],
        run_config_fields=["run_name", "wandb_project"],
    )

    print("\nDataFrame after parameter extraction:")
    print(result_df)
    print(f"\nNew columns: {list(result_df.columns)}")


def test_load_data_with_config_columns() -> None:
    """
    Test that load_data properly includes config columns when parameters are requested.
    """
    # Create a temporary CSV file with config columns
    csv_content = (
        "timestamp,dataset_name,experiment_name,checkpoint_name,"
        "retrieval_roc_auc,retrieval_precision_at_1,test_accuracy,"
        "test_balanced_accuracy,test_roc_auc,test_multiclass_f1,test_map,"
        "eval_config,training_params,run_config_params\n"
        "2025-06-20T11:21:35.975258,egyptian_fruit_bats,efficientnet_pretrained_test,"
        "final_model.pt,0.5732174218313791,0.2205,0.286,0.286,0.286,0.286,0.286,"
        '"{""device"": ""cuda"", ""seed"": 42}",'
        '"{""train_epochs"": 30, ""lr"": 0.0003}",'
        '"{""run_name"": ""efficientnet_base""}"\n'
        "2025-06-20T11:21:48.674987,egyptian_fruit_bats,efficientnet_frozen_test,"
        "None,0.6012269332774985,0.1935,0.178,0.178,0.178,0.178,0.178,"
        '"{""device"": ""cuda"", ""seed"": 42}",'
        '"{""train_epochs"": 30, ""lr"": 0.0003}",'
        '"{""run_name"": ""efficientnet_base""}"\n'
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        temp_csv_path = f.name

    try:
        print("Testing load_data with config parameter extraction:")

        # Test without config parameters (should not include config columns)
        df_no_config = load_data(temp_csv_path)
        print(f"\nColumns without config extraction: {list(df_no_config.columns)}")
        print(f"Number of columns: {len(df_no_config.columns)}")

        # Test with config parameters (should include config columns)
        df_with_config = load_data(
            temp_csv_path,
            eval_config_fields=["device", "seed"],
            training_param_fields=["train_epochs", "lr"],
            run_config_fields=["run_name"],
        )
        print(f"\nColumns with config extraction: {list(df_with_config.columns)}")
        print(f"Number of columns: {len(df_with_config.columns)}")

        # Verify that config columns are present
        config_columns = ["eval_config", "training_params", "run_config_params"]
        for col in config_columns:
            if col in df_with_config.columns:
                print(f"✓ Config column '{col}' is present")
            else:
                print(f"✗ Config column '{col}' is missing")

        # Verify that extracted parameters are present and populated
        extracted_columns = [
            "eval_device",
            "eval_seed",
            "training_train_epochs",
            "training_lr",
            "run_run_name",
        ]
        for col in extracted_columns:
            if col in df_with_config.columns:
                # Check if the column has non-None values
                non_null_count = df_with_config[col].notna().sum()
                print(
                    f"✓ Extracted column '{col}' is present with "
                    f"{non_null_count} non-null values"
                )
            else:
                print(f"✗ Extracted column '{col}' is missing")

        # Verify that all requested fields are present (even if empty)
        requested_fields = [
            "eval_device",
            "eval_seed",
            "training_train_epochs",
            "training_lr",
            "run_run_name",
        ]
        for field in requested_fields:
            if field in df_with_config.columns:
                print(f"✓ Requested field '{field}' is present in DataFrame")
            else:
                print(f"✗ Requested field '{field}' is missing from DataFrame")

    finally:
        # Clean up temporary file
        os.unlink(temp_csv_path)


def test_prepare_data_for_leaderboard() -> None:
    """Test that prepare_data_for_leaderboard hides JSON config columns."""
    # Create sample DataFrame with config columns and extracted parameters
    data = {
        "timestamp": ["2025-06-20T11:21:35.975258"],
        "dataset_name": ["egyptian_fruit_bats"],
        "experiment_name": ["efficientnet_pretrained_test"],
        "checkpoint_name": ["final_model.pt"],
        "retrieval_roc_auc": [0.5732174218313791],
        "eval_config": ['{"device": "cuda", "seed": 42}'],
        "training_params": ['{"train_epochs": 30, "lr": 0.0003}'],
        "run_config_params": ['{"run_name": "efficientnet_base"}'],
        "eval_device": ["cuda"],
        "eval_seed": [42],
        "training_train_epochs": [30],
        "training_lr": [0.0003],
        "run_run_name": ["efficientnet_base"],
    }

    df = pd.DataFrame(data)

    print("Original DataFrame columns:")
    print(list(df.columns))

    # Test prepare_data_for_leaderboard
    display_df = prepare_data_for_leaderboard(df)

    print("\nDisplay DataFrame columns:")
    print(list(display_df.columns))

    # Verify config columns are hidden
    config_columns = ["eval_config", "training_params", "run_config_params"]
    for col in config_columns:
        if col in display_df.columns:
            print(f"✗ Config column '{col}' should be hidden but is present")
        else:
            print(f"✓ Config column '{col}' is properly hidden")

    # Verify extracted parameters are still visible
    extracted_columns = [
        "eval_device",
        "eval_seed",
        "training_train_epochs",
        "training_lr",
        "run_run_name",
    ]
    for col in extracted_columns:
        if col in display_df.columns:
            print(f"✓ Extracted column '{col}' is visible")
        else:
            print(f"✗ Extracted column '{col}' should be visible but is missing")


def test_pre_population_approach() -> None:
    """Test that columns are pre-populated
    as empty and then filled with extracted values."""
    # Create sample DataFrame without config columns
    data = {
        "timestamp": ["2025-06-20T11:21:35.975258"],
        "dataset_name": ["egyptian_fruit_bats"],
        "experiment_name": ["efficientnet_pretrained_test"],
        "checkpoint_name": ["final_model.pt"],
        "retrieval_roc_auc": [0.5732174218313791],
    }

    df = pd.DataFrame(data)

    print("Original DataFrame:")
    print(f"Columns: {list(df.columns)}")
    print(f"Shape: {df.shape}")

    # Test extract_config_parameters with pre-populated columns
    # First, manually pre-populate the columns as empty
    eval_config_fields = ["device", "seed"]
    training_param_fields = ["train_epochs", "lr"]
    run_config_fields = ["run_name"]

    # Pre-populate columns
    for field in eval_config_fields:
        df[f"eval_{field}"] = None
    for field in training_param_fields:
        df[f"training_{field}"] = None
    for field in run_config_fields:
        df[f"run_{field}"] = None

    print("\nAfter pre-population:")
    print(f"Columns: {list(df.columns)}")
    print(f"Shape: {df.shape}")

    # Add config columns with sample data
    df["eval_config"] = '{"device": "cuda", "seed": 42}'
    df["training_params"] = '{"train_epochs": 30, "lr": 0.0003}'
    df["run_config_params"] = '{"run_name": "efficientnet_base"}'

    print("\nAfter adding config columns:")
    print(f"Columns: {list(df.columns)}")

    # Now extract parameters
    result_df = extract_config_parameters(
        df,
        eval_config_fields=eval_config_fields,
        training_param_fields=training_param_fields,
        run_config_fields=run_config_fields,
    )

    print("\nAfter parameter extraction:")
    print(f"Columns: {list(df.columns)}")

    # Check that pre-populated columns are now filled
    expected_columns = [
        "eval_device",
        "eval_seed",
        "training_train_epochs",
        "training_lr",
        "run_run_name",
    ]
    for col in expected_columns:
        if col in result_df.columns:
            value = result_df[col].iloc[0]
            print(f"✓ Column '{col}' = {value}")
        else:
            print(f"✗ Column '{col}' is missing")


def test_escaped_json_parsing() -> None:
    """Test parsing of JSON with escaped quotes (actual format from CSV)."""
    # This is the actual format from your CSV - with double quotes
    actual_csv_training_params = (
        '{"train_epochs": 30, "lr": 0.0003, "batch_size": 64, "optimizer": "adamw", '
        '"weight_decay": 0.01, "adam_betas": null, "amp": false, "amp_dtype": "bf16", '
        '"log_steps": 100, "gradient_checkpointing": false}'
    )

    print("Testing parse_config_fields with actual CSV JSON format:")

    # Test with the actual CSV format
    result = parse_config_fields(
        "",  # empty eval_config for this test
        actual_csv_training_params,
        "",  # empty run_config for this test
        eval_config_fields=[],
        training_param_fields=["train_epochs", "lr", "batch_size", "optimizer"],
        run_config_fields=[],
    )

    print("\nExtracted parameters from actual CSV JSON:")
    for key, value in result.items():
        print(f"  {key}: {value}")

    # Verify specific values
    expected_values = {
        "training_train_epochs": 30,
        "training_lr": 0.0003,
        "training_batch_size": 64,
        "training_optimizer": "adamw",
    }

    print("\nVerifying extracted values:")
    for key, expected_value in expected_values.items():
        actual_value = result.get(key)
        if actual_value == expected_value:
            print(f"✓ {key}: {actual_value} (correct)")
        else:
            print(f"✗ {key}: expected {expected_value}, got {actual_value}")

    # Test with a more complex example from the actual CSV
    actual_eval_config = (
        '{"experiments": [{"run_name": "efficientnet_pretrained_test", '
        '"run_config": "configs/run_configs/efficientnet_base_beans.yml", '
        '"pretrained": false, "layers": "model.avgpool", '
        '"checkpoint_path": "runs/efficientnet_base_test/2025-06-16_07-08-36/'
        'checkpoints/final_model.pt", "frozen": true}], "device": "cuda", '
        '"seed": 42, "num_workers": 8, "eval_modes": ["linear_probe", "retrieval"], '
        '"overwrite_embeddings": false}'
    )

    print("\n\nTesting with actual eval_config from CSV:")
    eval_result = parse_config_fields(
        actual_eval_config,
        "",
        "",
        eval_config_fields=["device", "seed", "num_workers"],
        training_param_fields=[],
        run_config_fields=[],
    )

    print("\nExtracted eval_config parameters:")
    for key, value in eval_result.items():
        print(f"  {key}: {value}")

    # Verify eval_config values
    eval_expected = {"eval_device": "cuda", "eval_seed": 42, "eval_num_workers": 8}

    print("\nVerifying eval_config values:")
    for key, expected_value in eval_expected.items():
        actual_value = eval_result.get(key)
        if actual_value == expected_value:
            print(f"✓ {key}: {actual_value} (correct)")
        else:
            print(f"✗ {key}: expected {expected_value}, got {actual_value}")


if __name__ == "__main__":
    print("Testing config parsing functionality...\n")

    test_parse_config_fields()
    print("\n" + "=" * 50 + "\n")
    test_extract_config_parameters()
    print("\n" + "=" * 50 + "\n")
    test_load_data_with_config_columns()
    print("\n" + "=" * 50 + "\n")
    test_pre_population_approach()
    print("\n" + "=" * 50 + "\n")
    test_escaped_json_parsing()
    print("\n" + "=" * 50 + "\n")
    test_prepare_data_for_leaderboard()

    print("\nAll tests completed!")
