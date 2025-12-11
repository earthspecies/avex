#!/usr/bin/env python
"""Test script to validate the evaluation pipeline components.

This script tests:
1. Model loading (BEATs/OpenBEATs)
2. Embedding extraction
3. Embedding dataset structure
4. Retrieval and clustering metrics

Run with: uv run python scripts/test_evaluation_pipeline.py
"""

import logging
import sys
import tempfile
from pathlib import Path

import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def test_embedding_dataset_structure():
    """Test that EmbeddingDataset returns correct structure."""
    print("\n" + "=" * 60)
    print("TEST: EmbeddingDataset structure")
    print("=" * 60)

    from representation_learning.evaluation.embedding_utils import EmbeddingDataset

    # Test multi-layer case
    embeddings = {
        "backbone": torch.randn(10, 1024),
        "layer_12": torch.randn(10, 768),
    }
    labels = torch.randint(0, 5, (10,))

    dataset = EmbeddingDataset(embeddings, labels)
    sample = dataset[0]

    print(f"Sample keys: {list(sample.keys())}")
    print(f"Sample key order: {list(sample.keys())}")

    # Check if 'label' is incorrectly being selected as "last layer"
    last_key = list(sample.keys())[-1]
    print(f"Last key in sample dict: '{last_key}'")

    if last_key == "label":
        print(
            "❌ BUG CONFIRMED: 'label' is the last key, would be incorrectly selected as embedding layer!"
        )
        return False
    else:
        print("✅ PASS: 'label' is not the last key")
        return True


def test_embedding_layer_selection():
    """Test the layer selection logic from run_evaluate.py."""
    print("\n" + "=" * 60)
    print("TEST: Layer selection logic (fixed behavior)")
    print("=" * 60)

    from representation_learning.evaluation.embedding_utils import EmbeddingDataset

    # Simulate what happens in run_evaluate.py
    embeddings = {"backbone": torch.randn(10, 1024)}
    labels = torch.randint(0, 5, (10,))

    dataset = EmbeddingDataset(embeddings, labels)
    sample = dataset[0]

    # Fixed logic from run_evaluate.py - excludes 'label' key
    if isinstance(sample, dict):
        embedding_keys = [k for k in sample.keys() if k != "label"]
        if not embedding_keys:
            print("❌ FAIL: No embedding layers found")
            return False
        last_layer_name = embedding_keys[-1]
        print(f"Fixed logic selects: '{last_layer_name}'")

        if last_layer_name == "label":
            print("❌ BUG: Would try to use 'label' as embeddings!")
            return False

        test_embeds = torch.stack(
            [dataset[i][last_layer_name] for i in range(len(dataset))]
        )
        print(f"Embeddings shape: {test_embeds.shape}")
        print("✅ PASS: Layer selection works correctly")
        return True

    print("✅ PASS: Layer selection works correctly")
    return True


def test_fixed_layer_selection():
    """Test the fixed layer selection logic."""
    print("\n" + "=" * 60)
    print("TEST: Fixed layer selection logic")
    print("=" * 60)

    from representation_learning.evaluation.embedding_utils import EmbeddingDataset

    embeddings = {"backbone": torch.randn(10, 1024)}
    labels = torch.randint(0, 5, (10,))

    dataset = EmbeddingDataset(embeddings, labels)
    sample = dataset[0]

    # Fixed logic: exclude 'label' from layer selection
    if isinstance(sample, dict):
        embedding_keys = [k for k in sample.keys() if k != "label"]
        if embedding_keys:
            last_layer_name = embedding_keys[-1]
            print(f"Fixed logic selects: '{last_layer_name}'")

            test_embeds = torch.stack(
                [dataset[i][last_layer_name] for i in range(len(dataset))]
            )
            print(f"Embeddings shape: {test_embeds.shape}")
            print("✅ PASS: Fixed layer selection works!")
            return True
        else:
            print("❌ No embedding layers found")
            return False

    return True


def test_retrieval_metrics():
    """Test retrieval metric computation."""
    print("\n" + "=" * 60)
    print("TEST: Retrieval metrics")
    print("=" * 60)

    from representation_learning.evaluation.retrieval import eval_retrieval

    # Create dummy embeddings
    n_samples = 100
    embed_dim = 256
    n_classes = 5

    # Create embeddings where same-class samples are similar
    embeddings = []
    labels = []
    for i in range(n_samples):
        class_id = i % n_classes
        # Base vector for each class + small noise
        base = torch.zeros(embed_dim)
        base[class_id * 50 : (class_id + 1) * 50] = 1.0
        emb = base + torch.randn(embed_dim) * 0.1
        embeddings.append(emb)
        labels.append(class_id)

    embeddings = torch.stack(embeddings)
    labels = torch.tensor(labels)

    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Labels shape: {labels.shape}")

    # Test retrieval
    try:
        metrics = eval_retrieval(embeddings, labels)
        print(f"Retrieval metrics: {metrics}")
        print("✅ PASS: Retrieval metrics computed successfully")
        return True
    except Exception as e:
        print(f"❌ FAIL: {e}")
        return False


def test_retrieval_with_none():
    """Test that retrieval fails gracefully with None inputs."""
    print("\n" + "=" * 60)
    print("TEST: Retrieval with None inputs (should fail gracefully)")
    print("=" * 60)

    from representation_learning.evaluation.retrieval import eval_retrieval

    try:
        eval_retrieval(None, torch.tensor([1, 2, 3]))
        print("❌ FAIL: Should have raised an error for None embeddings")
        return False
    except ValueError as e:
        print(f"✅ PASS: Got proper ValueError: {e}")
        return True
    except (AttributeError, TypeError) as e:
        print(f"❌ FAIL: Got unhelpful error: {type(e).__name__}: {e}")
        return False


def test_clustering_metrics():
    """Test clustering metric computation."""
    print("\n" + "=" * 60)
    print("TEST: Clustering metrics")
    print("=" * 60)

    from representation_learning.evaluation.clustering import eval_clustering

    # Create dummy embeddings
    n_samples = 100
    embed_dim = 256
    n_classes = 5

    embeddings = torch.randn(n_samples, embed_dim)
    labels = torch.randint(0, n_classes, (n_samples,))

    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Labels shape: {labels.shape}")

    try:
        metrics = eval_clustering(embeddings, labels)
        print(f"Clustering metrics: {metrics}")
        print("✅ PASS: Clustering metrics computed successfully")
        return True
    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_model_loading():
    """Test BEATs/OpenBEATs model loading."""
    print("\n" + "=" * 60)
    print("TEST: Model loading (OpenBEATs)")
    print("=" * 60)

    try:
        from representation_learning.configs import AudioConfig, ModelSpec
        from representation_learning.models.get_model import get_model

        audio_config = AudioConfig(
            sample_rate=16000,
            representation="raw",
            normalize=False,
            target_length_seconds=10,
        )

        model_spec = ModelSpec(
            name="openbeats",
            pretrained=True,
            device="cuda" if torch.cuda.is_available() else "cpu",
            audio_config=audio_config,
            model_id="openbeats-large-i3",
            model_size="large",
        )

        print(f"Loading model: {model_spec.name} ({model_spec.model_id})")
        model = get_model(model_spec, num_classes=10)
        print(f"Model loaded successfully!")
        print(f"Model type: {type(model).__name__}")

        # Test forward pass with dummy audio
        device = model_spec.device
        dummy_audio = torch.randn(2, 16000 * 5).to(device)  # 5 seconds
        model.eval()
        with torch.no_grad():
            output = model(dummy_audio)
        print(f"Output shape: {output.shape}")
        print("✅ PASS: Model loading and forward pass work")
        return True

    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("EVALUATION PIPELINE TESTS")
    print("=" * 60)

    results = {}

    # Run tests
    results["embedding_dataset_structure"] = test_embedding_dataset_structure()
    results["embedding_layer_selection"] = test_embedding_layer_selection()
    results["fixed_layer_selection"] = test_fixed_layer_selection()
    results["retrieval_metrics"] = test_retrieval_metrics()
    results["retrieval_none_handling"] = test_retrieval_with_none()
    results["clustering_metrics"] = test_clustering_metrics()

    # Model loading test (optional - requires HF download)
    if "--with-model" in sys.argv:
        results["model_loading"] = test_model_loading()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"  {status}: {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed < total:
        print("\n⚠️  Some tests failed - fixes needed!")
        return 1
    else:
        print("\n✅ All tests passed!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
