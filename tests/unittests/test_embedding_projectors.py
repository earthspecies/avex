"""Tests for embedding projectors."""

import pytest
import torch

from representation_learning.models.probes.embedding_projectors import (
    Conv4DProjector,
    EmbeddingProjector,
    Sequence3DProjector,
)


class TestConv4DProjector:
    """Test cases for Conv4DProjector."""

    def test_conv4d_projector_basic(self) -> None:
        """Test basic 4D to 3D conversion."""
        projector = Conv4DProjector()

        # Test input: (batch=2, channels=3, height=4, width=5)
        x = torch.randn(2, 3, 4, 5)
        output = projector(x)

        # Expected output: (batch=2, width=5, channels*height=12)
        assert output.shape == (2, 5, 12)
        assert output.dim() == 3

    def test_conv4d_projector_with_target_dim(self) -> None:
        """Test 4D projector with target feature dimension."""
        projector = Conv4DProjector(
            target_feature_dim=64,
        )

        x = torch.randn(2, 3, 4, 5)
        output = projector(x)

        # Expected output: (batch=2, width=5, target_dim=64)
        assert output.shape == (2, 5, 64)
        assert output.dim() == 3

    def test_conv4d_projector_invalid_input(self) -> None:
        """Test 4D projector with invalid input dimensions."""
        projector = Conv4DProjector()

        # Test with 3D input (should raise error)
        x = torch.randn(2, 3, 4)
        with pytest.raises(ValueError, match="Conv4DProjector expects 4D input"):
            projector(x)


class TestSequence3DProjector:
    """Test cases for Sequence3DProjector."""

    def test_sequence3d_projector_basic(self) -> None:
        """Test basic 3D standardization."""
        projector = Sequence3DProjector()

        # Test input: (batch=2, seq_len=10, features=64)
        x = torch.randn(2, 10, 64)
        output = projector(x)

        # Should maintain the same shape
        assert output.shape == (2, 10, 64)
        assert output.dim() == 3

    def test_sequence3d_projector_with_target_dim(self) -> None:
        """Test 3D projector with target feature dimension."""
        projector = Sequence3DProjector(
            target_feature_dim=128,
        )

        x = torch.randn(2, 10, 64)
        output = projector(x)

        # Expected output: (batch=2, seq_len=10, target_dim=128)
        assert output.shape == (2, 10, 128)
        assert output.dim() == 3

    def test_sequence3d_projector_invalid_input(self) -> None:
        """Test 3D projector with invalid input dimensions."""
        projector = Sequence3DProjector()

        # Test with 2D input (should raise error)
        x = torch.randn(2, 64)
        with pytest.raises(ValueError, match="Sequence3DProjector expects 3D input"):
            projector(x)


class TestEmbeddingProjector:
    """Test cases for EmbeddingProjector."""

    def test_embedding_projector_4d_input(self) -> None:
        """Test unified projector with 4D input."""
        projector = EmbeddingProjector()

        # Test 4D input: (batch=2, channels=3, height=4, width=5)
        x = torch.randn(2, 3, 4, 5)
        output = projector(x)

        # Expected output: (batch=2, width=5, channels*height=12)
        assert output.shape == (2, 5, 12)
        assert output.dim() == 3

    def test_embedding_projector_3d_input(self) -> None:
        """Test unified projector with 3D input."""
        projector = EmbeddingProjector()

        # Test 3D input: (batch=2, seq_len=10, features=64)
        x = torch.randn(2, 10, 64)
        output = projector(x)

        # Should maintain the same shape
        assert output.shape == (2, 10, 64)
        assert output.dim() == 3

    def test_embedding_projector_2d_input(self) -> None:
        """Test unified projector with 2D input."""
        projector = EmbeddingProjector(force_sequence_format=True)

        # Test 2D input: (batch=2, features=64)
        x = torch.randn(2, 64)
        output = projector(x)

        # Expected output: (batch=2, seq_len=1, features=64)
        assert output.shape == (2, 1, 64)
        assert output.dim() == 3

    def test_embedding_projector_2d_input_no_force(self) -> None:
        """Test unified projector with 2D input without forcing sequence format."""
        projector = EmbeddingProjector(force_sequence_format=False)

        # Test 2D input: (batch=2, features=64)
        x = torch.randn(2, 64)
        output = projector(x)

        # Should maintain the same shape
        assert output.shape == (2, 64)
        assert output.dim() == 2

    def test_embedding_projector_with_target_dim(self) -> None:
        """Test unified projector with target feature dimension."""
        projector = EmbeddingProjector(target_feature_dim=128, force_sequence_format=True)

        # Test 4D input
        x = torch.randn(2, 3, 4, 5)
        output = projector(x)

        # Expected output: (batch=2, width=5, target_dim=128)
        assert output.shape == (2, 5, 128)
        assert output.dim() == 3

    def test_embedding_projector_invalid_input(self) -> None:
        """Test unified projector with invalid input dimensions."""
        projector = EmbeddingProjector()

        # Test with 1D input (should raise error)
        x = torch.randn(64)
        with pytest.raises(ValueError, match="EmbeddingProjector supports 2D, 3D, and 4D tensors"):
            projector(x)

    def test_embedding_projector_get_output_shape_info(self) -> None:
        """Test getting output shape information."""
        projector = EmbeddingProjector(target_feature_dim=128)

        # Test 4D input shape info
        info_4d = projector.get_output_shape_info((2, 3, 4, 5))
        assert info_4d["input_shape"] == (2, 3, 4, 5)
        assert info_4d["output_shape"] == (2, 5, 128)
        assert info_4d["projector_type"] == "Conv4DProjector"
        assert info_4d["sequence_length"] == 5
        assert info_4d["feature_dim"] == 128

        # Test 3D input shape info
        info_3d = projector.get_output_shape_info((2, 10, 64))
        assert info_3d["input_shape"] == (2, 10, 64)
        assert info_3d["output_shape"] == (2, 10, 128)
        assert info_3d["projector_type"] == "Sequence3DProjector"
        assert info_3d["sequence_length"] == 10
        assert info_3d["feature_dim"] == 128

        # Test 2D input shape info
        info_2d = projector.get_output_shape_info((2, 64))
        assert info_2d["input_shape"] == (2, 64)
        assert info_2d["output_shape"] == (2, 1, 128)
        assert info_2d["projector_type"] == "2D->3D"
        assert info_2d["sequence_length"] == 1
        assert info_2d["feature_dim"] == 128


class TestProjectorIntegration:
    """Test integration scenarios."""

    def test_multiple_embedding_shapes(self) -> None:
        """Test handling multiple embedding shapes in sequence."""
        projector = EmbeddingProjector(
            target_feature_dim=256,
        )

        # Test different input shapes
        inputs = [
            torch.randn(2, 3, 4, 5),  # 4D
            torch.randn(2, 10, 64),  # 3D
            torch.randn(2, 128),  # 2D
        ]

        outputs = []
        for x in inputs:
            output = projector(x)
            outputs.append(output)
            assert output.dim() == 3
            assert output.shape[0] == 2  # batch size preserved

        # All outputs should have the same feature dimension
        for output in outputs:
            assert output.shape[2] == 256

    def test_gradient_flow(self) -> None:
        """Test that gradients flow through projectors."""
        projector = EmbeddingProjector(
            target_feature_dim=128,
        )

        # Test 4D input
        x = torch.randn(2, 3, 4, 5, requires_grad=True)
        output = projector(x)

        # Compute a simple loss
        loss = output.sum()
        loss.backward()

        # Check that gradients exist
        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestProjectorWithEmbeddingLists:
    """Test projectors with lists of embeddings containing tensors of
    different dimensions."""

    def test_three_4d_tensors_different_dims(self) -> None:
        """Test projector with three 4D tensors of different dimensions."""
        projector = EmbeddingProjector(
            target_feature_dim=256,
        )

        # Create three 4D tensors with different dimensions
        embeddings = [
            torch.randn(2, 3, 4, 5),  # (batch, channels, height, width)
            torch.randn(2, 6, 8, 10),  # Different 4D shape
            torch.randn(2, 2, 3, 7),  # Another different 4D shape
        ]

        print("Testing three 4D tensors with different dimensions:")
        projected_embeddings = []
        for i, emb in enumerate(embeddings):
            print(f"  Original embedding {i}: {emb.shape}")
            projected = projector(emb)
            projected_embeddings.append(projected)
            print(f"  Projected embedding {i}: {emb.shape} -> {projected.shape}")

            # Verify all are 3D with same feature dimension
            assert projected.dim() == 3
            assert projected.shape[2] == 256
            assert projected.shape[0] == 2  # batch size preserved

        # Test weighted combination (simulating probe behavior)
        weights = torch.softmax(torch.randn(len(projected_embeddings)), dim=0)
        min_seq_len = min(emb.shape[1] for emb in projected_embeddings)

        weighted_embeddings = torch.zeros(
            2,
            min_seq_len,
            256,
            device=projected_embeddings[0].device,
            dtype=projected_embeddings[0].dtype,
        )

        for _i, (emb, weight) in enumerate(zip(projected_embeddings, weights, strict=False)):
            truncated_emb = emb[:, :min_seq_len, :]
            weighted_embeddings += weight * truncated_emb

        print(f"  Final weighted embeddings shape: {weighted_embeddings.shape}")
        assert weighted_embeddings.shape == (2, min_seq_len, 256)

    def test_three_3d_tensors_different_dims(self) -> None:
        """Test projector with three 3D tensors of different dimensions."""
        projector = EmbeddingProjector(
            target_feature_dim=128,
        )

        # Create three 3D tensors with different dimensions
        embeddings = [
            torch.randn(2, 10, 64),  # (batch, seq_len, features)
            torch.randn(2, 20, 128),  # Different 3D shape
            torch.randn(2, 5, 32),  # Another different 3D shape
        ]

        print("Testing three 3D tensors with different dimensions:")
        projected_embeddings = []
        for i, emb in enumerate(embeddings):
            print(f"  Original embedding {i}: {emb.shape}")
            projected = projector(emb)
            projected_embeddings.append(projected)
            print(f"  Projected embedding {i}: {emb.shape} -> {projected.shape}")

            # Verify all are 3D with same feature dimension
            assert projected.dim() == 3
            assert projected.shape[2] == 128
            assert projected.shape[0] == 2  # batch size preserved

        # Test weighted combination
        weights = torch.softmax(torch.randn(len(projected_embeddings)), dim=0)
        min_seq_len = min(emb.shape[1] for emb in projected_embeddings)

        weighted_embeddings = torch.zeros(
            2,
            min_seq_len,
            128,
            device=projected_embeddings[0].device,
            dtype=projected_embeddings[0].dtype,
        )

        for _i, (emb, weight) in enumerate(zip(projected_embeddings, weights, strict=False)):
            truncated_emb = emb[:, :min_seq_len, :]
            weighted_embeddings += weight * truncated_emb

        print(f"  Final weighted embeddings shape: {weighted_embeddings.shape}")
        assert weighted_embeddings.shape == (2, min_seq_len, 128)

    def test_mixed_4d_and_3d_tensors(self) -> None:
        """Test projector with mixed 4D and 3D tensors."""
        projector = EmbeddingProjector(
            target_feature_dim=512,
        )

        # Create mixed 4D and 3D tensors
        embeddings = [
            torch.randn(2, 3, 4, 5),  # 4D tensor
            torch.randn(2, 6, 8, 10),  # Another 4D tensor
            torch.randn(2, 15, 64),  # 3D tensor
        ]

        print("Testing mixed 4D and 3D tensors:")
        projected_embeddings = []
        for i, emb in enumerate(embeddings):
            print(f"  Original embedding {i}: {emb.shape}")
            projected = projector(emb)
            projected_embeddings.append(projected)
            print(f"  Projected embedding {i}: {emb.shape} -> {projected.shape}")

            # Verify all are 3D with same feature dimension
            assert projected.dim() == 3
            assert projected.shape[2] == 512
            assert projected.shape[0] == 2  # batch size preserved

        # Test weighted combination
        weights = torch.softmax(torch.randn(len(projected_embeddings)), dim=0)
        min_seq_len = min(emb.shape[1] for emb in projected_embeddings)

        weighted_embeddings = torch.zeros(
            2,
            min_seq_len,
            512,
            device=projected_embeddings[0].device,
            dtype=projected_embeddings[0].dtype,
        )

        for _i, (emb, weight) in enumerate(zip(projected_embeddings, weights, strict=False)):
            truncated_emb = emb[:, :min_seq_len, :]
            weighted_embeddings += weight * truncated_emb

        print(f"  Final weighted embeddings shape: {weighted_embeddings.shape}")
        assert weighted_embeddings.shape == (2, min_seq_len, 512)

    def test_mixed_4d_3d_2d_tensors(self) -> None:
        """Test projector with mixed 4D, 3D, and 2D tensors."""
        projector = EmbeddingProjector(
            target_feature_dim=256,
        )

        # Create mixed 4D, 3D, and 2D tensors
        embeddings = [
            torch.randn(2, 3, 4, 5),  # 4D tensor
            torch.randn(2, 10, 64),  # 3D tensor
            torch.randn(2, 128),  # 2D tensor
        ]

        print("Testing mixed 4D, 3D, and 2D tensors:")
        projected_embeddings = []
        for i, emb in enumerate(embeddings):
            print(f"  Original embedding {i}: {emb.shape}")
            projected = projector(emb)
            projected_embeddings.append(projected)
            print(f"  Projected embedding {i}: {emb.shape} -> {projected.shape}")

            # Verify all are 3D with same feature dimension
            assert projected.dim() == 3
            assert projected.shape[2] == 256
            assert projected.shape[0] == 2  # batch size preserved

        # Test weighted combination
        weights = torch.softmax(torch.randn(len(projected_embeddings)), dim=0)
        min_seq_len = min(emb.shape[1] for emb in projected_embeddings)

        weighted_embeddings = torch.zeros(
            2,
            min_seq_len,
            256,
            device=projected_embeddings[0].device,
            dtype=projected_embeddings[0].dtype,
        )

        for _i, (emb, weight) in enumerate(zip(projected_embeddings, weights, strict=False)):
            truncated_emb = emb[:, :min_seq_len, :]
            weighted_embeddings += weight * truncated_emb

        print(f"  Final weighted embeddings shape: {weighted_embeddings.shape}")
        assert weighted_embeddings.shape == (2, min_seq_len, 256)

    def test_realistic_model_embeddings(self) -> None:
        """Test with realistic embedding shapes from different models."""
        projector = EmbeddingProjector(
            target_feature_dim=768,
        )

        # Simulate realistic embeddings from different models
        embeddings = [
            # EfficientNet layers (4D)
            torch.randn(2, 3, 4, 5),  # Early conv layer
            torch.randn(2, 6, 8, 10),  # Mid conv layer
            torch.randn(2, 12, 16, 20),  # Late conv layer
            # BEATs/AVES layers (3D)
            torch.randn(2, 10, 768),  # Early transformer layer
            torch.randn(2, 20, 768),  # Mid transformer layer
            torch.randn(2, 30, 768),  # Late transformer layer
            # BirdMAE output (2D)
            torch.randn(2, 768),  # Final pooled output
        ]

        print("Testing realistic model embeddings:")
        projected_embeddings = []
        for i, emb in enumerate(embeddings):
            print(f"  Original embedding {i}: {emb.shape}")
            projected = projector(emb)
            projected_embeddings.append(projected)
            print(f"  Projected embedding {i}: {emb.shape} -> {projected.shape}")

            # Verify all are 3D with same feature dimension
            assert projected.dim() == 3
            assert projected.shape[2] == 768
            assert projected.shape[0] == 2  # batch size preserved

        # Test weighted combination
        weights = torch.softmax(torch.randn(len(projected_embeddings)), dim=0)
        min_seq_len = min(emb.shape[1] for emb in projected_embeddings)

        weighted_embeddings = torch.zeros(
            2,
            min_seq_len,
            768,
            device=projected_embeddings[0].device,
            dtype=projected_embeddings[0].dtype,
        )

        for _i, (emb, weight) in enumerate(zip(projected_embeddings, weights, strict=False)):
            truncated_emb = emb[:, :min_seq_len, :]
            weighted_embeddings += weight * truncated_emb

        print(f"  Final weighted embeddings shape: {weighted_embeddings.shape}")
        assert weighted_embeddings.shape == (2, min_seq_len, 768)

        # Test that we can use this with attention
        attention = torch.nn.MultiheadAttention(embed_dim=768, num_heads=12, batch_first=True)

        attn_out, attn_weights = attention(
            weighted_embeddings, weighted_embeddings, weighted_embeddings
        )

        print(f"  Attention output shape: {attn_out.shape}")
        assert attn_out.shape == (2, min_seq_len, 768)
