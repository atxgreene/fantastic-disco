"""Tests for TurboQuant vector compression."""

import numpy as np
import pytest

from mnemosyne.compression.turboquant import (
    CompressedVector,
    TurboQuantCompressor,
    _fast_hadamard_transform,
    _pad_to_power_of_2,
)
from mnemosyne.compression.mixed_precision import MixedPrecisionPolicy, PrecisionTier


class TestHadamardTransform:
    def test_self_inverse(self):
        """Hadamard transform applied twice returns original (up to normalization)."""
        x = np.random.randn(16).astype(np.float32)
        transformed = _fast_hadamard_transform(x)
        recovered = _fast_hadamard_transform(transformed)
        np.testing.assert_allclose(recovered, x, atol=1e-5)

    def test_preserves_norm(self):
        """Hadamard rotation preserves L2 norm."""
        x = np.random.randn(64).astype(np.float32)
        transformed = _fast_hadamard_transform(x)
        np.testing.assert_allclose(
            np.linalg.norm(transformed),
            np.linalg.norm(x),
            rtol=1e-4,
        )

    def test_power_of_2_required(self):
        x = np.random.randn(16).astype(np.float32)
        result = _fast_hadamard_transform(x)
        assert result.shape == (16,)


class TestPadding:
    def test_already_power_of_2(self):
        x = np.ones(32, dtype=np.float32)
        padded, orig = _pad_to_power_of_2(x)
        assert len(padded) == 32
        assert orig == 32

    def test_needs_padding(self):
        x = np.ones(384, dtype=np.float32)  # MiniLM dimension
        padded, orig = _pad_to_power_of_2(x)
        assert len(padded) == 512  # Next power of 2
        assert orig == 384
        assert padded[384] == 0.0  # Padded with zeros


class TestTurboQuantCompressor:
    def test_compress_decompress_preserves_direction(self):
        """Compressed then decompressed vector should point in same direction."""
        compressor = TurboQuantCompressor(bits=4, qjl_dim=32)
        original = np.random.randn(384).astype(np.float32)
        original /= np.linalg.norm(original)

        compressed = compressor.compress(original)
        recovered = compressor.decompress(compressed)
        recovered /= np.linalg.norm(recovered)

        cosine_sim = np.dot(original, recovered)
        # At 4 bits, should be very close
        assert cosine_sim > 0.85, f"Cosine similarity too low: {cosine_sim}"

    def test_compression_ratio(self):
        """Should achieve meaningful compression."""
        compressor = TurboQuantCompressor(bits=3, qjl_dim=64)
        original = np.random.randn(384).astype(np.float32)
        compressed = compressor.compress(original)

        original_bytes = 384 * 4  # float32
        compressed_bytes = compressed.storage_bytes()
        ratio = compressed.compression_ratio(original_bytes)

        assert ratio > 3.0, f"Compression ratio too low: {ratio}"
        assert compressed_bytes < original_bytes

    def test_inner_product_unbiased(self):
        """Inner products on compressed vectors should be approximately correct."""
        compressor = TurboQuantCompressor(bits=4, qjl_dim=128)

        a = np.random.randn(384).astype(np.float32)
        b = np.random.randn(384).astype(np.float32)

        true_dot = float(np.dot(a, b))
        ca = compressor.compress(a)
        cb = compressor.compress(b)
        estimated_dot = compressor.inner_product(ca, cb)

        # Should be within reasonable range (not exact due to quantization)
        relative_error = abs(estimated_dot - true_dot) / (abs(true_dot) + 1e-10)
        assert relative_error < 0.5, f"Inner product error too high: {relative_error}"

    def test_zero_vector(self):
        compressor = TurboQuantCompressor(bits=3)
        zero = np.zeros(384, dtype=np.float32)
        compressed = compressor.compress(zero)
        assert compressed.norm == 0.0
        recovered = compressor.decompress(compressed)
        np.testing.assert_allclose(recovered, zero, atol=1e-6)

    def test_batch_compress(self):
        compressor = TurboQuantCompressor(bits=3)
        batch = np.random.randn(10, 384).astype(np.float32)
        results = compressor.compress_batch(batch)
        assert len(results) == 10
        assert all(isinstance(r, CompressedVector) for r in results)

    def test_stats(self):
        compressor = TurboQuantCompressor(bits=3)
        for _ in range(5):
            compressor.compress(np.random.randn(384).astype(np.float32))
        stats = compressor.get_stats()
        assert stats["total_compressed"] == 5
        assert stats["bits_per_coordinate"] == 3


class TestMixedPrecision:
    def test_tier_selection(self):
        policy = MixedPrecisionPolicy()
        # Index embeddings always get HIGH
        _, tier = policy.compress(np.random.randn(384).astype(np.float32), importance=0.1, is_index=True)
        assert tier == PrecisionTier.HIGH

        # High importance → HIGH
        _, tier = policy.compress(np.random.randn(384).astype(np.float32), importance=0.9)
        assert tier == PrecisionTier.HIGH

        # Low importance → LOW
        _, tier = policy.compress(np.random.randn(384).astype(np.float32), importance=0.2)
        assert tier == PrecisionTier.LOW

        # Medium importance → STANDARD
        _, tier = policy.compress(np.random.randn(384).astype(np.float32), importance=0.5)
        assert tier == PrecisionTier.STANDARD

    def test_average_bits_mixed(self):
        policy = MixedPrecisionPolicy()
        vec = np.random.randn(384).astype(np.float32)

        # Compress a mix
        policy.compress(vec, importance=0.9)   # HIGH (4 bit)
        policy.compress(vec, importance=0.5)   # STANDARD (3 bit)
        policy.compress(vec, importance=0.1)   # LOW (2 bit)

        avg = policy.average_bits()
        assert 2.0 < avg < 4.0  # Should be ~3.0
