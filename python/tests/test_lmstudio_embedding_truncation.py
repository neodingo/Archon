"""
Test script for verifying LM-Studio embedding dimension truncation.

This test verifies:
1. LMStudioEmbeddingAdapter correctly truncates embeddings to requested dimensions
2. The adapter is selected when provider is "lmstudio"
3. Truncation preserves the first N dimensions (MRL approach)
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from server.services.embeddings.embedding_service import (
    LMStudioEmbeddingAdapter,
    _get_embedding_adapter,
)


def test_lmstudio_adapter_selection():
    """Test that lmstudio provider selects the correct adapter."""
    print("Testing LM-Studio adapter selection...")

    mock_client = MagicMock()
    adapter = _get_embedding_adapter("lmstudio", mock_client)

    assert isinstance(
        adapter, LMStudioEmbeddingAdapter
    ), f"Expected LMStudioEmbeddingAdapter, got {type(adapter)}"
    print("✓ LM-Studio provider correctly selects LMStudioEmbeddingAdapter")

    return True


async def test_embedding_truncation():
    """Test that embeddings are truncated to requested dimensions."""
    print("\nTesting embedding dimension truncation...")

    # Create mock client that returns 4096-dimensional embeddings
    mock_client = MagicMock()
    mock_response = MagicMock()

    # Simulate 4096-dimensional embedding (like Qwen3-Embedding-8B)
    mock_embedding_4096 = list(range(4096))  # [0, 1, 2, ..., 4095]

    mock_item = MagicMock()
    mock_item.embedding = mock_embedding_4096
    mock_response.data = [mock_item]

    mock_client.embeddings.create = AsyncMock(return_value=mock_response)

    # Create adapter and request 1536 dimensions
    adapter = LMStudioEmbeddingAdapter(mock_client)
    result = await adapter.create_embeddings(
        texts=["test text"], model="qwen3-embedding-8b", dimensions=1536
    )

    # Verify truncation
    assert len(result) == 1, "Should return one embedding"
    assert len(result[0]) == 1536, f"Expected 1536 dimensions, got {len(result[0])}"

    # Verify it's the FIRST 1536 dimensions (MRL approach)
    assert result[0] == list(
        range(1536)
    ), "Should preserve first 1536 dimensions in order"

    print(f"✓ Successfully truncated from 4096 to 1536 dimensions")
    print(f"✓ Preserved first 1536 dimensions (MRL approach)")

    return True


async def test_no_truncation_when_dimensions_match():
    """Test that no truncation occurs when dimensions match."""
    print("\nTesting no truncation when dimensions already match...")

    mock_client = MagicMock()
    mock_response = MagicMock()

    # Simulate 1536-dimensional embedding
    mock_embedding_1536 = list(range(1536))

    mock_item = MagicMock()
    mock_item.embedding = mock_embedding_1536
    mock_response.data = [mock_item]

    mock_client.embeddings.create = AsyncMock(return_value=mock_response)

    adapter = LMStudioEmbeddingAdapter(mock_client)
    result = await adapter.create_embeddings(
        texts=["test text"], model="some-model", dimensions=1536
    )

    assert len(result[0]) == 1536, "Should maintain 1536 dimensions"
    assert result[0] == mock_embedding_1536, "Should not modify embeddings"

    print("✓ No truncation applied when dimensions already match")

    return True


async def test_no_truncation_when_dimensions_not_specified():
    """Test that no truncation occurs when dimensions parameter is None."""
    print("\nTesting no truncation when dimensions not specified...")

    mock_client = MagicMock()
    mock_response = MagicMock()

    mock_embedding_4096 = list(range(4096))

    mock_item = MagicMock()
    mock_item.embedding = mock_embedding_4096
    mock_response.data = [mock_item]

    mock_client.embeddings.create = AsyncMock(return_value=mock_response)

    adapter = LMStudioEmbeddingAdapter(mock_client)
    result = await adapter.create_embeddings(
        texts=["test text"], model="qwen3-embedding-8b", dimensions=None
    )

    assert len(result[0]) == 4096, "Should return full 4096 dimensions"

    print("✓ No truncation when dimensions=None")

    return True


async def test_batch_embedding_truncation():
    """Test truncation with multiple texts."""
    print("\nTesting batch embedding truncation...")

    mock_client = MagicMock()
    mock_response = MagicMock()

    # Create 3 mock embeddings
    mock_items = []
    for i in range(3):
        mock_item = MagicMock()
        # Each embedding starts at different offset for verification
        mock_item.embedding = list(range(i * 1000, i * 1000 + 4096))
        mock_items.append(mock_item)

    mock_response.data = mock_items
    mock_client.embeddings.create = AsyncMock(return_value=mock_response)

    adapter = LMStudioEmbeddingAdapter(mock_client)
    result = await adapter.create_embeddings(
        texts=["text1", "text2", "text3"], model="qwen3-embedding-8b", dimensions=1536
    )

    assert len(result) == 3, "Should return three embeddings"
    for i, embedding in enumerate(result):
        assert len(embedding) == 1536, f"Embedding {i} should have 1536 dimensions"
        expected = list(range(i * 1000, i * 1000 + 1536))
        assert (
            embedding == expected
        ), f"Embedding {i} should preserve first 1536 dimensions"

    print("✓ Batch truncation works correctly for multiple texts")

    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("LM-Studio Embedding Dimension Truncation Tests")
    print("=" * 60)

    results = []

    # Test 1: Adapter selection
    try:
        results.append(("Adapter Selection", test_lmstudio_adapter_selection()))
    except Exception as e:
        print(f"✗ Adapter selection test failed: {e}")
        import traceback

        traceback.print_exc()
        results.append(("Adapter Selection", False))

    # Test 2: Basic truncation
    try:
        result = asyncio.run(test_embedding_truncation())
        results.append(("Embedding Truncation", result))
    except Exception as e:
        print(f"✗ Embedding truncation test failed: {e}")
        import traceback

        traceback.print_exc()
        results.append(("Embedding Truncation", False))

    # Test 3: No truncation when dimensions match
    try:
        result = asyncio.run(test_no_truncation_when_dimensions_match())
        results.append(("No Truncation (Match)", result))
    except Exception as e:
        print(f"✗ No truncation test failed: {e}")
        import traceback

        traceback.print_exc()
        results.append(("No Truncation (Match)", False))

    # Test 4: No truncation when dimensions not specified
    try:
        result = asyncio.run(test_no_truncation_when_dimensions_not_specified())
        results.append(("No Truncation (None)", result))
    except Exception as e:
        print(f"✗ No truncation (None) test failed: {e}")
        import traceback

        traceback.print_exc()
        results.append(("No Truncation (None)", False))

    # Test 5: Batch truncation
    try:
        result = asyncio.run(test_batch_embedding_truncation())
        results.append(("Batch Truncation", result))
    except Exception as e:
        print(f"✗ Batch truncation test failed: {e}")
        import traceback

        traceback.print_exc()
        results.append(("Batch Truncation", False))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")

    all_passed = all(passed for _, passed in results)

    if all_passed:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
