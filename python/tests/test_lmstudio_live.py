"""
Live test of LM-Studio embedding dimension truncation with actual LM-Studio server.

This test connects to a running LM-Studio instance and verifies:
1. Connection to LM-Studio server works
2. Embeddings are generated successfully
3. Dimensions are correctly truncated from 4096 to 1536
4. Multiple texts are handled correctly
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import openai


async def test_raw_lmstudio_api():
    """Test raw LM-Studio API to see what dimensions it returns."""
    print("=" * 60)
    print("Test 1: Raw LM-Studio API (without our adapter)")
    print("=" * 60)

    lm_studio_url = "http://192.168.88.78:1234/v1"
    model = "text-embedding-qwen3-8b-text-embedding"

    try:
        client = openai.AsyncOpenAI(
            api_key="lm-studio",  # LM-Studio doesn't require real key
            base_url=lm_studio_url,
        )

        print(f"Connecting to: {lm_studio_url}")
        print(f"Model: {model}")
        print(f"Generating embedding for test text...")

        # Test without dimensions parameter
        response = await client.embeddings.create(
            model=model, input=["test text for embedding"]
        )

        embedding = response.data[0].embedding
        print(f"\n✓ Raw API Response:")
        print(f"  - Dimensions returned: {len(embedding)}")
        print(f"  - First 5 values: {embedding[:5]}")
        print(f"  - Expected: 4096 dimensions (Qwen3-8B default)")

        if len(embedding) == 4096:
            print(f"  ✓ Confirmed: LM-Studio returns full 4096 dimensions")
        else:
            print(f"  ⚠️  Unexpected dimension count: {len(embedding)}")

        # Test WITH dimensions parameter (should be ignored by LM-Studio)
        print(f"\n Testing with dimensions=1536 parameter...")
        response2 = await client.embeddings.create(
            model=model, input=["test text"], dimensions=1536
        )

        embedding2 = response2.data[0].embedding
        print(f"  - Dimensions returned: {len(embedding2)}")

        if len(embedding2) == 4096:
            print(
                f"  ✓ Confirmed: LM-Studio ignores dimensions parameter (still returns 4096)"
            )
        elif len(embedding2) == 1536:
            print(f"  ✓ Surprise! LM-Studio now respects dimensions parameter!")
        else:
            print(f"  ⚠️  Unexpected dimension count: {len(embedding2)}")

        await client.close()
        return len(embedding)

    except Exception as e:
        print(f"\n✗ Failed to connect to LM-Studio: {e}")
        import traceback

        traceback.print_exc()
        return None


async def test_with_our_adapter():
    """Test with our LMStudioEmbeddingAdapter that does client-side truncation."""
    print("\n" + "=" * 60)
    print("Test 2: With LMStudioEmbeddingAdapter (client-side truncation)")
    print("=" * 60)

    try:
        from server.services.embeddings.embedding_service import (
            LMStudioEmbeddingAdapter,
        )

        lm_studio_url = "http://192.168.88.78:1234/v1"
        model = "text-embedding-qwen3-8b-text-embedding"

        client = openai.AsyncOpenAI(
            api_key="lm-studio", base_url=lm_studio_url
        )

        adapter = LMStudioEmbeddingAdapter(client)

        print(f"Using LMStudioEmbeddingAdapter")
        print(f"Requesting 1536 dimensions...")

        # Single text test
        embeddings = await adapter.create_embeddings(
            texts=["This is a test sentence for embedding generation."],
            model=model,
            dimensions=1536,
        )

        print(f"\n✓ Single Text Test:")
        print(f"  - Dimensions returned: {len(embeddings[0])}")
        print(f"  - Expected: 1536 dimensions")

        if len(embeddings[0]) == 1536:
            print(f"  ✓ SUCCESS: Truncation working correctly!")
        else:
            print(f"  ✗ FAILED: Expected 1536, got {len(embeddings[0])}")

        # Batch test
        test_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning and artificial intelligence are transforming technology.",
            "Python is a versatile programming language used in data science.",
        ]

        print(f"\n✓ Batch Test ({len(test_texts)} texts):")
        embeddings_batch = await adapter.create_embeddings(
            texts=test_texts, model=model, dimensions=1536
        )

        all_correct = True
        for i, embedding in enumerate(embeddings_batch):
            correct = len(embedding) == 1536
            all_correct = all_correct and correct
            status = "✓" if correct else "✗"
            print(f"  {status} Text {i+1}: {len(embedding)} dimensions")

        if all_correct:
            print(f"\n  ✓ SUCCESS: All batch embeddings truncated to 1536!")
        else:
            print(f"\n  ✗ FAILED: Some embeddings have incorrect dimensions")

        # Test without dimensions parameter (should return full 4096)
        print(f"\n✓ Test without dimensions parameter:")
        embeddings_full = await adapter.create_embeddings(
            texts=["test"], model=model, dimensions=None
        )

        print(f"  - Dimensions returned: {len(embeddings_full[0])}")
        print(f"  - Expected: 4096 (full model output)")

        await client.close()
        return all_correct

    except Exception as e:
        print(f"\n✗ Adapter test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_with_full_service():
    """Test with the full embedding service (as it would be used in production)."""
    print("\n" + "=" * 60)
    print("Test 3: Full Embedding Service Integration")
    print("=" * 60)

    try:
        # This test requires the full service infrastructure
        # We'll just verify the configuration is correct
        from server.services.embeddings.embedding_service import _get_embedding_adapter
        import openai

        lm_studio_url = "http://192.168.88.78:1234/v1"

        client = openai.AsyncOpenAI(
            api_key="lm-studio", base_url=lm_studio_url
        )

        adapter = _get_embedding_adapter("lmstudio", client)

        print(f"Testing adapter selection...")
        print(f"  Provider: 'lmstudio'")
        print(f"  Adapter type: {type(adapter).__name__}")

        from server.services.embeddings.embedding_service import (
            LMStudioEmbeddingAdapter,
        )

        if isinstance(adapter, LMStudioEmbeddingAdapter):
            print(f"  ✓ Correct adapter selected!")
            await client.close()
            return True
        else:
            print(f"  ✗ Wrong adapter: expected LMStudioEmbeddingAdapter")
            await client.close()
            return False

    except Exception as e:
        print(f"\n✗ Service integration test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_semantic_similarity():
    """Test that truncated embeddings still maintain semantic similarity."""
    print("\n" + "=" * 60)
    print("Test 4: Semantic Similarity with Truncated Embeddings")
    print("=" * 60)

    try:
        from server.services.embeddings.embedding_service import (
            LMStudioEmbeddingAdapter,
        )
        import numpy as np

        lm_studio_url = "http://192.168.88.78:1234/v1"
        model = "text-embedding-qwen3-8b-text-embedding"

        client = openai.AsyncOpenAI(
            api_key="lm-studio", base_url=lm_studio_url
        )

        adapter = LMStudioEmbeddingAdapter(client)

        # Create embeddings for similar and dissimilar texts
        texts = [
            "The cat sits on the mat.",  # Reference
            "A feline rests on a rug.",  # Similar meaning
            "Python is a programming language.",  # Different meaning
        ]

        print(f"Generating embeddings for semantic similarity test...")
        embeddings = await adapter.create_embeddings(
            texts=texts, model=model, dimensions=1536
        )

        # Calculate cosine similarity
        def cosine_similarity(a, b):
            a_np = np.array(a)
            b_np = np.array(b)
            return np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np))

        ref_embedding = embeddings[0]
        similar_embedding = embeddings[1]
        different_embedding = embeddings[2]

        similarity_similar = cosine_similarity(ref_embedding, similar_embedding)
        similarity_different = cosine_similarity(ref_embedding, different_embedding)

        print(f"\nSemantic Similarity Results:")
        print(f"  Reference: '{texts[0]}'")
        print(f"  Similar:   '{texts[1]}'")
        print(f"    Similarity: {similarity_similar:.4f}")
        print(f"  Different: '{texts[2]}'")
        print(f"    Similarity: {similarity_different:.4f}")

        if similarity_similar > similarity_different:
            print(
                f"\n  ✓ SUCCESS: Similar texts have higher similarity than different texts!"
            )
            print(
                f"  ✓ Truncated embeddings preserve semantic meaning correctly."
            )
            success = True
        else:
            print(
                f"\n  ✗ WARNING: Similar texts don't have higher similarity"
            )
            success = False

        await client.close()
        return success

    except Exception as e:
        print(f"\n✗ Semantic similarity test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print(" LM-STUDIO EMBEDDING DIMENSION TRUNCATION - LIVE TESTS")
    print("=" * 70)
    print(f"LM-Studio URL: http://192.168.88.78:1234/v1")
    print(f"Model: text-embedding-qwen3-8b-text-embedding")
    print(f"Expected behavior: 4096 dims → 1536 dims (client-side truncation)")
    print("=" * 70)

    results = []

    # Test 1: Raw API
    raw_dims = await test_raw_lmstudio_api()
    results.append(("Raw API Connection", raw_dims is not None))

    # Test 2: Our adapter
    adapter_success = await test_with_our_adapter()
    results.append(("Adapter Truncation", adapter_success))

    # Test 3: Full service integration
    service_success = await test_with_full_service()
    results.append(("Service Integration", service_success))

    # Test 4: Semantic similarity
    semantic_success = await test_semantic_similarity()
    results.append(("Semantic Similarity", semantic_success))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:.<50} {status}")

    all_passed = all(passed for _, passed in results)

    print("=" * 70)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("\nThe LM-Studio embedding dimension fix is working correctly:")
        print("  • Connects to your LM-Studio instance")
        print("  • Truncates 4096 dimensions to 1536")
        print("  • Preserves semantic similarity")
        print("  • Ready for pgvector indexing")
    else:
        print("❌ SOME TESTS FAILED")
        print("\nPlease check:")
        print("  • LM-Studio is running at http://192.168.88.78:1234")
        print("  • Model 'text-embedding-qwen3-8b-text-embedding' is loaded")
        print("  • Network connectivity to 192.168.88.78")

    print("=" * 70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
