"""
Embeddings Module

Converts text to vector embeddings using OpenAI's text-embedding-3-small model.
This is the same model recommended by Microsoft for RAG applications.

Reference: https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-solution-design-and-evaluation-guide
"""

import os
import re
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

# Initialize OpenAI client
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Embedding model - same as used in Openclaw-mini
EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-3-small")


def preprocess_text(text: str) -> str:
    """
    Clean text before embedding.
    Removes Slack-specific formatting.
    """
    # Remove user mentions: <@U1234567>
    text = re.sub(r"<@[A-Z0-9]+>", "", text)

    # Convert channel links: <#C1234567|general> -> #general
    text = re.sub(r"<#[A-Z0-9]+\|([^>]+)>", r"#\1", text)

    # Convert URL links: <https://example.com|Example> -> Example
    text = re.sub(r"<https?://[^|>]+\|([^>]+)>", r"\1", text)

    # Remove plain URLs: <https://example.com>
    text = re.sub(r"<https?://[^>]+>", "", text)

    # Remove extra whitespace
    text = " ".join(text.split())

    return text.strip()


async def create_embedding(text: str) -> list[float]:
    """
    Create embedding vector for a single text.

    Args:
        text: The text to embed

    Returns:
        List of floats representing the embedding vector
    """
    cleaned_text = preprocess_text(text)

    if not cleaned_text:
        # Return zero vector for empty text
        return [0.0] * 1536  # text-embedding-3-small dimension

    response = await client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=cleaned_text,
    )

    return response.data[0].embedding


async def create_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Create embedding vectors for multiple texts.
    More efficient than calling create_embedding multiple times.

    Args:
        texts: List of texts to embed

    Returns:
        List of embedding vectors
    """
    cleaned_texts = [preprocess_text(t) for t in texts]

    # Filter out empty texts, keep track of indices
    non_empty = [(i, t) for i, t in enumerate(cleaned_texts) if t]

    if not non_empty:
        return [[0.0] * 1536 for _ in texts]

    # Create embeddings for non-empty texts
    response = await client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[t for _, t in non_empty],
    )

    # Map back to original indices
    result = [[0.0] * 1536 for _ in texts]
    for idx, (original_idx, _) in enumerate(non_empty):
        result[original_idx] = response.data[idx].embedding

    return result
