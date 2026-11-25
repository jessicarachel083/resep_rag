import logging
import os

import voyageai
from dotenv import load_dotenv

load_dotenv()

MODEL = "voyage-4-large"
DIMENSION = 1024  # must match vector(1024) in db/schema.sql
BATCH_SIZE = 64

logger = logging.getLogger("embeddings")

_api_key = os.getenv("VOYAGE_API_KEY")
if not _api_key:
    raise RuntimeError("VOYAGE_API_KEY is not set. Check your .env file.")

# The library default is max_retries=0, so one transient error would abort the whole ingest.
_client = voyageai.Client(api_key=_api_key, max_retries=3)


def _embed(texts: list[str], input_type: str) -> list[list[float]]:
    # truncation=False: fail loudly instead of silently cutting a long recipe.
    result = _client.embed(
        texts,
        model=MODEL,
        input_type=input_type,
        truncation=False,
        output_dimension=DIMENSION,
    )
    logger.info(
        "voyage embed: input_type=%s texts=%d tokens=%d",
        input_type,
        len(texts),
        result.total_tokens,
    )
    return result.embeddings


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed recipes for storage (used by ingest.py)."""
    vectors: list[list[float]] = []
    for start in range(0, len(texts), BATCH_SIZE):
        vectors.extend(_embed(texts[start:start + BATCH_SIZE], "document"))
    return vectors


def embed_query(text: str) -> list[float]:
    """Embed a search query (used by cari_resep in agent.py)."""
    logger.info("search query: %s", text)
    return _embed([text], "query")[0]
