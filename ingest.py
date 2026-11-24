import hashlib
import logging
from pathlib import Path

from pgvector import Vector

from db import get_connection
from embeddings import embed_documents

RESEP_DIR = Path(__file__).parent / "resep"
LOG_FILE = Path(__file__).parent / "ingest.log"

logger = logging.getLogger("ingest")

UPSERT_SQL = """
INSERT INTO resep (slug, judul, teks, embedding, content_hash,
                   pakai_santan, waktu_masak_menit, vegetarian, pedas)
VALUES (%(slug)s, %(judul)s, %(teks)s, %(embedding)s, %(content_hash)s,
        %(pakai_santan)s, %(waktu_masak_menit)s, %(vegetarian)s, %(pedas)s)
ON CONFLICT (slug) DO UPDATE SET
    judul = EXCLUDED.judul,
    teks = EXCLUDED.teks,
    embedding = EXCLUDED.embedding,
    content_hash = EXCLUDED.content_hash,
    pakai_santan = EXCLUDED.pakai_santan,
    waktu_masak_menit = EXCLUDED.waktu_masak_menit,
    vegetarian = EXCLUDED.vegetarian,
    pedas = EXCLUDED.pedas
"""


def _setup_logging() -> None:
    # Same messages go to the terminal and to ingest.log (appended on every run).
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()],
    )


def _field(lines: list[str], key: str) -> str:
    prefix = f"{key}:"
    for line in lines:
        if line.startswith(prefix):
            return line[len(prefix):].strip()
    raise ValueError(f"missing field '{key}'")


def _to_bool(value: str) -> bool:
    if value == "ya":
        return True
    if value == "tidak":
        return False
    raise ValueError(f"expected 'ya' or 'tidak', got '{value}'")


def parse_recipe(path: Path) -> dict:
    # read_text normalizes CRLF to LF, so the hash is identical on every OS.
    teks = path.read_text(encoding="utf-8").strip()
    lines = teks.splitlines()
    try:
        return {
            "slug": path.stem,
            "judul": _field(lines, "Judul"),
            "teks": teks,
            "content_hash": hashlib.sha256(teks.encode("utf-8")).hexdigest(),
            "pakai_santan": _to_bool(_field(lines, "Pakai santan")),
            "waktu_masak_menit": int(_field(lines, "Waktu masak (menit)")),
            "vegetarian": _to_bool(_field(lines, "Vegetarian")),
            "pedas": _to_bool(_field(lines, "Pedas")),
        }
    except ValueError as e:
        raise ValueError(f"{path.name}: {e}") from e


def main() -> None:
    # Parse everything first: a malformed file aborts before any API call or DB write.
    recipes = [parse_recipe(p) for p in sorted(RESEP_DIR.glob("*.txt"))]
    if not recipes:
        raise RuntimeError(f"No recipe files found in {RESEP_DIR}")

    with get_connection() as conn:
        existing = dict(conn.execute("SELECT slug, content_hash FROM resep").fetchall())
        changed = [r for r in recipes if existing.get(r["slug"]) != r["content_hash"]]
        unchanged = len(recipes) - len(changed)
        if not changed:
            logger.info("%d recipes, nothing changed (0 sent to Voyage).", len(recipes))
            return

        logger.info("%d recipes unchanged (not sent to Voyage).", unchanged)
        for recipe in changed:
            action = "update" if recipe["slug"] in existing else "new"
            logger.info("sending to Voyage: %s (%s)", recipe["slug"], action)

        vectors = embed_documents([r["teks"] for r in changed])
        for recipe, vector in zip(changed, vectors, strict=True):
            conn.execute(UPSERT_SQL, {**recipe, "embedding": Vector(vector)})
        logger.info("%d recipes, %d embedded and upserted.", len(recipes), len(changed))


if __name__ == "__main__":
    _setup_logging()
    try:
        main()
    except Exception:
        # Record the failure in ingest.log, then re-raise so the terminal still shows it.
        logger.exception("ingest failed")
        raise
