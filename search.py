import argparse

from pgvector import Vector
from psycopg.rows import dict_row

from db import get_connection
from embeddings import embed_query
from log_config import setup_logging


def cari_resep(
    kata_kunci: str,
    pakai_santan: bool | None = None,
    vegetarian: bool | None = None,
    pedas: bool | None = None,
    maks_menit: int | None = None,
    limit: int = 3,
) -> list[dict]:
    """Search recipes by meaning, optionally narrowed by exact metadata filters.

    A filter left as None is not applied. Smaller `jarak` means closer in meaning.
    """
    if not kata_kunci.strip():
        raise ValueError("kata_kunci must not be empty")

    # Filters are fixed SQL fragments; only their values go through query parameters.
    conditions: list[str] = []
    params: dict = {"embedding": Vector(embed_query(kata_kunci)), "limit": limit}
    for column, value in (
        ("pakai_santan", pakai_santan),
        ("vegetarian", vegetarian),
        ("pedas", pedas),
    ):
        if value is not None:
            conditions.append(f"{column} = %({column})s")
            params[column] = value
    if maks_menit is not None:
        conditions.append("waktu_masak_menit <= %(maks_menit)s")
        params["maks_menit"] = maks_menit

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    sql = f"""
        SELECT slug, judul, teks, waktu_masak_menit,
               embedding <=> %(embedding)s AS jarak
        FROM resep
        {where}
        ORDER BY embedding <=> %(embedding)s
        LIMIT %(limit)s
    """
    with get_connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def _yes_no(value: str) -> bool:
    return value == "ya"


if __name__ == "__main__":
    # Manual test: python search.py "masakan berkuah" --santan tidak --maks-menit 30
    parser = argparse.ArgumentParser()
    parser.add_argument("kata_kunci")
    parser.add_argument("--santan", choices=["ya", "tidak"])
    parser.add_argument("--vegetarian", choices=["ya", "tidak"])
    parser.add_argument("--pedas", choices=["ya", "tidak"])
    parser.add_argument("--maks-menit", type=int)
    args = parser.parse_args()

    setup_logging()
    hasil = cari_resep(
        args.kata_kunci,
        pakai_santan=_yes_no(args.santan) if args.santan else None,
        vegetarian=_yes_no(args.vegetarian) if args.vegetarian else None,
        pedas=_yes_no(args.pedas) if args.pedas else None,
        maks_menit=args.maks_menit,
    )
    if not hasil:
        print("no results")
    for r in hasil:
        print(f"{r['jarak']:.3f}  {r['waktu_masak_menit']:>3} mnt  {r['judul']}")
