-- schema.sql: tabel resep untuk project resep-rag
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS resep (
    id                BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    slug              TEXT NOT NULL UNIQUE,    -- nama file tanpa ekstensi, mis. 'opor-ayam'
    judul             TEXT NOT NULL,
    teks              TEXT NOT NULL,           -- isi resep, dikirim ke LLM sebagai konteks
    embedding         vector(1024) NOT NULL,   -- voyage-4-large, dimensi default
    content_hash      TEXT NOT NULL,           -- deteksi resep yang berubah
    pakai_santan      BOOLEAN NOT NULL,
    waktu_masak_menit INTEGER NOT NULL,
    vegetarian        BOOLEAN NOT NULL,
    pedas             BOOLEAN NOT NULL
);
