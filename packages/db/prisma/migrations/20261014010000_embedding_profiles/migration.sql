BEGIN;

-- Keep existing OpenAI vectors while adding native 1024-dimensional Qwen vectors.
DROP INDEX "RepositoryChunk_embedding_hnsw_idx";
ALTER TABLE "RepositoryChunk"
    ALTER COLUMN "embedding" TYPE vector USING "embedding"::vector,
    ADD COLUMN "embeddingProfile" TEXT,
    ADD CONSTRAINT "RepositoryChunk_embedding_dimensions_check"
        CHECK (vector_dims("embedding") IN (1024, 1536));

-- pgvector requires one dimension per HNSW index, even for an unbounded column.
CREATE INDEX "RepositoryChunk_embedding_1024_hnsw_idx"
    ON "RepositoryChunk" USING hnsw (("embedding"::vector(1024)) vector_cosine_ops)
    WHERE vector_dims("embedding") = 1024;
CREATE INDEX "RepositoryChunk_embedding_1536_hnsw_idx"
    ON "RepositoryChunk" USING hnsw (("embedding"::vector(1536)) vector_cosine_ops)
    WHERE vector_dims("embedding") = 1536;

COMMIT;
