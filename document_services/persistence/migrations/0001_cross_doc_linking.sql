-- One-off migration: adds cross-document linking (grouping nodes and
-- cross-docs relations) to an already-populated database.
--
-- docarag's schema is declarative and applied whole (see schema.py):
-- `ensure_schema` only ever acts on a completely empty database, and
-- `reset_schema` drops the entire `docarag` schema and all its data before
-- reapplying it. Neither is safe to run against an existing corpus once new
-- tables have been added to schema.py. This script is the safe alternative:
-- run it once, directly, against an existing database. It is idempotent --
-- rerunning it is a no-op.
--
-- Keep this file's table definitions identical to the ones in schema.py.

CREATE TABLE IF NOT EXISTS docarag.cross_doc_statement (
    tenant_id text NOT NULL,
    corpus_id text NOT NULL,
    stmt_key bigint GENERATED ALWAYS AS IDENTITY,
    predicate text NOT NULL,
    directed boolean NOT NULL DEFAULT true,
    weight real,
    confidence real,
    producer text,
    description_blob_key bigint REFERENCES docarag.text_blob(blob_key),
    attrs jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (tenant_id, corpus_id, stmt_key),
    FOREIGN KEY (tenant_id, corpus_id)
        REFERENCES docarag.corpora(tenant_id, corpus_id),
    CONSTRAINT cross_doc_statement_predicate_len
        CHECK (char_length(predicate) BETWEEN 1 AND 256)
);

CREATE INDEX IF NOT EXISTS cross_doc_statement_predicate_idx
    ON docarag.cross_doc_statement (tenant_id, corpus_id, predicate);

CREATE TABLE IF NOT EXISTS docarag.cross_doc_statement_role (
    tenant_id text NOT NULL,
    corpus_id text NOT NULL,
    stmt_key bigint NOT NULL,
    role text NOT NULL,
    entity_key bigint NOT NULL REFERENCES docarag.entity(entity_key) ON DELETE CASCADE,
    ordinal smallint NOT NULL DEFAULT 0,
    PRIMARY KEY (tenant_id, corpus_id, stmt_key, role, entity_key, ordinal),
    FOREIGN KEY (tenant_id, corpus_id, stmt_key)
        REFERENCES docarag.cross_doc_statement(tenant_id, corpus_id, stmt_key)
        ON DELETE CASCADE,
    CONSTRAINT cross_doc_statement_role_len CHECK (char_length(role) BETWEEN 1 AND 64)
);

CREATE INDEX IF NOT EXISTS cross_doc_statement_role_entity_idx
    ON docarag.cross_doc_statement_role (tenant_id, corpus_id, entity_key)
    INCLUDE (stmt_key, role);

CREATE TABLE IF NOT EXISTS docarag.cross_doc_statement_property (
    tenant_id text NOT NULL,
    corpus_id text NOT NULL,
    stmt_key bigint NOT NULL,
    name text NOT NULL,
    text_val text,
    num_val numeric,
    time_val timestamptz,
    PRIMARY KEY (tenant_id, corpus_id, stmt_key, name),
    FOREIGN KEY (tenant_id, corpus_id, stmt_key)
        REFERENCES docarag.cross_doc_statement(tenant_id, corpus_id, stmt_key)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS cross_doc_statement_property_lookup_idx
    ON docarag.cross_doc_statement_property (tenant_id, corpus_id, name, text_val);

-- Widen stage_state to track cross-doc-link runs. DROP+ADD rather than a
-- bare ALTER since Postgres has no "ADD CONSTRAINT IF NOT EXISTS".
DO $$
BEGIN
    ALTER TABLE docarag.stage_state DROP CONSTRAINT IF EXISTS stage_state_stage;
    ALTER TABLE docarag.stage_state ADD CONSTRAINT stage_state_stage CHECK (
        stage IN ('parse', 'tree', 'graph', 'vdb', 'persist', 'cross_doc_link')
    );
END $$;
