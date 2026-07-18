"""Declarative PostgreSQL schema for the docarag store: defined once, applied whole.

The store has two halves with opposite rules.

The **ledger** records that a document was admitted and what ran over it.  It is
append-only, enforced by triggers, and is the audit trail: nothing derived may
rewrite it.

The **evidence store** holds what parsing and extraction produced.  It is
entirely derived — a re-run rebuilds it — and is shaped for one job: answering a
question about a document quickly.  Three decisions follow from that.

Evidence is keyed by an interned revision.  The ledger identifies a revision by
five text columns; repeating that tuple on every block, node and statement costs
more than the evidence itself and is copied into every index over it.  The
ledger row owns ``revision_key``; evidence carries that alone.

Structure is one shape.  A table's cells, a formula's symbols and a diagram's
nodes are typed elements joined by typed edges inside one block, so they share
``block_structure`` / ``block_element`` / ``block_element_edge`` instead of three
parallel designs that would each need extending for the next structure kind.

Knowledge is separated from its occurrences.  ``entity`` is corpus-scoped and
outlives any single revision, while ``entity_mention`` records what one revision
saw and where it saw it, so the same real-world entity is recognised across
documents without being duplicated per run.
"""

from __future__ import annotations

import hashlib


LEDGER_SQL = r"""
CREATE TABLE docarag.tenants (
    tenant_id text PRIMARY KEY,
    display_name text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    CONSTRAINT tenants_id_length CHECK (char_length(tenant_id) BETWEEN 1 AND 512),
    CONSTRAINT tenants_name_length CHECK (char_length(display_name) BETWEEN 1 AND 1024)
);

CREATE TABLE docarag.corpora (
    tenant_id text NOT NULL,
    corpus_id text NOT NULL,
    display_name text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (tenant_id, corpus_id),
    FOREIGN KEY (tenant_id) REFERENCES docarag.tenants(tenant_id),
    CONSTRAINT corpora_id_length CHECK (char_length(corpus_id) BETWEEN 1 AND 512),
    CONSTRAINT corpora_name_length CHECK (char_length(display_name) BETWEEN 1 AND 1024)
);

CREATE TABLE docarag.documents (
    tenant_id text NOT NULL,
    corpus_id text NOT NULL,
    document_id text NOT NULL,
    display_name text,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (tenant_id, corpus_id, document_id),
    FOREIGN KEY (tenant_id, corpus_id)
        REFERENCES docarag.corpora(tenant_id, corpus_id),
    CONSTRAINT documents_id_length CHECK (char_length(document_id) BETWEEN 1 AND 512),
    CONSTRAINT documents_name_length CHECK (
        display_name IS NULL OR char_length(display_name) BETWEEN 1 AND 1024
    )
);

CREATE TABLE docarag.document_versions (
    tenant_id text NOT NULL,
    corpus_id text NOT NULL,
    document_id text NOT NULL,
    document_version_id text NOT NULL,
    source_sha256 char(64) NOT NULL,
    source_uri text NOT NULL,
    mime_type text NOT NULL,
    page_count integer,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (tenant_id, corpus_id, document_id, document_version_id),
    UNIQUE (tenant_id, corpus_id, document_id, document_version_id, source_sha256),
    FOREIGN KEY (tenant_id, corpus_id, document_id)
        REFERENCES docarag.documents(tenant_id, corpus_id, document_id),
    CONSTRAINT document_versions_id_length CHECK (
        char_length(document_version_id) BETWEEN 1 AND 512
    ),
    CONSTRAINT document_versions_sha256 CHECK (source_sha256 ~ '^[0-9a-f]{64}$'),
    CONSTRAINT document_versions_source_uri_length CHECK (
        char_length(source_uri) BETWEEN 1 AND 32768
    ),
    CONSTRAINT document_versions_mime_length CHECK (
        char_length(mime_type) BETWEEN 1 AND 512
    ),
    CONSTRAINT document_versions_page_count CHECK (page_count IS NULL OR page_count >= 0)
);

CREATE TABLE docarag.artifact_receipts (
    tenant_id text NOT NULL,
    corpus_id text NOT NULL,
    document_id text NOT NULL,
    document_version_id text NOT NULL,
    receipt_id text NOT NULL,
    artifact_role text NOT NULL,
    availability text NOT NULL,
    artifact_schema text NOT NULL,
    producer_task_id text NOT NULL,
    source_sha256 char(64) NOT NULL,
    artifact_key text NOT NULL,
    uri text,
    sha256 char(64),
    size_bytes bigint,
    media_type text NOT NULL,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (
        tenant_id, corpus_id, document_id, document_version_id, receipt_id
    ),
    UNIQUE (
        tenant_id, corpus_id, document_id, document_version_id,
        producer_task_id, artifact_role, artifact_key
    ),
    FOREIGN KEY (
        tenant_id, corpus_id, document_id, document_version_id, source_sha256
    ) REFERENCES docarag.document_versions(
        tenant_id, corpus_id, document_id, document_version_id, source_sha256
    ),
    CONSTRAINT artifact_receipts_availability CHECK (
        availability IN ('emitted', 'not_emitted')
    ),
    CONSTRAINT artifact_receipts_emission_shape CHECK (
        (
            availability = 'emitted'
            AND uri IS NOT NULL
            AND sha256 ~ '^[0-9a-f]{64}$'
            AND size_bytes > 0
        ) OR (
            availability = 'not_emitted'
            AND uri IS NULL
            AND sha256 IS NULL
            AND size_bytes IS NULL
        )
    ),
    CONSTRAINT artifact_receipts_uri_length CHECK (
        uri IS NULL OR char_length(uri) BETWEEN 1 AND 32768
    )
);

CREATE TABLE docarag.index_revisions (
    revision_key bigint GENERATED ALWAYS AS IDENTITY UNIQUE,
    tenant_id text NOT NULL,
    corpus_id text NOT NULL,
    document_id text NOT NULL,
    document_version_id text NOT NULL,
    revision_id text NOT NULL,
    configuration_sha256 char(64) NOT NULL,
    canonical_artifact_sha256 char(64) NOT NULL,
    qdrant_collection text NOT NULL,
    qdrant_vector_name text,
    state text NOT NULL DEFAULT 'building',
    tree_artifact_sha256 char(64),
    graph_artifact_sha256 char(64),
    retrieval_manifest_sha256 char(64),
    qdrant_point_count bigint,
    failure_code text,
    failure_message text,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    finished_at timestamptz,
    PRIMARY KEY (
        tenant_id, corpus_id, document_id, document_version_id, revision_id
    ),
    FOREIGN KEY (tenant_id, corpus_id, document_id, document_version_id)
        REFERENCES docarag.document_versions(
            tenant_id, corpus_id, document_id, document_version_id
        ),
    CONSTRAINT index_revisions_configuration_sha CHECK (
        configuration_sha256 ~ '^[0-9a-f]{64}$'
    ),
    CONSTRAINT index_revisions_canonical_sha CHECK (
        canonical_artifact_sha256 ~ '^[0-9a-f]{64}$'
    ),
    CONSTRAINT index_revisions_state CHECK (state IN ('building', 'ready', 'failed')),
    CONSTRAINT index_revisions_terminal_shape CHECK (
        (
            state = 'building'
            AND finished_at IS NULL
            AND failure_code IS NULL
            AND failure_message IS NULL
            AND tree_artifact_sha256 IS NULL
            AND graph_artifact_sha256 IS NULL
            AND retrieval_manifest_sha256 IS NULL
            AND qdrant_point_count IS NULL
        ) OR (
            state = 'ready'
            AND finished_at IS NOT NULL
            AND tree_artifact_sha256 ~ '^[0-9a-f]{64}$'
            AND graph_artifact_sha256 ~ '^[0-9a-f]{64}$'
            AND retrieval_manifest_sha256 ~ '^[0-9a-f]{64}$'
            AND qdrant_point_count >= 0
            AND failure_code IS NULL
            AND failure_message IS NULL
        ) OR (
            state = 'failed'
            AND finished_at IS NOT NULL
            AND failure_code IS NOT NULL
            AND failure_message IS NOT NULL
            AND tree_artifact_sha256 IS NULL
            AND graph_artifact_sha256 IS NULL
            AND retrieval_manifest_sha256 IS NULL
            AND qdrant_point_count IS NULL
        )
    )
);

CREATE TABLE docarag.stage_events (
    tenant_id text NOT NULL,
    corpus_id text NOT NULL,
    document_id text NOT NULL,
    document_version_id text NOT NULL,
    event_id text NOT NULL,
    trace_id text NOT NULL,
    stage_run_id text NOT NULL,
    sequence_no bigint NOT NULL,
    stage_name text NOT NULL,
    event_type text NOT NULL,
    status text NOT NULL,
    payload jsonb NOT NULL DEFAULT '{}'::jsonb,
    payload_sha256 char(64) NOT NULL,
    occurred_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    recorded_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (tenant_id, corpus_id, event_id),
    UNIQUE (tenant_id, corpus_id, stage_run_id, sequence_no),
    FOREIGN KEY (tenant_id, corpus_id, document_id, document_version_id)
        REFERENCES docarag.document_versions(
            tenant_id, corpus_id, document_id, document_version_id
        ),
    CONSTRAINT stage_events_sequence CHECK (sequence_no >= 0),
    CONSTRAINT stage_events_status CHECK (
        status IN ('started', 'progress', 'succeeded', 'failed', 'skipped')
    ),
    CONSTRAINT stage_events_payload_sha CHECK (payload_sha256 ~ '^[0-9a-f]{64}$')
);

CREATE TABLE docarag.work_items (
    tenant_id text NOT NULL,
    corpus_id text NOT NULL,
    document_id text NOT NULL,
    document_version_id text NOT NULL,
    work_item_id text NOT NULL,
    kind text NOT NULL,
    idempotency_key text NOT NULL,
    payload jsonb NOT NULL,
    payload_sha256 char(64) NOT NULL,
    priority smallint NOT NULL DEFAULT 0,
    state text NOT NULL DEFAULT 'pending',
    available_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    attempt_count integer NOT NULL DEFAULT 0,
    max_attempts integer NOT NULL,
    lease_owner text,
    lease_expires_at timestamptz,
    fencing_token bigint NOT NULL DEFAULT 0,
    result jsonb,
    error jsonb,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    finished_at timestamptz,
    PRIMARY KEY (tenant_id, corpus_id, work_item_id),
    UNIQUE (tenant_id, corpus_id, kind, idempotency_key),
    FOREIGN KEY (tenant_id, corpus_id, document_id, document_version_id)
        REFERENCES docarag.document_versions(
            tenant_id, corpus_id, document_id, document_version_id
        ),
    CONSTRAINT work_items_payload_sha CHECK (payload_sha256 ~ '^[0-9a-f]{64}$'),
    CONSTRAINT work_items_state CHECK (
        state IN ('pending', 'leased', 'succeeded', 'failed', 'dead')
    ),
    CONSTRAINT work_items_attempts CHECK (
        attempt_count >= 0 AND max_attempts >= 1 AND attempt_count <= max_attempts
    ),
    CONSTRAINT work_items_fencing CHECK (fencing_token >= 0),
    CONSTRAINT work_items_lease_shape CHECK (
        (
            state = 'leased'
            AND lease_owner IS NOT NULL
            AND lease_expires_at IS NOT NULL
            AND attempt_count >= 1
            AND fencing_token >= 1
            AND result IS NULL
            AND error IS NULL
            AND finished_at IS NULL
        ) OR (
            state <> 'leased'
            AND lease_owner IS NULL
            AND lease_expires_at IS NULL
        )
    ),
    CONSTRAINT work_items_terminal_shape CHECK (
        (state IN ('succeeded', 'failed', 'dead') AND finished_at IS NOT NULL)
        OR (state IN ('pending', 'leased') AND finished_at IS NULL)
    )
);

CREATE INDEX work_items_lease_queue_idx
    ON docarag.work_items (
        tenant_id, corpus_id, priority DESC, available_at, created_at
    )
    WHERE state IN ('pending', 'leased');

CREATE INDEX stage_events_trace_idx
    ON docarag.stage_events (tenant_id, corpus_id, trace_id, recorded_at);

CREATE INDEX index_revisions_ready_idx
    ON docarag.index_revisions (tenant_id, corpus_id, state, finished_at DESC);

CREATE FUNCTION docarag.reject_immutable_mutation()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION '% is append-only and cannot be changed with %',
        TG_TABLE_NAME, TG_OP
        USING ERRCODE = '55000';
END;
$$;

CREATE TRIGGER document_versions_are_immutable
    BEFORE UPDATE OR DELETE ON docarag.document_versions
    FOR EACH ROW EXECUTE FUNCTION docarag.reject_immutable_mutation();

CREATE TRIGGER artifact_receipts_are_immutable
    BEFORE UPDATE OR DELETE ON docarag.artifact_receipts
    FOR EACH ROW EXECUTE FUNCTION docarag.reject_immutable_mutation();

CREATE TRIGGER stage_events_are_append_only
    BEFORE UPDATE OR DELETE ON docarag.stage_events
    FOR EACH ROW EXECUTE FUNCTION docarag.reject_immutable_mutation();

CREATE FUNCTION docarag.guard_index_revision_mutation()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        RAISE EXCEPTION 'index revisions cannot be deleted' USING ERRCODE = '55000';
    END IF;
    IF OLD.state <> 'building' OR NEW.state NOT IN ('ready', 'failed') THEN
        RAISE EXCEPTION 'invalid index revision state transition'
            USING ERRCODE = '55000';
    END IF;
    IF (OLD.tenant_id, OLD.corpus_id, OLD.document_id, OLD.document_version_id,
        OLD.revision_id, OLD.configuration_sha256, OLD.canonical_artifact_sha256,
        OLD.qdrant_collection, OLD.qdrant_vector_name, OLD.metadata)
       IS DISTINCT FROM
       (NEW.tenant_id, NEW.corpus_id, NEW.document_id, NEW.document_version_id,
        NEW.revision_id, NEW.configuration_sha256, NEW.canonical_artifact_sha256,
        NEW.qdrant_collection, NEW.qdrant_vector_name, NEW.metadata) THEN
        RAISE EXCEPTION 'index revision identity is immutable'
            USING ERRCODE = '55000';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER index_revision_transitions_are_guarded
    BEFORE UPDATE OR DELETE ON docarag.index_revisions
    FOR EACH ROW EXECUTE FUNCTION docarag.guard_index_revision_mutation();
"""


EVIDENCE_SQL = r"""
CREATE EXTENSION IF NOT EXISTS ltree WITH SCHEMA public;

-- ================================================================ content ==
-- Text is deduplicated across the corpus by digest, so a footer repeated on ten
-- thousand pages is stored once and referenced by every block that shows it.

CREATE TABLE docarag.text_blob (
    blob_key bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    sha256 bytea NOT NULL UNIQUE,
    byte_size integer NOT NULL,
    body text NOT NULL,
    CONSTRAINT text_blob_sha_len CHECK (octet_length(sha256) = 32),
    CONSTRAINT text_blob_size CHECK (byte_size >= 0)
);

-- Bytes live in the content-addressed store; this row says what they are and
-- where to find them.  Assets are shared by every revision resolving to the
-- same digest, so they are never scoped to one.
CREATE TABLE docarag.asset (
    asset_key bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    sha256 bytea NOT NULL UNIQUE,
    media_type text NOT NULL,
    byte_size bigint NOT NULL,
    width integer,
    height integer,
    cas_uri text NOT NULL,
    attrs jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    CONSTRAINT asset_sha_len CHECK (octet_length(sha256) = 32),
    CONSTRAINT asset_size CHECK (byte_size >= 0),
    CONSTRAINT asset_dims CHECK (
        (width IS NULL OR width >= 0) AND (height IS NULL OR height >= 0)
    ),
    CONSTRAINT asset_media_len CHECK (char_length(media_type) BETWEEN 1 AND 512)
);

-- ================================================================== block ==
-- Every evidence table leads with revision_key, so one document's rows stay
-- together, are removed together, and never scan another document's heap.

CREATE TABLE docarag.block (
    revision_key bigint NOT NULL
        REFERENCES docarag.index_revisions(revision_key) ON DELETE CASCADE,
    block_key bigint GENERATED ALWAYS AS IDENTITY,
    block_uid text NOT NULL,
    page_no integer,
    block_no integer NOT NULL,
    block_kind text NOT NULL,
    subtype text,
    reading_order double precision,
    quality text,
    reconciliation text,
    content_sha256 char(64),
    geometry jsonb,
    source_observation_ids jsonb NOT NULL DEFAULT '[]'::jsonb,
    attrs jsonb NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (revision_key, block_key),
    CONSTRAINT block_uid_unique UNIQUE (revision_key, block_uid),
    CONSTRAINT block_page CHECK (page_no IS NULL OR page_no >= 0),
    CONSTRAINT block_kind_len CHECK (char_length(block_kind) BETWEEN 1 AND 64)
);

CREATE INDEX block_page_idx ON docarag.block (revision_key, page_no);
CREATE INDEX block_kind_idx ON docarag.block (revision_key, block_kind);

-- One row per text channel a block carries: parsed text, recovered OCR, a
-- model's description, a table's markup, a formula's LaTeX.
CREATE TABLE docarag.block_representation (
    revision_key bigint NOT NULL,
    block_key bigint NOT NULL,
    rep_no smallint NOT NULL,
    rep_kind text NOT NULL,
    blob_key bigint REFERENCES docarag.text_blob(blob_key),
    asset_key bigint REFERENCES docarag.asset(asset_key),
    lang text,
    confidence real,
    producer text,
    attrs jsonb NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (revision_key, block_key, rep_no),
    FOREIGN KEY (revision_key, block_key)
        REFERENCES docarag.block(revision_key, block_key) ON DELETE CASCADE,
    CONSTRAINT block_rep_kind_len CHECK (char_length(rep_kind) BETWEEN 1 AND 64),
    CONSTRAINT block_rep_confidence CHECK (
        confidence IS NULL OR (confidence >= 0 AND confidence <= 1)
    ),
    CONSTRAINT block_rep_has_content CHECK (
        blob_key IS NOT NULL OR asset_key IS NOT NULL
    )
);

CREATE INDEX block_rep_lookup_idx
    ON docarag.block_representation (revision_key, block_key)
    INCLUDE (rep_kind, blob_key);

CREATE TABLE docarag.block_asset (
    revision_key bigint NOT NULL,
    block_key bigint NOT NULL,
    asset_key bigint NOT NULL REFERENCES docarag.asset(asset_key),
    role text NOT NULL,
    attrs jsonb NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (revision_key, block_key, asset_key, role),
    FOREIGN KEY (revision_key, block_key)
        REFERENCES docarag.block(revision_key, block_key) ON DELETE CASCADE
);

-- ============================================================== structure ==
-- One shape for a table's cells, a formula's symbols, a diagram's nodes and a
-- chart's series: typed elements joined by typed edges, scoped to one block.

CREATE TABLE docarag.block_structure (
    revision_key bigint NOT NULL,
    block_key bigint NOT NULL,
    structure_key bigint GENERATED ALWAYS AS IDENTITY,
    structure_kind text NOT NULL,
    producer text NOT NULL,
    availability text NOT NULL DEFAULT 'complete',
    confidence real,
    source_uri text,
    attrs jsonb NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (revision_key, structure_key),
    FOREIGN KEY (revision_key, block_key)
        REFERENCES docarag.block(revision_key, block_key) ON DELETE CASCADE,
    CONSTRAINT block_structure_kind_len CHECK (
        char_length(structure_kind) BETWEEN 1 AND 64
    ),
    CONSTRAINT block_structure_availability CHECK (
        availability IN ('complete', 'partial')
    )
);

CREATE INDEX block_structure_block_idx
    ON docarag.block_structure (revision_key, block_key);

-- row_no/col_no carry a table's grid and stay null for every other kind, so a
-- cell lookup is an index scan rather than a search through markup.
CREATE TABLE docarag.block_element (
    revision_key bigint NOT NULL,
    structure_key bigint NOT NULL,
    element_key bigint GENERATED ALWAYS AS IDENTITY,
    local_key text NOT NULL,
    element_kind text NOT NULL,
    content text,
    role text,
    row_no integer,
    col_no integer,
    row_span smallint,
    col_span smallint,
    geometry jsonb,
    entity_key bigint,
    confidence real,
    attrs jsonb NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (revision_key, element_key),
    CONSTRAINT block_element_local_unique
        UNIQUE (revision_key, structure_key, local_key),
    FOREIGN KEY (revision_key, structure_key)
        REFERENCES docarag.block_structure(revision_key, structure_key)
        ON DELETE CASCADE,
    CONSTRAINT block_element_grid CHECK (
        (row_no IS NULL OR row_no >= 0) AND (col_no IS NULL OR col_no >= 0)
    )
);

CREATE INDEX block_element_grid_idx
    ON docarag.block_element (revision_key, structure_key, row_no, col_no)
    WHERE row_no IS NOT NULL;
CREATE INDEX block_element_entity_idx
    ON docarag.block_element (entity_key) WHERE entity_key IS NOT NULL;

CREATE TABLE docarag.block_element_edge (
    revision_key bigint NOT NULL,
    structure_key bigint NOT NULL,
    src_key bigint NOT NULL,
    dst_key bigint NOT NULL,
    edge_kind text NOT NULL,
    ordinal smallint NOT NULL DEFAULT 0,
    label text,
    attrs jsonb NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (revision_key, structure_key, src_key, dst_key, edge_kind, ordinal),
    FOREIGN KEY (revision_key, structure_key)
        REFERENCES docarag.block_structure(revision_key, structure_key)
        ON DELETE CASCADE,
    FOREIGN KEY (revision_key, src_key)
        REFERENCES docarag.block_element(revision_key, element_key) ON DELETE CASCADE,
    FOREIGN KEY (revision_key, dst_key)
        REFERENCES docarag.block_element(revision_key, element_key) ON DELETE CASCADE,
    CONSTRAINT block_element_edge_distinct CHECK (src_key <> dst_key)
);

CREATE INDEX block_element_edge_src_idx
    ON docarag.block_element_edge (revision_key, src_key);

-- =================================================================== tree ==
-- Adjacency is the truth and is writable during ingestion, before any path can
-- be known.  The ltree path is derived when the revision is sealed; a sealed
-- revision never changes, so the path never has to be maintained.

CREATE TABLE docarag.tree_node (
    revision_key bigint NOT NULL
        REFERENCES docarag.index_revisions(revision_key) ON DELETE CASCADE,
    node_key bigint GENERATED ALWAYS AS IDENTITY,
    node_no integer NOT NULL,
    parent_key bigint,
    depth smallint,
    sibling_no integer NOT NULL DEFAULT 0,
    path public.ltree,
    node_kind text NOT NULL,
    block_key bigint,
    label text,
    summary_blob_key bigint REFERENCES docarag.text_blob(blob_key),
    attrs jsonb NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (revision_key, node_key),
    CONSTRAINT tree_node_no_unique UNIQUE (revision_key, node_no),
    FOREIGN KEY (revision_key, parent_key)
        REFERENCES docarag.tree_node(revision_key, node_key) ON DELETE CASCADE,
    FOREIGN KEY (revision_key, block_key)
        REFERENCES docarag.block(revision_key, block_key) ON DELETE SET NULL,
    CONSTRAINT tree_node_depth CHECK (depth IS NULL OR depth >= 0),
    CONSTRAINT tree_node_not_own_parent
        CHECK (parent_key IS NULL OR parent_key <> node_key)
);

CREATE INDEX tree_node_path_idx ON docarag.tree_node USING gist (path);
CREATE INDEX tree_node_children_idx
    ON docarag.tree_node (revision_key, parent_key, sibling_no);
CREATE INDEX tree_node_block_idx
    ON docarag.tree_node (revision_key, block_key) WHERE block_key IS NOT NULL;
CREATE INDEX tree_node_kind_idx ON docarag.tree_node (revision_key, node_kind);

-- ============================================================== knowledge ==
-- An entity is corpus-scoped and outlives any single revision; a mention is
-- what one revision saw, where, and how it was written on the page.

CREATE TABLE docarag.entity (
    entity_key bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    tenant_id text NOT NULL,
    corpus_id text NOT NULL,
    entity_kind text NOT NULL,
    canonical text NOT NULL,
    description_blob_key bigint REFERENCES docarag.text_blob(blob_key),
    attrs jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    CONSTRAINT entity_identity_unique
        UNIQUE (tenant_id, corpus_id, entity_kind, canonical),
    FOREIGN KEY (tenant_id, corpus_id)
        REFERENCES docarag.corpora(tenant_id, corpus_id),
    CONSTRAINT entity_canonical_len CHECK (char_length(canonical) BETWEEN 1 AND 2048)
);

CREATE INDEX entity_kind_idx ON docarag.entity (tenant_id, corpus_id, entity_kind);

CREATE TABLE docarag.entity_mention (
    revision_key bigint NOT NULL
        REFERENCES docarag.index_revisions(revision_key) ON DELETE CASCADE,
    mention_key bigint GENERATED ALWAYS AS IDENTITY,
    entity_key bigint NOT NULL REFERENCES docarag.entity(entity_key) ON DELETE CASCADE,
    surface text,
    node_key bigint,
    block_key bigint,
    char_start integer,
    char_end integer,
    confidence real,
    PRIMARY KEY (revision_key, mention_key),
    FOREIGN KEY (revision_key, node_key)
        REFERENCES docarag.tree_node(revision_key, node_key) ON DELETE CASCADE,
    CONSTRAINT entity_mention_span CHECK (
        (char_start IS NULL AND char_end IS NULL)
        OR (char_start >= 0 AND char_end > char_start)
    )
);

-- The retrieval entry point: a query entity resolves to the nodes mentioning it.
CREATE INDEX entity_mention_lookup_idx
    ON docarag.entity_mention (revision_key, entity_key)
    INCLUDE (node_key, block_key);
CREATE INDEX entity_mention_node_idx
    ON docarag.entity_mention (revision_key, node_key);
CREATE INDEX entity_mention_entity_idx ON docarag.entity_mention (entity_key);

-- A statement is one asserted fact of any arity.  Two roles express the binary
-- edge retrieval uses today; more express facts a src/dst pair cannot hold.
CREATE TABLE docarag.statement (
    revision_key bigint NOT NULL
        REFERENCES docarag.index_revisions(revision_key) ON DELETE CASCADE,
    stmt_key bigint GENERATED ALWAYS AS IDENTITY,
    predicate text NOT NULL,
    directed boolean NOT NULL DEFAULT true,
    weight real,
    confidence real,
    producer text,
    description_blob_key bigint REFERENCES docarag.text_blob(blob_key),
    attrs jsonb NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (revision_key, stmt_key),
    CONSTRAINT statement_predicate_len CHECK (char_length(predicate) BETWEEN 1 AND 256)
);

CREATE INDEX statement_predicate_idx ON docarag.statement (revision_key, predicate);

CREATE TABLE docarag.statement_role (
    revision_key bigint NOT NULL,
    stmt_key bigint NOT NULL,
    role text NOT NULL,
    entity_key bigint NOT NULL REFERENCES docarag.entity(entity_key) ON DELETE CASCADE,
    ordinal smallint NOT NULL DEFAULT 0,
    PRIMARY KEY (revision_key, stmt_key, role, entity_key, ordinal),
    FOREIGN KEY (revision_key, stmt_key)
        REFERENCES docarag.statement(revision_key, stmt_key) ON DELETE CASCADE,
    CONSTRAINT statement_role_len CHECK (char_length(role) BETWEEN 1 AND 64)
);

CREATE INDEX statement_role_entity_idx
    ON docarag.statement_role (revision_key, entity_key) INCLUDE (stmt_key, role);

CREATE TABLE docarag.statement_property (
    revision_key bigint NOT NULL,
    stmt_key bigint NOT NULL,
    name text NOT NULL,
    text_val text,
    num_val numeric,
    time_val timestamptz,
    PRIMARY KEY (revision_key, stmt_key, name),
    FOREIGN KEY (revision_key, stmt_key)
        REFERENCES docarag.statement(revision_key, stmt_key) ON DELETE CASCADE
);

-- Why a statement is believed: the node, block or element that asserted it, and
-- the span within it when the producer reported one.
CREATE TABLE docarag.statement_evidence (
    revision_key bigint NOT NULL,
    stmt_key bigint NOT NULL,
    node_key bigint,
    block_key bigint,
    element_key bigint,
    char_start integer,
    char_end integer,
    PRIMARY KEY (revision_key, stmt_key, node_key, block_key, element_key),
    FOREIGN KEY (revision_key, stmt_key)
        REFERENCES docarag.statement(revision_key, stmt_key) ON DELETE CASCADE,
    CONSTRAINT statement_evidence_located CHECK (
        node_key IS NOT NULL OR block_key IS NOT NULL OR element_key IS NOT NULL
    )
);

-- Subgraph induction: given a subtree's nodes, the statements they support.
CREATE INDEX statement_evidence_node_idx
    ON docarag.statement_evidence (revision_key, node_key) INCLUDE (stmt_key);

-- ============================================================== retrieval ==
-- One row per node holding the single string a reranker scores.  Written when a
-- revision is sealed, so the hottest read in the query path -- hundreds of
-- candidates, one field each -- is a single index-only scan.

CREATE TABLE docarag.node_retrieval (
    revision_key bigint NOT NULL
        REFERENCES docarag.index_revisions(revision_key) ON DELETE CASCADE,
    node_key bigint NOT NULL,
    node_kind text NOT NULL,
    page_no integer,
    text_len integer NOT NULL,
    retrieval_text text NOT NULL,
    PRIMARY KEY (revision_key, node_key),
    FOREIGN KEY (revision_key, node_key)
        REFERENCES docarag.tree_node(revision_key, node_key) ON DELETE CASCADE,
    CONSTRAINT node_retrieval_len CHECK (text_len >= 0)
);

CREATE INDEX node_retrieval_feed_idx
    ON docarag.node_retrieval (revision_key)
    INCLUDE (node_key, node_kind, page_no, retrieval_text);
""".strip()


SCHEMA_SQL = LEDGER_SQL + "\n\n" + EVIDENCE_SQL

SCHEMA_SHA256 = hashlib.sha256(SCHEMA_SQL.encode("utf-8")).hexdigest()

# Children before parents, so a caller may remove a revision's evidence
# explicitly rather than relying on cascade ordering.
EVIDENCE_TABLES = (
    "node_retrieval",
    "statement_evidence",
    "statement_property",
    "statement_role",
    "statement",
    "entity_mention",
    "block_element_edge",
    "block_element",
    "block_structure",
    "block_asset",
    "block_representation",
    "tree_node",
    "block",
)


__all__ = [
    "EVIDENCE_SQL",
    "EVIDENCE_TABLES",
    "LEDGER_SQL",
    "SCHEMA_SHA256",
    "SCHEMA_SQL",
]
