"""Declarative PostgreSQL schema for the docarag store: defined once, applied whole."""

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
""".strip()


EVIDENCE_STORE_SQL = r"""
CREATE EXTENSION IF NOT EXISTS ltree WITH SCHEMA public;

CREATE TABLE docarag.assets (
    tenant_id text NOT NULL,
    corpus_id text NOT NULL,
    asset_id char(64) NOT NULL,
    mime_type text NOT NULL,
    byte_size bigint NOT NULL,
    width integer,
    height integer,
    cas_uri text NOT NULL,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (tenant_id, corpus_id, asset_id),
    FOREIGN KEY (tenant_id, corpus_id) REFERENCES docarag.corpora(tenant_id, corpus_id),
    CONSTRAINT assets_asset_id_sha CHECK (asset_id ~ '^[0-9a-f]{64}$'),
    CONSTRAINT assets_byte_size CHECK (byte_size >= 0),
    CONSTRAINT assets_dims CHECK (
        (width IS NULL OR width >= 0) AND (height IS NULL OR height >= 0)
    ),
    CONSTRAINT assets_cas_uri_length CHECK (char_length(cas_uri) BETWEEN 1 AND 32768),
    CONSTRAINT assets_mime_length CHECK (char_length(mime_type) BETWEEN 1 AND 512)
);

CREATE TABLE docarag.canonical_blocks (
    tenant_id text NOT NULL,
    corpus_id text NOT NULL,
    document_id text NOT NULL,
    document_version_id text NOT NULL,
    index_revision_id text NOT NULL,
    block_id text NOT NULL,
    revision_id text NOT NULL,
    block_type text NOT NULL,
    semantic_subtype text,
    page_id text,
    page_index integer NOT NULL,
    reading_order integer,
    parent_block_id text,
    content_sha256 char(64),
    quality_status text,
    reconciliation_method text,
    geometry jsonb,
    source_observation_ids jsonb NOT NULL DEFAULT '[]'::jsonb,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (
        tenant_id, corpus_id, document_id, document_version_id,
        index_revision_id, block_id
    ),
    FOREIGN KEY (tenant_id, corpus_id, document_id, document_version_id)
        REFERENCES docarag.document_versions(
            tenant_id, corpus_id, document_id, document_version_id
        ),
    CONSTRAINT canonical_blocks_page_index CHECK (page_index >= 0),
    CONSTRAINT canonical_blocks_reading_order CHECK (
        reading_order IS NULL OR reading_order >= 0
    ),
    CONSTRAINT canonical_blocks_content_sha CHECK (
        content_sha256 IS NULL OR content_sha256 ~ '^[0-9a-f]{64}$'
    )
);

CREATE INDEX canonical_blocks_page_idx
    ON docarag.canonical_blocks (
        tenant_id, corpus_id, document_version_id, index_revision_id,
        page_index, reading_order
    );
CREATE INDEX canonical_blocks_type_idx
    ON docarag.canonical_blocks (
        tenant_id, corpus_id, document_version_id, index_revision_id, block_type
    );

CREATE TABLE docarag.block_representations (
    tenant_id text NOT NULL,
    corpus_id text NOT NULL,
    document_id text NOT NULL,
    document_version_id text NOT NULL,
    index_revision_id text NOT NULL,
    block_id text NOT NULL,
    representation_no integer NOT NULL,
    kind text NOT NULL,
    content_sha256 char(64),
    content text,
    cas_uri text,
    confidence double precision,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (
        tenant_id, corpus_id, document_id, document_version_id,
        index_revision_id, block_id, representation_no
    ),
    FOREIGN KEY (
        tenant_id, corpus_id, document_id, document_version_id,
        index_revision_id, block_id
    ) REFERENCES docarag.canonical_blocks(
        tenant_id, corpus_id, document_id, document_version_id,
        index_revision_id, block_id
    ) ON DELETE CASCADE,
    CONSTRAINT block_representations_no CHECK (representation_no >= 0),
    CONSTRAINT block_representations_content_sha CHECK (
        content_sha256 IS NULL OR content_sha256 ~ '^[0-9a-f]{64}$'
    ),
    CONSTRAINT block_representations_confidence CHECK (
        confidence IS NULL OR (confidence >= 0 AND confidence <= 1)
    )
);

CREATE TABLE docarag.block_assets (
    tenant_id text NOT NULL,
    corpus_id text NOT NULL,
    document_id text NOT NULL,
    document_version_id text NOT NULL,
    index_revision_id text NOT NULL,
    block_id text NOT NULL,
    asset_id char(64) NOT NULL,
    role text NOT NULL,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (
        tenant_id, corpus_id, document_id, document_version_id,
        index_revision_id, block_id, asset_id, role
    ),
    FOREIGN KEY (
        tenant_id, corpus_id, document_id, document_version_id,
        index_revision_id, block_id
    ) REFERENCES docarag.canonical_blocks(
        tenant_id, corpus_id, document_id, document_version_id,
        index_revision_id, block_id
    ) ON DELETE CASCADE,
    FOREIGN KEY (tenant_id, corpus_id, asset_id)
        REFERENCES docarag.assets(tenant_id, corpus_id, asset_id),
    CONSTRAINT block_assets_asset_sha CHECK (asset_id ~ '^[0-9a-f]{64}$')
);

CREATE INDEX block_assets_asset_idx
    ON docarag.block_assets (tenant_id, corpus_id, asset_id);

CREATE TABLE docarag.tree_nodes (
    tenant_id text NOT NULL,
    corpus_id text NOT NULL,
    document_id text NOT NULL,
    document_version_id text NOT NULL,
    index_revision_id text NOT NULL,
    node_id integer NOT NULL,
    parent_id integer,
    node_path public.ltree NOT NULL,
    node_type text NOT NULL,
    level integer NOT NULL,
    reading_order integer,
    block_id text,
    summary_sha256 char(64),
    summary text,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (
        tenant_id, corpus_id, document_id, document_version_id,
        index_revision_id, node_id
    ),
    FOREIGN KEY (tenant_id, corpus_id, document_id, document_version_id)
        REFERENCES docarag.document_versions(
            tenant_id, corpus_id, document_id, document_version_id
        ),
    CONSTRAINT tree_nodes_node_id CHECK (node_id >= 0),
    CONSTRAINT tree_nodes_parent_id CHECK (parent_id IS NULL OR parent_id >= 0),
    CONSTRAINT tree_nodes_level CHECK (level >= 0),
    CONSTRAINT tree_nodes_summary_sha CHECK (
        summary_sha256 IS NULL OR summary_sha256 ~ '^[0-9a-f]{64}$'
    )
);

CREATE INDEX tree_nodes_path_idx
    ON docarag.tree_nodes USING gist (node_path);
CREATE INDEX tree_nodes_parent_idx
    ON docarag.tree_nodes (
        tenant_id, corpus_id, document_version_id, index_revision_id, parent_id
    );

CREATE TABLE docarag.kg_entities (
    tenant_id text NOT NULL,
    corpus_id text NOT NULL,
    document_id text NOT NULL,
    document_version_id text NOT NULL,
    index_revision_id text NOT NULL,
    entity_id text NOT NULL,
    entity_name text NOT NULL,
    entity_type text NOT NULL,
    description text,
    description_sha256 char(64),
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (
        tenant_id, corpus_id, document_id, document_version_id,
        index_revision_id, entity_id
    ),
    FOREIGN KEY (tenant_id, corpus_id, document_id, document_version_id)
        REFERENCES docarag.document_versions(
            tenant_id, corpus_id, document_id, document_version_id
        ),
    CONSTRAINT kg_entities_desc_sha CHECK (
        description_sha256 IS NULL OR description_sha256 ~ '^[0-9a-f]{64}$'
    )
);

CREATE INDEX kg_entities_type_idx
    ON docarag.kg_entities (
        tenant_id, corpus_id, document_version_id, index_revision_id, entity_type
    );
CREATE INDEX kg_entities_name_idx
    ON docarag.kg_entities (tenant_id, corpus_id, entity_name);

CREATE TABLE docarag.kg_relations (
    tenant_id text NOT NULL,
    corpus_id text NOT NULL,
    document_id text NOT NULL,
    document_version_id text NOT NULL,
    index_revision_id text NOT NULL,
    relation_id text NOT NULL,
    src_entity_id text NOT NULL,
    dst_entity_id text NOT NULL,
    relation_type text NOT NULL,
    directed boolean NOT NULL DEFAULT true,
    weight double precision,
    description text,
    evidence jsonb NOT NULL DEFAULT '[]'::jsonb,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (
        tenant_id, corpus_id, document_id, document_version_id,
        index_revision_id, relation_id
    ),
    FOREIGN KEY (tenant_id, corpus_id, document_id, document_version_id)
        REFERENCES docarag.document_versions(
            tenant_id, corpus_id, document_id, document_version_id
        ),
    CONSTRAINT kg_relations_weight CHECK (weight IS NULL OR weight >= 0)
);

CREATE INDEX kg_relations_src_idx
    ON docarag.kg_relations (
        tenant_id, corpus_id, document_version_id, index_revision_id, src_entity_id
    );
CREATE INDEX kg_relations_dst_idx
    ON docarag.kg_relations (
        tenant_id, corpus_id, document_version_id, index_revision_id, dst_entity_id
    );

CREATE TABLE docarag.gt_links (
    tenant_id text NOT NULL,
    corpus_id text NOT NULL,
    document_id text NOT NULL,
    document_version_id text NOT NULL,
    index_revision_id text NOT NULL,
    entity_id text NOT NULL,
    node_id integer NOT NULL,
    block_id text,
    provenance jsonb NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (
        tenant_id, corpus_id, document_id, document_version_id,
        index_revision_id, entity_id, node_id
    ),
    FOREIGN KEY (tenant_id, corpus_id, document_id, document_version_id)
        REFERENCES docarag.document_versions(
            tenant_id, corpus_id, document_id, document_version_id
        ),
    CONSTRAINT gt_links_node_id CHECK (node_id >= 0)
);

CREATE INDEX gt_links_node_idx
    ON docarag.gt_links (
        tenant_id, corpus_id, document_version_id, index_revision_id, node_id
    );
CREATE INDEX gt_links_entity_idx
    ON docarag.gt_links (
        tenant_id, corpus_id, document_version_id, index_revision_id, entity_id
    );
""".strip()


SCHEMA_SQL = LEDGER_SQL + "\n\n" + EVIDENCE_STORE_SQL

SCHEMA_SHA256 = hashlib.sha256(SCHEMA_SQL.encode("utf-8")).hexdigest()


__all__ = ["SCHEMA_SQL", "SCHEMA_SHA256", "LEDGER_SQL", "EVIDENCE_STORE_SQL"]
