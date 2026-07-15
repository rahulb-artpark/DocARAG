import logging
import argparse
import os
from contextlib import contextmanager
import pandas as pd
import yaml
from pathlib import Path
import time
from typing import Iterator
from urllib.parse import urlsplit
from rich.logging import RichHandler

from Core.configs.system_config import load_system_config, SourceIdentity, SystemConfig
from Core.configs.dataset_config import load_dataset_config, DatasetConfig
from Core.construct_index import (
    construct_GBC_index,
    construct_vdb,
    compute_mm_reranker,
    rebuild_graph_vdb,
)
from Core.inference import inference
from Core.pipelines.doc_tree_builder import CanonicalIngestionProvider
from docarag.run_logging import RunLogInfo, RunLogScope
from docarag.sharding import document_shard_number

log = logging.getLogger(__name__)  # Get logger for main

_SENSITIVE_CONFIG_KEYS = {
    "api_key",
    "authorization",
    "password",
    "secret",
    "access_token",
    "refresh_token",
}


def redact_config(value):
    """Return a log-safe copy without changing the runtime configuration."""
    if isinstance(value, dict):
        return {
            key: (
                "***REDACTED***"
                if str(key).casefold() in _SENSITIVE_CONFIG_KEYS
                else redact_config(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_config(item) for item in value]
    return value


def write_config_snapshot(path: Path, config: SystemConfig) -> None:
    """Atomically write, or safely reuse, an owner-only config snapshot."""

    payload = yaml.safe_dump(config.model_dump(), allow_unicode=True)
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing != payload:
            raise RuntimeError(
                f"Refusing to overwrite a different run configuration: {path}"
            )
        path.chmod(0o400)
        return

    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(temporary, "x", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o400)
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def create_args():
    """
    Configures the command-line arguments for the project.

    This function sets up a main parser with global options (like config paths)
    and then adds subparsers for the two main modes of operation: 'index' and 'rag'.
    """
    # --- Main Parser ---
    # This parser handles global arguments that are common to all commands.
    parser = argparse.ArgumentParser(
        description="A command-line interface for building an index or running RAG inference.",
        formatter_class=argparse.RawTextHelpFormatter,  # For better help text formatting
    )

    # Global arguments required by both 'index' and 'rag' commands
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to the main system configuration file (e.g., config/main_config.yaml).",
    )
    parser.add_argument(
        "-d",
        "--dataset_config",
        type=str,
        required=False,
        help="(Optional) Path to the dataset configuration file for batch processing.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode for more verbose logging and error output.",
    )

    parser.add_argument(
        "--nsplit",
        type=int,
        default=2,
        help="The total number of splits for parallel processing.",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=2,
        help="The current split number (1-indexed) to process.",
    )

    # --- Subparsers for Commands ---
    # This will hold the subparsers for 'index' and 'rag'
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Available commands"
    )

    # --- 'index' Command ---
    # The parser for the 'index' command
    parser_index = subparsers.add_parser(
        "index",
        help="Build an search index from the documents specified in the dataset config.",
    )
    # You can add arguments specific to indexing here if needed in the future
    # For example:
    # parser_index.add_argument("--force-rebuild", action="store_true", help="Force rebuilding the index if it already exists.")

    # --- 'rag' Command ---
    # The parser for the 'rag' (inference) command
    parser_rag = subparsers.add_parser(
        "rag", help="Run RAG inference using a pre-built index."
    )
    # You can add arguments specific to inference here
    # For example, to run a single query from the command line:
    parser_rag.add_argument(
        "-q",
        "--query",
        type=str,
        help="(Optional) A single query to run. If not provided, runs queries from the dataset config.",
    )

    # --- ADD THIS ARGUMENT ---
    # This allows the user to select a specific pipeline stage to execute.
    parser_index.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["tree", "graph", "vdb", "all", "mm_reranker", "rebuild_graph_vdb"],
        help="Specify which stage of the indexing pipeline to run: "
        "'tree' - Build and save the document tree only. "
        "'graph' - Build and save the knowledge graph (requires a tree). "
        "'vdb' - Build and save the vector database (requires a tree). "
        "'all' - Run all stages sequentially."
        "'mm_reranker' - Build and save the multi-modal reranker (requires a tree). "
        "'rebuild_graph_vdb' - Rebuild the graph and vector database (requires GBC Index).",
    )

    return parser.parse_args()


def build_index(
    config: SystemConfig,
    stage: str = "all",
    data_df: pd.DataFrame = None,
    *,
    parsing_provider: CanonicalIngestionProvider | None = None,
):
    """Build BookRAG artifacts from the legacy or injected ingestion path.

    ``parsing_provider=None`` preserves the supported MinerU compatibility
    behavior.  A composition root may inject a routed canonical provider.  In
    an ``all`` build the provider is consumed only by the tree stage; graph and
    vector stages read the already source-bound tree and therefore never parse
    the document a second or third time.
    """
    log.info(
        f"  - build_index called. Indexing '{config.pdf_path}' into '{config.save_path}'"
    )

    # Stage 1: Build the Document Tree
    if stage in ["tree", "all"]:
        log.info("  - STAGE: Building Document Tree...")
        # This function should build the tree and save it to config.save_path
        construct_GBC_index(
            config,
            tree_only=True,
            parsing_provider=parsing_provider,
        )

    # Stage 2: Build the Knowledge Graph
    if stage in ["graph", "all"]:
        log.info("  - STAGE: Building Knowledge Graph...")
        # This function should LOAD the pre-existing tree and then build/save the graph
        graph_flag = stage == "graph"
        construct_GBC_index(
            config,
            graph_only=graph_flag,
            parsing_provider=(parsing_provider if stage == "graph" else None),
        )

    # Stage 3: Build the Vector Database
    if stage in ["vdb", "all"]:
        log.info("  - STAGE: Building Vector Database...")
        # This function should LOAD the pre-existing tree and then build/save the VDB
        construct_vdb(
            config,
            parsing_provider=(parsing_provider if stage == "vdb" else None),
        )

    if stage == "mm_reranker":
        log.info("  - STAGE: Building MM Reranker Embedding...")
        compute_mm_reranker(
            config,
            data_df,
            parsing_provider=parsing_provider,
        )
    
    if stage == "rebuild_graph_vdb":
        log.info("  - STAGE: Rebuilding Graph VDB...")
        rebuild_graph_vdb(config)


@contextmanager
def routed_parsing_provider(stage: str) -> Iterator[CanonicalIngestionProvider | None]:
    """Build the explicitly selected routed runtime for the parse-bearing stage.

    The detached controller exports an immutable profile snapshot.  Every
    digest is revalidated inside the BookRAG interpreter before any worker is
    contacted.  Graph/vector-only subprocesses deliberately reuse the verified
    canonical tree without reacquiring parser models or GPU memory.
    """

    profile_path = os.environ.get("DOCARAG_ROUTED_PROFILE")
    if not profile_path or stage not in {"tree", "all"}:
        yield None
        return

    from docarag.routed_profile import load_routed_runtime_profile
    from docarag.runtime_composition import build_balanced_routed_runtime

    profile = load_routed_runtime_profile(profile_path)
    issues = profile.configuration_issues()
    if issues:
        rendered = "; ".join(
            f"{item.runtime_id}.{item.field}: {item.message}" for item in issues
        )
        raise RuntimeError(
            "routed parser profile is not configuration-ready: " + rendered
        )
    expected = (
        ("profile_id", os.environ.get("DOCARAG_ROUTED_PROFILE_ID"), profile.profile_id),
        (
            "source_sha256",
            os.environ.get("DOCARAG_ROUTED_PROFILE_SOURCE_SHA256"),
            profile.source_sha256,
        ),
        (
            "configuration_sha256",
            os.environ.get("DOCARAG_ROUTED_PROFILE_CONFIGURATION_SHA256"),
            profile.configuration_sha256,
        ),
    )
    for field_name, supplied, actual in expected:
        if not supplied:
            raise RuntimeError(
                f"routed indexing is missing immutable {field_name} binding"
            )
        if supplied != actual:
            raise RuntimeError(
                f"routed parser profile {field_name} differs from the admitted revision"
            )

    runtime = build_balanced_routed_runtime(profile.config)
    try:
        identities = ", ".join(
            f"{item.adapter_name}@{item.adapter_version}"
            for item in runtime.worker_identities
        )
        log.info(
            "Routed canonical ingestion ready from profile %s (%s)",
            profile.profile_id,
            identities,
        )
        yield runtime.provider
    finally:
        runtime.close()


def run_inference(config: SystemConfig, data_df: pd.DataFrame, dataset_name: str):
    log.info(f"  - run_inference called. Using index from '{config.save_path}'")
    inference(
        cfg=config,
        data_df=data_df,
        dataset_name=dataset_name,
    )


@contextmanager
def setup_logging(
    save_path: str,
    config_to_log: SystemConfig,
) -> Iterator[RunLogInfo]:
    """
    Isolate this run's logs without altering the process root logger.

    File records carry a context-local run ID, so concurrent requests cannot
    leak into another document's log.  An embedding application that already
    owns console logging remains in control of it.

    :param save_path: The base directory for the current run, where logs will be saved.
    :param config_to_log: The configuration object to be logged at the start.
    """
    root_logger = logging.getLogger()
    console_handler = None
    if not root_logger.handlers:
        console_handler = RichHandler(rich_tracebacks=True, show_path=False)

    scope = RunLogScope(
        save_path,
        logger_names=("Core", "docarag", log.name),
        console_handler=console_handler,
    )
    with scope as run_info:
        log.info(
            "Logging initialized for run %s. Log file: %s",
            run_info.run_id,
            run_info.log_file,
        )
        config_dict = redact_config(config_to_log.model_dump())
        config_yaml_string = yaml.safe_dump(
            config_dict,
            allow_unicode=True,
            default_flow_style=False,
        )
        log.info("--- Starting Run with Configuration ---\n%s", config_yaml_string)
        try:
            yield run_info
        except BaseException:
            log.exception("Run %s failed", run_info.run_id)
            raise


def process_document(
    base_system_cfg: SystemConfig,
    dataset_cfg: DatasetConfig,
    args,
    doc_uuid,
    doc_path,
    group: pd.DataFrame,
) -> None:
    """Process one document, allowing the caller to aggregate any failure."""
    current_config = base_system_cfg.model_copy(deep=True)
    pdf_full_path = Path(doc_path)
    output_full_path = Path(dataset_cfg.working_dir) / str(doc_uuid)

    current_config.pdf_path = str(pdf_full_path)
    current_config.save_path = str(output_full_path)
    first_row = group.iloc[0]
    original_uri = first_row.get("source_uri")
    display_name = first_row.get("source_name")
    source_sha256 = first_row.get("source_sha256")
    if not isinstance(original_uri, str) or not original_uri:
        original_uri = pdf_full_path.expanduser().resolve().as_uri()
    elif not urlsplit(original_uri).scheme:
        original_uri = Path(original_uri).expanduser().resolve().as_uri()
    if not isinstance(display_name, str) or not display_name:
        display_name = Path(original_uri).name or pdf_full_path.name
    if not isinstance(source_sha256, str) or not source_sha256:
        source_sha256 = None
    document_id = first_row.get("document_id")
    document_version_id = first_row.get("document_version_id")
    index_revision_id = first_row.get("index_revision_id")
    tenant_id = first_row.get("tenant_id")
    corpus_id = first_row.get("corpus_id")
    if not isinstance(index_revision_id, str) or not index_revision_id:
        index_revision_id = str(doc_uuid)
    if str(doc_uuid) != index_revision_id:
        raise ValueError("dataset doc_uuid must equal index_revision_id")
    for field_name, supplied, configured in (
        ("tenant_id", tenant_id, current_config.vdb.tenant_id),
        ("corpus_id", corpus_id, current_config.vdb.corpus_id),
    ):
        if isinstance(supplied, str) and supplied and supplied != configured:
            raise ValueError(
                f"dataset {field_name} differs from the configured PostgreSQL/Qdrant scope"
            )

    if source_sha256 is not None:
        from docarag.domain.evidence import DocumentRef

        expected_document = DocumentRef.identified(
            content_sha256=source_sha256,
            mime_type="application/pdf",
            source_uri=original_uri,
            display_name=display_name,
        )
        supplied_ids = (document_id, document_version_id)
        if any(isinstance(value, str) and value for value in supplied_ids):
            if (
                document_id != expected_document.document_id
                or document_version_id != expected_document.document_version_id
            ):
                raise ValueError(
                    "dataset document IDs do not match source_uri and source_sha256"
                )
        document_id = expected_document.document_id
        document_version_id = expected_document.document_version_id
        for field_name, configured, expected in (
            ("document_id", current_config.vdb.document_id, document_id),
            (
                "document_version_id",
                current_config.vdb.document_version_id,
                document_version_id,
            ),
            (
                "index_revision_id",
                current_config.vdb.index_revision_id,
                index_revision_id,
            ),
        ):
            if configured is not None and configured != expected:
                raise ValueError(
                    f"configured {field_name} differs from the dataset/source identity"
                )
        current_config.vdb.document_id = document_id
        current_config.vdb.document_version_id = document_version_id
        current_config.vdb.index_revision_id = index_revision_id
    elif any(
        isinstance(value, str) and value for value in (document_id, document_version_id)
    ):
        raise ValueError("dataset document IDs require source_sha256")
    current_config.source = SourceIdentity(
        display_name=display_name,
        original_uri=original_uri,
        sha256=source_sha256,
        snapshot_path=str(pdf_full_path),
        document_id=document_id if isinstance(document_id, str) else None,
        document_version_id=(
            document_version_id if isinstance(document_version_id, str) else None
        ),
        index_revision_id=index_revision_id,
    )

    output_full_path.mkdir(parents=True, exist_ok=True)
    with setup_logging(
        save_path=current_config.save_path,
        config_to_log=current_config,
    ):
        log.info(f"--- Processing Document UUID: {doc_uuid} ---")
        log.info(f"  - PDF Path: {pdf_full_path}")
        log.info(f"  - Save Path: {output_full_path}")

        data_df = group.reset_index(drop=True)
        if args.command == "index":
            config_snapshot_path = output_full_path / "run_config.yaml"
            write_config_snapshot(config_snapshot_path, current_config)
            log.info(f"  - Saved index config snapshot to: {config_snapshot_path}")
            with routed_parsing_provider(args.stage) as parsing_provider:
                build_index(
                    config=current_config,
                    stage=args.stage,
                    data_df=data_df,
                    parsing_provider=parsing_provider,
                )
            return

        rag_strategy = current_config.rag.strategy_config.strategy
        config_snapshot_path = output_full_path / f"rag_config_{rag_strategy}.yaml"
        write_config_snapshot(config_snapshot_path, current_config)
        log.info(f"  - Saved RAG config snapshot to: {config_snapshot_path}")
        run_inference(
            config=current_config,
            data_df=data_df,
            dataset_name=dataset_cfg.dataset_name,
        )


def main() -> int:
    """
    Run the requested command and return a process exit code.

    Batch document failures are collected so that one bad document does not prevent
    the remaining documents in the split from being attempted.  A non-zero result is
    returned after the split finishes if any document failed.
    """
    args = create_args()

    log.info("--- Arguments Loaded ---")
    log.info(f"Main Config Path: {args.config}")
    log.info(f"Selected Command: {args.command}")
    log.info("------------------------\n")

    base_system_cfg: SystemConfig = load_system_config(args.config)

    if args.dataset_config:
        if (
            isinstance(args.nsplit, bool)
            or not isinstance(args.nsplit, int)
            or args.nsplit < 1
            or isinstance(args.num, bool)
            or not isinstance(args.num, int)
            or args.num < 1
            or args.num > args.nsplit
        ):
            print(
                f"  - ERROR: --num ({args.num}) must be between 1 and "
                f"--nsplit ({args.nsplit}), and --nsplit must be positive."
            )
            return 1

        dataset_cfg: DatasetConfig = load_dataset_config(args.dataset_config)

        # 1. Load the entire dataset from the JSON file into a pandas DataFrame
        log.info(f"  - Loading dataset from: {dataset_cfg.dataset_path}")
        try:
            df = pd.read_json(dataset_cfg.dataset_path)
            print(f"Dataset shape: {df.shape}")
        except FileNotFoundError:
            log.error(f"ERROR: Dataset file not found at '{dataset_cfg.dataset_path}'")
            return 1
        except Exception as e:
            log.error(f" ERROR: Failed to parse JSON file. Reason: {e}")
            return 1

        # 2. Group by document identifiers to find unique documents
        document_groups = df.groupby(["doc_uuid", "doc_path"], sort=False)
        total_docs = document_groups.ngroups
        selected_docs = sum(
            1
            for doc_uuid, doc_path in document_groups.indices
            if document_shard_number(doc_uuid, doc_path, args.nsplit) == args.num
        )
        print(f"  - Found {total_docs} unique documents in the dataset.")

        print(
            f" This worker (stable shard {args.num}/{args.nsplit}) will process "
            f"{selected_docs} documents."
        )

        # 3. Loop over each unique document group
        index_error_list = []
        rag_error_list = []

        for (doc_uuid, doc_path), group in document_groups:
            if document_shard_number(doc_uuid, doc_path, args.nsplit) != args.num:
                continue
            try:
                process_document(
                    base_system_cfg=base_system_cfg,
                    dataset_cfg=dataset_cfg,
                    args=args,
                    doc_uuid=doc_uuid,
                    doc_path=doc_path,
                    group=group,
                )
            except Exception as e:
                if args.command == "index":
                    log.error(f"  - ERROR: Failed to build index. Reason: {e}")
                    index_error_list.append((doc_uuid, str(e)))
                else:
                    log.error(f"  - ERROR: Failed to run inference. Reason: {e}")
                    rag_error_list.append((doc_uuid, str(e)))

        # get the base directory of the script
        import os
        base_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_name = dataset_cfg.dataset_name
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        if index_error_list:
            error_log_path = (
                Path(base_dir)
                / f"{dataset_name}-index_error_split_{args.num}_{timestamp}.txt"
            )
            with open(error_log_path, "w", encoding="utf-8") as f:
                for item in index_error_list:
                    f.write(f"{item}\n")
            log.error(
                f"  - Indexing completed with errors. See {error_log_path} for details."
            )

        if rag_error_list:
            error_log_path = (
                Path(base_dir)
                / f"{dataset_name}-rag_error_split_{args.num}_{timestamp}.txt"
            )
            with open(error_log_path, "w", encoding="utf-8") as f:
                for item in rag_error_list:
                    f.write(f"{item}\n")
            log.error(
                f"  - RAG completed with errors. See {error_log_path} for details."
            )
        log.info(f"--- All documents in this split processed. ---\n")

        failure_count = len(index_error_list) + len(rag_error_list)
        if failure_count:
            log.error(
                "  - Split %s/%s completed with %s document failure(s).",
                args.num,
                args.nsplit,
                failure_count,
            )
            return 1

        return 0
    else:
        # SINGLE FILE MODE

        with setup_logging(
            save_path=base_system_cfg.save_path,
            config_to_log=base_system_cfg,
        ):
            log.info(f"🚀 No dataset config provided. Starting SINGLE mode...")
            log.info(f"  - Using paths and settings from '{args.config}'")

            if args.command == "index":
                with routed_parsing_provider(args.stage) as parsing_provider:
                    build_index(
                        config=base_system_cfg,
                        stage=args.stage,
                        parsing_provider=parsing_provider,
                    )

            elif args.command == "rag":
                # Check if the --query argument was provided
                if args.query:
                    log.info(f"  - Running inference on single query: '{args.query}'")
                    query_df = pd.DataFrame([{"question": args.query}])
                    run_inference(
                        config=base_system_cfg,
                        data_df=query_df,
                        dataset_name="single",
                    )
                else:
                    log.error(
                        "  - ERROR: RAG command in single mode requires a query."
                    )
                    log.error("  - Please provide one with the -q/--query argument.")
                    return 1

        return 0


if __name__ == "__main__":
    raise SystemExit(main())
