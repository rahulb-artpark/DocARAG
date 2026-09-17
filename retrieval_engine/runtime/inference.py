from contextlib import contextmanager
from collections.abc import Iterable, Mapping, Sized
import pandas as pd
from retrieval_engine.configuration.system import load_system_config, SystemConfig
from retrieval_engine.providers.token_tracker import (
    TokenTracker,
    TokenUsageScope,
    current_token_scope,
)
from retrieval_engine.rag import create_rag_agent
from retrieval_engine.rag.base_strategy import BaseRAG
from retrieval_engine.rag.session import RAGSession
from retrieval_engine.runtime.resource_loader import prepare_rag_dependencies
from retrieval_engine.rag.query_cache import (
    AtomicJsonArrayWriter,
    QueryCacheBinding,
    QueryCacheContext,
    atomic_write_json,
    cache_commit_payload,
    question_directory_key,
)

import json
from tqdm import tqdm
from pathlib import Path
import logging
import argparse
import time
from typing import Any, Iterator
from rich.logging import RichHandler
from document_services.operations.run_logging import RunLogScope

log = logging.getLogger(__name__)
# logging.basicConfig(
#     level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
# )


def run_rag(
    rag_agent: BaseRAG,
    output_dir: str,
    force_reprocess: bool = False,
    dataset_path: str = None,
    data_df: pd.DataFrame = None,
    data_iterable: Iterable[Mapping[str, Any]] | None = None,
    *,
    close_agent_on_exit: bool = True,
    token_scope: TokenUsageScope | None = None,
    cache_context: QueryCacheContext | None = None,
):
    """Run a dataset in an isolated token and resource-ownership scope.

    ``close_agent_on_exit=True`` is the documented compatibility default: the
    function takes ownership of the supplied agent and releases it on every
    exit path.  Long-lived services must pass ``False`` explicitly and close
    their borrowed agent elsewhere.  Supplying ``token_scope`` lets a request
    owner retain and inspect the exact ledger after this call.  Without one,
    an already-active request scope is reused; otherwise a fresh run scope is
    created automatically.
    """

    usage = (
        token_scope
        or current_token_scope()
        or TokenTracker.get_instance().scope(name="rag-run")
    )
    with usage.activate():
        usage.begin_stage_sequence()
        with RAGSession(
            rag_agent,
            owns_agent=close_agent_on_exit,
        ) as session:
            return _run_rag(
                rag_agent=session.rag_agent,
                output_dir=output_dir,
                force_reprocess=force_reprocess,
                dataset_path=dataset_path,
                data_df=data_df,
                data_iterable=data_iterable,
                cache_context=cache_context,
            )


def create_rag_session(cfg: SystemConfig) -> RAGSession[BaseRAG]:
    """Construct a fully owned RAG session from one system configuration.

    The dependency loader owns resources until they are passed to
    ``create_rag_agent``; that factory closes all partially constructed models
    and dependencies on failure.  Once an agent is returned, this session is
    the sole owner and protects all later setup/query failures.
    """

    dependencies = prepare_rag_dependencies(cfg=cfg)
    rag_agent = create_rag_agent(
        strategy_config=cfg.rag.strategy_config,
        llm_config=cfg.llm,
        vlm_config=cfg.vlm,
        **dependencies,
    )
    try:
        return RAGSession(rag_agent, owns_agent=True)
    except BaseException:
        # Defensive: a future session constructor may validate more state.
        # Ownership has already transferred from the factory at this point.
        rag_agent.close()
        raise


def create_query_cache_context(cfg: SystemConfig) -> QueryCacheContext | None:
    """Return a verified, secret-free query reuse context for one published run.

    Cache reuse is an optimization, never a correctness requirement.  Legacy
    or incomplete index directories therefore continue to answer queries but
    cannot reuse prior answers until they have a valid artifact publication.
    """

    from document_services.evidence.provenance import verify_query_artifacts

    try:
        if not cfg.save_path:
            raise ValueError("save_path is not configured")
        verification = verify_query_artifacts(Path(cfg.save_path))
        if not verification.valid or verification.artifact_set_sha256 is None:
            details = "; ".join(verification.errors[:3]) or "identity unavailable"
            log.warning(
                "Query cache disabled: artifact verification failed: %s", details
            )
            return None
        return QueryCacheContext.from_system_config(
            cfg,
            artifact_set_sha256=verification.artifact_set_sha256,
        )
    except (OSError, TypeError, ValueError) as exc:
        log.warning("Query cache disabled: safe binding unavailable: %s", exc)
        return None


def _run_rag(
    rag_agent: BaseRAG,
    output_dir: str,
    force_reprocess: bool = False,
    dataset_path: str = None,
    data_df: pd.DataFrame = None,
    data_iterable: Iterable[Mapping[str, Any]] | None = None,
    cache_context: QueryCacheContext | None = None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    output_dir.chmod(0o700)
    log.info("Results will be saved to: %s", output_dir)

    dataset, dataset_len_hint = _dataset_records(
        dataset_path=dataset_path,
        data_df=data_df,
        data_iterable=data_iterable,
    )

    start_time = time.time()
    load_cnt = 0
    dataset_len = 0
    final_res_path = output_dir / "final_results.json"
    with AtomicJsonArrayWriter(final_res_path) as final_results:
        progress = tqdm(dataset, total=dataset_len_hint, desc="Processing Query")
        try:
            for i, raw_item in enumerate(progress):
                dataset_len += 1
                if not isinstance(raw_item, Mapping):
                    raise TypeError(f"Dataset item {i} must be a JSON object")
                item = dict(raw_item)
                query = item.get("question")
                if not query:
                    log.warning("Skipping item %d due to missing 'question' field.", i)
                    continue
                if not isinstance(query, str):
                    raise TypeError(f"Dataset item {i} question must be a string")

                binding = (
                    QueryCacheBinding.create(query, cache_context)
                    if cache_context is not None
                    else None
                )
                directory_key = (
                    binding.cache_key
                    if binding is not None
                    else question_directory_key(query, None)
                )
                query_name = f"query_{directory_key}"
                query_output_dir = output_dir / query_name
                query_result_file = query_output_dir / "result.json"
                query_binding_file = query_output_dir / "cache-binding.json"

                existing_result = None
                if binding is not None and not force_reprocess:
                    existing_result = _load_cached_query_result(
                        query_result_file,
                        query_binding_file,
                        binding,
                    )
                if existing_result is not None:
                    # Cache only generated evidence. Dataset labels, IDs, and
                    # other row metadata always come from the current item.
                    current_result = {
                        **item,
                        "output": existing_result["output"],
                        "retrieved_node_ids": existing_result[
                            "retrieved_node_ids"
                        ],
                    }
                    _publish_query_result(
                        query_result_file,
                        query_binding_file,
                        current_result,
                        binding,
                    )
                    log.info("Reused bound result for %s.", query_name)
                    load_cnt += 1
                    final_results.append(current_result)
                    continue

                query_output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
                query_output_dir.chmod(0o700)
                # A marker is the cache commit point.  Remove any stale marker
                # before generation so a failure cannot make old bytes reusable.
                query_binding_file.unlink(missing_ok=True)
                answer, retrieved_node_ids = rag_agent.generation(
                    query, query_output_dir
                )
                current_result = {
                    **item,
                    "output": answer,
                    "retrieved_node_ids": retrieved_node_ids,
                }
                _publish_query_result(
                    query_result_file,
                    query_binding_file,
                    current_result,
                    binding,
                )
                final_results.append(current_result)
        finally:
            progress.close()

    end_time = time.time()
    total_time = end_time - start_time
    log.info("✅ RAG processing complete in %.2f seconds.", total_time)

    log.info("✅ RAG complete. All results are saved to %s", final_res_path)
    token_tracker = TokenTracker.get_instance()
    rag_cost = token_tracker.record_stage("rag_cost")
    log.info(f"The token cost of RAG in the current document: {rag_cost}")

    update_and_save_cost(
        output_dir=output_dir,
        new_cost=rag_cost,
        new_time=total_time,
        load_cnt=load_cnt,
        dataset_len=dataset_len,
        force_reprocess=force_reprocess,
    )


def _dataset_records(
    *,
    dataset_path: str | None,
    data_df: pd.DataFrame | None,
    data_iterable: Iterable[Mapping[str, Any]] | None,
) -> tuple[Iterable[Mapping[str, Any]], int | None]:
    """Return an iterable of rows without allocating another whole dataset.

    JSONL/NDJSON and caller-provided iterables remain streaming.  A legacy JSON
    array necessarily remains eagerly parsed by ``json.load`` for backward
    compatibility; callers processing large datasets should use JSONL.
    """

    selected = sum(
        source is not None
        for source in (
            dataset_path,
            data_df,
            data_iterable,
        )
    )
    if selected != 1:
        raise ValueError(
            "exactly one of dataset_path, data_df, or data_iterable is required"
        )

    if dataset_path is not None:
        path = Path(dataset_path)
        if not path.is_file():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        if path.suffix.lower() in {".jsonl", ".ndjson"}:
            return _iter_jsonl(path), None
        with path.open("r", encoding="utf-8") as source:
            loaded = json.load(source)
        if not isinstance(loaded, list):
            raise TypeError("Legacy JSON dataset must contain one top-level array")
        return loaded, len(loaded)

    if data_df is not None:
        columns = list(data_df.columns)

        def iter_dataframe() -> Iterator[Mapping[str, Any]]:
            for values in data_df.itertuples(index=False, name=None):
                yield dict(zip(columns, values))

        return iter_dataframe(), len(data_df)

    assert data_iterable is not None
    size = len(data_iterable) if isinstance(data_iterable, Sized) else None
    return data_iterable, size


def _iter_jsonl(path: Path) -> Iterator[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL record at {path}:{line_number}: {exc.msg}"
                ) from exc
            if not isinstance(item, Mapping):
                raise TypeError(
                    f"JSONL record at {path}:{line_number} must be an object"
                )
            yield item


def _load_cached_query_result(
    result_path: Path,
    binding_path: Path,
    expected_binding: QueryCacheBinding,
) -> dict[str, Any] | None:
    """Load only an exactly bound, committed result; otherwise miss safely."""

    if not result_path.is_file() or not binding_path.is_file():
        return None
    if result_path.is_symlink() or binding_path.is_symlink():
        log.warning("Ignoring unsafe symlinked query cache at %s", result_path.parent)
        return None
    try:
        result_bytes = result_path.read_bytes()
        commit = json.loads(binding_path.read_text(encoding="utf-8"))
        expected_commit = cache_commit_payload(expected_binding, result_bytes)
        if commit != expected_commit:
            log.warning("Ignoring mismatched query cache at %s", result_path.parent)
            return None
        result = json.loads(result_bytes)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        log.warning("Ignoring corrupt query cache at %s", result_path.parent)
        return None
    if (
        not isinstance(result, dict)
        or not result.get("output")
        or "retrieved_node_ids" not in result
    ):
        log.warning("Ignoring incomplete query cache at %s", result_path.parent)
        return None
    return result


def _publish_query_result(
    result_path: Path,
    binding_path: Path,
    result: Mapping[str, Any],
    binding: QueryCacheBinding | None,
) -> None:
    """Atomically publish result bytes and then their optional commit marker."""

    binding_path.unlink(missing_ok=True)
    result_bytes = atomic_write_json(result_path, dict(result))
    if binding is not None:
        atomic_write_json(
            binding_path,
            cache_commit_payload(binding, result_bytes),
        )


def update_and_save_cost(
    output_dir: Path,
    new_cost: Mapping[str, int],
    new_time: float,
    load_cnt: int,
    dataset_len: int,
    force_reprocess: bool,
):
    if load_cnt == dataset_len:
        log.info(f"All {load_cnt} samples were loaded from existing results.")
        log.info("Skipping saving token cost since no new inference was made.")
        return

    token_cost_path = output_dir / "token_cost.json"
    previous_cost = {}
    previous_time = 0

    if token_cost_path.exists() and load_cnt != 0 and not force_reprocess:
        log.info(
            f"Found existing cost file at {token_cost_path}. Reading previous values."
        )
        try:
            with open(token_cost_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
            previous_cost = existing_data.get("rag_cost", {})
            previous_time = existing_data.get("time", 0)
            log.info(
                f"Previous cost: {previous_cost}, Previous time: {previous_time:.2f}s"
            )
        except (json.JSONDecodeError, KeyError):
            log.warning(
                f"Could not read or parse existing cost file. Starting from zero."
            )
            previous_cost = {}
            previous_time = 0

    total_rag_cost = previous_cost.copy()
    for key, value in new_cost.items():
        total_rag_cost[key] = total_rag_cost.get(key, 0) + value

    total_processing_time = previous_time + new_time

    final_token_cost = {
        "rag_cost": total_rag_cost,
        "time": total_processing_time,
    }

    log.info(
        f"Saving accumulated cost: {total_rag_cost}, Total time: {total_processing_time:.2f}s"
    )
    atomic_write_json(token_cost_path, final_token_cost)


@contextmanager
def create_log_handler(
    cfg: SystemConfig,
    dataset_path: str,
) -> Iterator[Path]:
    """
    Yield the evaluation directory inside an isolated file-log scope.

    The compatibility name is retained for callers, but this no longer removes
    or adds process-root handlers. Library users keep ownership of their own
    logging configuration.
    """
    rag_strategy = cfg.rag.strategy_config.strategy
    log.info(f"Using RAG strategy: {rag_strategy}")

    dataset_file = Path(dataset_path)
    output_dir = Path(cfg.save_path) / f"eval_{dataset_file.stem}_{rag_strategy}"
    output_dir.mkdir(parents=True, exist_ok=True)
    with RunLogScope(
        output_dir,
        logger_names=("retrieval_engine", "document_services", log.name),
    ) as run_log:
        log.info("Evaluation log: %s", run_log.log_file)
        yield output_dir


def inference_base(cfg: SystemConfig, dataset_path: str):
    with create_log_handler(cfg, dataset_path) as output_dir:
        log.info(
            "Successfully loaded config. Using RAG strategy: %s",
            cfg.rag.strategy_config.strategy,
        )
        cache_context = create_query_cache_context(cfg)
        with create_rag_session(cfg) as session:
            rag_agent = session.rag_agent
            log.info(f"RAG agent created with strategy: {rag_agent.name}")

            run_rag(
                rag_agent=rag_agent,
                dataset_path=dataset_path,
                output_dir=output_dir,
                force_reprocess=True,
                close_agent_on_exit=False,
                cache_context=cache_context,
            )


def inference(cfg: SystemConfig, data_df: pd.DataFrame, dataset_name: str):
    cache_context = create_query_cache_context(cfg)
    with create_rag_session(cfg) as session:
        rag_agent = session.rag_agent
        log.info(f"RAG agent created with strategy: {rag_agent.name}")

        rag_strategy = cfg.rag.strategy_config.strategy
        log.info(f"Using RAG strategy: {rag_strategy}")
        if rag_strategy == "vanilla":
            retrieval_method = cfg.rag.strategy_config.retrieval_method
            output_dir = (
                Path(cfg.save_path) / f"eval_{dataset_name}_{retrieval_method}"
            )
        elif rag_strategy == "gbc":
            variant = cfg.rag.strategy_config.variant
            output_dir = (
                Path(cfg.save_path) / f"eval_{dataset_name}_{rag_strategy}_{variant}"
            )
        else:
            output_dir = Path(cfg.save_path) / f"eval_{dataset_name}_{rag_strategy}"

        # special topk for mmr, graph, and vanilla
        if rag_strategy in ["mmr", "graph", "vanilla"]:
            if hasattr(cfg.rag.strategy_config, "topk"):
                topk = cfg.rag.strategy_config.topk
            elif hasattr(cfg.rag.strategy_config, "topk_docs"):
                topk = cfg.rag.strategy_config.topk_docs
            else:
                topk = 5  # default value if not specified
            output_dir = output_dir.with_name(f"{output_dir.name}_top{topk}")

        output_dir.mkdir(parents=True, exist_ok=True)

        run_rag(
            rag_agent=rag_agent,
            output_dir=output_dir,
            force_reprocess=cfg.rag_force_reprocess,
            data_df=data_df,
            close_agent_on_exit=False,
            cache_context=cache_context,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG evaluation on a dataset.")
    parser.add_argument(
        "--config_path",
        type=str,
        default="/home/wangshu/multimodal/GBC-RAG/config/gbc.yaml",
        # default="/home/wangshu/multimodal/GBC-RAG/config/mm.yaml",
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path to the JSON dataset file with questions.",
        default="/home/wangshu/multimodal/GBC-RAG/test/test_qa/test_samples.json",
        # default="/home/wangshu/multimodal/GBC-RAG/test/sf/case-qa/sel_data_qa.json",
    )
    logging.basicConfig(
        level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
    )

    args = parser.parse_args()
    cfg = load_system_config(args.config_path)
    inference_base(cfg, args.dataset_path)
