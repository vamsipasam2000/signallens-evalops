from __future__ import annotations

# ruff: noqa: E402, I001

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from textwrap import shorten

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _ensure_project_environment() -> None:
    """Keep `python demo/evalops_demo.py` working when a local venv exists."""
    venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    if not venv_python.exists():
        return

    if Path(sys.executable).resolve() == venv_python.resolve():
        return

    try:
        import pydantic  # type: ignore

        pydantic_major = int(pydantic.__version__.split(".", maxsplit=1)[0])
    except Exception:
        pydantic_major = 0

    if pydantic_major >= 2:
        return

    os.environ["SIGNALLENS_EVALOPS_DEMO_REEXEC"] = "1"
    os.execv(str(venv_python), [str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]])


_ensure_project_environment()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.analysis.models import FailureThresholds
from app.analysis.repositories import InMemoryFailureAnalysisRepository
from app.analysis.service import ErrorAnalysisService
from app.core.errors import RetrievalValidationError
from app.dashboard.repositories import CompositeDashboardRepository
from app.dashboard.service import DashboardService
from app.metrics.repositories import TraceBackedMetricsRepository
from app.metrics.service import MetricsService
from app.quality.models import QualityThresholds
from app.quality.repositories import InMemoryQualityGateRepository
from app.quality.service import QualityGateService
from app.rag.chunking import TokenWindowChunker
from app.rag.embeddings import HashEmbeddingProvider
from app.rag.ingestion import IngestionService
from app.rag.parsers import default_parser_router
from app.retrieval.evaluation import RetrievalEvaluationService
from app.retrieval.leaderboard import InMemoryLeaderboardRepository
from app.retrieval.models import GroundTruthQuery, RetrievalEvaluationSummary, RetrievalQuery
from app.retrieval.retrievers import RepositoryRetriever
from app.retrieval.service import RetrievalService
from app.storage.repositories import InMemoryDocumentRepository
from app.traces.repositories import InMemoryTraceRepository
from app.traces.service import TraceListFilters, TraceService


CHUNK_SIZE_TOKENS = 34
CHUNK_OVERLAP_TOKENS = 6
EMBEDDING_DIMENSIONS = 64
DATASET_NAME = "signalLens-evalops-demo"

DEMO_DOCUMENTS: tuple[dict[str, object], ...] = (
    {
        "name": "01-ingestion-pipeline.md",
        "metadata": {"domain": "rag", "topic": "ingestion", "owner": "eval-platform"},
        "content": (
            "SignalLens EvalOps ingests product policies and engineering notes, parses "
            "them into clean text, splits the text into overlapping token windows, "
            "embeds every chunk, and stores the resulting vectors with metadata. "
            "The ingestion path records parser version, chunker version, token counts, "
            "chunk counts, source metadata, and latency so teams can debug corpus "
            "changes before they affect retrieval quality."
        ),
    },
    {
        "name": "02-trace-observability.md",
        "metadata": {"domain": "observability", "topic": "tracing", "owner": "eval-platform"},
        "content": (
            "A SignalLens retrieval trace captures the request payload, normalized query, "
            "retriever backend, embedding model, retrieved chunk IDs, document IDs, "
            "similarity scores, retrieval latency, total latency, status, and error "
            "message. Trace records make a failed query inspectable instead of anecdotal "
            "and connect online requests to offline evaluation experiments."
        ),
    },
    {
        "name": "03-quality-gates.md",
        "metadata": {"domain": "evalops", "topic": "quality", "owner": "eval-platform"},
        "content": (
            "Quality gates compare retrieval experiments against thresholds for "
            "precision at k, recall at k, mean reciprocal rank, NDCG, retrieval latency, "
            "and similarity score. A passing gate can release a retriever configuration; "
            "a failing gate produces failed checks and remediation guidance such as "
            "review chunking, adjust top k, improve embeddings, or apply metadata filters."
        ),
    },
)


@dataclass(frozen=True)
class DemoStack:
    document_repository: InMemoryDocumentRepository
    trace_repository: InMemoryTraceRepository
    failure_repository: InMemoryFailureAnalysisRepository
    quality_repository: InMemoryQualityGateRepository
    leaderboard_repository: InMemoryLeaderboardRepository
    embedding_provider: HashEmbeddingProvider
    ingestion_service: IngestionService
    retrieval_service: RetrievalService
    retrieval_evaluator: RetrievalEvaluationService
    metrics_service: MetricsService
    error_analysis_service: ErrorAnalysisService
    quality_gate_service: QualityGateService
    trace_service: TraceService
    dashboard_service: DashboardService


def main() -> None:
    stack = build_stack()

    print_header()
    ingestion_results = ingest_documents(stack)
    print_ingestion_section(ingestion_results)
    print_chunking_section(stack, ingestion_results)
    print_embeddings_section(stack, ingestion_results)
    print_vector_storage_section(stack)

    retrieval_result = run_retrieval(stack)
    print_retrieval_section(retrieval_result)

    ground_truth = build_ground_truth(ingestion_results)
    tight_summary = run_retrieval_evaluation(
        stack,
        records=ground_truth,
        run_id="demo-tight-keyword-top1",
        run_name="Tight metadata + keyword boosted",
        top_k=1,
        strategy="keyword_boosted",
    )
    broad_summary = run_retrieval_evaluation(
        stack,
        records=ground_truth,
        run_id="demo-broad-cosine-top3",
        run_name="Broad cosine top-3",
        top_k=3,
        strategy="cosine",
    )
    print_evaluation_section(tight_summary, broad_summary)

    failed_query_error = record_failed_trace(stack)
    print_trace_section(stack, retrieval_result.trace_id, failed_query_error)
    print_metrics_section(stack)

    tight_failures = stack.error_analysis_service.analyze_retrieval_evaluation(
        summary=tight_summary,
        records=ground_truth,
    )
    broad_failures = stack.error_analysis_service.analyze_retrieval_evaluation(
        summary=broad_summary,
        records=ground_truth,
    )
    print_error_analysis_section(tight_failures, broad_failures)

    tight_gate = stack.quality_gate_service.evaluate_retrieval_summary(tight_summary)
    broad_gate = stack.quality_gate_service.evaluate_retrieval_summary(broad_summary)
    print_quality_gate_section(tight_gate, broad_gate)

    print_leaderboard_section(stack)
    print_dashboard_section(stack)
    print_footer()


def build_stack() -> DemoStack:
    document_repository = InMemoryDocumentRepository()
    trace_repository = InMemoryTraceRepository()
    failure_repository = InMemoryFailureAnalysisRepository()
    quality_repository = InMemoryQualityGateRepository()
    leaderboard_repository = InMemoryLeaderboardRepository()
    embedding_provider = HashEmbeddingProvider(
        dimensions=EMBEDDING_DIMENSIONS,
        model_name=f"local-hash-{EMBEDDING_DIMENSIONS}d-demo",
    )
    ingestion_service = IngestionService(
        parser_router=default_parser_router(),
        chunker_factory=lambda chunk_size, overlap: TokenWindowChunker(
            chunk_size_tokens=chunk_size,
            overlap_tokens=overlap,
        ),
        embedding_provider=embedding_provider,
        repository=document_repository,
        default_chunk_size_tokens=CHUNK_SIZE_TOKENS,
        default_chunk_overlap_tokens=CHUNK_OVERLAP_TOKENS,
    )
    retriever = RepositoryRetriever(
        repository=document_repository,
        embedding_provider=embedding_provider,
    )
    retrieval_service = RetrievalService(
        retriever=retriever,
        trace_repository=trace_repository,
        embedding_model=embedding_provider.model_name,
    )
    retrieval_evaluator = RetrievalEvaluationService(
        retrieval_service=retrieval_service,
        backend=retriever.backend_name,
        embedding_model=embedding_provider.model_name,
    )
    metrics_service = MetricsService(
        repository=TraceBackedMetricsRepository(trace_repository=trace_repository),
    )
    error_analysis_service = ErrorAnalysisService(
        repository=failure_repository,
        thresholds=FailureThresholds(
            precision_at_k=0.75,
            recall_at_k=0.75,
            mrr=0.75,
            ndcg=0.75,
            latency_ms=500.0,
            avg_similarity_score=0.0,
        ),
    )
    quality_gate_service = QualityGateService(
        repository=quality_repository,
        thresholds=QualityThresholds(
            precision_at_k=0.8,
            recall_at_k=0.8,
            mrr=0.8,
            ndcg=0.8,
            retrieval_latency_ms=500.0,
            similarity_score=0.0,
        ),
    )
    trace_service = TraceService(repository=trace_repository)
    dashboard_service = DashboardService(
        repository=CompositeDashboardRepository(
            trace_repository=trace_repository,
            leaderboard_repository=leaderboard_repository,
            quality_gate_repository=quality_repository,
        ),
    )
    return DemoStack(
        document_repository=document_repository,
        trace_repository=trace_repository,
        failure_repository=failure_repository,
        quality_repository=quality_repository,
        leaderboard_repository=leaderboard_repository,
        embedding_provider=embedding_provider,
        ingestion_service=ingestion_service,
        retrieval_service=retrieval_service,
        retrieval_evaluator=retrieval_evaluator,
        metrics_service=metrics_service,
        error_analysis_service=error_analysis_service,
        quality_gate_service=quality_gate_service,
        trace_service=trace_service,
        dashboard_service=dashboard_service,
    )


def ingest_documents(stack: DemoStack):
    results = []
    for document in DEMO_DOCUMENTS:
        result = stack.ingestion_service.ingest_bytes(
            document_name=str(document["name"]),
            content=str(document["content"]).encode("utf-8"),
            content_type="text/markdown",
            source="evalops-demo",
            metadata=dict(document["metadata"]),
        )
        results.append(result)
    return results


def run_retrieval(stack: DemoStack):
    return stack.retrieval_service.retrieve(
        RetrievalQuery(
            query="What does SignalLens capture in a retrieval trace?",
            top_k=2,
            metadata_filter={"topic": "tracing"},
            strategy="keyword_boosted",
        ),
        trace_request_payload={
            "demo_step": "retrieval",
            "query": "What does SignalLens capture in a retrieval trace?",
            "top_k": 2,
            "metadata_filter": {"topic": "tracing"},
            "strategy": "keyword_boosted",
        },
    )


def build_ground_truth(ingestion_results) -> list[GroundTruthQuery]:
    document_ids = {
        result.document.name: result.document.document_id
        for result in ingestion_results
    }
    return [
        GroundTruthQuery(
            query_id="ingestion_pipeline",
            query="How does SignalLens turn documents into chunks and embeddings?",
            relevant_document_ids={document_ids["01-ingestion-pipeline.md"]},
            metadata_filter={"topic": "ingestion"},
        ),
        GroundTruthQuery(
            query_id="trace_observability",
            query="What fields are stored in a retrieval trace?",
            relevant_document_ids={document_ids["02-trace-observability.md"]},
            metadata_filter={"topic": "tracing"},
        ),
        GroundTruthQuery(
            query_id="quality_gate",
            query="Which metrics decide whether a retrieval experiment can ship?",
            relevant_document_ids={document_ids["03-quality-gates.md"]},
            metadata_filter={"topic": "quality"},
        ),
    ]


def run_retrieval_evaluation(
    stack: DemoStack,
    *,
    records: list[GroundTruthQuery],
    run_id: str,
    run_name: str,
    top_k: int,
    strategy,
) -> RetrievalEvaluationSummary:
    summary = stack.retrieval_evaluator.evaluate(
        records=records,
        top_k=top_k,
        strategy=strategy,
        run_id=run_id,
        trace_request_payload={
            "demo_step": "retrieval_evaluation",
            "dataset_name": DATASET_NAME,
            "run_name": run_name,
            "top_k": top_k,
            "strategy": strategy,
        },
    )
    entry = stack.retrieval_evaluator.to_leaderboard_entry(
        summary=summary,
        name=run_name,
        dataset_name=DATASET_NAME,
        chunk_size_tokens=CHUNK_SIZE_TOKENS,
        chunk_overlap_tokens=CHUNK_OVERLAP_TOKENS,
        parameters={
            "embedding_dimensions": EMBEDDING_DIMENSIONS,
            "demo": True,
        },
    )
    stack.leaderboard_repository.record(entry)
    return summary


def record_failed_trace(stack: DemoStack) -> str:
    try:
        stack.retrieval_service.retrieve(
            RetrievalQuery(query="   ", top_k=1),
            trace_request_payload={
                "demo_step": "failed_trace_guardrail",
                "reason": "blank query validation",
            },
        )
    except RetrievalValidationError as exc:
        return str(exc)
    return "no error recorded"


def print_header() -> None:
    print("=" * 78)
    print("SignalLens EvalOps end-to-end demo")
    print("One command: python demo/evalops_demo.py")
    print("Services: ingestion, retrieval, evaluation, traces, metrics, analysis, gates, dashboard")
    print("=" * 78)


def print_ingestion_section(ingestion_results) -> None:
    section("1. DOCUMENT INGESTION")
    for result in ingestion_results:
        document = result.document
        print(
            f"- {document.name}: id={short_id(document.document_id)} "
            f"tokens={document.token_count} chunks={document.chunk_count} "
            f"parser={document.parser_version} latency_ms={document.ingestion_latency_ms:.3f}"
        )


def print_chunking_section(stack: DemoStack, ingestion_results) -> None:
    section("2. CHUNKING")
    total_chunks = len(stack.document_repository.list_all_chunks())
    print(
        f"Token-window chunker: size={CHUNK_SIZE_TOKENS}, "
        f"overlap={CHUNK_OVERLAP_TOKENS}, total_chunks={total_chunks}"
    )
    for result in ingestion_results:
        first_chunk = stack.document_repository.list_chunks(result.document.document_id)[0]
        print(
            f"- {result.document.name} chunk[{first_chunk.chunk_index}] "
            f"tokens={first_chunk.token_count} span="
            f"{first_chunk.metadata['start_token']}:{first_chunk.metadata['end_token']}"
        )
        print(f"  preview: {preview(first_chunk.text)}")


def print_embeddings_section(stack: DemoStack, ingestion_results) -> None:
    section("3. EMBEDDINGS")
    first_chunk = stack.document_repository.list_chunks(
        ingestion_results[0].document.document_id
    )[0]
    non_zero_dimensions = sum(1 for value in first_chunk.embedding if value != 0)
    print(f"Embedding provider: {stack.embedding_provider.model_name}")
    print(f"Vector dimensions: {len(first_chunk.embedding)}")
    print(f"Non-zero dimensions in sample chunk: {non_zero_dimensions}")
    print(f"Sample vector head: {[round(value, 4) for value in first_chunk.embedding[:8]]}")


def print_vector_storage_section(stack: DemoStack) -> None:
    section("4. VECTOR STORAGE")
    documents = stack.document_repository.list_documents()
    chunks = stack.document_repository.list_all_chunks()
    print("Backend: memory vector store via InMemoryDocumentRepository + RepositoryRetriever")
    print(f"Stored documents: {len(documents)}")
    print(f"Stored vectorized chunks: {len(chunks)}")
    print(f"Metadata fields: {sorted(chunks[0].metadata.keys())}")


def print_retrieval_section(retrieval_result) -> None:
    section("5. RETRIEVAL")
    print(
        f"Query: {retrieval_result.query}\n"
        f"Backend={retrieval_result.backend} strategy={retrieval_result.strategy} "
        f"top_k={retrieval_result.top_k} latency_ms={retrieval_result.latency_ms:.3f} "
        f"trace_id={short_id(retrieval_result.trace_id)}"
    )
    for rank, chunk in enumerate(retrieval_result.results, start=1):
        print(
            f"- rank={rank} score={chunk.score:.4f} doc={short_id(chunk.document_id)} "
            f"chunk={short_id(chunk.chunk_id)} topic={chunk.metadata.get('topic')}"
        )
        print(f"  text: {preview(chunk.text)}")


def print_evaluation_section(
    tight_summary: RetrievalEvaluationSummary,
    broad_summary: RetrievalEvaluationSummary,
) -> None:
    section("6. EVALUATION")
    print_eval_summary("Tight run", tight_summary)
    print_eval_summary("Broad run", broad_summary)
    print("Interpretation: the broad run finds the right documents but wastes result slots.")


def print_trace_section(
    stack: DemoStack,
    retrieval_trace_id: str | None,
    failed_query_error: str,
) -> None:
    section("7. TRACE CREATION")
    trace_list = stack.trace_service.list(
        filters=TraceListFilters(),
        limit=20,
        offset=0,
        sort_by="timestamp",
        sort_order="desc",
    )
    retrieval_trace = stack.trace_service.get(retrieval_trace_id or "")
    print(f"Traces recorded: {trace_list.total}")
    if retrieval_trace is not None:
        print(
            f"Completed trace: id={short_id(retrieval_trace.trace_id)} "
            f"chunks={len(retrieval_trace.retrieved_chunks)} "
            f"scores={retrieval_trace.similarity_scores} "
            f"latency_ms={retrieval_trace.retrieval_latency_ms:.3f}"
        )
    print(f"Failed trace guardrail: {failed_query_error}")


def print_metrics_section(stack: DemoStack) -> None:
    section("8. METRICS GENERATION")
    snapshot = stack.metrics_service.summarize()
    print(f"Total requests: {snapshot.total_requests}")
    print(f"Success rate: {percent(snapshot.success_rate)}")
    print(f"Error rate: {percent(snapshot.error_rate)}")
    print(f"Avg retrieval latency ms: {snapshot.avg_retrieval_latency_ms:.3f}")
    print(f"P95 retrieval latency ms: {snapshot.p95_retrieval_latency_ms:.3f}")
    print(f"Avg chunks returned: {snapshot.avg_chunks_returned:.3f}")


def print_error_analysis_section(tight_failures, broad_failures) -> None:
    section("9. ERROR ANALYSIS")
    print(f"Tight run failures: {len(tight_failures)}")
    print(f"Broad run failures: {len(broad_failures)}")
    for failure in broad_failures[:4]:
        print(
            f"- {failure.failure_type}: {failure.metric_name}="
            f"{failure.metric_value:.4f} threshold={failure.threshold:.4f}"
        )
        print(f"  query: {preview(failure.query, width=86)}")
        print(f"  recommendation: {failure.recommendation}")


def print_quality_gate_section(tight_gate, broad_gate) -> None:
    section("10. QUALITY GATE EXECUTION")
    for label, gate in (("Tight run", tight_gate), ("Broad run", broad_gate)):
        print(f"{label}: {gate.status} failed_checks={len(gate.failed_checks)}")
        for check in gate.failed_checks:
            print(
                f"- {check.metric}: actual={check.actual:.4f} "
                f"required={check.required:.4f} reason={check.reason}"
            )


def print_leaderboard_section(stack: DemoStack) -> None:
    section("11. LEADERBOARD GENERATION")
    entries = stack.leaderboard_repository.list_entries(limit=5)
    for rank, entry in enumerate(entries, start=1):
        print(
            f"{rank}. {entry.name} | top_k={entry.top_k} strategy={entry.retrieval_strategy} "
            f"precision={entry.precision_at_k:.4f} recall={entry.recall_at_k:.4f} "
            f"mrr={entry.mrr:.4f} ndcg={entry.ndcg:.4f} "
            f"latency_ms={entry.avg_latency_ms:.3f}"
        )


def print_dashboard_section(stack: DemoStack) -> None:
    section("12. DASHBOARD SUMMARY")
    summary = stack.dashboard_service.summary()
    top_entries = stack.dashboard_service.leaderboard(
        limit=1,
        sort_by="precision_at_k",
        sort_order="desc",
    )
    print(f"Total traces: {summary.total_traces}")
    print(f"Successful requests: {summary.successful_requests}")
    print(f"Failed requests: {summary.failed_requests}")
    print(f"Dashboard success rate: {percent(summary.success_rate)}")
    print(f"Quality checks: {summary.total_quality_checks}")
    print(f"Failed quality gates: {summary.failed_quality_gates}")
    if top_entries:
        top = top_entries[0]
        print(
            f"Best visible experiment: {top.name} "
            f"precision={top.precision_at_k:.4f} gate={top.quality_gate_status}"
        )


def print_footer() -> None:
    print("\n" + "=" * 78)
    print(
        "Demo complete: one corpus, two retrieval runs, traces, metrics, gate, "
        "leaderboard, dashboard."
    )
    print("=" * 78)


def print_eval_summary(label: str, summary: RetrievalEvaluationSummary) -> None:
    print(
        f"{label}: run_id={summary.run_id} top_k={summary.top_k} strategy={summary.strategy} "
        f"precision={summary.precision_at_k:.4f} recall={summary.recall_at_k:.4f} "
        f"mrr={summary.mrr:.4f} ndcg={summary.ndcg:.4f} "
        f"avg_latency_ms={summary.avg_latency_ms:.3f}"
    )


def section(title: str) -> None:
    print("\n" + "-" * 78)
    print(title)
    print("-" * 78)


def short_id(value: str | None) -> str:
    if not value:
        return "none"
    return value[:8]


def preview(text: str, *, width: int = 96) -> str:
    return shorten(" ".join(text.split()), width=width, placeholder="...")


def percent(value: float) -> str:
    return f"{value * 100:.2f}%"


if __name__ == "__main__":
    main()
