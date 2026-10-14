"""Unit tests for Phase 4 Langfuse trace ingestion, anonymization, and failure diagnostics."""

from pathlib import Path
import pytest
from typer.testing import CliRunner

from msu_llm_lab.cli import app
from msu_llm_lab.traces.analyzer import TraceDiagnosticAnalyzer
from msu_llm_lab.traces.importer import TraceImporter
from msu_llm_lab.traces.models import (
    FailureCategory,
    ObservationSpan,
    TraceRecord,
)
from msu_llm_lab.traces.reporter import DiagnosticReporter

runner = CliRunner()


def test_importer_valid_json():
    """TraceImporter correctly parses Langfuse JSON format and normalizes latency."""
    json_path = Path("data/traces/synthetic_langfuse_traces.json")
    traces = TraceImporter.load_from_file(json_path)

    assert len(traces) == 5
    t1 = traces[0]
    assert t1.id == "trace-langfuse-001"
    assert t1.latency_ms == 1420.0  # normalized from 1.42s
    assert t1.total_tokens == 173
    assert len(t1.spans) == 2
    assert t1.spans[0].name == "intent_classifier"


def test_importer_valid_csv():
    """TraceImporter correctly parses Langfuse CSV format."""
    csv_path = Path("data/traces/synthetic_langfuse_traces.csv")
    traces = TraceImporter.load_from_file(csv_path)

    assert len(traces) == 3
    t1 = traces[0]
    assert t1.id == "trace-csv-001"
    assert "Clinical Pharmacy" in t1.input_query
    assert t1.latency_ms == 1350.0
    assert len(t1.spans) >= 1


def test_importer_anonymization():
    """TraceImporter redacts student email addresses and phone numbers."""
    raw_dict = {
        "id": "trace-pii-test",
        "input": "Contact student at john.doe@example.com or 012-3456789 about admission.",
        "output": "Notification sent to john.doe@example.com.",
        "latency": 0.5,
    }
    trace = TraceImporter._parse_dict_to_trace(raw_dict)
    assert "[REDACTED_EMAIL]" in trace.input_query
    assert "john.doe@example.com" not in trace.input_query
    assert "[REDACTED_PHONE]" in trace.input_query
    assert "012-3456789" not in trace.input_query


def test_importer_errors(tmp_path: Path):
    """TraceImporter raises informative ValueErrors on corrupted files."""
    # File not found
    with pytest.raises(FileNotFoundError):
        TraceImporter.load_from_file("non_existent_traces.json")

    # Unsupported format
    bad_ext = tmp_path / "traces.txt"
    bad_ext.write_text("random")
    with pytest.raises(ValueError) as exc:
        TraceImporter.load_from_file(bad_ext)
    assert "Unsupported export format" in str(exc.value)

    # Malformed JSON
    corrupt_json = tmp_path / "bad.json"
    corrupt_json.write_text("{not valid json")
    with pytest.raises(ValueError) as exc:
        TraceImporter.load_json(corrupt_json)
    assert "Malformed Langfuse JSON" in str(exc.value)


def test_analyzer_failure_categories():
    """TraceDiagnosticAnalyzer accurately diagnoses all 4 failure modes and healthy traces."""
    analyzer = TraceDiagnosticAnalyzer()

    # 1. Routing Failure: Postgrad inquiry sent to undergrad route
    t_route = TraceRecord(
        id="t-route",
        timestamp="2026-10-14T01:00:00Z",
        input_query="What are the PhD supervisor requirements in computer science?",
        output_response="For Diploma in IT, you need 3 SPM credits.",
        spans=[
            ObservationSpan(id="s1", name="router", type="span", output_data="undergraduate_admissions")
        ],
    )
    d_route = analyzer.analyze_trace(t_route)
    assert d_route.category == FailureCategory.ROUTING_FAILURE
    assert d_route.confidence >= 0.85

    # 2. Retrieval Failure: Empty chunks
    t_ret = TraceRecord(
        id="t-ret",
        timestamp="2026-10-14T01:00:00Z",
        input_query="What are the scholarship requirements for Bachelor of Design?",
        output_response="I cannot answer that.",
        spans=[
            ObservationSpan(id="s2", name="retrieval", type="retrieval", output_data=[])
        ],
    )
    d_ret = analyzer.analyze_trace(t_ret)
    assert d_ret.category == FailureCategory.RETRIEVAL_FAILURE

    # 3. Source Knowledge Failure: Conflicting figures in KB context
    t_kb = TraceRecord(
        id="t-kb",
        timestamp="2026-10-14T01:00:00Z",
        input_query="What is the PhD tuition fee?",
        output_response="The tuition fee is RM62,500 or RM63,500.",
        spans=[
            ObservationSpan(
                id="s3",
                name="retrieval",
                type="retrieval",
                output_data="The tuition fee is RM62,500. Other docs mention RM63,500.",
            )
        ],
    )
    d_kb = analyzer.analyze_trace(t_kb)
    assert d_kb.category == FailureCategory.SOURCE_KNOWLEDGE_FAILURE

    # 4. Generation Failure: Hallucinated fee not in context
    t_gen = TraceRecord(
        id="t-gen",
        timestamp="2026-10-14T01:00:00Z",
        input_query="What is the Bachelor of Accounting fee?",
        output_response="The fee is RM75,000 per year.",
        spans=[
            ObservationSpan(
                id="s4",
                name="retrieval",
                type="retrieval",
                output_data="Bachelor of Accounting fee is RM35,000 across 3 years.",
            )
        ],
    )
    d_gen = analyzer.analyze_trace(t_gen)
    assert d_gen.category == FailureCategory.GENERATION_FAILURE

    # 5. Healthy trace
    t_healthy = TraceRecord(
        id="t-healthy",
        timestamp="2026-10-14T01:00:00Z",
        input_query="How long is the Foundation in Science programme?",
        output_response="Foundation in Science is a 1-year programme.",
        spans=[
            ObservationSpan(
                id="s5",
                name="retrieval",
                type="retrieval",
                output_data="Foundation in Science is a 1-year programme.",
            )
        ],
    )
    d_healthy = analyzer.analyze_trace(t_healthy)
    assert d_healthy.category == FailureCategory.NO_FAILURE_DETECTED


def test_reporter_export(tmp_path: Path):
    """DiagnosticReporter exports CSV and Markdown reports accurately."""
    analyzer = TraceDiagnosticAnalyzer()
    traces = TraceImporter.load_from_file("data/traces/synthetic_langfuse_traces.json")
    report = analyzer.analyze_batch(traces)

    csv_out = tmp_path / "report.csv"
    md_out = tmp_path / "report.md"

    DiagnosticReporter.export_csv(report, csv_out)
    DiagnosticReporter.export_markdown(report, md_out)

    assert csv_out.exists()
    assert md_out.exists()

    csv_content = csv_out.read_text()
    assert "trace-langfuse-001" in csv_content
    assert "ROUTING_FAILURE" in csv_content

    md_content = md_out.read_text()
    assert "# MSU LLM Workbench — Trace Diagnostic Report" in md_content
    assert "trace-langfuse-001" in md_content


def test_cli_import_traces():
    """CLI import-traces command executes successfully via Typer runner."""
    result = runner.invoke(
        app,
        ["import-traces", "data/traces/synthetic_langfuse_traces.json"],
    )
    assert result.exit_code == 0
    assert "Trace Batch Summary" in result.output
    assert "Failure Category Distribution" in result.output
    assert "ROUTING_FAILURE" in result.output
