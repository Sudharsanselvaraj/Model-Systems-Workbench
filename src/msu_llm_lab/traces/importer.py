"""Importer for authorized, redacted Langfuse JSON and CSV export files."""

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Union

from msu_llm_lab.providers.base import scrub_secrets
from msu_llm_lab.traces.models import ObservationSpan, TraceRecord


class TraceImporter:
    """Parses, sanitizes, and validates Langfuse trace exports."""

    @staticmethod
    def _anonymize_text(text: str) -> str:
        """Removes personal identification (emails, phone numbers) from traces."""
        # Email redaction
        text = re.sub(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "[REDACTED_EMAIL]", text)
        # Malaysian phone numbers (e.g. +60123456789 or 012-3456789)
        text = re.sub(r"(\+?60\d{1,2}[-\s]?\d{7,8}|01\d{1}[-\s]?\d{7,8})", "[REDACTED_PHONE]", text)
        return text

    @classmethod
    def load_from_file(cls, file_path: Union[str, Path]) -> List[TraceRecord]:
        """Loads and parses traces from either a JSON or CSV export file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Trace export file not found: {path}")

        suffix = path.suffix.lower()
        if suffix == ".json":
            return cls.load_json(path)
        elif suffix == ".csv":
            return cls.load_csv(path)
        else:
            raise ValueError(f"Unsupported export format '{suffix}'. Expected .json or .csv.")

    @classmethod
    def load_json(cls, file_path: Path) -> List[TraceRecord]:
        """Parses Langfuse JSON export format."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Malformed Langfuse JSON export at {file_path}: {e}")

        # Langfuse exports can be a top-level list or wrapped under 'data' / 'traces'
        if isinstance(data, dict):
            raw_list = data.get("data") or data.get("traces") or data.get("items")
            if not isinstance(raw_list, list):
                raise ValueError("JSON export must be a list of trace objects or contain a 'data' array.")
        elif isinstance(data, list):
            raw_list = data
        else:
            raise ValueError(f"Unexpected JSON structure in {file_path}. Expected array or object.")

        records: List[TraceRecord] = []
        for idx, item in enumerate(raw_list):
            if not isinstance(item, dict):
                raise ValueError(f"Trace item #{idx} is not a valid dictionary object.")
            record = cls._parse_dict_to_trace(item)
            records.append(record)

        return records

    @classmethod
    def load_csv(cls, file_path: Path) -> List[TraceRecord]:
        """Parses flat Langfuse CSV export format."""
        records: List[TraceRecord] = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames:
                    raise ValueError(f"Empty or headerless CSV file: {file_path}")

                for idx, row in enumerate(reader):
                    record = cls._parse_csv_row_to_trace(row, idx)
                    records.append(record)
        except csv.Error as e:
            raise ValueError(f"CSV parsing error in {file_path}: {e}")

        return records

    @classmethod
    def _parse_dict_to_trace(cls, raw: Dict[str, Any]) -> TraceRecord:
        """Converts raw dictionary item into a sanitized TraceRecord."""
        scrubbed = scrub_secrets(raw)

        trace_id = str(scrubbed.get("id") or scrubbed.get("traceId") or scrubbed.get("trace_id") or "")
        if not trace_id:
            raise ValueError("Trace object missing required 'id' / 'traceId' field.")

        timestamp = str(scrubbed.get("timestamp") or scrubbed.get("createdAt") or scrubbed.get("time") or "")
        session_id = scrubbed.get("sessionId") or scrubbed.get("session_id")
        user_id = "[REDACTED_USER]"

        raw_in = scrubbed.get("input") or scrubbed.get("input_query") or scrubbed.get("prompt") or ""
        if isinstance(raw_in, (dict, list)):
            input_query = json.dumps(raw_in)
        else:
            input_query = str(raw_in)

        raw_out = scrubbed.get("output") or scrubbed.get("output_response") or scrubbed.get("completion") or ""
        if isinstance(raw_out, (dict, list)):
            output_response = json.dumps(raw_out)
        else:
            output_response = str(raw_out)

        input_query = cls._anonymize_text(input_query)
        output_response = cls._anonymize_text(output_response)

        latency_ms = float(scrubbed.get("latency") or scrubbed.get("latency_ms") or scrubbed.get("duration") or 0.0)
        # If latency is in seconds (e.g. 1.25s), normalize to ms
        if 0.0 < latency_ms < 50.0:
            latency_ms = latency_ms * 1000.0

        usage = scrubbed.get("usage") or {}
        total_tokens = int(scrubbed.get("total_tokens") or usage.get("total") or 0)
        prompt_tokens = int(scrubbed.get("prompt_tokens") or usage.get("prompt") or usage.get("input") or 0)
        completion_tokens = int(scrubbed.get("completion_tokens") or usage.get("completion") or usage.get("output") or 0)

        # Parse subordinate observations / spans
        spans: List[ObservationSpan] = []
        raw_obs = scrubbed.get("observations") or scrubbed.get("spans") or scrubbed.get("steps") or []
        if isinstance(raw_obs, list):
            for obs in raw_obs:
                if isinstance(obs, dict):
                    obs_id = str(obs.get("id") or obs.get("observationId") or "")
                    obs_name = str(obs.get("name") or obs.get("type") or "unnamed_span")
                    obs_type = str(obs.get("type") or "span")
                    obs_lat = float(obs.get("latency") or obs.get("latency_ms") or 0.0)
                    if 0.0 < obs_lat < 50.0:
                        obs_lat = obs_lat * 1000.0
                    spans.append(
                        ObservationSpan(
                            id=obs_id,
                            name=obs_name,
                            type=obs_type,
                            latency_ms=obs_lat,
                            input_data=obs.get("input"),
                            output_data=obs.get("output"),
                            metadata=obs.get("metadata") or {},
                        )
                    )

        tags = scrubbed.get("tags") or []
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",") if t.strip()]

        metadata = scrubbed.get("metadata") or {}

        return TraceRecord(
            id=trace_id,
            timestamp=timestamp,
            session_id=str(session_id) if session_id else None,
            user_id=user_id,
            input_query=input_query,
            output_response=output_response,
            latency_ms=latency_ms,
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            spans=spans,
            tags=tags,
            metadata=metadata,
        )

    @classmethod
    def _parse_csv_row_to_trace(cls, row: Dict[str, str], idx: int) -> TraceRecord:
        """Converts flat CSV row into a sanitized TraceRecord."""
        trace_id = row.get("id") or row.get("trace_id") or row.get("traceId") or f"trace-csv-{idx}"
        timestamp = row.get("timestamp") or row.get("createdAt") or row.get("time") or ""

        input_query = cls._anonymize_text(row.get("input") or row.get("query") or row.get("prompt") or "")
        output_response = cls._anonymize_text(row.get("output") or row.get("response") or row.get("completion") or "")

        try:
            latency_ms = float(row.get("latency") or row.get("latency_ms") or 0.0)
            if 0.0 < latency_ms < 50.0:
                latency_ms = latency_ms * 1000.0
        except ValueError:
            latency_ms = 0.0

        try:
            total_tokens = int(row.get("total_tokens") or row.get("tokens") or 0)
        except ValueError:
            total_tokens = 0

        try:
            prompt_tokens = int(row.get("prompt_tokens") or 0)
        except ValueError:
            prompt_tokens = 0

        try:
            completion_tokens = int(row.get("completion_tokens") or 0)
        except ValueError:
            completion_tokens = 0

        raw_tags = row.get("tags") or ""
        tags = [t.strip() for t in raw_tags.split(",") if t.strip()]

        # Generate synthetic spans from CSV columns if retrieval or routing metadata exists
        spans: List[ObservationSpan] = []
        if "retrieval_chunks" in row or "context" in row:
            ctx = row.get("retrieval_chunks") or row.get("context")
            spans.append(
                ObservationSpan(
                    id=f"{trace_id}-retrieval",
                    name="retrieval",
                    type="retrieval",
                    output_data=ctx,
                )
            )
        if "intent" in row or "route" in row:
            route = row.get("intent") or row.get("route")
            spans.append(
                ObservationSpan(
                    id=f"{trace_id}-routing",
                    name="intent_classifier",
                    type="span",
                    output_data=route,
                )
            )

        return TraceRecord(
            id=trace_id,
            timestamp=timestamp,
            session_id=row.get("session_id"),
            user_id="[REDACTED_USER]",
            input_query=input_query,
            output_response=output_response,
            latency_ms=latency_ms,
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            spans=spans,
            tags=tags,
            metadata={"source_format": "csv"},
        )
