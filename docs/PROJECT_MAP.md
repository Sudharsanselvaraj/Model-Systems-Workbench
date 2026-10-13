# Project Map: MSU LLM Research Workbench (`msu-llm-lab`)

**Status**: Phase 0 — Workspace Investigation (Approved with Corrections)  
**Date**: October 2026  
**Workspace Path**: `/Users/sudharsan/Documents/Times Media/LLM Workbench`

---

## 1. Confirmed Current Workspace State

### 1.1 Physical File Inventory

The workspace contains two raw knowledge-base source files:

| File Name | Path | Size | Line Count | Initial SHA-256 Hash |
| :--- | :--- | :--- | :--- | :--- |
| `Copy of MSU General KB - Local - Sudarshan.md` | [General KB](file:///Users/sudharsan/Documents/Times%20Media/LLM%20Workbench/Copy%20of%20MSU%20General%20KB%20-%20Local%20-%20Sudarshan.md) | 111,810 bytes | 2,097 lines | `138e6893a1d54f36cfb2b7abe33317c16647769eb016e71c35f1071a741cfc72` |
| `Copy of MSU Programmes KB - Local - Sudharsan.md` | [Programmes KB](file:///Users/sudharsan/Documents/Times%20Media/LLM%20Workbench/Copy%20of%20MSU%20Programmes%20KB%20-%20Local%20-%20Sudharsan.md) | 961,958 bytes | 17,526 lines | `a46773fea196d8fe6529f5d6d8f6adb9d26ca5a6511185004822f249b2d3282b` |

*Source File Integrity Rule*: Both files are strictly read-only. Hashes must be checked before and after all audit operations to guarantee zero modification.

### 1.2 Git & Version Control State
- **Status**: Not currently a Git repository (`fatal: not a git repository`).
- **Phase 1 Prerequisite**: Initialize Git repository (`git init`) and apply a strict `.gitignore` *before* staging files, ensuring no secrets, local environments, traces, or unapproved data are tracked.

### 1.3 Python & Tooling Environment
- **System Python**: `3.11.7` (`/Library/Frameworks/Python.framework/Versions/3.11/bin/python3`) — fulfills the Python 3.11 requirement.
- **Dependency Managers Available**:
  - `uv` is installed at `/opt/homebrew/bin/uv` (recommended for fast, reproducible venv and dependency resolution).
  - Standard `pip3` is installed at `/Library/Frameworks/Python.framework/Versions/3.11/bin/pip3`.
  - `poetry` is not installed.
- **Virtual Environment**: Isolated project virtualenv (`.venv`) to be initialized in Phase 1.
- **Test Framework**: `pytest`.

### 1.4 Configuration & Secrets State
- No `.env` or credential files exist in the repository.
- `OPENROUTER_API_KEY`: Not exported in current shell; deferred until Phase 3 where external provider benchmarking is implemented.
- `LANGFUSE_*`: No Langfuse keys configured.
- **Rule**: Secrets are read exclusively from environment variables or non-committed local `.env`. Never commit, print, or log credentials.

---

## 2. Knowledge Base Structure & Discovered Formatting Quirks

### 2.1 Counting Distinction: Marker Counts vs Parsed Records vs Confirmed Defects
It is critical to distinguish three separate metrics:
1. **Raw Marker Counts**: Literal occurrences of regex patterns (e.g., `\*\*Question:\*\*` or `\*\*Answer:\*\*`) in the raw text.
2. **Parsed Records**: Structural Q&A entities extracted by the parser after accounting for inline tags, multi-line blocks, unclosed headers, and formatting artifacts.
3. **Confirmed Defects**: Validated defects verified by human review or deterministic schema violations.

**Under no circumstances should preliminary regex marker counts be hardcoded as expected audit ground truth.**

### 2.2 General Knowledge Base Preliminary Observations
- **Raw Marker Counts**: 225 `Question` markers vs 222 `Answer` markers.
- **Heading Styles**: Mixture of escaped markdown headings (`# \# General Information`, `## \#\# Legitimacy`) and raw markdown headings (`## `).
- **Dangling Tokens & Incomplete Edits**: Line 16 contains a stray token `cou` immediately following an answer on line 15.
- **Empty Section Headers**: Lines 42 and 598 contain empty `## ` headings with no titles.

### 2.3 Programmes Knowledge Base Preliminary Observations
- **Raw Marker Counts**: 2,807 `Question` markers vs 2,813 `Answer` markers.
- **Exact Duplicate Q&As**: Lines 87–93 and Lines 95–101 are verbatim duplicates ("What is the job demand for English language and literature graduates in Malaysia?").
- **Inline Tag Formatting**: Many entries format Question and Answer on consecutive lines, while others place them inline or wrap them across multiple line breaks (e.g. `\*\*Question:\*\* What is the Bachelor...\n Studies? \*\*Answer:\*\* The English language program...`).
- **Data Inconsistencies**: Known issues noted in onboarding records (e.g. PhD in Applied Science fee stating RM62,500 and RM63,500 in different parts of the same entry).
- **Dividers**: Long hyphens (`\------------------------------------------------------------------------`) separating major degree programme blocks.

---

## 3. Target Repository Layout (`msu-llm-lab`)

```text
LLM Workbench/
├── .gitignore                          # Excludes .env, raw traces, venv, cache, internal data
├── pyproject.toml                      # Project metadata, CLI entry points, dependencies
├── README.md                           # Quickstart, CLI documentation, and usage guidelines
├── docs/                               # System documentation
│   ├── PROJECT_MAP.md                  # This file
│   ├── ARCHITECTURE.md                 # System architecture and data flow
│   └── IMPLEMENTATION_PLAN.md          # Multi-phase engineering roadmap
├── src/
│   └── msu_llm_lab/                    # Core Python package
│       ├── __init__.py                 # Version and package root
│       ├── cli.py                      # Typer CLI application root
│       ├── audit/                      # Phase 1: Knowledge-base audit engine
│       │   ├── __init__.py
│       │   ├── parser.py               # Robust markdown & escaped-tag parser
│       │   ├── validator.py            # Deterministic checks & heuristic detectors
│       │   └── reporter.py             # CSV and Markdown report generators
│       ├── schemas/                    # Phase 2: Pydantic schemas
│       │   ├── __init__.py
│       │   ├── evaluation.py           # EvaluationCase, VerificationStatus, DataClassification
│       │   ├── execution.py            # ModelExecutionRecord, UsageMetrics, CostEstimate
│       │   └── trace.py                # Redacted Trace, Observation, Latency schemas
│       ├── dataset/                    # Phase 2: Evaluation dataset manager
│       │   ├── __init__.py
│       │   ├── manager.py              # JSONL import/export, split management (train/dev/test)
│       │   └── leakage.py              # Deduplication and cross-split leakage detector
│       ├── providers/                  # Phase 3: Model provider abstractions
│       │   ├── __init__.py
│       │   ├── base.py                 # Abstract Base Provider interface & guardrails
│       │   └── openrouter.py           # OpenRouter client with fail-closed privacy gate
│       ├── evaluation/                 # Phase 3: Benchmarking & scoring engine
│       │   ├── __init__.py
│       │   ├── runner.py               # Benchmark execution coordinator
│       │   └── metrics.py              # Factual accuracy, groundedness, rubric scorers
│       ├── traces/                     # Phase 4: Langfuse trace importer & analysis
│       │   ├── __init__.py
│       │   ├── importer.py             # Authorized redacted JSON/CSV importer
│       │   └── analyzer.py             # Candidate diagnosis engine (routing, retrieval, generation)
│       ├── storage/                    # Local persistence
│       │   ├── __init__.py
│       │   └── db.py                   # Lightweight SQLite storage for experiment history
│       └── dashboard/                  # Phase 7: Local Streamlit interface
│           ├── __init__.py
│           └── app.py                  # Read-only review UI using msu_llm_lab services
├── tests/                              # Automated test suite (pytest)
│   ├── __init__.py
│   ├── conftest.py                     # Shared fixtures (safe mock data, isolated paths)
│   ├── test_audit.py                   # Tests for parser, validator, and edge cases
│   ├── test_schemas.py                 # Schema validation, serialization, and invariants
│   ├── test_providers.py               # Mocked provider tests & fail-closed confidentiality gate
│   └── test_traces.py                  # Trace import validation and edge cases
├── data/                               # Local data storage (never checked into git)
│   ├── evaluations/                    # Versioned JSONL benchmark test sets
│   └── trace_imports/                  # Authorized redacted trace exports
├── reports/                            # Output directory for audit & evaluation reports
├── Copy of MSU General KB - Local - Sudarshan.md     # Original General KB (read-only)
└── Copy of MSU Programmes KB - Local - Sudharsan.md   # Original Programmes KB (read-only)
```

---

## 4. Component Responsibilities & Boundaries

| Component | Responsibility | Boundary / Invariants |
| :--- | :--- | :--- |
| **CLI (`cli.py`)** | Single entry point for all workflows (`audit-kb`, `validate-dataset`, `benchmark`, `import-traces`). | Typer-based, outputs structured tables and logs to stdout/stderr. |
| **Audit Engine (`audit/`)** | Parse raw KB markdown files, distinguish raw marker counts from parsed records, detect duplicates, missing answers, conflicting numbers, and structural issues. | **Read-only**: Never modifies or overwrites original KB files. Source hashes verified before and after. |
| **Schema Engine (`schemas/`)** | Define validated Pydantic data contracts with strict verification state and confidentiality classification (`PUBLIC`, `SYNTHETIC`, `INTERNAL`, `CONFIDENTIAL`, `UNCLASSIFIED`). | Rejects missing source citations or unverified ground truth claims. |
| **Dataset Manager (`dataset/`)** | Read/write JSONL benchmark datasets, enforce train/dev/test isolation, flag leakage. | No synthetic data masquerading as verified ground truth. |
| **Provider Adapter (`providers/`)** | Secure OpenRouter API integration for candidate model benchmarking. | **Fail-Closed Confidentiality Gate**: Allows ONLY explicitly approved `PUBLIC` or `SYNTHETIC` records. Rejects `UNCLASSIFIED`, `INTERNAL`, and `CONFIDENTIAL` before any network call. Automated test verifies zero outbound requests on rejection. |
| **Trace Importer (`traces/`)** | Parse authorized, redacted Langfuse export files (JSON/CSV) as soon as available, without blocking on provider benchmarking. | Offline only. Label diagnostic findings as candidate hypotheses, not ground truth. |
| **Local Storage (`storage/`)** | SQLite database and file-based JSONL/CSV persistence for runs and outputs. | Local-only, portable, no external cloud dependencies. |
| **Local Dashboard (`dashboard/`)** | Read-only visual inspection interface (Streamlit) for engineering decisions. | Deferred to Phase 7; shares underlying service modules with CLI. |
