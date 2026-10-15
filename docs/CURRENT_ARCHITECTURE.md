# MSU LLM Workbench — Complete System Architecture (Phases 0–6)

```mermaid
graph TB
    %% CLI & Interface Layer
    subgraph Interfaces["1. User Interfaces & CLI Layer"]
        CLI["Typer CLI (cli.py)"]
        TUI["Rich Terminal Dashboard (tui/)"]
        
        CLI --> CMD_Audit["audit-kb"]
        CLI --> CMD_Remediate["remediate-kb"]
        CLI --> CMD_Validate["validate-dataset"]
        CLI --> CMD_Benchmark["benchmark & compare-runs"]
        CLI --> CMD_BenchmarkRAG["benchmark-rag"]
        CLI --> CMD_Serving["simulate-serving"]
        CLI --> CMD_Traces["import-traces"]
        CLI --> CMD_Budget["budget status"]
        CLI --> CMD_Models["models list"]
    end

    %% Phase 1 Subsystems
    subgraph Phase1["2. Knowledge Base Audit & Remediation (Phase 1)"]
        RawKB[("Source KB (Read-Only Markdown)")]
        SHA_Check["SHA-256 Immutability Guard"]
        Parser["KB Parser (Tokenize questions, answers, sections)"]
        Validator["KB Validator (Deterministic & Heuristic Rules)"]
        Remediation["Remediation Engine & Curated Registry"]
        CleanKB[("Versioned Clean KB Copy")]

        RawKB --> SHA_Check --> Parser --> Validator
        Validator --> Remediation --> CleanKB
    end

    %% Phase 2 Subsystems
    subgraph Phase2["3. Dataset Governance & Leakage Prevention (Phase 2)"]
        RawCases[("Evaluation Cases (JSONL)")]
        SchemaVal["Pydantic Schemas (EvaluationCase, VerificationRecord)"]
        Partition["Dataset Partitioning (TRAIN / DEV / TEST)"]
        LeakDetector["N-Gram Leakage Detector (Lexical Overlap)"]

        RawCases --> SchemaVal --> Partition --> LeakDetector
    end

    %% Phase 3 Subsystems
    subgraph Phase3["4. Model Evaluation & Provider Layer (Phase 3)"]
        BudgetPreflight{"Budget Preflight Check<br/>($0.50/run cap)"}
        ConfGate{"Fail-Closed Confidentiality Gate"}
        SecretScrubber["Secret Scrubber (API keys, bearer tokens)"]
        Provider["OpenRouter Adapter (OpenRouterAdapter)"]
        DryRunSim["Offline Deterministic Dry-Run Engine"]
        Rubric["Rubric Evaluator (Accuracy, Groundedness, Format)"]

        LeakDetector --> BudgetPreflight
        BudgetPreflight -->|Approved| ConfGate
        BudgetPreflight -.->|Breach| BudgetError["Abort: BudgetExceededError"]
        ConfGate -->|PUBLIC or SYNTHETIC| SecretScrubber
        ConfGate -.->|UNCLASSIFIED / INTERNAL / CONFIDENTIAL| ConfError["Abort: ConfidentialityViolationError"]
        SecretScrubber --> Provider
        Provider --> DryRunSim
        DryRunSim --> Rubric
    end

    %% Phase 4 Subsystems
    subgraph Phase4["5. RAG Observability & Trace Diagnostics (Phase 4)"]
        TraceFiles[("Langfuse Exports (JSON / CSV)")]
        TraceImp["Trace Importer & PII Anonymizer"]
        Analyzer["Trace Diagnostic Analyzer (4 Failure Modes)"]
        
        TraceFiles --> TraceImp --> Analyzer
        Analyzer --> Class_Routing["1. ROUTING_FAILURE"]
        Analyzer --> Class_Retrieval["2. RETRIEVAL_FAILURE"]
        Analyzer --> Class_Generation["3. GENERATION_FAILURE"]
        Analyzer --> Class_KB["4. SOURCE_KNOWLEDGE_FAILURE"]
        Analyzer --> Class_Healthy["5. NO_FAILURE_DETECTED"]
    end

    %% Phase 6 Subsystems
    subgraph Phase6["6. Inference Serving Simulation & Sizing (Phase 6)"]
        HardwareSpec["Hardware Profiles (H100, A100, Mac Studio)"]
        ServingEngine["Serving Simulator (vLLM Engine Specs)"]
        KVCalc["KV-Cache & Memory Footprint Calculator"]
        RampSim["Concurrency Ramp Simulator (1 to 150 Workers)"]
        Crossover["Cost Crossover Analysis (Hosted vs Cloud GPU)"]

        HardwareSpec --> ServingEngine --> KVCalc --> RampSim --> Crossover
    end

    %% Storage & Persistence Layer
    subgraph Storage["7. Local Storage & Artifacts (Zero-Network)"]
        SQLite[("Local SQLite Database (workbench.db - WAL Mode)")]
        LedgerTbl["budget_ledger (Transactions & Micro-USD Spend)"]
        RunsTbl["evaluation_runs (Aggregated Run Metrics)"]
        ResultsTbl["evaluation_results (Per-Case Scores & Outputs)"]
        ModelsTbl["models (Candidate Model Catalog)"]
        Reports[("Reports Directory (CSV & Markdown Artifacts)")]

        SQLite --- LedgerTbl
        SQLite --- RunsTbl
        SQLite --- ResultsTbl
        SQLite --- ModelsTbl

        Rubric --> RunsTbl
        Rubric --> ResultsTbl
        Provider --> LedgerTbl
        Analyzer --> Reports
        Validator --> Reports
        Crossover --> Reports
    end
```

---

## Component Interaction Lifecycles


### A. Model Evaluation & Spend Lifecycle (Phase 3)

```mermaid
sequenceDiagram
    autonumber
    actor Engineer as Research Engineer
    participant CLI as CLI / Runner
    participant Budget as BudgetService
    participant Gate as Confidentiality Gate
    participant Provider as OpenRouterAdapter
    participant Evaluator as RubricEvaluator
    participant DB as SQLite Storage

    Engineer->>CLI: msu-llm-lab benchmark --model <id> --split TEST --dry-run
    CLI->>Budget: check_preflight_budget(projected_cost)
    alt Projected Cost Exceeds Ceilings
        Budget-->>CLI: raise BudgetExceededError
        CLI-->>Engineer: Abort execution with policy warning
    else Budget Check Passed
        Budget-->>CLI: Spend permitted
    end

    loop For each Evaluation Case in Split
        CLI->>Gate: validate_case_confidentiality(case)
        alt Classification is INTERNAL / CONFIDENTIAL / UNCLASSIFIED
            Gate-->>CLI: raise ConfidentialityViolationError
            CLI-->>Engineer: Fail-Closed Halt (Zero Outbound Calls)
        else Classification is PUBLIC or SYNTHETIC
            Gate-->>CLI: Approved for dispatch
            CLI->>Provider: generate(request)
            Provider->>Provider: scrub_secrets() & dry-run simulation
            Provider-->>CLI: ProviderResponse (latency, tokens, cost_usd)
            CLI->>Evaluator: evaluate(output, expected_answer)
            Evaluator-->>CLI: EvaluationScore (accuracy, groundedness, rubric)
            CLI->>Budget: record_spend(transaction_id, cost_usd)
            Budget->>DB: INSERT INTO budget_ledger
        end
    end

    CLI->>DB: INSERT INTO evaluation_runs & evaluation_results
    CLI-->>Engineer: Render rich summary table & performance deltas
```

---

### B. Production Trace Ingestion & Diagnostic Lifecycle (Phase 4)

```mermaid
sequenceDiagram
    autonumber
    actor Admin as System Admin / Ops
    participant Importer as TraceImporter
    participant Anonymizer as PII Redactor & Secret Scrubber
    participant Analyzer as TraceDiagnosticAnalyzer
    participant Reporter as DiagnosticReporter

    Admin->>Importer: load_from_file(export.json / export.csv)
    Importer->>Anonymizer: redact_emails(), redact_phones(), scrub_secrets()
    Anonymizer-->>Importer: Sanitized TraceRecords & ObservationSpans
    
    Importer->>Analyzer: analyze_batch(traces)
    loop For each trace
        Analyzer->>Analyzer: 1. Check Routing Failure (query intent vs assigned route)
        Analyzer->>Analyzer: 2. Check Retrieval Failure (empty chunks, topic divergence)
        Analyzer->>Analyzer: 3. Check KB Defect (conflicting fees or stale years in context)
        Analyzer->>Analyzer: 4. Check Generation Failure (hallucinated figures, false refusal)
    end
    Analyzer-->>Importer: TraceDiagnosticReport (hypotheses with confidence & evidence)

    Importer->>Reporter: export_csv() & export_markdown()
    Reporter-->>Admin: Structured report files in reports/ & Rich terminal tables
```

---

## Subsystem Matrix (Phases 0 to 6)

| Subsystem | Primary Python Modules | CLI Command | Core Safety / Invariant |
|---|---|---|---|
| **Audit & Remediation** | `msu_llm_lab.audit.*`<br/>`msu_llm_lab.remediation.*` | `msu-llm-lab audit-kb`<br/>`msu-llm-lab remediate-kb` | Source KB is read-only; SHA-256 hashed before and after. Never auto-resolves contradictions. |
| **Dataset Governance** | `msu_llm_lab.schemas.*`<br/>`msu_llm_lab.dataset.*` | `msu-llm-lab validate-dataset` | Verified cases require explicit source citations. N-gram detector prevents cross-split leakage. |
| **Evaluation & Providers** | `msu_llm_lab.providers.*`<br/>`msu_llm_lab.evaluation.*` | `msu-llm-lab benchmark`<br/>`msu-llm-lab compare-runs` | Fail-closed gate blocks non-public cases. Dry-run by default. Preflight budget limits enforced. |
| **Decoupled RAG Benchmark** | `msu_llm_lab.rag.*` | `msu-llm-lab benchmark-rag` | Evaluates retrieval (Recall@k, Precision@k, MRR) against 3,030 clean audited KB records; isolates abstention denominator (v2.1 baseline). |
| **Trace Diagnostics** | `msu_llm_lab.traces.*` | `msu-llm-lab import-traces` | PII and API keys redacted offline. Classifications labeled as candidate hypotheses, not ground truth. |
| **Inference Serving & Sizing** | `msu_llm_lab.serving.*` | `msu-llm-lab simulate-serving` | Hardware sizing, KV-cache footprint, concurrency ramp (1–150 workers), and cloud vs self-hosted cost crossover. |
| **Spend Governance** | `msu_llm_lab.core.budget`<br/>`msu_llm_lab.storage.db` | `msu-llm-lab budget status` | Exact `Decimal` arithmetic and integer micro-dollars. Local governance ceilings ($0.50/$2.00/$10.00). |
| **Interactive Terminal** | `msu_llm_lab.tui.*` | `msu-llm-lab ui` | Hardware telemetry, tab navigation, model catalog, and budget ledger inspection. |
