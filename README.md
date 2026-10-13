# MSU LLM Research Workbench (`msu-llm-lab`)

Internal research workbench for investigating, evaluating, and improving the admissions AI counsellor for Management and Science University (MSU), Malaysia.

## Documentation
- [Project Map](docs/PROJECT_MAP.md)
- [System Architecture](docs/ARCHITECTURE.md)
- [Implementation Plan](docs/IMPLEMENTATION_PLAN.md)

## Usage

### Phase 1: Knowledge-Base Audit
Audit source knowledge-base markdown files for formatting defects, missing fields, duplicate questions, and conflicting data without altering source files:

```bash
msu-llm-lab audit-kb "Copy of MSU General KB - Local - Sudarshan.md"
msu-llm-lab audit-kb "Copy of MSU Programmes KB - Local - Sudharsan.md" --output-dir reports
```
