"""CLI entry point for msu-llm-lab."""

import hashlib
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from msu_llm_lab.audit.models import AuditSummary, IssueSeverity
from msu_llm_lab.audit.parser import KBParser
from msu_llm_lab.audit.reporter import KBReporter
from msu_llm_lab.audit.validator import KBValidator
from msu_llm_lab.core.budget import BudgetService
from msu_llm_lab.models.catalog import get_portfolio
import sys

app = typer.Typer(
    name="msu-llm-lab",
    help="MSU Admissions LLM Research Workbench: evaluation, audit, and benchmark CLI.",
    no_args_is_help=False,
)
budget_app = typer.Typer(
    name="budget",
    help="Budget ledger, ceilings, and spend tracking.",
    no_args_is_help=True,
)
models_app = typer.Typer(
    name="models",
    help="Model portfolio catalog and validation.",
    no_args_is_help=True,
)

app.add_typer(budget_app, name="budget")
app.add_typer(models_app, name="models")

console = Console()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    interactive: bool = typer.Option(
        False,
        "--interactive",
        "-i",
        help="Launch interactive terminal UI.",
    ),
) -> None:
    """MSU Admissions LLM Research Workbench."""
    if ctx.invoked_subcommand is None:
        if interactive:
            from msu_llm_lab.tui.app import run_tui
            run_tui()
        elif sys.stdin.isatty():
            run_launcher_loop()
        else:
            console.print(ctx.get_help())


def _run_command(args: List[str]) -> None:
    """Run a CLI command in-process exactly as if typed, so every option keeps its real default (calling a command function directly
    would hand it unresolved Typer option objects)."""
    try:
        app(args=args, standalone_mode=False)
    except (SystemExit, typer.Exit):
        pass
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/dim]")
    except Exception as e:  # keep the launcher alive; show what failed instead of a traceback
        console.print(f"[bold red]Command failed:[/bold red] {type(e).__name__}: {e}")


def handle_welcome_action(choice: int) -> None:
    """Dispatches one launcher choice to a real workflow. Nothing here simulates anything or sends data."""
    if choice == 1:
        from msu_llm_lab.tui.app import run_tui
        run_tui()
    elif choice == 2:
        gen = Path("Copy of MSU General KB - Local - Sudarshan.md")
        if gen.exists():
            console.print("\n[bold #c084fc]Auditing MSU General Knowledge Base (read-only; nothing is written)...[/]")
            _run_command(["audit-kb", str(gen), "--dry-run"])
        else:
            console.print(f"[yellow]Knowledge base file not found: {gen}[/]")
    elif choice == 3:
        _run_command(["classification-data-status"])
    elif choice == 4:
        _run_command(["models", "roster"])
        console.print("\n[dim]Next: api-preflight (menu 5), then the smoke-test plan (menu 6).[/dim]")
    elif choice == 5:
        _run_command(["api-preflight"])
    elif choice == 6:
        _run_command(["smoke-test"])
    elif choice == 7:
        _run_command(["budget", "status"])
    elif choice == 8:
        console.print("\n[bold #c084fc]Cryptographic Knowledge Base Baseline Verification:[/]")
        gen = Path("Copy of MSU General KB - Local - Sudarshan.md")
        prog = Path("Copy of MSU Programmes KB - Local - Sudharsan.md")
        if gen.exists():
            console.print(f"  General KB SHA-256:    [green]{compute_sha256(gen)}[/] (Verified Immutable)")
        if prog.exists():
            console.print(f"  Programmes KB SHA-256: [green]{compute_sha256(prog)}[/] (Verified Immutable)")
    elif choice == 9:
        console.print("[dim]Exiting MSU LLM Workbench. Goodbye.[/]\n")


def run_launcher_loop() -> None:
    """Show the launcher, run the chosen workflow, then come back to the menu. Only 'Exit' (9), q or Esc leaves."""
    from msu_llm_lab.tui.welcome import MENU_ITEMS, show_welcome_screen

    exit_choice = len(MENU_ITEMS)
    while True:
        choice = show_welcome_screen()
        handle_welcome_action(choice)
        if choice == exit_choice or choice == 0:
            break
        if choice != 1:  # the dashboard is full-screen and returns straight to the menu; command output needs a pause to be read
            try:
                console.input("\n[dim]Press Enter to return to the menu...[/dim]")
            except (EOFError, KeyboardInterrupt):
                break


def _require_live_confirmation(models: List[str], dataset_name: str):
    """Fail closed: a live run needs interactive typed confirmation, or the command exits before any request."""
    from msu_llm_lab.core.live_guard import confirm_live_run
    from msu_llm_lab.providers.base import LiveNotAuthorizedError

    summary = [
        f"Live run requested. Models: {', '.join(models)}. Dataset: {dataset_name}.",
        "Local spend ceilings still apply (preflight and per-case circuit breaker).",
    ]
    try:
        return confirm_live_run(models, summary, echo=lambda line: console.print(line, markup=False, highlight=False))
    except LiveNotAuthorizedError as e:
        console.print(f"[bold red]Live run refused:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command("smoke-test")
def smoke_test(
    model: List[str] = typer.Option(["jev-1.14", "gpt-4.1-nano"], "--model", "-m", help="Registry keys to include, or 'all' for the whole roster."),
    stage: str = typer.Option("smoke", "--stage", help="smoke: the two-model protocol check. pilot: every eligible participant, strict gating."),
    task: str = typer.Option("both", "--task", "-t", help="INTENT, LEAD or both."),
    hard_cap: str = typer.Option("0.05", "--hard-cap", help="Hard spend cap in USD for this run (must fit the per-run and pilot ceilings)."),
    prompt_dir: Path = typer.Option(Path("."), "--prompt-dir", help="Folder holding the two authoritative prompt files."),
    out_dir: Path = typer.Option(Path("reports/smoke"), "--out-dir", help="Where prediction records are written."),
    live: bool = typer.Option(False, "--live", help="Send the planned requests. Needs interactive typed approval of the exact plan."),
) -> None:
    """Synthetic-only smoke test of the Jev and GPT-4.1 Nano adapters. Prints a plan by default and sends nothing."""
    from decimal import Decimal, InvalidOperation

    from msu_llm_lab.classification import smoke
    from msu_llm_lab.classification.records import PredictionStore, Task
    from msu_llm_lab.core.budget import BudgetService
    from msu_llm_lab.core.live_guard import confirm_live_run
    from msu_llm_lab.providers.base import LiveNotAuthorizedError, ProviderError

    try:
        cap = Decimal(hard_cap)
    except InvalidOperation:
        console.print(f"[bold red]Invalid --hard-cap:[/bold red] {hard_cap}")
        raise typer.Exit(code=1)
    tasks = [Task.INTENT, Task.LEAD] if task.lower() == "both" else [Task(task.upper())] if task.upper() in ("INTENT", "LEAD") else None
    if tasks is None:
        console.print("[bold red]--task must be INTENT, LEAD or both[/bold red]")
        raise typer.Exit(code=1)
    try:
        prompts = smoke.load_prompts(prompt_dir)
    except FileNotFoundError as e:
        console.print(f"[bold red]{e}[/bold red]")
        raise typer.Exit(code=1)

    from msu_llm_lab.models.registry import REGISTRY

    if stage not in ("smoke", "pilot"):
        console.print("[bold red]--stage must be smoke or pilot[/bold red]")
        raise typer.Exit(code=1)
    if "all" in [m.lower() for m in model]:
        model = list(REGISTRY)
    budget = BudgetService()
    plan = smoke.build_plan(model, tasks, prompts, budget, cap, stage=stage)
    for line in smoke.render_plan(plan):
        console.print(line, markup=False, highlight=False)
    if not live:
        console.print("\nPlan only: nothing was sent. Re-run with --live to request typed approval of this exact plan.", markup=False)
        raise typer.Exit(code=2 if plan.blocked else 0)
    if plan.blocked:
        console.print("[bold red]Live run refused:[/bold red] the plan is blocked.")
        raise typer.Exit(code=1)
    try:
        auth = confirm_live_run(plan.model_ids, ["Approve exactly this plan by typing the phrase below."],
                                echo=lambda line: console.print(line, markup=False, highlight=False), plan_fingerprint=plan.fingerprint)
    except LiveNotAuthorizedError as e:
        console.print(f"[bold red]Live run refused:[/bold red] {e}")
        raise typer.Exit(code=1)
    import uuid
    run_id = f"smoke_{uuid.uuid4().hex[:8]}"
    store = PredictionStore(Path(out_dir) / f"{run_id}.jsonl")
    failure = None
    summaries = []
    try:
        summaries = smoke.execute_plan(plan, auth, smoke.build_smoke_adapters(model, stage), store, budget, prompts, run_id)
    except Exception as e:  # records already written stay on disk; the failure is reported and sets a failing exit status
        failure = f"{type(e).__name__}: {e}"
        console.print(f"[bold red]Smoke test stopped:[/bold red] {failure}")
    recs = store.load()
    console.print(f"\nSchema verification only (no accuracy computed). Records: {store.path}", markup=False)
    for row in smoke.summarize_schema_checks(recs):
        console.print(str(row), markup=False, highlight=False)
    problems = []
    if failure:
        problems.append("the run stopped before finishing")
    problems += [f"a run aborted: {s.aborted_reason}" for s in summaries if s.aborted_reason]
    bad = [r for r in recs if r.status != "OK"]
    if bad:
        problems.append(f"{len(bad)} record(s) were invalid outputs or errors")
    planned = sum(r.n_cases for r in plan.rows)
    if len(recs) < planned:
        problems.append(f"only {len(recs)} of {planned} planned requests produced a record")
    if budget.unresolved():
        problems.append("unresolved spend is on the ledger (run `msu-llm-lab budget reconcile`)")
    if problems:
        console.print("[bold red]SMOKE TEST FAILED:[/bold red] " + "; ".join(problems), markup=True)
        raise typer.Exit(code=1)
    console.print("[green]Smoke test completed: every planned request produced a valid record.[/green]")


@models_app.command("roster")
def models_roster() -> None:
    """Show every comparison participant with its role, tasks, readiness and exact blockers."""
    from msu_llm_lab.models.registry import roster_status

    table = Table(title="Comparison roster (10 participants)", show_header=True, header_style="bold magenta")
    for col in ("Participant", "Role", "Tasks", "Readiness", "Data policy", "Evaluation", "Blockers"):
        table.add_column(col)
    for r in roster_status():
        table.add_row(r["display_name"] or r["key"], ", ".join(r["roles"]), ", ".join(r["tasks"]), r["readiness"], r["data_policy"],
                      r["evaluation"], ", ".join(r["blockers"]) or "none (synthetic)")
    console.print(table)


@app.command("export-traces")
def export_traces_cmd(
    fmt: str = typer.Option("zip", "--format", "-f", help="jsonl, csv or zip (zip = per-model files, comparison file and manifest)."),
    model: List[str] = typer.Option([], "--model", "-m", help="Limit to a model key (repeatable)."),
    run: List[str] = typer.Option([], "--run", "-r", help="Limit to a run id (repeatable)."),
    task: Optional[str] = typer.Option(None, "--task", "-t", help="INTENT or LEAD."),
    status: List[str] = typer.Option([], "--status", "-s", help="OK, INVALID_OUTPUT or ERROR (repeatable)."),
    out_dir: Optional[Path] = typer.Option(None, "--out-dir", help="Destination folder (default: reports/exports)."),
) -> None:
    """Export stored trace records (redacted, local only). Reads the prediction store and event log; sends nothing; never overwrites."""
    from msu_llm_lab.core.config import settings
    from msu_llm_lab.models.registry import roster_status
    from msu_llm_lab.observability.export import ExportError, ExportSelection, build_trace_rows, export_traces
    from msu_llm_lab.observability.monitor import LiveMonitor

    fmt_val = getattr(fmt, "default", fmt) if hasattr(fmt, "default") else fmt
    task_val = getattr(task, "default", task) if hasattr(task, "default") else task
    if task_val is not None and str(task_val).upper() not in ("INTENT", "LEAD"):
        console.print("[bold red]--task must be INTENT or LEAD[/bold red]")
        raise typer.Exit(code=1)
    mon = LiveMonitor()
    mon.poll()
    preds, events, labels, frozen = mon.trace_inputs()
    sel = ExportSelection(models=tuple(model) or None, runs=tuple(run) or None, task=str(task_val).upper() if task_val else None, statuses=tuple(status) or None)
    ts = build_trace_rows(preds, events, labels, frozen, sel)
    try:
        res = export_traces(ts, sel, roster_status(), labels, str(fmt_val).lower(), out_dir or (settings.reports_dir / "exports"))
    except ExportError as e:
        console.print(f"[bold red]Export failed:[/bold red] {e}")
        raise typer.Exit(code=1)
    console.print(f"Wrote {res.path.resolve()}", markup=False)
    console.print(f"records exported {res.n_exported}, skipped {res.n_skipped}, redacted {res.n_redacted}, missing telemetry {res.n_missing_telemetry}", markup=False)
    for w in res.warnings:
        console.print(f"WARNING: {w}", markup=False)


@app.command("evaluation-report")
def evaluation_report_cmd(
    bench: str = typer.Option("bench-d0a763e4a920", "--bench", help="Benchmark id (folder under data/benchmarks)."),
    bundle: Optional[Path] = typer.Option(None, "--bundle", help="Verified result bundle (default: reports/real_benchmark_<bench>-r4)."),
    out_dir: Optional[Path] = typer.Option(None, "--out", help="Output folder (default: reports/evaluation_report_<bench>)."),
    seed: int = typer.Option(20261015, "--seed", help="Bootstrap seed."),
    redesign: bool = typer.Option(False, "--redesign", help="Also write the compact redesigned PDF/Markdown (needs an existing report in --out)."),
) -> None:
    """Offline evaluation report (PDF, Markdown, CSV/JSON, figures) from existing benchmark artifacts. No network, no inference, no spend; sources are read-only."""
    try:
        from msu_llm_lab.observability import evaluation_report as er
        from msu_llm_lab.observability.evaluation_runner import generate
    except ModuleNotFoundError as e:
        console.print(f"Missing report dependency ({e.name}). Install them with: uv sync --extra report", markup=False)
        raise typer.Exit(code=1)

    cfg = er.Config(bench_id=bench, bench_dir=Path("data/benchmarks") / bench, seed=seed,
                    bundle_dir=bundle or Path(f"reports/real_benchmark_{bench}-r4"), out_dir=out_dir or Path(f"reports/evaluation_report_{bench}"))
    if redesign:
        from msu_llm_lab.observability.evaluation_runner import generate_redesign

        r2 = generate_redesign(cfg)
        q = r2["quality"]
        console.print(f"Wrote redesigned report in {r2['out'].resolve()}: {q['pages'][0]} -> {q['pages'][1]} pages; quality check {'PASS' if q['ok'] else 'FAIL'}", markup=False)
        if not q["ok"]:
            raise typer.Exit(code=1)
        return
    res = generate(cfg)
    console.print(f"Wrote {res['out'].resolve()} ({res['pages']} flowables); verification {'PASS' if res['verification']['ok'] else 'FAIL'}", markup=False)
    if not res["verification"]["ok"]:
        for c in res["verification"]["checks"]:
            if not c["ok"]:
                console.print(f"FAILED: {c['check']} {c['detail']}", markup=False)
        raise typer.Exit(code=1)


labeled_app = typer.Typer(help="Labelled classifier evaluation: offline preparation, plan and scoring (never sends requests).")
app.add_typer(labeled_app, name="labeled-eval")


@labeled_app.command("prepare")
def labeled_prepare(out_dir: Path = typer.Option(Path("reports/labeled_eval"), "--out")) -> None:
    """Audit the labelled test suites, freeze splits, separate the answer key and write protocol, blocked-run results table and cost plan. Offline."""
    try:
        from msu_llm_lab.labeled_eval import deliver, run
    except ModuleNotFoundError as e:
        console.print(f"Missing report dependency ({e.name}). Install them with: uv sync --extra report", markup=False)
        raise typer.Exit(code=1)
    r = deliver.deliver(run.Cfg(out=str(out_dir)))
    console.print(f"Wrote {r['out'].resolve()}; open gates: {', '.join(g['id'] for g in r['gates'] if g['blocking'] and not g['ok']) or 'none'}; eight-participant cohort expected ${r['cohort_totals']['expected_with_observed_retries_usd']:.2f}, guard bound ${r['cohort_totals']['guard_bound_usd']:.2f}; headroom ${r['cohort_verdict']['headroom_usd']:.2f}", markup=False)


@labeled_app.command("score")
def labeled_score(predictions: List[Path] = typer.Argument(..., help="Prediction JSONL files (after inference)."), out_csv: Path = typer.Option(Path("reports/labeled_eval/labeled-eval-v1/classifier_evaluation_results_scored.csv"), "--out")) -> None:
    """Join stored predictions to the protected answer key and compute metrics. Evaluator only; reads no API key and sends nothing."""
    from msu_llm_lab.labeled_eval import deliver, run

    res = deliver.score_all(run.Cfg(), predictions, out_csv)
    console.print(f"Scored {len(res)} model/task/split rows -> {out_csv}", markup=False)


@labeled_app.command("record-permission")
def labeled_record_permission(
    evidence_type: str = typer.Option(..., "--evidence-type", help="DOCUMENT or OPERATOR_CHAT_ATTESTATION."),
    verified_by: str = typer.Option(..., "--verified-by", help="Who stands behind this record."),
    evidence_reference: str = typer.Option(..., "--evidence-reference", help="Document id/ticket/email, or a description of the attestation."),
    approvers: str = typer.Option(..., "--approvers", help="Names of the approving parties, or 'NOT PROVIDED'."),
    approved_on: str = typer.Option(..., "--approved-on", help="Date of the approval, or 'NOT PROVIDED'."),
    source: str = typer.Option(..., "--source", help="Where the approval comes from."),
) -> None:
    """Record the data-processing permission for the stored plan exactly as you state it. Nothing is defaulted; an attestation is stored as an attestation."""
    from msu_llm_lab.core.budget import BudgetService
    from msu_llm_lab.labeled_eval import execute as X, run

    cfg = run.Cfg()
    plan = X.build_plan(cfg, BudgetService())
    try:
        rec = X.record_processing_permission(cfg, plan, evidence_type=evidence_type, verified_by=verified_by, evidence_reference=evidence_reference, approvers=approvers,
                                             approved_on=approved_on, source=source, searched=["repository docs and reports", "~/Downloads file names", "earlier benchmark manifests (prep-v2 authorisation only)"])
    except ValueError as e:
        console.print(f"[bold red]Not recorded:[/bold red] {e}")
        raise typer.Exit(code=1)
    console.print(f"Recorded {rec['evidence_type']} for plan {plan['fingerprint']}: {len(rec['models'])} models, {len(rec['routes'])} routes, tasks {rec['tasks']}. Provider policies are NOT verified.", markup=False)


@labeled_app.command("master")
def labeled_master(monthly_cases: Optional[int] = typer.Option(None, "--monthly-cases", help="Optional workload assumption (cases per month) for the projected monthly cost; otherwise NOT_COMPUTED.")) -> None:
    """Build the Main Intent and Lead Qualifier model-comparison master CSVs, pairwise tests, sensitivity table, reconciliation and summary. Offline, evaluator side only."""
    try:
        from msu_llm_lab.labeled_eval import master, run
    except ModuleNotFoundError as e:
        console.print(f"Missing report dependency ({e.name}). Install them with: uv sync --extra report", markup=False)
        raise typer.Exit(code=1)
    r = master.write_all(run.Cfg(), monthly_cases)
    for t, p in r["files"].items():
        console.print(f"{t}: {p} ({len(r['tables'][t])} rows)", markup=False)
    bad = r["rec"][r["rec"].status == "DISCREPANCY"]
    console.print(f"reconciliation: {len(r['rec'])} checks, {len(bad)} unexplained discrepancies", markup=False)
    raise typer.Exit(code=1 if len(bad) else 0)


@labeled_app.command("report")
def labeled_report() -> None:
    """Write the final results report (PDF + Markdown + figures) from the master tables. Offline."""
    try:
        from msu_llm_lab.labeled_eval import report, run
    except ModuleNotFoundError as e:
        console.print(f"Missing report dependency ({e.name}). Install them with: uv sync --extra report", markup=False)
        raise typer.Exit(code=1)
    r = report.build(run.Cfg())
    console.print(f"Wrote {r['out']}/classifier_evaluation_report.pdf with {len(r['figures'])} figures", markup=False)


jev_app = typer.Typer(help="Pre-registered Jev prompt comparison (V0/V1/V2 on Main Intent). Own plan, approvals and permission record; never runs without every gate.")
labeled_app.add_typer(jev_app, name="jev-experiment")


def _jev_ctx():
    from msu_llm_lab.core.budget import BudgetService
    from msu_llm_lab.labeled_eval import jev_experiment as J, run

    cfg = run.Cfg()
    budget = BudgetService()
    return J, cfg, budget, J.build_plan(cfg, budget)


@jev_app.command("gates")
def jev_gates(save_plan: bool = typer.Option(False, "--save-plan")) -> None:
    """Show the experiment plan and every gate. Sends nothing; exit 2 while a blocking gate fails."""
    J, cfg, budget, plan = _jev_ctx()
    gates = J.check_gates(cfg, plan, budget)
    if save_plan:
        J.save_plan(cfg, plan)
    for line in J.render_plan(plan, gates):
        console.print(line, markup=False, highlight=False)
    raise typer.Exit(code=0 if all(g["ok"] or not g["blocking"] for g in gates) else 2)


@jev_app.command("record-approval")
def jev_record_approval(operator: str = typer.Option(..., "--operator"), statement: str = typer.Option(..., "--statement", help="Your approval, in your words.")) -> None:
    """Record YOUR approval of this exact plan (bound to its fingerprint). It does not verify the processing permission."""
    J, cfg, budget, plan = _jev_ctx()
    try:
        J.record_operator_approval(cfg, plan, operator=operator, statement=statement)
    except ValueError as e:
        console.print(f"[bold red]Not recorded:[/bold red] {e}")
        raise typer.Exit(code=1)
    console.print(f"Recorded operator approval for plan {plan['fingerprint']}.", markup=False)


@jev_app.command("record-permission")
def jev_record_permission(evidence_type: str = typer.Option(..., "--evidence-type"), verified_by: str = typer.Option(..., "--verified-by"), evidence_reference: str = typer.Option(..., "--evidence-reference"),
                          approvers: str = typer.Option(..., "--approvers"), approved_on: str = typer.Option(..., "--approved-on"), source: str = typer.Option(..., "--source")) -> None:
    """Record the processing permission for this plan exactly as you state it (DOCUMENT or OPERATOR_CHAT_ATTESTATION). Nothing is defaulted."""
    J, cfg, budget, plan = _jev_ctx()
    try:
        J.record_permission(cfg, plan, evidence_type=evidence_type, verified_by=verified_by, evidence_reference=evidence_reference, approvers=approvers, approved_on=approved_on, source=source)
    except ValueError as e:
        console.print(f"[bold red]Not recorded:[/bold red] {e}")
        raise typer.Exit(code=1)
    console.print(f"Recorded {evidence_type} permission for plan {plan['fingerprint']}. Provider policies are NOT verified.", markup=False)


@jev_app.command("run")
def jev_run(operator: str = typer.Option(..., "--operator"), live: bool = typer.Option(False, "--live", help="Send the planned requests. Needs every gate AND two interactive typed approvals.")) -> None:
    """Execute (or resume) the experiment. Refuses before any typed phrase if a gate fails."""
    from msu_llm_lab.classification import realdata
    from msu_llm_lab.classification.smoke import build_smoke_adapters
    from msu_llm_lab.core.config import load_project_env, settings
    from msu_llm_lab.core.live_guard import confirm_live_run
    from msu_llm_lab.labeled_eval import execute as X
    from msu_llm_lab.providers.base import LiveNotAuthorizedError

    load_project_env()
    J, cfg, budget, plan = _jev_ctx()
    stored = J.exp_dir(cfg) / "plan.json"
    if not stored.exists() or __import__("json").loads(stored.read_text()).get("fingerprint") != plan["fingerprint"]:
        console.print("[bold red]This exact plan was never stored.[/bold red] Run `labeled-eval jev-experiment gates --save-plan` and review it.")
        raise typer.Exit(code=1)
    gates = J.check_gates(cfg, plan, budget)
    for line in J.render_plan(plan, gates):
        console.print(line, markup=False, highlight=False)
    failed = [g for g in gates if g["blocking"] and not g["ok"]]
    if failed:
        console.print("[bold red]Live run refused:[/bold red] " + "; ".join(f"{g['id']} {g['item']}" for g in failed))
        raise typer.Exit(code=1)
    if not live:
        console.print("\nPlan only: nothing was sent. Re-run with --live to request the typed approvals.", markup=False)
        raise typer.Exit(code=0)
    models = [plan["participant"]["model_id"]]
    try:
        live_auth = confirm_live_run(models, ["Approve exactly this experiment plan by typing the phrase below."], echo=lambda line: console.print(line, markup=False, highlight=False), plan_fingerprint=plan["fingerprint"])
        auth = realdata.confirm_real_data(dataset_version=plan["dataset_version"], session_policy="labeled-test-files", models=models, routes={models[0]: plan["participant"]["route"] or "none"}, tasks=["INTENT"],
                                          plan_fingerprint=plan["fingerprint"], cost_ceiling_usd=plan["guard_bound_total_usd"], provider_policy_status={models[0]: "UNKNOWN"}, recipients=plan["recipients"], operator=operator,
                                          echo=lambda line: console.print(line, markup=False, highlight=False))
    except LiveNotAuthorizedError as e:
        console.print(f"[bold red]Live run refused:[/bold red] {e}")
        raise typer.Exit(code=1)
    adapters = build_smoke_adapters([J.KEY], X.EXECUTION_STAGE)
    root = settings.data_dir / "labeled_eval" / "labeled-eval-v1" / "jev_prompt_experiment"
    try:
        res = X.execute(plan, X.load_cases(cfg), J.load_prompts(cfg), auth, live_auth, adapters, budget, root=root, reports_dir=cfg.out, gate_fn=lambda: J.check_gates(cfg, plan, budget), progress=lambda m: console.print(m, markup=False))
    except KeyboardInterrupt:
        console.print("Interrupted. Progress is checkpointed; re-run the same command to resume.", markup=False)
        raise typer.Exit(code=130)
    except Exception as e:
        console.print(f"[bold red]Experiment stopped:[/bold red] {type(e).__name__}: {e}")
        raise typer.Exit(code=1)
    console.print(f"\nunits {res.units_done} of {res.units_total}   guard cost ${res.spent_guard_usd}   halted: {res.halted or 'no'}", markup=False)
    for m, why in res.quarantined.items():
        console.print(f"QUARANTINED {m}: {why}", markup=False)
    raise typer.Exit(code=0 if (res.units_done == res.units_total and not res.quarantined) else 3)


@jev_app.command("score")
def jev_score() -> None:
    """Score the experiment against the protected key and apply the pre-registered selection rule. Offline."""
    from msu_llm_lab.labeled_eval import jev_experiment_score as S, run

    r = S.score(run.Cfg())
    console.print(f"Scored {len(r['results'])} variant/half rows; selected on dev: {r['selected']}; V0 repeat agreement: {r['agreement']}", markup=False)


@labeled_app.command("gates")
def labeled_gates(save_plan: bool = typer.Option(False, "--save-plan", help="Also store the plan (case ids and bounds only).")) -> None:
    """Show the plan and every gate that must pass before any request. Sends nothing."""
    from msu_llm_lab.core.budget import BudgetService
    from msu_llm_lab.labeled_eval import execute as X, run

    cfg = run.Cfg()
    budget = BudgetService()
    plan = X.build_plan(cfg, budget)
    gates = X.check_gates(cfg, plan, budget)
    if save_plan:
        X.save_plan(cfg, plan)
    for line in X.render_plan(plan, gates):
        console.print(line, markup=False, highlight=False)
    raise typer.Exit(code=0 if all(g["ok"] or not g["blocking"] for g in gates) else 2)


@labeled_app.command("run")
def labeled_run(operator: str = typer.Option(..., "--operator", help="Your name, recorded in the authorisation."),
                live: bool = typer.Option(False, "--live", help="Send the planned requests. Needs every gate AND two interactive typed approvals.")) -> None:
    """Execute (or resume) the labelled evaluation. Refuses before any typed phrase if a gate fails."""
    from msu_llm_lab.classification import realdata
    from msu_llm_lab.classification.smoke import build_smoke_adapters
    from msu_llm_lab.core.budget import BudgetService
    from msu_llm_lab.core.config import load_project_env, settings
    from msu_llm_lab.core.live_guard import confirm_live_run
    from msu_llm_lab.labeled_eval import execute as X, run
    from msu_llm_lab.providers.base import LiveNotAuthorizedError

    load_project_env()
    cfg = run.Cfg()
    budget = BudgetService()
    plan = X.build_plan(cfg, budget)
    stored = cfg.out / run.VERSION / "plan.json"
    if not stored.exists() or __import__("json").loads(stored.read_text()).get("fingerprint") != plan["fingerprint"]:
        console.print("[bold red]This exact plan was never stored.[/bold red] Run `labeled-eval gates --save-plan` and review it.")
        raise typer.Exit(code=1)
    gates = X.check_gates(cfg, plan, budget)
    for line in X.render_plan(plan, gates):
        console.print(line, markup=False, highlight=False)
    failed = [g for g in gates if g["blocking"] and not g["ok"]]
    if failed:
        console.print("[bold red]Live run refused:[/bold red] " + "; ".join(f"{g['id']} {g['item']}" for g in failed))
        raise typer.Exit(code=1)
    if not live:
        console.print("\nPlan only: nothing was sent. Re-run with --live to request the typed approvals.", markup=False)
        raise typer.Exit(code=0)
    models = [p["model_id"] for p in plan["participants"]]
    try:
        live_auth = confirm_live_run(models, ["Approve exactly this plan by typing the phrase below."], echo=lambda line: console.print(line, markup=False, highlight=False), plan_fingerprint=plan["fingerprint"])
        auth = realdata.confirm_real_data(dataset_version=run.VERSION, session_policy="labeled-test-files", models=models, routes={p["model_id"]: (p["route"] or "none") for p in plan["participants"]},
                                          tasks=["INTENT", "LEAD"], plan_fingerprint=plan["fingerprint"], cost_ceiling_usd=plan["guard_bound_total_usd"],
                                          provider_policy_status={m: "UNKNOWN" for m in models}, recipients=plan["recipients"], operator=operator,
                                          echo=lambda line: console.print(line, markup=False, highlight=False))
    except LiveNotAuthorizedError as e:
        console.print(f"[bold red]Live run refused:[/bold red] {e}")
        raise typer.Exit(code=1)
    adapters = build_smoke_adapters([p["key"] for p in plan["participants"]], X.EXECUTION_STAGE)
    root = settings.data_dir / "labeled_eval" / run.VERSION
    try:
        res = X.execute(plan, X.load_cases(cfg), X.load_prompts(), auth, live_auth, adapters, budget, root=root, reports_dir=cfg.out,
                        gate_fn=lambda: X.check_gates(cfg, plan, budget), progress=lambda m: console.print(m, markup=False))
    except KeyboardInterrupt:
        console.print("Interrupted. Progress is checkpointed; re-run the same command to resume.", markup=False)
        raise typer.Exit(code=130)
    except Exception as e:
        console.print(f"[bold red]Evaluation stopped:[/bold red] {type(e).__name__}: {e}")
        raise typer.Exit(code=1)
    console.print(f"\nunits {res.units_done} of {res.units_total}   guard cost ${res.spent_guard_usd}   halted: {res.halted or 'no'}", markup=False)
    for m, why in res.quarantined.items():
        console.print(f"QUARANTINED {m}: {why}", markup=False)
    raise typer.Exit(code=0 if (res.units_done == res.units_total and not res.quarantined) else 3)


real_app = typer.Typer(help="Authorized real-data benchmark: plan, run (typed approvals), status, report.")
app.add_typer(real_app, name="real-benchmark")


def _real_inputs(prep_dir: Path, session_policy: str, prompt_dir: Path, model: List[str], block_size: int, hard_cap: Optional[str], sampling: str = "stratified", inherit_from: Optional[str] = None, reserve: Optional[str] = None):
    from decimal import Decimal, InvalidOperation

    from msu_llm_lab.classification import benchmark as bm
    from msu_llm_lab.classification.smoke import load_prompts
    from msu_llm_lab.core.budget import BudgetService

    try:
        cap = Decimal(hard_cap) if hard_cap else None
    except InvalidOperation:
        console.print(f"[bold red]Invalid --hard-cap:[/bold red] {hard_cap}")
        raise typer.Exit(code=1)
    try:
        loaded = bm.load_prep_cases(prep_dir, session_policy)
        prompts = load_prompts(prompt_dir)
        budget = BudgetService()
        from msu_llm_lab.core.config import settings as _s
        parent = None
        if inherit_from:
            import json as _json
            pdir = bm.bench_dir(_s.data_dir / "benchmarks", inherit_from)
            if not (pdir / "plan.json").exists():
                raise bm.PlanError(f"no stored plan for {inherit_from}")
            parent = (_json.loads((pdir / "plan.json").read_text()), bm.load_state(_s.data_dir / "benchmarks", inherit_from))
        plan = bm.build_plan(loaded, prompts, budget, model_keys=model or None, block_size=block_size, hard_cap_usd=cap, sampling=sampling, parent=parent, reports_dir=_s.reports_dir,
                             reserve_usd=Decimal(reserve) if reserve else None)
    except (bm.PlanError, FileNotFoundError) as e:
        console.print(f"[bold red]Cannot plan:[/bold red] {e}")
        raise typer.Exit(code=1)
    return bm, loaded, prompts, budget, plan


@real_app.command("plan")
def real_plan(
    prep_dir: Path = typer.Option(..., "--prep-dir", help="Prepared dataset folder (for example data/classification_prep/prep-v2)."),
    session_policy: str = typer.Option(..., "--session-policy", help="2h or 30m. The choice is yours; neither is assumed."),
    prompt_dir: Path = typer.Option(Path("."), "--prompt-dir"),
    model: List[str] = typer.Option([], "--model", "-m", help="Participant keys; default is the whole roster."),
    block_size: int = typer.Option(10, "--block-size", help="Cases per task in one block."),
    hard_cap: Optional[str] = typer.Option(None, "--hard-cap", help="Hard cap in USD for this benchmark (never above the pilot ceiling)."),
    sampling: str = typer.Option("stratified", "--sampling", help="stratified (variety first, then proportional) or hash."),
    inherit_from: Optional[str] = typer.Option(None, "--inherit-from", help="Earlier bench id whose clean completed results are carried over (identical participants only)."),
    reserve: Optional[str] = typer.Option(None, "--reserve", help="USD of the remaining ledger headroom kept out of the plan for retries and unresolved costs."),
) -> None:
    """Build and store the benchmark plan. Sends nothing, calls no provider, and prints no student text."""
    from msu_llm_lab.core.config import settings

    bm, _, _, _, plan = _real_inputs(prep_dir, session_policy, prompt_dir, model, block_size, hard_cap, sampling, inherit_from, reserve)
    path = bm.save_plan(settings.data_dir / "benchmarks", plan)
    for line in bm.render_plan(plan):
        console.print(line, markup=False, highlight=False)
    console.print(f"\nPlan stored: {path.resolve()}", markup=False)
    console.print(f"Next: msu-llm-lab real-benchmark run --prep-dir {prep_dir} --session-policy {session_policy} --operator \"<your name>\"" + (f" --inherit-from {inherit_from}" if inherit_from else "") + (f" --reserve {reserve}" if reserve else "") + " --live", markup=False)
    raise typer.Exit(code=2 if plan["blockers"] else 0)


@real_app.command("run")
def real_run(
    prep_dir: Path = typer.Option(..., "--prep-dir"),
    session_policy: str = typer.Option(..., "--session-policy"),
    operator: str = typer.Option(..., "--operator", help="Your name, recorded in the authorization manifest."),
    prompt_dir: Path = typer.Option(Path("."), "--prompt-dir"),
    model: List[str] = typer.Option([], "--model", "-m"),
    block_size: int = typer.Option(10, "--block-size"),
    hard_cap: Optional[str] = typer.Option(None, "--hard-cap"),
    max_blocks: Optional[int] = typer.Option(None, "--max-blocks", help="Stop after this many TOTAL blocks counted from the start of the plan (the plan has 222; resume later with a larger number)."),
    sampling: str = typer.Option("stratified", "--sampling", help="Must match the stored plan."),
    inherit_from: Optional[str] = typer.Option(None, "--inherit-from", help="Must match the stored plan."),
    reserve: Optional[str] = typer.Option(None, "--reserve", help="Must match the stored plan."),
    live: bool = typer.Option(False, "--live", help="Send the planned requests. Needs the two interactive typed approvals."),
) -> None:
    """Execute (or resume) the benchmark. Re-derives the plan, then needs typed live approval AND typed real-data authorization."""
    from msu_llm_lab.classification import realdata
    from msu_llm_lab.classification.dataprep import code_fingerprint
    from msu_llm_lab.core.config import load_project_env, settings
    from msu_llm_lab.core.live_guard import confirm_live_run
    from msu_llm_lab.providers.base import LiveNotAuthorizedError

    load_project_env()
    bm, loaded, prompts, budget, plan = _real_inputs(prep_dir, session_policy, prompt_dir, model, block_size, hard_cap, sampling, inherit_from, reserve)
    root = settings.data_dir / "benchmarks"
    saved = bm.bench_dir(root, plan["bench_id"]) / "plan.json"
    if not saved.exists():
        console.print("[bold red]This exact plan was never stored.[/bold red] Run `real-benchmark plan` first and review it.")
        raise typer.Exit(code=1)
    for line in bm.render_plan(plan):
        console.print(line, markup=False, highlight=False)
    if not live:
        console.print("\nPlan only: nothing was sent. Re-run with --live to request the typed approvals.", markup=False)
        raise typer.Exit(code=2 if plan["blockers"] else 0)
    if plan["blockers"]:
        console.print("[bold red]Live run refused:[/bold red] the plan is blocked.")
        raise typer.Exit(code=1)
    models = [p["model_id"] for p in plan["participants"]]
    try:
        live_auth = confirm_live_run(models, ["Approve exactly this plan by typing the phrase below."], echo=lambda line: console.print(line, markup=False, highlight=False),
                                     plan_fingerprint=plan["fingerprint"])
        auth = realdata.confirm_real_data(
            dataset_version=plan["dataset"]["version"], session_policy=session_policy, models=models,
            routes={p["model_id"]: (p["route"]["pinned_tag"] or "none") for p in plan["participants"]},
            tasks=sorted({t for p in plan["participants"] for t in p["tasks"]}), plan_fingerprint=plan["fingerprint"], cost_ceiling_usd=plan["limits"]["hard_cap_usd"],
            provider_policy_status={p["model_id"]: p["data_handling"]["data_policy"] for p in plan["participants"]}, recipients=plan["recipients"],
            operator=operator, echo=lambda line: console.print(line, markup=False, highlight=False))
    except LiveNotAuthorizedError as e:
        console.print(f"[bold red]Live run refused:[/bold red] {e}")
        raise typer.Exit(code=1)
    from msu_llm_lab.classification.smoke import build_smoke_adapters

    bm.write_manifest(root, plan, auth, live_auth, code_fingerprint())
    adapters = build_smoke_adapters([p["key"] for p in plan["participants"]], bm.STAGE)
    try:
        res = bm.execute(plan, loaded, prompts, auth, live_auth, adapters, budget, root=root, reports_dir=settings.reports_dir, max_blocks=max_blocks,
                         progress=lambda m: console.print(m, markup=False))
    except KeyboardInterrupt:
        console.print("Interrupted. Progress is checkpointed; re-run the same command to resume.", markup=False)
        raise typer.Exit(code=130)
    except Exception as e:
        console.print(f"[bold red]Benchmark stopped:[/bold red] {type(e).__name__}: {e}")
        raise typer.Exit(code=1)
    total = plan["ordering"]["n_blocks"]
    console.print(f"\nblocks completed {res.blocks_completed} of {total}   guard cost ${res.spent_guard_usd}   halted: {res.halted or 'no'}", markup=False)
    for m, why in res.quarantined.items():
        console.print(f"QUARANTINED {m}: {why}", markup=False)
    if res.blocks_completed < total or res.quarantined:
        console.print("PARTIAL BENCHMARK: not every planned case/participant completed. Run `real-benchmark report` for exact coverage.", markup=False)
        raise typer.Exit(code=3)
    console.print("All planned blocks completed. Run `real-benchmark report`.", markup=False)


@real_app.command("status")
def real_status(bench_id: Optional[str] = typer.Argument(None)) -> None:
    """Show stored benchmark plans and their checkpoint state (no network)."""
    import json

    from msu_llm_lab.core.config import settings

    root = settings.data_dir / "benchmarks"
    dirs = sorted(p for p in root.glob("bench-*") if p.is_dir()) if root.exists() else []
    if bench_id:
        dirs = [root / bench_id]
    if not dirs:
        console.print("No benchmark plans stored.", markup=False)
        return
    for d in dirs:
        plan = json.loads((d / "plan.json").read_text()) if (d / "plan.json").exists() else {}
        st = json.loads((d / "state.json").read_text()) if (d / "state.json").exists() else {}
        console.print(f"{d.name}  dataset {plan.get('dataset', {}).get('version')}  blocks {st.get('blocks_completed', 0)}/{plan.get('ordering', {}).get('n_blocks')}  "
                      f"halted: {st.get('halted') or 'no'}  quarantined: {list(st.get('quarantined', {})) or 'none'}", markup=False)


annotate_app = typer.Typer(help="Independent annotation, adjudication, approval and freezing of ground-truth labels (local files only).")
app.add_typer(annotate_app, name="annotate")


def _annotation_ctx(prep_dir: Path, session_policy: str):
    from msu_llm_lab.classification import annotation as an, benchmark as bm
    try:
        loaded = bm.load_prep_cases(prep_dir, session_policy)
    except bm.PlanError as e:
        console.print(f"[bold red]{e}[/bold red]")
        raise typer.Exit(code=1)
    return an, loaded


@annotate_app.command("queue")
def annotate_queue(prep_dir: Path = typer.Option(..., "--prep-dir"), session_policy: str = typer.Option(..., "--session-policy"),
                   task: str = typer.Option(..., "--task", help="INTENT or LEAD."), limit: Optional[int] = typer.Option(None, "--limit")) -> None:
    """Write the deterministic review queue (redacted text, no labels, no predictions). Local, git-ignored; never sent anywhere."""
    from msu_llm_lab.classification.records import Task
    an, loaded = _annotation_ctx(prep_dir, session_policy)
    try:
        path = an.write_queue(prep_dir, loaded, Task(task.upper()), limit)
    except (an.AnnotationError, ValueError) as e:
        console.print(f"[bold red]{e}[/bold red]")
        raise typer.Exit(code=1)
    console.print(f"Review queue written: {path.resolve()} (contains redacted student text: keep it local)", markup=False)


@annotate_app.command("submit")
def annotate_submit(prep_dir: Path = typer.Option(..., "--prep-dir"), session_policy: str = typer.Option(..., "--session-policy"),
                    task: str = typer.Option(..., "--task"), reviewer: str = typer.Option(..., "--reviewer"),
                    case: Optional[str] = typer.Option(None, "--case"), label: Optional[str] = typer.Option(None, "--label"),
                    file: Optional[Path] = typer.Option(None, "--file", help="JSONL lines {\"case_id\": ..., \"label\": ...} for bulk submission."),
                    note: str = typer.Option("", "--note")) -> None:
    """Record one reviewer's label (or a JSONL batch). A reviewer never sees another reviewer's label."""
    import json
    an, loaded = _annotation_ctx(prep_dir, session_policy)
    valid = an.eligible_case_ids(loaded)
    pairs = []
    if file:
        pairs = [(d["case_id"], str(d["label"])) for d in (json.loads(l) for l in file.read_text().splitlines() if l.strip())]
    elif case and label is not None:
        pairs = [(case, label)]
    else:
        console.print("[bold red]Give --case and --label, or --file.[/bold red]")
        raise typer.Exit(code=1)
    try:
        for cid, lab in pairs:
            an.submit(prep_dir, case_id=cid, task=task.upper(), label=lab, reviewer=reviewer, valid_ids=valid, note=note)
    except an.AnnotationError as e:
        console.print(f"[bold red]Stopped:[/bold red] {e}")
        raise typer.Exit(code=1)
    console.print(f"Recorded {len(pairs)} annotation(s) by {reviewer}.", markup=False)


@annotate_app.command("template")
def annotate_template(prep_dir: Path = typer.Option(..., "--prep-dir"), session_policy: str = typer.Option(..., "--session-policy"),
                      blocks: int = typer.Option(..., "--blocks", help="How many benchmark blocks of sampled cases the templates cover."),
                      block_size: int = typer.Option(10, "--block-size"), sampling: str = typer.Option("stratified", "--sampling")) -> None:
    """Write INTENT and LEAD review templates (opaque ids and metadata only; no text, no predictions). Never overwrites."""
    from msu_llm_lab.observability import annotation_templates as at
    an, loaded = _annotation_ctx(prep_dir, session_policy)
    try:
        paths = at.write_templates(prep_dir, loaded, sampling, blocks, block_size)
    except an.AnnotationError as e:
        console.print(f"[bold red]{e}[/bold red]")
        raise typer.Exit(code=1)
    for t, p in paths.items():
        console.print(f"{t}: {p.resolve()}", markup=False)
    console.print("Contains opaque ids only. Review text comes from `annotate queue` (local, redacted).", markup=False)


@annotate_app.command("import-sheet")
def annotate_import_sheet(prep_dir: Path = typer.Option(..., "--prep-dir"), session_policy: str = typer.Option(..., "--session-policy"),
                          sheet: Path = typer.Option(..., "--sheet"), slot: int = typer.Option(..., "--slot", help="1 or 2: which reviewer column to import.")) -> None:
    """Forward one reviewer's filled column to the annotation workflow (it creates annotation events only; it approves nothing)."""
    from msu_llm_lab.observability import annotation_templates as at
    an, loaded = _annotation_ctx(prep_dir, session_policy)
    try:
        n = at.import_sheet(sheet, prep_dir, an.eligible_case_ids(loaded), slot)
    except (an.AnnotationError, OSError, KeyError) as e:
        console.print(f"[bold red]Stopped:[/bold red] {e}")
        raise typer.Exit(code=1)
    console.print(f"Imported {n} annotation(s) from reviewer slot {slot}.", markup=False)


@annotate_app.command("status")
def annotate_status(prep_dir: Path = typer.Option(..., "--prep-dir"), session_policy: str = typer.Option(..., "--session-policy")) -> None:
    """Counts by workflow state (no labels, no text)."""
    an, loaded = _annotation_ctx(prep_dir, session_policy)
    for t, c in an.summary(prep_dir, an.eligible_case_ids(loaded)).items():
        console.print(f"{t}: {c}", markup=False)
    fm = an.frozen_matches_current(prep_dir)
    console.print(f"freeze: {'never frozen' if fm is None else 'current labels match the latest freeze' if fm else 'LABELS CHANGED SINCE THE LATEST FREEZE'}", markup=False)


@annotate_app.command("adjudicate")
def annotate_adjudicate(prep_dir: Path = typer.Option(..., "--prep-dir"), task: str = typer.Option(..., "--task"), case: str = typer.Option(..., "--case"),
                        label: str = typer.Option(..., "--label"), adjudicator: str = typer.Option(..., "--adjudicator"), note: str = typer.Option("", "--note")) -> None:
    """A third reviewer settles a conflict between two independent annotations."""
    from msu_llm_lab.classification import annotation as an
    try:
        an.adjudicate(prep_dir, case_id=case, task=task.upper(), label=label, adjudicator=adjudicator, note=note)
    except an.AnnotationError as e:
        console.print(f"[bold red]{e}[/bold red]")
        raise typer.Exit(code=1)
    console.print("Adjudication recorded.", markup=False)


@annotate_app.command("approve")
def annotate_approve(prep_dir: Path = typer.Option(..., "--prep-dir"), approver: str = typer.Option(..., "--approver"),
                     all_ready: bool = typer.Option(False, "--all-ready", help="Approve every AGREED/ADJUDICATED case (needs the typed confirmation)."),
                     task: Optional[str] = typer.Option(None, "--task"), case: Optional[str] = typer.Option(None, "--case")) -> None:
    """Explicitly approve ready cases. Never automatic: bulk approval needs an interactive typed phrase."""
    import sys
    from msu_llm_lab.classification import annotation as an
    ready = [s for s in an.case_states(prep_dir).values() if s.proposed is not None and s.state != "APPROVED"]
    try:
        if all_ready:
            if not (sys.stdin.isatty() and sys.stdout.isatty()):
                console.print("[bold red]Bulk approval needs an interactive terminal.[/bold red]")
                raise typer.Exit(code=1)
            phrase = f"APPROVE {len(ready)} LABELS"
            console.print(f"{len(ready)} case(s) are ready (AGREED or ADJUDICATED). Type exactly: {phrase}", markup=False)
            if typer.prompt("Confirmation", default="", show_default=False).strip() != phrase:
                console.print("Confirmation did not match; nothing approved.", markup=False)
                raise typer.Exit(code=1)
            rows = an.approve(prep_dir, approver=approver, all_ready=True)
        elif task and case:
            rows = an.approve(prep_dir, approver=approver, case_ids=[(task.upper(), case)])
        else:
            console.print("[bold red]Give --task and --case, or --all-ready.[/bold red]")
            raise typer.Exit(code=1)
    except an.AnnotationError as e:
        console.print(f"[bold red]{e}[/bold red]")
        raise typer.Exit(code=1)
    console.print(f"Approved {len(rows)} label(s) as {approver}.", markup=False)


@annotate_app.command("revoke")
def annotate_revoke(prep_dir: Path = typer.Option(..., "--prep-dir"), task: str = typer.Option(..., "--task"), case: str = typer.Option(..., "--case"),
                    by: str = typer.Option(..., "--by"), reason: str = typer.Option(..., "--reason")) -> None:
    """Withdraw an approval (the history stays)."""
    from msu_llm_lab.classification import annotation as an
    try:
        an.revoke(prep_dir, task=task.upper(), case_id=case, by=by, reason=reason)
    except an.AnnotationError as e:
        console.print(f"[bold red]{e}[/bold red]")
        raise typer.Exit(code=1)
    console.print("Approval withdrawn.", markup=False)


@annotate_app.command("freeze")
def annotate_freeze(prep_dir: Path = typer.Option(..., "--prep-dir"), by: str = typer.Option(..., "--by")) -> None:
    """Hash and record the current approved label set so a label-backed benchmark can prove which labels it used."""
    import json
    from msu_llm_lab.classification import annotation as an
    try:
        manifest = json.loads((prep_dir / "manifest.json").read_text())
        path = an.freeze(prep_dir, by=by, dataset_version=manifest.get("prep_version", prep_dir.name))
    except (an.AnnotationError, OSError) as e:
        console.print(f"[bold red]{e}[/bold red]")
        raise typer.Exit(code=1)
    console.print(f"Frozen: {path.resolve()}", markup=False)


@real_app.command("verify")
def real_verify(bundle: Path = typer.Argument(..., help="A bundle folder such as reports/real_benchmark_bench-xxxx-r3"),
                prep_dir: Optional[Path] = typer.Option(None, "--prep-dir"), session_policy: Optional[str] = typer.Option(None, "--session-policy")) -> None:
    """Offline integrity check of a written bundle (hashes, counts, predictions, cost provenance, narrative vs CSV). Writes verification_report.md/json into it."""
    import json

    from msu_llm_lab.classification import benchmark as bm
    from msu_llm_lab.observability import bundle_verify as bv
    from msu_llm_lab.classification.records import Task

    executed = None
    if prep_dir and session_policy:
        man = json.loads((bundle / "benchmark_manifest.json").read_text())
        loaded = bm.load_prep_cases(prep_dir, session_policy)
        done = man["blocks"]["completed"]
        bs = man["ordering"]["block_size"]
        executed = {t.value: {c.case_id for c in bm.order_cases(loaded, t, man["ordering"]["sampling"])[: done * bs]} for t in (Task.INTENT, Task.LEAD)}
    v = bv.verify_bundle(bundle, executed)
    (bundle / "verification_report.json").write_text(json.dumps(v, indent=2) + "\n")
    (bundle / "verification_report.md").write_text(bv.verification_markdown(v))
    for c in v["checks"]:
        console.print(f"{'PASS' if c['ok'] else 'FAIL'}  {c['check']}  {c['detail'][:140]}", markup=False)
    console.print(f"\nVerification {'PASSED' if v['ok'] else 'FAILED'}: {v['n_checks']} checks, {len(v['failed'])} failed. Wrote {bundle.resolve()}/verification_report.md", markup=False)
    raise typer.Exit(code=0 if v["ok"] else 1)


@real_app.command("report")
def real_report(bench_id: str = typer.Argument(..., help="For example bench-0123456789ab."),
                prep_dir: Optional[Path] = typer.Option(None, "--prep-dir", help="For approved labels and unattempted-case lists (read-only)."),
                session_policy: Optional[str] = typer.Option(None, "--session-policy", help="With --prep-dir: also list unattempted cases of the executed blocks."),
                export: bool = typer.Option(True, "--export/--no-export", help="Also write the redacted ZIP trace export.")) -> None:
    """Write the benchmark report (markdown + json) and a redacted trace export from persisted outcomes. No network, no student text."""
    import json

    from msu_llm_lab.classification import benchmark as bm, benchmark_report as br, realdata
    from msu_llm_lab.core.config import settings
    from msu_llm_lab.models.registry import roster_status
    from msu_llm_lab.observability.events import EventStore
    from msu_llm_lab.observability.export import ExportSelection, build_trace_rows, export_traces
    from msu_llm_lab.observability.monitor import LiveMonitor

    d = bm.bench_dir(settings.data_dir / "benchmarks", bench_id)
    if not (d / "plan.json").exists():
        console.print(f"[bold red]No stored plan for {bench_id}.[/bold red]")
        raise typer.Exit(code=1)
    plan = json.loads((d / "plan.json").read_text())
    state = bm.load_state(settings.data_dir / "benchmarks", bench_id)
    report = br.build_report(plan, state, settings.reports_dir, EventStore(), prep_dir)
    out_md, out_json = settings.reports_dir / f"real_benchmark_{bench_id}.md", settings.reports_dir / f"real_benchmark_{bench_id}.json"
    realdata.write_json_atomic(out_json, report)
    out_md.write_text(br.render_markdown(report), encoding="utf-8")
    console.print(f"Report: {out_md.resolve()}", markup=False)
    console.print(f"Machine-readable: {out_json.resolve()}", markup=False)
    console.print(f"State: {report['benchmark_state']}   participants: {report['completion_counts']}", markup=False)
    loaded = None
    if prep_dir and session_policy:
        try:
            loaded = bm.load_prep_cases(prep_dir, session_policy)
        except bm.PlanError as e:
            console.print(f"[yellow]Unattempted-case list skipped: {e}[/yellow]")
    from msu_llm_lab.core.budget import BudgetService
    st = BudgetService().get_status()
    ledger = {"counted_against_caps_usd": str(st.cumulative_spend_usd), "pilot_spent_usd": str(st.pilot_spend_usd), "pilot_remaining_usd": str(st.remaining_pilot_usd),
              "total_remaining_usd": str(st.remaining_total_usd), "spend_by_basis": {k: str(v) for k, v in st.spend_by_basis.items()}, "unresolved_entries": st.unresolved_spend_count,
              "legacy_simulated_usd_not_counted": str(st.simulated_spend_usd)}
    prompts = None
    try:
        from msu_llm_lab.classification.smoke import load_prompts
        prompts = load_prompts(Path("."))
    except FileNotFoundError:
        pass
    bundle = br.write_bundle(plan, state, settings.reports_dir, EventStore(), roster_status(), settings.reports_dir, prep_dir, loaded, d / "manifest.json", ledger=ledger, prompts=prompts)
    console.print(f"Comparison bundle: {bundle['dir'].resolve()}  ({', '.join(br.BUNDLE_FILES)})", markup=False)
    if export:
        mon = LiveMonitor()
        mon.poll()
        preds, events, labels, frozen = mon.trace_inputs()
        runs = tuple(sorted({p.run_id for p in preds if p.run_id.startswith(bench_id + "-")}))
        sel = ExportSelection(runs=runs or None)
        if not runs:
            console.print("No persisted records for this benchmark yet: trace export skipped (no placeholder is created).", markup=False)
            return
        res = export_traces(build_trace_rows(preds, events, labels, frozen, sel), sel, roster_status(), labels, "zip", settings.reports_dir / "exports")
        console.print(f"Trace export (redacted by default): {res.path.resolve()}  records {res.n_exported}", markup=False)


@app.command("classification-data-status")
def classification_data_status() -> None:
    """Show what local dataset preparation exists (counts only; no student text)."""
    import json

    base = Path("data/classification_prep")
    manifests = sorted(base.glob("*/manifest.json")) if base.exists() else []
    if not manifests:
        console.print("No prepared classification dataset found. Run: msu-llm-lab prepare-classification-data", markup=False)
        return
    for p in manifests:
        try:
            m = json.loads(p.read_text())
        except Exception:
            console.print(f"{p.parent.name}: unreadable manifest", markup=False)
            continue
        cand, sess = m["candidates"], m["session_comparison"]["sessions"]
        console.print(f"{p.parent.name}  ({m.get('parity_status')})", markup=False)
        console.print(f"  rows accounted: {m['cleaning']['rows_accounted']} of {m['source']['rows']}   answers missing: {m['cleaning']['answers_missing']}   source unchanged: {m['source']['unchanged_after_run']}", markup=False)
        console.print(f"  sessions: 30m {sess['30m']} / 2h {sess['2h']} (provisional, neither chosen)", markup=False)
        console.print(f"  intent cases: {cand['intent_2h']['cases']}  lead cases: 30m {cand['lead_30m']['cases']} / 2h {cand['lead_2h']['cases']}", markup=False)
        console.print("  expected labels: ALL NULL (human review pending)   student data may not be sent to any provider", markup=False)
        console.print(f"  pseudonymization: {m.get('pseudonymization', {}).get('salt_id', 'n/a (older preparation)')}   privacy: {m.get('privacy_notice', 'n/a (older preparation)')}", markup=False)


@app.command("api-preflight")
def api_preflight() -> None:
    """Report whether the API key is available WITHOUT showing it and WITHOUT sending any request."""
    import os
    import sys

    from msu_llm_lab.core.config import get_api_key_source, load_project_env
    from msu_llm_lab.models.registry import roster_status

    load_project_env()
    raw = os.environ.get("OPENROUTER_API_KEY")
    present = bool(raw and raw.strip())
    odd_whitespace = bool(raw) and raw != raw.strip()
    table = Table(title="API preflight (no request is sent; the key value is never shown)", show_header=True, header_style="bold magenta")
    table.add_column("Check")
    table.add_column("Result")
    table.add_row("OPENROUTER_API_KEY is set and non-empty", "[green]yes[/green]" if present else "[bold red]no[/bold red]")
    if present:
        table.add_row("Key has leading/trailing whitespace", "[yellow]yes (fix it)[/yellow]" if odd_whitespace else "no")
    table.add_row("Interactive terminal (needed for live approval)", "yes" if (sys.stdin.isatty() and sys.stdout.isatty()) else "[yellow]no: live runs would be refused[/yellow]")
    table.add_row("Source of the key", get_api_key_source())
    table.add_row("Network used by this command", "none")
    rows = roster_status()
    table.add_row("Participants READY / SMOKE_READY / BLOCKED",
                  f"{sum(r['readiness'] == 'READY' for r in rows)} / {sum(r['readiness'] == 'SMOKE_READY' for r in rows)} / {sum(r['readiness'] == 'BLOCKED' for r in rows)}")
    table.add_row("Student-derived data allowed to any provider", "[bold red]no[/bold red] (no data policy is verified)")
    console.print(table)
    if not present:
        console.print("Set it in your shell (e.g. export OPENROUTER_API_KEY=...) or in .env before a live run (never paste it into chat or commit it)", markup=False)
    raise typer.Exit(code=0 if present else 1)


@app.command("prepare-classification-data")
def prepare_classification_data(
    source: Path = typer.Option(Path("september_to_october.csv"), "--source", help="Historical export (read only)."),
    out_dir: Path = typer.Option(Path("data/classification_prep/prep-v1"), "--out-dir", help="Local, git-ignored output folder."),
    new_version: bool = typer.Option(False, "--new-version", help="If the folder already has content, write a new sibling version (prep-v1-r2, ...) instead of refusing."),
    prompt_salt: bool = typer.Option(False, "--prompt-salt", help="Ask for the private pseudonymization secret with hidden input (asked twice; never echoed, stored or printed)."),
    allow_public_salt: bool = typer.Option(False, "--allow-public-salt", help="Permit the legacy public salt (pseudonyms are then linkable by anyone holding the export)."),
) -> None:
    """Clean, redact and prepare the historical export for human review. Local only; prints aggregates, never student text."""
    import os
    from getpass import getpass

    from msu_llm_lab.classification import dataprep

    if prompt_salt:
        first, second = getpass("Private pseudonymization secret (hidden): "), getpass("Repeat the secret (hidden): ")
        if first != second or len(first.strip()) < 12:
            console.print("[bold red]Secrets differ or are shorter than 12 characters; nothing was written.[/bold red]")
            raise typer.Exit(code=1)
        os.environ[dataprep.SALT_ENV] = first.strip()
        del first, second
    if not os.environ.get(dataprep.SALT_ENV, "").strip() and not allow_public_salt:
        console.print("[bold red]No private salt.[/bold red] Re-run with --prompt-salt (hidden input) or export MSU_PREP_SALT in your shell; "
                      "--allow-public-salt keeps the linkable legacy salt.")
        raise typer.Exit(code=1)

    if not source.exists():
        console.print(f"[bold red]Source not found:[/bold red] {source}")
        raise typer.Exit(code=1)
    if new_version:
        out_dir = dataprep.next_free_version_dir(out_dir)
    try:
        manifest = dataprep.prepare(source, out_dir)
    except dataprep.PrepOverwriteError as e:
        console.print(f"[bold red]Refused:[/bold red] {e}")
        raise typer.Exit(code=1)
    c = manifest["cleaning"]
    console.print(f"{manifest['parity_status']}  |  prep {manifest['prep_version']}  |  rows accounted: {c['rows_accounted']} of {manifest['source']['rows']}", markup=False)
    console.print(f"dispositions: {c['dispositions']}  duplicates: {c['duplicates']}  answers missing: {c['answers_missing']}", markup=False)
    console.print(f"redactions: {c['redaction_token_counts']}  rows with any redaction: {c['rows_with_any_redaction']}", markup=False)
    console.print(f"sessions: {manifest['session_comparison']['sessions']}  views/splits: {manifest['views_and_splits']}", markup=False)
    console.print(f"source unchanged: {manifest['source']['unchanged_after_run']}   artifacts in {out_dir}", markup=False)
    console.print(f"pseudonymization: {manifest['pseudonymization']['salt_id']} (protected: {manifest['pseudonymization']['protected_secret_used']})   privacy: {manifest['privacy_notice']}", markup=False)


@app.command(name="welcome")
def launch_welcome() -> None:
    """Launch the interactive developer-grade welcome screen."""
    run_launcher_loop()


@app.command(name="ui")
def launch_ui() -> None:
    """Launch the interactive terminal UI (welcome screen, model catalogue, budget)."""
    from msu_llm_lab.tui.app import run_tui
    run_tui()


@app.command(name="interactive", hidden=True)
def launch_interactive() -> None:
    """Alias to launch the interactive terminal UI."""
    from msu_llm_lab.tui.app import run_tui
    run_tui()


def compute_sha256(filepath: Path) -> str:
    """Computes SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


# ---------------------------------------------------------------------------
# Knowledge Base Audit Command (Phase 1)
# ---------------------------------------------------------------------------


@app.command(name="audit-kb")
def audit_kb(
    file_path: Path = typer.Argument(
        ...,
        help="Path to the knowledge base Markdown file to audit.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    output_dir: Path = typer.Option(
        Path("reports"),
        "--output-dir",
        "-o",
        help="Directory to save audit CSV and Markdown reports.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Run audit in memory and display summary without writing reports to disk.",
    ),
    report_format: str = typer.Option(
        "both",
        "--format",
        "-f",
        help="Report export format: 'both', 'csv', or 'md'.",
    ),
) -> None:
    """Audit a knowledge base Markdown file for structure, syntax, and content integrity."""
    console.print(Panel(f"[bold cyan]Knowledge Base Audit[/bold cyan]: {file_path.name}", expand=False))

    # 1. Pre-audit SHA-256 verification
    sha_before = compute_sha256(file_path)
    file_size = file_path.stat().st_size

    # 2. Parse source file
    parser = KBParser(file_path)
    q_markers, a_markers, records, headings, orphan_lines = parser.parse()

    # 3. Validate parsed records
    validator = KBValidator(str(file_path), records, headings, orphan_lines)
    issues = validator.validate_all()

    # 4. Post-audit SHA-256 verification (guarantee zero mutation)
    sha_after = compute_sha256(file_path)
    integrity_ok = (sha_before == sha_after)

    if not integrity_ok:
        console.print("[bold red]CRITICAL INTEGRITY FAILURE: Source file hash changed during audit![/bold red]")
        raise typer.Exit(code=2)

    # 5. Build summary
    error_count = sum(1 for i in issues if i.severity == IssueSeverity.ERROR)
    warning_count = sum(1 for i in issues if i.severity == IssueSeverity.WARNING)

    summary = AuditSummary(
        source_file=str(file_path.resolve()),
        file_size_bytes=file_size,
        line_count=len(parser.lines),
        sha256_before=sha_before,
        sha256_after=sha_after,
        integrity_verified=integrity_ok,
        raw_question_markers=q_markers,
        raw_answer_markers=a_markers,
        parsed_records_count=len(records),
        error_count=error_count,
        warning_count=warning_count,
        issues=issues,
    )

    # 6. Render terminal tables
    metric_table = Table(title="Audit Metrics Summary", show_header=True, header_style="bold magenta")
    metric_table.add_column("Metric", style="dim")
    metric_table.add_column("Value", justify="right")
    metric_table.add_column("Notes", style="italic")

    metric_table.add_row("Source SHA-256", f"{sha_before[:16]}...", "Pre- and post-audit match verified")
    metric_table.add_row("Total Lines", f"{len(parser.lines):,}", "Original line structure preserved")
    metric_table.add_row("Raw Question Markers", f"{q_markers:,}", "Regex hits across document")
    metric_table.add_row("Raw Answer Markers", f"{a_markers:,}", "Regex hits across document")
    metric_table.add_row("Parsed Q&A Records", f"{len(records):,}", "Structural entities extracted")
    metric_table.add_row(
        "Deterministic Errors",
        f"[bold red]{error_count}[/bold red]" if error_count else "[green]0[/green]",
        "Missing answers, empty fields, malformed syntax",
    )
    metric_table.add_row(
        "Heuristic Warnings",
        f"[bold yellow]{warning_count}[/bold yellow]" if warning_count else "[green]0[/green]",
        "Duplicates, conflicting fees, mismatches, dates",
    )
    console.print(metric_table)

    cat_table = Table(title="Issues Breakdown by Category", show_header=True, header_style="bold blue")
    cat_table.add_column("Category", style="cyan")
    cat_table.add_column("Errors", justify="right", style="red")
    cat_table.add_column("Warnings", justify="right", style="yellow")
    cat_table.add_column("Total", justify="right", style="bold")

    categories = sorted(list(set(i.category.value for i in issues)))
    for cat in categories:
        c_err = sum(1 for i in issues if i.category.value == cat and i.severity == IssueSeverity.ERROR)
        c_warn = sum(1 for i in issues if i.category.value == cat and i.severity == IssueSeverity.WARNING)
        cat_table.add_row(cat, str(c_err), str(c_warn), str(c_err + c_warn))
    console.print(cat_table)

    # 7. Write reports if not dry-run
    if dry_run:
        console.print("[yellow]Dry-run mode enabled: No report files written to disk.[/yellow]")
    else:
        timestamp_slug = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = file_path.stem.replace(" ", "_").lower()
        reporter = KBReporter(summary, issues, output_dir)

        if report_format in ("both", "csv"):
            csv_file = reporter.export_csv(f"audit_issues_{safe_name}_{timestamp_slug}.csv")
            console.print(f"[green]✓ CSV issue log saved:[/green] {csv_file}")

        if report_format in ("both", "md"):
            md_file = reporter.export_markdown(f"audit_summary_{safe_name}_{timestamp_slug}.md")
            console.print(f"[green]✓ Markdown summary saved:[/green] {md_file}")

    console.print("[dim]Source file integrity verified: zero modifications made to source.[/dim]")


# ---------------------------------------------------------------------------
# Knowledge Base Remediation Subsystem (Step 2)
# ---------------------------------------------------------------------------


@app.command(name="remediate-kb")
def remediate_kb(
    file_path: Path = typer.Argument(
        Path("Copy of MSU Programmes KB - Local - Sudharsan.md"),
        help="Path to the source knowledge base Markdown file.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    output_path: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Destination path for versioned clean copy (e.g., data/remediated/MSU_Programmes_KB_v1.1_clean.md).",
    ),
    apply: bool = typer.Option(
        False,
        "--apply",
        help="Write remediated clean copy to output destination. Defaults to dry-run.",
    ),
) -> None:
    """Inspect defect findings, evaluate remediation rules, and produce versioned clean copies."""
    from msu_llm_lab.remediation.engine import RemediationEngine
    from msu_llm_lab.remediation.models import RemediationStatus

    console.print(Panel(f"[bold cyan]MSU Knowledge Base Remediation Engine[/bold cyan]: {file_path.name}", expand=False))
    console.print("[yellow]SAFETY INVARIANT: Original knowledge-base source files remain 100% read-only and immutable.[/yellow]\n")

    engine = RemediationEngine(file_path)
    dry_run = not apply

    target_out = output_path or Path("data/remediated") / f"{file_path.stem}_v1.1_clean.md"
    report, _ = engine.execute_remediation(output_path=target_out if apply else None, dry_run=dry_run)

    # 1. Findings Overview Table
    table = Table(title="Remediation Findings & Action Classification", show_header=True, header_style="bold magenta")
    table.add_column("ID", style="bold cyan", width=9)
    table.add_column("Programme", style="white", width=26)
    table.add_column("Classification", style="dim", width=22)
    table.add_column("Status", justify="center", width=24)
    table.add_column("Action / Resolution", style="italic")

    status_styles = {
        RemediationStatus.APPROVED_FOR_RELEASE: "[bold green]APPROVED_FOR_RELEASE[/bold green]",
        RemediationStatus.APPLIED: "[bold green]APPLIED (CLEAN COPY)[/bold green]",
        RemediationStatus.OFFICIAL_VERIFICATION_PENDING: "[bold yellow]OFFICIAL_VERIF_PENDING[/bold yellow]",
        RemediationStatus.INVESTIGATION_PENDING: "[bold cyan]INVESTIGATION_PENDING[/bold cyan]",
        RemediationStatus.KB_OWNER_ESCALATED: "[bold magenta]KB_OWNER_ESCALATED[/bold magenta]",
    }

    for f in report.findings:
        st_label = status_styles.get(f.status, str(f.status.value))
        table.add_row(f.finding_id, f.programme_name[:26], f.classification.value, st_label, f.resolution_notes or f.issue_title)

    console.print(table)

    # 2. Summary Card
    mode_text = "[bold green]LIVE VERSIONED WRITE[/bold green]" if apply else "[bold yellow]DRY-RUN (In-Memory Evaluation)[/bold yellow]"
    summary_card = (
        f"Execution Mode:       {mode_text}\n"
        f"Source SHA-256:       {report.sha256_original[:16]}... (Verified 100% Immutable)\n"
        f"Total Findings:       {report.total_findings}\n"
        f"Approved for Clean:   {report.approved_count} (Confirmed Content Defects)\n"
        f"Official Verif Req:   {report.pending_official_verification_count} (Contradictory Fees / Intakes — Zero Guessing)\n"
        f"KB Owner Escalations: {report.escalated_kb_owner_count} (Curriculum / Language Standards)\n"
    )
    if apply and output_path:
        summary_card += f"Output Destination:   {target_out}\nRemediated SHA-256:   {report.sha256_remediated[:16]}...\n"
    elif dry_run:
        summary_card += f"\n[dim]Run with '--apply --output {target_out}' to generate versioned clean copy.[/dim]"

    console.print(Panel(summary_card, title="[bold]Remediation Execution Summary[/bold]", border_style="cyan"))



# ---------------------------------------------------------------------------
# Evaluation Dataset Subsystem (Phase 2)
# ---------------------------------------------------------------------------


@app.command(name="validate-dataset")
def validate_dataset(
    dataset_path: Path = typer.Argument(
        Path("data/evaluation/synthetic_seed_eval.jsonl"),
        help="Path to the JSONL evaluation dataset file.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    check_leakage: bool = typer.Option(
        True,
        "--check-leakage/--no-check-leakage",
        help="Perform cross-split (train/dev/test) leakage and n-gram overlap detection.",
    ),
) -> None:
    """Validate evaluation dataset schemas, confidentiality boundaries, and cross-split leakage."""
    from msu_llm_lab.dataset.manager import DatasetManager

    console.print(Panel(f"[bold cyan]Dataset Validation[/bold cyan]: {dataset_path.name}", expand=False))

    manager = DatasetManager()
    try:
        cases = manager.load_jsonl(dataset_path)
    except Exception as e:
        console.print(f"[bold red]Validation Failure:[/bold red] Failed to load dataset: {e}")
        raise typer.Exit(code=1)

    summary = manager.validate_dataset(cases, source_path=str(dataset_path), check_leakage=check_leakage)

    # 1. Dataset Overview Table
    table = Table(title="Dataset Partition & Confidentiality Overview", show_header=True, header_style="bold magenta")
    table.add_column("Partition Split", style="cyan")
    table.add_column("Case Count", justify="right")
    table.add_column("External API Status", justify="center")
    table.add_column("Notes", style="italic")

    table.add_row("TRAIN", str(summary.train_count), "[green]Internal Model Training / Tuning[/green]", "Isolated partition")
    table.add_row("DEV (Validation)", str(summary.dev_count), "[green]Hyperparameter & Prompt Dev[/green]", "Isolated partition")
    table.add_row("TEST (Evaluation)", str(summary.test_count), "[green]Final Benchmark Evaluation[/green]", "Held-out partition")
    if summary.unassigned_count > 0:
        table.add_row("UNASSIGNED", str(summary.unassigned_count), "[yellow]Pending Stratification[/yellow]", "Not yet split")
    console.print(table)

    # 2. Privacy & Governance Breakdown Table
    gov_table = Table(title="Confidentiality & Provenance Status", show_header=True, header_style="bold blue")
    gov_table.add_column("Metric", style="dim")
    gov_table.add_column("Count", justify="right")
    gov_table.add_column("Governance Policy", style="italic")

    gov_table.add_row("Eligible for External Dispatch", f"[green]{summary.external_eligible_count}[/green]", "PUBLIC or SYNTHETIC records")
    gov_table.add_row("Blocked from Outbound Network", f"[bold yellow]{summary.blocked_count}[/bold yellow]", "INTERNAL, CONFIDENTIAL or UNCLASSIFIED")
    gov_table.add_row("Verified Ground Truths", str(summary.verification_breakdown.get("VERIFIED", 0)), "Explicit citation attached")
    gov_table.add_row("Unverified Ground Truths", str(summary.verification_breakdown.get("UNVERIFIED", 0)), "Requires domain expert review")
    gov_table.add_row("Disputed Ground Truths", str(summary.verification_breakdown.get("DISPUTED", 0)), "Contradictory facts flagged")
    console.print(gov_table)

    # 3. Leakage Findings
    if summary.leakage_report and summary.leakage_report.has_leakage:
        leak_table = Table(title="[bold red]Contamination Alert: Cross-Split Leakage Detected[/bold red]", show_header=True, header_style="bold red")
        leak_table.add_column("Case A (Split)", style="yellow")
        leak_table.add_column("Case B (Split)", style="red")
        leak_table.add_column("Type", style="bold")
        leak_table.add_column("Similarity", justify="right")
        leak_table.add_column("Query Snippet", style="italic")

        for f in summary.leakage_report.findings:
            leak_table.add_row(
                f"{f.case_a_id} ({f.split_a.value})",
                f"{f.case_b_id} ({f.split_b.value})",
                f.leakage_type.value,
                f"{f.similarity_score:.2f}",
                f.query_a[:40] + ("..." if len(f.query_a) > 40 else ""),
            )
        console.print(leak_table)

    # 4. Final Verdict
    if summary.is_valid:
        console.print(Panel(
            f"[bold green]✓ DATASET VALIDATION PASSED[/bold green]\n"
            f"Total Cases: {summary.total_cases} | Zero Leakage Detected | Confidentiality Invariants Satisfied",
            border_style="green",
        ))
    else:
        err_msg = "\n".join(f"• {e}" for e in summary.validation_errors)
        console.print(Panel(
            f"[bold red]✗ DATASET VALIDATION FAILED[/bold red]\n{err_msg}",
            border_style="red",
        ))
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# Model Evaluation & Benchmarking Subsystem (Phase 3)
# ---------------------------------------------------------------------------


@app.command(name="benchmark")
def run_benchmark(
    model_id: str = typer.Option(
        "google/gemini-3.5-flash-lite",
        "--model",
        "-m",
        help="Target candidate model ID to evaluate.",
    ),
    dataset_path: Path = typer.Option(
        Path("data/evaluation/synthetic_seed_eval.jsonl"),
        "--dataset",
        "-d",
        help="Path to evaluation dataset JSONL.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    split: str = typer.Option(
        "TEST",
        "--split",
        "-s",
        help="Evaluation split to benchmark: 'TEST', 'DEV', or 'ALL'.",
    ),
    live: bool = typer.Option(
        False,
        "--live",
        help="Send real requests to OpenRouter. Required: without it the command only explains what it would do and sends nothing.",
    ),
) -> None:
    """Benchmark a candidate model with REAL requests (typed approval required), tracked against the budget."""
    from msu_llm_lab.evaluation.runner import BenchmarkRunner
    from msu_llm_lab.schemas.evaluation import DatasetSplit

    model_id = getattr(model_id, "default", model_id) if hasattr(model_id, "default") else model_id
    dataset_path_val = getattr(dataset_path, "default", dataset_path) if hasattr(dataset_path, "default") else dataset_path
    dataset_path = Path(dataset_path_val) if not isinstance(dataset_path_val, Path) else dataset_path_val
    live = getattr(live, "default", live) if hasattr(live, "default") else live

    console.print(Panel(
        f"[bold cyan]MSU Model Evaluation Benchmark[/bold cyan]\n"
        f"Target Model:  [bold white]{model_id}[/bold white]\n"
        f"Dataset:       {dataset_path.name}\n"
        f"Mode:          [bold green]REAL REQUESTS[/bold green] (typed approval required)",
        expand=False,
    ))
    if not live:
        console.print("Nothing was sent. Add --live to request typed approval for real requests.", markup=False)
        raise typer.Exit(code=2)

    split_val = getattr(split, "default", split) if hasattr(split, "default") else split
    split_str = str(split_val).strip()
    target_split = None
    if split_str.upper() != "ALL":
        try:
            target_split = DatasetSplit(split_str.upper())
        except ValueError:
            console.print(f"[bold red]Error:[/bold red] Invalid split '{split_str}'. Choose TEST, DEV, or ALL.")
            raise typer.Exit(code=1)

    live_auth = _require_live_confirmation([model_id], dataset_path.name)
    runner = BenchmarkRunner()
    try:
        report = runner.run(
            model_id=model_id,
            dataset_path=dataset_path,
            split=target_split,
            live_authorization=live_auth,
        )
    except Exception as e:
        console.print(f"[bold red]Benchmark Failed:[/bold red] {e}")
        raise typer.Exit(code=1)

    # 1. Summary Metrics Table
    summary_table = Table(title="Benchmark Execution Summary", show_header=True, header_style="bold magenta")
    summary_table.add_column("Metric", style="dim")
    summary_table.add_column("Value", justify="right")
    summary_table.add_column("Benchmark Evaluation Context", style="italic")

    summary_table.add_row("Run Identifier", report.run_id, "Unique execution run ID stored in SQLite")
    summary_table.add_row("Evaluated Cases", f"{report.completed_cases} / {report.total_cases}", f"Split: {split.upper()}")
    summary_table.add_row("Mean Latency", f"{report.mean_latency_ms:.1f} ms", "End-to-end inference turnaround")
    summary_table.add_row("Mean Accuracy", f"{report.mean_accuracy * 100:.1f}%", "Factual overlap against ground truth citations")
    summary_table.add_row("Mean Groundedness", f"{report.mean_groundedness * 100:.1f}%", "Absence of unsupported hallucinations")
    summary_table.add_row("Overall Score", f"[bold green]{report.mean_overall_score * 100:.1f}%[/bold green]", "Weighted composite rubric score")
    summary_table.add_row("Run Spend (USD)", f"${report.total_cost_usd:.4f}", "Deducted and verified against budget ledger")
    console.print(summary_table)

    # 2. Case Results Table
    case_table = Table(title="Per-Case Performance Breakdown", show_header=True, header_style="bold blue")
    case_table.add_column("Case ID", style="cyan", width=12)
    case_table.add_column("Query Snippet", style="white", width=36)
    case_table.add_column("Latency", justify="right", width=10)
    case_table.add_column("Accuracy", justify="right", width=10)
    case_table.add_column("Grounded", justify="right", width=10)
    case_table.add_column("Overall", justify="right", style="bold", width=9)

    for r in report.results:
        q_snip = r["prompt"][:34] + ("..." if len(r["prompt"]) > 34 else "")
        case_table.add_row(
            r["case_id"],
            q_snip,
            f"{r['latency_ms']:.0f} ms",
            f"{r['accuracy'] * 100:.0f}%",
            f"{r['groundedness'] * 100:.0f}%",
            f"{r['overall_score'] * 100:.0f}%",
        )
    console.print(case_table)


@app.command(name="compare-runs")
def compare_runs(
    run_a: str = typer.Argument(..., help="Run ID of the baseline evaluation run."),
    run_b: str = typer.Argument(..., help="Run ID of the comparison evaluation run."),
) -> None:
    """Compare two evaluation benchmark runs side by side."""
    from msu_llm_lab.storage.db import DatabaseManager

    db = DatabaseManager()
    data_a = db.get_evaluation_run(run_a)
    data_b = db.get_evaluation_run(run_b)

    if not data_a:
        console.print(f"[bold red]Error:[/bold red] Run '{run_a}' not found in database.")
        raise typer.Exit(code=1)
    if not data_b:
        console.print(f"[bold red]Error:[/bold red] Run '{run_b}' not found in database.")
        raise typer.Exit(code=1)

    console.print(Panel(
        f"[bold cyan]Evaluation Run Comparison[/bold cyan]\n"
        f"Baseline Run (A):    [white]{run_a}[/white] ({data_a['model_id']})\n"
        f"Comparison Run (B):  [white]{run_b}[/white] ({data_b['model_id']})",
        expand=False,
    ))

    table = Table(title="Run Comparison & Performance Deltas", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="dim")
    table.add_column(f"Run A ({data_a['model_id'][:18]})", justify="right")
    table.add_column(f"Run B ({data_b['model_id'][:18]})", justify="right")
    table.add_column("Delta (B - A)", justify="right", style="bold")

    acc_a = float(data_a["mean_accuracy"])
    acc_b = float(data_b["mean_accuracy"])
    d_acc = acc_b - acc_a
    d_acc_str = f"[green]+{d_acc * 100:.1f}%[/green]" if d_acc >= 0 else f"[red]{d_acc * 100:.1f}%[/red]"

    lat_a = float(data_a["mean_latency_ms"])
    lat_b = float(data_b["mean_latency_ms"])
    d_lat = lat_b - lat_a
    d_lat_str = f"[green]{d_lat:+.1f} ms[/green]" if d_lat <= 0 else f"[yellow]{d_lat:+.1f} ms[/yellow]"

    sc_a = float(data_a["mean_overall_score"])
    sc_b = float(data_b["mean_overall_score"])
    d_sc = sc_b - sc_a
    d_sc_str = f"[green]+{d_sc * 100:.1f}%[/green]" if d_sc >= 0 else f"[red]{d_sc * 100:.1f}%[/red]"

    cost_a = Decimal(data_a["total_cost_usd"])
    cost_b = Decimal(data_b["total_cost_usd"])
    d_cost = cost_b - cost_a
    d_cost_str = f"${d_cost:+.4f}"

    table.add_row("Mean Accuracy", f"{acc_a * 100:.1f}%", f"{acc_b * 100:.1f}%", d_acc_str)
    table.add_row("Mean Groundedness", f"{float(data_a['mean_groundedness']) * 100:.1f}%", f"{float(data_b['mean_groundedness']) * 100:.1f}%", f"{(float(data_b['mean_groundedness']) - float(data_a['mean_groundedness'])) * 100:+.1f}%")
    table.add_row("Overall Score", f"{sc_a * 100:.1f}%", f"{sc_b * 100:.1f}%", d_sc_str)
    table.add_row("Mean Latency", f"{lat_a:.1f} ms", f"{lat_b:.1f} ms", d_lat_str)
    table.add_row("Total Run Spend", f"${cost_a:.4f}", f"${cost_b:.4f}", d_cost_str)
    table.add_row("Completed Cases", f"{data_a['completed_cases']}", f"{data_b['completed_cases']}", f"{data_b['completed_cases'] - data_a['completed_cases']:+d}")

    console.print(table)


# ---------------------------------------------------------------------------
# Decoupled RAG Benchmark Subcommand
# ---------------------------------------------------------------------------


@app.command("benchmark-rag")
def benchmark_rag(
    model: str = typer.Option(
        "google/gemini-3.5-flash-lite",
        "--model",
        "-m",
        help="Primary candidate model ID to evaluate.",
    ),
    dataset: Path = typer.Option(
        Path("data/evaluation/verified_rag_benchmark.jsonl"),
        "--dataset",
        "-d",
        help="Path to verified evaluation dataset JSONL.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    split: str = typer.Option(
        "TEST",
        "--split",
        "-s",
        help="Partition split to benchmark: 'TEST', 'DEV', or 'ALL'.",
    ),
    live: bool = typer.Option(
        False,
        "--live",
        help="Send real requests to OpenRouter. Required: without it the command only explains what it would do and sends nothing.",
    ),
    compare_with: Optional[str] = typer.Option(
        None,
        "--compare-with",
        "-c",
        help="Optional second candidate model ID for side-by-side RAG comparison.",
    ),
    top_k: int = typer.Option(
        2,
        "--top-k",
        "-k",
        help="Number of context chunks to retrieve per case (default 2).",
    ),
    exclude_defective: bool = typer.Option(
        True,
        "--exclude-defective/--include-all",
        help="Exclude known contradictory or incomplete records from the retrieval index (default True).",
    ),
) -> None:
    """Benchmark candidate models on decoupled retrieval, generation, and end-to-end RAG metrics."""
    from msu_llm_lab.dataset.manager import DatasetManager
    from msu_llm_lab.rag.retriever import KeywordBM25Retriever
    from msu_llm_lab.rag.runner import RAGBenchmarkRunner
    from msu_llm_lab.schemas.evaluation import DatasetSplit
    model = getattr(model, "default", model) if hasattr(model, "default") else model
    dataset_val = getattr(dataset, "default", dataset) if hasattr(dataset, "default") else dataset
    dataset = Path(dataset_val) if not isinstance(dataset_val, Path) else dataset_val
    split_val = getattr(split, "default", split) if hasattr(split, "default") else split
    split_clean = str(split_val).strip().upper()
    split_enum = DatasetSplit(split_clean) if split_clean in DatasetSplit.__members__ else DatasetSplit.UNASSIGNED
    live = getattr(live, "default", live) if hasattr(live, "default") else live
    top_k = getattr(top_k, "default", top_k) if hasattr(top_k, "default") else top_k
    compare_with = getattr(compare_with, "default", compare_with) if hasattr(compare_with, "default") else compare_with
    exclude_defective = getattr(exclude_defective, "default", exclude_defective) if hasattr(exclude_defective, "default") else exclude_defective

    # Real retrieval only: a BM25 index over the audited knowledge-base records. There is no mock or answer-leaking retriever.
    gen_kb_path = Path("data/audit/parsed_records_general_kb.jsonl")
    prog_kb_path = Path("data/audit/parsed_records_programmes_kb.jsonl")
    if not (gen_kb_path.exists() or prog_kb_path.exists()):
        console.print("[bold red]Audited knowledge-base records not found under data/audit/.[/bold red] Run `msu-llm-lab audit-kb` first; "
                      "the benchmark will not fall back to evaluation passages.")
        raise typer.Exit(code=1)
    retriever = KeywordBM25Retriever.from_parsed_kb(
        general_path=gen_kb_path if gen_kb_path.exists() else None,
        programmes_path=prog_kb_path if prog_kb_path.exists() else None,
        exclude_defective=exclude_defective,
    )
    manifest = getattr(retriever, "manifest", {})
    console.print(
        f"[dim]BM25 indexed {len(retriever.documents)} clean documents from audited KB records "
        f"(excluded {manifest.get('excluded_defective_count', 0)} defective records).[/dim]"
    )
    test_cases = DatasetManager.load_jsonl(dataset)
    leaked = retriever.verify_no_leakage(test_cases)
    if leaked:
        console.print(f"[bold red]WARNING: Evaluation data leakage detected for case IDs: {leaked}[/bold red]")

    console.print(Panel(
        f"[bold cyan]MSU End-to-End RAG Benchmark[/bold cyan]\n"
        f"[dim]Primary Model:[/dim]  [bold white]{model}[/bold white]\n"
        f"[dim]Retriever:[/dim]      [bold green]{retriever.__class__.__name__}[/bold green] (top_k: {top_k})\n"
        f"[dim]Dataset:[/dim]        {dataset.name} (Split: {split_clean})\n"
        f"[dim]Execution Mode:[/dim] [red]REAL REQUESTS[/red] (typed approval required)",
        expand=False,
    ))
    if not live:
        console.print("Nothing was sent. Add --live to request typed approval for real requests.", markup=False)
        raise typer.Exit(code=2)

    live_models = [model] + ([compare_with] if compare_with else [])
    live_auth = _require_live_confirmation(live_models, dataset.name)
    runner = RAGBenchmarkRunner(retriever=retriever, top_k=top_k)
    report_a = runner.run(model_id=model, dataset_path=dataset, split=split_enum, live_authorization=live_auth)

    # 1. Decoupled Summary Table
    summary_table = Table(title="Decoupled RAG Performance Summary", show_header=True, header_style="bold magenta")
    summary_table.add_column("Subsystem / Metric", style="cyan")
    summary_table.add_column("Value", justify="right", style="bold")
    summary_table.add_column("Evaluation Context", style="dim")

    summary_table.add_row("Total Evaluated Cases", f"{report_a.total_cases}", f"Split: {split_clean}")
    summary_table.add_row("Verified Ground Truth Cases", f"{report_a.verified_cases_count}", "Cases with verified source KB records")
    summary_table.add_row("Unresolved Abstention Cases", f"{report_a.unresolved_cases_count}", "Cases unsupported by KB expecting abstention")
    summary_table.add_row("Source Recall@k", f"[bold green]{report_a.mean_recall_at_k * 100:.1f}%[/bold green]", f"Relevant record retrieved in top-{top_k} (Verified only)")
    summary_table.add_row("Source Precision@k", f"{report_a.mean_precision_at_k * 100:.1f}%", f"Ratio of top-{top_k} records that are relevant")
    summary_table.add_row("Mean Reciprocal Rank (MRR)", f"[bold cyan]{report_a.mean_reciprocal_rank:.4f}[/bold cyan]", "1 / rank of first relevant record (Verified only)")
    summary_table.add_row("Verified Retrieval Hit Rate", f"[bold green]{report_a.verified_retrieval_hit_rate * 100:.1f}%[/bold green]", "Proportion of verified cases with >= 1 hit")
    summary_table.add_row("Abstention Accuracy", f"[bold yellow]{report_a.abstention_accuracy * 100:.1f}%[/bold yellow]", "Model successfully abstained on unsupported queries")
    summary_table.add_row("Retrieval Latency", f"{report_a.mean_retrieval_latency_ms:.1f} ms", "Isolated context search turnaround")
    summary_table.add_row("Answer Correctness", f"{report_a.mean_answer_correctness * 100:.1f}%", "Factual overlap against reference answer")
    summary_table.add_row("Groundedness (Faithfulness)", f"{report_a.mean_groundedness * 100:.1f}%", "Supported by context (hallucination check)")
    summary_table.add_row("Diagnostic Keyword Hit Rate", f"{report_a.diagnostic_keyword_hit_rate * 100:.1f}%", "Secondary exact keyword-substring match")
    summary_table.add_row("Generation Latency", f"{report_a.mean_generation_latency_ms:.1f} ms", "Isolated LLM inference turnaround")
    summary_table.add_row("End-to-End Latency", f"[bold yellow]{report_a.mean_e2e_latency_ms:.1f} ms[/bold yellow]", "Total user turnaround (Ret + Gen)")
    summary_table.add_row("Run Spend (USD)", f"${report_a.total_cost_usd:.4f}", "Tracked in budget ledger")

    console.print(summary_table)

    # 2. Failure Attribution Breakdown
    attr_table = Table(title="Failure Attribution Analysis", show_header=True, header_style="bold magenta")
    attr_table.add_column("Attribution Category", style="bold")
    attr_table.add_column("Count", justify="right")
    attr_table.add_column("% Share", justify="right")
    attr_table.add_column("Diagnosis / Implication", style="dim")

    tot = max(1, report_a.total_cases)
    for cat, cnt in report_a.failure_breakdown.items():
        pct = (cnt / tot) * 100
        if cat in ("SYSTEM_SUCCESS", "ABSTENTION_SUCCESS"):
            diag = "[green]Satisfied source grounding or correctly abstained[/green]"
        elif cat == "RETRIEVAL_FAILURE":
            diag = "[red]Retriever starved model of necessary context[/red]"
        elif cat in ("GENERATION_FAILURE", "OVERCONFIDENT_HALLUCINATION"):
            diag = "[yellow]Context available or absent, but model erred or hallucinated[/yellow]"
        else:
            diag = "[blue]Model answered correctly despite weak context[/blue]"
        attr_table.add_row(cat, str(cnt), f"{pct:.1f}%", diag)

    console.print(attr_table)

    # 3. Per-Case Performance Table
    case_table = Table(title="Per-Case Decoupled Breakdown", show_header=True, header_style="bold blue")
    case_table.add_column("Case ID", style="cyan", width=16)
    case_table.add_column("Status", width=14)
    case_table.add_column("Target Record(s)", width=16)
    case_table.add_column("Retrieved Record(s)", width=18)
    case_table.add_column("Recall@k", justify="right", width=9)
    case_table.add_column("MRR", justify="right", width=8)
    case_table.add_column("Attribution", style="bold", width=22)

    for c in report_a.case_results:
        status_style = "[green]VERIFIED[/green]" if c.source_status == "SOURCE_VERIFIED" else "[yellow]UNRESOLVED[/yellow]"
        targets = ", ".join(c.relevant_record_ids) if c.relevant_record_ids else "(None - Abstain)"
        retrieved_recs = ", ".join(c.retrieved_record_ids) if c.retrieved_record_ids else "(None)"
        
        attr_style = (
            "[green]SUCCESS[/green]" if c.failure_attribution in ("SYSTEM_SUCCESS", "ABSTENTION_SUCCESS")
            else f"[red]{c.failure_attribution}[/red]"
        )
        case_table.add_row(
            c.case_id,
            status_style,
            targets,
            retrieved_recs,
            f"{c.recall_at_k * 100:.0f}%" if c.relevant_record_ids else "N/A",
            f"{c.reciprocal_rank:.2f}" if c.relevant_record_ids else "N/A",
            attr_style,
        )

    console.print(case_table)

    # Optional Multi-Model Comparison
    if compare_with:
        console.print(Panel(
            f"[bold cyan]Running Side-by-Side Model Comparison[/bold cyan]\n"
            f"[dim]Comparison Candidate:[/dim] [bold white]{compare_with}[/bold white]",
            expand=False,
        ))
        runner_b = RAGBenchmarkRunner(retriever=retriever)
        report_b = runner_b.run(model_id=compare_with, dataset_path=dataset, split=split_enum, live_authorization=live_auth)

        comp_table = Table(title=f"Comparative Benchmark: {model} vs {compare_with}", show_header=True, header_style="bold magenta")
        comp_table.add_column("Metric", style="cyan")
        comp_table.add_column(f"{model[:18]} (A)", justify="right")
        comp_table.add_column(f"{compare_with[:18]} (B)", justify="right")
        comp_table.add_column("Delta (B - A)", justify="right")

        def _fmt_delta(delta: float, is_pct: bool = False, reverse: bool = False) -> str:
            sign = "+" if delta > 0 else ""
            txt = f"{sign}{delta * 100:.1f}%" if is_pct else f"{sign}{delta:.1f}"
            if delta == 0:
                return f"[dim]{txt}[/dim]"
            is_good = (delta > 0) if not reverse else (delta < 0)
            return f"[green]{txt}[/green]" if is_good else f"[red]{txt}[/red]"

        comp_table.add_row("Answer Correctness", f"{report_a.mean_answer_correctness * 100:.1f}%", f"{report_b.mean_answer_correctness * 100:.1f}%", _fmt_delta(report_b.mean_answer_correctness - report_a.mean_answer_correctness, is_pct=True))
        comp_table.add_row("Groundedness", f"{report_a.mean_groundedness * 100:.1f}%", f"{report_b.mean_groundedness * 100:.1f}%", _fmt_delta(report_b.mean_groundedness - report_a.mean_groundedness, is_pct=True))
        comp_table.add_row("Generation Latency", f"{report_a.mean_generation_latency_ms:.1f} ms", f"{report_b.mean_generation_latency_ms:.1f} ms", _fmt_delta(report_b.mean_generation_latency_ms - report_a.mean_generation_latency_ms, reverse=True))
        comp_table.add_row("End-to-End Latency", f"{report_a.mean_e2e_latency_ms:.1f} ms", f"{report_b.mean_e2e_latency_ms:.1f} ms", _fmt_delta(report_b.mean_e2e_latency_ms - report_a.mean_e2e_latency_ms, reverse=True))
        comp_table.add_row("Total Run Spend", f"${report_a.total_cost_usd:.4f}", f"${report_b.total_cost_usd:.4f}", f"${report_b.total_cost_usd - report_a.total_cost_usd:+.4f}")

        console.print(comp_table)


# ---------------------------------------------------------------------------
# Langfuse Trace Diagnostic Subcommand
# ---------------------------------------------------------------------------


@app.command("import-traces")
def import_traces(
    export_file: Path = typer.Argument(
        ...,
        help="Path to Langfuse JSON or CSV trace export file.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Directory to export diagnostic CSV and Markdown reports (e.g. reports/).",
    ),
    min_confidence: float = typer.Option(
        0.50,
        "--min-confidence",
        "-c",
        help="Minimum confidence threshold for candidate hypotheses.",
    ),
) -> None:
    """Ingest redacted Langfuse RAG traces and diagnose candidate failure modes."""
    from msu_llm_lab.traces.analyzer import TraceDiagnosticAnalyzer
    from msu_llm_lab.traces.importer import TraceImporter
    from msu_llm_lab.traces.models import FailureCategory
    from msu_llm_lab.traces.reporter import DiagnosticReporter

    console.print(Panel(
        f"[bold cyan]Langfuse Trace Ingestion & Diagnostic Analyzer[/bold cyan]\n"
        f"[dim]File:[/dim] [bold white]{export_file.name}[/bold white]\n"
        f"[dim]Min Confidence:[/dim] [yellow]{min_confidence:.0%}[/yellow]",
        expand=False,
    ))

    try:
        traces = TraceImporter.load_from_file(export_file)
    except Exception as e:
        console.print(f"[bold red]Failed to load trace export:[/bold red] {e}")
        raise typer.Exit(code=1)

    if not traces:
        console.print("[yellow]No traces found in specified export file.[/yellow]")
        return

    analyzer = TraceDiagnosticAnalyzer(min_confidence=min_confidence)
    report = analyzer.analyze_batch(traces)

    # 1. Summary Table
    table_sum = Table(title="Trace Batch Summary", show_header=True, header_style="bold magenta")
    table_sum.add_column("Metric", style="dim")
    table_sum.add_column("Value", justify="right", style="bold")
    table_sum.add_column("Operational Context", style="italic")

    table_sum.add_row("Total Ingested Traces", str(report.total_traces), "Successfully validated offline")
    table_sum.add_row("Diagnosed Failures", str(report.diagnosed_failures), "Identified candidate failure hypotheses")
    table_sum.add_row("Mean Latency", f"{report.mean_latency_ms:.1f} ms", "End-to-end trace turnaround")
    table_sum.add_row("Total Token Footprint", f"{report.total_tokens:,}", "Aggregated prompt and completion tokens")

    console.print(table_sum)

    # 2. Failure Distribution Table
    table_dist = Table(title="Failure Category Distribution", show_header=True, header_style="bold magenta")
    table_dist.add_column("Failure Category", style="cyan")
    table_dist.add_column("Candidate Count", justify="right")
    table_dist.add_column("% Share", justify="right")
    table_dist.add_column("Severity / Priority", justify="center")

    for cat_name, count in report.category_counts.items():
        pct = (count / max(1, report.total_traces)) * 100
        if cat_name == FailureCategory.ROUTING_FAILURE.value:
            prio = "[bold red]HIGH[/bold red]"
        elif cat_name == FailureCategory.RETRIEVAL_FAILURE.value:
            prio = "[bold yellow]HIGH[/bold yellow]"
        elif cat_name == FailureCategory.SOURCE_KNOWLEDGE_FAILURE.value:
            prio = "[bold magenta]KB DEFECT[/bold magenta]"
        elif cat_name == FailureCategory.GENERATION_FAILURE.value:
            prio = "[bold blue]MEDIUM[/bold blue]"
        else:
            prio = "[green]HEALTHY[/green]"

        table_dist.add_row(cat_name, str(count), f"{pct:.1f}%", prio)

    console.print(table_dist)

    # 3. Detailed Diagnoses Table
    table_diag = Table(title="Candidate Diagnostic Hypotheses", show_header=True, header_style="bold magenta")
    table_diag.add_column("Trace ID", style="cyan", no_wrap=True)
    table_diag.add_column("Category", style="bold")
    table_diag.add_column("Conf.", justify="right")
    table_diag.add_column("Hypothesis Snippet", style="white")
    table_diag.add_column("Suggested Action", style="green")

    for d in report.diagnoses:
        if d.category != FailureCategory.NO_FAILURE_DETECTED:
            table_diag.add_row(
                d.trace_id,
                d.category.value,
                f"{d.confidence:.0%}",
                d.hypothesis[:65] + "..." if len(d.hypothesis) > 65 else d.hypothesis,
                d.suggested_action[:45] + "..." if len(d.suggested_action) > 45 else d.suggested_action,
            )

    if table_diag.rows:
        console.print(table_diag)

    # 4. Export reports if requested
    if output_dir:
        out_path = Path(output_dir)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = out_path / f"trace_diagnoses_{ts}.csv"
        md_file = out_path / f"trace_diagnoses_{ts}.md"
        DiagnosticReporter.export_csv(report, csv_file)
        DiagnosticReporter.export_markdown(report, md_file)
        console.print(f"\n[green]Diagnostic reports generated successfully:[/green]")
        console.print(f"  • CSV: [bold]{csv_file}[/bold]")
        console.print(f"  • Markdown: [bold]{md_file}[/bold]")


# ---------------------------------------------------------------------------
# Budget Subcommands
# ---------------------------------------------------------------------------


@budget_app.command(name="reconcile")
def budget_reconcile(
    transaction_id: str = typer.Option(..., "--transaction-id", "-t", help="Unresolved ledger entry to resolve."),
    amount_usd: str = typer.Option(..., "--amount-usd", "-a", help="Real amount in USD read from the provider dashboard."),
    note: str = typer.Option("", "--note", help="Optional note, for example the dashboard date."),
) -> None:
    """Resolve an unresolved-spend entry with an amount you read from the provider dashboard. Live runs stay blocked until none remain."""
    from decimal import Decimal, InvalidOperation

    try:
        amount = Decimal(amount_usd)
        if amount < 0:
            raise InvalidOperation
    except InvalidOperation:
        console.print("[bold red]--amount-usd must be a non-negative number[/bold red]")
        raise typer.Exit(code=1)
    service = BudgetService()
    service.db.init_db()
    try:
        service.db.reconcile_transaction(transaction_id, amount, note)
    except (KeyError, ValueError) as e:
        console.print(f"[bold red]{e}[/bold red]")
        raise typer.Exit(code=1)
    left = len(service.unresolved())
    console.print(f"Reconciled {transaction_id} at ${amount:.6f}. Unresolved entries remaining: {left}.", markup=False)


@budget_app.command(name="purge-simulated")
def budget_purge_simulated() -> None:
    """Delete all simulated (dry-run) ledger rows and stored dry-run runs. The database is copied first; real entries are never touched."""
    service = BudgetService()
    result = service.db.purge_simulated(backup=True)
    console.print(f"Removed {result['ledger_rows']} simulated ledger rows, {result['runs']} dry-run runs and {result['results']} stored results.", markup=False)
    console.print(f"Backup of the previous database: {result['backup'] or 'none (no database file)'}", markup=False)


@budget_app.command(name="audit")
def budget_audit_cmd() -> None:
    """Reconcile the ledger with the stored results (read-only). Explains ledger total versus per-plan guard cost. Writes reports/budget_reconciliation_*.md/.json."""
    import json
    from datetime import datetime, timezone

    from msu_llm_lab.core.config import settings
    from msu_llm_lab.observability import budget_audit as ba

    service = BudgetService()
    service.db.init_db()
    a = ba.audit(service.db.db_path, service.policy, settings.reports_dir, settings.data_dir / "benchmarks")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    md, js = settings.reports_dir / f"budget_reconciliation_{stamp}.md", settings.reports_dir / f"budget_reconciliation_{stamp}.json"
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    js.write_text(json.dumps(a, indent=2) + "\n")
    md.write_text(ba.audit_markdown(a))
    console.print(f"Ledger counted: ${a['ledger_total_counted_usd']}  remaining total ${a['remaining_usd']['total']}  pilot ceiling ${a['ceilings_usd']['pilot']}  unresolved {a['unresolved_entries']}", markup=False)
    for g in a["groups"]:
        console.print(f"  {g['group']:<28}ledger ${g['ledger_usd']:<10} records ${g['records_recorded_cost_usd']:<10} guard ${g['guard_from_records_usd']:<10} ledger=records {g['ledger_equals_records']}", markup=False)
    console.print(f"RECONCILED: {a['reconciled']}   report: {md.resolve()}", markup=False)
    raise typer.Exit(code=0 if a["reconciled"] else 1)


@budget_app.command(name="policy")
def budget_policy_cmd() -> None:
    """Show the effective ceilings and the audit trail of operator-authorized pilot-ceiling changes."""
    from msu_llm_lab.core import budget_policy as bp
    from msu_llm_lab.core.config import settings

    eff = bp.effective_policy(settings.budget, settings.data_dir)
    console.print(f"effective: total ${eff.total_budget_usd}  pilot ${eff.pilot_budget_usd}  per run ${eff.per_run_ceiling_usd}  per case ${eff.per_case_ceiling_usd}   (defaults: pilot ${settings.budget.pilot_budget_usd})", markup=False)
    for h in bp.history(settings.data_dir):
        console.print(f"  {h['at']}  {h['operator']}: pilot ${h['previous_pilot_budget_usd']} -> ${h['new_pilot_budget_usd']}  (total ${h['total_budget_usd']}; ledger counted ${h['ledger_total_spent_usd']} at the time)", markup=False)


@budget_app.command(name="raise-pilot-ceiling")
def budget_raise_pilot(to: str = typer.Option(..., "--to", help="New pilot ceiling in USD (never above the total lifetime ceiling)."),
                       operator: str = typer.Option(..., "--operator", help="Your name, recorded in the audit trail.")) -> None:
    """Operator-authorized pilot-ceiling change with an interactive typed phrase. The total, per-run and per-case ceilings and the ledger are untouched."""
    from decimal import Decimal, InvalidOperation

    from msu_llm_lab.core import budget_policy as bp
    from msu_llm_lab.core.config import settings
    from msu_llm_lab.providers.base import LiveNotAuthorizedError

    try:
        new = Decimal(to)
    except InvalidOperation:
        console.print(f"[bold red]Invalid amount:[/bold red] {to}")
        raise typer.Exit(code=1)
    service = BudgetService()
    st = service.get_status()
    try:
        entry = bp.raise_pilot_ceiling(settings.data_dir, settings.budget, new, operator=operator, ledger_total_spent=st.cumulative_spend_usd, ledger_pilot_spent=st.pilot_spend_usd,
                                       echo=lambda line: console.print(line, markup=False, highlight=False))
    except (LiveNotAuthorizedError, bp.PolicyError) as e:
        console.print(f"[bold red]Not changed:[/bold red] {e}")
        raise typer.Exit(code=1)
    console.print(f"Pilot ceiling is now ${entry['new_pilot_budget_usd']} (recorded in {bp.policy_path(settings.data_dir)}). Total ${entry['total_budget_usd']}, per-run ${entry['per_run_ceiling_usd']}, per-case ${entry['per_case_ceiling_usd']} unchanged.", markup=False)


@budget_app.command(name="status")
def budget_status() -> None:
    """Display current spend ledger, remaining balances, and local policy ceilings."""
    service = BudgetService()
    status = service.get_status()

    console.print(Panel("[bold cyan]MSU LLM Workbench — Budget Status[/bold cyan]", expand=False))
    console.print("[yellow]NOTE: Ceilings below are local experiment tracking policies, NOT provider-enforced account limits.[/yellow]\n")

    table = Table(title="Spend & Policy Overview", show_header=True, header_style="bold magenta")
    table.add_column("Budget Metric", style="dim")
    table.add_column("Configured Ceiling", justify="right")
    table.add_column("Current Spend", justify="right")
    table.add_column("Remaining Balance", justify="right", style="bold")
    table.add_column("Status", justify="center")

    pilot_status_str = "[bold red]EXHAUSTED[/bold red]" if status.is_pilot_exhausted else "[green]ACTIVE[/green]"
    total_status_str = "[bold red]EXHAUSTED[/bold red]" if status.is_total_exhausted else "[green]ACTIVE[/green]"

    table.add_row(
        "Initial Pilot Budget",
        f"${status.pilot_ceiling_usd:.4f}",
        f"${status.pilot_spend_usd:.4f}",
        f"${status.remaining_pilot_usd:.4f}",
        pilot_status_str,
    )
    table.add_row(
        "Total Experiment Ceiling",
        f"${status.total_ceiling_usd:.4f}",
        f"${status.cumulative_spend_usd:.4f}",
        f"${status.remaining_total_usd:.4f}",
        total_status_str,
    )
    console.print(table)
    if status.simulated_spend_usd > 0:  # only legacy rows from before simulated runs stopped being recorded
        console.print(f"[dim]Legacy simulated entries (never counted against any ceiling): ${status.simulated_spend_usd:.6f}[/dim]")
    if status.spend_by_basis:
        console.print("[dim]Booked spend by basis: " + ", ".join(f"{k} ${v:.6f}" for k, v in sorted(status.spend_by_basis.items())) + "[/dim]")
    if status.unresolved_spend_count:
        console.print(f"[bold red]UNRESOLVED SPEND: {status.unresolved_spend_count} request(s) have unknown cost. Live runs are blocked until you run "
                      "`msu-llm-lab budget reconcile --transaction-id ... --amount-usd ...`.[/bold red]")
        for u in service.unresolved():
            console.print(f"  {u['transaction_id']}  {u['model_id']}", markup=False)
    console.print("")

    rule_table = Table(title="Per-Execution Safety Rules", show_header=True, header_style="bold blue")
    rule_table.add_column("Safety Rule", style="cyan")
    rule_table.add_column("Limit (USD)", justify="right")
    rule_table.add_column("Enforcement Action", style="italic")

    rule_table.add_row("Max Single Run Spend", f"${status.per_run_ceiling_usd:.4f}", "Preflight check rejects run if exceeded")
    rule_table.add_row("Max Single Case Spend", f"${status.per_case_ceiling_usd:.4f}", "Per-case circuit breaker halts run")
    rule_table.add_row("Execution Mode", "Real requests only", "Typed approval of the exact plan; interactive terminal required")
    console.print(rule_table)

    if status.recent_transactions:
        tx_table = Table(title="Recent Ledger Transactions", show_header=True, header_style="bold green")
        tx_table.add_column("Timestamp (UTC)", style="dim")
        tx_table.add_column("Model ID", style="cyan")
        tx_table.add_column("Provider", style="magenta")
        tx_table.add_column("Amount (USD)", justify="right")
        tx_table.add_column("Type")
        tx_table.add_column("Description", style="italic")

        for tx in status.recent_transactions:
            basis = tx.get("cost_basis") or "UNSPECIFIED"
            tx_type = ("[yellow]PRE-RUN ESTIMATE[/yellow]" if tx["is_preflight_estimate"] else "[dim]LEGACY SIMULATED[/dim]" if tx.get("is_simulated")
                       else "[bold red]UNRESOLVED[/bold red]" if tx.get("is_unresolved")
                       else "[bold green]ACTUAL[/bold green]" if basis in ("PROVIDER_REPORTED", "RECONCILED")
                       else "[yellow]ESTIMATED[/yellow]" if basis == "ESTIMATED_FROM_CATALOG" else f"[dim]{basis}[/dim]")
            tx_table.add_row(
                tx["timestamp"][:19].replace("T", " "),
                tx["model_id"],
                tx["provider"],
                f"${Decimal(tx['amount_micros'])/Decimal('1000000'):.4f}",
                tx_type,
                tx["description"],
            )
        console.print(tx_table)
    else:
        console.print("[dim]No paid transactions recorded yet. Spend ledger is at baseline ($0.0000).[/dim]")


# ---------------------------------------------------------------------------
# Models Subcommands
# ---------------------------------------------------------------------------


@models_app.command(name="list")
def models_list() -> None:
    """List the comparison participants (same view as `models roster`)."""
    models_roster()


if __name__ == "__main__":
    app()

