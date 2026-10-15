"""Fail-closed confirmation for live (outbound API) runs.

A live run needs an explicit, typed confirmation from an interactive terminal. Piped or
otherwise non-interactive execution can never confirm, so it can never reach a provider.
"""

import sys
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

from msu_llm_lab.providers.base import LiveNotAuthorizedError

CONFIRM_PREFIX = "RUN LIVE"


@dataclass(frozen=True)
class LiveAuthorization:
    """Proof that a human confirmed a live run for specific models. Created only by `confirm_live_run`."""
    models: Tuple[str, ...]

    def covers(self, model_id: str) -> bool:
        return model_id in self.models


def expected_phrase(models: Sequence[str], plan_fingerprint: Optional[str] = None) -> str:
    """Typed confirmation text. When a plan fingerprint is given, approval is bound to that exact plan."""
    base = f"{CONFIRM_PREFIX} {', '.join(models)}"
    return f"{base} PLAN {plan_fingerprint}" if plan_fingerprint else base


def confirm_live_run(
    models: Sequence[str],
    summary_lines: Sequence[str] = (),
    *,
    is_interactive: Optional[Callable[[], bool]] = None,
    prompt: Optional[Callable[[str], str]] = None,
    echo: Callable[[str], None] = print,
    plan_fingerprint: Optional[str] = None,
) -> LiveAuthorization:
    """Return a LiveAuthorization only if an interactive user types the exact confirmation phrase.

    Raises LiveNotAuthorizedError otherwise. Nothing here contacts any provider.
    """
    interactive = is_interactive if is_interactive is not None else (lambda: sys.stdin.isatty() and sys.stdout.isatty())
    if not models or any(not m for m in models):
        raise LiveNotAuthorizedError("Live run needs at least one explicit model id")
    if not interactive():
        raise LiveNotAuthorizedError(
            "Live runs require interactive confirmation; non-interactive execution cannot confirm and is refused."
        )
    phrase = expected_phrase(models, plan_fingerprint)
    for line in summary_lines:
        echo(line)
    echo(f"This will send data to an external provider and may spend money. Type exactly: {phrase}")
    if prompt is None:
        import typer
        prompt = lambda msg: typer.prompt(msg, default="", show_default=False)  # noqa: E731
    answer = prompt("Confirmation")
    if answer.strip() != phrase:
        raise LiveNotAuthorizedError("Confirmation phrase did not match; live run refused.")
    return LiveAuthorization(models=tuple(models))


def require_live_authorization(auth: Optional[LiveAuthorization], model_id: str) -> None:
    """Defence in depth for programmatic callers: a live run needs an authorization covering the model."""
    if auth is None or not auth.covers(model_id):
        raise LiveNotAuthorizedError(f"No live authorization covers model '{model_id}'; live run refused.")
