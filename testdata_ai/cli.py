"""CLI interface for testdata-ai."""

import csv
import io
import json
import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional

import click

from testdata_ai.contexts import get_context_schema, list_contexts
from testdata_ai.generator import TestDataGenerator


@click.group()
@click.version_option(package_name="testdata-ai")
def cli():
    """AI-powered test data generator for QA engineers."""


@cli.command()
@click.option(
    "--context",
    required=True,
    help="Context name (e.g. ecommerce_customer, banking_user, saas_trial).",
)
@click.option(
    "--count", default=10, show_default=True, help="Number of records to generate."
)
@click.option(
    "-o",
    "--output",
    default=None,
    type=click.Path(),
    help="Output file path. Defaults to stdout.",
)
@click.option(
    "-f",
    "--format",
    "fmt",
    default="json",
    show_default=True,
    type=click.Choice(["json", "csv"]),
    help="Output format.",
)
@click.option(
    "--provider", default=None, help="AI provider (overrides AI_PROVIDER env var)."
)
@click.option("--model", default=None, help="Model name (overrides default).")
@click.option(
    "--max-tokens",
    default=None,
    type=int,
    help="Max tokens for AI response (increase if you get fewer records than expected).",
)
@click.option(
    "--temperature",
    default=None,
    type=float,
    help="Sampling temperature 0.0-1.0 (higher = more creative).",
)
@click.option(
    "--no-validate", is_flag=True, help="Skip schema validation of generated data."
)
@click.option(
    "-q", "--quiet", is_flag=True, help="Suppress status messages (only output data)."
)
def generate(
    context, count, output, fmt, provider, model, max_tokens, temperature, no_validate, quiet
):
    """Generate realistic test data using AI."""
    _apply_overrides(provider, model, max_tokens, temperature)

    try:
        schema = get_context_schema(context)
    except ValueError as e:
        raise click.ClickException(str(e))

    try:
        gen = TestDataGenerator()
    except (ValueError, ImportError) as e:
        raise click.ClickException(str(e))

    _adjust_max_tokens(gen, schema, count, quiet)
    records = _run_generation(gen, context, count, no_validate, quiet)
    _report(records, count, gen.config.max_tokens, quiet)
    _emit(records, fmt, output, quiet)


def _apply_overrides(provider, model, max_tokens, temperature):
    """Push CLI flags into env vars for TestDataGenerator to pick up."""
    if provider:
        os.environ["AI_PROVIDER"] = provider
    resolved = (provider or os.environ.get("AI_PROVIDER", "openai")).upper()
    if model:
        os.environ[f"{resolved}_MODEL"] = model
    if max_tokens is not None:
        os.environ[f"{resolved}_MAX_TOKENS"] = str(max_tokens)
    if temperature is not None:
        os.environ[f"{resolved}_TEMPERATURE"] = str(temperature)


def _adjust_max_tokens(gen, schema, count, quiet):
    """Estimate required tokens and increase max_tokens if needed."""
    sample_tokens = len(json.dumps(schema.sample)) // 3
    estimated = int(sample_tokens * count * 1.3)
    current = gen.config.max_tokens

    if estimated <= current:
        return

    if quiet:
        gen.provider.max_tokens = estimated
        gen.config.max_tokens = estimated
        return

    click.echo(
        click.style(
            f"Estimated tokens needed: ~{estimated} (current max_tokens={current})",
            fg="yellow",
        ),
        err=True,
    )
    choice = click.prompt(
        "How would you like to proceed?",
        type=click.Choice(["increase", "continue", "cancel"]),
        default="increase",
        show_choices=True,
    )
    if choice == "cancel":
        raise click.Abort()
    if choice == "increase":
        gen.provider.max_tokens = estimated
        gen.config.max_tokens = estimated
        click.echo(click.style(f"max_tokens set to {estimated}", fg="green"), err=True)


def _run_generation(gen, context, count, no_validate, quiet):
    """Call the generator with a spinner and translate exceptions."""
    label = f"Generating {count} {context} records ({gen.config.provider}/{gen.config.model})"
    try:
        with _Spinner(label, silent=quiet):
            return gen.generate(context, count=count, validate=not no_validate)
    except ValueError as e:
        raise click.ClickException(str(e))
    except RuntimeError as e:
        raise click.ClickException(f"API error: {e}")


def _report(records, count, current_max, quiet):
    """Print generation summary to stderr."""
    if quiet:
        return
    if len(records) < count:
        click.echo(
            click.style(
                f"Warning: Requested {count} records but received {len(records)}. "
                f"Try increasing with --max-tokens {current_max * 2}",
                fg="yellow",
            ),
            err=True,
        )
    else:
        click.echo(click.style(f"Generated {len(records)} records.", fg="green"), err=True)


def _emit(records, fmt, output, quiet):
    """Format records and write to file or stdout."""
    text = _records_to_csv(records) if fmt == "csv" else json.dumps(records, indent=2)
    if output:
        with open(output, "w") as f:
            f.write(text)
        if not quiet:
            click.echo(click.style(f"Saved to {output}", fg="green"), err=True)
    else:
        click.echo(text)


@cli.command("list-contexts")
@click.option("--category", default=None, help="Filter by category.")
def list_contexts_cmd(category):
    """List all available data contexts."""
    names = list_contexts(category)
    if not names:
        click.echo("No contexts found.")
        return

    click.echo(
        click.style(
            f"{'Context':<25} {'Category':<15} Description", fg="cyan", bold=True
        )
    )
    click.echo("-" * 70)
    for name in names:
        schema = get_context_schema(name)
        click.echo(f"{name:<25} {schema.category:<15} {schema.description}")


@cli.command("show-context")
@click.argument("context")
def show_context(context):
    """Show details of a specific context."""
    try:
        schema = get_context_schema(context)
    except ValueError as e:
        raise click.ClickException(str(e))

    click.echo(click.style(f"Context: {context}", fg="cyan", bold=True))
    click.echo(f"Category: {schema.category}")
    click.echo(f"Description: {schema.description}")
    click.echo()
    click.echo(click.style("Fields:", bold=True))
    for field in schema.fields:
        click.echo(f"  - {field}")
    click.echo()
    click.echo(click.style("Sample record:", bold=True))
    click.echo(json.dumps(schema.sample, indent=2))
    click.echo()
    click.echo(click.style("Prompt hints:", bold=True))
    for hint in schema.prompt_hints:
        click.echo(f"  - {hint}")


class _Spinner:
    """Animated spinner with elapsed time for long-running operations."""

    _FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, message: str, silent: bool = False):
        self._message = message
        self._silent = silent
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def __enter__(self):
        if not self._silent:
            self._thread = threading.Thread(target=self._spin, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *args):
        self._stop.set()
        if self._thread:
            self._thread.join()
            # Clear the spinner line
            sys.stderr.write("\r\033[K")
            sys.stderr.flush()

    def _spin(self):
        start = time.monotonic()
        i = 0
        while not self._stop.wait(0.1):
            elapsed = time.monotonic() - start
            frame = self._FRAMES[i % len(self._FRAMES)]
            sys.stderr.write(
                f"\r{frame} {self._message} ({elapsed:.0f}s)"
            )
            sys.stderr.flush()
            i += 1


def _records_to_csv(records: List[Dict[str, Any]]) -> str:
    """Convert records to CSV string, flattening nested dicts."""
    if not records:
        return ""
    flat = [_flatten_dict(r) for r in records]
    # Union all keys across records for consistent columns
    fieldnames: List[str] = []
    seen: set = set()
    for row in flat:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(flat)
    return buf.getvalue()


def _flatten_dict(
    d: Dict[str, Any], parent_key: str = "", sep: str = "."
) -> Dict[str, Any]:
    """Flatten nested dict: {'a': {'b': 1}} -> {'a.b': 1}."""
    items: list = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep).items())
        elif isinstance(v, list):
            items.append((new_key, json.dumps(v)))
        else:
            items.append((new_key, v))
    return dict(items)
