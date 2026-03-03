"""CLI interface for testdata-ai."""

import csv
import io
import json
import sys
import time

import yaml
from typing import Any, Dict, List, Optional

import click

from testdata_ai.contexts import get_context_schema, list_contexts
from testdata_ai.generator import DataGenerator


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
    "fmt",
    default="json",
    show_default=True,
    type=click.Choice(["json", "jsonl", "csv", "yaml"]),
    help="Output format. Write to file via shell redirection: -o csv > data.csv",
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
    context, count, fmt, provider, model, max_tokens, temperature, no_validate, quiet
):
    """Generate realistic test data using AI."""
    try:
        schema = get_context_schema(context)
    except ValueError as e:
        raise click.ClickException(str(e))

    try:
        gen = DataGenerator(
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except (ValueError, ImportError) as e:
        raise click.ClickException(str(e))

    _adjust_max_tokens(gen, schema, count, quiet, user_set=max_tokens is not None)

    records = _run_generation(gen, context, count, no_validate, quiet)
    _report(records, count, gen.provider.max_tokens, quiet)
    _emit(records, fmt)


def _adjust_max_tokens(gen, context_schema, count, quiet, user_set):
    """Auto-increase max_tokens when the estimate exceeds the current limit.

    Skipped when the user explicitly passed --max-tokens (user_set=True).
    """
    if user_set:
        return
    tokens_per_record = max(len(json.dumps(context_schema.sample)) // 4, 50)
    estimated = count * tokens_per_record
    if gen.provider.max_tokens >= estimated:
        return
    new_value = max(gen.provider.max_tokens * 2, estimated)
    gen.set_max_tokens(new_value)
    if not quiet:
        click.echo(
            click.style(
                f"Auto-adjusted --max-tokens to {new_value} for {count} records.",
                fg="yellow",
            ),
            err=True,
        )


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


def _emit(records, fmt):
    """Format records and write to stdout."""
    if fmt == "csv":
        text = _records_to_csv(records)
    elif fmt == "jsonl":
        text = "\n".join(json.dumps(r) for r in records)
    elif fmt == "yaml":
        text = yaml.dump(records, allow_unicode=True, sort_keys=False)
    else:
        text = json.dumps(records, indent=2)
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
    """Simple context manager that prints start/done messages to stderr."""

    def __init__(self, message: str, silent: bool = False):
        self._message = message
        self._silent = silent
        self._start: Optional[float] = None

    def __enter__(self):
        if not self._silent:
            sys.stderr.write(f"  {self._message}...\n")
            sys.stderr.flush()
            self._start = time.monotonic()
        return self

    def __exit__(self, *args):
        if not self._silent and self._start is not None:
            elapsed = time.monotonic() - self._start
            sys.stderr.write(f"  Done ({elapsed:.1f}s)\n")
            sys.stderr.flush()


def _records_to_csv(records: List[Dict[str, Any]]) -> str:
    """Convert records to CSV string, flattening nested dicts."""
    if not records:
        return ""
    flat = [_flatten_dict(r) for r in records]
    fieldnames = list(dict.fromkeys(key for row in flat for key in row))
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(flat)
    return buf.getvalue()


def _flatten_dict(
    d: Dict[str, Any], parent_key: str = "", sep: str = "."
) -> Dict[str, Any]:
    """Flatten nested dict: {'a': {'b': 1}} -> {'a.b': 1}."""
    result: Dict[str, Any] = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            result.update(_flatten_dict(v, new_key, sep))
        elif isinstance(v, list):
            result[new_key] = json.dumps(v)
        else:
            result[new_key] = v
    return result
