"""CLI interface for testdata-ai."""

import contextlib
import csv
import io
import itertools
import json
import math
import sys
import threading
import time

import yaml
from typing import Any, Dict, List, Optional

import click

from testdata_ai.ai_providers import OllamaProvider
from testdata_ai.contexts import get_context_schema, list_contexts, load_contexts_from_file
from testdata_ai.generator import DataGenerator
from testdata_ai.schema_adapter import model_to_context_schema


@click.group()
@click.version_option(package_name="testdata-ai")
def cli():
    """AI-powered test data generator for QA engineers."""


_context_file_option = click.option(
    "--context-file",
    "context_files",
    multiple=True,
    type=click.Path(exists=True, dir_okay=False),
    help="YAML/JSON file with custom context definitions (repeatable).",
)


def _common_options(f):
    """Options shared by generate commands."""
    for opt in reversed([
        click.option(
            "-o", "--output", "fmt",
            default="json",
            show_default=True,
            type=click.Choice(["json", "jsonl", "csv", "yaml", "sql"]),
            help="Output format. Write to file via shell redirection: -o csv > data.csv",
        ),
        click.option("--provider", default=None, help="AI provider (overrides AI_PROVIDER env var)."),
        click.option("--model", default=None, help="Model name (overrides default)."),
        click.option(
            "--max-tokens", default=None, type=int,
            help="Max tokens for AI response (increase if you get fewer records than expected).",
        ),
        click.option(
            "--temperature", default=None, type=float,
            help="Sampling temperature 0.0-1.0 (higher = more creative).",
        ),
        click.option("--no-validate", is_flag=True, help="Skip schema validation of generated data."),
        click.option(
            "--batch-size", default=10, show_default=True, type=int,
            help="Records per AI call. For count > batch-size, records stream progressively.",
        ),
        click.option("-q", "--quiet", is_flag=True, help="Suppress status messages (only output data)."),
        click.option(
            "--locale", default=None,
            help="Locale/language for generated values (e.g. pl, ja, de). Overrides AI_LOCALE env var.",
        ),
    ]):
        f = opt(f)
    return f


def _make_generator(provider, model, max_tokens, temperature, locale) -> "DataGenerator":
    """Instantiate DataGenerator, converting errors to ClickException."""
    try:
        return DataGenerator(
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            locale=locale,
        )
    except (ValueError, ImportError) as e:
        raise click.ClickException(str(e))


def _load_context_files(context_files, quiet: bool = False) -> None:
    """Load custom context definitions from a sequence of file paths.

    Raises click.ClickException on any error so callers stay clean.
    """
    for path in context_files:
        try:
            names = load_contexts_from_file(path)
        except (ValueError, ImportError, OSError) as e:
            raise click.ClickException(f"--context-file: {e}")
        if not quiet:
            click.echo(
                click.style(f"Loaded context(s) from {path}: {', '.join(names)}", fg="cyan"),
                err=True,
            )


@cli.command()
@click.option(
    "--context",
    default=None,
    help="Context name (e.g. ecommerce_customer, banking_user, saas_trial).",
)
@click.option(
    "--schema-file",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="JSON or YAML file with a JSON Schema definition (alternative to --context).",
)
@click.option(
    "--graph-file",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="YAML or JSON file describing an entity relationship graph (alternative to --context).",
)
@click.option(
    "--count", default=10, show_default=True, help="Number of records to generate."
)
@_common_options
@click.option(
    "--table",
    default=None,
    help="Table name for SQL output (default: context name or 'records').",
)
@_context_file_option
def generate(
    context, schema_file, graph_file, count, fmt, provider, model, max_tokens, temperature,
    no_validate, batch_size, quiet, locale, context_files, table,
):
    """Generate realistic test data using AI."""
    inputs_given = sum(bool(x) for x in [context, schema_file, graph_file])
    if inputs_given == 0:
        raise click.ClickException("Provide --context, --schema-file, or --graph-file.")
    if inputs_given > 1:
        raise click.ClickException(
            "--context, --schema-file, and --graph-file are mutually exclusive."
        )

    _load_context_files(context_files, quiet)

    if table is None:
        table = context if context else "records"
    table = "".join(c if c.isalnum() or c == "_" else "_" for c in table) or "records"
    if table[0].isdigit():
        table = f"_{table}"

    if graph_file:
        _run_graph_file(graph_file, fmt, provider, model, max_tokens, temperature,
                        no_validate, batch_size, quiet, locale)
        return

    if context:
        try:
            schema = get_context_schema(context)
        except ValueError as e:
            raise click.ClickException(str(e))

    gen = _make_generator(provider, model, max_tokens, temperature, locale)

    if schema_file:
        _run_schema_file(gen, schema_file, count, fmt, no_validate, quiet, max_tokens, table)
        return

    _adjust_max_tokens(gen, schema, min(count, batch_size), quiet, user_set=max_tokens is not None)

    records = _run_streaming(gen, context, count, batch_size, fmt, no_validate, quiet, table)
    _report(records, count, gen.provider.max_tokens, quiet)


def _run_graph_file(graph_file, fmt, provider, model, max_tokens, temperature,
                    no_validate, batch_size, quiet, locale):
    """Generate related entities from a YAML/JSON graph file."""
    try:
        with open(graph_file) as f:
            content = f.read()
        if graph_file.endswith(".json"):
            graph = json.loads(content)
        else:
            graph = yaml.safe_load(content)
    except (OSError, ValueError, yaml.YAMLError) as e:
        raise click.ClickException(f"Failed to load graph file: {e}")

    if not isinstance(graph, dict):
        raise click.ClickException("Graph file must contain a YAML/JSON object (mapping).")

    # Apply CLI batch_size as default for nodes that don't specify their own
    for node in graph.values():
        if isinstance(node, dict):
            node.setdefault("batch_size", batch_size)

    gen = _make_generator(provider, model, max_tokens, temperature, locale)

    # Auto-adjust max_tokens for each node
    for node in graph.values():
        if not isinstance(node, dict):
            continue
        context_name = node.get("context")
        if not context_name:
            continue
        try:
            node_schema = get_context_schema(context_name)
        except ValueError:
            continue
        node_batch = node.get("batch_size", batch_size)
        _adjust_max_tokens(gen, node_schema, node_batch, quiet, user_set=max_tokens is not None)

    entity_names = list(graph.keys())
    if not quiet:
        click.echo(
            click.style(
                f"Generating related entities: {', '.join(entity_names)} "
                f"[{gen.config.provider}/{gen.config.model}]",
                fg="cyan",
            ),
            err=True,
        )

    with _Spinner("Starting…", silent=quiet) as spinner:
        try:
            result = gen.generate_with_relationships(
                graph,
                validate=not no_validate,
                progress_callback=spinner.update if not quiet else None,
            )
        except ValueError as e:
            raise click.ClickException(str(e))
        except Exception as e:
            raise click.ClickException(f"Generation failed: {e}")

    if fmt == "json":
        click.echo(json.dumps(result, indent=2))
    elif fmt == "jsonl":
        for entity_name, records in result.items():
            click.echo(json.dumps({"entity": entity_name, "records": records}))
    elif fmt == "yaml":
        click.echo(yaml.dump(result, allow_unicode=True, sort_keys=False), nl=False)
    elif fmt == "csv":
        for entity_name, records in result.items():
            click.echo(f"# entity: {entity_name}")
            click.echo(_records_to_csv(records), nl=False)
    elif fmt == "sql":
        for entity_name, records in result.items():
            click.echo(_records_to_sql(records, entity_name), nl=False)

    if not quiet:
        total = sum(len(recs) for recs in result.values())
        per_entity = ", ".join(f"{k}: {len(v)}" for k, v in result.items())
        click.echo(
            click.style(f"Generated {total} total records ({per_entity})", fg="green"),
            err=True,
        )


def _run_schema_file(gen, schema_file, count, fmt, no_validate, quiet, max_tokens_user_set, table="records"):
    """Generate data from a JSON/YAML schema file using generate_from_model."""
    try:
        with open(schema_file) as f:
            if schema_file.endswith((".yaml", ".yml")):
                schema_dict = yaml.safe_load(f)
            else:
                schema_dict = json.load(f)
    except (OSError, ValueError) as e:
        raise click.ClickException(f"--schema-file: {e}")

    try:
        schema = model_to_context_schema(schema_dict)
    except (TypeError, ValueError) as e:
        raise click.ClickException(f"--schema-file: {e}")

    _adjust_max_tokens(gen, schema, count, quiet, user_set=max_tokens_user_set is not None)

    label = f"Generating {count} records from schema ({gen.config.provider}/{gen.config.model})"
    try:
        with _Spinner(label, silent=quiet):
            records = gen.generate_from_model(schema_dict, count=count, validate=not no_validate)
    except (ValueError, TypeError) as e:
        raise click.ClickException(str(e))
    except RuntimeError as e:
        raise click.ClickException(f"API error: {e}")

    _output_records(records, fmt, table)
    _report(records, count, gen.provider.max_tokens, quiet)


def _output_records(records: List[Dict[str, Any]], fmt: str, table: str = "records") -> None:
    """Write records to stdout in the requested format."""
    if fmt == "json":
        click.echo(json.dumps(records, indent=2))
    elif fmt == "jsonl":
        for record in records:
            click.echo(json.dumps(record))
    elif fmt == "yaml":
        click.echo(yaml.dump(records, allow_unicode=True, sort_keys=False), nl=False)
    elif fmt == "csv":
        click.echo(_records_to_csv(records), nl=False)
    elif fmt == "sql":
        click.echo(_records_to_sql(records, table), nl=False)


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


def _run_streaming(gen, context, count, batch_size, fmt, no_validate, quiet, table="records"):
    """Generate in batches, streaming records to stdout as each batch completes."""
    total_batches = math.ceil(count / batch_size)
    label = f"Generating {count} {context} records ({gen.config.provider}/{gen.config.model})"
    all_records: List[Dict[str, Any]] = []
    done = 0
    jsonl_pretty = fmt == "jsonl" and sys.stdout.isatty()
    try:
        with _Spinner(label, silent=quiet) as spinner:
            for i, batch in enumerate(
                gen.generate_batched(context, count, batch_size, validate=not no_validate), 1
            ):
                done += len(batch)
                all_records.extend(batch)
                with spinner.hidden():
                    if fmt == "jsonl":
                        for record in batch:
                            if jsonl_pretty:
                                click.echo(json.dumps(record, indent=2))
                                click.echo("")
                            else:
                                click.echo(json.dumps(record))
                    elif fmt == "yaml":
                        click.echo(yaml.dump(batch, allow_unicode=True, sort_keys=False), nl=False)
                if total_batches > 1:
                    spinner.update(f"Batch {i}/{total_batches} done ({done}/{count} records)")
            with spinner.hidden():
                if fmt == "json":
                    click.echo(json.dumps(all_records, indent=2))
                elif fmt == "csv":
                    click.echo(_records_to_csv(all_records), nl=False)
                elif fmt == "sql":
                    click.echo(_records_to_sql(all_records, table), nl=False)
    except ValueError as e:
        raise click.ClickException(str(e))
    except RuntimeError as e:
        raise click.ClickException(f"API error: {e}")
    return all_records


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



@cli.command("list-contexts")
@click.option("--category", default=None, help="Filter by category.")
@_context_file_option
def list_contexts_cmd(category, context_files):
    """List all available data contexts."""
    _load_context_files(context_files)

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
@_context_file_option
def show_context(context, context_files):
    """Show details of a specific context."""
    _load_context_files(context_files)

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


@cli.command("list-models")
@click.option("--provider", default=None, help="AI provider (overrides AI_PROVIDER env var).")
def list_models_cmd(provider):
    """List models available in the configured Ollama instance."""
    try:
        gen = DataGenerator(provider=provider)
    except (ValueError, ImportError) as e:
        raise click.ClickException(str(e))

    if not isinstance(gen.provider, OllamaProvider):
        raise click.ClickException(
            f"list-models is only supported for Ollama "
            f"(current provider: {gen.config.provider}). "
            f"Use --provider ollama or set AI_PROVIDER=ollama."
        )

    try:
        models = gen.provider.list_models()
    except RuntimeError as e:
        raise click.ClickException(str(e))

    if not models:
        click.echo("No models found. Run: ollama pull <model>")
        return

    click.echo(click.style(f"{'Model':<40}", fg="cyan", bold=True))
    click.echo("-" * 40)
    for name in models:
        click.echo(name)


class _Spinner:
    """Animated spinner context manager (stderr). Falls back to static text on non-TTY."""

    FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    INTERVAL = 0.08

    def __init__(self, message: str, silent: bool = False):
        self._message = message
        self._silent = silent
        self._start: Optional[float] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._is_tty = sys.stderr.isatty()

    def __enter__(self):
        if not self._silent:
            self._start = time.monotonic()
            if self._is_tty:
                self._thread = threading.Thread(target=self._spin, daemon=True)
                self._thread.start()
            else:
                sys.stderr.write(f"  {self._message}...\n")
                sys.stderr.flush()
        return self

    def _spin(self) -> None:
        for frame in itertools.cycle(self.FRAMES):
            if self._stop.is_set():
                break
            with self._lock:
                if not self._stop.is_set():
                    sys.stderr.write(f"\r\033[K  {frame} {self._message}")
                    sys.stderr.flush()
            self._stop.wait(self.INTERVAL)

    def update(self, message: str) -> None:
        """Update the status message (no-op when silent)."""
        if self._silent:
            return
        with self._lock:
            self._message = message
        if not self._is_tty:
            sys.stderr.write(f"  {message}\n")
            sys.stderr.flush()

    @contextlib.contextmanager
    def hidden(self):
        """Temporarily clear the spinner line so stdout records are not overwritten."""
        if self._silent or not self._is_tty or self._thread is None:
            yield
            return
        self._lock.acquire()
        try:
            sys.stderr.write("\r\033[K")
            sys.stderr.flush()
            yield
        finally:
            self._lock.release()

    def __exit__(self, *args):
        if self._silent or self._start is None:
            return
        elapsed = time.monotonic() - self._start
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
            sys.stderr.write(f"\r\033[K  Done ({elapsed:.1f}s)\n")
        else:
            sys.stderr.write(f"  Done ({elapsed:.1f}s)\n")
        sys.stderr.flush()



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


def _records_to_csv(records: List[Dict[str, Any]]) -> str:
    """Convert records to CSV text (with header). Nested dicts are dot-flattened."""
    if not records:
        return ""
    flat = [_flatten_dict(r) for r in records]
    fieldnames = list(dict.fromkeys(k for row in flat for k in row))
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore", restval="")
    writer.writeheader()
    writer.writerows(flat)
    return buf.getvalue()


def _records_to_sql(records: List[Dict[str, Any]], table: str = "records") -> str:
    """Convert records to SQL DDL + INSERT statements. Nested dicts are dot-flattened."""
    if not records:
        return ""
    flat = [_flatten_dict(r) for r in records]
    fieldnames = list(dict.fromkeys(k for row in flat for k in row))

    def _col_name(name: str) -> str:
        sanitized = "".join(c if c.isalnum() or c == "_" else "_" for c in name)
        return f'"{sanitized}"'

    def _infer_type(col: str) -> str:
        vals = [row[col] for row in flat if col in row and row[col] is not None]
        if not vals:
            return "TEXT"
        if all(isinstance(v, bool) for v in vals):
            return "INTEGER"
        if all(isinstance(v, int) and not isinstance(v, bool) for v in vals):
            return "INTEGER"
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
            return "REAL"
        return "TEXT"

    def _sql_val(v: Any) -> str:
        if v is None:
            return "NULL"
        if isinstance(v, bool):
            return "1" if v else "0"
        if isinstance(v, (int, float)):
            return str(v)
        return "'" + str(v).replace("'", "''") + "'"

    col_defs = ",\n  ".join(f"{_col_name(f)} {_infer_type(f)}" for f in fieldnames)
    lines = [f"CREATE TABLE IF NOT EXISTS {table} (\n  {col_defs}\n);", ""]
    for row in flat:
        vals = ", ".join(_sql_val(row.get(f)) for f in fieldnames)
        lines.append(f"INSERT INTO {table} VALUES ({vals});")
    return "\n".join(lines) + "\n"
