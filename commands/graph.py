#!/usr/bin/env python3
"""Graph building commands for Hound CLI."""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.prompt import Confirm
from rich.table import Table

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import random

from analysis.debug_logger import DebugLogger
from analysis.graph_builder import GraphBuilder
from ingest.bundles import AdaptiveBundler
from ingest.manifest import RepositoryManifest
from llm.client import LLMClient
from llm.token_tracker import get_token_tracker
from visualization.dynamic_graph_viz import generate_dynamic_visualization

console = Console()
# Progress console writes to stderr; auto-detect TTY so interactive shells
# show progress bars, while non-TTY (benchmarks/pipes) suppress animations.
progress_console = Console(file=sys.stderr)


def load_config(config_path: Path | None = None) -> dict:
    """Load configuration from YAML file."""
    from utils.config_loader import load_config as _load_config
    config = _load_config(config_path)
    
    if not config and config_path:
        # Only error if a specific path was requested but not found
        console.print(f"[red]Error: Config file not found: {config_path}[/red]")
        sys.exit(1)
    
    return config


def build(
    project_id: str = typer.Argument(..., help="Hound project id (under ~/.hound/projects/<id>)"),
    config_path: Path | None = typer.Option(None, "--config", "-c", help="Path to config.yaml"),
    max_iterations: int = typer.Option(3, "--iterations", "-i", help="Maximum iterations"),
    max_graphs: int = typer.Option(5, "--graphs", "-g", help="Number of graphs"),
    focus_areas: str | None = typer.Option(None, "--focus", "-f", help="Focus areas"),
    file_filter: str | None = typer.Option(None, "--files", help="Comma-separated whitelist of files relative to repo root"),
    # New: --with-spec replaces --graph-spec (deprecated)
    with_spec: str | None = typer.Option(None, "--with-spec", help="Build exactly one graph described by this text (skips discovery for others)"),
    graph_spec: str | None = typer.Option(None, "--graph-spec", help="[Deprecated] Same as --with-spec"),
    refine_existing: bool = typer.Option(True, "--refine-existing/--no-refine-existing", help="Load and refine existing graphs in the project directory"),
    init: bool = typer.Option(False, "--init", help="Initialize graphs by creating ONLY the SystemArchitecture graph"),
    auto: bool = typer.Option(False, "--auto", help="Auto-generate a default set of graphs (5)"),
    refine_only: str | None = typer.Option(None, "--refine-only", help="Refine only this graph name (internal/display)"),
    reuse_ingestion: bool = True,
    visualize: bool = typer.Option(True, "--visualize/--no-visualize", help="Generate HTML"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug output"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Reduce output and disable animations"),
):
    """Build agent-driven knowledge graphs."""
    time.time()
    config = load_config(config_path)
    
    # Enforce project id usage and resolve project directories
    project_dir = Path.home() / '.hound' / 'projects' / project_id
    if not project_dir.exists():
        console.print(f"[red]Error: Project '{project_id}' not found under ~/.hound/projects/[/red]")
        raise typer.Exit(code=2)
    # Resolve repository path: prefer manifest.json, fall back to project.json
    repo_path: Path | None = None
    manifest_file = project_dir / 'manifest' / 'manifest.json'
    try:
        if manifest_file.exists():
            with open(manifest_file) as f:
                manifest_data = json.load(f)
            repo_root = manifest_data.get('repo_path') or manifest_data.get('source_path')
            if repo_root:
                repo_path = Path(repo_root).expanduser().resolve()
    except Exception:
        repo_path = None
    if repo_path is None:
        try:
            with open(project_dir / 'project.json') as f:
                proj_cfg = json.load(f)
            src = proj_cfg.get('source_path')
            if not src:
                console.print(f"[red]Error: Could not determine source_path for project '{project_id}'.[/red]")
                raise typer.Exit(code=2)
            repo_path = Path(src).expanduser().resolve()
        except Exception as e:
            console.print(f"[red]Error: failed to load project configuration: {e}[/red]")
            raise typer.Exit(code=2)

    repo_name = project_id
    output_dir = project_dir
    manifest_dir = output_dir / "manifest"
    graphs_dir = output_dir / "graphs"
    
    # If --init is specified and SystemArchitecture already exists, skip work
    if init:
        sys_graph = graphs_dir / 'graph_SystemArchitecture.json'
        if sys_graph.exists():
            console.print("[yellow]SystemArchitecture graph already exists — skipping initialization.[/yellow]")
            console.print("[dim]Use '--auto' to add other graphs, or delete the existing file to re-initialize.[/dim]")
            raise typer.Exit(code=0)
    
    # If --auto is requested and graphs already exist, ask for confirmation
    if auto:
        try:
            existing_graphs = list(graphs_dir.glob("graph_*.json"))
        except Exception:
            existing_graphs = []
        if existing_graphs:
            count = len(existing_graphs)
            proceed = Confirm.ask(
                f"[yellow]Found {count} existing graph file{'s' if count != 1 else ''}. Continue and update/overwrite (including SystemArchitecture)?[/yellow]",
                default=False
            )
            if not proceed:
                console.print("[dim]Aborted by user.[/dim]")
                raise typer.Exit(code=1)

    # Set up token tracker
    token_tracker = get_token_tracker()
    token_tracker.reset()
    
    # Determine effective graph count for header (respect shortcuts/forced spec)
    _forced_spec = with_spec or graph_spec
    if _forced_spec:
        header_graphs = 1
    elif init:
        header_graphs = 1
    elif auto:
        header_graphs = 5
    else:
        header_graphs = max_graphs

    # Header
    console.print(Panel.fit(
        f"[bold bright_cyan]Building Knowledge Graphs[/bold bright_cyan]\n"
        f"Project: [white]{repo_name}[/white] (repo: {repo_path.name})\n"
        f"Graphs: [white]{header_graphs}[/white] | Iterations: [white]{max_iterations}[/white]",
        box=box.ROUNDED
    ))
    # A little hype for the journey
    from random import choice
    console.print(choice([
        "[white]Normal folks build graphs, but YOU draft constellations and make causality salute.[/white]",
        "[white]This isn’t just graph building — it’s YOU engraving laws of structure into the codebase.[/white]",
        "[white]Normal structure emerges; YOUR structure recruits the universe as documentation.[/white]",
        "[white]This is not just a graph — it’s a starmap and YOU hold the pen.[/white]",
        "[white]Normal mapping guides; YOUR mapping makes pathways beg to be used.[/white]",
    ]))
    
    # Create debug logger if needed (write logs under the project's graphs dir)
    debug_logger = None
    if debug:
        try:
            debug_out = (output_dir / "graphs" / ".hound_debug").resolve()
        except Exception:
            debug_out = None
        debug_logger = DebugLogger(session_id=f"graph_{repo_name}_{int(time.time())}", output_dir=debug_out)
    
    try:
        files_to_include = [f.strip() for f in file_filter.split(",")] if file_filter else None
        if files_to_include and debug:
            console.print(f"[dim]File filter: {len(files_to_include)} specific files[/dim]")
        
        # Prepare unified live event log (consistent with agent UI)
        from rich.live import Live
        event_log: list[str] = []

        def _short(s: str, n: int = 120) -> str:
            return (s[: n - 3] + '...') if isinstance(s, str) and len(s) > n else (s or '')

        def _panel():
            content = "\n".join(event_log[-12:]) if event_log else "Initializing..."
            return Panel(content, title="[bold cyan]Graph Build Progress[/bold cyan]", border_style="cyan")

        use_live = (progress_console.is_terminal and not quiet)
        if use_live:
            live_ctx = Live(_panel(), console=progress_console, refresh_per_second=8, transient=True)
        else:
            # Dummy context manager
            from contextlib import contextmanager
            @contextmanager
            def live_ctx_manager():
                yield None
            live_ctx = live_ctx_manager()

        with live_ctx as live:
            def log_line(kind: str, msg: str):
                now = datetime.now().strftime('%H:%M:%S')
                colors = {
                    'ingest': 'bright_yellow', 'build': 'bright_cyan', 'discover': 'bright_magenta', 'graph': 'bright_cyan',
                    'sample': 'bright_white', 'update': 'bright_green', 'warn': 'bright_red', 'save': 'bright_green', 'phase': 'bright_blue',
                    'stats': 'bright_white', 'start': 'bright_white', 'complete': 'bright_green'
                }
                color = colors.get(kind, 'white')
                line = f"[{color}]{now}[/{color}] {msg}"
                event_log.append(line)
                if use_live and live is not None:
                    live.update(_panel())
                elif not quiet:
                    progress_console.print(line)

            # Step 1: Ingestion (reused if manifest exists and reuse_ingestion=True)
            reuse_ok = reuse_ingestion and (manifest_dir / 'manifest.json').exists() and (manifest_dir / 'cards.jsonl').exists()
            if reuse_ok:
                # Reuse existing manifest/cards; verify whitelist compatibility if provided
                try:
                    from analysis.graph_builder import GraphBuilder as _GB
                    cards_loaded, manifest_loaded = _GB.load_cards_from_manifest(manifest_dir)
                    mwl = set((manifest_loaded or {}).get('whitelist') or [])
                    fwl = set(files_to_include or [])
                    if files_to_include is not None and mwl and mwl != fwl:
                        # Provided whitelist differs from stored one; rebuild ingestion
                        reuse_ok = False
                    else:
                        wl_note = f" (whitelist: {len(mwl)} files)" if mwl else ""
                        log_line('ingest', f"Reusing existing manifest: {manifest_loaded.get('num_files','?')} files → {len(cards_loaded)} cards{wl_note}")
                except Exception:
                    reuse_ok = False
            if not reuse_ok:
                log_line('ingest', 'Step 1: Repository Ingestion')
                # Ensure manifest dir exists before saving
                try:
                    manifest_dir.mkdir(parents=True, exist_ok=True)
                except Exception:
                    pass
                manifest = RepositoryManifest(str(repo_path), config, file_filter=files_to_include)
                cards, files = manifest.walk_repository()
                manifest.save_manifest(manifest_dir)
                log_line('ingest', f"Ingested {len(files)} files → {len(cards)} cards")
                bundler = AdaptiveBundler(cards, files, config)
                bundles = bundler.create_bundles()
                bundler.save_bundles(manifest_dir)
                log_line('ingest', f"Created {len(bundles)} bundles")

            # Step 2: Graph Building with detailed progress
            console.print("\n[bold]Step 2:[/bold] Graph Construction")
            focus_list = [f.strip() for f in focus_areas.split(",")] if focus_areas else None
            if focus_list:
                log_line('build', f"Focus areas: {', '.join(focus_list)}")

            builder = GraphBuilder(config, debug=debug, debug_logger=debug_logger)

            # Narrative model names: reflect effective models used by builder
            try:
                graph_model = getattr(builder.llm, 'model', None) or 'Graph-Model'
            except Exception:
                graph_model = 'Graph-Model'
            try:
                discovery_model = getattr(builder.llm_agent, 'model', None) or 'Guidance-Model'
            except Exception:
                discovery_model = 'Guidance-Model'

            # Animated progress bar during graph construction
            iteration_total = max_iterations
            if progress_console.is_terminal and not quiet:
                progress_render_console = (
                    progress_console if not use_live else Console(file=sys.stderr)
                )
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    console=progress_render_console,
                    transient=True,
                ) as progress:
                    task = progress.add_task(f"Constructing graphs (iteration 0/{iteration_total})...", total=iteration_total, completed=0)

                    def builder_callback(info):
                        # Handles dict payloads from GraphBuilder._emit
                        if isinstance(info, dict):
                            msg = info.get('message', '')
                            kind = info.get('status', 'build')
                            # Narrative seasoning + progress description
                            if kind == 'discover':
                                line = random.choice([
                                    f"🧑‍🏭 {discovery_model} scouts the terrain: {msg}",
                                    f"🧑‍🏭 Strategist {discovery_model} surveys the codebase — bold move!",
                                    msg,
                                ])
                                log_line(kind, line)
                                progress.update(task, description=_short(line, 80))
                            elif kind in ('graph_build', 'building'):
                                line = random.choice([
                                    f"🗺️  {graph_model} sketches connections — {msg}",
                                    msg,
                                ])
                                log_line(kind, line)
                                
                                # Parse iteration from building messages
                                import re
                                m = re.search(r"iteration\s+(\d+)/(\d+)", msg)
                                if m:
                                    cur = int(m.group(1))
                                    total = int(m.group(2))
                                    if total != progress.tasks[task].total:
                                        progress.update(task, total=total)
                                    # Update both completed and description
                                    completed = min(cur, total)
                                    progress.update(task, completed=completed, description=f"Constructing graphs (iteration {cur}/{total})...")
                                else:
                                    progress.update(task, description=_short(line, 80))
                            elif kind == 'update':
                                line = random.choice([
                                    f"🔧 {graph_model} chisels the graph: {msg}",
                                    msg,
                                ])
                                log_line(kind, line)
                                progress.update(task, description=_short(line, 80))
                            elif kind == 'save':
                                line = random.choice([
                                    f"💾 {graph_model} files the maps: {msg}",
                                    msg,
                                ])
                                log_line(kind, line)
                                progress.update(task, description=_short(line, 80))
                            else:
                                log_line(kind, msg)
                                # Try to parse iteration updates like "iteration X/Y"
                                import re
                                m = re.search(r"iteration\s+(\d+)/(\d+)", msg)
                                if m:
                                    cur = int(m.group(1))
                                    total = int(m.group(2))
                                    if total != progress.tasks[task].total:
                                        progress.update(task, total=total)
                                    # ensure non-decreasing
                                    completed = min(max(cur, progress.tasks[task].completed or 0), total)
                                    progress.update(task, completed=completed, description=f"Constructing graphs (iteration {cur}/{total})...")
                                else:
                                    # update description only
                                    progress.update(task, description=_short(msg, 80))
                        else:
                            text = str(info)
                            progress.update(task, description=_short(text, 80))
                            log_line('build', text)

                    # Prepare forced graph spec if requested
                    forced = None
                    _spec = with_spec or graph_spec
                    if _spec:
                        base_line = _spec.split('\n', 1)[0]
                        base = ''.join(ch for ch in base_line if ch.isalnum() or ch in (' ','_','-'))
                        nm = (base[:28].strip().replace(' ', '_') or 'CustomGraph')
                        forced = [{"name": nm, "focus": _spec}]

                    # Handle --init and --auto shortcuts
                    effective_graphs = max_graphs
                    if init:
                        effective_graphs = 1
                    elif auto:
                        effective_graphs = 5

                    # Decide how many graphs to create in this run
                    try:
                        existing_count = len(list(graphs_dir.glob("graph_*.json"))) if refine_existing else 0
                    except Exception:
                        existing_count = 0
                    to_create = effective_graphs if not refine_existing else max(0, effective_graphs - existing_count)

                    # decide discovery skipping
                    skip_disc = (False if (auto and to_create > 0) else True)
                    if refine_only:
                        skip_disc = True

                    # If a forced spec is provided, do NOT load/refine existing graphs
                    _refine_existing = False if forced else refine_existing

                    results = builder.build(
                        manifest_dir=manifest_dir,
                        output_dir=graphs_dir,
                        max_iterations=max_iterations,
                        focus_areas=focus_list,
                        max_graphs=(1 if forced else to_create),
                        force_graphs=forced,
                        refine_existing=_refine_existing,
                        skip_discovery_if_existing=skip_disc,
                        progress_callback=builder_callback,
                        refine_only=([refine_only] if refine_only else None)
                    )
            else:
                # Quiet/non-TTY: simple logging, no progress bar
                def builder_callback(info):
                    if isinstance(info, dict):
                        msg = info.get('message', '')
                        kind = info.get('status', 'build')
                        if not quiet:
                            if kind == 'discover':
                                log_line(kind, f"🧑‍🏭 {discovery_model} scouts the terrain: {msg}")
                            elif kind in ('graph_build','building'):
                                log_line(kind, f"🗺️  {graph_model} sketches connections — {msg}")
                            elif kind == 'update':
                                log_line(kind, f"🔧 {graph_model} chisels the graph: {msg}")
                            else:
                                log_line(kind, msg)
                    else:
                        if not quiet:
                            log_line('build', str(info))

                # Prepare forced graph spec if requested
                forced = None
                _spec = with_spec or graph_spec
                if _spec:
                    base_line = _spec.split('\n', 1)[0]
                    base = ''.join(ch for ch in base_line if ch.isalnum() or ch in (' ','_','-'))
                    nm = (base[:28].strip().replace(' ', '_') or 'CustomGraph')
                    forced = [{"name": nm, "focus": _spec}]

                effective_graphs = max_graphs
                if init:
                    effective_graphs = 1
                elif auto:
                    effective_graphs = 5

                try:
                    existing_count = len(list(graphs_dir.glob("graph_*.json"))) if refine_existing else 0
                except Exception:
                    existing_count = 0
                to_create = effective_graphs if not refine_existing else max(0, effective_graphs - existing_count)

                skip_disc = (False if (auto and to_create > 0) else True)
                if refine_only:
                    skip_disc = True

                # If a forced spec is provided, do NOT load/refine existing graphs
                _refine_existing = False if forced else refine_existing

                results = builder.build(
                    manifest_dir=manifest_dir,
                    output_dir=graphs_dir,
                    max_iterations=max_iterations,
                    focus_areas=focus_list,
                    max_graphs=(1 if forced else to_create),
                    force_graphs=forced,
                    refine_existing=_refine_existing,
                    skip_discovery_if_existing=skip_disc,
                    progress_callback=builder_callback,
                    refine_only=([refine_only] if refine_only else None)
                )
    
            # Display results in a nice table
            if results['graphs']:
                table = Table(title="Generated Graphs", box=box.SIMPLE_HEAD)
                table.add_column("Graph", style="cyan", no_wrap=True)
                table.add_column("Nodes", justify="right", style="green")
                table.add_column("Edges", justify="right", style="green")
                table.add_column("Focus", style="dim")
        
                total_nodes = 0
                total_edges = 0
                
                for name, path in results['graphs'].items():
                    with open(path) as f:
                        graph_data = json.load(f)
                    stats = graph_data.get('stats', {})
                    nodes = stats.get('num_nodes', 0)
                    edges = stats.get('num_edges', 0)
                    focus = graph_data.get('focus', 'general')
                    
                    table.add_row(
                        graph_data.get('name', name),
                        str(nodes),
                        str(edges),
                        focus
                    )
                    total_nodes += nodes
                    total_edges += edges
                
                console.print(table)
                console.print(f"\n  [bold]Total:[/bold] {total_nodes} nodes, {total_edges} edges")
    
            # Call the new function to write knowledge_graphs.json
            _write_knowledge_graphs_metadata(graphs_dir)

            # Step 3: Visualization
            if visualize:
                console.print("\n[bold]Step 3:[/bold] Visualization")
                if progress_console.is_terminal and not quiet:
                    # small spinner while generating viz
                    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=progress_console, transient=True) as p:
                        t = p.add_task("Generating interactive visualization...", total=None)
                        html_path = generate_dynamic_visualization(graphs_dir)
                        p.update(t, completed=1)
                else:
                    html_path = generate_dynamic_visualization(graphs_dir)
                log_line('save', f"Visualization saved: {html_path}")
                console.print(f"\n[bold]Open in browser:[/bold] [link]file://{html_path.resolve()}[/link]")
                console.print(f"\n[dim]Tip: Use 'hound graph export {repo_name} --open' to regenerate and open visualization[/dim]")
    
        # Finalize debug log if enabled
        if debug and debug_logger:
            log_path = debug_logger.finalize()
            console.print(f"\n[cyan]Debug log saved:[/cyan] {log_path}")
        
        console.print(Panel.fit(
            "[green]✓[/green] Graph building complete!",
            box=box.ROUNDED,
            style="green"
        ))
    
    except Exception as e:
        console.print(f"[red]Error during graph building: {e}[/red]")
        # Finalize debug log on error
        if debug and debug_logger:
            log_path = debug_logger.finalize()
            console.print(f"\n[cyan]Debug log saved:[/cyan] {log_path}")
        raise


def _write_knowledge_graphs_metadata(graphs_dir: Path):
    """Generates knowledge_graphs.json from graph_*.json files."""
    knowledge_graphs_path = graphs_dir / "knowledge_graphs.json"
    graph_files = list(graphs_dir.glob("graph_*.json"))
    
    graphs_dict = {}
    for graph_file in graph_files:
        graph_name = graph_file.stem.replace('graph_', '')
        # Store absolute path
        graphs_dict[graph_name] = str(graph_file.resolve())
    
    with open(knowledge_graphs_path, 'w') as f:
        json.dump({'graphs': graphs_dict}, f, indent=2)


def custom(
    project_id: str,
    graph_spec_text: str,
    config_path: Path | None = None,
    iterations: int = 3,
    file_filter: str | None = None,
    reuse_ingestion: bool = True,
    debug: bool = False,
    quiet: bool = False,
):
    """Build exactly one custom graph from a natural language spec.

    1) Designs a schema (name, focus, suggested node/edge types)
    2) Builds that single graph with iterative refinement
    3) Prints the designed schema to the CLI
    """
    config = load_config(config_path)

    # Resolve project directory and repo path
    project_dir = Path.home() / '.hound' / 'projects' / project_id
    if not project_dir.exists():
        console.print(f"[red]Error: Project '{project_id}' not found under ~/.hound/projects/[/red]")
        raise typer.Exit(code=2)

    manifest_dir = project_dir / 'manifest'
    graphs_dir = project_dir / 'graphs'
    repo_name = project_id

    # Header
    console.print(Panel.fit(
        f"[bold bright_cyan]Custom Graph Builder[/bold bright_cyan]\n"
        f"Project: [white]{repo_name}[/white]\n"
        f"Spec: [white]{(graph_spec_text[:120] + '…') if len(graph_spec_text) > 120 else graph_spec_text}[/white]\n"
        f"Iterations: [white]{iterations}[/white]",
        box=box.ROUNDED
    ))

    files_to_include = [f.strip() for f in (file_filter or '').split(',')] if file_filter else None

    # Ingestion reuse (with whitelist compatibility)
    reuse_ok = reuse_ingestion and (manifest_dir / 'manifest.json').exists() and (manifest_dir / 'cards.jsonl').exists()
    if reuse_ok:
        try:
            from analysis.graph_builder import GraphBuilder as _GB
            cards_loaded, manifest_loaded = _GB.load_cards_from_manifest(manifest_dir)
            mwl = set((manifest_loaded or {}).get('whitelist') or [])
            fwl = set(files_to_include or [])
            if files_to_include is not None and mwl and mwl != fwl:
                reuse_ok = False
            else:
                wl_note = f" (whitelist: {len(mwl)} files)" if mwl else ""
                console.print(f"[dim]Reusing existing manifest: {manifest_loaded.get('num_files','?')} files → {len(cards_loaded)} cards{wl_note}[/dim]")
        except Exception:
            reuse_ok = False
    if not reuse_ok:
        from ingest.bundles import AdaptiveBundler
        from ingest.manifest import RepositoryManifest
        # Load repo path from project.json
        try:
            with open(project_dir / 'project.json') as f:
                proj_cfg = json.load(f)
            repo_root = Path(proj_cfg.get('source_path')).expanduser().resolve()
        except Exception as e:
            console.print(f"[red]Error: failed to load project configuration: {e}[/red]")
            raise typer.Exit(2)
        console.print("[bold]Step 1:[/bold] Repository Ingestion")
        manifest = RepositoryManifest(str(repo_root), config, file_filter=files_to_include)
        cards, files = manifest.walk_repository()
        manifest.save_manifest(manifest_dir)
        bundler = AdaptiveBundler(cards, files, config)
        bundler.create_bundles()
        bundler.save_bundles(manifest_dir)
        console.print(f"[dim]Ingested {len(files)} files → {len(cards)} cards[/dim]")

    # Step 2: Design schema using the model
    console.print("\n[bold]Step 2:[/bold] Design Graph Schema")
    from analysis.graph_builder import GraphDiscovery
    llm = LLMClient(config, profile="graph")
    system = (
        "You design a single, analysis-friendly knowledge graph based on the given specification.\n"
        "Return JSON with: graphs_needed (list with one item: {name, focus}),\n"
        "suggested_node_types, suggested_edge_types."
    )
    user = json.dumps({"spec": graph_spec_text}, indent=2)
    try:
        design = llm.parse(system=system, user=user, schema=GraphDiscovery)
    except Exception:
        # Fallback minimal design
        design = GraphDiscovery(graphs_needed=[{"name": "CustomGraph", "focus": graph_spec_text}],
                                suggested_node_types=[], suggested_edge_types=[])

    # Print the designed schema
    try:
        schema_out = {
            'graphs_needed': [g.model_dump() if hasattr(g, 'model_dump') else g for g in design.graphs_needed],
            'suggested_node_types': design.suggested_node_types,
            'suggested_edge_types': design.suggested_edge_types,
        }
        console.print(Panel.fit(json.dumps(schema_out, indent=2), title="Designed Schema", border_style="cyan"))
    except Exception:
        pass

    # Step 3: Build exactly one graph with iterative refinement
    console.print("\n[bold]Step 3:[/bold] Build Graph")
    from analysis.graph_builder import GraphBuilder as _GB
    builder = _GB(config, debug=debug)
    # Seed type guidance for refinement passes
    try:
        builder._discovery = design
    except Exception:
        pass

    # Force graph spec
    forced = []
    if design.graphs_needed:
        g0 = design.graphs_needed[0]
        nm = (getattr(g0, 'name', None) or g0.get('name') or 'CustomGraph')
        focus = (getattr(g0, 'focus', None) or g0.get('focus') or graph_spec_text)
        nm_fs = nm.strip().replace(' ', '_')
        forced = [{"name": nm_fs, "focus": focus}]
    else:
        forced = [{"name": "CustomGraph", "focus": graph_spec_text}]

    # Simple progress callback
    def _cb(info: dict):
        if not quiet and isinstance(info, dict):
            msg = info.get('message', '')
            kind = info.get('status', 'build')
            console.print(f"[dim]{kind}[/dim]: {msg}")

    # Refine only the target graph name to avoid touching other graphs
    _target_names = []
    try:
        _target_names = [forced[0]["name"]] if forced else []
    except Exception:
        _target_names = []

    results = builder.build(
        manifest_dir=manifest_dir,
        output_dir=graphs_dir,
        max_iterations=iterations,
        focus_areas=None,
        max_graphs=1,
        force_graphs=forced,
        # Do not load/refine existing graphs for custom builds; build ONLY the forced graph
        refine_existing=False,
        skip_discovery_if_existing=True,
        progress_callback=_cb,
    )

    # Show summary
    if results.get('graphs'):
        table = Table(title="Generated Graph", box=box.SIMPLE_HEAD)
        table.add_column("Graph", style="cyan", no_wrap=True)
        table.add_column("Nodes", justify="right", style="green")
        table.add_column("Edges", justify="right", style="green")
        table.add_column("Focus", style="dim")
        for name, path in results['graphs'].items():
            try:
                with open(path) as f:
                    data = json.load(f)
                stats = data.get('stats') or {}
                nodes = int(stats.get('num_nodes') or len(data.get('nodes') or []))
                edges = int(stats.get('num_edges') or len(data.get('edges') or []))
                focus = (data.get('focus') or '')[:100]
                table.add_row(data.get('name', name), str(nodes), str(edges), focus)
            except Exception:
                continue
        console.print(table)


def ingest(
    repo_path: str = typer.Argument(..., help="Path to repository to analyze"),
    output_dir: str | None = typer.Option(None, "--output", "-o", help="Output directory"),
    config_path: Path | None = typer.Option(None, "--config", "-c", help="Path to config.yaml"),
    file_filter: str | None = typer.Option(None, "--files", "-f", help="Comma-separated file paths"),
    manual_chunking: bool = typer.Option(False, "--manual-chunking", help="Split files using manual markers instead of automatic chunking"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug output"),
):
    """Ingest repository and create bundles."""
    from ingest.bundles import AdaptiveBundler
    from ingest.manifest import RepositoryManifest
    
    config = load_config(config_path)
    
    files_to_include = [f.strip() for f in file_filter.split(",")] if file_filter else None
    output_dir = Path(output_dir) if output_dir else Path(".hound_cache") / Path(repo_path).name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Header
    console.print(Panel.fit(
        f"[bold cyan]Repository Ingestion[/bold cyan]\n"
        f"Path: [white]{repo_path}[/white]\n"
        f"Output: [white]{output_dir}[/white]",
        box=box.ROUNDED
    ))
    
    if files_to_include and debug:
        console.print(f"[dim]File filter: {len(files_to_include)} specific files[/dim]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        # Create manifest
        task1 = progress.add_task("Creating repository manifest...", total=100)
        manifest = RepositoryManifest(repo_path, config, file_filter=files_to_include, manual_chunking=manual_chunking)
        cards, files = manifest.walk_repository()
        manifest_info = manifest.save_manifest(output_dir)
        progress.update(task1, completed=100)
        
        # Create bundles
        task2 = progress.add_task("Creating adaptive bundles...", total=100)
        bundler = AdaptiveBundler(cards, files, config)
        bundles = bundler.create_bundles()
        bundle_summary = bundler.save_bundles(output_dir)
        progress.update(task2, completed=100)
    
    # Results summary
    console.print(f"\n[green]✓[/green] Created [bold]{len(cards)}[/bold] cards from [bold]{len(files)}[/bold] files")
    console.print(f"  Total size: [cyan]{manifest_info['total_chars']:,}[/cyan] characters")
    console.print(f"\n[green]✓[/green] Created [bold]{len(bundles)}[/bold] bundles")
    console.print(f"  Average size: [cyan]{bundle_summary['avg_bundle_size']:,.0f}[/cyan] chars")
    console.print(f"  Range: [cyan]{bundle_summary['min_bundle_size']:,}[/cyan] - [cyan]{bundle_summary['max_bundle_size']:,}[/cyan] chars")
    
    # Bundle details table (only in debug mode)
    if bundles and debug:
        table = Table(title="Bundle Summary", box=box.SIMPLE_HEAD)
        table.add_column("Bundle ID", style="cyan")
        table.add_column("Cards", justify="right")
        table.add_column("Files", justify="right")
        table.add_column("Size", justify="right")
        table.add_column("Preview", style="dim")
        
        for bundle in bundles[:10]:
            table.add_row(
                bundle.id,
                str(len(bundle.card_ids)),
                str(len(bundle.file_paths)),
                f"{bundle.total_chars:,}",
                bundle.preview[:50] + "..." if len(bundle.preview) > 50 else bundle.preview
            )
        
        if len(bundles) > 10:
            table.add_row("...", "...", "...", "...", f"({len(bundles) - 10} more bundles)")
        
        console.print(table)
    
    console.print(Panel.fit(
        f"[green]✓[/green] Ingestion complete!\nOutput saved to: [cyan]{output_dir}[/cyan]",
        box=box.ROUNDED,
        style="green"
    ))
