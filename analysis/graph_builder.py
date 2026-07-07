#!/usr/bin/env python3
"""Dynamic Knowledge Graph Builder with agent-driven schema discovery."""

import hashlib
import json
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pydantic import AliasChoices, BaseModel, Field

from llm.client import LLMClient
from llm.tokenization import count_tokens


@dataclass
class DynamicNode:
    """Flexible node representation for dynamic graphs."""
    id: str
    type: str  # Dynamic type decided by agent
    label: str
    properties: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    # Optional structured fields
    description: str | None = None
    confidence: float = 1.0
    source_refs: list[str] = field(default_factory=list)  # File paths or card IDs
    created_by: str = "agent"  # Which agent/pass created this
    iteration: int = 0  # When in the process it was created
    
    # Analysis fields - facts about the system (NOT security issues)
    observations: list[dict[str, Any]] = field(default_factory=list)  # Verified facts, invariants, behaviors
    assumptions: list[dict[str, Any]] = field(default_factory=list)  # Unverified assumptions, constraints


@dataclass
class DynamicEdge:
    """Flexible edge representation for dynamic graphs."""
    id: str
    type: str  # Dynamic type decided by agent
    source_id: str
    target_id: str
    properties: dict[str, Any] = field(default_factory=dict)
    
    # Optional structured fields
    label: str | None = None
    confidence: float = 1.0
    evidence: list[str] = field(default_factory=list)  # Supporting evidence
    created_by: str = "agent"
    iteration: int = 0


@dataclass
class KnowledgeGraph:
    """A single knowledge graph with a specific focus"""
    name: str
    focus: str  # What this graph focuses on (structure, security, data flow, etc.)
    nodes: dict[str, DynamicNode] = field(default_factory=dict)
    edges: dict[str, DynamicEdge] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def add_node(self, node: DynamicNode) -> bool:
        """Add a node. Returns True if node was actually added (not duplicate)."""
        if node.id in self.nodes:
            # Node already exists, merge source_refs if provided
            existing_node = self.nodes[node.id]
            if node.source_refs:
                existing_refs = set(existing_node.source_refs)
                existing_refs.update(node.source_refs)
                existing_node.source_refs = list(existing_refs)
            return False
        self.nodes[node.id] = node
        return True
    
    def add_edge(self, edge: DynamicEdge) -> bool:
        """Add an edge. Returns True if edge was actually added (not duplicate)."""
        # Check for duplicate edges (same source, target, and type)
        for existing_edge in self.edges.values():
            if (existing_edge.source_id == edge.source_id and 
                existing_edge.target_id == edge.target_id and 
                existing_edge.type == edge.type):
                # Edge already exists, merge evidence if provided
                if hasattr(edge, 'evidence') and edge.evidence:
                    if hasattr(existing_edge, 'evidence'):
                        # Merge evidence lists, avoiding duplicates
                        existing_evidence = set(existing_edge.evidence)
                        existing_evidence.update(edge.evidence)
                        existing_edge.evidence = list(existing_evidence)
                return False  # Skip adding duplicate edge
        
        # No duplicate found, add the edge
        self.edges[edge.id] = edge
        return True
    
    def get_neighbors(self, node_id: str, edge_type: str | None = None) -> list[str]:
        """Get neighboring nodes, optionally filtered by edge type"""
        neighbors = []
        for edge in self.edges.values():
            if edge.source_id == node_id:
                if edge_type is None or edge.type == edge_type:
                    neighbors.append(edge.target_id)
            elif edge.target_id == node_id:
                if edge_type is None or edge.type == edge_type:
                    neighbors.append(edge.source_id)
        return neighbors

    def to_dict(self) -> dict:
        return {
            "name": self.metadata.get("display_name", self.name),
            "internal_name": self.name,
            "focus": self.focus,
            "nodes": [asdict(n) for n in self.nodes.values()],
            "edges": [asdict(e) for e in self.edges.values()],
            "metadata": self.metadata,
            "stats": {
                "num_nodes": len(self.nodes),
                "num_edges": len(self.edges),
                "node_types": list(set(n.type for n in self.nodes.values())),
                "edge_types": list(set(e.type for e in self.edges.values())),
            },
        }



class GraphSpec(BaseModel):
    """Specification for a graph to build"""
    model_config = {"extra": "forbid"}
    name: str = Field(description="Graph name")
    focus: str = Field(description="What this graph focuses on")


class GraphDiscovery(BaseModel):
    """Initial discovery of what graphs to build"""
    model_config = {"extra": "forbid"}
    graphs_needed: list[GraphSpec] = Field(
        default_factory=list,
        description="List of graphs to create"
    )
    suggested_node_types: list[str] = Field(
        default_factory=list,
        description="Custom node types needed for this codebase"
    )
    suggested_edge_types: list[str] = Field(
        default_factory=list,
        description="Custom edge types needed for this codebase"
    )




class Observation(BaseModel):
    """An observation or verified fact about a node (NOT security issues - use hypotheses for those)"""
    model_config = {"extra": "forbid"}
    description: str = Field(description="Description of the observation")
    type: str = Field(default="general", description="Type: invariant, behavior, pattern, constraint, property")
    confidence: float = Field(1.0, description="Confidence in this observation")
    evidence: list[str] = Field(default_factory=list, description="Evidence supporting this observation")


class Assumption(BaseModel):
    """An unverified assumption about a node"""
    model_config = {"extra": "forbid"}
    description: str = Field(description="Description of the assumption")
    type: str = Field(default="general", description="Type: constraint, precondition, invariant, etc.")
    confidence: float = Field(0.5, description="Confidence in this assumption")
    needs_verification: bool = Field(True, description="Whether this needs verification")


class NodeSpec(BaseModel):
    """Node to add to the graph"""
    model_config = {"extra": "forbid"}
    id: str = Field(description="Unique node identifier (e.g., 'func_calculate', 'module_utils')")
    type: str = Field(description="Node type (e.g., function, class, module)")
    label: str = Field(description="Human-readable label for the node")
    refs: list[str] = Field(
        default_factory=list,
        description="List of card IDs where this node appears",
        validation_alias=AliasChoices("refs", "Refs", "evidence", "cards"),
    )


class EdgeSpec(BaseModel):
    """Edge to add - connects two nodes"""
    model_config = {"extra": "forbid"}
    type: str = Field(description="Edge type (e.g., calls, uses, depends_on)")
    src: str = Field(
        description="Source node ID (must be an existing node ID, NOT a card ID)",
        validation_alias=AliasChoices("src", "source", "from", "source_id"),
    )
    dst: str = Field(
        description="Target node ID (must be an existing node ID, NOT a card ID)",
        validation_alias=AliasChoices("dst", "target", "to", "target_id"),
    )
    refs: list[str] = Field(
        default_factory=list,
        description="Card IDs that evidence this edge",
        validation_alias=AliasChoices("refs", "Refs", "evidence", "cards"),
    )


class NodeUpdate(BaseModel):
    """Update for an existing node"""
    model_config = {"extra": "forbid"}
    id: str = Field(description="Node ID to update")
    description: str | None = Field(None, description="New description")
    properties: str | None = Field(None, description="JSON string of properties to update")
    
    # New observations/assumptions to add
    new_observations: list[Observation] = Field(default_factory=list, description="[LEAVE EMPTY during graph building - only for agent analysis phase]")
    new_assumptions: list[Assumption] = Field(default_factory=list, description="[LEAVE EMPTY during graph building - only for agent analysis phase]")


class GraphUpdate(BaseModel):
    """Incremental update to a knowledge graph"""
    model_config = {"extra": "forbid"}
    target_graph: str = Field(default="", description="Name of graph to update")
    
    new_nodes: list[NodeSpec] = Field(
        default_factory=list,
        description="New nodes to add - return empty list if no new nodes found"
    )
    new_edges: list[EdgeSpec] = Field(
        default_factory=list,
        description="New edges to add - return empty list if no new edges found"
    )
    
    node_updates: list[NodeUpdate] = Field(
        default_factory=list,
        description="Updates to existing nodes with new invariants/observations"
    )

    # Optional completion hints from the model (ignored by builder heuristics for now)
    is_complete: bool | None = Field(
        default=None,
        description="Whether the model believes this graph is complete"
    )
    completeness_reason: str | None = Field(
        default=None,
        description="Reason provided by the model about completeness state"
    )





class GraphBuilder:
    """
    Agent-driven dynamic knowledge graph builder.
    
    Key principles:
    1. Agent decides what's important to model
    2. Multiple specialized graphs for different concerns
    3. Iterative refinement based on discoveries
    4. Minimal pre-processing - just code cards
    """
    
    def __init__(self, config: dict, debug: bool = False, debug_logger=None):
        # Use a local copy of config and honor user-provided model settings.
        # We now use a single model profile ('graph') for both discovery and build.
        import copy as _copy
        cfg = _copy.deepcopy(config) if isinstance(config, dict) else {}
        self.config = cfg
        self.debug = debug
        self.debug_logger = debug_logger

        # Initialize LLM client for graph building (also used for discovery)
        self.llm = LLMClient(self.config, profile="graph", debug_logger=debug_logger)
        self.llm_agent = self.llm  # single-model setup
        if debug:
            graph_model = self.config.get("models", {}).get("graph", {}).get("model", "unknown")
            print(f"[*] Graph model: {graph_model} (used for discovery and build)")
        
        # Knowledge graphs storage
        self.graphs: dict[str, KnowledgeGraph] = {}
        
        # Card storage for later retrieval
        self.card_store: dict[str, dict] = {}
        
        # Iteration counter
        self.iteration = 0
        # External progress sink
        self._progress_callback = None
        
        # Repository root for extracting full content
        self._repo_root: Path | None = None

    def _emit(self, status: str, message: str, **kwargs):
        """Emit progress events to callback and optionally print when debug."""
        if self._progress_callback:
            payload = {"status": status, "message": message, "iteration": self.iteration}
            payload.update(kwargs)
            try:
                # Prefer dict signature
                self._progress_callback(payload)
            except TypeError:
                # Backward compatibility: (iteration, message)
                try:
                    self._progress_callback(self.iteration, message)
                except Exception:
                    pass
        if self.debug:
            print(f"[{status}] {message}")
    
    def build(
        self,
        manifest_dir: Path,
        output_dir: Path,
        max_iterations: int = 5,
        focus_areas: list[str] | None = None,
        max_graphs: int = 2,
        force_graphs: list[dict[str, str]] | None = None,
        refine_existing: bool = True,
        skip_discovery_if_existing: bool = True,
        progress_callback: Callable[[dict], None] | None = None,
        refine_only: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Main entry point for dynamic graph building.
        
        Args:
            manifest_dir: Directory with code manifest and cards
            output_dir: Where to save results
            max_iterations: Maximum refinement iterations
            focus_areas: Optional list of areas to focus on
            max_graphs: Maximum number of graphs to create
            force_graphs: Optional list of specific graphs to force (e.g., [{"name": "CallGraph", "focus": "function calls"}])
        """
        start_time = time.time()
        self._progress_callback = progress_callback
        # Track strict refine-only mode
        try:
            self._refine_only = bool(refine_only)
        except Exception:
            self._refine_only = False
        
        # Remember output dir for incremental saves
        try:
            self._output_dir = output_dir
        except Exception:
            pass

        # Load code cards with full content
        manifest, cards = self._load_manifest(manifest_dir)
        
        # Calculate statistics
        total_chars = sum(
            len(card.get("content", "")) if card.get("content") else
            (len(card.get("peek_head", "")) + len(card.get("peek_tail", "")))
            for card in cards)
        # Estimate lines from peek content
        total_lines = sum(
            card.get("peek_head", "").count('\n') + 
            card.get("peek_tail", "").count('\n') + 2
            for card in cards
        )
        
        self._emit("start", "Dynamic Graph Building")
        self._emit("stats", f"Files: {manifest['num_files']}")
        self._emit("stats", f"Cards: {len(cards)}")
        self._emit("stats", f"Total lines: {total_lines:,}")
        self._emit("stats", f"Total chars: {total_chars:,}")
        self._emit("stats", f"Max iterations: {max_iterations}")
        
        # Load existing graphs if present (refinement)
        if refine_existing:
            try:
                self._load_existing_graphs(output_dir)
                if self.graphs:
                    self._emit("note", f"Loaded {len(self.graphs)} existing graph(s) for refinement")
            except Exception:
                pass

        # If refining only a specific subset, filter loaded graphs
        if refine_existing and refine_only:
            # Special sentinel to indicate refine-all without filtering
            if len(refine_only) == 1 and str(refine_only[0]).strip().lower() == '__all__':
                pass  # keep all loaded graphs, but remain in refine-only mode
            else:
                targets_norm = {str(n or '').strip().replace(' ', '_').lower() for n in refine_only}
                if self.graphs:
                    filtered: dict[str, KnowledgeGraph] = {}
                    for name, g in self.graphs.items():
                        disp = (g.metadata or {}).get('display_name') or name
                        candidates = {
                            name.strip().replace(' ', '_').lower(),
                            disp.strip().replace(' ', '_').lower(),
                            disp.strip().lower(),
                        }
                        if targets_norm & candidates:
                            filtered[name] = g
                    if filtered:
                        self.graphs = filtered
                        self._emit('note', f"Refining only: {', '.join(filtered.keys())}")
                    else:
                        self._emit('warn', 'Refine target not found among existing graphs; no graphs will be refined.')

        # Phase 1: Discovery - Let agent decide what to build (unless refining existing only)
        do_discovery = True
        if skip_discovery_if_existing and refine_existing and self.graphs and not force_graphs:
            do_discovery = False
            self._emit("note", "Skipping discovery (existing graphs present)")
        # If refining only, never run discovery
        if refine_existing and refine_only:
            do_discovery = False

        if do_discovery:
            self._emit("phase", "Graph Discovery")
            # Run discovery; emit a stern warning if sampling was required inside discovery
            self._discover_graphs(manifest, cards, focus_areas, max_graphs, force_graphs)
        
        # Phase 2: Iterative Graph Building
        self._build_had_failures = False
        if self.debug:
            print("\n[Phase 2] Iterative Graph Building")
        for i in range(max_iterations):
            self.iteration = i
            if self.debug:
                print(f"  Iteration {i+1}/{max_iterations}")
            self._emit("building", f"Building graphs: iteration {i+1}/{max_iterations}")
            
            # Build/refine graphs with early stopping if complete
            had_updates, had_failures = self._build_iteration(cards)
            if had_failures:
                self._build_had_failures = True

            # Early exit only if nothing was updated AND nothing failed: a
            # failure-only pass must not be mistaken for completion.
            if not had_updates and not had_failures and i > 0:  # Allow at least 2 iterations
                self._emit("early_exit", f"All graphs complete after {i+1} iterations")
                if self.debug:
                    print(f"  Early exit: All graphs complete after {i+1} iterations")
                break
        
        # Phase 3: Save results
        self._emit("phase", "Saving Results")
        results = self._save_results(output_dir, manifest)
        if isinstance(results, dict):
            results["had_failures"] = bool(getattr(self, "_build_had_failures", False))
        if getattr(self, "_build_had_failures", False):
            self._emit("warn", "Some graph-building LLM calls failed; the graph may be INCOMPLETE. "
                               "Re-run with --debug for details, and consider `graph rm --all` then rebuild.")

        duration = time.time() - start_time
        self._emit("complete", f"Complete in {duration:.1f}s")
        
        return results
    
    def _discover_graphs(
        self,
        manifest: dict,
        cards: list[dict],
        focus_areas: list[str] | None = None,
        max_graphs: int = 2,
        force_graphs: list[dict[str, str]] | None = None
    ):
        """Let agent discover what graphs to build"""
        
        # If specific graphs are forced, use them directly
        if force_graphs:
            self._emit("discover", "Using forced graph specifications...")
            for graph_spec in force_graphs:
                name = graph_spec["name"].replace(' ', '_').replace('/', '_')
                focus = graph_spec["focus"]
                self.graphs[name] = KnowledgeGraph(
                    name=name,
                    focus=focus,
                    metadata={"created_at": time.time(), "display_name": graph_spec["name"]}
                )
                self._emit("graph", f"Created graph: {graph_spec['name']} (focus: {focus})")
                # Save initial empty graph immediately
                try:
                    self._save_graph(output_dir=self._output_dir, graph=self.graphs[name])
                    self._emit("save", f"Saved schema for {name}")
                except Exception:
                    pass
            return
        
        # Use adaptive sampling based on token count for discovery phase
        self._emit("discover", "Analyzing codebase for graph discovery...")
        # Sample cards to stay within the active model's context limits
        code_samples = self._sample_cards_for_discovery(cards)
        if len(code_samples) != len(cards):
            msg = (
                f"Discovery context limit reached — sampling active: using {len(code_samples)}/{len(cards)} cards. "
                f"Consider restricting input files with --files (whitelist), and/or increasing models.graph.max_context."
            )
            self._emit("warn", msg)
            self._emit("sample", msg)
        
        # Allow forcing specific graph type through focus_areas (backward compatibility)
        if focus_areas and "call_graph" in focus_areas:
            system_prompt = f"""Create {max_graphs} call graph(s) showing function/method calls.
For each graph, provide:
- name: A short name for the graph (e.g., "CallGraph")
- focus: What this graph focuses on (e.g., "function call relationships")"""
        else:
            system_prompt = f"""Design EXACTLY {max_graphs} graph{'s' if max_graphs > 1 else ''} for this codebase.

REQUIRED: The FIRST graph MUST be a high-level system/component/flow graph that shows:
- Major components, modules, or contracts
- How they relate and interact
- Data/control flow between them
- System boundaries and external interfaces

Name it "SystemArchitecture".

DIVERSITY REQUIREMENTS (for remaining graphs):
- Each additional graph MUST present a distinct analytical lens (do NOT create multiple call graphs).
- Prefer domain-relevant structures with meaningful, typed relationships (edges with types like grants, authorizes, mints, reads, writes, depends_on, initializes, upgrades, pauses, emits, computes, bounded_by, etc.).
- Choose graphs that maximize utility for security analysis and understanding, not just topology.

Creativity guidance:
- Be maximally creative and tailor graph types to THIS codebase (examples are inspiration, not constraints).
- Propose novel, domain-specific graph types when they would reveal important structure or risk.
- Avoid redundancy across graphs; minimize overlap and pick the most informative lenses.

Ideas for strong, analysis-friendly graphs (pick those that fit this codebase):
- AuthorizationMap: who grants/assumes/authorizes which roles/actions (edges: creates, grants, assumes, authorizes, guarded_by).
- PermissionChecks: coverage of access modifiers and require checks per function (edges: guarded_by, unchecked, requires_role).
- AssetFlow: mint/burn/transfer/deposit/withdraw across contracts and accounts (edges: mints, burns, transfers, deposits, withdraws).
- StateMutation: storage variables and the functions that read/write them (edges: written_by, read_by, derived_from).
- UpgradeLifecycle: deployment/initialization/upgrade relationships (edges: deploys, initializes, upgrades, migrates_from).
- ExternalDeps: external/oracle/library dependencies and trust boundaries (edges: reads_from, depends_on, trusts, verifies).
- Reentrancy/ExternalCalls: external call graph with entrypoints and reentrant paths (edges: calls_external, reentrant_path, invokes_untrusted).
- InvariantsMap: key invariants/assumptions and where they’re enforced (edges: enforced_by, broken_by, relies_on).
- MathAlgorithm: break down core formulas/AMM math into steps/variables (edges: computes, uses_param, normalizes, clamps).
- EventMap: which events are emitted by which functions and with what state (edges: emitted_by, indexes, correlates_with).
- TimeWindows/RateLimits: time-based gates and limits (edges: gates, bounded_by, cooldown).

For each graph, you MUST provide:
- name: A short name for the graph
- focus: What this graph focuses on (be specific)

Additionally, provide top-level guidance to help refinement:
- suggested_node_types: list of node types you plan to use across graphs (e.g., function, storage, role, token, invariant, event)
- suggested_edge_types: list of edge types you plan to use (e.g., calls, guarded_by, writes, reads, mints, authorizes)

IMPORTANT: Return EXACTLY {max_graphs} graph{'s' if max_graphs > 1 else ''}, no more, no less.
The FIRST must be the system/component/flow overview."""
        
        user_prompt = {
            "repository": manifest.get("repo_path", "unknown"),
            "num_files": manifest["num_files"],
            "code_samples": code_samples,
            "focus_areas": focus_areas or ["general analysis"],
            "instruction": "Determine what knowledge graphs to build and what custom types are needed"
        }
        
        # Estimate token counts for the request
        user_prompt_str = json.dumps(user_prompt, indent=2)
        # Use the graph model for token estimation in single-model mode
        _graph_model_name = (self.config.get("models", {}).get("graph", {}) or {}).get("model", "gpt-4o-mini")
        system_tokens = count_tokens(system_prompt, "openai", _graph_model_name)
        user_tokens = count_tokens(user_prompt_str, "openai", _graph_model_name)
        total_tokens = system_tokens + user_tokens
        
        # Log token counts
        self._emit("debug", f"Initial graph design prompt tokens: system={system_tokens:,}, user={user_tokens:,}, total={total_tokens:,}")
        
        # Check if we're approaching context limit (use graph model's context in single-model mode)
        graph_model_config = self.config.get("models", {}).get("graph", {})
        graph_max_context = graph_model_config.get("max_context", self.config.get("context", {}).get("max_tokens", 256000))
        if total_tokens > graph_max_context * 0.9:
            self._emit("warning", f"Token count ({total_tokens:,}) approaching context limit ({graph_max_context:,})")
        
        # Use agent model for discovery (better reasoning)
        discovery = self.llm_agent.parse(
            system=system_prompt,
            user=user_prompt_str,
            schema=GraphDiscovery
        )
        
        # Store discovery for later reference in prompts
        self._discovery = discovery
        
        # Create the suggested graphs (limited to max_graphs)
        graphs_to_create = discovery.graphs_needed[:max_graphs]
        if len(discovery.graphs_needed) > max_graphs:
            self._emit("discover", f"LLM suggested {len(discovery.graphs_needed)} graphs, limiting to {max_graphs}")
        
        for i, graph_spec in enumerate(graphs_to_create):
            raw_name = graph_spec.name if hasattr(graph_spec, 'name') else graph_spec.get("name", f"graph_{len(self.graphs)}")
            focus = graph_spec.focus if hasattr(graph_spec, 'focus') else graph_spec.get("focus", "general")
            
            # Force first graph to be SystemArchitecture
            if i == 0 and len(self.graphs) == 0:
                raw_name = "SystemArchitecture"
            
            # Sanitize name for file system (replace spaces with underscores)
            name = raw_name.replace(' ', '_').replace('/', '_')
            
            self.graphs[name] = KnowledgeGraph(
                
                name=name,
                focus=focus,
                metadata={"created_at": time.time(), "display_name": raw_name}
            )
            self._emit("graph", f"Created graph: {raw_name} (focus: {focus})")
            try:
                self._save_graph(output_dir=self._output_dir, graph=self.graphs[name])
                self._emit("save", f"Saved schema for {name}")
            except Exception:
                pass
        
        # Note custom types (agent can use these later)
        if discovery.suggested_node_types:
            self._emit("note", f"Custom node types: {', '.join(discovery.suggested_node_types)}")
        if discovery.suggested_edge_types:
            self._emit("note", f"Custom edge types: {', '.join(discovery.suggested_edge_types)}")
    
    def _build_iteration(self, cards: list[dict]) -> tuple[bool, bool]:
        """
        Single iteration to build/refine graphs.
        Returns (any_updates, had_failures): any_updates is True if any graph
        gained nodes/edges; had_failures is True if any per-graph update call
        failed, so the caller must not treat this pass as completion.
        """
        
        any_updates = False
        had_failures = False
        
        # Always try to use ALL cards for maximum context
        # The model needs to see the entire codebase to make good decisions
        for graph_name, graph in self.graphs.items():
            orphan_count = len(self._get_orphaned_nodes(graph))
            self._emit("graph_build", f"{graph_name}: {len(graph.nodes)}N/{len(graph.edges)}E, {orphan_count} orphans")
            
            # Try to use ALL cards if possible within token limits
            relevant_cards = self._sample_cards(cards)
            if len(relevant_cards) != len(cards):
                msg = (
                    f"Context limit reached — sampling active: using {len(relevant_cards)}/{len(cards)} cards. "
                    f"Consider restricting input files with --files (whitelist), and/or using a model with a larger context or increasing models.graph.max_context."
                )
                self._emit("warn", msg)
                self._emit("sample", msg)
            
            # Update the graph
            update = self._update_graph(graph, relevant_cards)
            
            if update:
                # Apply updates and track what was actually added (deduplication happens in add_node/add_edge)
                nodes_added, edges_added = self._apply_update(graph, update)
                
                new_orphan_count = len(self._get_orphaned_nodes(graph))
                if nodes_added > 0 or edges_added > 0:
                    self._emit("update", f"Added: {nodes_added} nodes, {edges_added} edges (orphans {orphan_count}->{new_orphan_count})")
                    any_updates = True
                else:
                    self._emit("update", f"No new nodes/edges added (duplicates filtered, orphans: {new_orphan_count})")
                    # If no new nodes or edges were added, the graph might be complete
                    if self.iteration > 0:  # Only after first iteration
                        self._emit("complete", f"Graph '{graph_name}' appears complete (no new nodes/edges found)")
            else:
                had_failures = True
        
        # Propagate had_failures so the caller does not treat an error-only pass as completion.
        if had_failures and self.debug:
            print("  One or more update calls failed this iteration; skipping early completion heuristics.")
        return any_updates, had_failures
    
    def _get_orphaned_nodes(self, graph: KnowledgeGraph) -> set:
        """Find nodes with no edges (neither incoming nor outgoing)"""
        connected_nodes = set()
        for edge in graph.edges.values():
            connected_nodes.add(edge.source_id)
            connected_nodes.add(edge.target_id)
        
        orphaned = set(graph.nodes.keys()) - connected_nodes
        return orphaned
    
    def _update_graph(self, graph: KnowledgeGraph, cards: list[dict]) -> GraphUpdate | None:
        """Update graph with new nodes and edges based on current state"""
        
        # Store cards for later retrieval - filter out redundant peek fields
        cards_with_ids = []
        for card in cards:
            card_id = card.get("id", f"card_{len(self.card_store)}_{len(cards_with_ids)}")
            self.card_store[card_id] = card
            # Create a filtered version without peek_head/peek_tail for the prompt
            card_with_id = {
                "id": card_id,
                "relpath": card.get("relpath", ""),
                "content": card.get("content", "")
            }
            # Only include other fields if they exist and are meaningful
            if card.get("type"):
                card_with_id["type"] = card["type"]
            if card.get("description"):
                card_with_id["description"] = card["description"]
            cards_with_ids.append(card_with_id)
        
        # Adaptive prompting based on graph state
        if self.iteration == 0:
            # Initial build - focus on discovering CONNECTED nodes
            # Include suggested types in prompt if available
            type_guidance = ""
            if hasattr(self, '_discovery') and self._discovery:
                if self._discovery.suggested_node_types:
                    type_guidance += f"\n\nRECOMMENDED NODE TYPES (use these!):\n{', '.join(self._discovery.suggested_node_types[:15])}"
                if self._discovery.suggested_edge_types:
                    type_guidance += f"\n\nRECOMMENDED EDGE TYPES (use these!):\n{', '.join(self._discovery.suggested_edge_types[:15])}"
            
            system_prompt = f"""Build {graph.focus} graph.{type_guidance}
FOCUS: COMPREHENSIVELY model ALL aspects of this graph's focus.
You MUST capture the COMPLETE structure - don't leave anything out!

IMPORTANT: This is ONLY structural discovery - do NOT add observations or assumptions.
Those will be added later during analysis.

CRITICAL: Only include nodes for code that EXISTS in this codebase's source files.
DO NOT create nodes for:
- External dependencies (OpenZeppelin, Chainlink, etc.)
- Standard library contracts or interfaces imported from outside
- Third-party libraries
Only reference external dependencies in edge relationships if needed.

Nodes: id (unique string), type, label, refs (array of card IDs that contain this node)
  - Prefer function-level and storage-level nodes; contract-level nodes are acceptable but should not crowd out finer nodes.
  - MUST be defined in the project's source files (not just imported/used)
Edges: type, src (source NODE ID), dst (target NODE ID), refs (array of card IDs evidencing this relationship)

CRITICAL: Check existing_edges list and DO NOT recreate edges that already exist!

CRITICAL - refs field:
- Each node MUST have a refs array containing the IDs of cards where this node appears
- Each edge SHOULD have a refs array with card IDs where this relationship is visible (call site, data flow, state mutation)
- Look at the code_samples - each has an "id" field like "card_0_0", "card_0_1", etc.
- Include these card IDs in refs for nodes/edges found in those cards
- Example: if you find a function in card_0_0 and card_0_3, refs should be ["card_0_0", "card_0_3"]

IMPORTANT: 
- Edge src/dst must reference node IDs you created, NOT card IDs!
- Every node should have at least one edge (incoming or outgoing)
- Prioritize connected components over isolated nodes
Target: 20-40 nodes minimum (more if needed for completeness). Every significant component should be represented.
Prioritize COMPLETENESS over simplicity - it's better to have too many nodes than to miss important parts."""
        else:
            # Refinement - strongly prioritize connecting existing nodes
            orphaned_nodes = self._get_orphaned_nodes(graph)
            orphan_count = len(orphaned_nodes)
            
            if orphan_count > 5:
                # Many orphaned nodes - focus on connecting them
                orphan_sample = list(orphaned_nodes)[:10]  # Show first 10
                focus_instruction = f"CRITICAL: {orphan_count} nodes have NO connections! Connect these orphans: {orphan_sample}\nEvery node should have at least one edge!"
            elif len(graph.edges) < len(graph.nodes) * 1.5:
                focus_instruction = f"PRIORITY: Find MORE EDGES! With {len(graph.nodes)} nodes, you should have at least {int(len(graph.nodes) * 1.5)} edges. Look for all relationships!"
            else:
                focus_instruction = "Continue adding nodes and edges to ensure COMPLETE coverage. Look for any missing components or relationships."
            
            # Include type guidance in refinement too
            type_guidance = ""
            if hasattr(self, '_discovery') and self._discovery:
                if self._discovery.suggested_node_types:
                    unused_node_types = [t for t in self._discovery.suggested_node_types if not any(n.type == t for n in graph.nodes.values())]
                    if unused_node_types:
                        type_guidance += f"\n\nUNUSED NODE TYPES (consider using): {', '.join(unused_node_types[:10])}"
                if self._discovery.suggested_edge_types:
                    unused_edge_types = [t for t in self._discovery.suggested_edge_types if not any(e.type == t for e in graph.edges.values())]
                    if unused_edge_types:
                        type_guidance += f"\n\nUNUSED EDGE TYPES (consider using): {', '.join(unused_edge_types[:10])}"
            
            # Stricter constraints in refine-only mode
            refine_constraints = "\nOnly add new nodes when strictly required and CONNECT them immediately to existing nodes (≤ 3)." 
            try:
                if getattr(self, '_refine_only', False):
                    refine_constraints = "\nDO NOT add new nodes; focus on EDGES and NODE UPDATES only."
            except Exception:
                pass

            system_prompt = f"""Refine {graph.focus}. Current: {len(graph.nodes)}N/{len(graph.edges)}E, {orphan_count} orphans.
{focus_instruction}

GOAL: Create a COMPREHENSIVE model that captures ALL aspects of {graph.focus}.
The graph should be COMPLETE - every important component, relationship, and interaction should be represented.{type_guidance}{refine_constraints}

IMPORTANT: This is ONLY structural discovery - do NOT add observations or assumptions.
Those will be added later during analysis.

CRITICAL: Only include nodes for code that EXISTS in this codebase's source files.
DO NOT create nodes for:
- External dependencies (OpenZeppelin, Chainlink, etc.)
- Standard library contracts or interfaces imported from outside
- Third-party libraries
Only reference external dependencies in edge relationships if needed.

DO NOT add new nodes unless they connect to existing ones (and ONLY if strictly necessary).
Nodes: If adding new nodes, include refs array with card IDs where they appear; prefer function/storage granularity.
  - MUST be defined in the project's source files (not just imported/used)
Edges: type, src (existing NODE id), dst (existing NODE id), refs (card IDs evidencing the relationship). 
IMPORTANT: Use existing node IDs! Check the existing_nodes list.
CRITICAL: Check existing_edges list and DO NOT recreate edges that already exist!
For any new nodes, include refs field with card IDs from code_samples.
Return empty lists only if graph is TRULY complete and comprehensive."""
        
        # Build user prompt with existing nodes AND edges for reference
        existing_nodes_list = []
        existing_edges_list = []
        
        if self.iteration > 0:
            # Provide existing nodes to help model make connections
            if len(graph.nodes) > 0:
                for node_id, node in graph.nodes.items():
                    existing_nodes_list.append({
                        "id": node_id,
                        "type": node.type,
                        "label": node.label
                    })
            
            # Provide existing edges to avoid duplicates
            if len(graph.edges) > 0:
                for edge_id, edge in graph.edges.items():
                    existing_edges_list.append({
                        "type": edge.type,
                        "src": edge.source_id,
                        "dst": edge.target_id
                    })
        
        user_prompt = {
            "graph_name": graph.name,
            "graph_focus": graph.focus,
            "existing_nodes": existing_nodes_list if existing_nodes_list else f"{len(graph.nodes)} nodes",
            "existing_edges": existing_edges_list if existing_edges_list else f"{len(graph.edges)} edges",
            "code_samples": cards_with_ids,
            "iteration": self.iteration,
            "instruction": "Update graph. ONLY add NEW nodes/edges that don't already exist. Check existing_edges to avoid duplicates. Use NODE IDs for edges. Include refs arrays for nodes AND edges with card IDs from code_samples that evidence them."
        }
        
        try:
            update = self.llm.parse(
                system=system_prompt,
                user=json.dumps(user_prompt, indent=2),
                schema=GraphUpdate
            )
            update.target_graph = graph.name
            return update
        except Exception as e:
            # One-shot retry with stricter output instructions to correct common format issues
            self._emit("warn", f"Failed to get update: {e}")
            strict_suffix = (
                "\n\nSTRICT OUTPUT RULES:\n"
                "- Return ONLY a JSON object with keys: new_nodes, new_edges, node_updates.\n"
                "- Use lowercase keys exactly: refs, src, dst.\n"
                "- Do NOT include any schema, $defs, comments, or trailing text.\n"
            )
            try:
                update = self.llm.parse(
                    system=system_prompt + strict_suffix,
                    user=json.dumps(user_prompt, indent=2),
                    schema=GraphUpdate,
                )
                update.target_graph = graph.name
                return update
            except Exception as e2:
                self._emit("warn", f"Retry failed: {e2}")
                # Final fallback: chunk the cards into smaller batches and merge updates
                try:
                    chunks: list[list[dict]] = []
                    n = max(1, len(cards_with_ids)//2)
                    if n < len(cards_with_ids):
                        chunks = [cards_with_ids[:n], cards_with_ids[n:]]
                    else:
                        # If very small, try 2 random subsets for diversity
                        import random as _rand
                        a = cards_with_ids.copy()
                        _rand.shuffle(a)
                        chunks = [a[: max(1, len(a)//2)], a[max(1, len(a)//2):]]
                    agg = GraphUpdate(new_nodes=[], new_edges=[], node_updates=[])
                    combined = False
                    for part in chunks:
                        up = {**user_prompt, "code_samples": part}
                        try:
                            u = self.llm.parse(
                                system=system_prompt,
                                user=json.dumps(up, indent=2),
                                schema=GraphUpdate
                            )
                            agg.new_nodes.extend(u.new_nodes)
                            agg.new_edges.extend(u.new_edges)
                            agg.node_updates.extend(u.node_updates)
                            combined = True
                        except Exception:
                            continue
                    if combined:
                        agg.target_graph = graph.name
                        return agg
                except Exception:
                    pass
                return None
    
    
    def _apply_update(self, graph: KnowledgeGraph, update: GraphUpdate) -> tuple[int, int]:
        """Apply an update to a graph. Returns (nodes_added, edges_added)."""
        
        nodes_added = 0
        edges_added = 0
        
        # Optionally constrain new nodes during refinement
        new_nodes_specs = list(update.new_nodes or [])
        if getattr(self, '_refine_only', False):
            existing_ids = set(graph.nodes.keys())
            # Keep only new nodes referenced in edges that connect to an existing node; cap to 3
            referenced_with_existing: set[str] = set()
            for e in update.new_edges or []:
                if e.src in existing_ids and e.dst not in existing_ids:
                    referenced_with_existing.add(e.dst)
                if e.dst in existing_ids and e.src not in existing_ids:
                    referenced_with_existing.add(e.src)
            new_nodes_specs = [ns for ns in new_nodes_specs if ns.id in referenced_with_existing][:3]

        # Add new nodes
        for node_spec in new_nodes_specs:
            # Parse properties if provided as JSON string
            properties = {}
            
            node = DynamicNode(
                id=node_spec.id,
                type=node_spec.type,
                label=node_spec.label,
                properties=properties,
                description=None,  # No description in compact schema
                confidence=1.0,  # Default confidence
                source_refs=node_spec.refs,  # Use shortened field name
                created_by=f"iteration_{self.iteration}",
                iteration=self.iteration,
                # Empty security fields since we're focusing on structure
                observations=[],
                assumptions=[]
            )
            if graph.add_node(node):
                nodes_added += 1
        
        # Add new edges (will be deduplicated in add_edge method)
        for edge_spec in (update.new_edges or []):
            if getattr(self, '_refine_only', False):
                # Only add edges where both endpoints exist (after node filtering)
                if edge_spec.src not in graph.nodes or edge_spec.dst not in graph.nodes:
                    continue
            edge = DynamicEdge(
                id=self._generate_id("edge"),
                type=edge_spec.type,
                source_id=edge_spec.src,  # Use shortened field name
                target_id=edge_spec.dst,  # Use shortened field name
                properties={},
                label=edge_spec.type,  # Use type as label
                confidence=1.0,
                evidence=edge_spec.refs or [],
                created_by=f"iteration_{self.iteration}",
                iteration=self.iteration
            )
            if graph.add_edge(edge):
                edges_added += 1
        
        # Update existing nodes
        for node_update in update.node_updates:
            if node_update.id in graph.nodes:
                node = graph.nodes[node_update.id]
                if node_update.description:
                    node.description = node_update.description
                if node_update.properties:
                    try:
                        props = json.loads(node_update.properties)
                        node.properties.update(props)
                    except json.JSONDecodeError:
                        pass
                
                # Add new observations and assumptions
                for obs in node_update.new_observations:
                    node.observations.append(obs.model_dump())
                for assum in node_update.new_assumptions:
                    node.assumptions.append(assum.model_dump())
        
        # Save graph incrementally after applying this update
        try:
            self._save_graph(output_dir=self._output_dir, graph=graph)
            self._emit("save", f"Saved {graph.name}: +{nodes_added}N/+{edges_added}E")
        except Exception:
            pass
        return nodes_added, edges_added
    
    
    @staticmethod
    def load_cards_from_manifest(manifest_dir: Path) -> tuple[list[dict], dict]:
        """
        Load cards and manifest from a project's manifest directory.
        Returns: (cards, manifest)
        """
        import json
        
        if not manifest_dir.exists():
            raise ValueError(f"No manifest found at {manifest_dir}")
        
        # Load manifest
        with open(manifest_dir / "manifest.json") as f:
            manifest = json.load(f)
        
        # Load cards
        cards = []
        repo_root = None
        repo_path = manifest.get('repo_path')
        if repo_path:
            repo_root = Path(repo_path)
        
        from analysis.cards import extract_card_content
        
        with open(manifest_dir / "cards.jsonl") as f:
            for line in f:
                card = json.loads(line)
                if not card.get('content') and repo_root:
                    card['content'] = extract_card_content(card, repo_root)
                cards.append(card)
        
        return cards, manifest
    
    def prepare_code_context(self, cards: list[dict]) -> list[dict]:
        """
        Prepare code context from cards for LLM prompts.
        Filters out redundant fields and returns clean context.
        """
        code_context = []
        for card in cards:
            context_item = {
                "file": card.get("relpath", "unknown"),
                "content": card.get("content", "")
            }
            # Only include type if it exists
            if card.get("type"):
                context_item["type"] = card["type"]
            code_context.append(context_item)
        return code_context
    
    def sample_cards_for_prompt(self, cards: list[dict]) -> tuple[list[dict], int, int]:
        """
        Sample cards for LLM prompt, returning sampled cards and counts.
        Returns: (sampled_cards, original_count, sampled_count)
        """
        original_count = len(cards)
        sampled_cards = self._sample_cards(cards)
        return sampled_cards, original_count, len(sampled_cards)
    
    def _sample_cards_for_discovery(self, cards: list[dict]) -> list[dict]:
        """
        Sample cards specifically for the discovery phase.
        Single-model mode: use the graph model's context limit.
        """
        # Get context limit from graph model or fall back to global default
        graph_model_config = self.config.get("models", {}).get("graph", {})
        max_context_tokens = graph_model_config.get("max_context") or self.config.get("context", {}).get("max_tokens", 256000)
        if self.debug:
            print(f"      Discovery: Using graph model's max_context: {max_context_tokens:,} tokens")
        
        # Reserve more tokens for discovery (system prompt is larger, needs more response space)
        reserved_tokens = 50000  # More conservative reservation for discovery
        available_tokens = max_context_tokens - reserved_tokens
        target_tokens = int(available_tokens * 0.7)  # More conservative: 70% instead of 80%
        
        # Use the graph model for token counting
        model = graph_model_config.get("model", "gpt-4o-mini")
        
        # Count tokens for all cards
        total_tokens = 0
        card_tokens = []
        for card in cards:
            card_text = f"{card.get('relpath', '')}\n{card.get('content', '')}"
            tokens = count_tokens(card_text, "openai", model)
            card_tokens.append(tokens)
            total_tokens += tokens
        
        # If under threshold, use all cards
        if total_tokens <= target_tokens:
            if self.debug:
                usage_pct = (total_tokens * 100) // max_context_tokens
                print(f"      Discovery: Using ALL {len(cards)} cards ({total_tokens:,} tokens, {usage_pct}% of context)")
            return cards
        
        # Need to sample - use same diversity logic
        ratio = target_tokens / total_tokens
        sample_size = max(1, int(len(cards) * ratio))
        
        if self.debug:
            print(f"      Discovery: Sampling {sample_size} cards from {len(cards)} total (tokens: {total_tokens:,} > {target_tokens:,} target)")
        
        import random
        
        # Sample with file diversity
        files_to_cards = {}
        for card in cards:
            file_path = card.get("relpath", "unknown")
            if file_path not in files_to_cards:
                files_to_cards[file_path] = []
            files_to_cards[file_path].append(card)
        
        sampled = []
        files = list(files_to_cards.keys())
        random.shuffle(files)
        
        # Take cards from different files
        cards_per_file = max(1, min(3, sample_size // len(files))) if len(files) > 0 else sample_size
        
        for file_path in files:
            if len(sampled) >= sample_size:
                break
            file_cards = files_to_cards[file_path]
            num_from_file = min(cards_per_file, len(file_cards), sample_size - len(sampled))
            sampled.extend(random.sample(file_cards, num_from_file))
        
        # Fill remaining if needed
        if len(sampled) < sample_size:
            remaining = [c for c in cards if c not in sampled]
            additional = min(sample_size - len(sampled), len(remaining))
            if additional > 0:
                sampled.extend(random.sample(remaining, additional))
        
        # Final safety check - verify we're under limit
        final_tokens = 0
        for card in sampled:
            card_text = f"{card.get('relpath', '')}\n{card.get('content', '')}"
            final_tokens += count_tokens(card_text, "openai", model)
        
        # If still over, remove cards until we fit
        while final_tokens > target_tokens and len(sampled) > 1:
            # Remove a random card
            removed_idx = random.randint(0, len(sampled) - 1)
            removed_card = sampled.pop(removed_idx)
            removed_text = f"{removed_card.get('relpath', '')}\n{removed_card.get('content', '')}"
            final_tokens -= count_tokens(removed_text, "openai", model)
            if self.debug:
                print(f"      Discovery: Removing card to fit in context (now {final_tokens:,} tokens)")
        
        if self.debug:
            print(f"      Discovery: Final sample: {len(sampled)} cards, {final_tokens:,} tokens (target was {target_tokens:,})")
        
        return sampled
    
    def _sample_cards(
        self,
        cards: list[dict]
    ) -> list[dict]:
        """Adaptive sampling based on token count to stay within context limits"""
        
        # Get max tokens for graph model specifically, if configured
        # First check if graph model has its own max_context setting
        graph_model_config = self.config.get("models", {}).get("graph", {})
        graph_max_context = graph_model_config.get("max_context")
        
        if graph_max_context:
            # Use graph model's specific context limit
            max_context_tokens = graph_max_context
            if self.debug:
                print(f"      Using graph model's max_context: {max_context_tokens:,} tokens")
        else:
            # Fall back to global context limit
            max_context_tokens = self.config.get("context", {}).get("max_tokens", 256000)
        
        # Reserve tokens for system prompt and response (30k should be enough)
        reserved_tokens = 30000
        available_tokens = max_context_tokens - reserved_tokens
        
        # Target to use 80% of available tokens (more aggressive usage)
        target_tokens = int(available_tokens * 0.8)
        
        # Estimate tokens for all cards (only content and path, no peek fields)
        total_tokens = 0
        card_tokens = []
        # Use the actual graph model for token counting
        model = graph_model_config.get("model", "gpt-4")
        
        for card in cards:
            # Only count the fields we'll actually send
            card_text = f"{card.get('relpath', '')}\n{card.get('content', '')}"
            tokens = count_tokens(card_text, "openai", model)
            card_tokens.append(tokens)
            total_tokens += tokens
        
        # If under threshold, use all cards
        if total_tokens <= target_tokens:
            if self.debug:
                usage_pct = (total_tokens * 100) // max_context_tokens
                print(f"      Using ALL {len(cards)} cards (total tokens: {total_tokens:,} < {target_tokens:,} target)")
                print(f"      Context usage: {usage_pct}% of {max_context_tokens:,} available tokens")
            return cards
        
        # Calculate proportional sample size to stay under token limit
        ratio = target_tokens / total_tokens
        sample_size = max(1, int(len(cards) * ratio))
        
        if self.debug:
            print(f"      WARNING: Large codebase - sampling {sample_size} cards from {len(cards)} total (tokens: {total_tokens:,} > {target_tokens:,} target, ratio: {ratio:.3f})")
        
        # Use original file diversity sampling logic
        import random
        
        # Sample cards ensuring file diversity - no hardcoded priorities
        files_to_cards = {}
        for card in cards:
            file_path = card.get("relpath", "unknown")
            if file_path not in files_to_cards:
                files_to_cards[file_path] = []
            files_to_cards[file_path].append(card)
        
        # Sample cards ensuring we get representation from many files
        sampled = []
        files = list(files_to_cards.keys())
        random.shuffle(files)  # Random order to avoid bias
        
        # Calculate cards per file to maximize coverage
        if len(files) > 0:
            cards_per_file = max(1, sample_size // len(files))
            # But cap at 2-3 cards per file to ensure diversity
            cards_per_file = min(cards_per_file, 3)
        else:
            cards_per_file = sample_size
        
        for file_path in files:
            if len(sampled) >= sample_size:
                break
            file_cards = files_to_cards[file_path]
            # Take up to cards_per_file from each file
            num_from_file = min(cards_per_file, len(file_cards), sample_size - len(sampled))
            sampled.extend(random.sample(file_cards, num_from_file))
        
        # Fill up with random cards if needed
        if len(sampled) < sample_size:
            remaining = [c for c in cards if c not in sampled]
            additional = min(sample_size - len(sampled), len(remaining))
            sampled.extend(random.sample(remaining, additional))
        
        # Verify we're under the token threshold, trim if needed
        sampled_tokens = 0
        for card in sampled:
            card_text = f"{card.get('relpath', '')}\n{card.get('content', '')}"
            sampled_tokens += count_tokens(card_text, "openai", model)
        
        # If still over threshold, remove cards until we fit
        while sampled_tokens > target_tokens and len(sampled) > 1:
            # Remove the smallest card
            smallest_idx = 0
            smallest_tokens = count_tokens(
                f"{sampled[0].get('relpath', '')}\n{sampled[0].get('content', '')}", 
                "openai", model
            )
            for i, card in enumerate(sampled[1:], 1):
                card_text = f"{card.get('relpath', '')}\n{card.get('content', '')}"
                card_tokens = count_tokens(card_text, "openai", model)
                if card_tokens < smallest_tokens:
                    smallest_tokens = card_tokens
                    smallest_idx = i
            
            removed_card = sampled.pop(smallest_idx)
            removed_text = f"{removed_card.get('relpath', '')}\n{removed_card.get('content', '')}"
            sampled_tokens -= count_tokens(removed_text, "openai", model)
        
        if self.debug:
            print(f"      Sampled {len(sampled)} cards from {len(set(c.get('relpath', 'unknown') for c in sampled))} files (final tokens: {sampled_tokens:,})")
        
        return sampled
    
    def _load_manifest(self, manifest_dir: Path) -> tuple:
        """Load manifest and cards with full content extraction"""
        
        with open(manifest_dir / "manifest.json") as f:
            manifest = json.load(f)
        
        repo_path = manifest.get('repo_path')
        if repo_path:
            self._repo_root = Path(repo_path)
        
        from .cards import extract_card_content
        
        cards = []
        with open(manifest_dir / "cards.jsonl") as f:
            for line in f:
                card = json.loads(line)
                if not card.get('content') and self._repo_root:
                    card['content'] = extract_card_content(card, self._repo_root)
                cards.append(card)
        
        return manifest, cards
    
    def _save_results(self, output_dir: Path, manifest: dict) -> dict:
        """Save all graphs and analysis results"""
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            "manifest": manifest,
            "timestamp": time.time(),
            "iterations": self.iteration,
            "graphs": {},
            "total_nodes": 0,
            "total_edges": 0,
            "card_references": {}  # Map of card IDs to their content
        }
        
        # Collect all referenced cards (nodes and edges)
        all_card_ids = set()
        for graph in self.graphs.values():
            for node in graph.nodes.values():
                all_card_ids.update(node.source_refs)
            for edge in graph.edges.values():
                if hasattr(edge, 'evidence') and edge.evidence:
                    all_card_ids.update(edge.evidence)
        
        # Save each graph
        for name, graph in self.graphs.items():
            # Save individual graph using helper (atomic)
            graph_file = self._save_graph(output_dir, graph)
            
            results["graphs"][name] = str(graph_file)
            results["total_nodes"] += len(graph.nodes)
            results["total_edges"] += len(graph.edges)
        
        # Save card store separately for retrieval during security analysis
        # Only save cards that are actually referenced by nodes
        referenced_cards = {}
        for card_id in all_card_ids:
            if card_id in self.card_store:
                referenced_cards[card_id] = self.card_store[card_id]
        
        card_refs_file = output_dir / "card_store.json"
        with open(card_refs_file, "w") as f:
            json.dump(referenced_cards, f, indent=2)
        
        results["card_store_path"] = str(card_refs_file)
        results["cards_stored"] = len(referenced_cards)
        
        # Save combined results
        results_file = output_dir / "knowledge_graphs.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        self._emit("save", f"Saved {len(self.graphs)} graphs to {output_dir}")
        self._emit("save", f"Total: {results['total_nodes']} nodes, {results['total_edges']} edges")
        self._emit("save", f"Card store: {len(referenced_cards)} cards saved")
        
        # Report connectivity stats
        for name, graph in self.graphs.items():
            orphans = self._get_orphaned_nodes(graph)
            if orphans:
                pct = (len(orphans)*100//max(1,len(graph.nodes)))
                self._emit("warn", f"{name} has {len(orphans)} disconnected nodes ({pct}%)")
        
        return results

    def _save_graph(self, output_dir: Path, graph: KnowledgeGraph) -> Path:
        """Save a single graph JSON atomically and return its path."""
        output_dir.mkdir(parents=True, exist_ok=True)
        data = graph.to_dict()
        graph_file = output_dir / f"graph_{graph.name}.json"
        tmp = graph_file.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        tmp.replace(graph_file)
        # Remember output dir for incremental saves
        return graph_file

    def _load_existing_graphs(self, output_dir: Path) -> None:
        """Load existing graph JSON files into self.graphs for refinement."""
        self._output_dir = output_dir  # Store for incremental saves
        if not output_dir.exists():
            return
        for p in output_dir.glob("graph_*.json"):
            try:
                with open(p) as f:
                    data = json.load(f)
                name = data.get("internal_name") or data.get("name") or p.stem.replace("graph_", "")
                focus = data.get("focus", "")
                kg = KnowledgeGraph(name=name, focus=focus, metadata=data.get("metadata") or {})
                for nd in data.get("nodes", []) or []:
                    try:
                        kg.nodes[nd.get("id") or "unknown"] = DynamicNode(**nd)
                    except Exception:
                        continue
                for ed in data.get("edges", []) or []:
                    try:
                        kg.edges[ed.get("id") or f"e_{len(kg.edges)}"] = DynamicEdge(**ed)
                    except Exception:
                        continue
                self.graphs[name] = kg
            except Exception:
                continue
    
    def _generate_id(self, prefix: str) -> str:
        """Generate unique ID"""
        timestamp = str(time.time())
        hash_val = hashlib.sha256(timestamp.encode()).hexdigest()[:8]
        return f"{prefix}_{hash_val}"
