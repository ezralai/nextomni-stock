"""Interactive Query Engine for NextOmni Stock CodeGraph.

Enables developers and AI agents to query callers, callees, blast radiuses,
database table dataflows, dead code candidates, and test coverage gaps.
"""

import argparse
from collections import defaultdict, deque
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


class CodeGraphQueryEngine:
    """In-memory query engine over the serialized NextOmni CodeGraph."""

    def __init__(self, graph_data: Dict[str, Any]) -> None:
        self.metadata = graph_data.get("metadata", {})
        self.nodes = {n["id"]: n for n in graph_data.get("nodes", [])}
        self.edges = graph_data.get("edges", [])

        # Adjacency lists
        self.outbound: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.inbound: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for edge in self.edges:
            src = edge["source"]
            tgt = edge["target"]
            self.outbound[src].append(edge)
            self.inbound[tgt].append(edge)

    @classmethod
    def load_from_file(cls, path: Path) -> "CodeGraphQueryEngine":
        if not path.exists():
            raise FileNotFoundError(f"CodeGraph file not found at {path}. Run build_graph.py first.")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data)

    def find_node_ids(self, pattern: str) -> List[str]:
        """Find matching node IDs by exact match, suffix, or symbol name."""
        matches = []
        for node_id, node in self.nodes.items():
            if node_id == pattern or node_id.endswith(f":{pattern}") or node.get("name") == pattern:
                matches.append(node_id)
            elif pattern.lower() in node_id.lower():
                matches.append(node_id)
        return matches

    def get_callers(self, target_symbol: str) -> List[Dict[str, Any]]:
        """Find all functions, classes, and files that call, import, or test target_symbol."""
        target_nodes = self.find_node_ids(target_symbol)
        callers = []
        seen = set()

        for tgt in target_nodes:
            for edge in self.inbound.get(tgt, []):
                src = edge["source"]
                edge_key = (src, tgt, edge["type"])
                if edge_key not in seen:
                    seen.add(edge_key)
                    callers.append({
                        "caller_id": src,
                        "caller_file": self.nodes.get(src, {}).get("file", src.split(":")[0]),
                        "edge_type": edge["type"],
                        "target_id": tgt,
                        "line": edge.get("line"),
                    })
        return callers

    def get_callees(self, caller_symbol: str) -> List[Dict[str, Any]]:
        """Find all functions, classes, and modules invoked by caller_symbol."""
        caller_nodes = self.find_node_ids(caller_symbol)
        callees = []
        seen = set()

        for src in caller_nodes:
            for edge in self.outbound.get(src, []):
                tgt = edge["target"]
                edge_key = (src, tgt, edge["type"])
                if edge_key not in seen:
                    seen.add(edge_key)
                    callees.append({
                        "caller_id": src,
                        "callee_id": tgt,
                        "edge_type": edge["type"],
                        "symbol": edge.get("symbol", tgt.split(":")[-1]),
                        "line": edge.get("line"),
                    })
        return callees

    def compute_blast_radius(self, target_symbol: str, max_depth: int = 3) -> Dict[str, Any]:
        """Compute recursive upstream callers that would be affected by changing target_symbol."""
        targets = self.find_node_ids(target_symbol)
        if not targets:
            return {"target": target_symbol, "matched_nodes": [], "affected_files": [], "call_paths": []}

        queue: deque = deque([(t, 0, [t]) for t in targets])
        visited: Set[str] = set(targets)
        paths: List[Dict[str, Any]] = []
        affected_files: Set[str] = set()

        while queue:
            curr_id, depth, path = queue.popleft()
            curr_file = self.nodes.get(curr_id, {}).get("file", curr_id.split(":")[0])
            if curr_file and not curr_file.startswith("external:") and not curr_file.startswith("db_table:"):
                affected_files.add(curr_file)

            if depth < max_depth:
                for edge in self.inbound.get(curr_id, []):
                    parent_id = edge["source"]
                    new_path = path + [f"({edge['type']})-> {parent_id}"]
                    paths.append({
                        "source": parent_id,
                        "target": curr_id,
                        "depth": depth + 1,
                        "edge_type": edge["type"],
                        "path_str": " -> ".join(new_path),
                    })
                    if parent_id not in visited:
                        visited.add(parent_id)
                        queue.append((parent_id, depth + 1, new_path))

        return {
            "target": target_symbol,
            "matched_nodes": targets,
            "total_affected_nodes": len(visited),
            "affected_files": sorted(list(affected_files)),
            "call_paths_count": len(paths),
            "sample_paths": paths[:15],
        }

    def get_table_dataflow(self, table_name: str) -> Dict[str, Any]:
        """Find all functions and modules that read from or write to a specific database table."""
        tbl_node_id = f"db_table:{table_name}"
        readers = []
        writers = []

        for edge in self.inbound.get(tbl_node_id, []):
            edge_info = {
                "caller": edge["source"],
                "file": self.nodes.get(edge["source"], {}).get("file", edge["source"].split(":")[0]),
                "line": edge.get("line"),
            }
            if edge["type"] == "READS_TABLE":
                readers.append(edge_info)
            elif edge["type"] == "WRITES_TABLE":
                writers.append(edge_info)

        return {
            "table": table_name,
            "table_node_id": tbl_node_id,
            "readers_count": len(readers),
            "writers_count": len(writers),
            "readers": readers,
            "writers": writers,
        }

    def find_dead_code_candidates(self) -> List[Dict[str, Any]]:
        """Identify functions in src/ that have zero inbound CALLS and zero TESTS edges."""
        dead_candidates = []
        exempt_prefixes = ("main", "run", "cli", "__", "test_")

        for node_id, node in self.nodes.items():
            if node["type"] == "function" and node.get("file", "").startswith("src/"):
                fn_name = node["name"]
                if any(fn_name.startswith(p) for p in exempt_prefixes):
                    continue

                in_edges = self.inbound.get(node_id, [])
                calls_edges = [e for e in in_edges if e["type"] in {"CALLS", "TESTS"}]
                if not calls_edges:
                    dead_candidates.append({
                        "function_id": node_id,
                        "file": node["file"],
                        "name": fn_name,
                        "line": node.get("line"),
                        "total_inbound_edges": len(in_edges),
                    })
        return dead_candidates

    def suggest_missing_tests(self) -> List[Dict[str, Any]]:
        """List functions in src/ lacking test coverage and suggest corresponding test locations."""
        suggestions = []
        for node_id, node in self.nodes.items():
            if node["type"] == "function" and node.get("file", "").startswith("src/"):
                fn_name = node["name"]
                if fn_name.startswith("__") or fn_name == "main":
                    continue

                in_edges = self.inbound.get(node_id, [])
                test_edges = [e for e in in_edges if e["type"] == "TESTS"]
                if not test_edges:
                    src_file = Path(node["file"])
                    suggested_test_file = f"tests/test_{src_file.stem}.py"
                    suggestions.append({
                        "function_id": node_id,
                        "file": node["file"],
                        "name": fn_name,
                        "line": node.get("line"),
                        "suggested_test_file": suggested_test_file,
                        "suggested_test_name": f"test_{fn_name}",
                        "callers_count": len(in_edges),
                    })
        return suggestions

    # Alias for backward compatibility
    audit_test_coverage_gaps = suggest_missing_tests



def main() -> None:
    parser = argparse.ArgumentParser(description="Query NextOmni Stock CodeGraph")
    parser.add_argument(
        "--graph",
        type=str,
        default=".agents/skills/codegraph/graph.json",
        help="Path to graph.json",
    )
    parser.add_argument("--callers", type=str, help="Find all callers of a function or class symbol")
    parser.add_argument("--callees", type=str, help="Find all functions called by a function")
    parser.add_argument("--blast-radius", type=str, help="Calculate recursive blast radius for a symbol/file")
    parser.add_argument("--dataflow", type=str, help="Trace read/write dataflow for a SQLite database table")
    parser.add_argument("--dead-code", action="store_true", help="Find uncalled functions in src/ (dead code)")
    parser.add_argument("--suggest-tests", action="store_true", help="Suggest tests for untested functions in src/")
    parser.add_argument("--audit-tests", action="store_true", help="Audit functions lacking test coverage")
    parser.add_argument("--summary", action="store_true", help="Print CodeGraph topology summary")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")

    args = parser.parse_args()
    graph_path = Path(args.graph)

    if not graph_path.exists():
        from build_graph import extract_repository_graph
        root_dir = Path(".")
        graph_data = extract_repository_graph(root_dir)
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        with open(graph_path, "w", encoding="utf-8") as f:
            json.dump(graph_data, f, indent=2)
    else:
        with open(graph_path, "r", encoding="utf-8") as f:
            graph_data = json.load(f)

    engine = CodeGraphQueryEngine(graph_data)

    if args.summary:
        meta = engine.metadata
        if args.json:
            print(json.dumps(meta, indent=2))
        else:
            print("\n" + "=" * 70)
            print("  NextOmni Stock CodeGraph Topology Summary")
            print("=" * 70)
            for k, v in meta.items():
                print(f"  {k:<24}: {v}")
            print("=" * 70 + "\n")
        return

    if args.callers:
        results = engine.get_callers(args.callers)
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            print(f"\nCallers of symbol '{args.callers}' ({len(results)} found):")
            for r in results:
                print(f"  - [{r['edge_type']}] {r['caller_id']} (line {r.get('line')}) -> {r['target_id']}")
            print()
        return

    if args.callees:
        results = engine.get_callees(args.callees)
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            print(f"\nCallees called by symbol '{args.callees}' ({len(results)} found):")
            for r in results:
                print(f"  - [{r['edge_type']}] calls {r['callee_id']} (line {r.get('line')})")
            print()
        return

    if getattr(args, "blast_radius", None):
        target = args.blast_radius
        res = engine.compute_blast_radius(target)
        if args.json:
            print(json.dumps(res, indent=2))
        else:
            print(f"\nBlast Radius for '{target}':")
            print(f"  Matched Nodes:        {res['matched_nodes']}")
            print(f"  Total Affected Nodes: {res['total_affected_nodes']}")
            print(f"  Affected Files ({len(res['affected_files'])}):")
            for f in res["affected_files"]:
                print(f"    - {f}")
            print(f"  Sample Paths Trace:")
            for p in res["sample_paths"][:5]:
                print(f"    - {p['path_str']}")
            print()
        return

    if args.dataflow:
        res = engine.get_table_dataflow(args.dataflow)
        if args.json:
            print(json.dumps(res, indent=2))
        else:
            print(f"\nSQLite Dataflow for Table '{args.dataflow}':")
            print(f"  Readers ({res['readers_count']}):")
            for r in res["readers"][:10]:
                print(f"    - {r['caller']} (line {r.get('line')}) in {r['file']}")
            print(f"  Writers ({res['writers_count']}):")
            for w in res["writers"][:10]:
                print(f"    - {w['caller']} (line {w.get('line')}) in {w['file']}")
            print()
        return

    if args.dead_code:
        res = engine.find_dead_code_candidates()
        if args.json:
            print(json.dumps(res, indent=2))
        else:
            print(f"\nPotential Dead Code Candidates in src/ ({len(res)} found):")
            for d in res[:20]:
                print(f"  - {d['function_id']} (line {d.get('line')}) in {d['file']}")
            if len(res) > 20:
                print(f"  ... and {len(res) - 20} more.")
            print()
        return

    if args.suggest_tests:
        res = engine.suggest_missing_tests()
        if args.json:
            print(json.dumps(res, indent=2))
        else:
            print(f"\nSuggested Test Templates for Untested Functions ({len(res)} found):")
            for s in res[:15]:
                print(f"  - {s['name']} -> {s['suggested_test_file']}::{s['suggested_test_name']} (Callers: {s['callers_count']})")
            if len(res) > 15:
                print(f"  ... and {len(res) - 15} more.")
            print()
        return

    if args.audit_tests:
        gaps = engine.suggest_missing_tests()
        if args.json:
            print(json.dumps(gaps, indent=2))
        else:
            print(f"\nFunctions in src/ lacking direct TESTS edges ({len(gaps)} found):")
            for g in gaps[:20]:
                print(f"  - {g['function_id']} (Callers: {g['callers_count']})")
            if len(gaps) > 20:
                print(f"  ... and {len(gaps) - 20} more.")
            print()
        return

    parser.print_help()


if __name__ == "__main__":
    main()
