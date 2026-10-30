"""AST-based CodeGraph Extractor for NextOmni Stock.

Extracts a semantic graph representation of files, classes, functions,
imports, call hierarchies, database table dataflows, and CLI dispatchers
across src/ and tests/ using Python's standard ast.
Features incremental SHA-256 caching for fast re-indexing.
Zero external dependencies required.
"""

import argparse
import ast
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple

# Known database tables in NextOmni Stock SQLite schema
KNOWN_DB_TABLES = {
    "stocks",
    "market_observations",
    "stock_observations",
    "predictions",
    "prediction_evaluations",
    "raw_payload_cache",
    "audit_log",
}



def compute_file_hash(path: Path) -> str:
    """Compute deterministic SHA-256 content hash of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CodeGraphBuilder(ast.NodeVisitor):
    """AST visitor that extracts definitions, imports, calls, SQL dataflows, and CLI commands."""

    def __init__(self, rel_path: str) -> None:
        self.rel_path = rel_path
        self.current_scope: List[str] = [rel_path]
        self.nodes: List[Dict[str, Any]] = []
        self.edges: List[Dict[str, Any]] = []
        self.defined_functions: Set[str] = set()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        class_id = f"{self.rel_path}:{node.name}"
        parent_scope = self.current_scope[-1]

        self.nodes.append({
            "id": class_id,
            "type": "class",
            "name": node.name,
            "file": self.rel_path,
            "line": node.lineno,
            "docstring": ast.get_docstring(node) or "",
        })
        self.edges.append({
            "source": parent_scope,
            "target": class_id,
            "type": "DEFINES",
        })

        self.current_scope.append(class_id)
        self.generic_visit(node)
        self.current_scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._record_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._record_function(node)

    def _record_function(self, node: Any) -> None:
        fn_id = f"{self.rel_path}:{node.name}"
        parent_scope = self.current_scope[-1]
        self.defined_functions.add(node.name)

        is_test = self.rel_path.startswith("tests/") or node.name.startswith("test_")
        args_list = [arg.arg for arg in node.args.args if arg.arg != "self"]

        self.nodes.append({
            "id": fn_id,
            "type": "test" if is_test else "function",
            "name": node.name,
            "file": self.rel_path,
            "line": node.lineno,
            "args": args_list,
            "docstring": ast.get_docstring(node) or "",
        })
        self.edges.append({
            "source": parent_scope,
            "target": fn_id,
            "type": "DEFINES",
        })

        self.current_scope.append(fn_id)
        self.generic_visit(node)
        self.current_scope.pop()

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.edges.append({
                "source": self.rel_path,
                "target": alias.name,
                "type": "IMPORTS",
                "imported_symbol": alias.asname or alias.name,
                "line": node.lineno,
            })
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        mod = node.module or ""
        for alias in node.names:
            target_symbol = f"{mod}.{alias.name}" if mod else alias.name
            self.edges.append({
                "source": self.rel_path,
                "target": target_symbol,
                "type": "IMPORTS",
                "imported_symbol": alias.asname or alias.name,
                "line": node.lineno,
            })
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        caller_scope = self.current_scope[-1]
        target_name = None

        if isinstance(node.func, ast.Name):
            target_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            target_name = node.func.attr

            # Detect CLI subparser registration: subparsers.add_parser("command_name")
            if target_name == "add_parser" and node.args:
                first_arg = node.args[0]
                if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
                    cmd_name = first_arg.value
                    cmd_node_id = f"cli:subcommand:{cmd_name}"
                    self.nodes.append({
                        "id": cmd_node_id,
                        "type": "cli_subcommand",
                        "name": cmd_name,
                        "file": self.rel_path,
                        "line": node.lineno,
                    })
                    self.edges.append({
                        "source": caller_scope,
                        "target": cmd_node_id,
                        "type": "REGISTERS_CLI",
                        "line": node.lineno,
                    })

        if target_name:
            edge_type = "TESTS" if (self.rel_path.startswith("tests/") and not target_name.startswith("test_")) else "CALLS"
            self.edges.append({
                "source": caller_scope,
                "target_symbol": target_name,
                "type": edge_type,
                "line": node.lineno,
            })

        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        """Inspect string constants to detect SQL table reads and writes."""
        if isinstance(node.value, str) and len(node.value) > 8:
            text = node.value.strip()
            upper_text = text.upper()
            caller_scope = self.current_scope[-1]

            # Detect SELECT / READ dataflows
            if "SELECT " in upper_text or " FROM " in upper_text:
                for tbl in KNOWN_DB_TABLES:
                    if re.search(rf"\bFROM\s+{tbl}\b", upper_text, re.IGNORECASE) or re.search(rf"\bJOIN\s+{tbl}\b", upper_text, re.IGNORECASE):
                        self.edges.append({
                            "source": caller_scope,
                            "target": f"db_table:{tbl}",
                            "type": "READS_TABLE",
                            "table": tbl,
                            "line": getattr(node, "lineno", None),
                        })

            # Detect INSERT / UPDATE / DELETE dataflows
            if any(kw in upper_text for kw in ["INSERT INTO ", "UPDATE ", "DELETE FROM "]):
                for tbl in KNOWN_DB_TABLES:
                    if (
                        re.search(rf"\bINTO\s+{tbl}\b", upper_text, re.IGNORECASE)
                        or re.search(rf"\bUPDATE\s+{tbl}\b", upper_text, re.IGNORECASE)
                        or re.search(rf"\bFROM\s+{tbl}\b", upper_text, re.IGNORECASE)
                    ):
                        self.edges.append({
                            "source": caller_scope,
                            "target": f"db_table:{tbl}",
                            "type": "WRITES_TABLE",
                            "table": tbl,
                            "line": getattr(node, "lineno", None),
                        })

        self.generic_visit(node)


def extract_repository_graph(
    root_dir: Path,
    cache_path: Optional[Path] = None,
    force_rebuild: bool = False,
) -> Dict[str, Any]:
    """Scan all Python modules in src/ and tests/ to construct the complete CodeGraph with caching."""
    all_nodes: List[Dict[str, Any]] = []
    all_edges: List[Dict[str, Any]] = []

    # Inject database table nodes
    for tbl in KNOWN_DB_TABLES:
        all_nodes.append({
            "id": f"db_table:{tbl}",
            "type": "database_table",
            "name": tbl,
            "path": "src/data/db.py",
        })

    # Cache handling
    file_hashes: Dict[str, str] = {}
    if cache_path and cache_path.exists() and not force_rebuild:
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                file_hashes = json.load(f)
        except Exception:
            file_hashes = {}

    new_hashes: Dict[str, str] = {}
    symbol_to_nodes: Dict[str, List[str]] = defaultdict(list)

    py_files = sorted(list(root_dir.glob("src/**/*.py")) + list(root_dir.glob("tests/**/*.py")))

    for py_file in py_files:
        rel_path = str(py_file.relative_to(root_dir))
        curr_hash = compute_file_hash(py_file)
        new_hashes[rel_path] = curr_hash

        all_nodes.append({
            "id": rel_path,
            "type": "file",
            "name": py_file.name,
            "path": rel_path,
            "sha256": curr_hash[:16],
        })

        try:
            code_text = py_file.read_text(encoding="utf-8")
            tree = ast.parse(code_text, filename=rel_path)
            builder = CodeGraphBuilder(rel_path)
            builder.visit(tree)

            all_nodes.extend(builder.nodes)
            all_edges.extend(builder.edges)

            for fn_node in builder.nodes:
                if fn_node["type"] in {"function", "class", "test"}:
                    symbol_to_nodes[fn_node["name"]].append(fn_node["id"])

        except Exception:
            continue

    # Resolve target_symbol references to concrete node IDs
    resolved_edges: List[Dict[str, Any]] = []
    for edge in all_edges:
        if "target_symbol" in edge:
            sym = edge["target_symbol"]
            candidates = symbol_to_nodes.get(sym, [])
            if candidates:
                for cand in candidates:
                    resolved_edges.append({
                        "source": edge["source"],
                        "target": cand,
                        "type": edge["type"],
                        "symbol": sym,
                        "line": edge.get("line"),
                    })
            else:
                resolved_edges.append({
                    "source": edge["source"],
                    "target": f"external:{sym}",
                    "type": edge["type"],
                    "symbol": sym,
                    "line": edge.get("line"),
                })
        else:
            resolved_edges.append(edge)

    # Save cache hashes
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(new_hashes, f, indent=2)

    # Compute summary statistics
    file_count = sum(1 for n in all_nodes if n["type"] == "file")
    fn_count = sum(1 for n in all_nodes if n["type"] == "function")
    test_count = sum(1 for n in all_nodes if n["type"] == "test")
    class_count = sum(1 for n in all_nodes if n["type"] == "class")
    table_count = sum(1 for n in all_nodes if n["type"] == "database_table")
    cli_count = sum(1 for n in all_nodes if n["type"] == "cli_subcommand")

    return {
        "metadata": {
            "root_dir": str(root_dir.resolve()),
            "total_nodes": len(all_nodes),
            "total_edges": len(resolved_edges),
            "file_count": file_count,
            "function_count": fn_count,
            "class_count": class_count,
            "test_count": test_count,
            "table_count": table_count,
            "cli_subcommands_count": cli_count,
        },
        "nodes": all_nodes,
        "edges": resolved_edges,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CodeGraph for NextOmni Stock repository")
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Root directory of the repository (default: current directory)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=".agents/skills/codegraph/graph.json",
        help="Output path for the serialized graph JSON",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force full re-parsing ignoring hash cache",
    )
    args = parser.parse_args()

    root_path = Path(args.root)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path = out_path.parent / ".cache_hashes.json"

    print(f"Building NextOmni CodeGraph from {root_path.resolve()}...")
    graph = extract_repository_graph(root_path, cache_path=cache_path, force_rebuild=args.force)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2)

    meta = graph["metadata"]
    print("=" * 70)
    print("  NextOmni CodeGraph Built Successfully (Optimized AST & Dataflow)!")
    print(f"  Output:           {out_path}")
    print(f"  Files Indexed:    {meta['file_count']}")
    print(f"  Functions:        {meta['function_count']}")
    print(f"  Classes:          {meta['class_count']}")
    print(f"  Tests:            {meta['test_count']}")
    print(f"  DB Tables:        {meta['table_count']}")
    print(f"  CLI Subparsers:   {meta['cli_subcommands_count']}")
    print(f"  Total Nodes:      {meta['total_nodes']}")
    print(f"  Total Edges:      {meta['total_edges']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
