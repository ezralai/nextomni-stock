"""AST-based CodeGraph Extractor for NextOmni Stock.

Extracts a semantic graph representation of files, classes, functions,
imports, and call hierarchies across src/ and tests/ using Python's standard ast.
Zero external dependencies required.
"""

import argparse
import ast
from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


class CodeGraphBuilder(ast.NodeVisitor):
    """AST visitor that extracts definitions, imports, and function calls from a Python module."""

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

        if target_name:
            edge_type = "TESTS" if (self.rel_path.startswith("tests/") and not target_name.startswith("test_")) else "CALLS"
            self.edges.append({
                "source": caller_scope,
                "target_symbol": target_name,
                "type": edge_type,
                "line": node.lineno,
            })

        self.generic_visit(node)


def extract_repository_graph(root_dir: Path) -> Dict[str, Any]:
    """Scan all Python modules in src/ and tests/ to construct the complete CodeGraph."""
    all_nodes: List[Dict[str, Any]] = []
    all_edges: List[Dict[str, Any]] = []

    # Map function name to list of defined node IDs
    symbol_to_nodes: Dict[str, List[str]] = defaultdict(list)

    py_files = sorted(list(root_dir.glob("src/**/*.py")) + list(root_dir.glob("tests/**/*.py")))

    for py_file in py_files:
        rel_path = str(py_file.relative_to(root_dir))
        all_nodes.append({
            "id": rel_path,
            "type": "file",
            "name": py_file.name,
            "path": rel_path,
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

        except Exception as ex:
            # Non-blocking skip on syntax or encoding anomalies
            continue

    # Resolve target_symbol references to concrete node IDs where unambiguous
    resolved_edges: List[Dict[str, Any]] = []
    for edge in all_edges:
        if "target_symbol" in edge:
            sym = edge["target_symbol"]
            candidates = symbol_to_nodes.get(sym, [])
            if candidates:
                # Target resolved to existing function(s)
                for cand in candidates:
                    resolved_edges.append({
                        "source": edge["source"],
                        "target": cand,
                        "type": edge["type"],
                        "symbol": sym,
                        "line": edge.get("line"),
                    })
            else:
                # Builtin or external package call (e.g., np.mean, len, print)
                resolved_edges.append({
                    "source": edge["source"],
                    "target": f"external:{sym}",
                    "type": edge["type"],
                    "symbol": sym,
                    "line": edge.get("line"),
                })
        else:
            resolved_edges.append(edge)

    # Compute summary statistics
    file_count = sum(1 for n in all_nodes if n["type"] == "file")
    fn_count = sum(1 for n in all_nodes if n["type"] == "function")
    test_count = sum(1 for n in all_nodes if n["type"] == "test")
    class_count = sum(1 for n in all_nodes if n["type"] == "class")

    return {
        "metadata": {
            "root_dir": str(root_dir.resolve()),
            "total_nodes": len(all_nodes),
            "total_edges": len(resolved_edges),
            "file_count": file_count,
            "function_count": fn_count,
            "class_count": class_count,
            "test_count": test_count,
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
    args = parser.parse_args()

    root_path = Path(args.root)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Building NextOmni CodeGraph from {root_path.resolve()}...")
    graph = extract_repository_graph(root_path)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2)

    meta = graph["metadata"]
    print("=" * 70)
    print("  NextOmni CodeGraph Built Successfully!")
    print(f"  Output:           {out_path}")
    print(f"  Files Indexed:    {meta['file_count']}")
    print(f"  Functions:        {meta['function_count']}")
    print(f"  Classes:          {meta['class_count']}")
    print(f"  Tests:            {meta['test_count']}")
    print(f"  Total Nodes:      {meta['total_nodes']}")
    print(f"  Total Edges:      {meta['total_edges']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
