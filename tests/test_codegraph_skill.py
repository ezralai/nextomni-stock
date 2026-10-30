"""Unit test suite for the NextOmni Stock CodeGraph Skill."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

# Dynamically import build_graph and query_graph from .agents/skills/codegraph/scripts
SCRIPTS_DIR = Path(".agents/skills/codegraph/scripts")

spec_build = importlib.util.spec_from_file_location("build_graph", SCRIPTS_DIR / "build_graph.py")
build_graph = importlib.util.module_from_spec(spec_build)
spec_build.loader.exec_module(build_graph)

spec_query = importlib.util.spec_from_file_location("query_graph", SCRIPTS_DIR / "query_graph.py")
query_graph = importlib.util.module_from_spec(spec_query)
spec_query.loader.exec_module(query_graph)


@pytest.fixture
def sample_graph_data(tmp_path: Path) -> dict:
    """Create a mini synthetic workspace and extract its CodeGraph."""
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()

    # Module A: defines helper and core_calc
    (src_dir / "module_a.py").write_text(
        "def helper(x: float) -> float:\n"
        "    return x * 2.0\n\n"
        "def core_calc(val: float) -> float:\n"
        "    return helper(val) + 1.0\n",
        encoding="utf-8",
    )

    # Module B: calls core_calc
    (src_dir / "module_b.py").write_text(
        "from src.module_a import core_calc\n\n"
        "def orchestrate(data: float) -> float:\n"
        "    return core_calc(data)\n",
        encoding="utf-8",
    )

    # Test file: tests module_a
    (tests_dir / "test_module_a.py").write_text(
        "from src.module_a import core_calc\n\n"
        "def test_core_calc() -> None:\n"
        "    assert core_calc(10.0) == 21.0\n",
        encoding="utf-8",
    )

    return build_graph.extract_repository_graph(tmp_path)


def test_extract_repository_graph_structure(sample_graph_data: dict) -> None:
    """Verify that AST graph extraction correctly identifies nodes and edges."""
    meta = sample_graph_data["metadata"]
    nodes = {n["id"]: n for n in sample_graph_data["nodes"]}
    edges = sample_graph_data["edges"]

    assert meta["file_count"] == 3
    assert meta["function_count"] == 3  # helper, core_calc, orchestrate
    assert meta["test_count"] == 1  # test_core_calc

    # Check function node definitions
    assert "src/module_a.py:helper" in nodes
    assert "src/module_a.py:core_calc" in nodes
    assert "src/module_b.py:orchestrate" in nodes
    assert "tests/test_module_a.py:test_core_calc" in nodes

    # Check CALLS edge from orchestrate -> core_calc
    call_edges = [
        e for e in edges
        if e["type"] == "CALLS" and "orchestrate" in e["source"] and "core_calc" in e["target"]
    ]
    assert len(call_edges) >= 1

    # Check TESTS edge from test_core_calc -> core_calc
    test_edges = [
        e for e in edges
        if e["type"] == "TESTS" and "test_core_calc" in e["source"] and "core_calc" in e["target"]
    ]
    assert len(test_edges) >= 1


def test_query_engine_callers_and_callees(sample_graph_data: dict) -> None:
    """Verify in-memory query engine resolves callers and callees."""
    engine = query_graph.CodeGraphQueryEngine(sample_graph_data)

    # Find callers of core_calc
    callers = engine.get_callers("core_calc")
    caller_ids = [c["caller_id"] for c in callers]
    assert any("orchestrate" in cid for cid in caller_ids)
    assert any("test_core_calc" in cid for cid in caller_ids)

    # Find callees of orchestrate
    callees = engine.get_callees("orchestrate")
    callee_ids = [c["callee_id"] for c in callees]
    assert any("core_calc" in cid for cid in callee_ids)


def test_query_engine_blast_radius(sample_graph_data: dict) -> None:
    """Verify recursive blast radius propagation."""
    engine = query_graph.CodeGraphQueryEngine(sample_graph_data)

    # If helper changes, core_calc, orchestrate, and test_module_a are affected
    res = engine.compute_blast_radius("helper", max_depth=3)
    assert res["total_affected_nodes"] >= 3
    assert "src/module_a.py" in res["affected_files"]
    assert "src/module_b.py" in res["affected_files"]


def test_query_engine_audit_test_coverage(sample_graph_data: dict) -> None:
    """Verify test coverage gap detection for untested functions."""
    engine = query_graph.CodeGraphQueryEngine(sample_graph_data)

    gaps = engine.audit_test_coverage_gaps()
    gap_names = [g["name"] for g in gaps]
    # helper and orchestrate are not directly called by a test
    assert "helper" in gap_names or "orchestrate" in gap_names
    # core_calc is tested, so should NOT be in gaps
    assert "core_calc" not in gap_names


def test_live_repository_build_and_query() -> None:
    """Verify that build_graph and query_graph run successfully on the actual repository."""
    root_dir = Path(".")
    graph = build_graph.extract_repository_graph(root_dir)

    assert graph["metadata"]["file_count"] >= 50
    assert graph["metadata"]["function_count"] >= 200
    assert graph["metadata"]["test_count"] >= 100

    engine = query_graph.CodeGraphQueryEngine(graph)

    # Test caller lookup for central db function
    callers = engine.get_callers("get_observations")
    assert len(callers) >= 5

    # Test blast radius on security validator
    blast = engine.compute_blast_radius("validate_ticker")
    assert len(blast["affected_files"]) >= 5
