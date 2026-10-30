---
name: codegraph
description: Query repository call graphs, semantic dependencies, blast radius impact analysis, and test coverage gaps across NextOmni Stock before modifying code.
---

# CodeGraph Skill for NextOmni Stock

The **CodeGraph** skill provides structural, semantic graph intelligence over the NextOmni Stock quantitative intelligence codebase. It maps files, classes, functions, CLI commands, database tables, and tests into an interconnected graph $G = (V, E)$.

Use this skill to perform **impact analysis**, verify **blast radius**, and ensure **zero unintended regressions** before making any code changes.

---

## 1. When to Use This Skill

1. **Before Modifying Shared Services:**
   - Always query the graph before editing core components like `src/data/db.py`, `src/security/validators.py`, or `src/features/builder.py`.
2. **Formulating Implementation Proposals:**
   - Use the CodeGraph to populate Section 2 (*"Files and components affected"*) and Section 3 (*"Risks, dependencies, and edge cases"*) of your 6-part proposal with 100% graph-based precision.
3. **Locating Function Callers & Callees:**
   - Determine which CLI commands or Streamlit dashboard tabs invoke an analytics engine.
4. **Auditing Test Coverage & Dead Code:**
   - Identify orphan functions that have no test coverage or are never invoked.

---

## 2. Core Commands & Workflows

### A. Extract / Refresh the CodeGraph
Re-index the entire repository after adding or refactoring files:
```bash
python3 .agents/skills/codegraph/scripts/build_graph.py
```
Outputs `.agents/skills/codegraph/graph.json` containing all nodes and resolved edges.

### B. Trace Blast Radius (Downstream Impact)
Find all files, functions, and tests that depend on a specific symbol or module:
```bash
python3 .agents/skills/codegraph/scripts/query_graph.py --blast-radius get_observations
```
Output reveals:
- Total affected nodes across the repository.
- Complete list of impacted files.
- Exact call path traces from root caller to callee.

### C. Find Inbound Callers
Discover which functions or CLI commands call a specific method:
```bash
python3 .agents/skills/codegraph/scripts/query_graph.py --callers evaluate_optimal_execution_plan
```

### D. Find Outbound Callees
Discover what external modules and functions a target function calls:
```bash
python3 .agents/skills/codegraph/scripts/query_graph.py --callees compute_almgren_chriss_trajectory
```

### E. Audit Test Coverage Gaps
Pinpoint any non-test functions in `src/` that lack direct unit test assertions:
```bash
python3 .agents/skills/codegraph/scripts/query_graph.py --audit-tests
```

### F. Repository Topology Summary
Display total counts of files, classes, functions, tests, and edges:
```bash
python3 .agents/skills/codegraph/scripts/query_graph.py --summary
```

---

## 3. Programmatic Usage in Python

You can also query the graph programmatically from scripts or tests:
```python
from pathlib import Path
from .agents.skills.codegraph.scripts.query_graph import CodeGraphQueryEngine

engine = CodeGraphQueryEngine.load_from_file(Path(".agents/skills/codegraph/graph.json"))
callers = engine.get_callers("validate_ticker")
blast_radius = engine.compute_blast_radius("get_observations")
```

---

## 4. Architectural Rules & Best Practices

- **Zero External Dependencies:** The CodeGraph runs entirely on Python's native `ast` library. Do not introduce external dependencies like Neo4j, Java, or C-extensions.
- **Pre-Change Verification:** Before making any code modifications, query the blast radius to assess potential side effects and update dependent test cases accordingly.
