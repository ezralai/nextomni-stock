---
name: codegraph
description: Query repository call graphs, semantic dependencies, database table dataflow, blast radius impact analysis, and test coverage gaps across NextOmni Stock before modifying code.
---

# CodeGraph Skill for NextOmni Stock

The **CodeGraph** skill provides structural, semantic graph intelligence over the NextOmni Stock quantitative intelligence codebase. It maps files, classes, functions, CLI commands, database tables, and tests into an interconnected directed graph $G = (V, E)$.

Use this skill to perform **impact analysis**, verify **blast radius**, audit **database dataflow**, detect **dead code**, and ensure **zero unintended regressions** before making any code changes.

---

## 1. When to Use This Skill

1. **Before Modifying Shared Services:**
   - Always query the graph before editing core components like `src/data/db.py`, `src/security/validators.py`, or `src/features/builder.py`.
2. **Formulating Implementation Proposals:**
   - Use the CodeGraph to populate Section 2 (*"Files and components affected"*) and Section 3 (*"Risks, dependencies, and edge cases"*) of your 6-part proposal with 100% graph-based precision.
3. **Database Schema Audits & Migrations:**
   - Trace which functions read from or write to a specific SQLite table using `--dataflow <table>`.
4. **Locating Function Callers & Interface Wiring:**
   - Determine which CLI commands (`DISPATCHES_TO`) or Streamlit tabs invoke an analytics engine.
5. **Auditing Test Coverage & Pruning Dead Code:**
   - Use `--dead-code` to identify uncalled functions and `--suggest-tests` to generate targeted test fixtures.

---

## 2. Core Commands & Workflows

### A. Extract / Refresh the CodeGraph (Incremental)
Re-index the repository using SHA-256 hash caching:
```bash
python3 .agents/skills/codegraph/scripts/build_graph.py
```
*Force full re-parsing without cache:*
```bash
python3 .agents/skills/codegraph/scripts/build_graph.py --force
```

### B. Trace Blast Radius (Downstream Impact)
Find all files, functions, and tests that depend on a specific symbol or module:
```bash
python3 .agents/skills/codegraph/scripts/query_graph.py --blast-radius get_observations
```
Output reveals:
- Total affected nodes across the repository.
- Complete list of impacted files.
- Exact call path traces from root caller to callee.

### C. Trace Database Table Dataflow
Find every function that performs `SELECT`, `INSERT`, `UPDATE`, or `DELETE` on a table:
```bash
python3 .agents/skills/codegraph/scripts/query_graph.py --dataflow stock_observations
python3 .agents/skills/codegraph/scripts/query_graph.py --dataflow predictions
```

### D. Find Inbound Callers & Outbound Callees
```bash
# Inbound callers
python3 .agents/skills/codegraph/scripts/query_graph.py --callers evaluate_optimal_execution_plan

# Outbound callees
python3 .agents/skills/codegraph/scripts/query_graph.py --callees compute_almgren_chriss_trajectory
```

### E. Audit Dead Code Candidates
Find functions in `src/` that have zero callers and zero tests:
```bash
python3 .agents/skills/codegraph/scripts/query_graph.py --dead-code
```

### F. Suggest Missing Tests
Identify untested functions and view recommended test filenames and test methods:
```bash
python3 .agents/skills/codegraph/scripts/query_graph.py --suggest-tests
```

### G. Repository Topology Summary
Display total counts of files, classes, functions, database tables, CLI commands, and edges:
```bash
python3 .agents/skills/codegraph/scripts/query_graph.py --summary
```

---

## 3. Programmatic Usage in Python

```python
from pathlib import Path
from .agents.skills.codegraph.scripts.query_graph import CodeGraphQueryEngine

engine = CodeGraphQueryEngine.load_from_file(Path(".agents/skills/codegraph/graph.json"))
callers = engine.get_callers("validate_ticker")
blast_radius = engine.compute_blast_radius("get_observations")
dataflow = engine.get_table_dataflow("stock_observations")
dead_code = engine.find_dead_code_candidates()
```

---

## 4. Architectural Rules & Best Practices

- **Zero External Dependencies:** Built strictly on Python standard library modules (`ast`, `pathlib`, `json`, `hashlib`, `re`).
- **Mandatory Pre-Change Check:** Every AI coding cycle must execute `--blast-radius` before modifying existing functions to ensure 100% regression avoidance.
