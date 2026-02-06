# Codex Working Agreement (Read-first)

## 0. Prime Directive
**Do not modify the repository unless the user explicitly says "EDIT NOW" (exact phrase).**
Until then, operate in **read-only** mode: analysis, proposals, and diffs only.

## 1. Required workflow (always)
When you think a change is needed, you must provide in this order:

1) **Summary** (what & why, 3-7 bullets)  
2) **Scope** (which files, which functions/classes)  
3) **Risk/Trade-offs** (what could break, how to verify)  
4) **Proposed patch** in **git diff** format (no actual edits yet)  
5) Ask for approval: **"Approve this patch? Reply: EDIT NOW"**

## 2. Change size limits
- One patch = **one purpose** (roughly one commit)
- Prefer the smallest possible diff
- Avoid wide refactors; do not rename/move many files at once

## 3. Forbidden without explicit approval
Never do these unless user says **EDIT NOW** and you re-confirm:

- Large refactors (architecture rework, mass renames, folder moves)
- Dependency changes (pip/npm install, requirements/pyproject/package.json edits)
- Formatting across the repo (black/ruff/prettier on many files)
- Deleting files or generating new large files
- Modifying data files (csv/parquet/xlsx) or outputs

## 4. Allowed areas (default)
- You may propose changes to: `src/`, `tests/` (if they exist)
- Do not touch by default: `data/`, `notebooks/`, `outputs/`, `*.csv`, `*.xlsx`

If a change outside allowed areas is necessary, explain why and request approval.

## 5. Verification expectations
For any patch, include at least one of:
- How to run tests (pytest/unittest)
- A minimal reproducible check (command list)
- Expected outputs / acceptance criteria

## 6. If uncertain
Do not guess. Ask a question or provide 2-3 options with pros/cons.
