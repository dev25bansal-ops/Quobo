---
tags: [adr, packaging]
status: accepted
date: 2026-09-20
up: "[[config]]"
---

# ADR-002: Declare pydantic in core dependencies

## Context
`config_schema.py` hard-imports pydantic v2 (lazy-imported by `config.validate_config`),
but pydantic appeared in neither `pyproject.toml` dependencies nor
`constraints.txt` — it only worked because a transitive install happened to
provide 2.13.5. A clean `pip install -r requirements.txt` environment could
break config validation.

## Decision
- `pyproject.toml`: added `"pydantic>=2,<3"` to core dependencies.
- `constraints.txt`: pinned `pydantic==2.13.5` (the version behind the green baseline).
- `ci.yml`: added pydantic to the explicit install list.
- The continuous build's **schema-import step** now guards this class of drift
  permanently ([[Continuous Build]]).

## Consequences
- Fresh environments validate configs correctly; CI parity restored.
- Remaining known schema/code drifts documented, not fixed: `qsvm.max_memory_mb`
  has no consumer; `KNOWN_SUBKEYS` ↔ schema manual sync still required.
