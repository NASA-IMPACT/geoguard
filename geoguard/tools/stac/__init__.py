"""STAC-catalog tools, organized by catalog and by layer.

- `veda`  — primitives for NASA VEDA's STAC + raster APIs: plain async
  search / inspect / analyze functions with no registry side effects.
- `agent` — the exploration sub-agent that wraps those primitives, and the
  `find_veda_evidence` tool it exposes to the verifier. Importing it
  registers that tool (the package deliberately does NOT import it here,
  so applications opt in per the usual import-to-register convention).

A future catalog (e.g. another STAC deployment) gets its own sibling
module beside `veda`.
"""
