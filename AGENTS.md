# Project instructions

This is an early research repository for collaborative training on weak devices.

## Scope and evidence

- Read `README.md`, `docs/design.md`, and `docs/roadmap.md` before extending the protocol.
- Preserve the existing MIT license.
- Do not claim academic novelty, high-end GPU equivalence, energy savings, or practical speedups without evidence.
- Keep mathematical/interface checks separate from end-to-end training and real networking.
- The current dense Jacobian is a correctness oracle, not the low-memory backend.
- Count contract construction, repair, metadata, committed features, coordinator storage, and batch/sequence factors. Do not report only compressed payloads as total savings.
- Keep the data/model epoch fixed during contract reuse. Anchor or basis changes invalidate old node summaries.
- Treat summaries as trusted numerical inputs; this code does not verify malicious devices.
- Record new experimental observations in generated JSON with parameters and explicit scope.

## Development

- Use Python 3.10+ and NumPy; standard-library `unittest` is the current test runner.
- Run `python -m unittest discover -s tests -v` after code changes.
- Run `python -m prairie_fire.experiments --output results/latest.json` when changing mathematical bounds or reference behavior.
- Keep tests tied to mathematical properties, failure cases, or protocol invariants.
- Prefer the next bounded milestone in `docs/roadmap.md` over new orchestration, UI, or agent features.
- Do not overwrite reference results without explaining the changed seed, implementation, or assumptions.
