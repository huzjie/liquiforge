# Contributing

1. Keep the project **zero third-party dependencies** (stdlib only).
2. All randomness must be deterministic (use `liquiforge.utils.rng`).
3. Metrics must be computed from a real forward pass, never hardcoded.
4. Add a unit test under `tests/` for new behavior.
5. Run `python -m unittest discover -s tests -t .` before opening a PR.
