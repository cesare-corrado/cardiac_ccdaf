# Nightly tests — deliberately empty

This directory will hold the tests that need real Carto exports, run on a
private mirror with a local runner so the data is never shared. What goes
here, with the numbers each check must reproduce, is specified in
`docs/eam-real-data-verification.md`.

Nothing in `tests/unit/` may depend on external data; everything synthetic
belongs there.
