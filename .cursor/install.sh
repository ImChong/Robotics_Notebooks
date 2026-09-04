#!/usr/bin/env bash
# Idempotent Cloud Agent bootstrap for Robotics_Notebooks.
# Safe to run repeatedly (build snapshot + per-agent refresh).
set -euo pipefail

cd "$(dirname "$0")/.."

# 1. System package: pip-audit (make ci-test) needs python3.12-venv, which is
#    not part of the default image.
if ! dpkg -s python3.12-venv >/dev/null 2>&1; then
  sudo apt-get update -qq
  sudo apt-get install -y --no-install-recommends python3.12-venv
fi

# 2. Python dev tooling (ruff, mypy, pytest, pip-audit, radon, pre-commit, ...).
#    Installed to ~/.local/bin, which ~/.profile prepends to PATH for login shells.
#    --break-system-packages only bypasses the PEP 668 guard; --user keeps every
#    package in the user site, never touching the system interpreter.
python3 -m pip install --user --break-system-packages -r requirements-dev.txt

# 3. Node tooling (ESLint for docs/main.js).
npm ci

# 4. Regenerate gitignored site data so `make test` / `make ci-test` and the
#    static-site preview server work out of the box (equivalent to `make export graph`).
python3 scripts/export_minimal.py
python3 scripts/generate_link_graph.py
python3 scripts/generate_home_stats.py
python3 scripts/graph_exports_sync.py

echo "✅ Robotics_Notebooks environment ready."
