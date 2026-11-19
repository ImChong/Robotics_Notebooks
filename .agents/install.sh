#!/usr/bin/env bash
# Idempotent cloud-environment setup for Robotics_Notebooks.
# Installs Python 3.12 + dev dependencies into a user-space virtualenv (no sudo),
# installs Node.js dev dependencies, unshallows git history if needed, and
# generates gitignored site/graph exports for tests and local previews.
set -euo pipefail

cd "$(dirname "$0")/.."

# 1. Ensure ~/.bashrc is sourced cleanly in non-interactive shells.
touch "$HOME/.bashrc"
sed -i '/case \$- in/,/esac/d' "$HOME/.bashrc" 2>/dev/null || true

# 2. Provision Python 3.12 and install requirements-dev.txt into a user-space venv
#    outside the repository so `git status` stays clean.
VENV_DIR="$HOME/.venvs/robotics-notebooks"
mkdir -p "$HOME/.venvs"

if command -v uv >/dev/null 2>&1; then
  uv python install 3.12
  if [ ! -x "$VENV_DIR/bin/python3" ]; then
    uv venv --python 3.12 --seed "$VENV_DIR"
  fi
  uv pip install --python "$VENV_DIR/bin/python3" -r requirements-dev.txt
else
  if [ ! -x "$VENV_DIR/bin/python3" ]; then
    python3 -m venv "$VENV_DIR"
  fi
  "$VENV_DIR/bin/python3" -m pip install --no-input -r requirements-dev.txt
fi

PATH_LINE='export PATH="$HOME/.venvs/robotics-notebooks/bin:$HOME/.local/bin:$PATH"'
grep -qxF "$PATH_LINE" "$HOME/.bashrc" || echo "$PATH_LINE" >> "$HOME/.bashrc"
export PATH="$VENV_DIR/bin:$HOME/.local/bin:$PATH"

# 3. Install Node.js dev tooling (ESLint, Mermaid validation, frontend tests).
npm ci --no-audit --no-fund

# 4. Ensure full git history is available for `generate_link_graph.py` /
#    `graph_exports_sync.py` (shallow clones fail git-based recency checks).
if [ "$(git rev-parse --is-shallow-repository 2>/dev/null || echo false)" = "true" ]; then
  git fetch --unshallow
fi

# 5. Regenerate gitignored site/graph exports so `make test`, `make ci-test`,
#    and static-site previews work out of the box.
python3 scripts/export_minimal.py
python3 scripts/generate_link_graph.py
python3 scripts/generate_home_stats.py
python3 scripts/graph_exports_sync.py

echo "✅ Robotics_Notebooks cloud environment ready."
