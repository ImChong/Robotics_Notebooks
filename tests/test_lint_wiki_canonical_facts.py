"""Tests for lint_wiki.load_canonical_facts."""

from __future__ import annotations

import json
from pathlib import Path

import lint_wiki as lw


def test_load_canonical_facts_returns_empty_when_file_missing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(lw, "CANONICAL_FACTS_FILE", tmp_path / "missing.json")
    assert lw.load_canonical_facts() == {}


def test_load_canonical_facts_reads_json(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "cf.json"
    path.write_text(json.dumps({"rules": []}), encoding="utf-8")
    monkeypatch.setattr(lw, "CANONICAL_FACTS_FILE", path)
    assert lw.load_canonical_facts() == {"rules": []}
