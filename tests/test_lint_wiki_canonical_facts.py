"""Tests for lint_wiki.load_canonical_facts and schema/canonical-facts.json."""

from __future__ import annotations

import json
import re
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


def test_canonical_facts_schema_has_bounded_valid_regexes() -> None:
    """schema/canonical-facts.json 中所有规则应可编译且不含无界贪婪 .*。"""
    cf = lw.load_canonical_facts()
    assert len(cf) >= 200
    for name, rule in cf.items():
        for key in ("terms", "pos_claims", "neg_claims"):
            for pat in rule.get(key, []):
                re.compile(pat)
                assert not re.search(r"(?<!\\)\.\*", pat), f"{name}.{key} 含无界 .* : {pat}"
