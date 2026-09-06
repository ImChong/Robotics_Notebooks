import copy
import json
from pathlib import Path

import pytest
from check_export_quality import validate_page_exports
from export_minimal import write_json, write_page_exports


def publish(tmp_path, payload):
    source, mirror = tmp_path / "source", tmp_path / "mirror"
    for directory in (source, mirror):
        write_json(directory / "site-data-v1.json", payload)
        write_page_exports(payload, directory)
    return source, mirror


@pytest.fixture
def payload():
    return {
        "version": "v1",
        "pages": {
            "detail_pages": {
                "wiki-methods-π0-policy": {
                    "id": "wiki-methods-π0-policy",
                    "title": "π0",
                    "path": "wiki/methods/π0-policy.md",
                    "summary": "策略摘要",
                    "content_markdown": "# π0\n\n公式 $x$\n```mermaid\ngraph TD\nA-->B\n```",
                },
                "roadmap-motion-control": {
                    "id": "roadmap-motion-control",
                    "title": "路线",
                    "type": "roadmap_page",
                    "content_markdown": "",
                },
            },
            "roadmap_pages": {"roadmap-motion-control": {"stages": [{"title": "入门"}]}},
            "page_aliases": {"old-id": "wiki-methods-π0-policy"},
        },
    }


def test_split_round_trip_preserves_unicode_empty_body_routes_and_aliases(tmp_path, payload):
    before = copy.deepcopy(payload)
    source, mirror = publish(tmp_path, payload)
    assert validate_page_exports(source, mirror) == 2
    assert payload == before
    catalog = json.loads((source / "site-catalog-v1.json").read_text())
    assert "content_markdown" not in json.dumps(catalog)
    assert catalog["pages"]["page_aliases"] == payload["pages"]["page_aliases"]


def test_hashes_are_stable_and_rebuild_prunes_only_generated_bodies(tmp_path, payload):
    source, mirror = publish(tmp_path, payload)
    old = {p.name for p in (source / "page-content").glob("*.json")}
    write_page_exports(payload, source)
    assert old == {p.name for p in (source / "page-content").glob("*.json")}
    unrelated = source / "keep.json"
    unrelated.write_text("keep")
    del payload["pages"]["detail_pages"]["roadmap-motion-control"]
    payload["pages"]["detail_pages"]["wiki-methods-π0-policy"]["content_markdown"] += "\nChanged"
    publish(tmp_path, payload)
    assert validate_page_exports(source, mirror) == 1
    assert not old.intersection(p.name for p in (source / "page-content").glob("*.json"))
    assert unrelated.read_text() == "keep"


@pytest.mark.parametrize("corruption", ["catalog", "body", "missing", "id", "inline", "path"])
def test_gate_rejects_corrupt_or_incomplete_split_exports(tmp_path, payload, corruption):
    source, mirror = publish(tmp_path, payload)
    catalog_path = mirror / "site-catalog-v1.json"
    catalog = json.loads(catalog_path.read_text())
    page = catalog["pages"]["detail_pages"]["wiki-methods-π0-policy"]
    body_path = mirror / page["content_url"].removeprefix("exports/")
    if corruption == "catalog":
        catalog_path.write_text("{}")
    elif corruption == "body":
        body_path.write_text("corrupt")
    elif corruption == "missing":
        body_path.unlink()
    else:
        if corruption == "id":
            del catalog["pages"]["detail_pages"]["roadmap-motion-control"]
        elif corruption == "inline":
            page["content_markdown"] = "leaked body"
        else:
            page["content_url"] = "../outside.json"
        for directory in (source, mirror):
            write_json(directory / "site-catalog-v1.json", catalog)
    with pytest.raises((ValueError, OSError)):
        validate_page_exports(source, mirror)


def test_all_generated_pages_reconstruct_the_legacy_export():
    root = Path(__file__).resolve().parents[1]
    count = validate_page_exports(root / "exports", root / "docs" / "exports")
    assert count > 3000
