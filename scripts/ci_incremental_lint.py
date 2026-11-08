#!/usr/bin/env python3
"""Targeted wiki validation for content-only ingest pull requests.

Reuse the production lint rules on changed and affected pages. Global identity
checks remain in check_project_nodes.py; full lint/preflight still run on main.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import lint_wiki as wiki

ROOT = Path(__file__).resolve().parents[1]

# Checks in lint_wiki.lint() deliberately not run here, with the reason. A test keeps this
# in sync with lint(), so a new full-lint check cannot silently skip the incremental path.
FULL_LINT_ONLY_CHECKS = {
    "_check_missing_concepts": "info only",
    "_check_missing_concept_pages": "info only",
    "_check_log_activity": "repository activity; an ingest PR can only add activity",
    "_check_readme_badges": "README.md is outside the ingest scope (changes run full CI)",
    "_check_graph_orphans": "needs generated graph-stats; per-page orphan check covers affected",
    "_check_methods_without_practitioner_query": "info only",
    "_check_stale_claims": "info only",
    "_check_physics_concept_crosslink": "info only",
    "_check_contact_control_crosslink": "info only",
    "_check_embodied_fm_crosslink": "info only",
    "_check_eval_benchmark_crosslink": "info only",
    "_check_actuator_drive_chain_crosslink": "info only",
    "_check_perception_stack_crosslink": "info only",
}


def old_links(path: str, base: str) -> set[Path]:
    if not base:
        return set()
    result = subprocess.run(
        ["git", "show", f"{base}:{path}"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        return set()  # New page; old path of a rename is included in the scope.
    return set(wiki.extract_internal_links(result.stdout, ROOT / path))


def affected_pages(scope: dict, pages: list[Path], broken: dict) -> list[Path]:
    page_set = {path.resolve() for path in pages}
    changed = set(scope["changed_wiki"])
    affected = {
        (ROOT / path).resolve()
        for path in changed
        if (ROOT / path).is_file() and (ROOT / path).resolve() in page_set
    }

    # Links removed from an edited/deleted page can orphan their old targets.
    for name in changed:
        for target in old_links(name, scope["base"]):
            if target in page_set:
                affected.add(target)
        current = ROOT / name
        if current.is_file():
            for target in wiki.extract_internal_links(current.read_text(encoding="utf-8"), current):
                if target in page_set:
                    affected.add(target)

    removed_wiki = {
        (ROOT / name).resolve() for name in scope["removed"] if name.startswith("wiki/")
    }
    changed_sources = {(ROOT / name).resolve() for name in scope["changed_sources"]}
    if removed_wiki or changed_sources:
        for page in pages:
            content = page.read_text(encoding="utf-8")
            # Revalidate pages whose original: archive or Markdown links changed.
            targets = set(wiki.extract_internal_links(content, page))
            if any(
                (page.parent / entry).resolve() in changed_sources
                for entry in wiki.original_entries(content)
            ):
                affected.add(page.resolve())
            if targets & (removed_wiki | changed_sources):
                affected.add(page.resolve())

    # Check all pages linking to newly deleted wiki targets (including broken links).
    for page, targets in broken.items():
        if any(
            (ROOT / name).resolve() in removed_wiki for name in targets if name.startswith("wiki/")
        ):
            affected.add(page)
    return sorted(path for path in affected if path in page_set)


def check_source_links(scope: dict, results: dict) -> None:
    """New/edited source archives may not introduce dangling wiki links."""
    for name in scope["changed_sources"]:
        path = ROOT / name
        if not path.is_file() or not name.startswith("sources/papers/"):
            continue
        content = path.read_text(encoding="utf-8")
        for match in re.finditer(r"\]\(([^)]*wiki/[^)]+\.md)\)", content):
            href = match.group(1).split("#")[0]
            if not (path.parent / href).resolve().is_file():
                results["sources_orphans"].append(f"{name} → {href}")


def contradiction_fact_ids(scope: dict) -> set[str]:
    """Canonical facts a changed page mentions: only these can gain a new contradiction.

    Deleted pages are skipped because removing a page can only remove claims.
    """
    facts = wiki.load_canonical_facts()
    hit: set[str] = set()
    for name in scope["changed_wiki"]:
        path = ROOT / name
        if not path.is_file():
            continue
        content = wiki.strip_misconception_sections(path.read_text(encoding="utf-8"))
        hit.update(
            fact_id
            for fact_id, fact in facts.items()
            if all(re.search(term, content, re.IGNORECASE) for term in fact["terms"])
        )
    return hit


def lint_changed(scope: dict) -> dict:
    if scope["mode"] != "ingest" or not scope["base"]:
        raise ValueError("incremental lint requires a detected PR ingest scope")
    pages = wiki.get_wiki_pages()
    page_set = {page.resolve() for page in pages}
    inbound, broken = wiki._build_link_index(pages, page_set)
    selected = affected_pages(scope, pages, broken)
    results = wiki._empty_results()
    print(
        f"Incremental wiki lint: {len(selected)} affected pages "
        f"from {len(scope['paths'])} changed paths",
        flush=True,
    )
    wiki._check_per_page(selected, inbound, broken, results)
    wiki._check_frontmatter(selected, results)
    wiki._check_original_reading(selected, results)
    wiki._check_query_format(selected, results)
    wiki._check_formalizations(selected, results)
    wiki._check_methods_entities(selected, results)
    wiki._check_paper_entity_metadata(selected, results)
    wiki._check_dataset_entity_metadata(selected, results)
    wiki._check_tool_institutions(selected, results)
    wiki._check_frontmatter_topic(selected, results)

    # Full-index invariants are cheap relative to full lint and must not be lost.
    wiki._check_duplicate_arxiv(pages, results)
    # Contradictions are cross-page: compare all pages, but only for facts a changed page hits.
    wiki._check_contradictions(pages, results, contradiction_fact_ids(scope))
    if any(name.startswith("wiki/") for name in scope["removed"]):
        wiki._check_sources_health(results)
    check_source_links(scope, results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope-file", type=Path, required=True)
    args = parser.parse_args()
    scope = json.loads(args.scope_file.read_text(encoding="utf-8"))
    results = lint_changed(scope)
    print(wiki.format_report(results))
    failures = wiki._failing_total(results)
    print(f"Incremental wiki lint: {failures} blocking issues")
    if failures:
        parser.exit(1, f"{failures} incremental lint issues found\n")


if __name__ == "__main__":
    main()
