"""Unit tests for scripts/generate_discovery_picks.py."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import generate_discovery_picks as gdp

BJT = timezone(timedelta(hours=8))


def test_extract_topics_and_domain_filter() -> None:
    topics, score = gdp.extract_topics(
        "Unitree G1 Whole-Body Control with VLA and Dexterous Manipulation",
        ["humanoid", "sim2real"],
    )
    assert "Humanoid" in topics
    assert "VLA" in topics
    assert "WBC" in topics
    assert score > 8.0

    neg_topics, neg_score = gdp.extract_topics(
        "Pretrained Tabular Features for Financial Portfolio Prediction"
    )
    assert neg_topics == []
    assert neg_score < 0.0


def test_parse_arxiv_atom_extracts_dates_and_links() -> None:
    today = date(2026, 11, 27)
    xml = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2611.99999v1</id>
    <published>2026-11-26T12:00:00Z</published>
    <updated>2026-11-26T12:00:00Z</updated>
    <title>DexHand-VLA: Dexterous Robot Manipulation via Action Chunking</title>
    <summary>We propose a dexterous robot manipulation policy. Code at https://github.com/testlab/dexhand-vla and website https://testlab.github.io/dexhand.</summary>
    <author><name>Alice Zhang</name></author>
    <author><name>Bob Li</name></author>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2611.00001v1</id>
    <published>2026-11-01T12:00:00Z</published>
    <updated>2026-11-01T12:00:00Z</updated>
    <title>Old Humanoid Robot Paper</title>
    <summary>Older than 7 days.</summary>
    <author><name>Carol</name></author>
  </entry>
</feed>
"""
    items = gdp.parse_arxiv_atom(xml, today, max_age_days=7)
    assert len(items) == 1
    item = items[0]
    assert item.arxiv_id == "2611.99999"
    assert item.date == "2026-11-26"
    assert item.date_type == "published"
    assert item.github_repo == "testlab/dexhand-vla"
    assert item.links["code"] == "https://github.com/testlab/dexhand-vla"
    assert item.links["project"] == "https://testlab.github.io/dexhand"


def test_parse_upstar_html_extracts_ranked_repos_and_star_deltas() -> None:
    html = """
    <html><head>
      <meta name="description" content="Robotics on GitHub, Nov 27, 2026: acme/humanoid-wbc +42 stars, lab/vla-sim +18.">
      <script type="application/ld+json">
        {"@context":"https://schema.org","@type":"CollectionPage","mainEntity":{"@type":"ItemList","itemListElement":[
          {"@type":"ListItem","name":"acme/humanoid-wbc","position":1},
          {"@type":"ListItem","name":"lab/vla-sim","position":2}
        ]}}
      </script>
    </head></html>
    """
    ranked, deltas = gdp.parse_upstar_html(html, "day")
    assert ranked == [(1, "acme/humanoid-wbc"), (2, "lab/vla-sim")]
    assert deltas["acme/humanoid-wbc"] == 42
    assert deltas["lab/vla-sim"] == 18


def test_merge_dedupe_kb_match_and_window_expansion(tmp_path: Path) -> None:
    entities_dir = tmp_path / "wiki" / "entities"
    entities_dir.mkdir(parents=True)
    (entities_dir / "paper-dexhand.md").write_text(
        """---
type: entity
project_id: dexhand-vla
arxiv: "2611.99999"
code: https://github.com/testlab/dexhand-vla
summary: "面向灵巧操作的多模态策略框架。"
---

# DexHand-VLA｜灵巧手操作 VLA 策略
""",
        encoding="utf-8",
    )
    kb_index = gdp.build_kb_index(tmp_path)
    today = date(2026, 11, 27)

    paper_item = gdp.CandidateItem(
        item_id="arxiv-2611.99999",
        kind="paper",
        title_en="DexHand-VLA: Dexterous Robot Manipulation",
        date="2026-11-26",
        date_type="published",
        team="TestLab",
        contribution_en="A dexterous manipulation policy.",
        topics=["VLA", "Dexterous"],
        links={"paper": "https://arxiv.org/abs/2611.99999"},
        signals=[
            {
                "source_id": "hf-daily-papers",
                "source_name": "Hugging Face Daily Papers",
                "label_zh": "HF Daily Papers 50 赞",
                "label_en": "HF Daily Papers 50 upvotes",
                "observed_at": "2026-11-27",
            }
        ],
        arxiv_id="2611.99999",
        github_repo="testlab/dexhand-vla",
        score=12.0,
    )
    repo_item = gdp.CandidateItem(
        item_id="repo-testlab-dexhand-vla",
        kind="project",
        title_en="testlab/dexhand-vla",
        date="2026-11-27",
        date_type="updated",
        team="testlab",
        contribution_en="Short desc",
        topics=["Dexterous"],
        links={"code": "https://github.com/testlab/dexhand-vla"},
        signals=[
            {
                "source_id": "upstar-robotics-day",
                "source_name": "Upstar Robotics 24h Rising",
                "label_zh": "Upstar 24h 榜 #1",
                "label_en": "Upstar 24h #1",
                "observed_at": "2026-11-27",
            }
        ],
        github_repo="testlab/dexhand-vla",
        score=9.0,
    )
    older_item = gdp.CandidateItem(
        item_id="arxiv-2611.88888",
        kind="paper",
        title_en="Humanoid Whole-Body Control Benchmark",
        date="2026-11-22",
        date_type="published",
        team="LabB",
        contribution_en="Benchmark for humanoid WBC.",
        topics=["Humanoid", "WBC"],
        links={"paper": "https://arxiv.org/abs/2611.88888"},
        signals=[],
        arxiv_id="2611.88888",
        score=8.5,
    )

    merged = gdp.merge_and_dedupe([paper_item, repo_item, older_item], kb_index)
    assert len(merged) == 2

    selected, win_meta = gdp.select_picks(merged, today, limit=10, window_days=7)
    assert len(selected) == 2
    assert win_meta["expanded"] is True
    assert win_meta["count_24h"] == 1
    assert win_meta["count_7d"] == 1

    first = selected[0]
    assert first["id"] == "arxiv-2611.99999"
    assert first["date_type"] == "published"
    assert first["links"]["code"] == "https://github.com/testlab/dexhand-vla"
    assert first["kb_node"]["detail_id"] == "entity-paper-dexhand"
    assert first["title_zh"] == "灵巧手操作 VLA 策略"
    assert len(first["signals"]) == 2


def test_generate_picks_payload_preserves_snapshot_on_total_failure(tmp_path: Path) -> None:
    out_file = tmp_path / "discovery-picks.json"
    previous = {
        "schema_version": 1,
        "updated_at": "2026-11-26T09:00:00+08:00",
        "updated_date_bjt": "2026-11-26",
        "status": {"state": "ok"},
        "window": {"primary_hours": 24, "expanded": False, "max_days": 7},
        "items": [{"id": "cached-item", "kind": "paper", "title_en": "Cached"}],
    }
    out_file.write_text(json.dumps(previous), encoding="utf-8")

    def failing_fetch(_url: str) -> str:
        raise RuntimeError("network offline")

    payload = gdp.generate_picks_payload(
        repo_root=tmp_path,
        output_path=out_file,
        now_bjt=datetime(2026, 11, 27, 9, 0, tzinfo=BJT),
        fetch_text=failing_fetch,
        enrich_github=False,
    )
    assert payload["status"]["state"] == "stale"
    assert len(payload["status"]["sources_failed"]) == 6
    assert payload["updated_date_bjt"] == "2026-11-26"
    assert payload["items"] == previous["items"]


def test_generate_picks_payload_partial_status_and_emoji_stripping(tmp_path: Path) -> None:
    out_file = tmp_path / "discovery-picks.json"

    def partial_fetch(url: str) -> str:
        if "huggingface.co/api/daily_papers" in url:
            return json.dumps(
                [
                    {
                        "title": "Embodied World-Action Policy",
                        "publishedAt": "2026-11-27T01:00:00.000Z",
                        "paper": {
                            "id": "2611.77777",
                            "upvotes": 42,
                            "ai_summary": "🔴 New robot manipulation policy.",
                            "ai_keywords": ["robotics", "vla"],
                        },
                    }
                ]
            )
        raise RuntimeError("upstream unavailable")

    payload = gdp.generate_picks_payload(
        repo_root=tmp_path,
        output_path=out_file,
        now_bjt=datetime(2026, 11, 27, 9, 0, tzinfo=BJT),
        fetch_text=partial_fetch,
        enrich_github=False,
    )
    assert payload["status"]["state"] == "partial"
    assert "hf-daily-papers" not in payload["status"]["sources_failed"]
    assert "arxiv-cs-ro" in payload["status"]["sources_failed"]
    assert len(payload["items"]) == 1
    assert "🔴" not in payload["items"][0]["contribution_en"]
