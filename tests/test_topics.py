"""图谱主题（社区）：schema/topics.json 注册表、派生优先级、邻居传播与 lint。"""

from __future__ import annotations

from pathlib import Path

import generate_link_graph as glg
import lint_wiki
from utils.community_labels import community_search_aliases_for_path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_registry_has_twenty_unique_topics() -> None:
    ids = [t["id"] for t in glg.TOPICS]
    assert len(ids) == 20
    assert len(set(ids)) == 20


def test_registry_seeds_exist_and_are_unique() -> None:
    seeds = [seed for t in glg.TOPICS for seed in t["seeds"]]
    assert len(seeds) == len(set(seeds))
    for topic in glg.TOPICS:
        assert topic["seeds"], topic["id"]
    for seed in seeds:
        assert (REPO_ROOT / seed).is_file(), seed


def test_registry_tags_are_lowercase_and_owned_by_one_topic() -> None:
    tags = [tag for t in glg.TOPICS for tag in t["tags"]]
    assert len(tags) == len(set(tags))
    assert all(tag == tag.lower() for tag in tags)


def test_parse_frontmatter_topics_scalar_list_and_block() -> None:
    assert glg.parse_frontmatter_topics("---\ntype: entity\ntopic: VLA\n---\n# X\n") == ["vla"]
    assert glg.parse_frontmatter_topics("---\ntopic: [vla, manipulation, vla]\n---\n# X\n") == [
        "vla",
        "manipulation",
    ]
    assert glg.parse_frontmatter_topics("---\ntopic:\n  - sim2real\n---\n# X\n") == ["sim2real"]
    assert glg.parse_frontmatter_topics("---\ntags: [vla]\n---\n# X\n") == []
    assert glg.parse_frontmatter_topics("# no frontmatter\ntopic: vla\n") == []


def test_derive_node_topics_priority() -> None:
    seed = "wiki/tasks/locomotion.md"
    # explicit 覆盖种子与 tags；非法 id 忽略；最多 2 个
    assert glg.derive_node_topics(seed, ["bogus", "vla", "manipulation", "sim2real"], ["rl"]) == (
        ["vla", "manipulation"],
        "explicit",
    )
    # 种子页主主题固定，tags 提供次主题
    assert glg.derive_node_topics(seed, [], ["paper", "rl", "locomotion"]) == (
        ["locomotion", "reinforcement-learning"],
        "seed",
    )
    # tags 按书写顺序：第一个命中为主，第二个不同主题为次
    assert glg.derive_node_topics(
        "wiki/entities/x.md", [], ["paper", "curated-index", "awesome-world-models", "vla"]
    ) == (["world-models", "vla"], "tags")
    assert glg.derive_node_topics("wiki/entities/x.md", [], ["paper", "humanoid"]) == ([], "")


def test_propagate_topics_majority_vote_and_tie_break() -> None:
    adjacency = {
        "a": {"x"},
        "b": {"x"},
        "c": {"x", "y"},
        "x": {"a", "b", "c"},
        "y": {"c", "z"},
        "z": {"y"},
        "lonely": set(),
    }
    assigned = {"a": "vla", "b": "vla", "c": "locomotion"}
    added = glg.propagate_topics(assigned, adjacency)
    assert added["x"] == "vla"
    # y 首轮只有 c（locomotion）一票；z 次轮继承 y
    assert added["y"] == "locomotion"
    assert added["z"] == "locomotion"
    assert "lonely" not in added
    # 平票取注册表靠前的主题（locomotion 在 vla 之前）
    tie = glg.propagate_topics(
        {"a": "vla", "b": "locomotion"}, {"a": {"m"}, "b": {"m"}, "m": {"a", "b"}}
    )
    assert tie["m"] == "locomotion"


def test_assign_communities_uses_topic_ids_and_secondary() -> None:
    nodes = [
        {"id": "wiki/tasks/locomotion.md", "label": "Locomotion", "_tags": ["rl"]},
        {"id": "wiki/entities/p.md", "label": "P", "_tags": ["paper", "vla", "manipulation"]},
        {"id": "wiki/entities/q.md", "label": "Q", "_tags": ["paper"]},
        {"id": "wiki/entities/r.md", "label": "R"},
    ]
    edges = [{"source": "wiki/entities/q.md", "target": "wiki/tasks/locomotion.md"}]
    communities, meta = glg.assign_communities(nodes, edges)
    by_id = {n["id"]: n for n in nodes}

    loco = by_id["wiki/tasks/locomotion.md"]
    assert loco["community"] == "community-locomotion"
    assert loco["community_secondary"] == "community-reinforcement-learning"
    assert by_id["wiki/entities/p.md"]["community"] == "community-vla"
    assert by_id["wiki/entities/p.md"]["community_secondary"] == "community-manipulation"
    assert by_id["wiki/entities/q.md"]["community"] == "community-locomotion"
    assert by_id["wiki/entities/q.md"]["_topic_source"] == "propagated"
    assert "community_secondary" not in by_id["wiki/entities/q.md"]
    assert by_id["wiki/entities/r.md"]["community"] == glg.OTHER_COMMUNITY_ID

    assert meta["community-locomotion"]["size"] == 2
    assert meta["community-locomotion"]["hub_id"] == "wiki/tasks/locomotion.md"
    assert meta["community-locomotion"]["label"] == "运动控制（Locomotion） 社区"
    assert communities[-1]["id"] == glg.OTHER_COMMUNITY_ID
    # 空主题不导出
    assert "community-evaluation" not in meta


def test_lint_flags_invalid_topic(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(lint_wiki, "REPO_ROOT", tmp_path)
    good = tmp_path / "good.md"
    good.write_text("---\ntopic: [vla, manipulation]\n---\n# G\n", encoding="utf-8")
    unknown = tmp_path / "unknown.md"
    unknown.write_text("---\ntopic: robotics\n---\n# U\n", encoding="utf-8")
    too_many = tmp_path / "many.md"
    too_many.write_text("---\ntopic: [vla, manipulation, sim2real]\n---\n# M\n", encoding="utf-8")
    results: dict = {"invalid_topic": []}

    lint_wiki._check_frontmatter_topic([good, unknown, too_many], results)

    flagged = " ".join(results["invalid_topic"])
    assert "good.md" not in flagged
    assert "unknown.md" in flagged
    assert "many.md" in flagged


def test_topic_anchor_page_carries_topic_search_aliases() -> None:
    aliases = community_search_aliases_for_path("wiki/overview/hub-systems-engineering.md")
    assert "系统工程与部署" in aliases
    assert "Systems Engineering and Deployment" in aliases
    # 历史基名别名仍保留
    assert "机器人系统工程" in aliases
