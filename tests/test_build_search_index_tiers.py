"""两层搜索索引：全文索引写 df、轻量索引只含元数据，文档集合一致。"""

from __future__ import annotations

import math

from build_search_index import LITE_FIELDS, build_search_payloads


def _doc(doc_id: str, body: str) -> dict:
    return {
        "id": doc_id,
        "path": f"wiki/{doc_id}.md",
        "title": doc_id.upper(),
        "summary": f"{doc_id} 摘要",
        "page_type": "wiki_page",
        "tags": ["rl"],
        "search_aliases": [],
        "body": body,
    }


def test_full_index_carries_df_and_lite_index_only_metadata() -> None:
    full, lite = build_search_payloads(
        [_doc("ppo", "ppo clip airtime"), _doc("slam", "slam loop closure")]
    )

    assert full["meta"]["tier"] == "full" and full["meta"]["N"] == 2
    assert "idf" not in full
    assert full["df"]["ppo"] == 1 and full["df"]["rl"] == 2
    # 前端按同一公式现算 idf：ln(1 + (N - df + 0.5) / (df + 0.5))
    assert math.isclose(math.log(1 + (2 - 1 + 0.5) / (1 + 0.5)), 0.6931, rel_tol=1e-3)
    assert full["docs"][0]["tokens"]["airtime"] == 1

    assert lite["meta"]["tier"] == "lite" and lite["meta"]["N"] == 2
    assert [d["id"] for d in lite["docs"]] == ["ppo", "slam"]
    assert all(set(d) == set(LITE_FIELDS) for d in lite["docs"])
    assert "tokens" not in lite["docs"][0] and "dl" not in lite["docs"][0]
    assert lite["docs"][0]["title"] == "PPO" and lite["docs"][0]["tags"] == ["rl"]
