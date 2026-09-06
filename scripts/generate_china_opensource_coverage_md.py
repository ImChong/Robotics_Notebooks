#!/usr/bin/env python3
"""Generate coverage query + overview sections from china-opensource-424-mappings.json."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MAPPINGS = REPO / "exports" / "china-opensource-424-mappings.json"
QUERY_OUT = REPO / "wiki" / "queries" / "china-domestic-opensource-424-coverage.md"
OVERVIEW_OUT = (
    REPO / "wiki" / "overview" / "china-domestic-embodied-opensource-76-companies-technology-map.md"
)

LAYER_ORDER = [
    ("第一层", "第一层 · 整机厂商与各地人形机器人创新中心"),
    ("第二层", "第二层 · 具身模型 / VLA / 世界模型 / 数据与空间智能公司"),
    ("第三层", "第三层 · 灵巧手与触觉传感公司"),
    ("第四层", "第四层 · 大厂与底层平台（互联网 / 云 / 芯片与中间件）"),
    ("第五层", "第五层 · 产业链与移动平台公司（机械臂 / 底盘 / 控制器 / 传感器等）"),
]


def link(slug: str, name: str) -> str:
    return f"[{name}](../entities/{slug}.md)"


def main() -> None:
    data = json.loads(MAPPINGS.read_text(encoding="utf-8"))
    by_layer: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in data:
        by_layer[row["layer"]][row["company"]].append(row)
    new_count = sum(1 for r in data if r["new"])
    reused = len(data) - new_count

    # Coverage query
    lines = [
        "---",
        "title: 国内具身开源全景 424 项 · 本库导读",
        "type: query",
        "status: complete",
        "created: 2026-09-06",
        "updated: 2026-09-06",
        'summary: "把具身智能研究室「76 家机构 · 424 个开源项目」全景清单逐条接到本库独立详情页；复用既有实体，缺口以 cn-os-* 实体补齐。"',
        "sources:",
        "  - ../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md",
        "  - ../../sources/repos/humanoid-motion-intelligence.md",
        "---",
        "",
        "> **Query 产物**：「国内 76 家机构 424 个开源项目，在本库分别对应哪一页？」",
        "",
        "# 国内具身开源全景 424 项 · 本库导读",
        "",
        "## 一句话定义",
        "",
        "把 [国内具身智能开源全景](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)（76 家 · 424 项）接到本库 **独立 `wiki/entities/*` 详情节点**，避免同名仓库在清单与 HMI 主表间迷路。",
        "",
        "## 英文缩写速查",
        "",
        "| 缩写 | 英文全称 | 简要说明 |",
        "|------|----------|----------|",
        "| HMI | Humanoid Motion Intelligence | 具身智能研究室 GitHub 知识库 |",
        "| VLA | Vision-Language-Action | 视觉–语言–动作策略 |",
        "| SDK | Software Development Kit | 真机控制与驱动接口 |",
        "| RL | Reinforcement Learning | 强化学习训练栈 |",
        "",
        "## 规模",
        "",
        f"| 指标 | 数值 |",
        f"|------|------|",
        f"| 清单条目 | {len(data)} |",
        f"| 本库可点开详情 | {len(data)} |",
        f"| 复用既有实体 | {reused} |",
        f"| 本 ingest 新建实体 | {new_count} |",
        "",
        "## 导读总表（按五层格局）",
        "",
    ]
    for layer_key, layer_title in LAYER_ORDER:
        if layer_key not in by_layer:
            continue
        lines.append(f"### {layer_title}")
        lines.append("")
        for company, rows in sorted(by_layer[layer_key].items()):
            org = rows[0]["org_url"].split("、")[0]
            lines.append(f"#### {company}（{len(rows)}）")
            lines.append("")
            lines.append("| 项目 | 类别 | 本库详情 |")
            lines.append("| --- | --- | --- |")
            for r in rows:
                note = "新建" if r["new"] else "复用"
                lines.append(
                    f"| {r['name']} | {r['category']} | {link(r['slug'], r['name'])} · {note} |"
                )
            lines.append("")
        lines.append("")
    lines.extend(
        [
            "## 关联页面",
            "",
            "- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)",
            "- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)",
            "- [HMI 开源项目主表导读](./hmi-opensource-projects-coverage.md)",
            "",
            "## 参考来源",
            "",
            "- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)",
            "- [Humanoid Motion Intelligence（GitHub）](../../sources/repos/humanoid-motion-intelligence.md)",
            "",
            "## 推荐继续阅读",
            "",
            "- [GitHub：人形机器人运动智能知识库](https://github.com/RealXiaoze/humanoid-motion-intelligence)",
        ]
    )
    QUERY_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Overview (shorter)
    olines = [
        "---",
        "type: overview",
        "tags: [overview, survey, china, open-source, embodied-ai, technology-map]",
        "status: complete",
        "updated: 2026-09-06",
        "related:",
        "  - ../entities/humanoid-motion-intelligence.md",
        "  - ../queries/china-domestic-opensource-424-coverage.md",
        "  - ../queries/hmi-opensource-projects-coverage.md",
        "sources:",
        "  - ../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md",
        'summary: "依据具身智能研究室 2026-09-06 公众号「76 家公司、424+ 开源项目」全景，按五层格局组织国内具身开源生态阅读坐标。"',
        "---",
        "",
        "# 国内具身智能开源全景（76 家 · 424 项）",
        "",
        "> **本页定位**：[国内具身智能的开源全景](https://mp.weixin.qq.com/s/L2XQBhesU8EiS2nKM7HErw)（2026-09-06）的阅读坐标；**424/424 独立详情节点**见 [覆盖索引](../queries/china-domestic-opensource-424-coverage.md)。",
        "",
        "## 一句话观点",
        "",
        "**国内具身开源已从单点仓库扩展为「整机全链路 + 模型大脑 + 灵巧手 + 大厂平台 + 产业链 SDK」五层格局；选型应先定层，再进独立实体页核对训练/部署入口。**",
        "",
        "## 英文缩写速查",
        "",
        "| 缩写 | 英文全称 | 简要说明 |",
        "|------|----------|----------|",
        "| OEM | Original Equipment Manufacturer | 整机厂商与各地创新中心 |",
        "| VLA | Vision-Language-Action | 视觉–语言–动作模型 |",
        "| SDK | Software Development Kit | 真机驱动与控制接口 |",
        "| HMI | Humanoid Motion Intelligence | 同源 GitHub 知识库 |",
        "",
        "## 五层格局",
        "",
        "```mermaid",
        "flowchart TB",
        "  L1[① 整机厂商与创新中心<br/>27 家 · 全链路开源]",
        "  L2[② 模型 / VLA / 世界模型 / 数据<br/>21 家 · 大脑与燃料]",
        "  L3[③ 灵巧手与触觉<br/>6 家 · 手部 SDK 与数据]",
        "  L4[④ 大厂与底层平台<br/>6 家 · 模型与工具链卡位]",
        "  L5[⑤ 产业链与移动平台<br/>16 家 · 机械臂/传感器 SDK]",
        "  L1 --> L2",
        "  L2 --> L3",
        "  L1 --> L5",
        "  L4 --> L2",
        "```",
        "",
        "| 层 | 机构数（文内） | 开源特征 | 本库入口 |",
        "|---|---:|---|---|",
    ]
    layer_counts = {k: sum(len(v) for v in d.values()) for k, d in by_layer.items()}
    layer_meta = [
        ("第一层", "27", "本体资产 + RL + 仿真 + 部署 SDK 成套公开"),
        ("第二层", "21", "VLA/世界模型/数据格式立标准"),
        ("第三层", "6", "手部 SDK、仿真、遥操作、触觉数据"),
        ("第四层", "6", "模型层与开发者工具链"),
        ("第五层", "16", "机械臂/相机/雷达 ROS 驱动"),
    ]
    layer_titles = {k: t for k, t in LAYER_ORDER}
    for layer_key, inst, feat in layer_meta:
        cnt = layer_counts.get(layer_key, 0)
        short = layer_titles[layer_key].split("·")[0].strip()
        olines.append(
            f"| {short} | {inst} | {feat} | [{cnt} 项索引](../queries/china-domestic-opensource-424-coverage.md) |"
        )
    olines.extend(
        [
            "",
            f"## 节点策略（本 ingest）",
            "",
            f"- **424/424 独立 `wiki/entities/*` 详情节点**（静态站 `detail.html?id=entity-…`）。",
            f"- **复用 {reused}** 既有实体（Unitree/智元/HMI 主表等已覆盖项）；**新建 {new_count}** `cn-os-*` 实体补齐缺口。",
            "- 与 [HMI 开源项目主表 166 项](./hmi-opensource-projects-coverage.md) **互补**：主表按技术路线深读算法；本全景按 **国内机构** 查仓库入口。",
            "",
            "## 读法建议",
            "",
            "1. **选整机厂** — 从智元/宇树/天工等实体页沿 RL → Sim2Sim → SDK 链路读。",
            "2. **选 VLA/世界模型** — 第二层公司实体 + [VLA](../methods/vla.md)。",
            "3. **查是否已有方法页** — 覆盖索引标注「复用」时优先读原方法/论文页。",
            "",
            "## 关联页面",
            "",
            "- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)",
            "- [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md)",
            "- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)",
            "",
            "## 参考来源",
            "",
            "- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)",
            "",
            "## 推荐继续阅读",
            "",
            "- [GitHub：人形机器人运动智能知识库](https://github.com/RealXiaoze/humanoid-motion-intelligence)",
        ]
    )
    OVERVIEW_OUT.write_text("\n".join(olines) + "\n", encoding="utf-8")
    print(f"Wrote {QUERY_OUT} and {OVERVIEW_OUT}")


if __name__ == "__main__":
    main()
