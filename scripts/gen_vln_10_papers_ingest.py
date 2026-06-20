#!/usr/bin/env python3
"""One-off generator for VLN 10-papers ingest (2026-06-20)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WECHAT_URL = "https://mp.weixin.qq.com/s/2_dYaN6IeWn_vvS_jmGqRQ"
BLOG = "wechat_shenlan_vln_10_papers_survey.md"
RAW = "wechat_vln_10_papers_2026-06-20.md"
CATALOG = "vln_10_papers_catalog.md"

PAPERS = [
    {
        "num": "01",
        "slug": "r2r",
        "short": "R2R",
        "title": "Vision-and-Language Navigation: Interpreting Visually-Grounded Navigation Instructions in Real Environments",
        "cat": "01",
        "cat_name": "01 数据集与仿真平台",
        "venue": "CVPR 2018",
        "arxiv": "1711.07280",
        "inst": "澳大利亚国立大学、阿德莱德大学等",
        "cites": "约 2,258",
        "summary": "提出 R2R 数据集、Matterport3D 导航图仿真与 VLN 评测基准，将任务形式化为「全景序列 + 语言指令 → 逐步动作」。",
        "why": "VLN 领域奠基工作：离散导航图 + Seq2Seq 基线，成为后续几乎所有 benchmark 的基础设施。",
    },
    {
        "num": "02",
        "slug": "vln-ce",
        "short": "VLN-CE",
        "title": "Beyond the Nav-Graph: Vision-and-Language Navigation in Continuous Environments",
        "cat": "01",
        "cat_name": "01 数据集与仿真平台",
        "venue": "ECCV 2020",
        "arxiv": "2004.02857",
        "inst": "俄勒冈州立大学、佐治亚理工学院、Facebook AI Research",
        "cites": "约 622",
        "summary": "基于 Habitat 将 VLN 从离散导航图迁移到连续 3D 环境，智能体以底层前进/转向在物理空间移动。",
        "why": "弥合离散图跳转与真实机器人连续运动之间的差距，提供 R2R 对应的连续环境标准 benchmark。",
    },
    {
        "num": "03",
        "slug": "reverie",
        "short": "REVERIE",
        "title": "REVERIE: Remote Embodied Visual Referring Expression in Real Indoor Environments",
        "cat": "01",
        "cat_name": "01 数据集与仿真平台",
        "venue": "CVPR 2020",
        "arxiv": "1904.10151",
        "inst": "Yuankai Qi, Qi Wu, Peter Anderson, Xin Wang 等",
        "cites": "约 584",
        "summary": "高层指令（目标物体 + 大致位置）下的远程导航与目标定位；结合路径标注与物体边界框。",
        "why": "把 VLN 从「逐步指路」推进到「找到并指认目标」，与视觉 referring 任务深度耦合。",
    },
    {
        "num": "04",
        "slug": "prevalent",
        "short": "PREVALENT",
        "title": "Towards Learning a Generic Agent for Vision-and-Language Navigation via Pre-training",
        "cat": "02",
        "cat_name": "02 算法框架",
        "venue": "CVPR 2020",
        "arxiv": "2002.10638",
        "inst": "杜克大学、微软研究院",
        "cites": "约 438",
        "summary": "首次在 VLN 引入预训练-微调：图像-文本-动作三元组上 Masked LM + 动作预测自监督，再下游微调。",
        "why": "开创 VLN 预训练范式，为 VLN↻BERT、HAMT 等后续工作奠定方法论基础。",
    },
    {
        "num": "05",
        "slug": "vln-bert",
        "short": "VLN↻BERT",
        "title": "VLN↻BERT: A Recurrent Vision-and-Language BERT for Navigation",
        "cat": "02",
        "cat_name": "02 算法框架",
        "venue": "CVPR 2021",
        "arxiv": "2011.13922",
        "inst": "澳大利亚国立大学、阿德莱德大学",
        "cites": "约 483",
        "summary": "在 Transformer 中引入循环状态 token，将历史压缩后与当前全景、指令联合编码以输出下一步动作。",
        "why": "使 BERT 类模型能处理部分可观测的时序导航输入，成为当时多 benchmark SOTA 基线架构。",
    },
    {
        "num": "06",
        "slug": "hamt",
        "short": "HAMT",
        "title": "History Aware Multimodal Transformer for Vision-and-Language Navigation",
        "cat": "02",
        "cat_name": "02 算法框架",
        "venue": "NeurIPS 2021",
        "arxiv": "2110.13309",
        "inst": "法国国家信息与自动化研究所等",
        "cites": "约 455",
        "summary": "层次化 ViT 编码完整历史全景序列，与历史动作、当前观测一同输入跨模态 Transformer。",
        "why": "相对单状态 token 压缩，保留完整历史视觉信息，成为历史感知 VLN 的重要参考。",
    },
    {
        "num": "07",
        "slug": "duet",
        "short": "DUET",
        "title": "Think Global, Act Local: Dual-scale Graph Transformer for Vision-and-Language Navigation",
        "cat": "02",
        "cat_name": "02 算法框架",
        "venue": "CVPR 2022",
        "arxiv": "2202.11742",
        "inst": "法国国家信息与自动化研究所等",
        "cites": "约 369",
        "summary": "在线拓扑建图 + 粗粒度全局规划与细粒度局部动作编码动态融合，双尺度图 Transformer 输出决策。",
        "why": "拓扑建图路线的代表性工作，在 R2R、REVERIE、SOON 等多 benchmark 达到当时 SOTA。",
    },
    {
        "num": "08",
        "slug": "scalevln",
        "short": "ScaleVLN",
        "title": "Scaling Data Generation in Vision-and-Language Navigation",
        "cat": "02",
        "cat_name": "02 算法框架",
        "venue": "ICCV 2023",
        "arxiv": "2307.15644",
        "inst": "澳大利亚国立大学、上海 AI Lab",
        "cites": "约 185",
        "summary": "从 HM3D/Gibson 1200+ 场景自动采样路径、风格迁移与指令生成，构建 490 万指令-轨迹对预训练数据。",
        "why": "证明数据规模扩展可显著抬升 R2R 等 benchmark 上限（文中 R2R SR 约 69%→80%）。",
    },
    {
        "num": "09",
        "slug": "etpnav",
        "short": "ETPNav",
        "title": "ETPNav: Evolving Topological Planning for Vision-Language Navigation in Continuous Environments",
        "cat": "02",
        "cat_name": "02 算法框架",
        "venue": "IEEE TPAMI 2024",
        "arxiv": "2304.03047",
        "inst": "中国科学院大学",
        "cites": "约 231",
        "summary": "连续环境 VLN-CE 下拓扑建图、跨模态规划与底层控制（含避障）三模块串联的端到端框架。",
        "why": "连续环境拓扑规划代表基线，在 R2R-CE、RxR-CE 上报告大幅性能提升。",
    },
    {
        "num": "10",
        "slug": "navid",
        "short": "NaVid",
        "title": "NaVid: Video-based VLM Plans the Next Step for Vision-and-Language Navigation",
        "cat": "02",
        "cat_name": "02 算法框架",
        "venue": "RSS 2024",
        "arxiv": "2402.15852",
        "inst": "北京大学、BAAI 等",
        "cites": "约 324",
        "summary": "将导航历史建模为视频流输入 Vicuna-7B VLM，仅依赖 RGB 视频流输出前进/转向等底层动作。",
        "why": "代表 VLN 与大模型融合方向：无需显式地图/里程计/深度，仿真与 Turtlebot4 真机均验证可行。",
    },
]


def write_paper_source(p: dict) -> None:
    path = ROOT / "sources/papers" / f"vln_survey_{p['num']}_{p['slug'].replace('-', '_')}.md"
    content = f"""# {p["short"]}: {p["title"]}

> 来源归档（ingest · VLN 10 篇盘点 第 {p["num"]}/10）

- **标题：** {p["title"]}
- **类型：** paper
- **VLN 分类：** {p["cat_name"]}（[深蓝具身智能 10 篇编译](../blogs/{BLOG})）
- **机构：** {p["inst"]}
- **出处：** {p["venue"]} · arXiv:{p["arxiv"]}
- **索引来源：** [{BLOG}](../blogs/{BLOG})（<{WECHAT_URL}>）
- **入库日期：** 2026-06-20
- **一句话说明：** {p["summary"]}

## 核心摘录（策展，非全文）

- **在 VLN 地图中的位置：** {p["cat_name"]}，编号 **{p["num"]}/10**。
- **公众号导读要点：** {p["why"]}
- **Google Scholar 引用量（文内）：** {p["cites"]} 次（以原文发表后统计为准，会随时间变化）。

## 对 wiki 的映射

- [paper-vln-{p["num"]}-{p["slug"]}](../../wiki/entities/paper-vln-{p["num"]}-{p["slug"]}.md)
- [vln-10-papers-technology-map](../../wiki/overview/vln-10-papers-technology-map.md)
- [vln-category-{p["cat"]}-{"datasets-platforms" if p["cat"] == "01" else "algorithm-frameworks"}](../../wiki/overview/vln-category-{p["cat"]}-{"datasets-platforms" if p["cat"] == "01" else "algorithm-frameworks"}.md)

## 参考来源（原始）

- 论文：<https://arxiv.org/abs/{p["arxiv"]}>
- 微信公众号编译：[{BLOG}](../blogs/{BLOG})
"""
    path.write_text(content, encoding="utf-8")


def write_paper_entity(p: dict) -> None:
    path = ROOT / "wiki/entities" / f"paper-vln-{p['num']}-{p['slug']}.md"
    cat_hub = (
        "vln-category-01-datasets-platforms"
        if p["cat"] == "01"
        else "vln-category-02-algorithm-frameworks"
    )
    content = f"""---
type: entity
tags: [paper, vln, vln-survey, navigation, embodied-ai]
status: complete
updated: 2026-06-20
arxiv: "{p["arxiv"]}"
summary: "{p["summary"]}"
related:
  - ../overview/vln-10-papers-technology-map.md
  - ../overview/{cat_hub}.md
  - ../tasks/vision-language-navigation.md
sources:
  - ../../sources/papers/vln_survey_{p["num"]}_{p["slug"].replace("-", "_")}.md
  - ../../sources/blogs/{BLOG}
  - ../../sources/papers/{CATALOG}
---

# {p["short"]}

**{p["short"]}** 收录于 [深蓝具身智能 · VLN 10 项代表性研究]({WECHAT_URL}) **第 {p["num"]}/10** 篇，归类为 **{p["cat_name"]}**。本页为知识库 **策展摘要**；方法细节以论文 PDF 为准。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLN | Vision-and-Language Navigation | 依据自然语言指令在环境中导航的具身任务 |
| R2R | Room-to-Room | Matterport3D 上经典逐步导航指令数据集 |
| VLM | Vision-Language Model | 视觉-语言多模态大模型，NaVid 等路线的骨干 |

## 为什么重要

- {p["why"]}
- 在 [VLN 10 篇技术地图](../overview/vln-10-papers-technology-map.md) 中属于 **[{p["cat_name"]}](../overview/{cat_hub}.md)**。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | {p["num"]}/10 |
| 分组 | {p["cat_name"]} |
| 出处 | {p["venue"]} · arXiv:{p["arxiv"]} |
| 机构 | {p["inst"]} |

## 与其他页面的关系

- 技术地图：[vln-10-papers-technology-map.md](../overview/vln-10-papers-technology-map.md)
- 分类 hub：[{cat_hub}.md](../overview/{cat_hub}.md)
- 任务页：[vision-language-navigation.md](../tasks/vision-language-navigation.md)
- 原始 source：[vln_survey_{p["num"]}_{p["slug"].replace("-", "_")}.md](../../sources/papers/vln_survey_{p["num"]}_{p["slug"].replace("-", "_")}.md)

## 实验与评测

- 本页为 **策展索引级** 摘要；量化 benchmark、消融与实机指标以 **原文 PDF** 为准。

## 参考来源

- [vln_survey_{p["num"]}_{p["slug"].replace("-", "_")}.md](../../sources/papers/vln_survey_{p["num"]}_{p["slug"].replace("-", "_")}.md) — VLN 10 篇策展摘录
- [{CATALOG}](../../sources/papers/{CATALOG})
- [{BLOG}](../../sources/blogs/{BLOG})
- 论文：<https://arxiv.org/abs/{p["arxiv"]}>

## 推荐继续阅读

- [VLN 10 篇技术地图](../overview/vln-10-papers-technology-map.md)
- [视觉–语言导航（VLN）](../tasks/vision-language-navigation.md)
"""
    path.write_text(content, encoding="utf-8")


def main() -> None:
    for p in PAPERS:
        write_paper_source(p)
        write_paper_entity(p)
    print(f"Generated {len(PAPERS) * 2} paper files")


if __name__ == "__main__":
    main()
