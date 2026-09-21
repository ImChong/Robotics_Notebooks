#!/usr/bin/env python3
"""Generator for 具身小站 contact-rich simulation 10-papers ingest (2026-09-21)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WECHAT_URL = "https://mp.weixin.qq.com/s/OCz5OShnrRSeSNb8dcmXDg"
BLOG = "wechat_embodied_station_contact_rich_sim_10_papers_2026-09-21.md"
RAW = "wechat_embodied_station_contact_rich_sim_10_papers_2026-09-21.md"
MAP = "contact-rich-sim-10-papers-technology-map"

PAPERS = [
    {
        "num": "01",
        "slug": "rapid-vlm-rl",
        "short": "RAPID",
        "title": "Scaling Vision-Language Reward Learning for Robot Manipulation in Parallel Simulation",
        "arxiv": "2609.21767",
        "tier": "深读",
        "oss": "已开源",
        "code": "https://github.com/rapid-vlm/rapid-vlm-rl",
        "site": None,
        "inst": "（待论文正式披露）",
        "tags": ["paper", "vla", "rl", "simulation", "reward-learning"],
        "summary": "GPU 并行 rollout + 单请求偏好标注 + 代表性图像采样 + 自动奖励稳定化；5 个 IsaacLab Franka 任务平均训练 9.18h→3.13h，全组件 1.15h、896 次 API 调用。",
        "why": "VLM 奖励 RL 的瓶颈常在标注吞吐而非策略网络；RAPID 把 rollout、标注、采样与更新统一调度。",
        "source_file": "rapid_vlm_rl_arxiv_2609_21767.md",
        "repo_file": "rapid-vlm-rl.md",
        "seq": True,
    },
    {
        "num": "02",
        "slug": "gala",
        "short": "GALA",
        "title": "GALA: Geometry-Aware Latent Action Modeling for Vision-Language-Action Model Pretraining across Embodiments",
        "arxiv": "2609.21948",
        "tier": "跟进",
        "oss": "待发布",
        "code": None,
        "site": "https://puzhenyuan.github.io/GALA-website/",
        "inst": "清华大学；上海期智研究院",
        "tags": ["paper", "vla", "latent-action", "cross-embodiment", "manipulation"],
        "summary": "UEMR 把 3D 末端几何运动纳入 latent action 预训练；RoboCasa-GR1 68.3%、真机四任务平均 75.5%，移除 UEMR 后真机降至 68.5%。",
        "why": "跨形态 VLA 预训练时图像 latent 常丢手指/夹爪细粒度几何；GALA 用 UEMR 对齐几何与场景动态。",
        "source_file": "gala_arxiv_2609_21948.md",
        "repo_file": None,
        "seq": False,
    },
    {
        "num": "03",
        "slug": "crisp",
        "short": "CRISP",
        "title": "CRISP: Contact-Rich Robotic Simulation Platform with Extensive Geometries and Contact Solvers",
        "arxiv": "2609.21761",
        "tier": "跟进",
        "oss": "已开源",
        "code": "https://github.com/INRoL/crisp",
        "site": "https://inrol.github.io/crisp/",
        "inst": "（待论文正式披露）",
        "tags": ["paper", "simulation", "contact-rich", "manipulation", "physics-engine"],
        "summary": "统一 mesh/SDF 几何 + CANAL/SubADMM 接触求解；50–200 μm peg-in-hole 与 bolt-nut 装配平均穿透深度低于 MuJoCo/Isaac Sim 对照。",
        "why": "插入/螺纹装配对碰撞几何与接触约束双敏感；CRISP 把几何表示与求解器放进同一平台。",
        "source_file": "crisp_arxiv_2609_21761.md",
        "repo_file": "crisp.md",
        "seq": True,
    },
    {
        "num": "04",
        "slug": "seeq",
        "short": "SeeQ",
        "title": "SeeQ: Training Generalist Value Functions for Long-Horizon Robotic Manipulation",
        "arxiv": "2609.22085",
        "tier": "扫读",
        "oss": "待发布",
        "code": None,
        "site": "https://saksham002.github.io/seeq/",
        "inst": "卡内基梅隆大学（CMU）",
        "tags": ["paper", "value-function", "long-horizon", "manipulation", "bimanual"],
        "summary": "Q 函数先预测当前子任务再估短时域 TD 价值，对 π₀.₅ base policy 做 best-of-N 排序；四真机双臂任务 24 次试验均有提升。",
        "why": "稀疏任务级奖励下直接学 Q 需长时域信用分配；SeeQ 把价值学习拆到活跃子任务。",
        "source_file": "seeq_arxiv_2609_22085.md",
        "repo_file": None,
        "seq": False,
    },
    {
        "num": "05",
        "slug": "skelwam",
        "short": "SkelWAM",
        "title": "SkelWAM: A Skeleton-Guided World-Action Model for Zero-Shot Cross-Embodiment Manipulation",
        "arxiv": "2609.21983",
        "tier": "扫读",
        "oss": "待发布",
        "code": None,
        "site": "http://www.liukepku.com/skelwam/index.html",
        "inst": "北京大学；清华大学；Feagine",
        "tags": ["paper", "wam", "cross-embodiment", "manipulation", "zero-shot"],
        "summary": "25-D skeleton state（中心线 + TCP + 夹爪）跨形态共享；LIBERO-Cross10 零样本 43.3%，相对最佳 baseline +36.2 pp。",
        "why": "跨形态迁移时视觉与动作几何同时变；SkelWAM 用显式骨架几何连接感知与控制。",
        "source_file": "skelwam_arxiv_2609_21983.md",
        "repo_file": None,
        "seq": False,
    },
    {
        "num": "06",
        "slug": "parts",
        "short": "PARTS",
        "title": "From Pretraining to Proficiency: Real-World Subtask RL for Long-Horizon Manipulation with Minimal Human Intervention",
        "arxiv": "2609.21788",
        "tier": "扫读",
        "oss": "待发布",
        "code": None,
        "site": "https://destiny000621.github.io/PARTS/",
        "inst": "德克萨斯大学奥斯汀分校；Autel US；加州大学伯克利分校",
        "tags": ["paper", "rl", "long-horizon", "manipulation", "residual-policy"],
        "summary": "冻结预训练策略，仅在抓取/插入等瓶颈子任务学 residual correction；YAM 32%→61%，Franka 50%→95%。",
        "why": "长时程任务常卡在少数瓶颈；PARTS 把真机 RL 预算集中到这些子任务。",
        "source_file": "parts_arxiv_2609_21788.md",
        "repo_file": None,
        "seq": False,
    },
    {
        "num": "07",
        "slug": "carf",
        "short": "CARF",
        "title": "CARF: Contrastive Attraction-Repulsion of Failure-Guided Flow Matching",
        "arxiv": "2609.21982",
        "tier": "扫读",
        "oss": "待发布",
        "code": None,
        "site": "https://zhao-sq.github.io/carf/",
        "inst": "加州大学伯克利分校",
        "tags": ["paper", "imitation-learning", "flow-matching", "manipulation", "failure-data"],
        "summary": "progress scorer 区分推进片段与失败关键段，flow matching 对前者吸引、对后者排斥；仿真与真机均优于 success-only/all-data IL。",
        "why": "失败轨迹含异质片段，粗粒度丢弃或全量模仿都浪费；CARF 显式分离有用与有害段。",
        "source_file": "carf_arxiv_2609_21982.md",
        "repo_file": None,
        "seq": False,
    },
    {
        "num": "08",
        "slug": "psr-vla",
        "short": "PSR-VLA",
        "title": "PSR: Predictive Sensorimotor Representation Learning for Contact-Rich Manipulation",
        "arxiv": "2609.21753",
        "tier": "扫读",
        "oss": "待发布",
        "code": None,
        "site": "https://psr-vla.pages.dev/",
        "inst": "（匿名投稿，待正式披露）",
        "tags": ["paper", "vla", "contact-rich", "force-torque", "manipulation"],
        "summary": "从视觉/力矩/本体感知历史预测未来接触动力学，六层 sensorimotor hierarchy 注入 VLA；6 真机接触任务总体 91.7%，相对力觉基线 +19.2–30.0 pp。",
        "why": "接触丰富任务需提前感知接触如何演化；PSR 用预测式 sensorimotor 表征桥接物理历史与动作生成。",
        "source_file": "psr_vla_arxiv_2609_21753.md",
        "repo_file": None,
        "seq": False,
    },
    {
        "num": "09",
        "slug": "wm-compositional-cl-benchmark",
        "short": "Compositional CL WM Benchmark",
        "title": "Benchmarking World Models for Continual Learning on Compositional Tasks",
        "arxiv": "2609.22055",
        "tier": "扫读",
        "oss": "待发布",
        "code": None,
        "site": "https://object814.github.io/Compositional-Continual-Learning/",
        "inst": "牛津大学",
        "tags": ["paper", "world-model", "continual-learning", "benchmark", "manipulation"],
        "summary": "把 world model 持续学习拆成 action/perception 组合轴，用重组已见 primitives 的任务序列单独测 reuse 与 forgetting；模块化 PWM 更平衡但仍未完全解决。",
        "why": "现有 CL benchmark 常把复用与快速学习新交互混在一起；该工作用组合任务末端隔离知识复用。",
        "source_file": "wm_compositional_cl_benchmark_arxiv_2609_22055.md",
        "repo_file": None,
        "seq": False,
    },
    {
        "num": "10",
        "slug": "sim2real-chunk-vla-pipeline",
        "short": "Sim2Real Chunk VLA Pipeline",
        "title": "A Sim-to-Real Integration Pipeline for Training and Deployment of Chunk-Based VLA Manipulation Policies",
        "arxiv": "2609.21817",
        "tier": "扫读",
        "oss": "已开源",
        "code": "https://gitlab.isir.upmc.fr/kappel/sim2real_public_chunk_control",
        "site": None,
        "inst": "索邦大学 ISIR（UPMC）",
        "tags": ["paper", "sim2real", "vla", "deployment", "manipulation"],
        "summary": "仿真生成 expert trajectories，真机 Franka FR3 open-loop replay 记录视觉/本体感知，同一部署栈闭环评估 chunk VLA；配对数据直接量化 sim-to-real gap。",
        "why": "chunk VLA 的 sim-to-real gap 缺少统一硬件闭环测量；该协议用同一栈贯通训练与部署评测。",
        "source_file": "sim2real_chunk_vla_pipeline_arxiv_2609_21817.md",
        "repo_file": "sim2real-public-chunk-control.md",
        "seq": True,
    },
]


def tag_yaml(tags: list[str]) -> str:
    return "\n".join(f"  - {t}" for t in tags)


def write_paper_source(p: dict) -> None:
    path = ROOT / "sources/papers" / p["source_file"]
    code_line = f"- **代码：** {p['code']}\n" if p["code"] else ""
    site_line = f"- **项目页：** {p['site']}\n" if p["site"] else ""
    content = f"""# {p["short"]}（arXiv:{p["arxiv"]}）

> 来源归档（paper）

- **标题：** {p["title"]}
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/{p["arxiv"]}>
- **PDF：** <https://arxiv.org/pdf/{p["arxiv"]}>
{site_line}{code_line}- **入库日期：** 2026-09-21
- **一句话说明：** {p["summary"]}

## 开源状态

- **{p["oss"]}**（步骤 2.5 核查，2026-09-21）
- 项目页/arXiv 截至入库日的可运行代码链接核查结论见上。

## 核心摘录

1. **策展档位：** {p["tier"]}（[具身小站 10 篇盘点](../blogs/{BLOG})）
2. **机构：** {p["inst"]}
3. **导读要点：** {p["why"]}

## 对 wiki 的映射

- [paper-{p["slug"]}](../../wiki/entities/paper-{p["slug"]}.md)
- [10 篇技术地图](../../wiki/overview/{MAP}.md)
"""
    path.write_text(content, encoding="utf-8")


def write_repo(p: dict) -> None:
    if not p["repo_file"]:
        return
    path = ROOT / "sources/repos" / p["repo_file"]
    content = f"""# {p["short"]} 官方仓库

> 来源归档（repo）

- **名称：** {p["short"]}
- **类型：** repo
- **URL：** {p["code"]}
- **关联论文：** arXiv:{p["arxiv"]}
- **入库日期：** 2026-09-21
- **开源结论：** **{p["oss"]}**

## 对 wiki 的映射

- [paper-{p["slug"]}](../../wiki/entities/paper-{p["slug"]}.md)
- [{p["source_file"]}](../papers/{p["source_file"]})
"""
    path.write_text(content, encoding="utf-8")


def write_site(p: dict) -> None:
    if not p["site"]:
        return
    site_slug = p["slug"].replace("_", "-")
    path = ROOT / "sources/sites" / f"{site_slug}.md"
    code_line = f"- **代码：** {p['code']}\n" if p["code"] else ""
    content = f"""# {p["short"]} 项目页

> 来源归档（site）

- **标题：** {p["title"]}
- **类型：** site
- **URL：** {p["site"]}
{code_line}- **关联 arXiv：** {p["arxiv"]}
- **入库日期：** 2026-09-21
- **开源结论：** **{p["oss"]}**

## 对 wiki 的映射

- [paper-{p["slug"]}](../../wiki/entities/paper-{p["slug"]}.md)
"""
    path.write_text(content, encoding="utf-8")


def seq_section(p: dict) -> str:
    if not p["seq"]:
        return """## 源码运行时序图

**不适用**（截至 2026-09-21 项目页/arXiv 未发布可运行官方代码，或仓库尚无可辨识训练/推理入口）。
"""
    repo_ref = f"../../sources/repos/{p['repo_file']}"
    return f"""## 源码运行时序图

```mermaid
sequenceDiagram
  autonumber
  participant Dev as 开发者
  participant Repo as 官方仓库
  participant Sim as 仿真/数据
  Dev->>Repo: clone + README 环境依赖
  Dev->>Sim: 准备 Isaac Sim/仿真或数据集
  Dev->>Repo: 训练/推理/示例入口
  Repo-->>Dev: 指标、日志或部署输出
```

节点对齐 [`sources/repos/{p["repo_file"]}`]({repo_ref}) 与 README 入口。
"""


def write_entity(p: dict) -> None:
    path = ROOT / "wiki/entities" / f"paper-{p['slug']}.md"
    sources = [
        f"  - ../../sources/papers/{p['source_file']}",
        f"  - ../../sources/blogs/{BLOG}",
    ]
    if p["repo_file"]:
        sources.insert(1, f"  - ../../sources/repos/{p['repo_file']}")
    if p["site"]:
        sources.insert(1, f"  - ../../sources/sites/{p['slug'].replace('_', '-')}.md")
    sources_yaml = "\n".join(sources)
    code_front = f"\ncode: {p['code']}" if p["code"] else ""
    content = f"""---
type: entity
tags:
{tag_yaml(p["tags"])}
status: complete
updated: 2026-09-21
arxiv: "{p["arxiv"]}"{code_front}
related:
  - ../overview/{MAP}.md
  - ../concepts/sim2real.md
  - ../tasks/manipulation.md
  - ../methods/vla.md
sources:
{sources_yaml}
summary: "{p["short"]}（arXiv:{p["arxiv"]}）：{p["summary"]}"
---

# {p["short"]}（arXiv:{p["arxiv"]}）

**{p["short"]}**（*{p["title"]}*，[arXiv:{p["arxiv"]}](https://arxiv.org/abs/{p["arxiv"]})）来自 [具身智能小站 10 篇盘点](../../sources/blogs/{BLOG})（策展档位：**{p["tier"]}**）。

## 一句话定义

**{p["summary"]}**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| RL | Reinforcement Learning | 强化学习 |
| SR | Success Rate | 任务成功率 |
| Sim2Real | Simulation to Real | 仿真到真机迁移 |
| WAM | World-Action Model | 联合未来观测与动作的策略 |

## 为什么重要

- 公众号将本文归入「接触丰富操作为何总在仿真里失真」专题；{p["tier"]}档位。
- **{p["inst"]}**；开源结论：**{p["oss"]}**（步骤 2.5，2026-09-21）。
- {p["why"]}

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [{p["arxiv"]}](https://arxiv.org/abs/{p["arxiv"]}) |
| **开源** | **{p["oss"]}** |
| **策展摘要** | {p["summary"]} |

{seq_section(p)}
## 实验与评测

- 定量指标与 baseline 协议以 arXiv PDF 与项目页为准；本页为清单级摘要。
- 读法：先确认任务设定（仿真/真机、传感器、成功定义）再对比 headline 数字。

## 与其他工作对比

> 本页为清单级摘要，下表只做**定位对照**；数字未与下列各页核对同一评测协议，不可横比。

| 对照 | 差异读法 |
|------|----------|
| [RAPID](./paper-rapid-vlm-rl.md) | 同专辑 **深读** 档：并行仿真 + VLM 奖励管线，关注训练吞吐与 API 成本 |
| [CRISP](./paper-crisp.md) | 同专辑 **跟进** 档：接触仿真几何与求解器，关注 peg-in-hole/装配物理准确性 |
| [10 篇技术地图](../overview/{MAP}.md) | 同批次横向对照入口：本文列 **{p["tier"]}** 档位 |

## 结论

**{p["short"]} 代表「{p["tier"]}」档位的 {p["tags"][1] if len(p["tags"]) > 1 else "manipulation"} 方向样本——部署前以开源状态与评测协议为准绳。**

1. 开源状态：**{p["oss"]}**；勿凭 PDF 臆断可复现性。
2. 与同专辑 [RAPID](./paper-rapid-vlm-rl.md) / [GALA](./paper-gala.md) / [CRISP](./paper-crisp.md) 形成「并行奖励 → 跨形态表征 → 接触仿真」阅读链。
3. 若做工程选型，先对齐传感器栈、仿真器与任务是否匹配文内设定。
4. 关注项目页/arXiv 版本更新与代码发布。

## 关联页面

- [{MAP}](../overview/{MAP}.md)
- [sim2real](../concepts/sim2real.md)
- [manipulation](../tasks/manipulation.md)
- [vla](../methods/vla.md)

## 参考来源

- [{p["source_file"]}](../../sources/papers/{p["source_file"]})
- [{BLOG}](../../sources/blogs/{BLOG})
- [arXiv:{p["arxiv"]}](https://arxiv.org/abs/{p["arxiv"]})

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/{p["arxiv"]})
- [10 篇技术地图](../overview/{MAP}.md)
"""
    path.write_text(content, encoding="utf-8")


def write_blog_and_map() -> None:
    raw_path = ROOT / "sources/raw" / RAW
    raw_path.write_text(
        f"# 接触丰富操作为何总在仿真里失真？几何与求解器一起看\n\n"
        f"原始链接：{WECHAT_URL}\n\n"
        f"（WebFetch 抓取，2026-09-21）\n",
        encoding="utf-8",
    )
    rows = []
    for p in PAPERS:
        rows.append(
            f"| {p['num']} | {p['short']} | [{p['arxiv']}](https://arxiv.org/abs/{p['arxiv']}) | "
            f"**{p['oss']}** | [paper-{p['slug']}](../../wiki/entities/paper-{p['slug']}.md) |"
        )
    table = "\n".join(rows)
    blog = f"""# 接触丰富操作为何总在仿真里失真？几何与求解器一起看

> 来源归档（blog / 微信公众号）

- **标题：** 接触丰富操作为何总在仿真里失真？几何与求解器一起看
- **类型：** blog
- **作者：** 具身智能小站（微信公众号）
- **原始链接：** {WECHAT_URL}
- **发表日期：** 2026-09-21
- **入库日期：** 2026-09-21
- **抓取方式：** WebFetch
- **原始抓取落盘：** [`sources/raw/{RAW}`](../raw/{RAW})
- **一句话说明：** 10 篇具身论文（并行 VLM 奖励 RL、跨形态 VLA、接触仿真、长时程价值/子任务 RL、失败数据、接触感知、持续学习、sim-to-real 流程）；**10/10 独立详情节点**（本 ingest **新建 10**、**复用 0**）。

## 10 篇 → 本库节点

| # | 论文 | arXiv | 开源结论 | wiki |
|---|------|-------|----------|------|
{table}

## 对 wiki 的映射

- **10/10 独立详情节点**；**0 重复 arXiv 节点**
- 阅读坐标：[接触丰富仿真 10 篇技术地图](../../wiki/overview/{MAP}.md)
- 交叉：[sim2real](../../wiki/concepts/sim2real.md)、[vla](../../wiki/methods/vla.md)、[manipulation](../../wiki/tasks/manipulation.md)

## 当前提炼状态

- [x] 公众号正文抓取
- [x] 10 篇独立节点（10 新建 / 0 复用）
- [x] 项目页/仓库开源状态核查（步骤 2.5）
"""
    (ROOT / "sources/blogs" / BLOG).write_text(blog, encoding="utf-8")

    entity_links = "\n".join(f"  - ../entities/paper-{p['slug']}.md" for p in PAPERS)
    idx_rows = []
    for p in PAPERS:
        idx_rows.append(
            f"| {p['num']} | {p['short']} | **{p['oss']}** | "
            f"[paper-{p['slug']}](../entities/paper-{p['slug']}.md) |"
        )
    map_content = f"""---
type: overview
tags: [overview, survey, simulation, contact-rich, vla, technology-map]
status: complete
updated: 2026-09-21
related:
{entity_links}
  - ../concepts/sim2real.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/blogs/{BLOG}
  - ../../sources/raw/{RAW}
summary: "具身智能小站 2026-09-21 十篇盘点：并行 VLM 奖励、跨形态 VLA、接触仿真、长时程策略与 sim-to-real 闭环。"
---

# 接触丰富仿真：10 篇论文阅读坐标

> **本页定位**：为 [具身智能小站 · 10 篇盘点]({WECHAT_URL})（2026-09-21）提供按阅读优先级组织的索引。

## 一句话观点

**接触丰富操控在仿真里失真，往往不只是「看得不够」——奖励标注吞吐、跨形态几何表征、碰撞几何与接触求解器、以及 sim-to-real 闭环评测需一起读。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 视觉–语言多模态模型 |
| RL | Reinforcement Learning | 强化学习 |
| Sim2Real | Simulation to Real | 仿真到真机迁移 |

## 阅读档位

1. **深读：** [RAPID](../entities/paper-rapid-vlm-rl.md)
2. **跟进：** [GALA](../entities/paper-gala.md)、[CRISP](../entities/paper-crisp.md)
3. **扫读：** 其余 7 篇（见下表）

## 完整索引

| # | 论文 | 开源 | 详情 |
|---|------|------|------|
{chr(10).join(idx_rows)}

## 关联页面

- [sim2real](../concepts/sim2real.md)
- [vla](../methods/vla.md)
- [manipulation](../tasks/manipulation.md)
- [RAPID](../entities/paper-rapid-vlm-rl.md)

## 参考来源

- [{BLOG}](../../sources/blogs/{BLOG})
"""
    (ROOT / "wiki/overview" / f"{MAP}.md").write_text(map_content, encoding="utf-8")


def main() -> None:
    for p in PAPERS:
        write_paper_source(p)
        write_entity(p)
        write_repo(p)
        write_site(p)
    write_blog_and_map()
    print(f"Generated {len(PAPERS)} papers + blog + map")


if __name__ == "__main__":
    main()
