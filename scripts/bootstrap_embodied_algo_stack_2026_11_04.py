#!/usr/bin/env python3
"""Bootstrap ingest: 新机器视觉（转自「人工智能AI与算法」）《2026年了，主流具身智能算法有哪些？》（2026-11-04）。

- 转载综述，没有给出处。文中点名 20 个算法 / 模型：15 个已有节点复用；
  3 个有一手论文但库内无独立节点 → 新建（Two-Steps Diffusion Policy、FlowVLA（Visual CoT）、S²-VLA（操作））；
  CSO、Latent PPO 两项文章无出处、未能定位一手论文 → 不建节点，仅在地图页注明。
- 一手核对：arXiv 摘要页（5 个新节点的标题 / 摘要 / 作者 / 日期）、OpenVLA 与 Octo 仓库 README、WebSearch 定位论文。
- 用法：python3 scripts/bootstrap_embodied_algo_stack_2026_11_04.py <抓取的 txt>
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TODAY = date.today().isoformat()
URL = "https://mp.weixin.qq.com/s/aIYDRlrluZIsmj3fmMsXIg"
BLOG = "wechat_xinjiqishijue_embodied_algorithm_stack_2026-11-04.md"
MAP = "embodied-algorithm-stack-2026-vla-diffusion-rl-world-model.md"
MAP_TITLE = "2026 具身智能算法栈：VLA、扩散策略、强化学习与世界模型（一手核对版）"

# (文中名称, 路径 wiki/<dir>/<node>.md, 层, 复用?, 核对)
MODELS: list[dict[str, Any]] = [
    dict(
        name="OpenVLA",
        path="entities/paper-openvla",
        layer="VLA",
        reuse=True,
        check="README：代码 MIT 许可，但预训练模型源自 Llama 2，受其许可约束（文章只写「MIT许可」，不完整）；LoRA 微调示例用**单张 A100**（文章写「单张 H100」）；7B 与 970K OXE 轨迹未在已读 README 段落逐项核对。",
    ),
    dict(
        name="Octo",
        path="entities/paper-octo",
        layer="VLA",
        reuse=True,
        check="README 给出 Octo-Base 93M / Octo-Small 27M 参数（与文章一致），1×RTX 4090 上推理 13 / 17 it/sec；**文章写「20–30 Hz」，与 README 的 13 / 17 it/sec 不一致**（二者口径是否相同未核实）；MIT 许可。",
    ),
    dict(
        name="SmolVLA",
        path="entities/paper-smolvla",
        layer="VLA",
        reuse=True,
        check="库内节点已核对 arXiv:2506.01844 与 LeRobot 仓库。",
    ),
    dict(
        name="π0",
        path="entities/paper-pi0",
        layer="VLA",
        reuse=True,
        check="库内节点：PaliGemma + Flow Matching 动作专家；文章「50Hz」未在本次核对。",
    ),
    dict(
        name="π0.5",
        path="entities/paper-pi05-open-world-vla",
        layer="VLA",
        reuse=True,
        check="库内节点（arXiv:2504.16054，开放世界泛化）；文章称「分层任务泛化」。",
    ),
    dict(
        name="π0.6（π*0.6）",
        path="entities/paper-rcl-2511-14759-0-6-a-vla-that-learns-from-experience",
        layer="VLA",
        reuse=True,
        check="库内节点（arXiv:2511.14759）。arXiv 摘要：用 RECAP 经优势条件化策略做 VLA 的 RL，对应文章「π0.6 融合强化学习」。",
    ),
    dict(
        name="π0.7",
        path="methods/pi07-policy",
        layer="VLA",
        reuse=True,
        check="库内方法页；文章称「集成世界模型与具身记忆」，库内页面写 π₀.₇ 使用 MEM 式历史视觉，二者表述不同，未判断。",
    ),
    dict(
        name="HiFOR",
        path="entities/paper-hifor-hierarchical-foresight-vla",
        layer="VLA",
        reuse=True,
        check="库内节点（arXiv:2610.40373），与文章「分层预见框架」一致。",
    ),
    dict(
        name="S²-VLA",
        path="entities/paper-s2-vla-state-space-guided",
        layer="VLA",
        reuse=False,
        check="新建（arXiv:2606.27872，IJCAI 2026）。**库内已有的 paper-s-squared-vla 是另一篇（驾驶 VLA，arXiv:2607.13926），不是文章所指的操作 VLA**，所以单独建节点。",
    ),
    dict(
        name="FlowVLA",
        path="entities/paper-flowvla-visual-cot",
        layer="VLA",
        reuse=False,
        check="新建（arXiv:2508.18269，Visual CoT）。库内已有的两个 FlowVLA 节点（pluggable-3d-flow、quantitative-robotic-pouring）是别的论文。",
    ),
    dict(
        name="Diffusion Policy",
        path="entities/paper-diffusion-policy",
        layer="扩散策略 / 模仿学习",
        reuse=True,
        check="库内节点；文章对 DDPM 去噪、多模态的描述未在本次逐项核对。",
    ),
    dict(
        name="ACT",
        path="entities/paper-act",
        layer="扩散策略 / 模仿学习",
        reuse=True,
        check="库内节点；「CVAE Transformer、约 50 次演示收敛」为文章表述，未在本次核对。",
    ),
    dict(
        name="Two-Steps Diffusion Policy",
        path="entities/paper-two-steps-diffusion-policy",
        layer="扩散策略 / 模仿学习",
        reuse=False,
        check="新建（arXiv:2510.21991，NeurIPS 2025）。摘要同时给出「少至 5 次 NFE」与「2 次 NFE 解决困难任务」，文章只取了 5 次；论文用 genetic denoising，在 D4RL / Robomimic 14 项任务上评测。",
    ),
    dict(
        name="REFINE-DP",
        path="entities/paper-loco-manip-161-157-refine-dp",
        layer="扩散策略 / 模仿学习",
        reuse=True,
        check="库内节点（arXiv:2603.13707）：分层 DP 规划器 + RL 跟踪器，DPPO/PPO 联合微调；文章「仿真中超过 90%」未在本次核对。",
    ),
    dict(name="PPO", path="methods/ppo", layer="强化学习", reuse=True, check="库内方法页。"),
    dict(
        name="SuperMimic",
        path="entities/paper-supermimic",
        layer="强化学习",
        reuse=True,
        check="库内节点（arXiv:2610.28480）已写明：1 张 H100 × 3 小时对比 SONIC 的 128 张 L40 × 7 天，作者保守表述 GPU-hours 减少超过 1000 倍，与文章「千倍、3 小时」一致。",
    ),
    dict(
        name="ELEVATE",
        path="entities/paper-elevate",
        layer="强化学习",
        reuse=True,
        check="库内节点（arXiv:2610.38388）；文章「98.4%–100%」未在库内节点中找到，未核实。",
    ),
    dict(
        name="WMPO",
        path="entities/paper-sa-2511-09515-wmpo-world-model-based-policy-optimization-for-v",
        layer="世界模型",
        reuse=True,
        check="库内节点（arXiv:2511.09515）；ICLR 2026（WebSearch）+ arXiv 摘要页核对：像素级动作条件视频世界模型 + on-policy GRPO，无需真实交互。",
    ),
]

UNLOCATED = [
    (
        "CSO",
        "文章称「通过对齐技能分布精炼策略，LIBERO 成功率从 80.1% 提升至 97.5%」",
        "文章未给全称与出处；我用关键词搜索没有找到与之对应的机器人论文（命中的 CSO 是 LLM 智能体方法，Skill1 的 97.5% 是 ALFWorld，均不是），**未能定位一手论文，不建节点**",
    ),
    (
        "Latent PPO",
        "文章称「面向轮腿机器人的 Latent PPO」",
        "文章未给出处；搜索没有找到名为 Latent PPO 的论文（最接近的 ATRos 为 PPO + 潜在嵌入辅助，是否即所指无法确认），**不建节点**",
    ),
]

NEW: dict[str, dict[str, Any]] = {
    "paper-two-steps-diffusion-policy": dict(
        arxiv="2510.21991",
        name="Two-Steps Diffusion Policy",
        slug="two-steps-diffusion-policy",
        venue="NeurIPS 2025",
        site=None,
        authors="Clemente, Brunswic, Yang, Zhao, Khalil 等（arXiv 摘要页前 5 位）",
        date="2025-10-24",
        tags=["paper", "diffusion-policy", "imitation-learning", "inference-efficiency"],
        one="Two-Steps Diffusion Policy 针对具身任务动作分布低维、结构化的特点定制去噪过程，使扩散策略仅需少量神经函数评估（NFE）即可有效运行，并提出 population-based 的 genetic denoising 提升性能与稳定性。",
        abstract="摘要称：扩散策略在模仿专家示范上达到 SOTA，但许多推理策略直接从视觉任务迁移而来；针对动作分布低维、结构化的特点定制去噪，可让扩散策略在少至 5 次 NFE 下有效运行；在此基础上提出 genetic denoising，选择分布外风险低的去噪轨迹；方法仅用 2 次 NFE 就解决困难任务且性能提升或持平；在 D4RT / Robomimic 的 14 项操作任务（摘要写 D4RL and Robomimic）上、超过 200 万次评测中持续优于标准扩散策略，最高约 20% 性能提升。",
        article="文章把 Two-Steps Diffusion Policy 与「各类实时加速器」并列为 2026 年效率优化重点：仅需 5 次函数评估，目标是保留鲁棒性的同时压低部署延迟。",
        check="文章写「仅需 5 次函数评估」是摘要的第一个表述；摘要同时称 2 次 NFE 即可解决困难任务，论文方法是 genetic denoising 而非 RL 微调。",
        related=[
            "../methods/diffusion-policy.md",
            "../entities/paper-diffusion-policy.md",
            "../methods/imitation-learning.md",
            "../methods/action-chunking.md",
        ],
    ),
    "paper-flowvla-visual-cot": dict(
        arxiv="2508.18269",
        name="FlowVLA（Visual CoT）",
        slug="flowvla-visual-cot",
        venue="（arXiv，v3 更名）",
        site="https://irpn-lab.github.io/FlowVLA/",
        authors="Zhong, Yan, Li, Liu, Gong 等（arXiv 摘要页前 5 位）",
        date="2025-08-25",
        tags=["paper", "vla", "world-model", "optical-flow", "visual-cot"],
        one="FlowVLA 提出 Visual Chain of Thought：让 VLA 内部的世界模型先推理运动动力学再生成未来帧，显式链路为 v_t → f_t → v_{t+1}，其中 f_t 是编码运动的中间光流预测。",
        abstract="摘要称：许多 VLA 内置的世界模型靠下一帧预测 v_t → v_{t+1}，缺少显式的运动推理，常导致物理上不合理的视觉预测与低效的策略学习；FlowVLA 是自回归 Transformer，显式物化 v_t → f_t → v_{t+1}，使「动力学预测」的预训练目标与「动作生成」的下游任务对齐；在操作基准与真机评测上，既生成更连贯、物理合理的预测，又取得 SOTA 策略性能与显著提升的样本效率。（WebSearch 补：早期版本标题为 FlowVLA: Thinking in Motion with a Visual Chain of Thought；作者来自 HKUST(GZ) 与上海交大，未在摘要页核对机构。）",
        article="文章把 FlowVLA 归为 2026 下半年针对长程与推理的架构创新之一：「引入视觉思维链进行运动推理」。",
        check="摘要与文章一致。库内另有两篇同名 FlowVLA（pluggable-3d-flow、quantitative-robotic-pouring），是别的论文，本节点对应 arXiv:2508.18269。",
        related=[
            "../methods/vla.md",
            "../concepts/world-action-models.md",
            "../entities/paper-flowvla-pluggable-3d-flow.md",
            "../entities/paper-flowvla-quantitative-robotic-pouring.md",
        ],
    ),
    "paper-s2-vla-state-space-guided": dict(
        arxiv="2606.27872",
        name="S²-VLA（状态空间引导的长程操作）",
        slug="s2-vla-state-space-guided",
        venue="IJCAI 2026（主会）",
        site=None,
        authors="Xie, Han, Wei, Sun, Li, Zhao 等（arXiv 摘要页）",
        date="2026-06-26",
        tags=["paper", "vla", "long-horizon", "manipulation", "attention"],
        one="S²-VLA 针对长程任务中误差累积的问题，引入状态空间引导自适应注意力（SSGAA）：用跟踪任务进度的信念状态生成动态门控权重，自适应融合视觉、任务意图与时序动作三类特征。",
        abstract="摘要称：VLA 在长程任务中因累积误差传播而性能显著下降，部分原因是静态特征融合用固定权重组合视觉、语言与动作表示，无法适应任务不同阶段；SSGAA 维护信念状态跟踪任务进展，生成动态门控权重，融合视觉特征（空间感知）、任务意图（高层规划）与时序动作序列（执行一致性）；约 2B 参数，持续优于更大的 7B 级模型，在 LIBERO 与 SimplerEnv 等长程操作基准上达到 SOTA。（IJCAI 2026 收录与页码来自 WebSearch，未在 IJCAI 站点核对。）",
        article="文章称「S²-VLA 利用状态空间引导长程操作」。",
        check="摘要与文章一致。**库内已有的 `paper-s-squared-vla`（arXiv:2607.13926）是另一篇驾驶 VLA，不是同一篇**；本节点对应 arXiv:2606.27872。",
        related=[
            "../methods/vla.md",
            "../entities/paper-hifor-hierarchical-foresight-vla.md",
            "../entities/paper-s-squared-vla.md",
            "../methods/action-chunking.md",
        ],
    ),
}

SCENARIOS = [
    (
        "工业分拣 / 上下料",
        "轻量 VLA（Octo / SmolVLA）+ Diffusion Policy + 少量 RL 精修",
        "边界清晰、结果可量化，优先速度与稳定性",
    ),
    ("双臂遥操作 / 柔性装配", "ACT + VLA 基座微调", "训练快、演示效率高，适合快速迭代"),
    (
        "腿足巡检 / 搬运",
        "PPO / Latent PPO + 世界模型预训练",
        "运动控制靠 RL，长程规划靠世界模型降本",
    ),
    (
        "家庭服务长程任务",
        "π0 系列 / OpenVLA + 世界模型 + 具身记忆",
        "需要分层规划、高频控制与跨任务泛化",
    ),
    (
        "灵巧手精细操作",
        "Flow Matching 动作头 + 扩散策略 + RL 失败驱动",
        "高频控制 + 多模态动作分布 + 长尾技能精修",
    ),
]

MARKET = [
    "智元「精灵 G2」在龙旗科技工厂 6 天测试累计 64,828 次作业、综合成功率 99.99%",
    "2026 年上半年智元出货约 9,700 台、占全球人形机器人市场超过 43%",
    "临界点灵巧手在真实工位完成 72 道工序、成功率 98%、年量产突破 15,000 台、整机故障率低于 0.5%",
    "2026 年上半年国内具身智能融资额 934 亿元、为 2025 年同期的五倍；中国具身智能市场规模预计 2026 年达 1.09 万亿元",
    "全球人形机器人出货结构中娱乐表演、科研仍占 60% 以上，智能制造和仓储物流分别仅 13% 和 5%",
]


def _yl(items: list[str]) -> str:
    return "\n".join(f"  - {x}" for x in items)


def _entity(node: str, p: dict[str, Any]) -> str:
    rel = [*p["related"], f"../overview/{MAP}"]
    rel_lines = "\n".join(f"- [{r.split('/')[-1][:-3]}]({r})" for r in rel)
    site = f"project: {p['site']}\n" if p.get("site") else ""
    summ = p["one"].replace('"', "'")
    src = p["slug"].replace("-", "_")
    return f"""---
type: entity
project_id: {p["slug"]}
tags:
{_yl(p["tags"])}
status: complete
updated: {TODAY}
arxiv: "{p["arxiv"]}"
{site}related:
{_yl(rel)}
sources:
  - ../../sources/papers/{src}_arxiv_{p["arxiv"].replace(".", "_")}.md
  - ../../sources/blogs/{BLOG}
summary: "{p["name"]}（arXiv:{p["arxiv"]}）：{summ}"
---

# {p["name"]}

**{p["name"]}**（[arXiv:{p["arxiv"]}](https://arxiv.org/abs/{p["arxiv"]}){"，[项目页](" + p["site"] + ")" if p.get("site") else ""}）是 [新机器视觉《2026年了，主流具身智能算法有哪些？》]({URL}) 点名的算法之一。本页以 arXiv 摘要页为一手资料，文章说法放在「文章说法核对」。

## 一句话定义

**{p["one"]}**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作模型 |

## 为什么重要

- 在文章的 2026 算法栈里属于：{next(m["layer"] for m in MODELS if m["path"].endswith(node))}层；总览见 [{MAP_TITLE}](../overview/{MAP})。

## 一手资料（arXiv 摘要页）

- **标题 / 编号：** [{p["name"]}](https://arxiv.org/abs/{p["arxiv"]})（提交 {p["date"]}；{p["venue"]}）
- **作者：** {p["authors"]}
- **摘要要点：** {p["abstract"]}

## 文章说法核对

- **文章：** {p["article"]}
- **核对：** {p["check"]}

## 关联页面

{rel_lines}

## 参考来源

- [{src}_arxiv_{p["arxiv"].replace(".", "_")}.md](../../sources/papers/{src}_arxiv_{p["arxiv"].replace(".", "_")}.md)
- [{BLOG}](../../sources/blogs/{BLOG})
- [arXiv:{p["arxiv"]}](https://arxiv.org/abs/{p["arxiv"]})
"""


def _paper_src(node: str, p: dict[str, Any]) -> str:
    return f"""# {p["name"]}（arXiv:{p["arxiv"]}）

> 来源归档（paper）

- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/{p["arxiv"]}>
- **PDF：** <https://arxiv.org/pdf/{p["arxiv"]}>
- **作者：** {p["authors"]}
- **提交：** {p["date"]}；{p["venue"]}
{"- **项目页：** <" + p["site"] + ">" if p.get("site") else ""}
- **入库日期：** {TODAY}
- **线索来源：** [{BLOG}](../blogs/{BLOG})

## arXiv 摘要要点（人工转述）

{p["abstract"]}

## 对 wiki 的映射

- [{node}](../../wiki/entities/{node}.md)
"""


def _blog() -> str:
    rows = "\n".join(
        f"| {m['name']} | {m['layer']} | [{m['path'].split('/')[-1]}](../../wiki/{m['path']}.md) | {'复用' if m['reuse'] else '新建'} | {m['check']} |"
        for m in MODELS
    )
    un = "\n".join(f"| {a} | {b} | {c} |" for a, b, c in UNLOCATED)
    mk = "\n".join(f"- {x}" for x in MARKET)
    return f"""# 2026年了，主流具身智能算法有哪些？一文彻底讲透VLA、扩散策略和世界模型

> 来源归档（blog / 微信公众号转载）

- **标题：** 2026年了，主流具身智能算法有哪些？一文彻底讲透VLA、扩散策略和世界模型
- **类型：** blog
- **来源账号：** 新机器视觉（转自公众号「人工智能AI与算法」；文首注明「本文仅用于学术分享，如有侵权请联系后台作删文处理」）
- **原始链接：** {URL}
- **发表日期：** 2026-11-04
- **入库日期：** {TODAY}
- **抓取方式：** `scripts/fetch_wechat_article.py`（Chromium + 微信内置 UA），正文 3527 字符
- **一句话说明：** 综述 2026 年具身智能的四层算法栈（VLA 基座、扩散策略保可靠性、RL 啃硬技能、世界模型赌未来泛化），并给出五类场景的算法组合建议。
- **说明：** 转载综述，**没有给出处**。我对文中点名的每个算法回到一手资料核对（arXiv 摘要页、OpenVLA / Octo 仓库 README、论文检索）：15 个复用已有节点，3 个新建，2 个（CSO、Latent PPO）无法定位。市场数据全部是文章表述、未核实。

## 算法 → 节点与一手核对

| 文中名称 | 层 | 节点 | 状态 | 一手核对 |
|----------|----|------|------|----------|
{rows}

## 未能定位一手论文的两项

| 名称 | 文章说法 | 核对结果 |
|------|----------|----------|
{un}

## 文章的市场数据（作者表述，未核实）

{mk}

> 这些数据没有给出处；文章自己也提醒「演示呈现最优表现，部署要求最不利情况下仍满足产线标准」。

## 对 wiki 的映射

- 总览：[{MAP_TITLE}](../../wiki/overview/{MAP})
"""


def _map() -> str:
    rows = "\n".join(
        f"| {m['name']} | {m['layer']} | [{m['path'].split('/')[-1]}](../{m['path']}.md){'（复用）' if m['reuse'] else '（新建）'} |"
        for m in MODELS
    )
    un = "\n".join(f"| {a} | {c} |" for a, b, c in UNLOCATED)
    sc = "\n".join(f"| {a} | {b} | {c} |" for a, b, c in SCENARIOS)
    mk = "\n".join(f"- {x}" for x in MARKET)
    rel = [f"../{m['path']}.md" for m in MODELS[:6]] + [
        "../methods/vla.md",
        "../methods/diffusion-policy.md",
        "../methods/reinforcement-learning.md",
    ]
    return f"""---
type: overview
tags: [overview, vla, diffusion-policy, reinforcement-learning, world-model, algorithm-stack, wechat]
status: complete
updated: {TODAY}
related:
{_yl(rel)}
sources:
  - ../../sources/blogs/{BLOG}
summary: "新机器视觉转载综述的阅读坐标：2026 具身智能四层算法栈（VLA / 扩散策略 / RL / 世界模型），文中点名的算法逐个回到一手资料核对（15 复用 / 3 新建 / 2 无法定位），并指出文章与 README、论文不一致之处。"
---

# {MAP_TITLE}

> **本页定位**：为 [2026年了，主流具身智能算法有哪些？]({URL})（新机器视觉转载，2026-11-04）提供阅读坐标。文章是没有出处的转载综述，本页对每个算法回到一手资料核对，**以一手资料为准、文章说法仅作对照**。

## 一句话观点

**文章的结论：别追求单一算法通吃——VLA 当基座、扩散策略保可靠性、RL 啃硬技能、世界模型赌未来泛化，按场景的可靠性、算力与数据条件组合。**（文章观点，工程上是合理的分层叙事；其中具体数字有若干与一手资料不一致，见下。）

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作模型 |
| NFE | Number of Function Evaluations | 扩散采样的神经函数评估次数 |
| GRPO | Group Relative Policy Optimization | 组相对策略优化（WMPO 使用） |
| RECAP | RL with Experience and Corrections via Advantage-conditioned Policies | π*0.6 的训练方法 |

## 为什么单独做这张地图

- 库内相关节点散落各处，且存在**同名不同篇**的陷阱（如 S²-VLA 在库内已有驾驶版、FlowVLA 已有两篇别的论文），需要一张明确对应关系的表。
- 文章的若干具体数字与一手资料不符（Octo 推理频率、OpenVLA 适配硬件与许可），必须显式标注。

## 四层算法栈与节点

| 文中名称 | 层 | 节点 |
|----------|----|------|
{rows}

## 无法定位一手论文的两项

| 名称 | 核对结果 |
|------|----------|
{un}

## 场景选型建议（文章）

| 场景 | 推荐组合 | 理由 |
|------|----------|------|
{sc}

## 文章的市场数据（作者表述，未核实）

{mk}

## 关联页面

- [VLA 方法总览](../methods/vla.md)
- [Diffusion Policy](../methods/diffusion-policy.md)
- [强化学习](../methods/reinforcement-learning.md)
- [VLA 演进简史（Khushi 长文）](./vla-history-rt1-to-molmoact-technology-map.md)

## 参考来源

- [{BLOG}](../../sources/blogs/{BLOG})
"""


def _patch(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    head, sep, body = text.partition("\n---\n")
    if not sep:
        raise SystemExit(f"frontmatter 缺失: {path}")
    lines = head.split("\n")
    blog = f"../../sources/blogs/{BLOG}"
    page = f"../overview/{MAP}"

    def _append(key: str, item: str) -> None:
        if item in (x.strip().lstrip("- ").strip() for x in lines):
            return
        idx = next((i for i, x in enumerate(lines) if x.startswith(f"{key}:")), None)
        if idx is None:
            lines.extend([f"{key}:", f"  - {item}"])
            return
        if lines[idx].strip() != f"{key}:":
            raise SystemExit(f"{path}: {key} 为行内写法，需手工补")
        j = idx + 1
        indent = "  "
        if j < len(lines) and lines[j].lstrip().startswith("- "):
            indent = lines[j][: len(lines[j]) - len(lines[j].lstrip())]
        while j < len(lines) and lines[j].lstrip().startswith("- "):
            j += 1
        lines.insert(j, f"{indent}- {item}")

    _append("sources", blog)
    _append("related", page)
    lines = [f"updated: {TODAY}" if x.startswith("updated:") else x for x in lines]
    path.write_text("\n".join(lines) + sep + body, encoding="utf-8")


def main() -> None:
    txt = Path(sys.argv[1]).read_text(encoding="utf-8")
    for m in MODELS:
        exists = (ROOT / "wiki" / f"{m['path']}.md").exists()
        if m["reuse"] and not exists:
            raise SystemExit(f"复用节点不存在: {m['path']}")
        if not m["reuse"] and exists:
            raise SystemExit(f"节点已存在: {m['path']}")
    for node, p in NEW.items():
        for r in p["related"]:
            if not (ROOT / "wiki/entities" / r).resolve().exists():
                raise SystemExit(f"关联页不存在: {r} ({node})")
    (ROOT / "sources/raw" / BLOG).write_text(
        f"# 2026年了，主流具身智能算法有哪些？\n\n> 原始抓取（{TODAY}，`scripts/fetch_wechat_article.py`），仅供溯源；来源：{URL}\n\n{txt}",
        encoding="utf-8",
    )
    (ROOT / "sources/blogs" / BLOG).write_text(_blog(), encoding="utf-8")
    (ROOT / "wiki/overview" / MAP).write_text(_map(), encoding="utf-8")
    for node, p in NEW.items():
        src = p["slug"].replace("-", "_")
        (ROOT / "sources/papers" / f"{src}_arxiv_{p['arxiv'].replace('.', '_')}.md").write_text(
            _paper_src(node, p), encoding="utf-8"
        )
        (ROOT / "wiki/entities" / f"{node}.md").write_text(_entity(node, p), encoding="utf-8")
    for m in MODELS:
        if m["reuse"]:
            _patch(ROOT / "wiki" / f"{m['path']}.md")
    print(
        f"Wrote blog + map + {len(NEW)} new entities, patched {sum(1 for m in MODELS if m['reuse'])} reuse nodes"
    )


if __name__ == "__main__":
    main()
