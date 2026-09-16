#!/usr/bin/env python3
"""Bootstrap ingest: 具身智能小站 12 篇论文（2026-09-16 公众号盘点）."""

from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TODAY = date.today().isoformat()
BLOG = "wechat_embodied_station_12_papers_vla_deploy_2026-09-16.md"
BLOG_PATH = ROOT / "sources/blogs" / BLOG
RAW_SRC = Path(
    "/tmp/wx-out/代码真的能帮部署吗？FluxVLA、JEPLO与DIDO等12篇机器人论文资源整理/"
    "代码真的能帮部署吗？FluxVLA、JEPLO与DIDO等12篇机器人论文资源整理.md"
)
RAW_DST = ROOT / "sources/raw/wechat_embodied_station_12_papers_vla_deploy_2026-09-16.md"
MAP_PATH = ROOT / "wiki/overview/vla-deploy-12-papers-technology-map.md"

REUSE = {
    "2609.17210": {
        "wiki": "wiki/entities/fluxvla-engine.md",
        "label": "FluxVLA Engine",
        "open": "已开源",
    },
    "2609.15988": {
        "wiki": "wiki/entities/paper-ressafe.md",
        "label": "ResSafe",
        "open": "待发布",
    },
}

PAPERS = [
    {
        "slug": "jeplo",
        "title": "JEPLO: Joint-Embedding Predictive Learning for LiDAR-Based Legged Locomotion",
        "short": "JEPLO",
        "arxiv": "2609.15770",
        "code": "https://github.com/ASIG-X/JEPLO",
        "project": None,
        "open": "已开源",
        "tags": ["paper", "locomotion", "lidar", "jepa", "sim2real", "unitree"],
        "one_liner": "PE-JEPA 学局部地形 latent，CJTS 教师—学生接到四足运动策略；强调遮挡/稀疏/噪声感知退化下的鲁棒 sim-to-real。",
        "why": "足式在楼梯、箱体与遮挡环境中，感知退化往往比策略本身更先造成失稳；无图导航与板载轻量计算的可迁移表征。",
        "mechanism": "PE-JEPA 从原始 LiDAR + 本体状态学局部地形表征；CJTS 管线把 latent 接到 locomotion policy；评测强调退化感知条件。",
        "metrics": "多地形 sim-to-real；依赖 Unitree Go2、Mid-360 LiDAR、Isaac Lab/MuJoCo 与 Jetson 等具体条件。",
        "conclusion": "JEPLO 用 JEPA 式预测表征替代显式建图，把退化感知当作一等评测维度；部署前核对硬件栈与 sim2real 口径。",
        "related": [
            "../methods/reinforcement-learning.md",
            "../tasks/locomotion.md",
            "../concepts/sim2real.md",
            "./unitree-g1.md",
        ],
        "abbrev": [
            ("JEPLO", "Joint-Embedding Predictive Learning for Locomotion", "本文框架"),
            ("JEPA", "Joint-Embedding Predictive Architecture", "联合嵌入预测表征学习"),
            ("LiDAR", "Light Detection and Ranging", "激光雷达外感知"),
            ("Sim2Real", "Simulation to Real", "仿真到真机迁移"),
        ],
    },
    {
        "slug": "dido-wam",
        "title": "DIDO: Distilling Interaction-Centric Dynamics into One-Step Denoising for World Action Models",
        "short": "DIDO",
        "arxiv": "2609.15570",
        "code": "https://github.com/LoveJu1y/DIDO-WAM",
        "project": "https://loveju1y.github.io/DIDO/",
        "open": "已开源",
        "tags": ["paper", "world-model", "wam", "distillation", "manipulation"],
        "one_liner": "把多步 WAM 去噪蒸馏为一步，用交互实体 bbox token + DINOv3 对齐，避免背景保留、接触动态丢失。",
        "why": "WAM 迭代去噪侵蚀闭环频率；简单截断易保留场景结构、丢掉夹爪—物体交互动态。",
        "mechanism": "分布匹配外，对夹爪/目标物/交互区域做 bbox 推理 token；DINOv3 特征对齐目标物表征。",
        "metrics": "LIBERO 99.0%、LIBERO-Plus 76.6%、RoboTwin 92.0%（作者报告）；含真机长程与泛化实验。",
        "conclusion": "DIDO 证明一步 WAM 可行前提是蒸馏目标盯住交互实体，而非只对齐全局像素分布。",
        "related": [
            "../concepts/world-action-models.md",
            "../methods/generative-world-models.md",
            "../tasks/manipulation.md",
            "./paper-wm-loco.md",
        ],
        "abbrev": [
            ("DIDO", "Distilling Interaction-Centric Dynamics into One-Step", "本文一步蒸馏 WAM"),
            ("WAM", "World Action Model", "世界预测与动作联合建模"),
            ("DiT", "Diffusion Transformer", "扩散 Transformer 骨干"),
            ("LIBERO", "Lifelong Robot Learning Benchmark", "操作基准套件"),
        ],
    },
    {
        "slug": "slotdit",
        "title": "SlotDiT: Object-Centric Representations for Diffusion Transformers",
        "short": "SlotDiT",
        "arxiv": "2609.17414",
        "code": None,
        "project": "https://slot-dit.github.io/",
        "open": "待发布",
        "tags": ["paper", "world-model", "diffusion", "object-centric", "video-prediction"],
        "one_liner": "把场景分解为对象级 slots，在统一 DiT 下比较 slot、VAE 与语义对齐表示的机器人视频预测。",
        "why": "像素或 VAE latent 对「哪个物体发生了什么」缺少显式结构；为表示选型提供清晰切口。",
        "mechanism": "对象级 slot 分解 + DiT 视频预测；对照 VAE latent 与语义对齐表征。",
        "metrics": "以项目页与原文为准；入库日项目页未列官方 GitHub。",
        "conclusion": "SlotDiT 把对象级结构引入 DiT 视频预测，适合先读表示层设计再决定是否跟代码。",
        "related": [
            "../methods/generative-world-models.md",
            "../concepts/world-action-models.md",
            "../tasks/manipulation.md",
            "./paper-dido-wam.md",
        ],
        "abbrev": [
            ("SlotDiT", "Slot-based Diffusion Transformer", "本文对象级 DiT"),
            ("DiT", "Diffusion Transformer", "扩散 Transformer"),
            ("VAE", "Variational Autoencoder", "变分自编码器 latent"),
            ("WM", "World Model", "环境前向预测模型"),
        ],
    },
    {
        "slug": "robresilience",
        "title": "RobResilience: Implementing and Evaluating a Resilience Framework for Cyber-Physical Embodied Systems",
        "short": "RobResilience",
        "arxiv": "2609.17349",
        "code": "https://github.com/mahyamkashani/RobResilience",
        "project": None,
        "open": "已开源",
        "tags": ["paper", "cybersecurity", "ros2", "resilience", "humanoid"],
        "one_liner": "Webots PR2/ROS2 上运行时检查可容忍扰动、可容忍退化与缓解可行性，支撑具身系统安全状态机。",
        "why": "检测到攻击不等于知道该降级、缓解还是停机；需要可复现的韧性评估框架。",
        "mechanism": "运行时监测扰动容忍边界与缓解动作可行性；PR2 + ROS2 + Webots 实例。",
        "metrics": "框架级案例研究；以论文与仓库 README 为准。",
        "conclusion": "RobResilience 把「攻击后还能不能安全跑」变成可测状态，而非仅做检测告警。",
        "related": [
            "../concepts/safety-filter.md",
            "../methods/safe-rl.md",
            "../entities/ros2-control.md",
            "./paper-ressafe.md",
        ],
        "abbrev": [
            ("CPS", "Cyber-Physical System", "信息物理系统"),
            ("ROS2", "Robot Operating System 2", "机器人中间件"),
            ("PR2", "Personal Robot 2", "Willow Garage 研究平台"),
            ("HIL", "Hardware-in-the-Loop", "半实物仿真"),
        ],
    },
    {
        "slug": "machine-zygote",
        "title": "Machine Zygote: Causal Biparental Heredity Before Learning in a Germline--Soma Artificial Agent",
        "short": "Machine Zygote",
        "arxiv": "2609.17300",
        "code": "https://github.com/LyesSaadSaoud/machine-zygote",
        "project": None,
        "open": "已开源",
        "tags": ["paper", "simulation", "causal-inference", "evolutionary-robotics"],
        "one_liner": "模拟智能体在「学习前」通过双亲 germline 重组与冻结 soma 测试因果遗传 vs 表观相似。",
        "why": "区分亲子行为相似与真正因果遗传；论文明确不外推到生物遗传或真实机器人。",
        "mechanism": "双亲 germline 重组、冻结 soma、干预实验；无学习阶段的模拟智能体。",
        "metrics": "模拟干预实验；非机器人部署论文。",
        "conclusion": "Machine Zygote 是方法论/因果实验论文，读法应限定在模拟智能体遗传推断，勿当机器人算法。",
        "related": [
            "../methods/reinforcement-learning.md",
            "../methods/reinforcement-learning.md",
            "../tasks/manipulation.md",
            "./paper-robresilience.md",
        ],
        "abbrev": [
            ("germline", "Germline", "可遗传配置层"),
            ("soma", "Soma", "个体表现型/执行层"),
            ("MDP", "Markov Decision Process", "序贯决策形式化"),
            ("IL", "Imitation Learning", "模仿学习（本文刻意不在学习阶段）"),
        ],
    },
    {
        "slug": "wholebodywam",
        "title": "WholeBodyWAM: Generalizing Pre-trained World-Action Priors to Humanoid Loco-Manipulation via WBC-Grounded Coordination",
        "short": "WholeBodyWAM",
        "arxiv": "2609.16644",
        "code": None,
        "project": "https://wholebodywam.github.io/",
        "open": "待发布",
        "tags": ["paper", "humanoid", "wam", "loco-manipulation", "wbc"],
        "one_liner": "保留预训练世界—动作先验，用 WBC 语义接地协调模块扩展到人形全身 loco-manipulation。",
        "why": "桌面 WAM 多；人形需同时协调行走、全身控制与手部动作，不宜从零重学全身行为。",
        "mechanism": "预训练 WAM 先验 + 异构 WBC 语义协调模块；项目页入库日无 GitHub。",
        "metrics": "项目页强调可扩展全身智能；具体数值以原文为准。",
        "conclusion": "WholeBodyWAM 代表「先验复用 + WBC 接地」路线，工程复现需等官方代码。",
        "related": [
            "../concepts/world-action-models.md",
            "../concepts/whole-body-control.md",
            "../tasks/loco-manipulation.md",
            "./paper-dido-wam.md",
        ],
        "abbrev": [
            ("WAM", "World Action Model", "世界—动作联合模型"),
            ("WBC", "Whole-Body Control", "全身控制"),
            ("Loco-Manip", "Loco-Manipulation", "移动操作联合任务"),
            ("Prior", "Pre-trained Prior", "预训练世界—动作先验"),
        ],
    },
    {
        "slug": "proxidex",
        "title": "ProxiDex: Learning Dynamics-Guided Proximity Policy for Dexterous Manipulation",
        "short": "ProxiDex",
        "arxiv": "2609.16586",
        "code": None,
        "project": "https://proxidex.github.io/",
        "open": "待发布",
        "tags": ["paper", "dexterous-manipulation", "tactile", "proximity", "manipulation"],
        "one_liner": "把手—物体接近关系转为硬件无关交互表征，学习动作条件下的接近动态，补视觉不可靠时的策略信号。",
        "why": "灵巧手操作中手部遮挡与触觉硬件差异使接触状态难稳定观测。",
        "mechanism": "接近关系表征 + 动作条件接近动态；项目页入库日为占位模板，无有效 GitHub。",
        "metrics": "以项目页与原文为准。",
        "conclusion": "ProxiDex 把「接近」当可迁移中间表征，适合跟踪视觉退化下的灵巧操作，但代码待发布。",
        "related": [
            "../tasks/manipulation.md",
            "../methods/imitation-learning.md",
            "../concepts/tactile-sensing.md",
            "./paper-stereopatch.md",
        ],
        "abbrev": [
            ("ProxiDex", "Proximity-guided Dexterous policy", "本文接近动态策略"),
            ("Dex", "Dexterous", "灵巧手操作"),
            ("IL", "Imitation Learning", "模仿学习管线"),
            ("Sim2Real", "Simulation to Real", "仿真到真机"),
        ],
    },
    {
        "slug": "goal-oriented-comms-physical-ai",
        "title": "Goal-Oriented Communications for Physical AI: Design and Testbed",
        "short": "Goal-Oriented Comms for Physical AI",
        "arxiv": "2609.15895",
        "code": None,
        "project": "https://sites.google.com/view/goc-physical-ai-testbed",
        "open": "待发布",
        "tags": ["paper", "communications", "5g", "edge-computing", "physical-ai"],
        "one_liner": "用 3D 框、2D/3D 场景图替代原始视频流，在 5G—边缘—机器人链路上做目标导向通信实测。",
        "why": "物理 AI 高频视频闭环挤压带宽、时延与边缘算力；需任务相关语义通信。",
        "mechanism": "语义提取 → 5G/边缘传输 → 控制执行全链路 testbed；入库日项目页未列训练代码仓。",
        "metrics": "系统级时延/带宽/任务成功率；以 testbed 论文为准。",
        "conclusion": "本文是系统/通信视角的 Physical AI 基建，与 VLA 算法页互补而非替代。",
        "related": [
            "../overview/embodied-infra-2026-panorama.md",
            "../methods/vla.md",
            "../tasks/manipulation.md",
            "./paper-robresilience.md",
        ],
        "abbrev": [
            ("GoC", "Goal-Oriented Communications", "目标导向通信"),
            ("Physical AI", "Physical Artificial Intelligence", "物理世界闭环 AI"),
            ("5G", "Fifth-Generation Mobile Network", "第五代移动通信"),
            ("Edge", "Edge Computing", "边缘计算节点"),
        ],
    },
    {
        "slug": "wla3",
        "title": "WLA^3: World Latent Action Modeling for Semantics, Dynamics, and Kinematics",
        "short": "WLA³",
        "arxiv": "2609.15870",
        "code": None,
        "project": "https://wla-3.github.io/",
        "open": "待发布",
        "tags": ["paper", "latent-action", "world-model", "pretraining", "manipulation"],
        "one_liner": "从相邻世界状态变化学 latent action，同一表征复用于语义、动力学与运动学；人类视频可参与预训练。",
        "why": "异构数据缺少统一低噪声动作监督，限制通用策略扩展。",
        "mechanism": "世界状态差分 → latent action；跨语义/动力学/运动学复用；项目页 arXiv 链入库日标注 coming soon。",
        "metrics": "以原文与项目页为准。",
        "conclusion": "WLA³ 把「世界变化」当动作共同语言，适合与世界模型/潜动作文献对照阅读。",
        "related": [
            "../concepts/world-action-models.md",
            "./paper-act-lam.md",
            "../methods/vla.md",
            "./paper-dido-wam.md",
        ],
        "abbrev": [
            ("WLA³", "World Latent Action Modeling", "本文三域潜动作框架"),
            ("LAM", "Latent Action Model", "潜动作模型"),
            ("WM", "World Model", "世界状态预测"),
            ("IL", "Imitation Learning", "异构示范预训练"),
        ],
    },
    {
        "slug": "stereopatch",
        "title": "StereoPatch: Patch-Aligned RGB-Depth Fusion for Spatial Perception in Robot Manipulation",
        "short": "StereoPatch",
        "arxiv": "2609.15509",
        "code": "https://github.com/YananZHOU5555/stereopatch",
        "project": "https://aus.bot/research/stereopatch/",
        "open": "部分开源",
        "tags": ["paper", "depth", "manipulation", "perception", "rgbd"],
        "one_liner": "把注册后的度量深度绑定到动作预测所用的 RGB patch，非对称 cross-attention 融合，消解控制相关几何歧义。",
        "why": "外观相似场景可能因高度/位置/接触几何不同而需不同动作；深度需与策略看的 patch 对齐。",
        "mechanism": "patch 级 RGB–Depth 绑定 + 非对称 cross-attention；仓库入库日以媒体 release 为主。",
        "metrics": "RoboMimic 等仿真任务与真机 rollout 视频（项目页）；具体成功率以原文为准。",
        "conclusion": "StereoPatch 强调「深度对齐到动作 patch」而非全局早期融合，适合空间感知选型。",
        "related": [
            "../tasks/manipulation.md",
            "../methods/vla.md",
            "./paper-proxidex.md",
            "../concepts/visuo-tactile-fusion.md",
        ],
        "abbrev": [
            ("RGB-D", "RGB + Depth", "彩色图与深度融合"),
            ("Patch", "Image Patch", "与策略骨干对齐的图像块"),
            ("CA", "Cross-Attention", "跨模态注意力融合"),
            ("Sim2Real", "Simulation to Real", "仿真到真机"),
        ],
    },
]


def _yaml_list(items: list[str], indent: int = 2) -> str:
    pad = " " * indent
    return "\n".join(f"{pad}- {x}" for x in items)


def _paper_source(p: dict) -> str:
    ax = p["arxiv"].replace(".", "_")
    lines = [
        f"# {p['short']}（arXiv:{p['arxiv']}）",
        "",
        "> 来源归档（paper）",
        "",
        f"- **标题：** {p['title']}",
        "- **类型：** paper",
        f"- **arXiv：** <https://arxiv.org/abs/{p['arxiv']}>",
        f"- **PDF：** <https://arxiv.org/pdf/{p['arxiv']}>",
    ]
    if p.get("project"):
        lines.append(f"- **项目页：** <{p['project']}>")
    if p.get("code"):
        lines.append(f"- **代码：** <{p['code']}>")
    lines += [
        f"- **入库日期：** {TODAY}",
        f"- **一句话说明：** {p['one_liner']}",
        "",
        "## 开源状态",
        "",
        f"- **{p['open']}**（步骤 2.5 核查，{TODAY}）",
        "",
        "## 核心摘录",
        "",
        p["mechanism"],
        "",
        f"**文内指标：** {p['metrics']}",
        "",
        "## 对 wiki 的映射",
        "",
        f"- [paper-{p['slug']}](../../wiki/entities/paper-{p['slug']}.md)",
        f"- [12 篇技术地图](../../wiki/overview/vla-deploy-12-papers-technology-map.md)",
    ]
    return "\n".join(lines) + "\n"


def _entity(p: dict) -> str:
    ax = p["arxiv"]
    slug = p["slug"]
    src_paper = f"../../sources/papers/{slug}_arxiv_{ax.replace('.', '_')}.md"
    src_blog = f"../../sources/blogs/{BLOG}"
    repo_src = ""
    if p.get("code"):
        repo_src = f"  - ../../sources/repos/{slug.replace('-', '_')}.md\n"
    site_src = ""
    if p.get("project"):
        site_src = f"  - ../../sources/sites/{slug}.md\n"
    code_line = ""
    if p.get("code"):
        code_line = f'code: {p["code"]}\n'
    abbrev = "\n".join(
        f"| {a} | {b} | {c} |" for a, b, c in p["abbrev"]
    )
    seq = ""
    if p["open"] == "已开源" and p.get("code"):
        seq = """
## 源码运行时序图

```mermaid
sequenceDiagram
  autonumber
  participant U as 用户/脚本
  participant R as 官方仓库
  participant M as 训练/推理
  participant E as 仿真或真机
  U->>R: clone + 安装依赖
  U->>M: 加载配置/权重
  U->>E: rollout / 评测
  M-->>E: 动作或轨迹
  E-->>U: 指标日志
```
"""
    elif p["open"] == "部分开源":
        seq = f"""
## 源码运行时序图

**部分开源** — 入库日以项目页媒体/权重发布为主；完整训练管线以官方后续更新为准。
"""
    else:
        seq = f"""
## 源码运行时序图

**不适用（{p['open']}）** — 截至 {TODAY} 项目页未列可运行官方仓库。
"""
    proj_line = ""
    if p.get("project"):
        proj_line = f"- [项目页]({p['project']})\n"
    code_read = ""
    if p.get("code"):
        code_read = f"- [{p['code']}]({p['code']})\n"
    return f"""---
type: entity
tags:
{_yaml_list(p['tags'], 2)}
status: complete
updated: {TODAY}
arxiv: "{ax}"
{code_line}related:
{_yaml_list(p['related'] + [f"../overview/vla-deploy-12-papers-technology-map.md"], 2)}
sources:
  - {src_paper}
{repo_src}{site_src}  - {src_blog}
summary: "{p['short']}（arXiv:{ax}）：{p['one_liner'][:120]}"
---

# {p['short']}（arXiv:{ax}）

**{p['short']}**（*{p['title']}*，[arXiv:{ax}](https://arxiv.org/abs/{ax}){f"，[项目页]({p['project']})" if p.get("project") else ""}{f"，[代码]({p['code']})" if p.get("code") else ""}）来自 [具身智能小站 12 篇盘点](../../sources/blogs/{BLOG})。

## 一句话定义

**{p['one_liner']}**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
{abbrev}

## 为什么重要

- {p['why']}
- 开源结论：**{p['open']}**（步骤 2.5，{TODAY}）。
- 与 [12 篇技术地图](../overview/vla-deploy-12-papers-technology-map.md) 中同类工作可横向对照。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [{ax}](https://arxiv.org/abs/{ax}) |
| **开源** | **{p['open']}** |
| **要点** | {p['mechanism']} |
| **文内指标** | {p['metrics']} |

{seq}

## 实验与评测

- {p['metrics']}
- **读法：** 索引级摘要；逐项对照与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [12 篇技术地图](../overview/vla-deploy-12-papers-technology-map.md)；与同 arXiv 节点不重复造页。

## 结论

**{p['conclusion']}**

1. 开源边界：**{p['open']}** — 以项目页实际链接为准（入库日 {TODAY}）。
2. 核心机制：{p['mechanism'][:80]}…
3. 部署前核对任务协议与硬件条件，勿直接横比公众号摘录数字。

## 关联页面

{_yaml_list([f"[{r.split('/')[-1].replace('.md','')}]({r})" for r in p['related']], 0)}

## 参考来源

- [{slug}_arxiv_{ax.replace('.', '_')}.md](../../sources/papers/{slug}_arxiv_{ax.replace('.', '_')}.md)
- [{BLOG}](../../sources/blogs/{BLOG})
- [arXiv:{ax}](https://arxiv.org/abs/{ax})

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/{ax})
{proj_line}{code_read}
"""


def _repo(p: dict) -> str:
    return f"""# {p['short']} 官方仓库

> 来源归档（repo）

- **标题：** {p['title']}
- **类型：** repo
- **链接：** {p['code']}
- **arXiv：** <https://arxiv.org/abs/{p['arxiv']}>
- **入库日期：** {TODAY}
- **一句话说明：** {p['one_liner']}
- **沉淀到 wiki：** [`wiki/entities/paper-{p['slug']}.md`](../../wiki/entities/paper-{p['slug']}.md)

## 开源状态

- **{p['open']}**：公开仓库（以 README 与 release 为准）。
"""


def _site(p: dict) -> str:
    return f"""# {p['short']} 项目页

> 来源归档（site）

- **标题：** {p['title']}
- **类型：** site
- **链接：** {p['project']}
- **arXiv：** <https://arxiv.org/abs/{p['arxiv']}>
- **入库日期：** {TODAY}
- **一句话说明：** {p['one_liner']}
- **沉淀到 wiki：** [`wiki/entities/paper-{p['slug']}.md`](../../wiki/entities/paper-{p['slug']}.md)

## 开源状态

- **{p['open']}**（步骤 2.5 核查，{TODAY}）。
"""


def _blog() -> str:
    rows = []
    n = 1
    rows.append(
        "| 01 | FluxVLA Engine | [2609.17210](https://arxiv.org/abs/2609.17210) | **已开源** | [fluxvla-engine](../../wiki/entities/fluxvla-engine.md)（**复用**） |"
    )
    for p in PAPERS:
        if p["slug"] == "ressafe":
            continue
        n += 1
        wiki = f"[paper-{p['slug']}](../../wiki/entities/paper-{p['slug']}.md)"
        rows.append(
            f"| {n:02d} | {p['short']} | [{p['arxiv']}](https://arxiv.org/abs/{p['arxiv']}) | **{p['open']}** | {wiki} |"
        )
    # insert ResSafe at position 9
    table_body = "\n".join(rows[:8]) + "\n"
    table_body += "| 09 | ResSafe | [2609.15988](https://arxiv.org/abs/2609.15988) | **待发布** | [paper-ressafe](../../wiki/entities/paper-ressafe.md)（**复用**） |\n"
    table_body += "\n".join(rows[8:])
    return f"""# 代码真的能帮部署吗？FluxVLA、JEPLO与DIDO等12篇机器人论文资源整理

> 来源归档（blog / 微信公众号）

- **标题：** 代码真的能帮部署吗？FluxVLA、JEPLO与DIDO等12篇机器人论文资源整理
- **类型：** blog
- **作者：** 具身智能小站（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/nsAslK7HCyhUaViGkSVgWA
- **发表日期：** 2026-09-16
- **入库日期：** {TODAY}
- **抓取方式：** wechat-article-for-ai（Camoufox）
- **原始抓取落盘：** [`sources/raw/wechat_embodied_station_12_papers_vla_deploy_2026-09-16.md`](../raw/wechat_embodied_station_12_papers_vla_deploy_2026-09-16.md)
- **一句话说明：** 12 篇具身论文盘点，覆盖 VLA 工程部署、LiDAR 足式、一步 WAM 蒸馏、对象级 DiT、系统韧性、人形安全过滤与语义通信等；**12/12 独立详情节点**（本 ingest **新建 10**、**复用 2**）。

## 12 篇 → 本库节点

| # | 论文 | arXiv | 开源结论 | wiki |
|---|------|-------|----------|------|
{table_body}

## 对 wiki 的映射

- **12/12 独立详情节点**；**0 重复 arXiv 节点**
- 阅读坐标：[VLA 部署 12 篇技术地图](../../wiki/overview/vla-deploy-12-papers-technology-map.md)
- 交叉：[VLA](../../wiki/methods/vla.md)、[World Action Models](../../wiki/concepts/world-action-models.md)、[Safety Filter](../../wiki/concepts/safety-filter.md)、[Locomotion](../../wiki/tasks/locomotion.md)

## 当前提炼状态

- [x] 公众号正文抓取
- [x] 12 篇独立节点（10 新建 / 2 复用）
- [x] 项目页/仓库开源状态核查（步骤 2.5）
"""


def _map() -> str:
    return f"""---
type: overview
tags: [overview, survey, vla, deployment, world-model, humanoid, technology-map]
status: complete
updated: {TODAY}
related:
  - ../entities/fluxvla-engine.md
  - ../entities/paper-jeplo.md
  - ../entities/paper-dido-wam.md
  - ../entities/paper-ressafe.md
  - ../methods/vla.md
  - ../concepts/world-action-models.md
sources:
  - ../../sources/blogs/{BLOG}
summary: "具身智能小站 2026-09-16 十二篇盘点：VLA 工程闭环、退化感知足式、一步 WAM、系统韧性与语义通信五条阅读线。"
---

# VLA 部署与系统可靠性：12 篇论文阅读坐标

> **本页定位**：为 [具身智能小站 · 12 篇盘点](https://mp.weixin.qq.com/s/nsAslK7HCyhUaViGkSVgWA)（2026-09-16）提供按问题组织的阅读坐标。

## 一句话观点

**「代码能否帮部署」取决于工程契约是否统一、感知退化是否被评测、以及安全/通信是否进入闭环——而非再多一个策略结构。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| WAM | World Action Model | 世界–动作联合模型 |
| WBC | Whole-Body Control | 人形全身控制 |
| GoC | Goal-Oriented Communications | 目标导向通信 |

## 为什么单独做这张地图

- 一次列出 12 篇方向跨度大的工作；需要横切面索引。
- **12/12 独立节点**：新建 10 + 复用 FluxVLA Engine、ResSafe；**0 重复 arXiv**。

## 流程总览

```mermaid
flowchart TB
  subgraph ENG["工程与部署"]
    FV[FluxVLA Engine]
    RR[RobResilience]
    GC[Goal-Oriented Comms]
  end
  subgraph WM["世界模型与策略"]
    DIDO[DIDO 一步 WAM]
    WB[WholeBodyWAM]
    WLA[WLA³ 潜动作]
    SD[SlotDiT]
  end
  subgraph PER["感知与操作"]
    JP[JEPLO LiDAR]
    PX[ProxiDex]
    SP[StereoPatch]
  end
  subgraph SAFE["安全"]
    RS[ResSafe]
  end
  ENG --> GOAL[可部署具身系统]
  WM --> GOAL
  PER --> GOAL
  SAFE --> GOAL
```

## 分组索引

### VLA 工程与系统

| 论文 | 节点 | 开源 |
|------|------|------|
| FluxVLA Engine | [fluxvla-engine](../entities/fluxvla-engine.md) | 已开源 |
| RobResilience | [paper-robresilience](../entities/paper-robresilience.md) | 已开源 |
| Goal-Oriented Comms | [paper-goal-oriented-comms-physical-ai](../entities/paper-goal-oriented-comms-physical-ai.md) | 待发布 |

### 世界模型与动作

| 论文 | 节点 | 开源 |
|------|------|------|
| DIDO | [paper-dido-wam](../entities/paper-dido-wam.md) | 已开源 |
| WholeBodyWAM | [paper-wholebodywam](../entities/paper-wholebodywam.md) | 待发布 |
| WLA³ | [paper-wla3](../entities/paper-wla3.md) | 待发布 |
| SlotDiT | [paper-slotdit](../entities/paper-slotdit.md) | 待发布 |

### 感知与足式

| 论文 | 节点 | 开源 |
|------|------|------|
| JEPLO | [paper-jeplo](../entities/paper-jeplo.md) | 已开源 |
| ProxiDex | [paper-proxidex](../entities/paper-proxidex.md) | 待发布 |
| StereoPatch | [paper-stereopatch](../entities/paper-stereopatch.md) | 部分开源 |

### 安全与其它

| 论文 | 节点 | 开源 |
|------|------|------|
| ResSafe | [paper-ressafe](../entities/paper-ressafe.md) | 待发布 |
| Machine Zygote | [paper-machine-zygote](../entities/paper-machine-zygote.md) | 已开源 |

## 关联页面

- [VLA](../methods/vla.md)
- [World Action Models](../concepts/world-action-models.md)
- [Safety Filter](../concepts/safety-filter.md)
- [VLA 部署指南](../queries/vla-deployment-guide.md)

## 参考来源

- [{BLOG}](../../sources/blogs/{BLOG})

## 推荐继续阅读

- [FluxVLA Engine](../entities/fluxvla-engine.md)
- [ResSafe](../entities/paper-ressafe.md)
"""


def _patch_reuse() -> None:
    fv = ROOT / "wiki/entities/fluxvla-engine.md"
    text = fv.read_text(encoding="utf-8")
    if 'arxiv: "2609.17210"' not in text:
        text = text.replace("updated: 2026-07-16", f'updated: {TODAY}\narxiv: "2609.17210"')
    blog_src = f"  - ../../sources/blogs/{BLOG}\n"
    if BLOG not in text:
        text = text.replace(
            "  - ../../sources/repos/fluxvla.md\n",
            f"  - ../../sources/repos/fluxvla.md\n{blog_src}",
        )
    if "12 篇技术地图" not in text:
        text = text.replace(
            "  - ../queries/vla-deployment-guide.md\n",
            "  - ../queries/vla-deployment-guide.md\n  - ../overview/vla-deploy-12-papers-technology-map.md\n",
        )
    fv.write_text(text, encoding="utf-8")

    rs = ROOT / "wiki/entities/paper-ressafe.md"
    rt = rs.read_text(encoding="utf-8")
    if BLOG not in rt:
        rt = rt.replace(
            "  - ../../sources/sites/ressafe-sciautonomy.md\n",
            f"  - ../../sources/sites/ressafe-sciautonomy.md\n  - ../../sources/blogs/{BLOG}\n",
        )
    if "vla-deploy-12-papers-technology-map" not in rt:
        rt = rt.replace(
            "  - ./paper-fmp-motion-priors.md\n",
            "  - ./paper-fmp-motion-priors.md\n  - ../overview/vla-deploy-12-papers-technology-map.md\n",
        )
    rt = rt.replace("updated: 2026-09-16", f"updated: {TODAY}")
    rs.write_text(rt, encoding="utf-8")


def main() -> None:
    RAW_DST.parent.mkdir(parents=True, exist_ok=True)
    if RAW_SRC.exists():
        shutil.copy2(RAW_SRC, RAW_DST)
    BLOG_PATH.write_text(_blog(), encoding="utf-8")
    MAP_PATH.write_text(_map(), encoding="utf-8")
    for p in PAPERS:
        ax_file = f"{p['slug']}_arxiv_{p['arxiv'].replace('.', '_')}.md"
        (ROOT / "sources/papers" / ax_file).write_text(_paper_source(p), encoding="utf-8")
        (ROOT / "wiki/entities" / f"paper-{p['slug']}.md").write_text(_entity(p), encoding="utf-8")
        if p.get("code"):
            (ROOT / "sources/repos" / f"{p['slug'].replace('-', '_')}.md").write_text(
                _repo(p), encoding="utf-8"
            )
        if p.get("project"):
            (ROOT / "sources/sites" / f"{p['slug']}.md").write_text(_site(p), encoding="utf-8")
    _patch_reuse()
    print(f"Wrote blog, map, {len(PAPERS)} new entities, patched 2 reuse nodes")


if __name__ == "__main__":
    main()
