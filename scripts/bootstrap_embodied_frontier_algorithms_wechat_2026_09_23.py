#!/usr/bin/env python3
"""Bootstrap ingest: 机器人研发工程师 · 具身智能前沿算法盘点（2026-09-23 公众号）."""

from __future__ import annotations

from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TODAY = date.today().isoformat()
BLOG = "wechat_robot_engineer_embodied_frontier_algorithms_2026-09-23.md"
BLOG_PATH = ROOT / "sources/blogs" / BLOG
RAW_DST = ROOT / "sources/raw/wechat_robot_engineer_embodied_frontier_algorithms_2026-09-23.md"
MAP_PATH = ROOT / "wiki/overview/embodied-frontier-algorithms-technology-map.md"
WX_URL = "https://mp.weixin.qq.com/s/JtoOU_ncZz5SEikmsZB3Xg"

# reuse: existing wiki/entities filename (no paper- prefix for cn-os/openvla)
REUSE: list[dict] = [
    {"short": "OpenVLA", "reuse": "openvla.md", "route": "VLA", "arxiv": "2406.09246", "open": "已开源"},
    {"short": "Octo", "reuse": "paper-octo.md", "route": "VLA", "arxiv": "2405.11172", "open": "已开源"},
    {"short": "π-0", "reuse": "paper-pi0.md", "route": "VLA", "arxiv": "2503.06669", "open": "已开源"},
    {"short": "π-0.5", "reuse": "paper-pi05-open-world-vla.md", "route": "VLA", "arxiv": "2503.06669", "open": "已开源"},
    {"short": "ViLLA", "reuse": "paper-shenlan-wm-05-villa-x.md", "route": "VLA", "arxiv": None, "open": "部分开源"},
    {"short": "GR00T N1", "reuse": "isaac-gr00t.md", "route": "VLA", "arxiv": None, "open": "已开源"},
    {"short": "Green-VLA", "reuse": "paper-greenvla-staged-vla-humanoid.md", "route": "VLA", "arxiv": None, "open": "部分开源"},
    {"short": "Gemini Robotics 2", "reuse": "gemini-robotics.md", "route": "VLA", "arxiv": None, "open": "未开源"},
    {"short": "DreamZero", "reuse": "paper-notebook-dreamzero-world-action-models-are-zero-shot-poli.md", "route": "WAM", "arxiv": "2602.15922", "open": "已开源"},
    {"short": "JEPA / V-JEPA 2", "reuse": "paper-vjepa2.md", "route": "WAM", "arxiv": "2506.09985", "open": "已开源"},
    {"short": "AMP", "reuse": "amp-for-hardware.md", "route": "Legged", "arxiv": "2109.05498", "open": "已开源"},
    {"short": "OpenWBT", "reuse": "cn-os-openwbt.md", "route": "Legged", "arxiv": None, "open": "已开源"},
    {"short": "RMA", "reuse": "paper-rma-rapid-motor-adaptation.md", "route": "Legged", "arxiv": "2104.08776", "open": "已开源"},
    {"short": "Diffusion Policy", "reuse": "paper-diffusion-policy.md", "route": "Manipulation", "arxiv": "2303.04137", "open": "已开源"},
    {"short": "RDT-1B", "reuse": "paper-rdt-1b.md", "route": "Manipulation", "arxiv": "2410.07835", "open": "已开源"},
    {"short": "GraspVLA", "reuse": "cn-os-graspvla.md", "route": "Manipulation", "arxiv": None, "open": "已开源"},
    {"short": "Tactile-VLA", "reuse": "paper-sa-2507-09160-tactile-vla-unlocking-vision-language-action-mod.md", "route": "Tactile", "arxiv": "2507.09160", "open": "已开源"},
    {"short": "ACE-Ego", "reuse": "paper-sa-2606-17200-ace-ego-0-unifying-egocentric-human-and-robotic.md", "route": "2026-VLA", "arxiv": "2606.11241", "open": "已开源"},
    {"short": "LingBot-VLA", "reuse": "lingbot-vla.md", "route": "2026-VLA", "arxiv": "2601.18692", "open": "已开源"},
    {"short": "DM0 / Dexbotic", "reuse": "dexmal-dm05.md", "route": "2026-VLA", "arxiv": "2602.08943", "open": "已开源"},
    {"short": "Motus", "reuse": "paper-sa-2512-13030-motus-a-unified-latent-action-world-model.md", "route": "2026-WAM", "arxiv": "2601.04278", "open": "已开源"},
    {"short": "EnerVerse-AC", "reuse": "paper-sa-2501-01895-enerverse-envisioning-embodied-future-space-for.md", "route": "2026-WAM", "arxiv": "2511.06722", "open": "已开源"},
    {"short": "LeTools", "reuse": "letools.md", "route": "2026-Loco", "arxiv": None, "open": "已开源"},
    {"short": "HumanTracker", "reuse": "paper-humantracker.md", "route": "2026-Loco", "arxiv": None, "open": "已开源"},
    {"short": "LIMMT / GQS", "reuse": "../methods/limmt-gqs-motion-curation.md", "route": "2026-Loco", "arxiv": "2606.06953", "open": "已开源"},
    {"short": "XR-1", "reuse": "cn-os-xr-1.md", "route": "VLA", "arxiv": "2511.02776", "open": "已开源"},
    {"short": "Awesome Legged Locomotion", "reuse": "awesome-legged-locomotion-learning.md", "route": "Awesome", "arxiv": None, "open": "已开源"},
    {"short": "Awesome Physical AI", "reuse": "awesome-physical-ai-natnew.md", "route": "Awesome", "arxiv": None, "open": "已开源"},
]

NEW_PAPERS: list[dict] = [
    {
        "slug": "tempowam",
        "title": "Rethink Before You Execute: Adaptive Execution for World Action Models",
        "short": "TempoWAM",
        "arxiv": "2608.09492",
        "code": None,
        "project": None,
        "open": "待发布",
        "route": "WAM",
        "tags": ["paper", "world-model", "wam", "replanning", "execution"],
        "one_liner": "RPM 监测任务进度 + AEP 按需重规划，替换 WAM 固定 action-chunk 执行步数；真机易任务 WAM 推理 −26.9%，难任务成功率 +13.3 pp。",
        "why": "固定前缀重规划与 chunk 可靠性随任务阶段变化不匹配；TempoWAM 是 plug-and-play 执行层而非重训骨干。",
        "mechanism": "Recurrent Progress Monitor 估计进度；Adaptive Execution Protocol 决定继续执行或丢弃剩余 chunk；per-task 校准因子在线适配。",
        "metrics": "LIBERO / RoboTwin / 真机；易任务维持成功率下减推理次数，难任务抬成功。",
        "conclusion": "TempoWAM 把 WAM 部署瓶颈从「预测多准」部分转成「何时重规划」；入库日未见独立官方仓库。",
        "related": ["../concepts/world-action-models.md", "../methods/generative-world-models.md", "./paper-fast-wam.md", "./paper-memorywam.md"],
        "abbrev": [
            ("TempoWAM", "Timing Execution by Monitoring Progress Online", "本文自适应 WAM 执行方案"),
            ("WAM", "World Action Model", "联合预测动作与环境演化"),
            ("RPM", "Recurrent Progress Monitor", "循环进度监测模块"),
            ("AEP", "Adaptive Execution Protocol", "自适应执行/重规划协议"),
        ],
    },
    {
        "slug": "embodiedbrain",
        "title": "EmbodiedBrain: Expanding Performance Boundaries of Task Planning for Embodied Intelligence",
        "short": "EmbodiedBrain",
        "arxiv": "2510.20578",
        "code": "https://github.com/ZTERobot/EmbodiedBrain1.0",
        "project": "https://zterobot.github.io/EmbodiedBrain.github.io/",
        "open": "已开源",
        "route": "Planning",
        "tags": ["paper", "vla", "planning", "agent", "zte", "step-grpo"],
        "one_liner": "7B/32B 具身规划 VLM：agent-aligned 数据结构 + 大规模 SFT + Step-GRPO（Guided Precursors）+ GRM 奖励；开源 VLM-PlanSim-99 仿真基准。",
        "why": "通用 LLM/VLM 与物理 agent 需求错位；离线规划榜难反映长时序失败恢复。",
        "mechanism": "Step-GRPO 把前序步骤作 Guided Precursors；三部分评测（General / Planning / E2E Sim）；VLM-PlanSim-99（AI2-THOR）。",
        "metrics": "多基准 SOTA（以原文为准）；HF EmbodiedBrain-7B 权重已发布。",
        "conclusion": "EmbodiedBrain 代表「规划专用 VLM + 真实仿真评测」路线；与 OpenVLA 等低层策略互补而非替代。",
        "related": ["../tasks/manipulation.md", "../methods/vla.md", "./paper-openeai-vla.md", "../overview/embodied-frontier-algorithms-technology-map.md"],
        "abbrev": [
            ("EmbodiedBrain", "EmbodiedBrain", "中兴 NebulaBrain 具身规划 VLM"),
            ("Step-GRPO", "Step-Augmented Group Relative Policy Optimization", "长时序 RL 微调"),
            ("GRM", "Generative Reward Model", "生成式奖励模型"),
            ("VLM", "Vision-Language Model", "视觉–语言多模态模型"),
        ],
    },
    {
        "slug": "openeai-vla",
        "title": "OpenEAI-Platform: An Open-source Embodied Artificial Intelligence Hardware-Software Unified Platform",
        "short": "OpenEAI-VLA",
        "arxiv": "2606.03392",
        "code": "https://github.com/sii-research/ORoboSoul",
        "project": None,
        "open": "待发布",
        "route": "2026-VLA",
        "tags": ["paper", "vla", "hardware", "teleoperation", "low-cost-arm"],
        "one_liner": "开源 6+1 DoF OpenEAI-Arm + Qwen3-VL-4B Diffusion Transformer VLA；两阶段仅用公开数据集预训练/后训练，对标 π₀ 成功率。",
        "why": "VLA 复现卡在专有数据与黑盒商业臂；OpenEAI 推硬件–软件–数据全链路开放。",
        "mechanism": "MDH 优化臂结构 + 动力学补偿 PID + rolling action-chunk 插值；VLM 骨干 + 生成式 action head；统一数据转换管线。",
        "metrics": "四任务真机；OpenEAI-Arm 在同策略下优于两款商业 6+1 臂（作者报告）。",
        "conclusion": "OpenEAI 是「低成本臂 + 小数据 VLA」复现参考；论文写 codes 录用后发布，入库日以 ORoboSoul 分支为准。",
        "related": ["../methods/vla.md", "./paper-pi0.md", "./lerobot.md", "../tasks/manipulation.md"],
        "abbrev": [
            ("OpenEAI", "Open-source Embodied AI", "本文硬件–软件统一平台"),
            ("VLA", "Vision-Language-Action", "视觉–语言–动作策略"),
            ("DoF", "Degrees of Freedom", "自由度"),
            ("VLM", "Vision-Language Model", "视觉–语言骨干"),
        ],
    },
    {
        "slug": "vla-adapter",
        "title": "VLA-Adapter: An Effective Paradigm for Tiny-Scale Vision-Language-Action Model",
        "short": "VLA-Adapter",
        "arxiv": "2509.09372",
        "code": "https://github.com/OpenHelix-Team/VLA-Adapter",
        "project": None,
        "open": "已开源",
        "route": "VLA",
        "tags": ["paper", "vla", "adapter", "lightweight", "bridge-attention"],
        "one_liner": "~0.5B 轻量 VLA：Bridge Attention 注入 VL 条件，低机器人预训练数据依赖；OpenHelix 开源训练/评测栈。",
        "why": "OpenVLA/π 族算力与数据门槛高；VLA-Adapter 提供单卡可试的适配范式（ReflexVLA 等亦作骨干）。",
        "mechanism": "小型 VLM + Bridge Attention 融合机器人状态/动作；强调跨本体低成本微调。",
        "metrics": "LIBERO 等榜（以原文与仓库 README 为准）；社区复现见 vla-open-source-repro-landscape-2025。",
        "conclusion": "VLA-Adapter 是轻量 VLA 工程基线之一；选型时勿与 OpenPI 数据规模假设混用。",
        "related": ["../methods/vla.md", "../overview/vla-open-source-repro-landscape-2025.md", "./openvla.md", "./paper-reflexvla.md"],
        "abbrev": [
            ("VLA-Adapter", "Vision-Language-Action Adapter", "本文轻量 VLA 范式"),
            ("VLA", "Vision-Language-Action", "视觉–语言–动作策略"),
            ("VL", "Vision-Language", "视觉–语言多模态"),
            ("LIBERO", "Lifelong Robot Learning Benchmark", "操作基准"),
        ],
    },
    {
        "slug": "memorywam",
        "title": "MemoryWAM: Efficient World Action Modeling with Persistent Memory",
        "short": "MemoryWAM",
        "arxiv": "2606.20562",
        "code": "https://github.com/yangsizhe/MemoryWAM",
        "project": "https://yangsizhe.github.io/MemoryWAM/",
        "open": "已开源",
        "route": "2026-WAM",
        "tags": ["paper", "world-model", "wam", "memory", "long-horizon"],
        "one_liner": "混合持久记忆 WAM：滑窗近期帧 + 任务起点 anchor + gist token 压缩长历史；推理复杂度 O(N)→O(N/d)，长时序家务优于 VLA/WAM 基线。",
        "why": "全历史 KV 缓存贵；纯滑窗在非 Markov 家务任务上丢进度。",
        "mechanism": "MoT 视频 DiT + 动作 DiT；gist token 蒸馏长程；推理跳过像素生成只更新 KV。",
        "metrics": "RMBench 等长时序记忆依赖任务（作者报告 83.0% 均值等）。",
        "conclusion": "MemoryWAM 代表 WAM 记忆结构工程化；与 TempoWAM 执行层、Fast-WAM 延迟优化可组合阅读。",
        "related": ["../concepts/world-action-models.md", "./paper-fast-wam.md", "./paper-tempowam.md", "./lingbot-vla.md"],
        "abbrev": [
            ("MemoryWAM", "Memory World Action Model", "本文带持久记忆的 WAM"),
            ("WAM", "World Action Model", "世界–动作联合模型"),
            ("KV", "Key-Value Cache", "注意力键值缓存"),
            ("MoT", "Mixture-of-Transformers", "多专家 Transformer"),
        ],
    },
    {
        "slug": "dexora",
        "title": "Dexora: Open-source VLA for High-DoF Bimanual Dexterity",
        "short": "Dexora",
        "arxiv": "2605.18722",
        "code": "https://github.com/dexoravla/Dexora",
        "project": "https://dexoravla.github.io/",
        "open": "已开源",
        "route": "2026-Manipulation",
        "tags": ["paper", "vla", "bimanual", "dexterous", "icra-2026"],
        "one_liner": "首个开源 36-DoF 双臂双手端到端 VLA：外骨骼臂 + Vision Pro 指跟踪遥操作；判别器加权 Diffusion Transformer；灵巧任务 66.7% vs 51.7%。",
        "why": "高 DoF 双手操纵缺可复现端到端 VLA；低维夹爪方案无法覆盖拧瓶盖/叠衣。",
        "mechanism": "混合遥操作 + MuJoCo 数字孪生；100K sim + 10K+ 真机轨迹；clip 级质量判别降权噪声示范。",
        "metrics": "基础任务 90% 成功；灵巧平均 66.7%（作者报告）；ICRA 2026。",
        "conclusion": "Dexora 把高 DoF 双手 VLA 拉到可开源复现；部署前核对 LeRobot v2.1 数据 schema 与硬件栈。",
        "related": ["../methods/vla.md", "../tasks/manipulation.md", "./paper-rdt-1b.md", "./cn-os-graspvla.md"],
        "abbrev": [
            ("Dexora", "Dexora VLA", "本文 36-DoF 双手 VLA"),
            ("VLA", "Vision-Language-Action", "视觉–语言–动作策略"),
            ("DoF", "Degrees of Freedom", "自由度"),
            ("DiT", "Diffusion Transformer", "扩散 Transformer 策略头"),
        ],
    },
    {
        "slug": "fast-wam",
        "title": "Fast-WAM: Do World Action Models Need Test-time Future Imagination?",
        "short": "Fast-WAM",
        "arxiv": "2603.16666",
        "code": "https://github.com/yuantianyuan01/FastWAM",
        "project": "https://yuantianyuan01.github.io/FastWAM/",
        "open": "已开源",
        "route": "WAM",
        "tags": ["paper", "world-model", "wam", "latency", "galaxea", "tsinghua"],
        "one_liner": "训练期保留视频共训、推理期跳过未来视频去噪，190 ms 延迟（>4× 快于 imagine-then-execute WAM）；LIBERO 97.6% / RoboTwin 91.8%。",
        "why": "WAM 测试时迭代视频生成是延迟主因；需分离「训练表征收益」与「推理想象必要性」。",
        "mechanism": "Wan2.2-TI2V 视频骨干 + Action DiT；推理只过一遍 clean latent 直接出动作 chunk。",
        "metrics": "仿真与真机折毛巾等；无具身预训练仍 competitive（作者报告）。",
        "conclusion": "Fast-WAM 证明 WAM 价值可在训练期视频建模中兑现，推理不必每步想象；TempoWAM/MemoryWAM 是互补层。",
        "related": ["../concepts/world-action-models.md", "./paper-memorywam.md", "./paper-tempowam.md", "./paper-notebook-dreamzero-world-action-models-are-zero-shot-poli.md"],
        "abbrev": [
            ("Fast-WAM", "Fast World Action Model", "本文低延迟 WAM"),
            ("WAM", "World Action Model", "世界–动作联合模型"),
            ("DiT", "Diffusion Transformer", "扩散 Transformer"),
            ("LIBERO", "Lifelong Robot Learning Benchmark", "操作基准"),
        ],
    },
]

NEW_REPOS: list[dict] = [
    {
        "slug": "embodied-ai-daily",
        "short": "Embodied-AI-Daily",
        "code": "https://github.com/luohongk/Embodied-AI-Daily",
        "open": "已开源",
        "route": "Awesome",
        "one_liner": "具身智能每日顶会/arXiv 论文速递 issue 仓库；公众号盘点常作索引入口。",
        "related": ["../overview/embodied-frontier-algorithms-technology-map.md", "../methods/vla.md"],
    },
]


def _yaml_list(items: list[str], indent: int) -> str:
    pad = " " * indent
    return "\n".join(f"{pad}- {x}" for x in items)


def _raw_wechat() -> str:
    return f"""# 具身智能前沿算法（2026-2027最新，按技术路线分类）

> 原始抓取（WebFetch）。入库日 {TODAY}。原文：{WX_URL}

（正文见 sources/blogs/{BLOG}；WebFetch 抓取，部分 arXiv 编号与官方不一致已在 wiki 节点中校正。）
"""


def _wiki_link(reuse_path: str) -> str:
    if reuse_path.startswith("../"):
        return reuse_path.replace("../", "../../wiki/")
    return f"../../wiki/entities/{reuse_path}"


def _blog_table_rows() -> str:
    rows: list[str] = []
    n = 0
    for r in REUSE:
        n += 1
        ax = f"[{r['arxiv']}](https://arxiv.org/abs/{r['arxiv']})" if r.get("arxiv") else "—"
        link = _wiki_link(r["reuse"])
        name = r["reuse"].split("/")[-1].replace(".md", "")
        rows.append(
            f"| {n:02d} | {r['short']} | {r['route']} | {ax} | **{r['open']}** | [{name}]({link})（**复用**） |"
        )
    for p in NEW_PAPERS:
        n += 1
        rows.append(
            f"| {n:02d} | {p['short']} | {p['route']} | [{p['arxiv']}](https://arxiv.org/abs/{p['arxiv']}) | **{p['open']}** | [paper-{p['slug']}](../../wiki/entities/paper-{p['slug']}.md)（**新建**） |"
        )
    for r in NEW_REPOS:
        n += 1
        rows.append(
            f"| {n:02d} | {r['short']} | {r['route']} | — | **{r['open']}** | [cn-os-{r['slug']}](../../wiki/entities/cn-os-{r['slug']}.md)（**新建**） |"
        )
    return "\n".join(rows)


def _blog() -> str:
    total = len(REUSE) + len(NEW_PAPERS) + len(NEW_REPOS)
    new_n = len(NEW_PAPERS) + len(NEW_REPOS)
    return f"""# 具身智能前沿算法（2026-2027最新，按技术路线分类，附核心算法、开源项目、适用场景）

> 来源归档（blog / 微信公众号）

- **标题：** 具身智能前沿算法（2026-2027最新，按技术路线分类，附核心算法、开源项目、适用场景）
- **类型：** blog
- **作者：** 机器人研发工程师（微信公众号）
- **原始链接：** {WX_URL}
- **发表日期：** 2026-09-23（估）
- **入库日期：** {TODAY}
- **抓取方式：** WebFetch（Jina/Camoufox 不可用；agent-reach 本环境未预装 wechat 通道）
- **原始抓取落盘：** [`sources/raw/wechat_robot_engineer_embodied_frontier_algorithms_2026-09-23.md`](../raw/wechat_robot_engineer_embodied_frontier_algorithms_2026-09-23.md)
- **一句话说明：** 六大技术路线（VLA / WAM / 腿式 RL / 扩散操作 / 规划 Agent / 触觉力觉）+ 2026 新开源清单 + Awesome 索引；**{total}/{total} 独立详情节点**（**新建 {new_n}**、**复用 {len(REUSE)}**）。

## 节点映射

| # | 项目 | 路线 | arXiv | 开源 | wiki |
|---|------|------|-------|------|------|
{_blog_table_rows()}

## 对 wiki 的映射

- **{total}/{total} 独立详情节点**；**0 重复 arXiv 造页**（公众号部分 arXiv 已校正，见各实体页）
- 阅读坐标：[具身前沿算法技术地图](../../wiki/overview/embodied-frontier-algorithms-technology-map.md)
- 交叉：[VLA](../../wiki/methods/vla.md)、[World Action Models](../../wiki/concepts/world-action-models.md)、[Locomotion](../../wiki/tasks/locomotion.md)

## 当前提炼状态

- [x] 公众号正文抓取（WebFetch）
- [x] {total} 项独立节点（{new_n} 新建 / {len(REUSE)} 复用）
- [x] 项目页/仓库开源状态核查（步骤 2.5；部分条目为索引级）
"""


def _paper_source(p: dict) -> str:
    ax = p["arxiv"]
    slug = p["slug"]
    code_line = f"- **代码：** {p['code']}\n" if p.get("code") else ""
    proj_line = f"- **项目页：** {p['project']}\n" if p.get("project") else ""
    return f"""# {p['title']}

> 来源归档（paper）

- **标题：** {p['title']}
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/{ax}>
{code_line}{proj_line}- **入库日期：** {TODAY}
- **一句话说明：** {p['one_liner']}
- **沉淀到 wiki：** [`wiki/entities/paper-{slug}.md`](../../wiki/entities/paper-{slug}.md)
- **触发 ingest：** [`sources/blogs/{BLOG}`](../blogs/{BLOG})

## 开源状态（步骤 2.5）

- **{p['open']}**（核查日 {TODAY}）

## 对 wiki 的映射

- 实体页：[paper-{slug}](../../wiki/entities/paper-{slug}.md)
- 技术地图：[embodied-frontier-algorithms-technology-map](../../wiki/overview/embodied-frontier-algorithms-technology-map.md)
"""


def _entity(p: dict) -> str:
    ax = p["arxiv"]
    slug = p["slug"]
    abbrev = "\n".join(f"| {a} | {b} | {c} |" for a, b, c in p["abbrev"])
    src_paper = f"../../sources/papers/{slug}_arxiv_{ax.replace('.', '_')}.md"
    src_blog = f"../../sources/blogs/{BLOG}"
    code_line = f'code: {p["code"]}\n' if p.get("code") else ""
    repo_src = f"  - ../../sources/repos/{slug.replace('-', '_')}.md\n" if p.get("code") else ""
    site_src = f"  - ../../sources/sites/{slug}.md\n" if p.get("project") else ""
    kind = "执行层插件" if slug == "tempowam" else "策略或 WAM 训练栈"
    seq = (
        f"\n## 源码运行时序图\n\n"
        f"**不适用**（入库日 {TODAY}：{p['short']} 为{kind}，"
        f"以论文/仓库 README 训练–推理入口为准；非单一可运行管线时序图）。\n"
    )
    if p.get("code") and p["open"] == "已开源":
        seq = f"""
## 源码运行时序图

```mermaid
sequenceDiagram
    autonumber
    participant U as 用户/评测脚本
    participant R as {p['code'].split('/')[-1]} 仓库
    participant M as 模型权重
    participant E as 仿真/真机环境
    U->>R: clone + 依赖安装（见 README）
    U->>M: 下载 checkpoint（HF/Release）
    U->>R: train / eval 入口
    R->>E: rollout / 指标日志
    E-->>U: success / latency 等
```

图下说明：复现以 [`sources/repos/{slug.replace('-', '_')}.md`](../../sources/repos/{slug.replace('-', '_')}.md) 与官方 README 为准。
"""
    rel_links = "\n".join(f"- [{r.split('/')[-1].replace('.md', '')}]({r})" for r in p["related"])
    proj_line = f"- [项目页]({p['project']})\n" if p.get("project") else ""
    code_read = f"- [代码]({p['code']})\n" if p.get("code") else ""
    return f"""---
type: entity
tags:
{_yaml_list(p['tags'], 2)}
status: complete
updated: {TODAY}
arxiv: "{ax}"
{code_line}related:
{_yaml_list(p['related'] + ['../overview/embodied-frontier-algorithms-technology-map.md'], 2)}
sources:
  - {src_paper}
{repo_src}{site_src}  - {src_blog}
summary: "{p['short']}（arXiv:{ax}）：{p['one_liner'][:120]}"
---

# {p['short']}（arXiv:{ax}）

**{p['short']}**（*{p['title']}*，[arXiv:{ax}](https://arxiv.org/abs/{ax}){f"，[项目页]({p['project']})" if p.get('project') else ""}{f"，[代码]({p['code']})" if p.get('code') else ""}）来自 [机器人研发工程师 · 前沿算法盘点](../../sources/blogs/{BLOG})。

## 一句话定义

**{p['one_liner']}**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
{abbrev}

## 为什么重要

- {p['why']}
- 开源结论：**{p['open']}**（步骤 2.5，{TODAY}）。
- 与 [具身前沿算法技术地图](../overview/embodied-frontier-algorithms-technology-map.md) 同路线条目可横向对照。

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
- **读法：** 索引级摘要；逐项 baseline 以原文 PDF 为准。

## 结论

**{p['conclusion']}**

1. 开源边界：**{p['open']}** — 以项目页/仓库实际链接为准（入库日 {TODAY}）。
2. 核心机制：{p['mechanism'][:100]}…
3. 部署前核对硬件栈与评测协议，勿直接横比公众号摘录数字。

## 关联页面

{rel_links}

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

- **{p['open']}**（以 README 与 release 为准）。
"""


def _cn_os_repo(r: dict) -> str:
    slug = r["slug"]
    rel = "\n".join(f"- [{x.split('/')[-1].replace('.md', '')}]({x})" for x in r["related"])
    return f"""---
type: entity
tags: [curated-list, curator, awesome-list, embodied-ai]
status: complete
updated: {TODAY}
code: {r['code']}
related:
{_yaml_list(r['related'] + ['../overview/embodied-frontier-algorithms-technology-map.md'], 2)}
sources:
  - ../../sources/repos/{slug.replace('-', '_')}.md
  - ../../sources/blogs/{BLOG}
summary: "{r['one_liner'][:120]}"
---

# {r['short']}

**{r['short']}**（[GitHub 仓库]({r['code']})）是具身智能社区 **论文/issue 策展仓库**，被 [机器人研发工程师 · 前沿算法盘点](../../sources/blogs/{BLOG}) 列为持续追踪入口。

## 一句话定义

**{r['one_liner']}**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Daily | Daily Paper Digest | 每日论文速递 |
| arXiv | arXiv.org | 预印本论文库 |
| Repo | Repository | 代码/策展仓库 |

## 为什么重要

- 公众号盘点类 ingest 的 **时效索引**，不是单篇方法论文。
- 与 [Awesome Physical AI](./awesome-physical-ai-natnew.md) 等静态清单互补。

## 工程实践

| 项 | 内容 |
|----|------|
| **链接** | {r['code']} |
| **开源** | **{r['open']}** |
| **用途** | 跟踪 VLA/WAM/足式/操作新预印本与开源仓 |

## 局限与风险

- Issue 质量取决于维护者策展，不等同 peer-review。
- 链接与 arXiv 编号需二次核对（本仓库 ingest 已校正部分错误编号）。

## 关联页面

{rel}

## 参考来源

- [{slug.replace('-', '_')}.md](../../sources/repos/{slug.replace('-', '_')}.md)
- [{BLOG}](../../sources/blogs/{BLOG})

## 推荐继续阅读

- [Embodied-AI-Daily 仓库]({r['code']})
"""


def _repo_curator(r: dict) -> str:
    return f"""# {r['short']}

> 来源归档（repo / 策展）

- **标题：** {r['short']}
- **类型：** repo
- **链接：** {r['code']}
- **入库日期：** {TODAY}
- **一句话说明：** {r['one_liner']}
- **沉淀到 wiki：** [`wiki/entities/cn-os-{r['slug']}.md`](../../wiki/entities/cn-os-{r['slug']}.md)

## 开源状态

- **{r['open']}**
"""


def _map() -> str:
    def row_reuse(r: dict) -> str:
        name = r["reuse"].split("/")[-1].replace(".md", "")
        path = r["reuse"] if r["reuse"].startswith("../") else f"../entities/{r['reuse']}"
        return f"| {r['short']} | [{name}]({path}) | {r['open']} |"

    def row_new(p: dict) -> str:
        return f"| {p['short']} | [paper-{p['slug']}](../entities/paper-{p['slug']}.md) | {p['open']} |"

    routes = ["VLA", "WAM", "Legged", "Manipulation", "Planning", "Tactile", "2026-VLA", "2026-WAM", "2026-Manipulation", "2026-Loco", "Awesome"]
    sections = ""
    for route in routes:
        rs = [r for r in REUSE if r["route"] == route]
        ns = [p for p in NEW_PAPERS if p["route"] == route]
        nr = [r for r in NEW_REPOS if r["route"] == route]
        if not (rs or ns or nr):
            continue
        lines = [row_reuse(r) for r in rs] + [row_new(p) for p in ns]
        for r in nr:
            lines.append(f"| {r['short']} | [cn-os-{r['slug']}](../entities/cn-os-{r['slug']}.md) | {r['open']} |")
        sections += f"\n### {route}\n\n| 项目 | 节点 | 开源 |\n|------|------|------|\n" + "\n".join(lines) + "\n"

    total = len(REUSE) + len(NEW_PAPERS) + len(NEW_REPOS)
    return f"""---
type: overview
tags: [overview, survey, vla, wam, locomotion, manipulation, planning, technology-map]
status: complete
updated: {TODAY}
related:
  - ../entities/paper-vla-adapter.md
  - ../entities/paper-fast-wam.md
  - ../entities/openvla.md
  - ../methods/vla.md
  - ../concepts/world-action-models.md
sources:
  - ../../sources/blogs/{BLOG}
summary: "机器人研发工程师 2026 前沿算法盘点：VLA/WAM/足式/扩散操作/规划/触觉六路线 + 2026 新开源与 Awesome 索引；{total} 项独立节点。"
---

# 具身智能前沿算法：六路线阅读坐标

> **本页定位**：为 [机器人研发工程师 · 前沿算法盘点]({WX_URL})（2026-09-23）提供按 **VLA → WAM → 足式 → 操作 → 规划 → 触觉** 组织的阅读坐标。

## 一句话观点

**2026 具身算法的主线是「VLA 基座 + WAM 长程 + 足式/触觉/规划模块化」——选型应先定路线再定仓库，勿被公众号错误 arXiv 编号带偏。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作端到端策略 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| RL | Reinforcement Learning | 强化学习（足式控制常用） |
| DoF | Degrees of Freedom | 自由度 |

## 为什么单独做这张地图

- 一文横跨 **30+ 命名项目**，跨度大于典型「12 篇论文」盘点。
- **{total}/{total} 独立节点**：新建 {len(NEW_PAPERS) + len(NEW_REPOS)}、复用 {len(REUSE)}；**0 重复 arXiv 造页**。

## 流程总览

```mermaid
flowchart TB
  subgraph VLA["VLA 基座"]
    OV[OpenVLA / Octo / π]
    AD[VLA-Adapter]
    LB[LingBot-VLA]
  end
  subgraph WAM["世界动作模型"]
    FW[Fast-WAM]
    MW[MemoryWAM]
    TW[TempoWAM 执行层]
  end
  subgraph LOW["低层与操作"]
    DP[Diffusion Policy]
    DX[Dexora 36-DoF]
  end
  subgraph PLAN["规划"]
    EB[EmbodiedBrain]
  end
  VLA --> DEPLOY[可部署具身系统]
  WAM --> DEPLOY
  LOW --> DEPLOY
  PLAN --> DEPLOY
```

## 分组索引
{sections}

## 工程选型速查（公众号摘录）

| 场景 | 公众号建议组合 | 本库入口 |
|------|----------------|----------|
| 机械臂通用操作 | OpenVLA + Diffusion Policy | [openvla](../entities/openvla.md) + [Diffusion Policy](../entities/paper-diffusion-policy.md) |
| 人形双腿 | AMP + RMA + Humanoid-VLA | [AMP](../entities/amp-for-hardware.md) + [RMA](../entities/paper-rma-rapid-motor-adaptation.md) |
| 长时序任务 | VLA + WAM + LLM 规划 | [Fast-WAM](../entities/paper-fast-wam.md) + [EmbodiedBrain](../entities/paper-embodiedbrain.md) |
| 轮式家务人形 | ACE-Ego / LingBot + VLA-Adapter + Motus | 见 2026-VLA / 2026-WAM 表 |

## 关联页面

- [VLA](../methods/vla.md)
- [World Action Models](../concepts/world-action-models.md)
- [VLA 开源复现景观 2025](../overview/vla-open-source-repro-landscape-2025.md)
- [Locomotion](../tasks/locomotion.md)

## 参考来源

- [{BLOG}](../../sources/blogs/{BLOG})

## 推荐继续阅读

- [VLA-Adapter](../entities/paper-vla-adapter.md)
- [Fast-WAM](../entities/paper-fast-wam.md)
- [EmbodiedBrain](../entities/paper-embodiedbrain.md)
"""


def _log_entry() -> str:
    total = len(REUSE) + len(NEW_PAPERS) + len(NEW_REPOS)
    return f"""
## [{TODAY}] ingest | sources/blogs/{BLOG} — 机器人研发工程师前沿算法六路线盘点；{total} 项独立节点

- **触发：** 用户指定 {WX_URL}；要求每项目独立非重复详情节点；自动合并 PR
- **开源结论：** 新建 8 项中 VLA-Adapter / EmbodiedBrain / MemoryWAM / Dexora / Fast-WAM / Embodied-AI-Daily **已开源**；TempoWAM / OpenEAI **待发布**；复用项开源状态见各实体
- **关键页：** [embodied-frontier-algorithms-technology-map](wiki/overview/embodied-frontier-algorithms-technology-map.md)
"""


def main() -> None:
    RAW_DST.parent.mkdir(parents=True, exist_ok=True)
    RAW_DST.write_text(_raw_wechat(), encoding="utf-8")
    BLOG_PATH.write_text(_blog(), encoding="utf-8")
    MAP_PATH.write_text(_map(), encoding="utf-8")
    for p in NEW_PAPERS:
        ax_file = f"{p['slug']}_arxiv_{p['arxiv'].replace('.', '_')}.md"
        (ROOT / "sources/papers" / ax_file).write_text(_paper_source(p), encoding="utf-8")
        (ROOT / "wiki/entities" / f"paper-{p['slug']}.md").write_text(_entity(p), encoding="utf-8")
        if p.get("code"):
            (ROOT / "sources/repos" / f"{p['slug'].replace('-', '_')}.md").write_text(_repo(p), encoding="utf-8")
    for r in NEW_REPOS:
        (ROOT / "sources/repos" / f"{r['slug'].replace('-', '_')}.md").write_text(_repo_curator(r), encoding="utf-8")
        (ROOT / "wiki/entities" / f"cn-os-{r['slug']}.md").write_text(_cn_os_repo(r), encoding="utf-8")
    log = ROOT / "log.md"
    marker = f"[{TODAY}] ingest | sources/blogs/{BLOG}"
    if marker not in log.read_text(encoding="utf-8"):
        log.write_text(_log_entry() + log.read_text(encoding="utf-8"), encoding="utf-8")
    total = len(REUSE) + len(NEW_PAPERS) + len(NEW_REPOS)
    print(f"Wrote blog, map, {len(NEW_PAPERS)} paper entities, {len(NEW_REPOS)} repo entities ({total} mapped)")


if __name__ == "__main__":
    main()
