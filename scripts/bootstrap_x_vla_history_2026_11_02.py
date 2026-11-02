#!/usr/bin/env python3
"""Bootstrap ingest: Khushi（@starlitmatcha）X Article《From RT1 to MolMoAct2: A brief history of VLAs》。

- 抓取：X 页面是 JS 壳，改用 api.fxtwitter.com 取推文与 Article 结构化正文（原文 JSON 落盘到 sources/raw）。
- 文中按小节介绍 10 个模型/数据集：SayCan、RT-1、RT-2、Open X-Embodiment(RT-X)、Octo、OpenVLA、π0、
  SmolVLA、MolmoAct、MolmoAct 2 → 10 个唯一节点：新建 2（SmolVLA、MolmoAct），复用 8。
- 用法：python3 scripts/bootstrap_x_vla_history_2026_11_02.py <article.md> <fx.json>
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TODAY = date.today().isoformat()
URL = "https://x.com/starlitmatcha/status/2116918927913804223"
BLOG = "x_starlitmatcha_vla_history_rt1_to_molmoact2_2026-11-01.md"
MAP = "vla-history-rt1-to-molmoact-technology-map.md"
MAP_TITLE = "VLA 演进简史：从 SayCan / RT-1 到 MolmoAct 2（Khushi 长文阅读坐标）"

# (序号, 文中名称, 节点, 新建?)
MODELS = [
    (1, "SayCan", "paper-saycan", False),
    (2, "RT-1", "paper-rt-1", False),
    (3, "RT-2", "paper-rt-2", False),
    (4, "Open X-Embodiment / RT-X", "paper-open-x-embodiment", False),
    (5, "Octo", "paper-octo", False),
    (6, "OpenVLA", "paper-openvla", False),
    (7, "π0", "paper-pi0", False),
    (8, "SmolVLA", "paper-smolvla", True),
    (9, "MolmoAct", "paper-molmoact", True),
    (10, "MolmoAct 2", "paper-molmoact2", False),
]

NEW: dict[str, dict[str, Any]] = {
    "paper-smolvla": dict(
        arxiv="2506.01844",
        name="SmolVLA",
        repo="huggingface/lerobot",
        tags=["paper", "vla", "lightweight", "flow-matching", "lerobot", "open-source"],
        one="SmolVLA 是小而高效、由社区数据驱动的 VLA：可在单 GPU 上训练、在消费级 GPU 甚至 CPU 上部署，并用解耦感知/动作预测与执行的异步推理栈提高控制频率。",
        abstract="论文摘要称：SmolVLA 是 small、efficient、community-driven 的 VLA，大幅降低训练与推理成本且保持有竞争力的性能；设计为可单 GPU 训练、消费级 GPU 或 CPU 部署；引入异步推理栈，使感知与动作预测与动作执行解耦，配合 chunked 动作生成获得更高控制率；性能可比 10 倍大的 VLA；并开源全部代码、预训练模型与训练数据。",
        article="文章把它作为「真正能拿来微调」的轻量选项：约 450M 参数（约为 OpenVLA 的 1/16），真实世界基础模型在不足 3 万条社区共享数据 episode 上预训练；基于 SmolVLM2（SigLIP 图像编码器 + SmolLM2 语言模型），与 π0 一样用 flow matching 输出连续动作 chunk；只用 VLM 语言解码器的前一半层、每图压缩为 64 个视觉 token；异步推理在执行当前 chunk 时就请求下一个 chunk，文章称吞吐大约翻倍、等待减少，代价是排队的动作基于更早的观测。",
        check="论文摘要与文章一致的部分：小型、社区数据、异步推理、flow matching 之外的「10 倍更大 VLA 可比」；文章里的 450M、前一半层、64 token、「吞吐翻倍」等具体数字来自文章，未在摘要中核对。lerobot 仓库 README 的 VLA 模型列表含 SmolVLA（文档 policy_smolvla_README），仓库 git ls-remote 可达。",
        related=[
            "../methods/vla.md",
            "../methods/action-chunking.md",
            "../concepts/flow-matching-embodied-policy.md",
            "../entities/lerobot.md",
            "../entities/paper-ros2smolvla.md",
            "../entities/paper-smolvla-onnx-libero.md",
            "../entities/paper-mamba-smolvla-expert.md",
            "../entities/paper-pi0.md",
            "../entities/paper-openvla.md",
        ],
        abbr=[
            ("VLA", "Vision-Language-Action", "视觉–语言–动作模型"),
            ("VLM", "Vision-Language Model", "视觉语言模型"),
            ("SigLIP", "Sigmoid Loss for Language-Image Pre-training", "图文匹配视觉编码器"),
        ],
        authors="Shukor, Aubakirova, Capuano, Kooijmans, Palma, Zouitine 等（arXiv 摘要页前 6 位）",
        date="2025-06-02",
    ),
    "paper-molmoact": dict(
        arxiv="2508.07917",
        name="MolmoAct",
        repo="allenai/molmoact",
        tags=["paper", "vla", "action-reasoning", "depth", "open-source", "ai2"],
        one="MolmoAct 提出 Action Reasoning Models（ARM）：把感知、规划与控制放进结构化三阶段——深度感知 token、可编辑的中层空间轨迹、低层动作——使行为可解释、可引导。",
        abstract="论文摘要称：MolmoAct 将观测与指令编码为深度感知 token，生成作为可编辑的轨迹的中层空间规划，再预测精确低层动作；MolmoAct-7B-D 在 SimplerEnv Visual Matching 零样本 70.5%，超过闭源 Pi-0 与 GR00T N1.5；LIBERO 平均成功率 86.6%，长时程比 ThinkAct 多 6.3%；真实世界微调较 Pi-0-FAST 单臂多 10%、双臂多 22.7% 任务进度；分布外泛化多 23.3%；并首次发布含 1 万+ 高质量轨迹的 MolmoAct Dataset（中期训练，平均带来 5.5% 提升），开源全部权重、训练代码与数据。",
        article="文章称其三阶段为：用 Depth Anything v2 与 VQ-VAE 把深度图压成 10×10 离散 token（作为训练目标，推理时不需再跑深度估计器）→ 预测至多 5 个图像空间点描述末端轨迹 → 输出 8 个离散化动作的 chunk；优点是能检查预测的深度与意图轨迹而非只看最终动作，缺点是生成中间 token 增加推理延迟，且「轨迹看着合理不等于执行正确」。文章称 MolmoAct 2 在其基础上提升空间推理、自适应深度预测并换成连续 flow-matching 动作专家。",
        check="摘要与文章一致：深度感知 token → 可编辑轨迹 → 动作的三阶段结构。文章的 10×10 token、5 个点、8 个动作 chunk 等细节来自文章，未在摘要中核对。allenai/molmoact 仓库 README 称模型与数据集已于 2025/08/12 发布，训练 / 微调 / 评估代码已释出，并于 2026/5/5 发布 MolmoAct2（allenai/molmoact2）。",
        related=[
            "../methods/vla.md",
            "../entities/paper-molmoact2.md",
            "../entities/molmo2-vlm.md",
            "../entities/paper-pi0.md",
            "../entities/paper-openvla.md",
        ],
        abbr=[
            ("ARM", "Action Reasoning Model", "动作推理模型"),
            ("VLA", "Vision-Language-Action", "视觉–语言–动作模型"),
            ("VQ-VAE", "Vector-Quantized Variational Autoencoder", "离散化潜变量自编码器"),
        ],
        authors="Lee, Duan, Fang, Deng, Liu, Li 等（arXiv 摘要页前 6 位）",
        date="2025-08-11",
    ),
}

SECTIONS = [
    (
        "为什么需要 VLA",
        "机器人在条件稍变（如料筐挪两英寸）就失效；VLM 能理解但不能动，经典控制能动但不泛化。VLA 的洞见：给 VLM 增加动作输出，用机器人示范端到端训练。",
    ),
    (
        "VLA 如何工作",
        "相机图像经视觉编码器成视觉 embedding，指令分词成文本 embedding，许多模型还输入关节角、夹爪开度等状态；Transformer 联合处理后由动作头预测末端位移、旋转与开合——可以是离散动作 token，也可以是连续值或 diffusion / flow matching，常一次预测多步动作 chunk；控制器执行后再观测、循环。",
    ),
    (
        "动作表示的三种路线",
        "离散动作 token（RT-1、RT-2、OpenVLA）→ 扩散出的连续动作 chunk（Octo）→ VLM + flow matching 动作专家（π0、SmolVLA、MolmoAct 2）。",
    ),
    (
        "开放问题（作者观点）",
        "(1) 示范只能教起点，机器人也应从自身错误中学习，VLA 与 RL 是互补而非对立；(2) 把 VLA 与世界模型结合，预测动作后果、比较候选动作；作者承认预测不准与额外计算延迟是风险。",
    ),
]


def _yl(items):
    return "\n".join(f"  - {x}" for x in items)


def _entity(node: str, p: dict) -> str:
    rel = [*p["related"], f"../overview/{MAP}"]
    abbr = "\n".join(f"| {a} | {b} | {c} |" for a, b, c in p["abbr"])
    rel_lines = "\n".join(f"- [{r.split('/')[-1][:-3]}]({r})" for r in rel)
    summ = p["one"].replace('"', "'")
    src = node.replace("paper-", "").replace("-", "_")
    return f"""---
type: entity
project_id: {node.replace("paper-", "")}
tags:
{_yl(p["tags"])}
status: complete
updated: {TODAY}
arxiv: "{p["arxiv"]}"
code: https://github.com/{p["repo"]}
related:
{_yl(rel)}
sources:
  - ../../sources/papers/{src}_arxiv_{p["arxiv"].replace(".", "_")}.md
  - ../../sources/blogs/{BLOG}
summary: "{p["name"]}（arXiv:{p["arxiv"]}）：{summ}"
---

# {p["name"]}

**{p["name"]}**（[arXiv:{p["arxiv"]}](https://arxiv.org/abs/{p["arxiv"]})，[GitHub](https://github.com/{p["repo"]})）是 [Khushi 的 VLA 演进长文]({URL}) 中单独成节介绍的模型之一。本页依据 arXiv 摘要页与该长文；技术细节以论文正文为准。

## 一句话定义

**{p["one"]}**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
{abbr}

## 为什么重要

- 在长文的演进线里属于「{"轻量、可微调的 flow-matching 路线" if "smol" in node else "显式空间推理（深度 → 轨迹 → 动作）路线"}」。
- 摘要要点：{p["abstract"]}

## 长文的讲法（作者表述）

{p["article"]}

## 核对

- **arXiv：** [{p["arxiv"]}](https://arxiv.org/abs/{p["arxiv"]})（提交 {p["date"]}），作者：{p["authors"]}
- {p["check"]}

## 关联页面

{rel_lines}

## 参考来源

- [{src}_arxiv_{p["arxiv"].replace(".", "_")}.md](../../sources/papers/{src}_arxiv_{p["arxiv"].replace(".", "_")}.md)
- [{BLOG}](../../sources/blogs/{BLOG})
- [arXiv:{p["arxiv"]}](https://arxiv.org/abs/{p["arxiv"]})
"""


def _paper_src(node: str, p: dict) -> str:
    return f"""# {p["name"]}（arXiv:{p["arxiv"]}）

> 来源归档（paper）

- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/{p["arxiv"]}>
- **PDF：** <https://arxiv.org/pdf/{p["arxiv"]}>
- **作者：** {p["authors"]}
- **提交：** {p["date"]}
- **代码：** <https://github.com/{p["repo"]}>（git ls-remote 可达）
- **入库日期：** {TODAY}
- **线索来源：** [{BLOG}](../blogs/{BLOG})

## arXiv 摘要要点（人工转述）

{p["abstract"]}

## 对 wiki 的映射

- [{node}](../../wiki/entities/{node}.md)
"""


def _blog(art: str) -> str:
    rows = "\n".join(
        f"| {n} | {name} | [{node}](../../wiki/entities/{node}.md) | {'新建' if new else '复用'} |"
        for n, name, node, new in MODELS
    )
    secs = "\n".join(f"- **{a}**：{b}" for a, b in SECTIONS)
    return f"""# From RT1 to MolMoAct2: A brief history of VLAs

> 来源归档（X Article / 个人博客）

- **标题：** From RT1 to MolMoAct2: A brief history of VLAs
- **类型：** blog（X Article）
- **作者：** Khushi（@starlitmatcha，简介「learning superduperintelligence for robotics」）
- **原始链接：** {URL}（推文）；文章 https://x.com/i/article/2116902127171932161
- **发表日期：** 2026-11-01
- **入库日期：** {TODAY}
- **抓取方式：** X 页面为 JS 壳，用第三方接口 `api.fxtwitter.com` 取推文与 Article 结构化内容（139 个文本块）；原始 JSON 落盘于 [`sources/raw/{BLOG.replace(".md", ".json")}`](../raw/{BLOG.replace(".md", ".json")})；文章内的图片与实验图未取回
- **一句话说明：** 个人学习笔记式长文，沿 SayCan → RT-1 → RT-2 → Open X-Embodiment/RT-X → Octo → OpenVLA → π0 → SmolVLA → MolmoAct / MolmoAct 2 梳理 VLA 架构演进，并附一个 RT-1 离散化的小实验。
- **说明：** 文中数字与论断为作者表述；与论文的核对见各模型节点。文中提到的 GPT-4V、Gemini、LLaVA、PaLI-X、PaLM-E、Prismatic、PaliGemma 等只作背景提及，不单独建节点。

## 文章结构与要点

{secs}

## 模型 → 节点（10 个模型，10 个唯一节点）

| # | 文中名称 | 节点 | 状态 |
|---|----------|------|------|
{rows}

## 离散化实验（作者表述）

作者取 3 条 LIBERO 示范共 843 个动作，把每个动作维度舍入到 16 / 32 / 64 / 128 / 256 个均匀格点，并比较「覆盖全范围」与「裁到 1–99 百分位」两种量化范围：16 格时裁剪略优；256 格时全范围的重构误差约低 5.5 倍；结论是动作分辨率取决于格点数与覆盖范围两者，且单步更准不保证长时程行为同比例更好。（实验图未取回，数字未复现。）

## 文章内的笔误 / 重复（已按原样处理）

- 作者把 Karol Hausman 写作「Carol Hausman」；OpenVLA 的离散化段落在正文中重复出现两次；这些不影响结论，页面不转录笔误。
- 文章把 MolmoAct 2 称为「更快、更实用」，具体指标文中未给。

## 对 wiki 的映射

- 阅读坐标：[{MAP_TITLE}](../../wiki/overview/{MAP})
"""


def _map() -> str:
    rows = "\n".join(
        f"| {n} | {name} | [{node}](../entities/{node}.md){'（新建）' if new else '（复用）'} |"
        for n, name, node, new in MODELS
    )
    secs = "\n".join(f"- **{a}**：{b}" for a, b in SECTIONS)
    rel = [f"../entities/{m[2]}.md" for m in MODELS[:4]] + [
        "../methods/vla.md",
        "../methods/action-chunking.md",
    ]
    return f"""---
type: overview
tags: [overview, vla, history, robot-learning, technology-map, x-article]
status: complete
updated: {TODAY}
related:
{_yl(rel)}
sources:
  - ../../sources/blogs/{BLOG}
summary: "Khushi 长文《From RT1 to MolMoAct2》的阅读坐标：按时间线把 SayCan、RT-1、RT-2、Open X-Embodiment、Octo、OpenVLA、π0、SmolVLA、MolmoAct、MolmoAct 2 十个模型对应到唯一节点，并梳理动作表示的三条路线。"
---

# {MAP_TITLE}

> **本页定位**：为 [Khushi（@starlitmatcha）的 X Article]({URL})（2026-11-01）提供按时间线组织的阅读坐标；文中 10 个模型各有唯一详情节点（新建 2 / 复用 8）。数字与论断为作者表述，各节点另有论文核对。

## 一句话观点

**作者的结论：没有单一赢家架构——不同设计在泛化、动作精度、推理延迟与所需机器人数据之间取舍；demo 越来越惊艳，但「做成一次」与「在不受控环境里可靠做成」之间仍有很大差距。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作模型 |
| VLM | Vision-Language Model | 视觉语言模型 |
| RT | Robotics Transformer | Google 的机器人 Transformer 系列 |
| OXE | Open X-Embodiment | 跨机器人开放数据集合作 |
| ARM | Action Reasoning Model | MolmoAct 提出的动作推理模型 |

## 时间线与节点

| # | 模型 | 节点 |
|---|------|------|
{rows}

## 作者的叙事线

{secs}

## 动作表示三条路线（作者归纳）

| 路线 | 代表 | 取舍（作者表述） |
|------|------|------------------|
| 离散动作 token | RT-1（256 bins，约 3 Hz）、RT-2、OpenVLA | 把动作当语言，简单；但量化范围与分辨率相互牵制（见来源页的离散化实验） |
| 扩散连续动作 chunk | Octo | 可表达多模态可行动作，避免均值化；Transformer 只跑一次、迭代去噪在小动作头内 |
| VLM + flow-matching 动作专家 | π0、SmolVLA、MolmoAct 2 | 兼得 VLM 语义与连续高频动作；每个 chunk 仍需多步精炼，chunk 内动作不一定用到新观测 |

## 关联页面

- [VLA 方法总览](../methods/vla.md)
- [动作分块 Action Chunking](../methods/action-chunking.md)
- [SayCan](../methods/saycan.md)
- [RT 系列](../methods/robotics-transformer-rt-series.md)
- [Octo 模型](../methods/octo-model.md)

## 参考来源

- [{BLOG}](../../sources/blogs/{BLOG})
- [原推文]({URL})
"""


def _patch(node: str, blog: str, mapl: str) -> None:
    path = ROOT / "wiki/entities" / f"{node}.md"
    text = path.read_text(encoding="utf-8")
    head, sep, body = text.partition("\n---\n")
    if not sep:
        raise SystemExit(f"frontmatter 缺失: {path}")
    lines = head.split("\n")

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
    _append("related", mapl)
    lines = [f"updated: {TODAY}" if x.startswith("updated:") else x for x in lines]
    path.write_text("\n".join(lines) + sep + body, encoding="utf-8")


def main() -> None:
    art = Path(sys.argv[1]).read_text(encoding="utf-8")
    fx = Path(sys.argv[2]).read_text(encoding="utf-8")
    for n, name, node, new in MODELS:
        exists = (ROOT / "wiki/entities" / f"{node}.md").exists()
        if new == exists:
            raise SystemExit(f"节点状态与预期不符: {node} exists={exists}")
    (ROOT / "sources/raw" / BLOG.replace(".md", ".json")).write_text(fx, encoding="utf-8")
    (ROOT / "sources/raw" / BLOG).write_text(
        f"# 原始抓取（{TODAY}，api.fxtwitter.com 转 Markdown）\n\n来源：{URL}\n\n{art}",
        encoding="utf-8",
    )
    (ROOT / "sources/blogs" / BLOG).write_text(_blog(art), encoding="utf-8")
    (ROOT / "wiki/overview" / MAP).write_text(_map(), encoding="utf-8")
    for node, p in NEW.items():
        src = node.replace("paper-", "").replace("-", "_")
        (ROOT / "sources/papers" / f"{src}_arxiv_{p['arxiv'].replace('.', '_')}.md").write_text(
            _paper_src(node, p), encoding="utf-8"
        )
        (ROOT / "wiki/entities" / f"{node}.md").write_text(_entity(node, p), encoding="utf-8")
    for n, name, node, new in MODELS:
        if not new:
            _patch(node, f"../../sources/blogs/{BLOG}", f"../overview/{MAP}")
    print(
        f"Wrote blog + map + {len(NEW)} new entities, patched {sum(1 for m in MODELS if not m[3])} reuse nodes"
    )


if __name__ == "__main__":
    main()
