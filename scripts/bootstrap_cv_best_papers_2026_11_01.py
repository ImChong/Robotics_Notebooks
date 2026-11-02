#!/usr/bin/env python3
"""Bootstrap ingest: 深蓝具身智能《四年十篇顶会Best Paper，谁在持续霸榜？》（2026-11-01）。

- 抓取方式：`scripts/fetch_wechat_article.py`（Chromium + 微信 UA）。
- 文章标题称「十篇」，正文实际列出 9 篇 CV 三大会 Best Paper；每篇一个独立 `wiki/entities/paper-*.md`：
  新建 7，复用 2（UniAD、D4RT 的 arXiv 已在库内）。
- arXiv 编号由 arXiv 摘要页核对标题；开源结论用 `git ls-remote` + 项目页 curl 核查（入库日 2026-11-01）。
- 用法：python3 scripts/bootstrap_cv_best_papers_2026_11_01.py <抓取的 txt>
"""

from __future__ import annotations

import re
import sys
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TODAY = date.today().isoformat()

ARTICLES: dict[str, dict[str, Any]] = {
    "a1": {
        "key": "cv_best_papers_2023_2026_labs",
        "title": "四年十篇顶会Best Paper，谁在持续霸榜？",
        "url": "https://mp.weixin.qq.com/s/CJki5-KhJ438RdAIhzNbCw",
        "date": "2026-11-01",
        "n": 9,
        "map": "cv-best-papers-2023-2026-technology-map.md",
        "map_title": "从视觉编程到 4D 重建：CV 三大会 2023–2026 Best Paper 阅读坐标",
        "map_summary": "深蓝具身智能 2026-11-01 盘点的 CV 三大会（CVPR / ICCV / ECCV）2023–2026 Best Paper：视觉编程、端到端驾驶、可控生成、静图动态、极简视觉、前馈 3D / 4D 重建与物理可搭建生成等 9 篇；9/9 独立节点（新建 7 / 复用 2）。",
        "view": "四年 Best Paper 的主线是计算机视觉从静态 2D 像素处理走向动态、符合物理规律的 3D / 4D 世界重构：视觉编程与可控生成之后，VGGT 把前馈几何做到统一，D4RT 再把它扩展到动态 4D；UniAD、Minimalist Vision、BrickGPT 分别代表端到端驾驶、极简传感与物理可行生成。",
    },
}


def blog_name(a: str) -> str:
    return f"wechat_shenlan_{ARTICLES[a]['key']}.md"


M = "../methods/"
C = "../concepts/"
VRP = C + "visual-representation-for-policy.md"
STATE = C + "state-estimation.md"
SPATIAL = C + "3d-spatial-vqa.md"
GWM = M + "generative-world-models.md"

OPEN = "已开源"
PART = "部分开源"
TBD = "待发布"
NOLINK = "项目页未列 GitHub"

REUSE: dict[str, str] = {
    "2212.10156": "paper-uniad",
    "2512.08924": "paper-d4rt",
}

GROUP_TAGS = {
    "视觉推理与可控生成": ["computer-vision", "generative"],
    "3D / 4D 几何重建": ["computer-vision", "3d-reconstruction"],
    "端到端自动驾驶": ["computer-vision", "autonomous-driving"],
    "计算成像与极简视觉": ["computer-vision", "computational-imaging"],
}

# 序号按文章出现顺序；venue 为公众号表述（未逐条核实奖项）
PAPERS_DATA: list[dict[str, Any]] = [
    dict(
        n=1,
        g="视觉推理与可控生成",
        short="VisProg",
        slug="visprog",
        arxiv="2211.11559",
        title="Visual Programming: Compositional Visual Reasoning Without Training",
        venue="CVPR 2023 Best Paper（公众号表述）",
        code="https://github.com/allenai/visprog",
        project="https://prior.allenai.org/projects/visprog",
        open=OPEN,
        open_note="仓库 allenai/visprog 经 git ls-remote 可达；项目页 HTTP 200。",
        one="VisProg 用大语言模型的上下文学习把自然语言指令写成 Python 式模块化视觉程序，调用现成视觉模型逐步求解，无需任务专门训练。",
        why="把大模型推理与传统视觉模块组合，得到可解释的执行过程（公众号归属：AI2 PRIOR 团队）。",
        mech="LLM 生成模块化程序，程序每一行调用检测、分割、VQA 等现成模型或 Python 函数，中间结果传给后续步骤，并给出可解释的推理依据（arXiv 摘要）。",
        metrics="arXiv 摘要称在 4 类任务上验证灵活性（含组合式视觉问答与图像编辑等）；具体数字未在本页转录。",
        concl="VisProg：神经符号路线——让 LLM 写程序而不是训练专用模型。",
        abbr=[
            ("VisProg", "Visual Programming", "大模型生成视觉程序的神经符号系统"),
            ("LLM", "Large Language Model", "大语言模型"),
            ("VQA", "Visual Question Answering", "视觉问答"),
        ],
        related=[VRP],
    ),
    dict(
        n=2,
        g="端到端自动驾驶",
        short="UniAD",
        arxiv="2212.10156",
        reuse=True,
        open=OPEN,
    ),
    dict(
        n=3,
        g="视觉推理与可控生成",
        short="ControlNet",
        slug="controlnet",
        arxiv="2302.05543",
        title="Adding Conditional Control to Text-to-Image Diffusion Models",
        venue="ICCV 2023 Best Paper（Marr Prize，公众号表述）",
        code="https://github.com/lllyasviel/ControlNet",
        project=None,
        open=OPEN,
        open_note="arXiv 摘要页链接的仓库 lllyasviel/ControlNet 经 git ls-remote 可达。",
        one="ControlNet 锁定预训练文生图扩散模型，复用其编码层作骨干，并经零初始化卷积（zero convolution）学习边缘、深度、分割、人体姿态等空间控制条件。",
        why="让大型扩散模型获得精确的空间可控性，是可控生成的基础组件（公众号归属：斯坦福 Brown 媒体创新研究所）。",
        mech="冻结生产级扩散模型，复制编码层作可训练分支，用零卷积逐步增加参数以避免有害噪声破坏微调（arXiv 摘要）。",
        metrics="arXiv 摘要称在 Stable Diffusion 上用单条件或多条件、带或不带文本提示测试多种控制；具体数字未在本页转录。",
        concl="ControlNet：锁住大模型、旁路学习条件控制。",
        abbr=[
            ("ControlNet", "Conditional Control Network", "给扩散模型加空间条件的网络结构"),
            ("SD", "Stable Diffusion", "开源文生图扩散模型"),
        ],
        related=[GWM],
    ),
    dict(
        n=4,
        g="视觉推理与可控生成",
        short="Generative Image Dynamics",
        slug="generative-image-dynamics",
        arxiv="2309.07906",
        title="Generative Image Dynamics",
        venue="CVPR 2024 Best Paper（公众号表述）",
        code=None,
        project="https://generative-dynamics.github.io/",
        open=NOLINK,
        open_note="项目页 HTTP 200，未见 GitHub 链接或 coming soon 字样（页面仅含项目页模板链接）。",
        one="Generative Image Dynamics 在傅里叶域学习图像空间的场景运动先验，由单张图预测频谱体并转成跨整段视频的运动纹理，让静图「动」起来。",
        why="用真实视频里树、花、烛火、衣物等振荡运动训练先验，可用于静图转循环视频与交互式动态（公众号归属：Google Research）。",
        mech="频率协调的扩散采样预测频谱体（spectral volume），转换为运动纹理，配合基于图像的渲染模块生成视频（arXiv 摘要）。",
        metrics="arXiv 摘要未给单一数字指标；本页不引用。",
        concl="Generative Image Dynamics：把运动建模成傅里叶域的图像空间先验。",
        abbr=[
            ("GID", "Generative Image Dynamics", "傅里叶域场景运动先验"),
            ("FFT", "Fast Fourier Transform", "频域表示的基础工具"),
        ],
        related=[GWM],
    ),
    dict(
        n=5,
        g="视觉推理与可控生成",
        short="Rich Human Feedback",
        slug="rich-human-feedback",
        arxiv="2312.10240",
        title="Rich Human Feedback for Text-to-Image Generation",
        venue="CVPR 2024 Best Paper（公众号表述）",
        code="https://github.com/google-research/google-research/tree/master/richhf_18k",
        project=None,
        open=OPEN,
        open_note="arXiv 摘要页链接 google-research/richhf_18k（数据集）；母仓库 google-research/google-research 经 git ls-remote 可达，子目录内容未逐文件核对。",
        one="Rich Human Feedback 把人类对文生图的反馈从单一分数扩展为可定位的区域标注（不合理或与文本不符的区域、对应词），并训练多模态模型预测这些细粒度反馈。",
        why="借鉴 LLM 的 RLHF，丰富 T2I 的反馈信号：区域标记 + 文本词标注 + 细粒度评分（公众号归属：Google Research 联合 UCSD 等）。",
        mech="标注数据集 RichHF-18K，并训练多模态 Transformer 预测富反馈，用于挑选训练数据、微调生成模型或做区域修复（arXiv 摘要）。",
        metrics="数据集名 RichHF-18K（名称暗示约 1.8 万条）；其余数字未转录。",
        concl="Rich Human Feedback：反馈要有空间位置，才能指导生成模型修哪里。",
        abbr=[
            ("RLHF", "Reinforcement Learning from Human Feedback", "基于人类反馈的强化学习"),
            ("T2I", "Text-to-Image", "文生图"),
            ("RichHF", "Rich Human Feedback", "富人类反馈"),
        ],
        related=[GWM],
    ),
    dict(
        n=6,
        g="计算成像与极简视觉",
        short="Minimalist Vision",
        slug="minimalist-vision-freeform-pixels",
        arxiv="2501.00142",
        title="Minimalist Vision with Freeform Pixels",
        venue="ECCV 2024 Best Paper（公众号表述）",
        code=None,
        project=None,
        open=NOLINK,
        open_note="arXiv 摘要页未列代码或项目页链接；本次未做进一步检索，不能判定是否开源。",
        one="Minimalist Vision 用任意形状的自由形态像素（光电探测器 + 光学掩膜）构成极简相机，把相机硬件建模成神经网络第一层，用极少像素完成特定视觉任务。",
        why="少即是多：用极少像素降低功耗并从物理层面保护隐私（公众号归属：哥伦比亚大学 CAVE 实验室，作者 Klotz 与 Nayar）。",
        mech="按任务训练网络得到各自由形态像素的形状；摘要称设计了监测室内空间（8 个像素）等极简相机原型（arXiv 摘要）。",
        metrics="公众号转述：用几个到十几个像素完成统计房间人数、检测门是否打开等任务；摘要称室内监测相机仅用 8 个像素。",
        concl="Minimalist Vision：传感器与网络联合设计，像素越少越要把硬件当网络层来训。",
        abbr=[
            ("Freeform Pixel", "Freeform Pixel", "任意形状的光电探测像素"),
            ("CAVE", "Computer Vision Laboratory (Columbia)", "哥伦比亚大学计算机视觉实验室"),
        ],
        related=[VRP],
    ),
    dict(
        n=7,
        g="3D / 4D 几何重建",
        short="VGGT",
        slug="vggt-visual-geometry-grounded-transformer",
        arxiv="2503.11651",
        title="VGGT: Visual Geometry Grounded Transformer",
        venue="CVPR 2025 Best Paper（公众号表述）",
        code="https://github.com/facebookresearch/vggt",
        project="https://vgg-t.github.io/",
        open=OPEN,
        open_note="仓库 facebookresearch/vggt 经 git ls-remote 可达；项目页 HTTP 200。",
        one="VGGT 是前馈神经网络，直接从一张、几张或上百张视图推出相机参数、点图、深度图与 3D 点轨迹等关键 3D 属性。",
        why="统一此前各自专用的 3D 视觉任务；摘要称重建不到 1 秒，且无需几何优化后处理仍优于需要后处理的方法（公众号归属：牛津 VGG 组 + Meta AI）。",
        mech="单个 Transformer 前馈输出相机参数、多视图深度、稠密点云与点跟踪（arXiv 摘要）。",
        metrics="arXiv 摘要称在相机参数估计、多视图深度估计、稠密点云重建、3D 点跟踪等任务达到 SOTA；具体数字未转录。",
        concl="VGGT：前馈统一多任务几何，D4RT 在其基础上扩展到动态 4D。",
        abbr=[
            ("VGGT", "Visual Geometry Grounded Transformer", "前馈统一 3D 几何的 Transformer"),
            ("SOTA", "State of the Art", "当前最优水平"),
        ],
        related=[STATE, SPATIAL],
    ),
    dict(
        n=8,
        g="视觉推理与可控生成",
        short="BrickGPT",
        slug="brickgpt",
        arxiv="2505.05469",
        title="Generating Physically Stable and Buildable Brick Structures from Text",
        venue="ICCV 2025 Best Paper（Marr Prize，公众号表述）",
        code="https://github.com/AvaLovelace1/BrickGPT",
        project="https://avalovelace1.github.io/BrickGPT/",
        open=OPEN,
        open_note="仓库 AvaLovelace1/BrickGPT 经 git ls-remote 可达；项目页 HTTP 200。",
        one="BrickGPT 从文本生成物理稳定、可互锁搭建的积木结构：自回归 LLM 逐块预测下一块，并在推理时用物理与装配约束做有效性检查和回滚。",
        why="把物理可行性放进生成过程，生成结果可由人或机器人实际搭建（公众号归属：CMU 计算机科学系）。",
        mech="构建大规模物理稳定积木数据集与配套文本描述，next-token 预测下一块，推理时用物理定律与装配约束剪除不可行的 token（arXiv 摘要）。",
        metrics="arXiv 摘要称生成结果稳定、多样、美观；具体数字未转录。",
        concl="BrickGPT：生成式模型加物理校验回滚，让输出可搭建。",
        abbr=[
            ("BrickGPT", "Brick Generative Pre-trained Transformer", "积木结构自回归生成模型"),
            ("LLM", "Large Language Model", "大语言模型"),
        ],
        related=[GWM],
    ),
    dict(
        n=9,
        g="3D / 4D 几何重建",
        short="D4RT",
        arxiv="2512.08924",
        reuse=True,
        open=NOLINK,
    ),
]


def _sentences(lines: list[str]) -> list[str]:
    out: list[str] = []
    for l in lines:
        l = re.sub(r"^(一句话判断|看点|结果|方法亮点)[：:]", "", l.lstrip("• ").strip())
        out += [s.strip() + "。" for s in re.split(r"。", l) if s.strip()]
    return out


def parse(txt: str) -> list[dict]:
    lines = [l.strip() for l in txt[txt.index("先看这一篇") :].split("\n")]
    blocks: list[dict] = []
    for l in lines:
        m = re.match(r"^(\d{1,2})([^\d\s].+)$", l)
        if m:
            blocks.append(dict(n=int(m.group(1)), head=m.group(2), lines=[]))
        elif blocks and l:
            blocks[-1]["lines"].append(l)
    skip = re.compile(
        r"^(另外 \d+ 篇|\d+ 篇一眼|下面每篇|关注本号|整理不易|具身智能小站|Like|Share|Boost|Comment|​)"
    )
    out = []
    for b in blocks:
        ls = b["lines"]
        links: dict[str, str] = {}
        body = []
        for l in ls[1:]:
            mm = re.match(r"^(论文|代码|项目页|资源)：(\S+)$", l)
            if mm:
                links[mm.group(1)] = mm.group(2)
            elif not skip.match(l) and l != "你可以带走的两点":
                body.append(l)
        out.append(dict(n=b["n"], head=b["head"], title=ls[0], body=body, links=links))
    return out


def _abbr(short: str, title: str) -> list[tuple[str, str, str]]:
    rows = [(short, title, "本文方法 / 基准的简称")]
    for key, full, zh in [
        ("Vision-Language-Action", "Vision-Language-Action", "视觉–语言–动作策略"),
        ("VLA", "Vision-Language-Action", "视觉–语言–动作策略"),
        ("VLM", "Vision-Language Model", "视觉语言模型"),
        ("UMI", "Universal Manipulation Interface", "手持通用操作采集接口"),
        ("WAM", "World Action Model", "世界–动作联合模型"),
        ("CSI", "Channel State Information", "WiFi 信道状态信息"),
        ("SysML", "Systems Modeling Language", "系统建模语言"),
        ("3D", "3D", "三维"),
    ]:
        if key in title and zh not in [r[2] for r in rows]:
            rows.append((key if key != "Vision-Language-Action" else "VLA", full, zh))
    return rows


def _yaml_list(items: list[str], indent: int = 2) -> str:
    pad = " " * indent
    return "\n".join(f"{pad}- {x}" for x in items)


def _ax_file(p: dict) -> str:
    return f"{p['slug'].replace('-', '_')}_arxiv_{p['arxiv'].replace('.', '_')}.md"


def _repo_file(p: dict) -> str:
    return f"{p['slug'].replace('-', '_')}.md"


def _site_file(p: dict) -> str:
    return f"{p['slug']}.md"


def _paper_source(p: dict) -> str:
    a = ARTICLES[p["a"]]
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
    if p.get("resource"):
        lines.append(f"- **资源：** <{p['resource']}>")
    lines += [
        f"- **入库日期：** {TODAY}",
        f"- **线索来源：** [{a['title']}](../blogs/{blog_name(p['a'])})（深蓝具身智能，{a['date']}）",
        f"- **一句话说明：** {p['one']}",
        "",
        "## 开源状态",
        "",
        f"- **{p['open']}**（步骤 2.5 核查，{TODAY}）：{p['open_note']}",
        "",
        "## 核心摘录（公众号转述，未核验原文表格）",
        "",
        f"- {p['why']}",
        f"- {p['mech']}",
        f"- 指标：{p['metrics']}",
        "",
        "## 对 wiki 的映射",
        "",
        f"- [paper-{p['slug']}](../../wiki/entities/paper-{p['slug']}.md)",
        f"- [{a['map_title']}](../../wiki/overview/{a['map']})",
    ]
    return "\n".join(lines) + "\n"


def _repo(p: dict) -> str:
    return f"""# {p["short"]} 官方仓库

> 来源归档（repo）

- **标题：** {p["title"]}
- **类型：** repo
- **链接：** {p["code"]}
- **arXiv：** <https://arxiv.org/abs/{p["arxiv"]}>
- **入库日期：** {TODAY}
- **一句话说明：** {p["one"]}
- **沉淀到 wiki：** [`wiki/entities/paper-{p["slug"]}.md`](../../wiki/entities/paper-{p["slug"]}.md)

## 开源状态

- **{p["open"]}**：{p["open_note"]} 仓库内容以 README 与 release 为准（本次仅用 `git ls-remote` 核实可达，未逐个阅读 README）。
"""


def _site(p: dict) -> str:
    code = f"- **代码：** <{p['code']}>\n" if p.get("code") else ""
    return f"""# {p["short"]} 项目页

> 来源归档（site）

- **标题：** {p["title"]}
- **类型：** site
- **链接：** {p["project"]}
- **arXiv：** <https://arxiv.org/abs/{p["arxiv"]}>
{code}- **入库日期：** {TODAY}
- **一句话说明：** {p["one"]}
- **沉淀到 wiki：** [`wiki/entities/paper-{p["slug"]}.md`](../../wiki/entities/paper-{p["slug"]}.md)

## 开源状态

- **{p["open"]}**（步骤 2.5 核查，{TODAY}）：{p["open_note"]}
"""


def _seq(p: dict) -> str:
    if p["open"] in (OPEN, PART) and p.get("code"):
        repo = p["code"].split("github.com/")[-1]
        return f"""## 源码运行时序图

以下为按仓库入口（`{repo}`）推断的通用运行链路；入库日仅核实仓库可达，未逐文件阅读，具体脚本名以 README 为准。

```mermaid
sequenceDiagram
  participant U as 用户
  participant R as {repo}
  participant E as 环境 / 数据
  participant M as 模型 / 策略
  U->>R: clone + 安装依赖（README）
  U->>E: 准备基准 / 数据集或真机接口
  U->>R: 训练或加载预训练权重
  R->>M: 构建 {p["short"]} 模型
  U->>R: 运行评测脚本
  R->>E: 重置环境、取观测
  E-->>M: 观测
  M-->>E: 动作 / 预测
  R-->>U: 指标与日志
```
"""
    if p["open"] == PART:
        return f"""## 源码运行时序图

**部分开源** — {p["open_note"]} 待可运行代码公开后补图。
"""
    return f"""## 源码运行时序图

**不适用（{p["open"]}）** — {p["open_note"]} 不编造运行脚本；代码公开后再补。
"""


def _entity(p: dict) -> str:
    ax = p["arxiv"]
    slug = p["slug"]
    a = ARTICLES[p["a"]]
    srcs = [f"../../sources/papers/{_ax_file(p)}"]
    if p.get("code"):
        srcs.append(f"../../sources/repos/{_repo_file(p)}")
    if p.get("project"):
        srcs.append(f"../../sources/sites/{_site_file(p)}")
    srcs.append(f"../../sources/blogs/{blog_name(p['a'])}")
    fm_extra = ""
    if p.get("project"):
        fm_extra += f"project: {p['project']}\n"
    if p.get("code"):
        fm_extra += f"code: {p['code']}\n"
    abbrev = "\n".join(f"| {x} | {y} | {z} |" for x, y, z in p["abbr"])
    links = []
    if p.get("project"):
        links.append(f"[项目页]({p['project']})")
    if p.get("code"):
        links.append(f"[代码]({p['code']})")
    if p.get("resource"):
        links.append(f"[资源]({p['resource']})")
    link_txt = ("，" + "，".join(links)) if links else ""
    related = p["related"] + [f"../overview/{a['map']}"]
    rel_lines = "\n".join(f"- [{r.split('/')[-1].replace('.md', '')}]({r})" for r in related)
    reads = [f"- [arXiv PDF](https://arxiv.org/pdf/{ax})"]
    if p.get("project"):
        reads.append(f"- [项目页]({p['project']})")
    if p.get("code"):
        reads.append(f"- [{p['code']}]({p['code']})")
    if p.get("resource"):
        reads.append(f"- [{p['resource']}]({p['resource']})")
    summary = f"{p['short']}（arXiv:{ax}）：{p['one']}".replace('"', "'")
    return f"""---
type: entity
project_id: {slug}
tags:
{_yaml_list(p["tags"], 2)}
status: complete
updated: {TODAY}
arxiv: "{ax}"
{fm_extra}related:
{_yaml_list(related, 2)}
sources:
{_yaml_list(srcs, 2)}
summary: "{summary}"
---

# {p["short"]}（arXiv:{ax}）

**{p["short"]}**（*{p["title"]}*，[arXiv:{ax}](https://arxiv.org/abs/{ax}){link_txt}）来自 [深蓝具身智能 {a["date"]} 盘点](../../sources/blogs/{blog_name(p["a"])})，属「{p["g"]}」组。

## 一句话定义

**{p["one"]}**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
{abbrev}

## 为什么重要

- {p["why"]}
- 开源结论：**{p["open"]}**（步骤 2.5，{TODAY}）。{p["open_note"]}
- 同期同组工作见 [{a["map_title"]}](../overview/{a["map"]})。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [{ax}](https://arxiv.org/abs/{ax}) |
| **开源** | **{p["open"]}** |
| **要点** | {p["mech"]} |
| **文内指标** | {p["metrics"]} |

{_seq(p)}
## 实验与评测

- {p["metrics"]}
- **读法：** 本页为索引级摘要，依据公众号转述的官方摘要与仓库信息整理；逐项数字与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [{a["map_title"]}](../overview/{a["map"]})；同 arXiv 不重复造页。

## 结论

**{p["concl"]}**

1. 开源边界：**{p["open"]}** — {p["open_note"]}
2. 核心机制：{p["mech"]}
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

{rel_lines}

## 参考来源

- [{_ax_file(p)}](../../sources/papers/{_ax_file(p)})
- [{blog_name(p["a"])}](../../sources/blogs/{blog_name(p["a"])})
- [arXiv:{ax}](https://arxiv.org/abs/{ax})

## 推荐继续阅读

{chr(10).join(reads)}
"""


def _real(a: str) -> list[dict]:
    return [p for p in PAPERS if p["a"] == a]


def _wiki_of(p: dict) -> str:
    return REUSE[p["arxiv"]] if p.get("reuse") else f"paper-{p['slug']}"


def _rows(a: str) -> list[tuple[str, str, str, str, str, str, bool]]:
    rows = []
    for p in _real(a):
        w = _wiki_of(p)
        reuse = bool(p.get("reuse"))
        tag = "（**复用**）" if reuse else ""
        tag2 = "（复用）" if reuse else ""
        rows.append(
            (
                p["g"],
                p["short"],
                p["arxiv"],
                p["open"],
                f"[{w}](../../wiki/entities/{w}.md){tag}",
                f"[{w}](../entities/{w}.md){tag2}",
                reuse,
            )
        )
    return rows


def _blog(a: str) -> str:
    A = ARTICLES[a]
    rows = _rows(a)
    new = sum(1 for r in rows if not r[6])
    body = "\n".join(
        f"| {i:02d} | {r[0]} | {r[1]} | [{r[2]}](https://arxiv.org/abs/{r[2]}) | **{r[3]}** | {r[4]} |"
        for i, r in enumerate(rows, 1)
    )
    return f"""# {A["title"]}

> 来源归档（blog / 微信公众号）

- **标题：** {A["title"]}
- **类型：** blog
- **作者：** 深蓝具身智能（微信公众号）
- **原始链接：** {A["url"]}
- **发表日期：** {A["date"]}
- **入库日期：** {TODAY}
- **抓取方式：** `scripts/fetch_wechat_article.py`（Chromium + 微信 UA）
- **原始抓取落盘：** [`sources/raw/wechat_shenlan_{A["key"]}.md`](../raw/wechat_shenlan_{A["key"]}.md)
- **一句话说明：** {A["view"]} 共 {A["n"]} 篇；**{A["n"]}/{A["n"]} 独立详情节点**（本 ingest **新建 {new}**、**复用 {len(rows) - new}**）。

## {A["n"]} 篇 → 本库节点

| # | 组 | 论文 | arXiv | 开源结论 | wiki |
|---|----|------|-------|----------|------|
{body}

## 对 wiki 的映射

- **{A["n"]}/{A["n"]} 独立详情节点**；同 arXiv 不重复造页（复用节点已补本文来源与地图关联）。
- 阅读坐标：[{A["map_title"]}](../../wiki/overview/{A["map"]})
- 开源结论口径：GitHub 仓库用 `git ls-remote` 核实可达；项目页用 curl 读取扫描 GitHub 链接与 "coming soon"；「项目页未列 GitHub」表示入库日页面无代码链接且未见开源承诺字样。

## 当前提炼状态

- [x] 公众号正文抓取（{TODAY}）
- [x] {A["n"]} 篇独立节点（{new} 新建 / {len(rows) - new} 复用）
- [x] 项目页 / 仓库开源状态核查（步骤 2.5）
- [ ] 原文实验表格核验（节点内数字仅为公众号转述）
"""


def _map(a: str) -> str:
    A = ARTICLES[a]
    rows = _rows(a)
    groups: dict[str, list] = {}
    for r in rows:
        groups.setdefault(r[0], []).append(r)
    sections = []
    for g, rs in groups.items():
        tbl = "\n".join(f"| {r[1]} | {r[5]} | {r[3]} |" for r in rs)
        sections.append(
            f"### {g}（{len(rs)} 篇）\n\n| 论文 | 节点 | 开源 |\n|------|------|------|\n{tbl}\n"
        )
    related = [f"../entities/{_wiki_of(p)}.md" for p in _real(a)[:3]] + []
    flow_nodes = "\n".join(
        f'  G{i}["{g}（{len(rs)}）"] --> GOAL' for i, (g, rs) in enumerate(groups.items())
    )
    return f"""---
type: overview
tags: [overview, survey, computer-vision, best-paper, technology-map, wechat]
status: complete
updated: {TODAY}
related:
{_yaml_list(related, 2)}
sources:
  - ../../sources/blogs/{blog_name(a)}
summary: "{A["map_summary"]}"
---

# {A["map_title"]}

> **本页定位**：为 [深蓝具身智能 · {A["title"]}]({A["url"]})（{A["date"]}）提供按研究方向组织的阅读坐标；{A["n"]} 篇全部有独立详情节点。

## 一句话观点

**{A["view"]}**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| CVPR | IEEE/CVF Conference on Computer Vision and Pattern Recognition | 计算机视觉三大会之一 |
| ICCV | International Conference on Computer Vision | 计算机视觉三大会之一 |
| ECCV | European Conference on Computer Vision | 计算机视觉三大会之一 |

## 为什么单独做这张地图

- 公众号按年份盘点近四年 CV 三大会 Best Paper（{A["n"]} 篇有独立节点；标题称「十篇」，正文实际列出 {A["n"]} 篇），按年份排列，缺少按主题的横切面；本页按主题重组。
- 节点摘要依据公众号编译与 arXiv 摘要页；Best Paper 奖项与实验室归属为公众号表述，未逐条核实；开源结论为入库日核查结果，见各节点「为什么重要」。

## 流程总览

```mermaid
flowchart TB
{flow_nodes}
  GOAL[可部署具身系统]
```

## 分组索引

{chr(10).join(sections)}
## 关联页面

- [Visual Representation for Policy](../concepts/visual-representation-for-policy.md)

## 参考来源

- [{blog_name(a)}](../../sources/blogs/{blog_name(a)})

## 推荐继续阅读

- [{rows[0][1]}]({rows[0][5].split("](")[1].split(")")[0]})
- [{rows[1][1]}]({rows[1][5].split("](")[1].split(")")[0]})
"""


def _patch_reuse(a: str) -> None:
    """复用节点：补本期博客来源与地图关联（幂等）。"""
    blog = f"  - ../../sources/blogs/{blog_name(a)}\n"
    mapl = f"  - ../overview/{ARTICLES[a]['map']}\n"
    for w in REUSE.values():
        path = ROOT / "wiki/entities" / f"{w}.md"
        text = path.read_text(encoding="utf-8")
        head, sep, body = text.partition("\n---\n")
        if not sep:
            raise SystemExit(f"frontmatter 缺失: {path}")
        lines = head.split("\n")

        def _append(key: str, item: str) -> None:
            if item.strip() in (l.strip() for l in lines):
                return
            try:
                i = next(i for i, l in enumerate(lines) if l.startswith(f"{key}:"))
            except StopIteration:
                lines.append(f"{key}:")
                lines.append(item.rstrip("\n"))
                return
            if lines[i].strip() != f"{key}:":
                raise SystemExit(f"{path}: {key} 为行内写法，需手工补")
            j = i + 1
            while j < len(lines) and lines[j].startswith("  "):
                j += 1
            lines.insert(j, item.rstrip("\n"))

        _append("sources", blog)
        _append("related", mapl)
        lines = [f"updated: {TODAY}" if l.startswith("updated:") else l for l in lines]
        path.write_text("\n".join(lines) + sep + body, encoding="utf-8")


def build_papers() -> list[dict]:
    out = []
    for d in PAPERS_DATA:
        p = dict(d, a="a1")
        if not p.get("reuse"):
            p.setdefault("resource", None)
            p["tags"] = ["paper"] + GROUP_TAGS[p["g"]]
        out.append(p)
    return out


def main() -> None:
    raw = Path(sys.argv[1])
    txt = raw.read_text(encoding="utf-8")
    global PAPERS
    PAPERS = build_papers()
    if len(PAPERS) != 9:
        raise SystemExit(f"应为 9 篇，实际 {len(PAPERS)}")
    new = [p for p in PAPERS if not p.get("reuse")]
    seen: set[str] = set()
    slugs: set[str] = set()
    for p in PAPERS:
        if p["arxiv"] in seen:
            raise SystemExit(f"重复 arXiv: {p['arxiv']}")
        seen.add(p["arxiv"])
        if p.get("reuse"):
            if not (ROOT / "wiki/entities" / f"{REUSE[p['arxiv']]}.md").exists():
                raise SystemExit(f"复用节点不存在: {p['arxiv']}")
            continue
        if p["slug"] in slugs:
            raise SystemExit(f"slug 重复: {p['slug']}")
        slugs.add(p["slug"])
        if (ROOT / "wiki/entities" / f"paper-{p['slug']}.md").exists():
            raise SystemExit(f"节点已存在: {p['slug']}")
        for rel in p["related"]:
            if not (ROOT / "wiki/entities" / rel).resolve().exists():
                raise SystemExit(f"关联页不存在: {rel} ({p['slug']})")
    for a, A in ARTICLES.items():
        dst = ROOT / "sources/raw" / f"wechat_shenlan_{A['key']}.md"
        dst.write_text(
            f"# {A['title']}\n\n> 原始抓取（{TODAY}，`scripts/fetch_wechat_article.py`），仅供溯源；来源：{A['url']}\n\n"
            + txt,
            encoding="utf-8",
        )
        (ROOT / "sources/blogs" / blog_name(a)).write_text(_blog(a), encoding="utf-8")
        (ROOT / "wiki/overview" / str(A["map"])).write_text(_map(a), encoding="utf-8")
        _patch_reuse(a)
    for p in new:
        (ROOT / "sources/papers" / _ax_file(p)).write_text(_paper_source(p), encoding="utf-8")
        (ROOT / "wiki/entities" / f"paper-{p['slug']}.md").write_text(_entity(p), encoding="utf-8")
        if p.get("code"):
            (ROOT / "sources/repos" / _repo_file(p)).write_text(_repo(p), encoding="utf-8")
        if p.get("project"):
            (ROOT / "sources/sites" / _site_file(p)).write_text(_site(p), encoding="utf-8")
    print(f"Wrote 1 blog, 1 map, {len(new)} new entities, patched {len(REUSE)} reuse nodes")


PAPERS: list[dict] = []

if __name__ == "__main__":
    main()
