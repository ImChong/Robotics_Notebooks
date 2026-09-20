#!/usr/bin/env python3
"""Bootstrap ingest: 具身智能小站 11 篇论文（2026-09-20 约束控制盘点）."""

from __future__ import annotations

from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TODAY = date.today().isoformat()
BLOG = "wechat_embodied_station_11_papers_constraint_control_2026-09-20.md"
BLOG_PATH = ROOT / "sources/blogs" / BLOG
MAP_PATH = ROOT / "wiki/overview/constraint-control-11-papers-technology-map.md"
WX_URL = "https://mp.weixin.qq.com/s/RozDRLth62xgulo4ccIBMw"

REUSE: dict[str, dict[str, str]] = {
    "2609.18193": {
        "wiki": "wiki/entities/paper-wave-go.md",
        "label": "WAVE-Go",
        "open": "已开源",
        "code": "https://github.com/vigorlee/wave-go",
    },
    "2609.19138": {
        "wiki": "wiki/entities/paper-gpt-policy.md",
        "label": "GPT-Policy",
        "open": "已开源",
    },
    "2609.18197": {
        "wiki": "wiki/entities/paper-wholebodywam-unimotion-4k.md",
        "label": "WholeBodyWAM",
        "open": "待发布",
    },
    "2609.18651": {
        "wiki": "wiki/entities/paper-fierce.md",
        "label": "FIERCE",
        "open": "已开源",
    },
    "2609.17824": {
        "wiki": "wiki/entities/paper-decentralized-multi-humanoid-pickup.md",
        "label": "decMHT",
        "open": "待发布",
    },
}

PAPERS: list[dict] = [
    {
        "slug": "elastiqp",
        "short": "ElastiQP",
        "title": "ElastiQP: An Always-Feasible QP Solver for Constrained Robot Control",
        "arxiv": "2609.19080",
        "code": "https://github.com/StanfordASL/elastiqp",
        "project": None,
        "open": "已开源",
        "tags": ["paper", "control", "qp", "constraints", "wbc", "stanford"],
        "one_liner": "不等式约束带精确 L1 软化并折进凝聚 QP，等式动力学保持硬约束；不可行时违约集中到冲突不等式，微秒级控制循环仍返回控制量。",
        "why": "真实系统常同时遇到碰撞、安全、动力学与任务约束；传统 QP 不可行即停摆，外层启发式降级难解释。",
        "mechanism": "每个不等式约束精确 L1 惩罚；等式硬约束；消元把松弛变量折进凝聚系统；违约位置与幅度可解释。",
        "metrics": "机器人控制基准微秒级求解；不可行时违约限制在冲突不等式；速度最高比最佳替代快 40×（作者报告）。",
        "conclusion": "ElastiQP 把「始终可返回控制量」作为 QP 求解器第一设计目标，适合现有 QP 控制器替换求解器做对照。",
        "related": [
            "../concepts/whole-body-control.md",
            "../methods/trajectory-optimization.md",
            "../tasks/locomotion.md",
            "./paper-wave-go.md",
        ],
        "abbrev": [
            ("QP", "Quadratic Programming", "二次规划"),
            ("WBC", "Whole-Body Control", "全身控制"),
            ("L1", "L1 Penalty", "不等式软约束惩罚"),
            ("ASL", "Autonomous Systems Lab", "Stanford 自主系统实验室"),
        ],
        "seq": True,
    },
    {
        "slug": "dreaming-sound-of-contact",
        "short": "Dreaming the Sound of Contact",
        "title": "Dreaming the Sound of Contact: Leveraging Video and Audio Generation for Zero-Shot Force-Aware Manipulation and Data Generation",
        "arxiv": "2609.19137",
        "code": None,
        "project": "https://dreamingcontactsound.github.io/",
        "open": "待发布",
        "tags": [
            "paper",
            "manipulation",
            "force-control",
            "audio",
            "video-generation",
            "zero-shot",
        ],
        "one_liner": "Seedance 2.0 联合生成视频与音频；视频得运动轨迹，音频响度构造期望力曲线，1 kHz 阻抗+力调节闭环执行；四类任务 40 次试验力感知 90% vs 运动学 20%。",
        "why": "擦拭、剥离、按压等任务正确轨迹不等于正确接触力；视频生成只显示「去哪」不显示「多用力」。",
        "mechanism": "MolmoPoint+SAM2+TAPIP3D 从视频得 EE 路径；SAM-Audio 分离接触声并映射响度→力曲线；Franka 力调节闭环。",
        "metrics": "四类任务共 40 次：力感知流程 36/40 vs 运动学 8/40；Diffusion Policy 数据引擎 34/40（含力输入）vs 29/40。",
        "conclusion": "接触音频可作为零样本力监督信号；音频形力曲线优于常数力阶跃，避免 overshoot 触发安全停机。",
        "related": [
            "../tasks/manipulation.md",
            "../methods/imitation-learning.md",
            "../concepts/contact-rich-manipulation.md",
            "./paper-fetch-my-beer.md",
        ],
        "abbrev": [
            ("DP", "Diffusion Policy", "扩散策略模仿学习"),
            ("EE", "End-Effector", "末端执行器"),
            ("SAM", "Segment Anything Model", "分割与音频分离模块族"),
            ("Zero-Shot", "Zero-Shot", "无任务特定训练直接执行"),
        ],
        "seq": False,
    },
    {
        "slug": "robovad",
        "short": "RoboVAD",
        "title": "RoboVAD: A Large Cross-Domain Evaluation Benchmark for Anomaly Detection in Robotic Arm Manipulation Videos",
        "arxiv": "2609.17843",
        "code": None,
        "project": "https://zenodo.org/records/22754659",
        "open": "部分开源",
        "tags": ["paper", "benchmark", "anomaly-detection", "manipulation", "video"],
        "one_liner": "1,078 episode、5 类任务、5 类异常、2 视角；跨域未见任务为测试重点；最难设置下所有方法帧级 micro-AUC 均低于 70%。",
        "why": "机械臂异常检测常被单一任务/视角限制；跨域泛化缺口需要统一基准暴露。",
        "mechanism": "大规模跨域机械臂操作视频；正常/异常 episode 标注；强调未见任务域迁移评测。",
        "metrics": "1,078 episodes；5 tasks × 5 anomaly types × 2 cameras；最难 split 全部方法 micro-AUC < 70%。",
        "conclusion": "RoboVAD 表明机械臂视频异常检测在跨域设置下仍远未解决；适合作为方法选型与泛化下限对照。",
        "related": [
            "../tasks/manipulation.md",
            "../methods/imitation-learning.md",
            "./paper-pointzero.md",
            "../concepts/safety-filter.md",
        ],
        "abbrev": [
            ("VAD", "Video Anomaly Detection", "视频异常检测"),
            ("AUC", "Area Under Curve", "ROC 曲线下面积"),
            ("OOD", "Out-of-Distribution", "分布外/未见任务域"),
            ("AD", "Anomaly Detection", "异常检测"),
        ],
        "seq": False,
    },
    {
        "slug": "pointzero",
        "short": "PointZero",
        "title": "PointZero: 3D Point Track Completion for Learning Transferable 3D Dynamics",
        "arxiv": "2609.19142",
        "code": "https://github.com/Duisterhof/pointzero",
        "project": "https://pointzero-wm.github.io/",
        "open": "已开源",
        "tags": ["paper", "world-model", "3d-dynamics", "point-tracks", "sim2real", "manipulation"],
        "one_liner": "RGB-D + 稀疏 3D 点轨迹预测未来轨迹；290 万合成帧覆盖刚体/关节/可变形；微调到动作预测与 IL 后 7 任务中 6 个达或超基线，无需机器人动作标签预训练。",
        "why": "机器人动作标签难规模化；网络视频含丰富物体运动，可用点轨迹补全学可迁移 3D 动力学。",
        "mechanism": "从 RGB-D 与稀疏点轨迹预测未来 3D point tracks；大规模合成预训练 → 下游动作预测/模仿微调。",
        "metrics": "2.9M 合成帧；7 个仿真+真机操作任务中 6 个 ≥ baseline（作者报告）。",
        "conclusion": "PointZero 用点轨迹补全把 web 视频动力学先验迁到机器人，适合无动作标签预训练路线。",
        "related": [
            "../methods/generative-world-models.md",
            "../concepts/world-action-models.md",
            "../tasks/manipulation.md",
            "./paper-robovad.md",
        ],
        "abbrev": [
            ("WM", "World Model", "环境/物体动力学预测"),
            ("RGB-D", "RGB-Depth", "彩色深度观测"),
            ("IL", "Imitation Learning", "模仿学习"),
            ("Sim2Real", "Simulation to Real", "仿真到真机迁移"),
        ],
        "seq": True,
    },
    {
        "slug": "fetch-my-beer",
        "short": "Fetch My Beer",
        "title": "Fetch My Beer: Synthetic-to-real Hierarchical Policy for Smooth Pick-and-place",
        "arxiv": "2609.18119",
        "code": None,
        "project": "https://fetch-my-beer.github.io/",
        "open": "待发布",
        "tags": [
            "paper",
            "manipulation",
            "liquid-transport",
            "hierarchical-policy",
            "diffusion",
            "sim2real",
        ],
        "one_liner": "流体仿真筛选稳定轨迹 + VLM 过滤不稳定姿态；高层语言视觉给 SE(3) 目标，潜扩散控制器生成平滑动作块；仅合成示范零样本 sim-to-real 液体运输。",
        "why": "装满液体的容器即使抓取成功，急停/转向仍可能洒出；需轨迹级动态稳定而非仅到达目标。",
        "mechanism": "合成抓取+流体仿真验证 → 分层：高层 SE(3) 目标 + 潜空间扩散密集动作块；强调 motion smoothness。",
        "metrics": "液体 pick-and-place；相对 SOTA 操作策略在运输平滑度与动态稳定性上更优（项目页/RA-L 2026）。",
        "conclusion": "Fetch My Beer 把「不洒」作为 pick-and-place 的一等目标；代码/数据截至入库日标注 Coming soon。",
        "related": [
            "../tasks/manipulation.md",
            "../concepts/sim2real.md",
            "../methods/imitation-learning.md",
            "./paper-dreaming-sound-of-contact.md",
        ],
        "abbrev": [
            ("SE(3)", "Special Euclidean Group", "刚体位姿空间"),
            ("VLM", "Vision-Language Model", "视觉–语言模型筛选姿态"),
            ("RA-L", "IEEE Robotics and Automation Letters", "发表期刊"),
            ("Sim2Real", "Simulation to Real", "合成到真机迁移"),
        ],
        "seq": False,
    },
    {
        "slug": "opendexgrasp",
        "short": "OpenDexGrasp",
        "title": "OpenDexGrasp: Open-vocabulary Task-Oriented Dexterous Grasping",
        "arxiv": "2609.18117",
        "code": None,
        "project": "https://opendexgrasp.github.io/",
        "open": "待发布",
        "tags": ["paper", "dexterous-manipulation", "grasping", "vlm", "flow-matching", "pku"],
        "one_liner": "自然语言功能意图 + 多视角 RGB + 点云几何 → 高 DoF 任务导向抓取；C2A 数据：1.24M 自动合成 + 27.42k 遥操作对齐；真机平均成功率 72.0%。",
        "why": "同一物体「拿稳」与「拿得能用」不同；开放词汇任务导向灵巧抓取需语义–几何–动作统一表示。",
        "mechanism": "VLM 语义 token + 点云几何 → flow-matching action expert 直接生成功能抓取；affordance 作辅助监督非级联瓶颈。",
        "metrics": "Seen functional SR 68.07%；Unseen 62.96%；真机平均 72.0% vs DexGraspNet 2.0* 59.0%。",
        "conclusion": "OpenDexGrasp 把 affordance 与抓取生成视为同一任务条件分布的两视图；部署前等官方代码。",
        "related": [
            "../tasks/manipulation.md",
            "../methods/vla.md",
            "../concepts/contact-rich-manipulation.md",
            "./paper-fierce.md",
        ],
        "abbrev": [
            ("DoF", "Degrees of Freedom", "灵巧手自由度"),
            ("VLM", "Vision-Language Model", "开放词汇语义编码"),
            ("C2A", "Coverage-to-Alignment", "先扩覆盖再遥操作对齐的数据配方"),
            ("SR", "Success Rate", "抓取/任务成功率"),
        ],
        "seq": False,
    },
]


def _yaml_list(items: list[str], indent: int = 2) -> str:
    pad = " " * indent
    return "\n".join(f"{pad}- {x}" for x in items)


def _abbrev(rows: list[tuple[str, str, str]]) -> str:
    lines = ["| 缩写 | 英文全称 | 简要说明 |", "|------|----------|----------|"]
    for a, b, c in rows:
        lines.append(f"| {a} | {b} | {c} |")
    return "\n".join(lines)


def _paper_source(p: dict) -> str:
    ax = p["arxiv"]
    extra = ""
    if p.get("code"):
        extra += f"- **代码：** {p['code']}\n"
    if p.get("project"):
        extra += f"- **项目页：** {p['project']}\n"
    return f"""# {p["short"]}（arXiv:{ax}）

> 来源归档（paper）

- **标题：** {p["title"]}
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/{ax}>
- **PDF：** <https://arxiv.org/pdf/{ax}>
{extra}- **入库日期：** {TODAY}
- **一句话说明：** {p["one_liner"]}

## 开源状态

- **{p["open"]}**（步骤 2.5 核查，{TODAY}）

## 核心摘录

1. **策展来源：** [具身智能小站 11 篇盘点](../../sources/blogs/{BLOG})
2. **机制：** {p["mechanism"]}

**对 wiki 的映射**

- [paper-{p["slug"]}](../../wiki/entities/paper-{p["slug"]}.md)
"""


def _seq_block(p: dict) -> str:
    if not p.get("seq"):
        return """## 源码运行时序图

**不适用**（截至 {today} 未发布可运行官方代码或待核实）。
""".format(today=TODAY)
    if p["slug"] == "elastiqp":
        return f"""## 源码运行时序图

节点对齐 [`sources/repos/elastiqp.md`](../../sources/repos/elastiqp.md) 与 [StanfordASL/elastiqp]({p["code"]})。

```mermaid
sequenceDiagram
    autonumber
    participant Ctrl as 上层 QP 控制器<br/>WBC / MPC
    participant EQ as ElastiQP 求解器<br/>C++ / Python / JAX
    participant Dyn as 动力学等式约束
    participant Ineq as 不等式约束族<br/>碰撞/安全/任务
    Ctrl->>EQ: 构建 QP（硬等式 + 软不等式）
    EQ->>Dyn: 保持等式硬约束
    EQ->>Ineq: L1 软化并入凝聚系统
    alt 可行
        EQ-->>Ctrl: 最优控制量
    else 不可行
        EQ-->>Ctrl: 可行近似解 + 违约定位
    end
    Ctrl->>Ctrl: 微秒级控制循环
```

- **最短路径：** 克隆仓库 → 用 C++ 头文件或 Python/JAX 绑定替换现有 QP 后端 → 对照不可行场景违约分布。
"""
    if p["slug"] == "pointzero":
        return f"""## 源码运行时序图

节点对齐 [`sources/repos/pointzero.md`](../../sources/repos/pointzero.md) 与 [Duisterhof/pointzero]({p["code"]})。

```mermaid
sequenceDiagram
    autonumber
    participant Data as RGB-D + 稀疏点轨迹
    participant PT as PointZero 模型
    participant WM as 未来点轨迹预测
    participant Down as 下游动作预测 / IL
    Data->>PT: 编码当前观测
    PT->>WM: 补全未来 3D point tracks
    WM-->>Down: 可迁移动力学表征
    Down->>Down: 微调至操作/模仿任务
```

- **最短路径：** 克隆仓库 → 按 README 预训练/微调 → 在 7 任务协议上对照 baseline。
"""
    return ""


def _entity(p: dict) -> str:
    ax = p["arxiv"]
    slug = p["slug"]
    src_paper = f"../../sources/papers/{slug}_arxiv_{ax.replace('.', '_')}.md"
    src_blog = f"../../sources/blogs/{BLOG}"
    code_line = f"code: {p['code']}\n" if p.get("code") else ""
    repo_src = f"  - ../../sources/repos/{slug.replace('-', '_')}.md\n" if p.get("code") else ""
    site_src = f"  - ../../sources/sites/{slug}.md\n" if p.get("project") else ""
    proj_line = f"- [项目页]({p['project']})\n" if p.get("project") else ""
    code_read = f"- [官方代码]({p['code']})\n" if p.get("code") else ""
    related = p["related"] + ["../overview/constraint-control-11-papers-technology-map.md"]
    link_bits = ""
    if p.get("project"):
        link_bits += f"，[项目页]({p['project']})"
    if p.get("code"):
        link_bits += f"，[代码]({p['code']})"
    return f"""---
type: entity
tags:
{_yaml_list(p["tags"], 2)}
status: complete
updated: {TODAY}
arxiv: "{ax}"
{code_line}related:
{_yaml_list(related, 2)}
sources:
  - {src_paper}
{repo_src}{site_src}  - {src_blog}
summary: "{p["short"]}（arXiv:{ax}）：{p["one_liner"][:120]}"
---

# {p["short"]}（arXiv:{ax}）

**{p["short"]}**（*{p["title"]}*，[arXiv:{ax}](https://arxiv.org/abs/{ax}){link_bits}）来自 [具身智能小站 11 篇盘点](../../sources/blogs/{BLOG})（2026-09-20）。

## 一句话定义

**{p["one_liner"]}**

## 英文缩写速查

{_abbrev(p["abbrev"])}

## 为什么重要

- {p["why"]}
- 开源结论：**{p["open"]}**（步骤 2.5，{TODAY}）。
- 与 [11 篇技术地图](../overview/constraint-control-11-papers-technology-map.md) 中同类工作可横向对照。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [{ax}](https://arxiv.org/abs/{ax}) |
| **开源** | **{p["open"]}** |
| **要点** | {p["mechanism"]} |
| **文内指标** | {p["metrics"]} |

{_seq_block(p)}

## 实验与评测

- {p["metrics"]}
- **读法：** 索引级摘要；逐项对照与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [11 篇技术地图](../overview/constraint-control-11-papers-technology-map.md)；与同 arXiv 节点不重复造页。

## 结论

**{p["conclusion"]}**

1. 开源边界：**{p["open"]}** — 以项目页实际链接为准（入库日 {TODAY}）。
2. 核心机制：{p["mechanism"][:80]}…
3. 部署前核对任务协议与硬件条件，勿直接横比公众号摘录数字。

## 关联页面

{_yaml_list([f"[{r.split('/')[-1].replace('.md', '')}]({r})" for r in p["related"]], 0)}

## 参考来源

- [{slug}_arxiv_{ax.replace(".", "_")}.md](../../sources/papers/{slug}_arxiv_{ax.replace(".", "_")}.md)
- [{BLOG}](../../sources/blogs/{BLOG})
- [arXiv:{ax}](https://arxiv.org/abs/{ax})

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/{ax})
{proj_line}{code_read}
"""


def _repo(p: dict) -> str:
    return f"""# {p["short"]} 官方仓库

> 来源归档（repo）

- **标题：** {p["title"]}
- **类型：** repo
- **链接：** {p["code"]}
- **arXiv：** <https://arxiv.org/abs/{p["arxiv"]}>
- **入库日期：** {TODAY}
- **一句话说明：** {p["one_liner"]}
- **沉淀到 wiki：** [`wiki/entities/paper-{p["slug"]}.md`](../../wiki/entities/paper-{p["slug"]}.md)

## 开源状态

- **{p["open"]}**：公开仓库（以 README 与 release 为准）。
"""


def _site(p: dict) -> str:
    return f"""# {p["short"]} 项目页

> 来源归档（site）

- **标题：** {p["title"]}
- **类型：** site
- **链接：** {p["project"]}
- **arXiv：** <https://arxiv.org/abs/{p["arxiv"]}>
- **入库日期：** {TODAY}
- **一句话说明：** {p["one_liner"]}
- **沉淀到 wiki：** [`wiki/entities/paper-{p["slug"]}.md`](../../wiki/entities/paper-{p["slug"]}.md)

## 开源状态

- **{p["open"]}**（步骤 2.5 核查，{TODAY}）。
"""


def _blog() -> str:
    rows = []
    order = [
        ("01", "ElastiQP", "2609.19080", "paper-elastiqp", "新建"),
        ("02", "WAVE-Go", "2609.18193", "paper-wave-go", "复用"),
        ("03", "GPT-Policy", "2609.19138", "paper-gpt-policy", "复用"),
        (
            "04",
            "Dreaming the Sound of Contact",
            "2609.19137",
            "paper-dreaming-sound-of-contact",
            "新建",
        ),
        ("05", "WholeBodyWAM", "2609.18197", "paper-wholebodywam-unimotion-4k", "复用"),
        ("06", "FIERCE", "2609.18651", "paper-fierce", "复用"),
        ("07", "RoboVAD", "2609.17843", "paper-robovad", "新建"),
        ("08", "PointZero", "2609.19142", "paper-pointzero", "新建"),
        ("09", "Fetch My Beer", "2609.18119", "paper-fetch-my-beer", "新建"),
        ("10", "OpenDexGrasp", "2609.18117", "paper-opendexgrasp", "新建"),
        ("11", "decMHT", "2609.17824", "paper-decentralized-multi-humanoid-pickup", "复用"),
    ]
    open_map = {
        "2609.19080": "已开源",
        "2609.18193": "已开源",
        "2609.19138": "已开源",
        "2609.19137": "待发布",
        "2609.18197": "待发布",
        "2609.18651": "已开源",
        "2609.17843": "部分开源",
        "2609.19142": "已开源",
        "2609.18119": "待发布",
        "2609.18117": "待发布",
        "2609.17824": "待发布",
    }
    for num, label, ax, wiki_slug, kind in order:
        rows.append(
            f"| {num} | {label} | [{ax}](https://arxiv.org/abs/{ax}) | **{open_map[ax]}** | "
            f"[{wiki_slug}](../../wiki/entities/{wiki_slug}.md)（**{kind}**） |"
        )
    table = "\n".join(rows)
    return f"""# 机器人控制遇到约束冲突，怎样保证动作还能继续？｜附代码与评测入口

> 来源归档（blog / 微信公众号）

- **标题：** 机器人控制遇到约束冲突，怎样保证动作还能继续？｜附代码与评测入口
- **类型：** blog
- **作者：** 具身智能小站（微信公众号）
- **原始链接：** {WX_URL}
- **发表日期：** 2026-09-20
- **入库日期：** {TODAY}
- **抓取方式：** WebFetch（Camoufox 工具链不可用时的兜底）
- **原始抓取落盘：** [`sources/raw/wechat_embodied_station_11_papers_constraint_control_2026-09-20.md`](../raw/wechat_embodied_station_11_papers_constraint_control_2026-09-20.md)
- **一句话说明：** 11 篇具身论文盘点，覆盖约束 QP、世界模型导航、VLM in-context、力感知操作、人形 WAM、异常检测基准、3D 动力学、液体运输、任务导向灵巧抓取与多人形协作；**11/11 独立详情节点**（本 ingest **新建 6**、**复用 5**）。

## 11 篇 → 本库节点

| # | 论文 | arXiv | 开源结论 | wiki |
|---|------|-------|----------|------|
{table}

## 对 wiki 的映射

- **11/11 独立详情节点**；**0 重复 arXiv 节点**
- 阅读坐标：[约束控制 11 篇技术地图](../../wiki/overview/constraint-control-11-papers-technology-map.md)
- 交叉：[Whole-Body Control](../../wiki/concepts/whole-body-control.md)、[World Action Models](../../wiki/concepts/world-action-models.md)、[VLA](../../wiki/methods/vla.md)、[Loco-Manipulation](../../wiki/tasks/loco-manipulation.md)

## 当前提炼状态

- [x] 公众号正文抓取
- [x] 11 篇独立节点（6 新建 / 5 复用）
- [x] 项目页/仓库开源状态核查（步骤 2.5）
"""


def _map() -> str:
    return f"""---
type: overview
tags: [overview, survey, control, navigation, vla, manipulation, humanoid, technology-map]
status: complete
updated: {TODAY}
related:
  - ../entities/paper-elastiqp.md
  - ../entities/paper-wave-go.md
  - ../entities/paper-gpt-policy.md
  - ../entities/paper-wholebodywam-unimotion-4k.md
  - ../methods/vla.md
  - ../concepts/whole-body-control.md
sources:
  - ../../sources/blogs/{BLOG}
summary: "具身智能小站 2026-09-20 十一篇盘点：约束 QP 可部署、可中断 WM 导航、VLM in-context、力/液体/灵巧操作与人形协作五条阅读线。"
---

# 约束冲突与可继续执行：11 篇论文阅读坐标

> **本页定位**：为 [具身智能小站 · 11 篇盘点]({WX_URL})（2026-09-20）提供按问题组织的阅读坐标。

## 一句话观点

**「动作还能继续」取决于三层：QP/控制器在不可行时如何软化、世界模型/智能体在执行中如何反悔与验证、操作层如何把力与动态稳定写进闭环——而非单点换更大模型。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| QP | Quadratic Programming | 约束二次规划 |
| WM | World Model | 世界/动作预测模型 |
| VLM | Vision-Language Model | 视觉–语言模型智能体 |
| WAM | World Action Model | 世界–动作联合模型 |

## 为什么单独做这张地图

- 一次列出 11 篇方向跨度大的工作；需要横切面索引。
- **11/11 独立节点**：新建 6 + 复用 5；**0 重复 arXiv**。

## 流程总览

```mermaid
flowchart TB
  subgraph CTRL["控制与导航"]
    EQ[ElastiQP 始终可行 QP]
    WG[WAVE-Go 可中断 WM 导航]
  end
  subgraph AGENT["智能体与学习"]
    GP[GPT-Policy VLM in-context]
    PZ[PointZero 3D 动力学先验]
    FI[FIERCE 专才 RL]
  end
  subgraph MANIP["接触与操作"]
    DS[Dreaming Sound 音频力]
    FB[Fetch My Beer 液体稳定]
    OD[OpenDexGrasp 任务抓取]
  end
  subgraph HUM["人形与协作"]
    WB[WholeBodyWAM 全身 WAM]
    MH[decMHT 多人形搬运]
  end
  subgraph EVAL["评测"]
    RV[RoboVAD 异常检测基准]
  end
  CTRL --> GOAL[可继续执行的具身系统]
  AGENT --> GOAL
  MANIP --> GOAL
  HUM --> GOAL
  EVAL --> GOAL
```

## 分组索引

### 控制与导航（深读 + 跟进）

| 论文 | 节点 | 开源 |
|------|------|------|
| ElastiQP | [paper-elastiqp](../entities/paper-elastiqp.md) | 已开源 |
| WAVE-Go | [paper-wave-go](../entities/paper-wave-go.md) | 已开源 |
| GPT-Policy | [paper-gpt-policy](../entities/paper-gpt-policy.md) | 已开源 |

### 力感知、液体与灵巧操作

| 论文 | 节点 | 开源 |
|------|------|------|
| Dreaming the Sound of Contact | [paper-dreaming-sound-of-contact](../entities/paper-dreaming-sound-of-contact.md) | 待发布 |
| Fetch My Beer | [paper-fetch-my-beer](../entities/paper-fetch-my-beer.md) | 待发布 |
| OpenDexGrasp | [paper-opendexgrasp](../entities/paper-opendexgrasp.md) | 待发布 |

### 世界模型、专才与基准

| 论文 | 节点 | 开源 |
|------|------|------|
| WholeBodyWAM | [paper-wholebodywam-unimotion-4k](../entities/paper-wholebodywam-unimotion-4k.md) | 待发布 |
| PointZero | [paper-pointzero](../entities/paper-pointzero.md) | 已开源 |
| FIERCE | [paper-fierce](../entities/paper-fierce.md) | 已开源 |
| RoboVAD | [paper-robovad](../entities/paper-robovad.md) | 部分开源 |

### 多人形协作

| 论文 | 节点 | 开源 |
|------|------|------|
| decMHT | [paper-decentralized-multi-humanoid-pickup](../entities/paper-decentralized-multi-humanoid-pickup.md) | 待发布 |

## 关联页面

- [Whole-Body Control](../concepts/whole-body-control.md)
- [World Action Models](../concepts/world-action-models.md)
- [VLA](../methods/vla.md)
- [Loco-Manipulation](../tasks/loco-manipulation.md)

## 参考来源

- [{BLOG}](../../sources/blogs/{BLOG})

## 推荐继续阅读

- [ElastiQP](../entities/paper-elastiqp.md)
- [GPT-Policy](../entities/paper-gpt-policy.md)
"""


def _append_blog_source(wiki_path: Path) -> None:
    text = wiki_path.read_text(encoding="utf-8")
    src_line = f"  - ../../sources/blogs/{BLOG}"
    if BLOG in text:
        return
    if "sources:\n" in text:
        text = text.replace("sources:\n", f"sources:\n{src_line}\n", 1)
    else:
        text = text.replace("summary:", f"sources:\n{src_line}\nsummary:", 1)
    text = text.replace(f"updated: {TODAY}", f"updated: {TODAY}")  # noqa: SIM210
    if "constraint-control-11-papers-technology-map" not in text:
        if "related:\n" in text:
            text = text.replace(
                "related:\n",
                "related:\n  - ../overview/constraint-control-11-papers-technology-map.md\n",
                1,
            )
    wiki_path.write_text(text, encoding="utf-8")


def _patch_wave_go() -> None:
    path = ROOT / "wiki/entities/paper-wave-go.md"
    text = path.read_text(encoding="utf-8")
    text = text.replace("**待发布**", "**已开源**")
    text = text.replace("**待发布**（步骤 2.5，2026-09-20）", "**已开源**（步骤 2.5，2026-09-20）")
    if "code: https://github.com/vigorlee/wave-go" not in text:
        text = text.replace(
            'arxiv: "2609.18193"\n',
            'arxiv: "2609.18193"\ncode: https://github.com/vigorlee/wave-go\n',
        )
    if "sources/repos/wave_go.md" not in text:
        text = text.replace(
            "  - ../../sources/papers/wave-go_arxiv_2609_18193.md\n",
            "  - ../../sources/papers/wave-go_arxiv_2609_18193.md\n"
            "  - ../../sources/repos/wave_go.md\n",
        )
    _append_blog_source(path)
    path.write_text(text, encoding="utf-8")
    repo_path = ROOT / "sources/repos/wave_go.md"
    if not repo_path.exists():
        repo_path.write_text(
            f"""# WAVE-Go 官方仓库

> 来源归档（repo）

- **标题：** WAVE-Go: World-Model Navigation with Adaptive Execution for Wheel-Legged Robots
- **类型：** repo
- **链接：** https://github.com/vigorlee/wave-go
- **arXiv：** <https://arxiv.org/abs/2609.18193>
- **入库日期：** {TODAY}
- **一句话说明：** Go2-W 世界模型动作适配、无图充电验证与 Cosmos3-Edge 混合长距导航 demo。
- **沉淀到 wiki：** [`wiki/entities/paper-wave-go.md`](../../wiki/entities/paper-wave-go.md)

## 开源状态

- **已开源**：公开仓库（以 README 与 release 为准）。
""",
            encoding="utf-8",
        )


def main() -> None:
    BLOG_PATH.write_text(_blog(), encoding="utf-8")
    MAP_PATH.write_text(_map(), encoding="utf-8")
    for p in PAPERS:
        ax_file = f"{p['slug']}_arxiv_{p['arxiv'].replace('.', '_')}.md"
        (ROOT / "sources/papers" / ax_file).write_text(_paper_source(p), encoding="utf-8")
        (ROOT / "wiki/entities" / f"paper-{p['slug']}.md").write_text(_entity(p), encoding="utf-8")
        if p.get("code"):
            repo_name = p["slug"].replace("-", "_")
            (ROOT / "sources/repos" / f"{repo_name}.md").write_text(_repo(p), encoding="utf-8")
        if p.get("project"):
            (ROOT / "sources/sites" / f"{p['slug']}.md").write_text(_site(p), encoding="utf-8")
    for info in REUSE.values():
        wiki_path = ROOT / info["wiki"]
        if wiki_path.exists():
            _append_blog_source(wiki_path)
        else:
            print(f"WARN missing reuse target {wiki_path}")
    _patch_wave_go()
    print(f"Wrote blog, map, {len(PAPERS)} new entities, patched {len(REUSE)} reuse nodes")


if __name__ == "__main__":
    main()
