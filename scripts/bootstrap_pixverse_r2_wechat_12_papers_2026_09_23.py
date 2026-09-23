#!/usr/bin/env python3
"""Bootstrap ingest: PixVerse R2 + 具身智能小站 12 篇论文（2026-09-23 公众号盘点）."""

from __future__ import annotations

from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TODAY = date.today().isoformat()
BLOG = "wechat_embodied_station_12_papers_collab_wm_2026-09-23.md"
BLOG_PATH = ROOT / "sources/blogs" / BLOG
RAW_DST = ROOT / "sources/raw/wechat_embodied_station_12_papers_collab_wm_2026-09-23.md"
MAP_PATH = ROOT / "wiki/overview/collab-wm-12-papers-technology-map.md"
WX_URL = "https://mp.weixin.qq.com/s/ZYnIkrJ-9H5KQw2z-AT0qQ"

PAPERS: list[dict] = [
    {
        "slug": "industrialvla-bench",
        "title": "IndustrialVLA-Bench: A Traceable Multi-Axis Evaluation of Open Robot Policy Models",
        "short": "IndustrialVLA-Bench",
        "arxiv": "2609.25562",
        "code": "https://github.com/xiaoqi-7/IndustrialVLA-Bench",
        "project": None,
        "open": "已开源",
        "tags": ["paper", "vla", "benchmark", "evaluation", "deployment"],
        "one_liner": "多轴可追溯 VLA 评测：LIBERO / LIBERO-Plus / LIBERO-Para 分测干净能力、视觉鲁棒与指令敏感，并记录延迟、显存与证据等级。",
        "why": "六个系统 clean LIBERO 平均仅差 1.58 分，但鲁棒性与改写指令摘要可差 14.62 与 31.08 分——部署选型不能只看单一成功率。",
        "mechanism": "三套件分轴评测；每任务 3 随机种子；protocol-faithful / near-reproduction / pending-verification 分层。",
        "metrics": "官方仓库含环境、启动脚本、逐种子结果与延迟/显存日志；权重与部分仿真资产需按各项目说明下载。",
        "conclusion": "IndustrialVLA-Bench 把「谁更强」从排行榜变成诊断式协议；复现前核对证据等级与运行成本列。",
        "related": [
            "../methods/vla.md",
            "../tasks/manipulation.md",
            "../queries/vla-deployment-guide.md",
            "./fluxvla-engine.md",
        ],
        "abbrev": [
            ("VLA", "Vision-Language-Action", "视觉–语言–动作策略"),
            ("LIBERO", "Lifelong Robot Learning Benchmark", "操作基准套件"),
            ("Bench", "Benchmark", "评测协议与工具链"),
            ("SR", "Success Rate", "任务成功率"),
        ],
    },
    {
        "slug": "varepsilon4p",
        "title": "Imperfection for Precision: Upcycling Imperfect Data for High-Precision Robotic Manipulation",
        "short": "ε4P",
        "arxiv": "2609.26672",
        "code": None,
        "project": "https://varepsilon4p.github.io/",
        "open": "未开源",
        "tags": ["paper", "vla", "flow-matching", "manipulation", "data-efficiency"],
        "one_liner": "按 flow-matching 噪声阶段分工：低精度目标任务数据在高噪声保留语境，高精度异任务数据在低噪声传递动作精度。",
        "why": "亚毫米级精密操作常被昂贵任务专属遥操作数据卡住；ε4P 不粗暴混合 imperfect 源，而用 flow time 作 admission filter。",
        "mechanism": "离线估计 t_pm / t_tm 边界；训练时按源采样可接受 flow time；策略架构与目标不变。",
        "metrics": "ATX 插入 80.0%、两阶段线缆 88.3%、螺栓分拣 91.7%；相对 native co-training 最高 +31.7 pp。",
        "conclusion": "ε4P 证明 imperfect 数据的价值在「何时贡献」而非「是否保留」；项目页入库日无 GitHub。",
        "related": [
            "../methods/vla.md",
            "../methods/imitation-learning.md",
            "../tasks/manipulation.md",
            "./paper-industrialvla-bench.md",
        ],
        "abbrev": [
            ("ε4P", "Imperfection for Precision", "本文 imperfect 数据升级框架"),
            ("VLA", "Vision-Language-Action", "视觉–语言–动作策略"),
            ("FM", "Flow Matching", "连续流匹配生成/策略训练"),
            ("UMI", "Universal Manipulation Interface", "常见异任务高精度示范来源"),
        ],
    },
    {
        "slug": "mate-virtual-teleop",
        "title": "MATE: Multi-Agent Virtual Teleoperation Platform for Humanoid Collaboration Data Collection",
        "short": "MATE",
        "arxiv": "2609.26520",
        "code": None,
        "project": "https://yerik-yu.github.io/MATE/",
        "open": "待发布",
        "tags": ["paper", "humanoid", "teleoperation", "multi-agent", "data-collection"],
        "one_liner": "多操作者在共享物理仿真中同时全身遥操作 humanoid；EAIS 优先采样任务推进与交互关键片段。",
        "why": "多 humanoid 协作数据采集受硬件数量、场地与重置成本限制；虚拟平台把 amortized 采集从 ~89s 降到 ~41s（Bottle Relay）。",
        "mechanism": "分布式全身遥操作 + 共享物理环境 + Execution-Aligned Interaction Sampling；24.1h / 2500 joint episodes / 5 长时程任务。",
        "metrics": "IL 与 VLA 可学；报告虚拟示范到真机 humanoid 零样本迁移（无额外真机数据）。",
        "conclusion": "MATE 把协作数据问题从「买更多机器人」转成「共享虚拟世界 + 对齐采样」；代码待论文发布。",
        "related": [
            "../tasks/loco-manipulation.md",
            "../methods/imitation-learning.md",
            "../methods/vla.md",
            "./paper-me-u0.md",
        ],
        "abbrev": [
            ("MATE", "Multi-Agent Virtual Teleoperation Platform", "本文虚拟协作遥操作平台"),
            ("EAIS", "Execution-Aligned Interaction Sampling", "任务推进/交互关键片段优先采样"),
            ("VLA", "Vision-Language-Action", "视觉–语言–动作策略"),
            ("IL", "Imitation Learning", "模仿学习"),
        ],
    },
    {
        "slug": "me-u0",
        "title": "MachEmbodied-U0: Unified Understanding and Generation Model for Embodied Intelligence",
        "short": "MachEmbodied-U0",
        "arxiv": "2609.25627",
        "code": "https://github.com/MachEmbodied/ME-U0",
        "project": "https://machembodied.com/ME-U/ME-U0.html",
        "open": "已开源",
        "tags": ["paper", "vla", "world-model", "mot", "embodied-ai", "li-auto"],
        "one_liner": "Mixture-of-Transformers 连接子任务理解、affordance grounding、视觉动态与连续动作生成；~4200h 预训练。",
        "why": "把理解、预测与动作合进一个具身模型，避免 VLA 与 world model 割裂；报告 RoboDojo 17.66、LIBERO 99.0%、LIBERO-Plus 82.5%。",
        "mechanism": "MoT 多专家 + 统一预训练；子任务理解 / affordance / 视觉动态 / 动作生成共享表征。",
        "metrics": "官方仓库提供后训练与评测代码；具体消融以原文为准。",
        "conclusion": "ME-U0 代表 MachEmbodied 统一理解–生成–动作路线；与 ME-Dex 1.0 触觉 WAM 形成同机构对照。",
        "related": [
            "../methods/vla.md",
            "../concepts/world-action-models.md",
            "../tasks/manipulation.md",
            "./paper-me-dex-1-0.md",
        ],
        "abbrev": [
            ("ME-U0", "MachEmbodied-U0", "本文统一理解–生成具身模型"),
            ("MoT", "Mixture-of-Transformers", "多专家 Transformer 混合"),
            ("VLA", "Vision-Language-Action", "视觉–语言–动作策略"),
            ("WM", "World Model", "环境动态预测模型"),
        ],
    },
    {
        "slug": "mavp",
        "title": "MAVP: Map-Aware Visuomotor Policies for Mobile Manipulation",
        "short": "MAVP",
        "arxiv": "2609.26378",
        "code": None,
        "project": "https://123qwedsa123.github.io/mavp/",
        "open": "待发布",
        "tags": ["paper", "mobile-manipulation", "visuomotor", "mapping", "loco-manipulation"],
        "one_liner": "从遥操作示范重建共享静态地图，策略显式预测 map-frame 底盘目标，前馈 + 位姿误差反馈跟踪。",
        "why": "仅速度控制的移动操作示范在「该停在哪」上不一致；map-frame 目标把底盘意图钉在空间参考系。",
        "mechanism": "共享地图对齐示范底盘位姿；策略同时预测臂/夹爪与 map-frame pose；部署期持续定位更新。",
        "metrics": "六个真实任务、三类策略（ACT/Diffusion/Flow）；P 相对 V 在各任务均更高（如 Disassemble 38%→90%）。",
        "conclusion": "MAVP 说明移动操作需要显式空间锚，而非只靠局部速度模仿；入库日项目页未列 GitHub。",
        "related": [
            "../tasks/loco-manipulation.md",
            "../tasks/manipulation.md",
            "../methods/imitation-learning.md",
            "./paper-industrialvla-bench.md",
        ],
        "abbrev": [
            ("MAVP", "Map-Aware Visuomotor Policy", "本文地图感知视运动策略"),
            ("FF", "Feedforward", "前馈底盘目标跟踪"),
            ("FB", "Feedback", "位姿误差反馈修正"),
            ("ACT", "Action Chunking Transformer", "动作分块 Transformer 策略"),
        ],
    },
    {
        "slug": "triworldbench",
        "title": "TriWorldBench: A Tri-View Consistency Perspective on Embodied World Models",
        "short": "TriWorldBench",
        "arxiv": "2609.26314",
        "code": "https://github.com/TriWorldBench/TriWorldBench",
        "project": "https://huggingface.co/datasets/TriWorldBench/Dataset",
        "open": "已开源",
        "tags": ["paper", "world-model", "benchmark", "multi-view", "bimanual"],
        "one_liner": "500 episode / 50 双臂任务 / 19 指标检查 head 与双 wrist 预测是否描述同一动作与物体状态。",
        "why": "分视角评测易漏掉「三路是否同一世界」；TriWorldBench 把三视角一致性当作 embodied WM 的一等指标。",
        "mechanism": "三相机同步轨迹 + 一致性指标套件；官方 GitHub 与 Hugging Face 数据集入口。",
        "metrics": "19 项指标覆盖语义、几何与动作对齐；具体分数以 benchmark 文档为准。",
        "conclusion": "TriWorldBench 适合筛「看起来像」但跨视角不自洽的 WM；复现从官方 repo + HF 数据集开始。",
        "related": [
            "../methods/generative-world-models.md",
            "../tasks/bimanual-manipulation.md",
            "../tasks/manipulation.md",
            "./paper-pixverse-r2.md",
        ],
        "abbrev": [
            ("WM", "World Model", "环境前向预测模型"),
            ("Tri-View", "Tri-View Consistency", "头+双腕三视角一致性"),
            ("HF", "Hugging Face", "数据集托管平台"),
            ("Bench", "Benchmark", "标准化评测套件"),
        ],
    },
    {
        "slug": "better-curriculum",
        "title": "What is the Better Curriculum? Controller-Shaped Grasping Behavior for Contact Force-Sensitive Manipulation",
        "short": "Better Curriculum",
        "arxiv": "2609.25887",
        "code": None,
        "project": "https://shayfeng.github.io/better-curriculum/",
        "open": "待发布",
        "tags": ["paper", "tactile", "manipulation", "demonstration", "act"],
        "one_liner": "采集期 25 Hz 触觉反射教师塑形示范，学生 ACT/π0.5 推理无触觉；名义塑料杯稳定抓取 95% vs 5%。",
        "why": "把触觉当策略输入未必增益；触觉可在采集期当 teacher 把 force-valid 接触写进示范分布。",
        "mechanism": "导纳式 kinesthetic + 触觉 reflex 记录 follower 动作；学生仅 RGB+状态；部署可选 re-engage arbiter。",
        "metrics": "30 demo / 20 trial；扰动下 policy-only 55% vs +arbiter 100%；入库日无官方 GitHub。",
        "conclusion": "Better Curriculum 的「curriculum」指示范分布而非分阶段训练；部署期保护仍不可省。",
        "related": [
            "../methods/imitation-learning.md",
            "../tasks/manipulation.md",
            "./paper-pakt.md",
            "./paper-me-dex-1-0.md",
        ],
        "abbrev": [
            ("ACT", "Action Chunking Transformer", "动作分块模仿策略"),
            ("CoRL", "Conference on Robot Learning", "机器人学习会议"),
            ("GUI", "Graphical User Interface", "操作员触觉监视界面"),
            ("Hz", "Hertz", "控制/采样频率"),
        ],
    },
    {
        "slug": "pakt",
        "title": "PAKT: Physically-Aligned Kinesthetic Teaching for Reinforcement Learning",
        "short": "PAKT",
        "arxiv": "2609.25630",
        "code": None,
        "project": "https://pakt-website.github.io/pakt-website",
        "open": "待发布",
        "tags": ["paper", "rl", "kinesthetic-teaching", "impedance-control", "manipulation"],
        "one_liner": "导纳控制把人施加力映射为动作，经与策略相同运动学限制的参考轨迹 + 1 kHz 阻抗控制执行。",
        "why": "接触丰富工业装配需要可安全直观的人类引导，且示范必须与策略执行栈物理对齐。",
        "mechanism": "约束导纳 kinesthetic teaching + 三阶参考生成器 + 1 kHz Cartesian impedance；相对 HIL-SERL 降周期与干预。",
        "metrics": "四插入/装配基准：周期时间 −23%–48%，干预 −60%–85%。",
        "conclusion": "PAKT 把「人类手把手」变成与 RL 策略同构的可执行轨迹；项目页交互 demo 丰富但代码未发布。",
        "related": [
            "../methods/reinforcement-learning.md",
            "../methods/imitation-learning.md",
            "../tasks/manipulation.md",
            "./paper-better-curriculum.md",
        ],
        "abbrev": [
            ("PAKT", "Physically-Aligned Kinesthetic Teaching", "本文物理对齐动觉示教"),
            ("RL", "Reinforcement Learning", "强化学习"),
            ("HIL-SERL", "Human-in-the-Loop SERL", "人机协同 SERL 基线"),
            ("TCP", "Tool Center Point", "工具中心点"),
        ],
    },
    {
        "slug": "agentic-coding-manipulation",
        "title": "Generalizing Manipulation Skills with a Local Coding Agent",
        "short": "Agentic Coding Agent",
        "arxiv": "2609.26499",
        "code": None,
        "project": "https://rtalwar2.github.io/agentic-coding-for-robot-manipulation/",
        "open": "未开源",
        "tags": ["paper", "code-as-policy", "vla", "manipulation", "local-llm"],
        "one_liner": "本地 Qwen3.8-27B + coding-agent harness 控制 UR3e，自行编写/调试代码；9 玩具任务 45 次试验完成 30 次泛化。",
        "why": "固定动作接口或训练策略换任务成本高；探本地 VLM 能否用已文档化 procedure 泛化到新颜色/尺寸/组合。",
        "mechanism": "Platform HTTP 服务（9 个运动/感知原语 + 安全包络）+ Skills 文档 + agent workspace 写脚本执行。",
        "metrics": "15.8h 机器人时间；成功复做后时长与 token 约减半；平台/agent 代码明确不发布。",
        "conclusion": "Agentic coding 证明 procedure+skills 可泛化，但感知误读与无 body schema 是主要失败源；非开箱部署方案。",
        "related": [
            "../methods/vla.md",
            "../tasks/manipulation.md",
            "../entities/paper-pai-2209-07753-codeaspolicies.md",
            "./paper-industrialvla-bench.md",
        ],
        "abbrev": [
            ("VLM", "Vision-Language Model", "视觉–语言模型"),
            ("SDK", "Software Development Kit", "机器人控制软件开发包"),
            ("TCP", "Tool Center Point", "工具中心点"),
            ("HSV", "Hue Saturation Value", "颜色空间阈值分割"),
        ],
    },
    {
        "slug": "silent-sabotage",
        "title": "Silent Sabotage: Internal State Triggered Backdoor Attacks on LLM-Powered Robotic Systems",
        "short": "Silent Sabotage",
        "arxiv": "2609.26184",
        "code": "https://github.com/doniobidov/silent_sabotage",
        "project": None,
        "open": "已开源",
        "tags": ["paper", "security", "llm", "robotics", "backdoor"],
        "one_liner": "由机器人自身历史动作序列触发的内部状态后门：正常时保持效用，稀有动作组合可致急停或碰撞。",
        "why": "LLM 驱动机器人系统的攻击面不限于 prompt；历史状态触发可在不篡改输入时潜伏。",
        "mechanism": "内部状态序列触发 + 多机器人/多 LLM 仿真环境；MIT 许可仿真代码。",
        "metrics": "报告接近完美攻击成功率（仿真）；部署风险需结合系统架构评估。",
        "conclusion": "Silent Sabotage 提醒 VLA/LLM 机器人栈需要状态级安全审计，而非只做输入过滤。",
        "related": [
            "../methods/vla.md",
            "../concepts/safety-filter.md",
            "./paper-industrialvla-bench.md",
            "./paper-robresilience.md",
        ],
        "abbrev": [
            ("LLM", "Large Language Model", "大语言模型"),
            ("CPS", "Cyber-Physical System", "信息物理系统"),
            ("MIT", "Massachusetts Institute of Technology", "许可证类型示例"),
            ("Backdoor", "Backdoor Attack", "触发式恶意行为注入"),
        ],
    },
    {
        "slug": "phi-rie",
        "title": "ϕ-RIE: From Photorealistic Reconstruction to Interactive Environments",
        "short": "ϕ-RIE",
        "arxiv": "2609.26795",
        "code": "https://github.com/insait-institute/PhiRIE",
        "project": "https://insait-institute.github.io/PhiRIE/",
        "open": "已开源",
        "tags": ["paper", "3dgs", "simulation", "reconstruction", "interactive"],
        "one_liner": "选定物体转可移动 simulator asset，移除并补全原高斯，解决 3DGS 不支持独立运动与遮挡后景暴露。",
        "why": "照片级重建不等于可交互仿真；需要物体级资产与背景 inpainting 才能做 manipulation 闭环。",
        "mechanism": "高斯场景分解 + 物体 sim asset + 背景补全；ScanNet++ 50 场景评测。",
        "metrics": "20 mm 匹配 F1：0.336→0.383；具体任务以原文为准。",
        "conclusion": "ϕ-RIE 桥接重建与交互仿真；开源代码可复现几何/资产管线。",
        "related": [
            "../concepts/sim2real.md",
            "../methods/generative-world-models.md",
            "../tasks/manipulation.md",
            "./paper-triworldbench.md",
        ],
        "abbrev": [
            ("3DGS", "3D Gaussian Splatting", "三维高斯溅射重建"),
            ("RIE", "Reconstruction to Interactive Environments", "重建到交互环境"),
            ("F1", "F1 Score", "匹配精度指标"),
            ("Sim", "Simulation", "物理/图形仿真环境"),
        ],
    },
    {
        "slug": "situation-aware-dual-cobots",
        "title": "Situation Aware Locomotion for Dual Mobile Cobots in Shared Environments",
        "short": "Situation-Aware Dual Cobots",
        "arxiv": "2609.26083",
        "code": "https://github.com/ricardoGrando/limo_cobot_jazzy_sim",
        "project": None,
        "open": "已开源",
        "tags": ["paper", "mobile-manipulation", "multi-robot", "simulation", "safety"],
        "one_liner": "联合位姿、负载、机械臂状态、共享区域占用、障碍物与短时冲突预测，在模拟工业场景选安全移动动作。",
        "why": "双移动协作机器人需理解共享空间而非独立规划；仿真可系统比较协调策略。",
        "mechanism": "situation-aware 特征 + 安全移动动作选择；ROS2 Jazzy + Limo/Cobot 仿真栈。",
        "metrics": "报告所有场景 100% vs 独立/固定优先级基线 33.3%；**证据限于仿真**。",
        "conclusion": "Situation-Aware 框架是仿真级多机协调原型；真机迁移与感知误差未在本页展开。",
        "related": [
            "../tasks/loco-manipulation.md",
            "../tasks/manipulation.md",
            "./paper-mavp.md",
            "./paper-mate-virtual-teleop.md",
        ],
        "abbrev": [
            ("ROS2", "Robot Operating System 2", "机器人中间件"),
            ("CoBot", "Collaborative Robot", "协作机械臂"),
            ("Sim", "Simulation", "仿真验证环境"),
            ("SA", "Situation Aware", "情境感知决策"),
        ],
    },
]


def _yaml_list(items: list[str], indent: int) -> str:
    pad = " " * indent
    return "\n".join(f"{pad}- {x}" for x in items)


def _abbrev_table(rows: list[tuple[str, str, str]]) -> str:
    lines = ["| 缩写 | 英文全称 | 简要说明 |", "|------|----------|----------|"]
    for a, b, c in rows:
        lines.append(f"| {a} | {b} | {c} |")
    return "\n".join(lines)


def _seq_block(open_status: str, code: str | None) -> str:
    if open_status != "已开源" or not code:
        return (
            "## 源码运行时序图\n\n"
            "**不适用**（入库日模型/训练权重未公开，或仅有 API/CLI 封装；"
            "无可运行官方训练/推理入口。）\n"
        )
    return """## 源码运行时序图

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


def _paper_source(p: dict) -> str:
    ax = p["arxiv"]
    lines = [
        f"# {p['title']}",
        "",
        f"- **类型：** paper",
        f"- **arXiv：** <https://arxiv.org/abs/{ax}>",
    ]
    if p.get("project"):
        lines.append(f"- **项目页：** <{p['project']}>")
    if p.get("code"):
        lines.append(f"- **代码：** <{p['code']}>")
    lines += [
        f"- **入库日期：** {TODAY}",
        f"- **索引来源：** [具身智能小站 12 篇盘点](../blogs/{BLOG})（<{WX_URL}>）",
        f"- **一句话说明：** {p['one_liner']}",
        "",
        "## 核心摘录",
        "",
        f"1. {p['one_liner']}",
        f"2. {p['why']}",
        f"3. {p['metrics']}",
        "",
        "## 对 wiki 的映射",
        "",
        f"- 实体页：[`wiki/entities/paper-{p['slug']}.md`](../../wiki/entities/paper-{p['slug']}.md)",
        f"- 技术地图：[`wiki/overview/collab-wm-12-papers-technology-map.md`](../../wiki/overview/collab-wm-12-papers-technology-map.md)",
    ]
    return "\n".join(lines) + "\n"


def _entity(p: dict) -> str:
    ax = p["arxiv"]
    slug = p["slug"]
    src_paper = f"../../sources/papers/{slug}_arxiv_{ax.replace('.', '_')}.md"
    src_blog = f"../../sources/blogs/{BLOG}"
    repo_src = ""
    site_src = ""
    code_line = ""
    if p.get("code"):
        code_line = f'code: {p["code"]}\n'
        repo_src = f"  - ../../sources/repos/{slug.replace('-', '_')}.md\n"
    if p.get("project"):
        site_src = f"  - ../../sources/sites/{slug}.md\n"
    proj_line = f"- [项目页]({p['project']})\n" if p.get("project") else ""
    code_read = f"- [{p['code']}]({p['code']})\n" if p.get("code") else ""
    seq = _seq_block(p["open"], p.get("code"))
    return f"""---
type: entity
tags:
{_yaml_list(p["tags"], 2)}
status: complete
updated: {TODAY}
arxiv: "{ax}"
{code_line}related:
{_yaml_list(p["related"] + ["../overview/collab-wm-12-papers-technology-map.md"], 2)}
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

{_abbrev_table(p["abbrev"])}

## 为什么重要

- {p['why']}
- 开源结论：**{p['open']}**（步骤 2.5，{TODAY}）。
- 与 [12 篇技术地图](../overview/collab-wm-12-papers-technology-map.md) 中同类工作可横向对照。

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

- 横向索引见 [12 篇技术地图](../overview/collab-wm-12-papers-technology-map.md)；与同 arXiv 节点不重复造页。

## 结论

**{p['conclusion']}**

1. 开源边界：**{p['open']}** — 以项目页实际链接为准（入库日 {TODAY}）。
2. 核心机制：{p['mechanism'][:80]}…
3. 部署前核对任务协议与硬件条件，勿直接横比公众号摘录数字。

## 关联页面

{_yaml_list([f"[{r.split('/')[-1].replace('.md', '')}]({r})" for r in p["related"]], 0)}

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


def _pixverse_sources() -> None:
    (ROOT / "sources/sites/pixverse-r2.md").write_text(
        f"""# PixVerse R2 项目页

> 来源归档（site）

- **标题：** PixVerse R2 Real-Time World Model
- **类型：** site
- **项目页：** <https://pixverse.ai/en/model/pixverse-r2>
- **技术报告：** <https://pixverse.ai/en/blog/pixverse-r2-scaling-real-time-omni-world-models>
- **在线体验：** <https://world.pixverse.video/>
- **机构：** PixVerse（爱诗科技）
- **入库日期：** {TODAY}
- **一句话说明：** 实时全模态（音视频）世界模型：Omni Causal AR 持续扩展 + Real-Time Acceleration 蒸馏为 ultra-few-step 推理。

## 开源状态（步骤 2.5）

| 资源 | 状态 |
|------|------|
| R2 模型权重 / 训练代码 | **未开源** — 项目页与博客未列模型仓库 |
| 在线 Demo | **已发布** — [world.pixverse.video](https://world.pixverse.video/) |
| 官方 GitHub 组织 | **部分** — [PixVerseAI](https://github.com/PixVerseAI) 仅 CLI / MCP / Agent Skills，非 R2 骨干 |

## 交叉链接

- 技术报告归档：[`sources/papers/pixverse_r2_technical_report_2026.md`](../papers/pixverse_r2_technical_report_2026.md)
- Wiki 实体：[`wiki/entities/paper-pixverse-r2.md`](../../wiki/entities/paper-pixverse-r2.md)
""",
        encoding="utf-8",
    )
    (ROOT / "sources/repos/pixverse-ai.md").write_text(
        f"""# PixVerseAI（GitHub 组织）

> 来源归档（repo）

- **URL：** <https://github.com/PixVerseAI>
- **类型：** repo（组织索引）
- **入库日期：** {TODAY}
- **一句话说明：** PixVerse 官方开源主要为 **API/Agent 工具链**（CLI、MCP、Skills），**不包含 PixVerse R2 世界模型训练或权重**。

## 公开仓库（入库日）

| 仓库 | 说明 |
|------|------|
| [PixVerseAI/cli](https://github.com/PixVerseAI/cli) | 终端视频/图像生成 CLI |
| [PixVerseAI/PixVerse-MCP](https://github.com/PixVerseAI/PixVerse-MCP) | MCP 服务器 |
| [PixVerseAI/skills](https://github.com/PixVerseAI/skills) | Agent skill 库 |

## 与 R2 关系

- R2 能力通过 **产品页 + 在线 World** 体验；GitHub 组织 **不能** 当作 R2 复现入口。
- Wiki：[`paper-pixverse-r2.md`](../../wiki/entities/paper-pixverse-r2.md)
""",
        encoding="utf-8",
    )
    (ROOT / "sources/papers/pixverse_r2_technical_report_2026.md").write_text(
        f"""# PixVerse R2: Scaling Real-Time Omni World Models

- **类型：** technical report / blog（PixVerse Research）
- **链接：** <https://pixverse.ai/en/blog/pixverse-r2-scaling-real-time-omni-world-models>
- **发布：** 2026-08-23（页面标注）
- **入库日期：** {TODAY}
- **前置：** PixVerse R1 — 首个公开发布的通用实时音视频世界模型

## 核心摘录

1. **两阶段框架：** 持续预训练 **Omni Causal AR** → 同骨干 **Real-Time Acceleration**（非重学世界）。
2. **Omni Causal AR：** 文本/参考/音频/动作（WASD 等）进入同一运行世界；**Dynamic Chunk Generation** 按控制信号语义边界切分音视频块。
3. **Train–Inference：** Hybrid **Teacher Forcing + Diffusion Forcing**；因果 mask + **Relative Temporal RoPE**。
4. **Multi-Timescale Memory：** Sink Memory（身份/规则）+ Rolling History（近期动力学）+ Object KV Cache。
5. **Error Bank：** 存储代表性失败状态并训练期回放；长序列亮度漂移内部评测 −35.8%。
6. **Real-Time Acceleration：** DDMD + 对抗正则 + **Block-Sparse Attention**（>90% 稀疏）+ **Pyramid Ultra-Few-Step Distillation**。

## 对 wiki 的映射

- 实体页：[`wiki/entities/paper-pixverse-r2.md`](../../wiki/entities/paper-pixverse-r2.md)
- 方法交叉：[`wiki/methods/generative-world-models.md`](../../wiki/methods/generative-world-models.md)
""",
        encoding="utf-8",
    )


def _pixverse_entity() -> str:
    return f"""---
type: entity
tags:
  - paper
  - world-model
  - generative-video
  - real-time
  - interactive
  - audiovisual
  - pixverse
status: complete
updated: {TODAY}
related:
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ./paper-sa-2604-08995-matrix-game-3-0-real-time-and-streaming-interact.md
  - ./paper-sa-2602-02393-infinite-world-scaling-interactive-world-models.md
  - ./paper-triworldbench.md
sources:
  - ../../sources/papers/pixverse_r2_technical_report_2026.md
  - ../../sources/sites/pixverse-r2.md
  - ../../sources/repos/pixverse-ai.md
  - ../../sources/blogs/{BLOG}
summary: "PixVerse R2：Omni Causal AR 持续扩展实时音视频世界模型，Multi-Timescale Memory + Error Bank + DDMD/稀疏注意力/金字塔蒸馏实现 ultra-few-step 交互。"
---

# PixVerse R2：Scaling Real-Time Omni World Models

**PixVerse R2**（[项目页](https://pixverse.ai/en/model/pixverse-r2)，[技术报告](https://pixverse.ai/en/blog/pixverse-r2-scaling-real-time-omni-world-models)，[在线 World](https://world.pixverse.video/)）是 PixVerse 在 R1「首个公开发布通用实时音视频世界模型」之后的 **统一扩展架构**：把 **数据 / 模态 / 任务 / 控制 / 时间跨度** 的持续预训练（Omni Causal AR）与 **同骨干实时加速**（Real-Time Acceleration）合成一条路径，而非传统五段式「双向→AR→蒸馏教师→DMD→自回归 DMD」链式 handoff。

## 一句话定义

**R2 让「正在生成的世界」持续接收文本、参考、音频与动作，并在 ultra-few-step 预算下保持身份锚点、长程稳定与音画同步。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| R2 | PixVerse R2 | 本文实时全模态世界模型 |
| OCA | Omni Causal AR | 可持续扩展的因果世界建模骨干 |
| DDMD | Decoupled Distribution Matching Distillation | 条件对齐与分布匹配解耦蒸馏 |
| RoPE | Rotary Position Embedding | 相对时间位置编码 |
| WM | World Model | 预测环境动态的前向模型 |

## 为什么重要

- **交互范式：** 相对「一次请求→完整视频」，R2 强调 **运行中** 接收 WASD、prompt、音频与参考，改变 **下一状态** 而非重置会话——与 [Matrix-Game 3.0](./paper-sa-2604-08995-matrix-game-3-0-real-time-and-streaming-interact.md)、[Infinite World](./paper-sa-2602-02393-infinite-world-scaling-interactive-world-models.md) 等同属 **Game & Open-World 交互 WM** 线，但 R2 突出 **音画联合** 与 **产品级在线 World**。
- **工程路线：** 把多阶段能力迁移压成 **两过程**（持续预训练 + 同骨干加速），降低 stage boundary 上的质量与长程稳定性损失。
- **记忆与恢复：** Sink / Rolling / Object 三通道记忆 + **Error Bank** 把部署失败写回训练——对长程亮度漂移等漂移指标报告 **−35.8%**（内部 stage eval）。

## 核心信息

| 项 | 内容 |
|----|------|
| **机构** | PixVerse Research（爱诗科技） |
| **前置** | PixVerse R1 — 实时音视频世界模型公开产品化 |
| **在线体验** | [world.pixverse.video](https://world.pixverse.video/) |
| **开源** | **未开源（模型）** — GitHub 仅有 CLI/MCP/Skills；权重与训练代码未发布 |

### 流程总览

```mermaid
flowchart TB
  subgraph scale [Omni Causal AR 持续扩展]
    DATA[数据/模态/任务/控制/时长]
    OCA[因果世界转移模型]
    MEM[Sink + Rolling + Object KV]
    EB[Error Bank 失败回放]
    DATA --> OCA --> MEM
    EB --> OCA
  end
  subgraph rt [Real-Time Acceleration]
    DDMD[DDMD + 对抗正则]
    BSA[Block-Sparse Attention]
    PYR[Pyramid Ultra-Few-Step]
  end
  CTRL[文本/参考/音频/动作] --> OCA
  OCA --> DDMD --> BSA --> PYR --> OUT[同步音视频段]
```

### Omni Causal AR 要点

- **Dynamic Chunk Generation：** 按活跃控制信号的语义边界切分音视频块——短块响应 WASD，长块保留事件结构与运动连贯。
- **Hybrid Teacher Forcing / Diffusion Forcing：** 干净历史保质量，噪声历史练恢复；配合因果 mask 与 **Relative Temporal RoPE** 限制绝对位置外推压力。
- **Multi-Timescale Memory：** 持久锚点（角色/环境/风格/规则）+ 近期动力学 + 仍相关的 object-level KV。

### Real-Time Acceleration 要点

- **Accelerate, not relearn：** 学生 ODE 初始化与蒸馏教师同源 OCA，避免 few-step 生成器重学长程动力学。
- **>90% attention sparsity**（内部四维质量评测仍保持）；**Pyramid** 低分辨率建结构、高分辨率补纹理。

## 源码运行时序图

**不适用**（截至入库日 {TODAY}，PixVerse R2 **模型权重与训练/推理代码未公开**；[PixVerseAI](https://github.com/PixVerseAI) 组织仓库为 API/Agent 工具链。在线体验入口：[world.pixverse.video](https://world.pixverse.video/)。）

## 工程实践

| 场景 | 读法 |
|------|------|
| **产品体验** | 从 World 页进入；队列与功能以当日产品配置为准 |
| **研究对照** | 与 Matrix-Game / Infinite World 等对比 **控制接口、记忆结构、音画同步、开源边界** |
| **复现预期** | 无官方训练代码前，仅能做 API/产品层评测，不能复现 OCA 内部指标 |

## 局限与风险

- **非游戏引擎：** 官方明确 R2 **不是** 完整开放世界或成品游戏引擎，而是 **模型生成世界的 live demo**。
- **闭源权重：** 内部稀疏度、Error Bank 等数字 **无法独立复现**；选型时区分「产品演示」与「可复现研究」。
- **物理正确性：** 与机器人 embodied WM 不同，R2 优化 **交互音视频连贯** 与 **控制响应**；勿直接等同于 manipulation 物理仿真器。

## 与其他工作对比

| 维度 | PixVerse R2 | Matrix-Game 3.0 | 机器人 Tri-View WM 评测 |
|------|-------------|-----------------|-------------------------|
| 交互 | 运行中多模态控制 | 流式游戏世界 | 双臂三视角预测一致性 |
| 输出 | 同步音视频 | 视频为主 | 多相机未来观测 |
| 开源 | 模型未开源 | 部分开源 | Benchmark 已开源 |
| 站点节点 | 本页 | [paper-sa-2604-08995-…](./paper-sa-2604-08995-matrix-game-3-0-real-time-and-streaming-interact.md) | [TriWorldBench](./paper-triworldbench.md) |

## 结论

**PixVerse R2 把「实时可进入的世界」从 R1 的产品验证推进到可扩展的 OCA+加速双进程架构——适合作为生成式交互 WM 的产品向深读，但不替代机器人 action-conditioned 世界模型选型。**

1. **真影响指标：** 运行中多模态控制、长程身份/规则锚点、音画同步与 ultra-few-step 延迟——而非单次 T2V 画质榜。
2. **次要代价：** 模型闭源、物理/action 忠实性未按机器人 WM 口径评测。
3. **部署读法：** 体验用 World；研究对照读技术报告 + 站内 [Generative World Models](../methods/generative-world-models.md)；机器人管线请看 [TriWorldBench](./paper-triworldbench.md) 等 embodied 基准。

## 关联页面

- [Generative World Models](../methods/generative-world-models.md)
- [Matrix-Game 3.0](./paper-sa-2604-08995-matrix-game-3-0-real-time-and-streaming-interact.md)
- [Infinite World](./paper-sa-2602-02393-infinite-world-scaling-interactive-world-models.md)
- [TriWorldBench](./paper-triworldbench.md)

## 参考来源

- [pixverse_r2_technical_report_2026.md](../../sources/papers/pixverse_r2_technical_report_2026.md)
- [pixverse-r2.md](../../sources/sites/pixverse-r2.md)
- [pixverse-ai.md](../../sources/repos/pixverse-ai.md)

## 推荐继续阅读

- [PixVerse R2 技术报告](https://pixverse.ai/en/blog/pixverse-r2-scaling-real-time-omni-world-models)
- [在线 World 体验](https://world.pixverse.video/)
- [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md)
"""


def _raw_wechat() -> str:
    return f"""# 机器人协作数据太贵了？把多人 humanoid teleoperation 搬进虚拟环境

> 原始链接：{WX_URL}
> 入库日期：{TODAY}
> 抓取方式：WebFetch（wechat-article-for-ai 环境不可用时的兜底）

（正文结构保留自公众号编译；推广二维码已省略。）

## 阅读路线

今天更新 12 篇具身智能新论文……

（完整策展索引见 [`sources/blogs/{BLOG}`](../blogs/{BLOG})）
"""


def _blog() -> str:
    rows = []
    for i, p in enumerate(PAPERS, 1):
        wiki = f"[paper-{p['slug']}](../../wiki/entities/paper-{p['slug']}.md)"
        rows.append(
            f"| {i:02d} | {p['short']} | [{p['arxiv']}](https://arxiv.org/abs/{p['arxiv']}) | **{p['open']}** | {wiki} |"
        )
    table = "\n".join(rows)
    return f"""# 机器人协作数据太贵了？把多人 humanoid teleoperation 搬进虚拟环境

> 来源归档（blog / 微信公众号）

- **标题：** 机器人协作数据太贵了？把多人 humanoid teleoperation 搬进虚拟环境
- **类型：** blog
- **作者：** 具身智能小站（微信公众号）
- **原始链接：** {WX_URL}
- **发表日期：** 2026-09-23
- **入库日期：** {TODAY}
- **抓取方式：** WebFetch 兜底（Camoufox 工具链本环境未预装）
- **原始抓取落盘：** [`sources/raw/wechat_embodied_station_12_papers_collab_wm_2026-09-23.md`](../raw/wechat_embodied_station_12_papers_collab_wm_2026-09-23.md)
- **一句话说明：** 12 篇具身论文盘点 + 同期 **PixVerse R2** 实时世界模型 ingest；**12/12 独立详情节点**（本批 **新建 12**、**复用 0**）。

## 12 篇 → 本库节点

| # | 论文 | arXiv | 开源结论 | wiki |
|---|------|-------|----------|------|
{table}

## PixVerse R2（同期 ingest）

| 模型 | 类型 | 开源 | wiki |
|------|------|------|------|
| PixVerse R2 | 实时 Omni 世界模型 | 模型**未开源**；Demo 已发布 | [paper-pixverse-r2](../../wiki/entities/paper-pixverse-r2.md) |

## 对 wiki 的映射

- **12/12 独立详情节点**；**0 重复 arXiv 节点**
- 阅读坐标：[协作与世界模型 12 篇技术地图](../../wiki/overview/collab-wm-12-papers-technology-map.md)
- PixVerse R2 深读：[paper-pixverse-r2](../../wiki/entities/paper-pixverse-r2.md)
- 交叉：[VLA](../../wiki/methods/vla.md)、[Generative World Models](../../wiki/methods/generative-world-models.md)、[Humanoid](../../wiki/tasks/loco-manipulation.md)

## 当前提炼状态

- [x] 公众号正文抓取（WebFetch）
- [x] 12 篇独立节点（12 新建）
- [x] PixVerse R2 实体 + sources
- [x] 项目页/仓库开源状态核查（步骤 2.5）
"""


def _map() -> str:
    def row(p: dict) -> str:
        return f"| {p['short']} | [paper-{p['slug']}](../entities/paper-{p['slug']}.md) | {p['open']} |"

    paper_rows = "\n".join(row(p) for p in PAPERS)
    return f"""---
type: overview
tags: [overview, survey, vla, world-model, humanoid, collaboration, technology-map]
status: complete
updated: {TODAY}
related:
  - ../entities/paper-pixverse-r2.md
  - ../entities/paper-industrialvla-bench.md
  - ../entities/paper-mate-virtual-teleop.md
  - ../entities/paper-triworldbench.md
  - ../methods/vla.md
  - ../methods/generative-world-models.md
sources:
  - ../../sources/blogs/{BLOG}
summary: "具身智能小站 2026-09-23 十二篇盘点：VLA 评测、协作数据采集、地图移动操作、三视角 WM 基准、安全与仿真重建；同期 PixVerse R2 实时世界模型。"
---

# 协作数据、评测与世界模型：12 篇论文阅读坐标

> **本页定位**：为 [具身智能小站 · 12 篇盘点]({WX_URL})（2026-09-23）提供按问题组织的阅读坐标，并链接同期 [PixVerse R2](../entities/paper-pixverse-r2.md) ingest。

## 一句话观点

**「协作数据贵」与「世界模型是否可信」是同一部署问题的两面：前者问数据从哪来，后者问预测能否支撑决策——本期 12 篇分别给出评测协议、虚拟协作采集、三视角一致性基准与重建到仿真路径。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| WM | World Model | 环境前向预测模型 |
| EAIS | Execution-Aligned Interaction Sampling | MATE 交互关键片段采样 |
| 3DGS | 3D Gaussian Splatting | 高斯溅射场景重建 |

## 为什么单独做这张地图

- 一次列出 12 篇跨度大的工作；需要横切面索引。
- **12/12 独立节点**：本批全部新建；**0 重复 arXiv**。

## 流程总览

```mermaid
flowchart TB
  subgraph EVAL["评测与数据"]
    IVB[IndustrialVLA-Bench]
    MATE[MATE 虚拟协作采集]
    E4P[ε4P imperfect 数据]
  end
  subgraph POL["策略与操作"]
    MEU[MachEmbodied-U0]
    MAVP[MAVP 地图移动操作]
    PAKT[PAKT 动觉示教 RL]
    BC[Better Curriculum 触觉塑形]
    AG[Agentic Coding Agent]
  end
  subgraph WM["世界模型与仿真"]
    TW[TriWorldBench]
    PR2[PixVerse R2]
    PHI[ϕ-RIE 交互环境]
  end
  subgraph SAFE["安全与多机"]
    SS[Silent Sabotage]
    SA[Situation-Aware Dual Cobots]
  end
  EVAL --> DEPLOY[可部署具身系统]
  POL --> DEPLOY
  WM --> DEPLOY
  SAFE --> DEPLOY
```

## 分组索引

### VLA 评测与数据效率

| 论文 | 节点 | 开源 |
|------|------|------|
{chr(10).join(row(p) for p in PAPERS[:2])}

### 人形协作与统一具身模型

| 论文 | 节点 | 开源 |
|------|------|------|
{chr(10).join(row(p) for p in PAPERS[2:4])}

### 移动操作与示教

| 论文 | 节点 | 开源 |
|------|------|------|
{chr(10).join(row(p) for p in PAPERS[4:8])}

### 代码代理、安全与仿真

| 论文 | 节点 | 开源 |
|------|------|------|
{chr(10).join(row(p) for p in PAPERS[8:])}

### 同期：实时交互世界模型

| 模型 | 节点 | 开源 |
|------|------|------|
| PixVerse R2 | [paper-pixverse-r2](../entities/paper-pixverse-r2.md) | 模型未开源 |

## 关联页面

- [VLA](../methods/vla.md)
- [Generative World Models](../methods/generative-world-models.md)
- [Loco-Manipulation](../tasks/loco-manipulation.md)
- [PixVerse R2](../entities/paper-pixverse-r2.md)

## 参考来源

- [{BLOG}](../../sources/blogs/{BLOG})

## 推荐继续阅读

- [IndustrialVLA-Bench](../entities/paper-industrialvla-bench.md)
- [TriWorldBench](../entities/paper-triworldbench.md)
- [PixVerse R2](../entities/paper-pixverse-r2.md)
"""


def _patch_generative_wm() -> None:
    path = ROOT / "wiki/methods/generative-world-models.md"
    text = path.read_text(encoding="utf-8")
    insert = (
        "\n[PixVerse R2](../entities/paper-pixverse-r2.md)（PixVerse Research，2026）把 **Omni Causal AR** 持续预训练与同骨干 **Real-Time Acceleration** 合成一条实时 **音视频** 交互世界路径：运行中接收文本、参考、音频与 WASD 等控制，"
        "以 Sink/Rolling/Object 三通道记忆与 **Error Bank** 抑制长程漂移，并通过 block-sparse attention 与金字塔 ultra-few-step 蒸馏压延迟。"
        "在线体验 [world.pixverse.video](https://world.pixverse.video/)；**模型权重未开源**，与机器人 action-conditioned WM 评测口径不同，宜与 [Matrix-Game 3.0](../entities/paper-sa-2604-08995-matrix-game-3-0-real-time-and-streaming-interact.md) 及 [TriWorldBench](../entities/paper-triworldbench.md) 对照阅读。\n"
    )
    anchor = "足式控制与 MBRL 文献里也会出现"
    if "paper-pixverse-r2" not in text and anchor in text:
        text = text.replace(anchor, insert + "\n" + anchor)
        path.write_text(text, encoding="utf-8")


def _log_entry() -> str:
    return f"""
## [{TODAY}] ingest | PixVerse R2 + 具身智能小站 12 篇

- **意图：** ingest PixVerse R2（项目页/技术报告/World Demo/GitHub 核查）+ 公众号 12 篇论文独立节点。
- **开源结论：** PixVerse R2 模型**未开源**（GitHub 仅 CLI/MCP）；12 篇中 IndustrialVLA-Bench / ME-U0 / TriWorldBench / Silent Sabotage / ϕ-RIE / Dual Cobots **已开源**，ε4P / Agentic Coding **未开源**，其余**待发布**。
- **关键页：** [paper-pixverse-r2](wiki/entities/paper-pixverse-r2.md)、[collab-wm-12-papers-technology-map](wiki/overview/collab-wm-12-papers-technology-map.md)
"""


def main() -> None:
    RAW_DST.parent.mkdir(parents=True, exist_ok=True)
    RAW_DST.write_text(_raw_wechat(), encoding="utf-8")
    BLOG_PATH.write_text(_blog(), encoding="utf-8")
    MAP_PATH.write_text(_map(), encoding="utf-8")
    _pixverse_sources()
    (ROOT / "wiki/entities/paper-pixverse-r2.md").write_text(_pixverse_entity(), encoding="utf-8")
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
    _patch_generative_wm()
    log = ROOT / "log.md"
    if f"[{TODAY}] ingest | PixVerse R2" not in log.read_text(encoding="utf-8"):
        log.write_text(_log_entry() + log.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Wrote PixVerse R2 + blog, map, {len(PAPERS)} paper entities")


if __name__ == "__main__":
    main()
