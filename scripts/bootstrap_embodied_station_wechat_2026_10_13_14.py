#!/usr/bin/env python3
"""Bootstrap ingest: 具身智能小站 2026-10-13（34 篇）与 2026-10-14（41 篇）两期公众号盘点。

- 抓取方式：`scripts/fetch_wechat_article.py`（Chromium + 微信 UA），正文落盘 `sources/raw/`。
- 75 条论文 → 75 个独立 `wiki/entities/paper-*.md` 节点：新建 66，复用 9（arXiv 已在库内）。
- 开源结论（步骤 2.5）：对每个 GitHub 仓库用 `git ls-remote` 核实可达；项目页用 curl 读取并扫描
  GitHub 链接 / "coming soon" 字样（入库日 2026-10-14）。
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TODAY = date.today().isoformat()

ARTICLES = {
    "a1": {
        "key": "41_papers_streamingwam_2026-10-14",
        "title": "机器人动作为什么总在换块时卡顿？｜附代码与项目入口",
        "url": "https://mp.weixin.qq.com/s/pxBq6lFyzH05LGiBkkR9mQ",
        "date": "2026-10-14",
        "n": 41,
        "map": "embodied-station-2026-10-14-41-papers-technology-map.md",
        "map_title": "流式动作块与世界模型记忆：具身智能小站 10-14 期 41 篇阅读坐标",
        "map_summary": "具身智能小站 2026-10-14 盘点：StreamingWAM 流式动作块、异步 VLA 边界对齐、组合泛化评测，以及世界模型记忆、导航/足式、多机器人协作等 41 篇；41/41 独立节点。",
        "view": "「换块卡顿」是动作块策略部署的共性问题：相邻动作块是否接得上、新观察到达后能否持续细化，比单次推理更快更关键。本期深读 StreamingWAM，跟进 FreeBar 与 RoboCombine。",
    },
    "a2": {
        "key": "34_papers_mulligan_2026-10-13",
        "title": "机器人越熟练，采集数据越要盯住失败｜附代码与项目入口",
        "url": "https://mp.weixin.qq.com/s/mGe8FNbW8HXnr5TUb7PHGw",
        "date": "2026-10-13",
        "n": 34,
        "map": "embodied-station-2026-10-13-34-papers-technology-map.md",
        "map_title": "失败导向采集与可组合世界动作模型：具身智能小站 10-13 期 34 篇阅读坐标",
        "map_summary": "具身智能小站 2026-10-13 盘点：Mulligan 失败导向真机采集、OpenWAM 可组合 WAM、EigenDEXplore 结构化探索，以及 RL 后训练、世界模型、灵巧操作、人形 34 篇；34/34 独立节点（新建 25 / 复用 9）。",
        "view": "真机策略越熟练，均匀采集越浪费预算：把采集起点、价值函数与人工介入作为可优化变量，是部署后持续学习的主线。本期深读 Mulligan，跟进 OpenWAM 与 EigenDEXplore。",
    },
}


def blog_name(a: str) -> str:
    return f"wechat_embodied_station_{ARTICLES[a]['key']}.md"


# 已在库内的 arXiv → 复用节点（不新建）
REUSE: dict[str, dict] = {
    "2610.07922": {
        "wiki": "paper-openwam-composable-world-action-models",
        "label": "OpenWAM",
        "open": "已开源",
        "group": "世界模型与仿真",
    },
    "2610.07681": {
        "wiki": "paper-eigendexplore",
        "label": "EigenDEXplore",
        "open": "已开源",
        "group": "灵巧操作",
    },
    "2610.05734": {
        "wiki": "paper-freespeed",
        "label": "FreeSpeed",
        "open": "项目页未列 GitHub",
        "group": "策略与后训练",
    },
    "2610.08789": {
        "wiki": "paper-qf3-filtered-q-gradients",
        "label": "QF3",
        "open": "项目页未列 GitHub",
        "group": "策略与后训练",
    },
    "2610.08726": {
        "wiki": "paper-egolap-egocentric-language-action-reasoning",
        "label": "EgoLAP",
        "open": "待发布",
        "group": "策略与后训练",
    },
    "2610.08320": {
        "wiki": "paper-humanoid-horizon-long-horizon-loco-manipulation",
        "label": "Humanoid Horizon",
        "open": "项目页未列 GitHub",
        "group": "人形与移动操作",
    },
    "2610.06850": {
        "wiki": "paper-intermimicgen",
        "label": "InterMimicGen",
        "open": "项目页未列 GitHub",
        "group": "人形与移动操作",
    },
    "2610.05324": {
        "wiki": "paper-codance",
        "label": "CoDance",
        "open": "项目页未列 GitHub",
        "group": "人形与移动操作",
    },
    "2610.07752": {
        "wiki": "paper-core-collaboration-role-experts",
        "label": "CoRE",
        "open": "待发布",
        "group": "多机器人与评测",
    },
}

# 关联页简写
M = "../methods/"
C = "../concepts/"
T = "../tasks/"
WAM = C + "world-action-models.md"
VLA = M + "vla.md"
RL = M + "reinforcement-learning.md"
MANIP = T + "manipulation.md"
LOCO = T + "locomotion.md"
HUM = T + "humanoid-locomotion.md"
LOCOMANIP = T + "loco-manipulation.md"
NAV = T + "zero-shot-object-navigation.md"
S2R = C + "sim2real.md"
FM = C + "flow-matching-embodied-policy.md"
CHUNK = M + "action-chunking.md"
GWM = M + "generative-world-models.md"
IL = M + "imitation-learning.md"
MARL = M + "marl.md"
SIMEVAL = C + "simulation-evaluation-infrastructure.md"
MPC = M + "model-predictive-control.md"
DEX = C + "dexterous-kinematics.md"
TELEOP = T + "teleoperation.md"
BFM = C + "behavior-foundation-model.md"
SAFE = C + "safety-filter.md"
TACT = C + "tactile-sensing.md"
V2S = C + "video-as-simulation.md"
BIM = T + "bimanual-manipulation.md"
MULTI = C + "humanoid-multi-robot-coordination.md"
DP = M + "diffusion-policy.md"
RETARGET = C + "motion-retargeting.md"
SAFERL = C + "safe-real-world-rl-fine-tuning.md"

# 开源结论常量
OPEN = "已开源"
PART = "部分开源"
TBD = "待发布"
NOLINK = "项目页未列 GitHub"

PAPERS: list[dict] = [
    # ---------------- 10-14 期（41 篇，新建 41） ----------------
    dict(
        a="a1",
        g="VLA 与动作生成",
        slug="streamingwam",
        short="StreamingWAM",
        arxiv="2610.15585",
        title="StreamingWAM: Streaming World Action Modeling via Chunk-Wise Staircase Flow Matching",
        code="https://github.com/renshaojie233/StreamingWAM",
        project="https://renshaojie233.github.io/StreamingWAM/",
        open=OPEN,
        open_note="仓库可达；公众号称当前实现范围主要是 LIBERO-Plus，真机与其它基准以后续更新为准。",
        tags=["paper", "wam", "flow-matching", "action-chunking", "streaming", "manipulation"],
        one="把整块生成改为滚动动作缓冲 + 阶梯式分块噪声 + 历史条件，新观察到达后持续细化未来动作，同时压低响应延迟与跨块顿挫。",
        why="世界动作模型部署时单次预测快还不够，相邻动作块接不上仍让机器人顿挫；首次暖启动与稳态更新的延迟须分开衡量。",
        mech="滚动缓冲中的动作按时间距离分配不同噪声层级（staircase），前缀延续已承诺动作，历史记忆作为条件；每来一帧新观察即对缓冲做一次流匹配细化。",
        metrics="作者报告覆盖 LIBERO、LIBERO-Plus、RoboTwin 与真机；公众号未核验表格，本页不引用具体数字。",
        concl="流式 WAM 的关键不是更快的一步生成，而是把分块噪声、前缀延续与历史条件放进同一更新流程，使延迟与连续性可同时评估。",
        related=[WAM, FM, CHUNK, MANIP],
        abbr=[
            ("WAM", "World Action Model", "世界预测与动作联合建模"),
            ("FM", "Flow Matching", "流匹配生成"),
            ("LIBERO", "Lifelong Robot Learning Benchmark", "操作基准套件"),
            ("RTC", "Real-Time Chunking", "实时动作块执行"),
        ],
    ),
    dict(
        a="a1",
        g="VLA 与动作生成",
        slug="freebar-barrier-guidance-async-vla",
        short="FreeBar",
        arxiv="2610.14997",
        title="Barrier-Constrained Guidance For Real-time Asynchronous VLA Execution",
        code="https://github.com/NamelessCrew/FreeBar",
        project=None,
        open=OPEN,
        open_note="仓库可达；复现需另备策略检查点与场景资产。",
        tags=[
            "paper",
            "vla",
            "flow-matching",
            "asynchronous-execution",
            "action-chunking",
            "guidance",
        ],
        one="推理期用障碍约束引导对齐异步 VLA 的动作块边界：已承诺前缀严格约束、近未来柔性衔接、远未来保留自由度。",
        why="异步执行时新动作块必须接上机器人正在执行的旧动作；硬拼接或全局约束都会牺牲远期调整空间。",
        mech="按时间分区施加一致性约束，用 barrier 形式在采样过程中引导流匹配策略，不改训练；仿真与真机报告改善。",
        metrics="作者在仿真与真机报告改善；具体数字以原文为准。",
        concl="分区对齐是异步动作块执行的通用配方：约束强度随时间衰减，比整块硬约束更保留策略原有能力。",
        related=[VLA, FM, CHUNK, "./paper-streamingwam.md"],
        abbr=[
            ("VLA", "Vision-Language-Action", "视觉–语言–动作策略"),
            ("CBF", "Control Barrier Function", "障碍函数约束"),
            ("FM", "Flow Matching", "流匹配生成"),
        ],
    ),
    dict(
        a="a1",
        g="多机器人协作与评测",
        slug="robocombine-combinatorial-generalization",
        short="RoboCombine",
        arxiv="2610.15425",
        title="Exploring the Limits of Combinatorial Generalization in Embodied Manipulation",
        code="https://github.com/AutoLab-SAI-SJTU/RoboCombine",
        project=None,
        open=PART,
        open_note="仓库可达，但公众号称主要提供评测运行时，需自备策略与仿真资产。",
        tags=["paper", "benchmark", "evaluation", "long-horizon", "manipulation", "sjtu"],
        one="把长程操作失败定位到子任务交接状态：分别统计有序成功率与交接成功率，区分理想触发与预测触发。",
        why="技能单独成功不意味着串联后成功；没有交接级诊断，组合泛化失败只能被笼统归为「长程难」。",
        mech="评测框架记录每个子任务起止状态；提出扩展原子起始状态的 PRTI 以覆盖交接分布。",
        metrics="评测协议为主；数值以原文为准。",
        concl="评长程策略先看交接处：扩大原子技能的起始状态覆盖，往往比加大模型更直接。",
        related=[MANIP, SIMEVAL, IL],
        abbr=[
            (
                "PRTI",
                "Perturbed/Extended atomic Initial states",
                "扩展原子起始状态的训练策略（论文提出）",
            ),
            ("SR", "Success Rate", "成功率"),
            ("IL", "Imitation Learning", "模仿学习"),
        ],
    ),
    dict(
        a="a1",
        g="导航、足式与规划",
        slug="nanoflann-product-manifold-kdtree",
        short="nanoflann",
        arxiv="2610.15215",
        title="nanoflann: A Header-Only KD-Tree Library for Exact Nearest-Neighbor Search on Compile-Time Product Manifolds",
        code="https://github.com/jlblancoc/nanoflann",
        project=None,
        open=OPEN,
        open_note="长期维护的 header-only C++ 库；乘积流形接口需 C++17。",
        tags=[
            "paper",
            "library",
            "kd-tree",
            "nearest-neighbor",
            "motion-planning",
            "cpp",
            "almeria",
        ],
        one="把旋转、位姿等乘积流形的距离与剪枝规则编译期纳入精确 KD-tree 近邻搜索，避免对原始坐标直接用欧氏距离。",
        why="机器人状态空间的近邻搜索若忽略几何，采样规划与定位会得到错误邻居。",
        mech="编译期组合各因子流形的度量与包围盒剪枝；另提供动态索引。",
        metrics="库论文；性能对比以原文与仓库 benchmark 为准。",
        concl="采样规划或 SLAM 需要 SE(3) 近邻时，优先用支持流形度量的库而非对坐标拼接做欧氏搜索。",
        related=[M + "trajectory-optimization.md", M + "a-star.md", C + "configuration-space.md"],
        abbr=[
            ("KD-tree", "k-dimensional tree", "空间划分近邻索引"),
            ("NN", "Nearest Neighbor", "近邻搜索"),
            ("SE(3)", "Special Euclidean group", "刚体位姿空间"),
        ],
    ),
    dict(
        a="a1",
        g="VLA 与动作生成",
        slug="cs-vgb-inference-time-scaling",
        short="CS-VGB",
        arxiv="2610.15281",
        title="Imperfect Verifiers, Exact Targets: Inference-Time Scaling for Continuous Generation",
        code="https://github.com/KraitGit/CS_VGB",
        project=None,
        open=OPEN,
        open_note="仓库可达。",
        tags=["paper", "inference-time-scaling", "diffusion", "verifier", "multi-robot-planning"],
        one="通过配对去噪与再加噪的回溯搜索，在验证器有误差时仍保持目标奖励倾斜分布的精确性。",
        why="推理期用验证器引导连续生成时，验证器误差会把采样分布推离目标；需要有保证的搜索规则。",
        mech="搜索在去噪/再加噪配对转移上进行；理论保证依赖其转移与正则条件；除图像外测试多机器人规划。",
        metrics="图像生成与多机器人规划实验；数值以原文为准。",
        concl="推理期扩展不等于贪心选高分样本；配对转移 + 回溯能在不完美验证器下保住目标分布。",
        related=[C + "diffusion-model.md", MARL, M + "trajectory-optimization.md"],
        abbr=[
            ("VGB", "Verifier-Guided Backtracking", "验证器引导回溯（论文方法缩写）"),
            ("ITS", "Inference-Time Scaling", "推理期扩展"),
            ("MRP", "Multi-Robot Planning", "多机器人规划"),
        ],
    ),
    dict(
        a="a1",
        g="世界模型与记忆",
        slug="recap-long-horizon-world-model-memory",
        short="ReCAP",
        arxiv="2610.14583",
        title="Remembering What Matters for Long-Horizon Robotic World Models",
        code=None,
        project="https://alexander-wu.github.io/recap-project-page/",
        open=PART,
        open_note="Hugging Face 提供模型权重（easylearning/recap-robot-models）；项目页写代码即将发布，入库日未列 GitHub。",
        resource="https://huggingface.co/easylearning/recap-robot-models",
        tags=["paper", "world-model", "memory", "long-horizon", "video-prediction"],
        one="冻结世界模型，只保留初始场景锚点与近期完整视觉动作块，移除陈旧生成历史，改善长时给定动作的视频预测。",
        why="长程滚动预测中陈旧的自生成历史会污染上下文；记忆该留什么比留多少更重要。",
        mech="上下文 = 初始锚点 + 近期块；不重训骨干。记忆冲突解释可能与上下文长度效应混合。",
        metrics="长时视频预测指标；以原文为准。",
        concl="长程世界模型先做上下文裁剪再谈扩容：锚点 + 近期窗口是便宜且有效的基线。",
        related=[GWM, WAM, C + "functional-taxonomy-world-models.md"],
        abbr=[
            ("WM", "World Model", "环境前向预测模型"),
            ("HF", "Hugging Face", "模型托管平台"),
            ("FVD", "Fréchet Video Distance", "视频生成质量指标"),
        ],
    ),
    dict(
        a="a1",
        g="VLA 与动作生成",
        slug="flowvla-pluggable-3d-flow",
        short="FlowVLA",
        arxiv="2610.16160",
        title="FlowVLA: Enabling World Dynamics Understanding through Pluggable 3D Flow Learning",
        code="https://github.com/MMAI-Laboratory/flowvla",
        project="https://mmai-laboratory.github.io/flowvla/",
        open=OPEN,
        open_note="公众号只列项目页；项目页链接的 GitHub 仓库经 ls-remote 可达。",
        tags=["paper", "vla", "3d-flow", "auxiliary-supervision", "manipulation"],
        one="只在训练时加入点级与末端执行器级三维流监督，让二维 VLA 对齐场景运动与控制信号；推理移除辅助头。",
        why="二维 VLA 缺少动态理解；把三维知识留在训练阶段可避免推理成本上升。",
        mech="可插拔 3D flow 头；监督来自点流与 EE 流；推理与原 VLA 等价。",
        metrics="以原文为准。",
        concl="辅助三维监督是提升 VLA 动态理解的低成本路径，前提是推理时不额外依赖三维输入。",
        related=[VLA, MANIP, WAM],
        abbr=[
            ("VLA", "Vision-Language-Action", "视觉–语言–动作策略"),
            ("EE", "End-Effector", "末端执行器"),
            ("3D flow", "3D scene flow", "三维场景流"),
        ],
    ),
    dict(
        a="a1",
        g="VLA 与动作生成",
        slug="delta-flow-matching-robot-foundation-models",
        short="Δ-Flow",
        arxiv="2610.15794",
        title="Δ Flow Matching for Robot Foundation Models",
        code="https://github.com/NTUMARS/delta-flow-matching",
        project="https://zuo-kuangji.github.io/delta-flow/",
        open=OPEN,
        open_note="项目页链接的 GitHub 仓库可达。",
        tags=["paper", "flow-matching", "vla", "wam", "one-step", "ntu"],
        one="用带轻微噪声的近期历史替代固定高斯起点，使动作与视觉生成成为残差预测，缩短生成路径并研究单步推理。",
        why="从纯噪声起步浪费了动作/视频的时间连续性；历史起点让一步推理更可行。",
        mech="起点 = 噪声化历史；免重训替换仅限所述预训练 WAM 设置。",
        metrics="在 VLA 与 WAM 上研究单步推理；数值以原文为准。",
        concl="改起点比改求解器更省：残差式流匹配是 VLA/WAM 降延迟的通用技巧，但免重训结论有适用范围。",
        related=[FM, WAM, VLA, "./paper-cnp-flow-conditional-source-distribution.md"],
        abbr=[
            ("FM", "Flow Matching", "流匹配"),
            ("WAM", "World Action Model", "世界动作模型"),
            ("VLA", "Vision-Language-Action", "视觉–语言–动作策略"),
        ],
    ),
    dict(
        a="a1",
        g="人形与移动操作",
        slug="aida-agent-interface-aligned-mobile-manipulation",
        short="AIDA",
        arxiv="2610.15572",
        title="Look Before You Move: Agent-Interface-Aligned Supervision for Open-World Mobile Manipulation",
        code=None,
        project="https://connerqiu.github.io/look-before-you-move/",
        open=NOLINK,
        open_note="项目页入库日未列 GitHub。",
        tags=["paper", "mobile-manipulation", "active-vision", "demonstration", "sim2real"],
        one="让教师通过部署时相同的技能接口采集示范，显式覆盖「保持目标可见」与「丢失后恢复」，仿真训练后真机迁移。",
        why="移动操作数据质量不只取决于轨迹，还取决于示范是否包含主动看的行为。",
        mech="接口对齐示范 + 主动视觉监督。",
        metrics="仿真训练、真机迁移；以原文为准。",
        concl="采示范前先对齐接口：教师与策略用同一套技能接口，数据才可直接复用。",
        related=[LOCOMANIP, IL, NAV],
        abbr=[
            (
                "AIDA",
                "Agent-Interface-Aligned Demonstration/supervision",
                "接口对齐监督（论文方法）",
            ),
            ("MM", "Mobile Manipulation", "移动操作"),
            ("IL", "Imitation Learning", "模仿学习"),
        ],
    ),
    dict(
        a="a1",
        g="VLA 与动作生成",
        slug="prism-precision-forcing-vla",
        short="PRISM (Precision Forcing)",
        arxiv="2610.14704",
        title="PRISM: Precision Forcing for Generalist Vision-Language-Action Policies",
        code="https://github.com/seen-e/PRISM",
        project="https://seen-e.github.io/PRISM/",
        open=OPEN,
        open_note="项目页链接的 GitHub 仓库可达。与库内同名 PRISM 节点为不同论文。",
        tags=["paper", "vla", "flow-matching", "precision-manipulation", "loss-weighting"],
        one="按精度需求对流匹配监督重加权，并把精度需求写入条件表示，让训练更关注容差敏感时刻。",
        why="流匹配损失平均分摊到所有时刻，连接器插拔、拧螺丝等关键瞬间得不到足够监督。",
        mech="精度权重 + 条件编码；实验涉及连接器与拧螺丝。",
        metrics="以原文为准。",
        concl="精密操作先改监督分配再扩模型：对关键时刻加权是便宜的第一步。",
        related=[VLA, FM, MANIP],
        abbr=[
            ("VLA", "Vision-Language-Action", "视觉–语言–动作策略"),
            ("FM", "Flow Matching", "流匹配"),
            ("PF", "Precision Forcing", "精度强制加权"),
        ],
    ),
    dict(
        a="a1",
        g="操作与柔顺控制",
        slug="residual-compliant-policy",
        short="Residual Compliant Policy",
        arxiv="2610.14035",
        title="Residual Compliant Policy: Learning Compliant Policies from Few Demonstrations",
        code=None,
        project="https://residual-compliant-policy.github.io/",
        open=TBD,
        open_note="项目页含「coming soon」字样，入库日未列 GitHub。",
        tags=["paper", "compliance", "admittance-control", "residual-rl", "few-shot"],
        one="遥操作中显式选择柔顺模式，结合轨迹重定向与物理优化，再由残差 RL 调整导纳控制参考。",
        why="示范中的柔顺性通常隐含在力交互里，难以被表达和迁移。",
        mech="模式选择 → 重定向 + 物理优化 → 残差 RL 调导纳参考；实验集中于 FR3 翻转与推动。",
        metrics="少示范；结论限论文任务范围。",
        concl="让柔顺成为显式变量（模式 + 参考）再学残差，比端到端学力控更数据高效。",
        related=[M + "admittance-control.md", M + "residual-policy-learning.md", MANIP],
        abbr=[
            ("RL", "Reinforcement Learning", "强化学习"),
            ("FR3", "Franka Research 3", "Franka 机械臂"),
            ("Admittance", "Admittance Control", "导纳控制"),
        ],
    ),
    dict(
        a="a1",
        g="多机器人协作与评测",
        slug="anyrank-anytime-policy-ranking",
        short="AnyRank",
        arxiv="2610.14603",
        title="AnyRank: Reliable Anytime Policy Ranking via Sequential Testing",
        code="https://github.com/irom-princeton/multi_hypothesis_test",
        project="https://abadithela.github.io/anyrank-page/",
        open=OPEN,
        open_note="仓库可达（另有 irom-princeton/AnyRank）。",
        tags=["paper", "evaluation", "sequential-testing", "policy-ranking", "princeton"],
        one="用序贯检验把真机试验优先分配给难区分的策略，任意停止时刻都控制错误排名概率；可利用仿真/世界模型先验。",
        why="小样本成功率排名常被误读；需要可随时停止且有统计保证的评测流程。",
        mech="多假设序贯检验 + 自适应分配；先验可不完美。",
        metrics="以原文为准。",
        concl="策略比较要带统计保证：先验帮分配，序贯检验保结论。",
        related=[SIMEVAL, C + "sim-vs-real-eval-gap.md", M + "roboarena.md"],
        abbr=[
            ("SPRT", "Sequential Probability Ratio Test", "序贯检验"),
            ("SR", "Success Rate", "成功率"),
            ("WM", "World Model", "可提供先验的世界模型"),
        ],
    ),
    dict(
        a="a1",
        g="多机器人协作与评测",
        slug="collaborative-teleoperation-multi-operator-multi-robot",
        short="多操作员多机器人遥操作界面",
        arxiv="2610.15001",
        title="Design and Evaluation of a Collaborative Teleoperation Interface for Multi-Operator Multi-Robot Systems in Challenging Environments",
        code="https://github.com/kenanalperen/NPP-Digital-Twin",
        project=None,
        open=OPEN,
        open_note="核电站数字孪生资源仓库可达。",
        tags=["paper", "teleoperation", "multi-robot", "digital-twin", "hri"],
        one="在核电站数字孪生中比较共处共享、远程共享与共处独立三种界面，同时衡量任务效率、认知负荷与态势感知。",
        why="多操作员多机器人任务中态势共享方式直接影响协作效率。",
        mech="用户实验；不能替代真实灾害现场验证。",
        metrics="用户实验指标；以原文为准。",
        concl="界面设计要把态势感知与负荷一起量化，而不只看完成时间。",
        related=[TELEOP, MULTI],
        abbr=[
            ("NPP", "Nuclear Power Plant", "核电站场景"),
            ("SA", "Situational Awareness", "态势感知"),
            ("HRI", "Human-Robot Interaction", "人机交互"),
        ],
    ),
    dict(
        a="a1",
        g="多机器人协作与评测",
        slug="b-human-26-scalable-team-play",
        short="B-Human '26",
        arxiv="2610.14733",
        title="B-Human '26 - Scalable Team Play up to 11 vs. 11",
        code="https://github.com/bhuman/MachineLearning/tree/master/IsaacGymRL",
        project=None,
        open=OPEN,
        open_note="仓库可达。",
        tags=["paper", "robocup", "humanoid-soccer", "multi-robot", "team-play"],
        one="总结从标准平台向更大型人形机器人的迁移，以及支持不同队伍规模直至 11 vs 11 的行为架构。",
        why="足球队伍规模扩大后，行为架构与角色分配需要可扩展设计。",
        mech="十一人对抗来自展示赛经验，应与常规比赛和受控基准区分。",
        metrics="赛事经验；以原文为准。",
        concl="RoboCup 架构向大队伍扩展时，角色分配与通信比单机技能更关键。",
        related=[T + "humanoid-soccer.md", MULTI],
        abbr=[
            ("SPL", "Standard Platform League", "RoboCup 标准平台组"),
            ("RL", "Reinforcement Learning", "强化学习"),
            ("HL", "Humanoid League", "人形组"),
        ],
    ),
    dict(
        a="a1",
        g="世界模型与记忆",
        slug="physimcode-physics-video-to-code",
        short="PhySimCode",
        arxiv="2610.15722",
        title="PhySimCode: A Benchmark and Evaluation Method for Physics Video to Code Generation",
        code="https://github.com/SourajaKundu/PhySimCode",
        project="https://sourajakundu.github.io/PhySimCode/",
        open=OPEN,
        open_note="仓库可达。",
        tags=["paper", "benchmark", "video-to-code", "physics", "simulation"],
        one="要求模型从原始视频恢复物理规律、参数并生成可运行仿真代码；分别评规律、仿真等价性、可执行性与参数恢复。",
        why="会描述物理不等于会重建动力学；需要可执行的评测。",
        mech="基准 + 多维评测方法。",
        metrics="以原文为准。",
        concl="视频→仿真评测必须跑代码；只比描述会高估模型。",
        related=[V2S, SIMEVAL, "./paper-video2world-coding-agents-benchmark.md"],
        abbr=[
            ("V2C", "Video-to-Code", "视频到代码生成"),
            ("Sim", "Simulation", "仿真"),
            ("LLM", "Large Language Model", "大语言模型"),
        ],
    ),
    dict(
        a="a1",
        g="世界模型与记忆",
        slug="bridgewam-latent-bridge-queries",
        short="BridgeWAM",
        arxiv="2610.14746",
        title="BridgeWAM: Bridging Video and Action Experts via Latent Bridge Queries",
        code="https://github.com/jxfang04/BridgeWAM-code",
        project=None,
        open=OPEN,
        open_note="仓库可达。",
        tags=["paper", "wam", "video-expert", "action-expert", "latent-queries"],
        one="用潜在桥接查询汇聚视觉、语言与本体信息，动作专家按需读取，避免视频与动作专家逐层密集耦合。",
        why="视频专家与动作专家密集耦合代价高；只传递动作所需信息更省。",
        mech="latent bridge queries；含泛化与延迟分析。轻量动作头不代表整体轻量。",
        metrics="以原文为准。",
        concl="WAM 的视频–动作接口应窄而可控：桥接查询是一种可行的信息瓶颈。",
        related=[WAM, GWM, VLA],
        abbr=[
            ("WAM", "World Action Model", "世界动作模型"),
            ("LBQ", "Latent Bridge Query", "潜在桥接查询"),
            ("MoE", "Mixture of Experts", "专家混合"),
        ],
    ),
    dict(
        a="a1",
        g="导航、足式与规划",
        slug="gg-nav-goal-grounding-cot",
        short="GG-Nav",
        arxiv="2610.14541",
        title="GG-Nav: Goal-oriented Grounding Chain-of-Thought Elicits Reasoning in Navigation Foundation Models",
        code="https://github.com/DuangZhu/GG_Nav",
        project="https://duangzhu.github.io/GG-Nav-Project-Page/",
        open=OPEN,
        open_note="仓库可达。",
        tags=["paper", "navigation", "chain-of-thought", "foundation-model", "unitree-go2"],
        one="结构化推理先把中间帧转为目标证据，再结合当前观察决定行动或追问；覆盖目标导航基准与 Go2 真机。",
        why="导航基础模型常漏看目标或停错位置；把沿途观察积累成证据可减少此类失败。",
        mech="goal-grounding CoT；行动 / 追问两种输出。",
        metrics="以原文为准。",
        concl="导航推理的证据链应显式化，便于诊断漏看与停止错误。",
        related=[NAV, T + "vision-language-navigation.md"],
        abbr=[
            ("CoT", "Chain-of-Thought", "思维链"),
            ("ObjNav", "Object Navigation", "目标导航"),
            ("NFM", "Navigation Foundation Model", "导航基础模型"),
        ],
    ),
    dict(
        a="a1",
        g="世界模型与记忆",
        slug="imprint-multi-stream-latent-memory",
        short="IMPRINT",
        arxiv="2610.14406",
        title="Memory Without Curation: Learning Multi-Stream Latent Memory for Robot Policies",
        code=None,
        project="https://yilin-wu98.github.io/imprint_website/",
        open=NOLINK,
        open_note="项目页入库日未列 GitHub。",
        tags=["paper", "memory", "policy", "long-horizon", "latent-memory"],
        one="联合记忆观察、动作与本体信息：先训练历史重建与进度估计，再与策略联合微调；推理增量更新潜在记忆。",
        why="长历史带来延迟与视觉捷径学习；需要无需人工挑选的增量记忆。",
        mech="多流潜在记忆 + 重建/进度预训练。",
        metrics="以原文为准。",
        concl="策略记忆的训练信号可以来自历史重建与进度，而非人工标注。",
        related=[IL, MANIP, "./paper-ttt-rm-residual-memory.md"],
        abbr=[
            ("TTT", "Test-Time Training", "测试时训练"),
            ("IL", "Imitation Learning", "模仿学习"),
            ("Proprio", "Proprioception", "本体感知"),
        ],
    ),
    dict(
        a="a1",
        g="操作与柔顺控制",
        slug="bampc-compute-budget-sampling-mpc",
        short="BAMPC",
        arxiv="2610.14064",
        title="Do My Samples Help? Compute Budget Allocation for Real-World Sampling-Based MPC in Contact-Rich Manipulation",
        code="https://github.com/bampc-framework/bampc-code",
        project="https://bampc-framework.github.io/",
        open=OPEN,
        open_note="项目页链接的 bampc-code 仓库可达（基于 hydrax）。",
        tags=["paper", "mpc", "sampling-based-control", "contact-rich", "compute-budget"],
        one="同一控制栈下比较动作探索、动力学/状态不确定性、重规划频率与时间对齐的预算分配；动力学随机化并非总最优。",
        why="采样 MPC 的算力有限，预算该花在哪取决于主要仿真迁移误差来源。",
        mech="统一框架切换预算分配；接触操作真机实验。",
        metrics="以原文为准。",
        concl="先诊断 sim2real 误差来源，再决定预算投向探索还是不确定性采样。",
        related=[MPC, M + "mppi.md", C + "contact-rich-manipulation.md", S2R],
        abbr=[
            ("MPC", "Model Predictive Control", "模型预测控制"),
            ("MPPI", "Model Predictive Path Integral", "采样式 MPC"),
            ("DR", "Domain Randomization", "域随机化"),
        ],
    ),
    dict(
        a="a1",
        g="操作与柔顺控制",
        slug="rain-region-aware-interaction-networks",
        short="RAIN",
        arxiv="2610.14039",
        title="RAIN: Robotic Task Generalization with Region-Aware Interaction Networks",
        code="https://github.com/Seung-Hun-Lee/RAIN",
        project="https://seung-hun-lee.github.io/projects/RAIN/",
        open=OPEN,
        open_note="项目页链接的 GitHub 仓库可达。",
        tags=["paper", "manipulation", "task-generalization", "region-mask", "libero"],
        one="用动作类型 + 目标区域掩码表示子任务，结合跨视角编码、轨迹预测与完成判断；参考轨迹重定向合成新目标交互。",
        why="技能绑定到目标区域后才能跨任务组合。",
        mech="LIBERO-Analogy 与真机桌面任务。",
        metrics="以原文为准。",
        concl="子任务的「动作 × 区域」表示是组合泛化的实用中间层。",
        related=[MANIP, IL, "./paper-robocombine-combinatorial-generalization.md"],
        abbr=[
            ("RAIN", "Region-Aware Interaction Network", "本文方法"),
            ("LIBERO", "Lifelong Robot Learning Benchmark", "操作基准"),
            ("IL", "Imitation Learning", "模仿学习"),
        ],
    ),
    dict(
        a="a1",
        g="VLA 与动作生成",
        slug="vision-predictor-action-jepa",
        short="VPA",
        arxiv="2610.15556",
        title="Vision-Predictor-Action: Towards Efficient Embodied AI via Neuro-Symbolic JEPA Predictors",
        code="https://github.com/ylee-sling/Vision-Predictor-Action",
        project=None,
        open=OPEN,
        open_note="仓库可达。",
        tags=["paper", "jepa", "flow-matching", "action-chunking", "neuro-symbolic"],
        one="组合离散技能选择、JEPA 潜在预测与流匹配动作块生成，以潜在预测循环降低串行推理成本。",
        why="VLA 串行推理昂贵；潜在预测器可替代部分视觉–语言前向。",
        mech="当前证据是 LIBERO-10 单次训练概念验证；速度与成功率来自不同相机配置。",
        metrics="概念验证；以原文为准。",
        concl="JEPA 预测器作动作生成中枢是方向性尝试，结论尚需更强实验。",
        related=[FM, CHUNK, VLA],
        abbr=[
            ("JEPA", "Joint-Embedding Predictive Architecture", "联合嵌入预测"),
            ("VPA", "Vision-Predictor-Action", "本文框架"),
            ("FM", "Flow Matching", "流匹配"),
        ],
    ),
    dict(
        a="a1",
        g="VLA 与动作生成",
        slug="cfg-policy-unified-diffusion-policy",
        short="CFG-Policy",
        arxiv="2610.14936",
        title="CFG-Policy: Unified Diffusion Policy Learning Framework via Classifier-free Guidance",
        code=None,
        project="https://zhy-liu.github.io/cfg-policy-page",
        open=TBD,
        open_note="项目页含「coming soon」字样，入库日未列代码仓库。",
        tags=["paper", "diffusion-policy", "classifier-free-guidance", "reinforcement-learning"],
        one="把探索与利用拆成两条生成分支，采样时组合分数/速度场，以减少 RL 微调中的行为多样性损失。",
        why="扩散策略 RL 微调常坍缩多样性；CFG 式组合可在采样期调节。",
        mech="在线与离线转在线设置冻结的分支不同。",
        metrics="以原文为准。",
        concl="扩散策略 RL 微调可借 CFG 把探索/利用分离为可调旋钮。",
        related=[DP, RL, C + "diffusion-model.md"],
        abbr=[
            ("CFG", "Classifier-Free Guidance", "无分类器引导"),
            ("DP", "Diffusion Policy", "扩散策略"),
            ("RL", "Reinforcement Learning", "强化学习"),
        ],
    ),
    dict(
        a="a1",
        g="人形与移动操作",
        slug="ttco-test-time-compute-humanoid-bfm",
        short="TTCO",
        arxiv="2610.14896",
        title="Test-Time Compute for Humanoid Behavior Foundation Models",
        code=None,
        project="https://bfmttc.github.io/",
        open=TBD,
        open_note="项目页列出 GitHub 链接（Gamma-Robot/bfm_ttc），但入库日 ls-remote 不可达，按待发布处理。",
        tags=["paper", "behavior-foundation-model", "humanoid", "test-time-optimization"],
        one="闭环仿真中搜索更合适的输入命令，让既有行为基础模型更接近期望运动，并可生成模型专属动作标签；无需重训。",
        why="BFM 的命令–执行差距常由命令而非模型造成。",
        mech="依赖仿真与采样计算。",
        metrics="以原文为准。",
        concl="先用测试时优化压缩命令执行差距，再决定是否重训 BFM。",
        related=[BFM, HUM, RL],
        abbr=[
            ("BFM", "Behavior Foundation Model", "行为基础模型"),
            ("TTC", "Test-Time Compute", "测试时计算"),
            ("WBC", "Whole-Body Control", "全身控制"),
        ],
    ),
    dict(
        a="a1",
        g="导航、足式与规划",
        slug="tempoloco-motion-adaptive-temporal-receptive-fields",
        short="TempoLoco",
        arxiv="2610.14753",
        title="Beyond Fixed Visual History: Learning Motion-Adaptive Temporal Receptive Fields for Complex-Terrain Quadruped Locomotion",
        code=None,
        project="https://capability-aware.github.io/",
        open=PART,
        open_note="项目页链接 HaoZhangRobotics/capability_rl 与 capabilitylab（可达），与本文代码的对应关系待核实；页面另含「coming soon」。",
        tags=["paper", "quadruped", "locomotion", "perceptive-locomotion", "depth"],
        one="从长深度历史学习互补时间视图，再按本体状态与地形动态融合，让视觉提前量适应运动能力。",
        why="固定视觉历史无法随速度/地形变化调整提前量。",
        mech="复杂地形仿真 + 带固定机械臂的 B2 真机。",
        metrics="以原文为准。",
        concl="感知时间窗应随运动状态自适应，而非固定帧数。",
        related=[
            LOCO,
            T + "stair-obstacle-perceptive-locomotion.md",
            M + "pie-perceptive-locomotion.md",
        ],
        abbr=[
            ("TRF", "Temporal Receptive Field", "时间感受野"),
            ("B2", "Unitree B2", "四足平台"),
            ("RL", "Reinforcement Learning", "强化学习"),
        ],
    ),
    dict(
        a="a1",
        g="导航、足式与规划",
        slug="loadnav-load-aware-quadruped-navigation",
        short="LOADNav",
        arxiv="2610.14596",
        title="Load-Aware Quadruped Navigation with Online Contact-based Geometry Estimation",
        code="https://github.com/qiii04/LOADNav",
        project="https://qiii04.github.io/LOADNav/",
        open=OPEN,
        open_note="项目页链接的 GitHub 仓库可达。",
        tags=["paper", "quadruped", "navigation", "load-carrying", "contact-estimation"],
        one="学习随负载几何调整路径与间距，CoGE 从偶发接触在线推断未知负载形状，避免固定足迹膨胀的过保守。",
        why="四足负载导航若不估计物体外形，要么碰撞要么过保守。",
        mech="几何估计依赖接触信息逐步更新。",
        metrics="以原文为准。",
        concl="负载导航把几何估计做成在线变量，比固定安全裕度更实用。",
        related=[
            LOCO,
            C + "contact-estimation.md",
            C + "hierarchical-quadruped-navigation-stack.md",
        ],
        abbr=[
            ("CoGE", "Contact-based Geometry Estimation", "接触几何估计"),
            ("Nav", "Navigation", "导航"),
            ("RL", "Reinforcement Learning", "强化学习"),
        ],
    ),
    dict(
        a="a1",
        g="多机器人协作与评测",
        slug="care-team-response-beliefs-multi-robot",
        short="CARE",
        arxiv="2610.14551",
        title="Before We Act: Learning Action-Conditioned Team-Response Beliefs for Decentralized Multi-Robot Manipulation",
        code=None,
        project="https://jeong-zju.github.io/care/",
        open=NOLINK,
        open_note="项目页入库日未列 GitHub。",
        tags=["paper", "multi-robot", "decentralized", "manipulation", "belief"],
        one="从局部历史学习团队响应信念，把候选动作的直接收益与队友适应收益分开建模，选择性修正名义策略。",
        why="去中心化协作中队友响应会改变最佳动作，需显式预测。",
        mech="动作条件信念 + 选择性修正。",
        metrics="以原文为准。",
        concl="多机器人操作先预测「我动之后队友怎么动」，再修正自身动作。",
        related=[MARL, MULTI, MANIP, "./paper-akule-sparse-interaction-diffusion.md"],
        abbr=[
            ("CARE", "aCtion-conditioned teAm-REsponse beliefs", "本文方法缩写"),
            ("MARL", "Multi-Agent RL", "多智能体强化学习"),
            ("Dec", "Decentralized", "去中心化"),
        ],
    ),
    dict(
        a="a1",
        g="感知与 4D 世界",
        slug="tartanmatch-universal-dense-correspondence",
        short="TartanMatch",
        arxiv="2610.14513",
        title="TartanMatch: Towards Universal Dense Correspondence Across Modalities",
        code="https://github.com/castacks/tartanmatch",
        project="https://tartanmatch.github.io/",
        open=OPEN,
        open_note="项目页链接的 GitHub 仓库可达（CMU AirLab）。",
        tags=["paper", "perception", "dense-correspondence", "multimodal", "cmu"],
        one="单一网络适配 RGB、事件、热成像、深度与投影 LiDAR 之间的稠密匹配，混合真实与合成数据训练零样本跨模态对应。",
        why="跨传感器对应是多模态定位与配准的基础。",
        mech="共享网络 + 混合数据。",
        metrics="以原文为准。",
        concl="跨模态稠密匹配可用一个模型统一，但零样本性能依赖训练模态覆盖。",
        related=[M + "stereo-matching-foundation-models.md", C + "lidar-sensing.md"],
        abbr=[
            ("RGB", "Red-Green-Blue", "彩色图像"),
            ("LiDAR", "Light Detection and Ranging", "激光雷达"),
            ("Dense corr.", "Dense Correspondence", "稠密对应"),
        ],
    ),
    dict(
        a="a1",
        g="世界模型与记忆",
        slug="pilot-visual-subgoal-world-model-planning",
        short="PILOT (Visual Subgoals)",
        arxiv="2610.14335",
        title="PILOT: Guiding Long-Horizon World-Model Planning for Robot Manipulation with Visual Subgoals and Policy Priors",
        code=None,
        project="https://pilot-planner.github.io/",
        open=TBD,
        open_note="项目页含「coming soon」字样，入库日未列 GitHub。与库内 PILOT 足式节点为不同论文。",
        tags=["paper", "world-model", "planning", "cem", "subgoal", "manipulation"],
        one="先生成中间视觉子目标，再用策略初始化 CEM 搜索，通过动作条件世界模型滚动优化。",
        why="长程世界模型规划需要子目标与先验压缩搜索空间。",
        mech="UR5e 与仿真；需计入在线规划预算。",
        metrics="以原文为准。",
        concl="视觉子目标 + 策略先验是让 WM 规划在长程任务可用的两个杠杆。",
        related=[GWM, WAM, MANIP, M + "cma-es.md"],
        abbr=[
            ("CEM", "Cross-Entropy Method", "交叉熵采样优化"),
            ("WM", "World Model", "世界模型"),
            ("UR5e", "Universal Robots UR5e", "机械臂平台"),
        ],
    ),
    dict(
        a="a1",
        g="多机器人协作与评测",
        slug="akule-sparse-interaction-diffusion",
        short="Akule",
        arxiv="2610.14248",
        title="Akule: Fast and Scalable Multi-Robot Motion Planning via Sparse Interaction Diffusion",
        code="https://github.com/akule-diff/Akule",
        project="https://akule-diff.github.io/",
        open=OPEN,
        open_note="项目页链接的 GitHub 仓库可达。",
        tags=["paper", "multi-robot", "motion-planning", "diffusion", "scalability"],
        one="冻结单机器人扩散先验，学习成对修正、交互选择与组合权重，只为必要交互分配计算；生成轨迹仍交搜索规划器验证修复。",
        why="大队伍联合规划计算爆炸；稀疏交互是可扩展的关键。",
        mech="应比较完整规划流程而非仅生成器。",
        metrics="以原文为准。",
        concl="多机器人扩散规划的可扩展性来自稀疏交互选择，而非更大模型。",
        related=[MARL, M + "trajectory-optimization.md", C + "diffusion-model.md"],
        abbr=[
            ("MRMP", "Multi-Robot Motion Planning", "多机器人运动规划"),
            ("DM", "Diffusion Model", "扩散模型"),
            ("CBS", "Conflict-Based Search", "多智能体搜索规划器"),
        ],
    ),
    dict(
        a="a1",
        g="策略与后训练",
        slug="ditar-task-relevant-latent-dynamics-reward",
        short="DiTaR",
        arxiv="2610.14159",
        title="Disentangling Task-Relevant Latent Dynamics for Reward Modeling",
        code=None,
        project="https://ditarr.github.io/",
        open=TBD,
        open_note="项目页列出 GitHub 链接（phai-lab/DiTaR），入库日 ls-remote 不可达，按待发布处理。",
        tags=["paper", "reward-model", "video", "latent-dynamics", "reinforcement-learning"],
        one="先从多帧视频解耦任务相关潜在动态，再以历史与语言条件流模型构造方向性稠密奖励。",
        why="奖励模型应追踪任务变化而非画面变化；视觉干扰会破坏奖励可靠性。",
        mech="仿真及真机离线重标注。",
        metrics="以原文为准。",
        concl="视频奖励模型先解耦任务相关动态，再谈稠密化。",
        related=[
            RL,
            M + "inverse-reinforcement-learning.md",
            C + "robot-data-supervision-signal-types.md",
        ],
        abbr=[
            ("RM", "Reward Model", "奖励模型"),
            ("FM", "Flow Matching", "流匹配"),
            ("RL", "Reinforcement Learning", "强化学习"),
        ],
    ),
    dict(
        a="a1",
        g="操作与柔顺控制",
        slug="hardware-aware-llm-vlm-zero-shot-task-planning",
        short="硬件感知 LLM-VLM 零样本规划",
        arxiv="2610.14106",
        title="Embodied Zero-Shot Task Planning via Hardware-Aware LLM-VLM Integration for Grounded Robotic Manipulation",
        code="https://github.com/Ltankersley15/SK-Zero-Shot-Research",
        project=None,
        open=OPEN,
        open_note="仓库可达。",
        tags=["paper", "llm", "vlm", "task-planning", "isaac-sim"],
        one="LLM 输出符号任务，VLM 确认语义，确定性模块处理坐标、抓取、碰撞与可达性；在 Isaac Sim FR3 数字孪生验证。",
        why="语言规划与几何执行分工明确，才能减少幻觉落地错误。",
        mech="仿真操作方案。",
        metrics="以原文为准。",
        concl="LLM 只做符号层，几何可行性交给确定性模块。",
        related=[M + "saycan.md", C + "llm-robotics-control-interfaces.md", MANIP],
        abbr=[
            ("LLM", "Large Language Model", "大语言模型"),
            ("VLM", "Vision-Language Model", "视觉语言模型"),
            ("FR3", "Franka Research 3", "机械臂"),
        ],
    ),
    dict(
        a="a1",
        g="感知与 4D 世界",
        slug="tide-tides-4d-liquid-forecasting",
        short="TIDE/TIDES",
        arxiv="2610.16167",
        title="Interactive 4D Volumetric Liquid Forecasting under Moving-Solid Interaction",
        code="https://github.com/jongwonryu/TIDE-TIDES",
        project=None,
        open=OPEN,
        open_note="仓库可达。",
        tags=["paper", "fluid", "physics-prediction", "4d", "simulation"],
        one="根据已知未来固体运动预测稠密液体速度场，引入界面残差与粒子网格反馈；非双向闭环流固仿真。",
        why="液体操作需要对固体动作引发的流体响应建模。",
        mech="条件输入为给定固体轨迹。",
        metrics="以原文为准。",
        concl="液体预测模型的输入边界（固体轨迹已知）决定其在闭环控制中的用法。",
        related=[C + "differentiable-simulation.md", GWM],
        abbr=[
            ("4D", "3D + time", "时空体积"),
            ("PIC", "Particle-In-Cell", "粒子网格方法"),
            ("CFD", "Computational Fluid Dynamics", "计算流体力学"),
        ],
    ),
    dict(
        a="a1",
        g="导航、足式与规划",
        slug="lerr-nav-exploration-recovery-reminiscing",
        short="LERR-Nav",
        arxiv="2610.15362",
        title="LERR-Nav: Learning to Coordinate Exploration, Recovery, and Reminiscing for Object Navigation",
        code="https://github.com/kim-os/LERR-Nav",
        project=None,
        open=OPEN,
        open_note="仓库可达。",
        tags=["paper", "navigation", "object-navigation", "vlm", "recovery"],
        one="高层视觉语言策略协调探索、停滞恢复与记忆回访，结合文字执行摘要与地图历史；HM3D / MP3D。",
        why="导航失去进展后如何脱困应纳入策略学习。",
        mech="仿真基准。",
        metrics="以原文为准。",
        concl="脱困决策是导航策略的一等组件，而非兜底规则。",
        related=[NAV, T + "autonomous-exploration.md"],
        abbr=[
            ("ObjNav", "Object Navigation", "目标导航"),
            ("HM3D", "Habitat-Matterport 3D", "导航数据集"),
            ("VLM", "Vision-Language Model", "视觉语言模型"),
        ],
    ),
    dict(
        a="a1",
        g="多机器人协作与评测",
        slug="coplanvlm-coordinated-path-planning",
        short="CoPlanVLM",
        arxiv="2610.14341",
        title="CoPlanVLM: Coordinated Path Planning for a Robot Team using Vision Language Models",
        code="https://github.com/AICPS/CoPlanVLM",
        project=None,
        open=OPEN,
        open_note="仓库可达。",
        tags=["paper", "vlm", "multi-robot", "path-planning", "turtlebot"],
        one="结合自由空间叠图、任务分类与专用推理提示，把语言指令转为双机器人协同路径；Gazebo 与 Turtlebot 4 验证。",
        why="用 VLM 做多机器人路径需要显式空间表示。",
        mech="路径长度比较仅针对双方均成功的试验。",
        metrics="以原文为准。",
        concl="VLM 规划要配空间叠图与明确失败统计口径。",
        related=[MARL, NAV],
        abbr=[
            ("VLM", "Vision-Language Model", "视觉语言模型"),
            ("MRPP", "Multi-Robot Path Planning", "多机器人路径规划"),
            ("Gazebo", "Gazebo simulator", "ROS 仿真器"),
        ],
    ),
    dict(
        a="a1",
        g="感知与 4D 世界",
        slug="surgcast-surgical-world-model-skeletons",
        short="SurgCast",
        arxiv="2610.15645",
        title="SurgCast: Action-Conditioned Future Skeletons for Controllable Surgical World Models",
        code=None,
        project="https://surgcast.netlify.app/",
        open=NOLINK,
        open_note="项目页入库日未列 GitHub。",
        tags=["paper", "world-model", "surgical-robotics", "video-generation"],
        one="先由机器人状态与未来动作预测器械骨架，再把几何与运动描述注入视频生成器，提升手术视频预测的结构一致性。",
        why="手术世界模型需要可控结构，不能只靠像素生成。",
        mech="两套手术数据；不能解释为临床操作能力。",
        metrics="以原文为准。",
        concl="骨架作为中间表示是让视频世界模型可控的通用技巧。",
        related=[GWM, WAM],
        abbr=[
            ("WM", "World Model", "世界模型"),
            ("dVRK", "da Vinci Research Kit", "手术机器人平台"),
            ("Skeleton", "Instrument Skeleton", "器械骨架"),
        ],
    ),
    dict(
        a="a1",
        g="感知与 4D 世界",
        slug="demiurge-navigable-4d-worlds",
        short="Demiurge",
        arxiv="2610.14685",
        title="Demiurge: Forging Navigable 4D Worlds via Autoregressive Video Rerendering",
        code="https://github.com/HanxinZhu-lab/Demiurge",
        project="https://hanxinzhu-lab.github.io/demiurge_project_page/",
        open=OPEN,
        open_note="项目页链接的 GitHub 仓库可达。",
        tags=["paper", "4d", "video-generation", "world-generation", "navigable"],
        one="用动态 4D 点云缓存串联视频重渲染，结合深度补全、双流变形与几何细化维持跨轨迹一致性。",
        why="可导航视觉世界需要不同视角共享同一动态世界。",
        mech="尚不等同机器人闭环控制。",
        metrics="以原文为准。",
        concl="4D 缓存是把视频生成变成可导航世界的桥梁，但距离控制闭环仍远。",
        related=[GWM, V2S],
        abbr=[
            ("4D", "3D + time", "时空点云"),
            ("AR", "Autoregressive", "自回归生成"),
            ("NVS", "Novel View Synthesis", "新视角合成"),
        ],
    ),
    dict(
        a="a1",
        g="导航、足式与规划",
        slug="quadrotor-tendon-soft-continuum-arm-mujoco",
        short="四旋翼软连续体臂 MuJoCo 框架",
        arxiv="2610.15308",
        title="An Open MuJoCo Simulation Framework for a Quadrotor Carrying a Tendon-Driven Soft Continuum Arm",
        code="https://github.com/elhussieny/quad-soft-continuum",
        project=None,
        open=OPEN,
        open_note="仓库可达。",
        tags=["paper", "simulation", "mujoco", "aerial-manipulation", "soft-robotics"],
        one="把腱驱动连续体臂近似为柔性铰链连接的分段刚体链，与四旋翼控制联合仿真。",
        why="空中柔性操作缺少开放仿真平台。",
        mech="轨迹、增益、工作空间与扰动实验均为仿真。",
        metrics="以原文为准。",
        concl="分段刚体近似是软臂进入 MuJoCo 的实用路径。",
        related=[C + "robot-simulation-three-layers.md", SIMEVAL],
        abbr=[
            ("MuJoCo", "Multi-Joint dynamics with Contact", "物理仿真器"),
            ("UAV", "Unmanned Aerial Vehicle", "无人机"),
            ("PCC", "Piecewise Constant Curvature", "软臂近似模型"),
        ],
    ),
    dict(
        a="a1",
        g="多机器人协作与评测",
        slug="class-conditional-conformal-stl-inference",
        short="类条件保形 STL 推断",
        arxiv="2610.14352",
        title="Class-Conditional Conformal Prediction for Signal Temporal Logic Inference with Many Classes",
        code=None,
        project="https://sites.google.com/ucr.edu/multiclasswithconditionalcp?usp=sharing",
        open=NOLINK,
        open_note="项目页（Google Sites）入库日未列 GitHub。",
        tags=["paper", "formal-methods", "conformal-prediction", "temporal-logic", "ucr"],
        one="指出时序逻辑分类的准确率与公式满足度错位，固定结构后调整谓词阈值并逐类校准预测集合。",
        why="分类正确不代表轨迹满足时序公式。",
        mech="有限样本覆盖依赖类内可交换性。",
        metrics="以原文为准。",
        concl="可解释轨迹分类要同时报告准确率与公式满足度。",
        related=[SAFE, C + "control-barrier-function.md"],
        abbr=[
            ("STL", "Signal Temporal Logic", "信号时序逻辑"),
            ("CP", "Conformal Prediction", "保形预测"),
            ("UCR", "UC Riverside", "加州大学河滨分校"),
        ],
    ),
    dict(
        a="a1",
        g="人形与移动操作",
        slug="robocompiler-humanoids-composable-dynamics",
        short="RoboCompiler-Humanoids",
        arxiv="2610.14237",
        title="Composable Dynamics for Cooperative Humanoids: A RoboCompiler Extension",
        code="https://github.com/Mehdi-Heydari-Shahna/RoboCompiler-Humanoids",
        project=None,
        open=OPEN,
        open_note="仓库可达。",
        tags=["paper", "humanoid", "dynamics", "multi-robot", "cooperative"],
        one="以成员级装配与两级约简组合多个人形的动力学、手部连接与地面支撑；数值校验与 MuJoCo 仿真验证一致性。",
        why="协作人形需要可组合的动力学模型而非逐例手推。",
        mech="仿真跟踪误差不能外推到真机。",
        metrics="以原文为准。",
        concl="多人形协作的建模应模块化装配，约束在组合层处理。",
        related=[C + "floating-base-dynamics.md", MULTI, HUM],
        abbr=[
            ("WBC", "Whole-Body Control", "全身控制"),
            ("DoF", "Degrees of Freedom", "自由度"),
            ("MuJoCo", "Multi-Joint dynamics with Contact", "仿真器"),
        ],
    ),
    dict(
        a="a1",
        g="多机器人协作与评测",
        slug="rl-heterogeneous-task-allocation-spatio-temporal",
        short="海上异构任务分配 RL",
        arxiv="2610.15237",
        title="Reinforcement Learning Based Heterogeneous Task Allocation Under Spatio-Temporal Environmental Uncertainty for Multi-Robot Teams",
        code="https://github.com/RobustFieldAutonomyLab/Spatio_Temporal_MRTA",
        project=None,
        open=OPEN,
        open_note="仓库可达。",
        tags=["paper", "multi-robot", "task-allocation", "reinforcement-learning", "marine"],
        one="不同能力无人艇在时空随机扰动与动态任务下的分配，用去中心化 RL 应对观测偏差。",
        why="海上扰动使静态分配失效。",
        mech="结论适用于论文仿真设定。",
        metrics="以原文为准。",
        concl="异构队伍任务分配需把环境不确定性作为状态的一部分。",
        related=[MARL, RL],
        abbr=[
            ("MRTA", "Multi-Robot Task Allocation", "多机器人任务分配"),
            ("USV", "Unmanned Surface Vehicle", "无人艇"),
            ("RL", "Reinforcement Learning", "强化学习"),
        ],
    ),
    dict(
        a="a1",
        g="VLA 与动作生成",
        slug="pi05-learn-to-call-astra",
        short="π0.5 CALL Astra",
        arxiv="2610.14117",
        title="Make π0.5 Learn to Call Astra for Reliable Robot Manipulation",
        code=None,
        project="https://kykgeek.github.io/Pi05_CALL_Astra/",
        open=NOLINK,
        open_note="项目页入库日未列 GitHub。",
        tags=["paper", "vla", "pi0", "help-seeking", "libero"],
        one="冻结 π0.5，用复用策略特征的 CALL 头判断何时请求外部模型介入恢复；证据来自 LIBERO 仿真。",
        why="本地策略按需求助可提升可靠性，但要看调用频率与成本。",
        mech="LIBERO；关注成功、调用频率与每回合成本。",
        metrics="以原文为准。",
        concl="求助策略的评价是三元的：成功率、外部调用率、单回合成本。",
        related=[VLA, M + "π0-policy.md"],
        abbr=[
            ("VLA", "Vision-Language-Action", "视觉–语言–动作策略"),
            ("CALL", "Call head", "触发外部求助的分类头"),
            ("LIBERO", "Lifelong Robot Learning Benchmark", "操作基准"),
        ],
    ),
    # ---------------- 10-13 期（34 篇，新建 25） ----------------
    dict(
        a="a2",
        g="强化学习与数据采集",
        slug="mulligan-performance-guided-data-collection",
        short="Mulligan",
        arxiv="2610.05882",
        title="Mulligan: Performance-Guided Data Collection for Efficient On-Robot Learning",
        code="https://github.com/ankile/mulligan",
        project="https://mulligan.page/",
        open=OPEN,
        open_note="仓库可达。",
        tags=["paper", "data-collection", "on-robot-learning", "human-in-the-loop", "dagger"],
        one="先评测再采集：优先重试失败起点、补足未探索状态，策略用成功人类片段训练、价值函数利用全部交互。",
        why="真机策略越熟练，均匀采集越重复已掌握状态；采集起点是部署后学习的可优化变量。",
        mech="采集预算、初始状态范围与人工介入条件需同时控制；与价值重排序分开检验。",
        metrics="最终轮插笔任务 HiL-IDQL+Mulligan 38/50 vs 均匀 HG-DAgger 21/50；三项真机任务共 2,550 次留出盲测（公众号转述）。",
        concl="把采集预算花在失败起点上，比扩大均匀采集更划算；失败轨迹对价值函数仍有用。",
        related=[M + "dagger.md", SAFERL, C + "data-flywheel.md", RL],
        abbr=[
            ("HiL", "Human-in-the-Loop", "人在回路"),
            ("IDQL", "Implicit Diffusion Q-Learning", "离线 RL 算法"),
            ("HG-DAgger", "Human-Gated DAgger", "人工门控数据聚合"),
        ],
    ),
    dict(
        a="a2",
        g="世界模型与仿真",
        slug="wamjet-wam-acceleration-harness",
        short="WAMJET",
        arxiv="2610.03797",
        title="WAMJET: A Harness for World Action Model Acceleration",
        code="https://github.com/liulixinkerry/WAMJET",
        project="https://liulixinkerry.github.io/WAMJET/index.html",
        open=OPEN,
        open_note="仓库可达。",
        tags=["paper", "wam", "inference-acceleration", "coding-agent", "harness"],
        one="用测量、修改、验证工具让编码智能体围绕瓶颈优化 WAM 推理，跨模型、智能体与显卡测试。",
        why="WAM 推理加速是工程活；让 agent 系统化做可复现。",
        mech="区分保持输出质量的工程优化与引入近似后的延迟收益。",
        metrics="以原文为准。",
        concl="加速结果要按「是否改变输出」分类报告。",
        related=[
            WAM,
            "./paper-realtimewam-training-free-wam-acceleration.md",
            C + "agentic-coding-software-fundamentals.md",
        ],
        abbr=[
            ("WAM", "World Action Model", "世界动作模型"),
            ("GPU", "Graphics Processing Unit", "显卡"),
            ("Agent", "Coding Agent", "编码智能体"),
        ],
    ),
    dict(
        a="a2",
        g="世界模型与仿真",
        slug="affordcraft-task-ready-simulation-assets",
        short="AffordCraft",
        arxiv="2610.06643",
        title="AffordCraft: Scalable Construction of Task-Ready Simulation Assets from Single Images",
        code="https://github.com/AffordCraft/AffordCraft",
        project="https://affordcraft.github.io",
        open=OPEN,
        open_note="仓库可达。",
        tags=["paper", "simulation-assets", "articulated-objects", "real2sim", "retrieval"],
        one="根据图像与任务检索已有铰接资产并适配外观，保留部件与关节；能力依赖资产库覆盖。",
        why="从照片得到可操作仿真资产是扩展仿真训练的瓶颈。",
        mech="大规模照片物理测试支持检索路线。",
        metrics="以原文为准。",
        concl="资产构建的检索路线比逐物重建更可扩展，但受库覆盖限制。",
        related=[M + "crisp-real2sim.md", SIMEVAL, C + "robot-simulation-three-layers.md"],
        abbr=[
            ("Real2Sim", "Real to Simulation", "真实到仿真"),
            ("URDF", "Unified Robot Description Format", "资产/机器人描述格式"),
            ("Afford.", "Affordance", "可操作性"),
        ],
    ),
    dict(
        a="a2",
        g="策略与后训练",
        slug="robotuse-allocating-computation-context-decisions",
        short="RobotUse",
        arxiv="2610.04929",
        title="RobotUse: Allocating Computation, Context, and Decisions",
        code="https://github.com/robotuse-team/RobotUse",
        project="https://robotuse-team.github.io/",
        open=OPEN,
        open_note="仓库可达。",
        tags=["paper", "agent", "vlm", "task-planning", "manipulation"],
        one="智能体视觉选择目标与位姿，后端完成几何规划与控制；子目标上下文与持续更新的操作手册支持反复修正。",
        why="把物理决策留给智能体、执行交给后端，才能复用经验。",
        mech="评测包含真机经验迁移。",
        metrics="以原文为准。",
        concl="agent 式操作系统的核心是分工边界与可更新的手册。",
        related=[
            C + "llm-robotics-control-interfaces.md",
            MANIP,
            "./paper-hardware-aware-llm-vlm-zero-shot-task-planning.md",
        ],
        abbr=[
            ("VLM", "Vision-Language Model", "视觉语言模型"),
            ("Agent", "LLM Agent", "语言模型智能体"),
            ("IK", "Inverse Kinematics", "逆运动学"),
        ],
    ),
    dict(
        a="a2",
        g="世界模型与仿真",
        slug="video2world-coding-agents-benchmark",
        short="Video2World",
        arxiv="2610.04432",
        title="Video2World: Benchmarking Coding Agents for Interactive World Modeling from Embodied Videos",
        code="https://github.com/AetherLabsAI/Video2World",
        project="https://aetherlabsai.github.io/Video2World",
        open=OPEN,
        open_note="仓库可达。",
        tags=["paper", "benchmark", "coding-agent", "video-to-simulation", "world-model"],
        one="把视频到仿真视为可执行软件任务，同时检查几何、动力学与任务完成；视觉还原更好不保证交互更可靠。",
        why="评编码智能体建世界，必须跑交互。",
        mech="基准 + 执行检查。",
        metrics="以原文为准。",
        concl="视频→仿真评测的核心是可执行与可交互，而非画面相似。",
        related=[V2S, SIMEVAL, "./paper-physimcode-physics-video-to-code.md"],
        abbr=[
            ("V2W", "Video-to-World", "视频到世界建模"),
            ("Agent", "Coding Agent", "编码智能体"),
            ("Sim", "Simulation", "仿真"),
        ],
    ),
    dict(
        a="a2",
        g="灵巧操作",
        slug="eyerobot-2-active-gaze-manipulation",
        short="EyeRobot 2.0",
        arxiv="2610.03710",
        title="EyeRobot 2.0: Active Gaze for Precise Manipulation without Wrist Cameras",
        code="https://github.com/kerrj/eyerobot2",
        project="https://eyerobot2.github.io/",
        open=OPEN,
        open_note="项目页链接的 GitHub 仓库可达。",
        tags=["paper", "active-vision", "bimanual", "gaze", "manipulation"],
        one="联合主动注视、中心高分辨率视觉与注视坐标系动作表示，用单套双目做精细双臂操作，无需腕部相机。",
        why="腕部相机昂贵且受限；主动注视是替代方案。",
        mech="多任务真机比较被动双目与腕部相机。",
        metrics="以原文为准。",
        concl="注视坐标系动作表示让单套主动视觉替代腕部相机成为可能。",
        related=[BIM, MANIP, C + "humanoid-policy-observation-inputs.md"],
        abbr=[
            ("Gaze", "Active Gaze", "主动注视"),
            ("Fovea", "Foveated Vision", "中心高分辨率视觉"),
            ("BC", "Behavior Cloning", "行为克隆"),
        ],
    ),
    dict(
        a="a2",
        g="人形与移动操作",
        slug="mobiagent-dual-loop-self-improvement",
        short="MobiAgent",
        arxiv="2610.03476",
        title="MobiAgent: Dual-Loop Recursive Policy Self-Improvement for Long-Horizon Mobile Manipulation",
        code="https://github.com/kaiknower/MobiAgent",
        project="https://kaiknower.github.io/mobiagent",
        open=PART,
        open_note="仓库可达；公众号称模型下载有访问条件。",
        tags=["paper", "mobile-manipulation", "self-improvement", "skill-library", "long-horizon"],
        one="内循环动态组合技能并视觉反思，外循环整理部署轨迹与更新技能库。",
        why="长程移动操作需要执行反思与离线学习双循环。",
        mech="仿真与实体平台。",
        metrics="以原文为准。",
        concl="自改进系统把部署轨迹变成技能库更新，是闭环数据飞轮的具体做法。",
        related=[LOCOMANIP, C + "data-flywheel.md"],
        abbr=[
            ("MM", "Mobile Manipulation", "移动操作"),
            ("VLM", "Vision-Language Model", "视觉语言模型"),
            ("Skill lib.", "Skill Library", "技能库"),
        ],
    ),
    dict(
        a="a2",
        g="强化学习与数据采集",
        slug="uniintervene-plus-plus-adaptive-intervention",
        short="UniIntervene++",
        arxiv="2610.03620",
        title="UniIntervene++: An Adaptive Intervention Agent for Efficient Real-World Reinforcement Learning",
        code="https://github.com/dannyyudong/An-Adaptive-Intervention-Agent-for-Efficient-Real-World-Reinforcement-Learning",
        project=None,
        open=OPEN,
        open_note="仓库可达。",
        tags=["paper", "real-world-rl", "intervention", "human-in-the-loop", "options"],
        one="把自主策略、轨迹纠正与代码辅助纳入统一选项框架，周期探测自主能力以动态分配辅助控制。",
        why="随策略变强，人工介入比例应自适应下降。",
        mech="五项真机任务；介入率有专门统计口径。",
        metrics="以原文为准。",
        concl="真机 RL 的介入策略本身可学：先探测能力，再分配控制。",
        related=[SAFERL, RL, "./paper-mulligan-performance-guided-data-collection.md"],
        abbr=[
            ("RL", "Reinforcement Learning", "强化学习"),
            ("HiL", "Human-in-the-Loop", "人在回路"),
            ("Option", "Options framework", "分层 RL 选项框架"),
        ],
    ),
    dict(
        a="a2",
        g="策略与后训练",
        slug="vla-augm-data-augmentation-post-training",
        short="VLA 后训练数据增强",
        arxiv="2610.05994",
        title="How (and How Not) to Use Data Augmentation in VLA Post-Training",
        code="https://github.com/bramgrooten/vla-augm",
        project="https://bramgrooten.nl/vla-augm/",
        open=OPEN,
        open_note="项目页链接的 GitHub 仓库可达（基于 RLinf）。",
        tags=["paper", "vla", "post-training", "data-augmentation", "ppo"],
        one="PPO 后训练中只增强价值模块输入更利于视觉泛化，增强策略输入反而破坏训练。",
        why="增强加在哪个模块决定成败。",
        mech="两种 VLA 与仿真基准。",
        metrics="以原文为准。",
        concl="RL 后训练的增强先加 critic 不加 actor。",
        related=[
            VLA,
            M + "ppo.md",
            M + "generative-data-augmentation.md",
            "./paper-dissect-vla-post-training.md",
        ],
        abbr=[
            ("VLA", "Vision-Language-Action", "视觉–语言–动作策略"),
            ("PPO", "Proximal Policy Optimization", "近端策略优化"),
            ("DA", "Data Augmentation", "数据增强"),
        ],
    ),
    dict(
        a="a2",
        g="灵巧操作",
        slug="pears-tactile-failure-reasoning-diffusion-steering",
        short="PEARS",
        arxiv="2610.08784",
        title="PEARS: Physical-Prior-Guided Efficient Adaptation via Failure Reasoning and Diffusion Steering for Tactile Manipulation",
        code=None,
        project="https://song-kun.github.io/pears",
        open=NOLINK,
        open_note="项目页入库日未列 GitHub。",
        tags=["paper", "tactile", "diffusion-policy", "failure-reasoning", "adaptation"],
        one="VLM 失败诊断转为接触力边界，再用触觉条件噪声引导冻结策略在线适配。",
        why="触觉失败信息应参与策略适配而非只报警。",
        mech="仿真和两项真机实验。",
        metrics="以原文为准。",
        concl="失败推理→物理约束→采样引导，是不重训的适配路径。",
        related=[TACT, DP, C + "visuo-tactile-fusion.md"],
        abbr=[
            ("VLM", "Vision-Language Model", "视觉语言模型"),
            ("DP", "Diffusion Policy", "扩散策略"),
            ("Tactile", "Tactile Sensing", "触觉感知"),
        ],
    ),
    dict(
        a="a2",
        g="导航、足式与规划",
        slug="cusp-survival-hazard-alarms-offroad",
        short="CUSP",
        arxiv="2610.07882",
        title="CUSP: CUSUM-Governed Survival Hazard Alarms at the Perception Onset for Off-Road Navigation",
        code=None,
        project="https://in-uk.github.io/cusp-project/",
        open=TBD,
        open_note="项目页含「coming soon」字样，入库日未列 GitHub。",
        tags=["paper", "off-road", "navigation", "safety", "anomaly-detection"],
        one="把人类感知危险时刻与实际接管时刻分开标注，累积生存风险信号；五个未见场地统一误报率评测。",
        why="越野导航要在接管前报警。",
        mech="CUSUM + 生存分析。",
        metrics="以原文为准。",
        concl="报警系统的标签应对齐「感知起点」而非「接管时刻」。",
        related=[SAFE, C + "robot-safety-state-machine.md"],
        abbr=[
            ("CUSUM", "Cumulative Sum", "累积和变化检测"),
            ("Hazard", "Survival Hazard", "生存风险率"),
            ("FPR", "False Positive Rate", "误报率"),
        ],
    ),
    dict(
        a="a2",
        g="VLA 与动作生成",
        slug="stairvla-stage-aware-hierarchical-action",
        short="StairVLA",
        arxiv="2610.07756",
        title="StairVLA: Stage-Aware Hierarchical Action Generation for Vision-Language-Action Models",
        code="https://github.com/Dicomsky/StairVLA",
        project="https://dicomsky.github.io/projects/stairvla",
        open=OPEN,
        open_note="项目页链接的 GitHub 仓库可达（基于 starVLA）。",
        tags=["paper", "vla", "hierarchical", "flow-matching", "latency"],
        one="重型 VLA 做粗规划、轻量模型做高频修正，用部分去噪轨迹衔接两层以摊薄昂贵推理。",
        why="VLA 推理延迟与闭环修正的矛盾需要分层。",
        mech="延迟指标是动作块摊销值。",
        metrics="以原文为准。",
        concl="分层 VLA 的衔接点选在部分去噪轨迹，是值得复用的接口设计。",
        related=[VLA, FM, M + "star-vla.md", C + "control-inference-frequency-decoupling.md"],
        abbr=[
            ("VLA", "Vision-Language-Action", "视觉–语言–动作策略"),
            ("FM", "Flow Matching", "流匹配"),
            ("Hz", "Hertz", "控制频率"),
        ],
    ),
    dict(
        a="a2",
        g="灵巧操作",
        slug="gott-cross-embodiment-contact-primitive",
        short="GOTT",
        arxiv="2610.03861",
        title="GOTT: Object-centric Dexterous Manipulation with a Reusable Cross-Embodiment Primitive",
        code=None,
        project="https://dex-gott.github.io/",
        open=TBD,
        open_note="项目页含「coming soon」字样，入库日未列 GitHub。",
        tags=["paper", "dexterous", "cross-embodiment", "primitive", "contact"],
        one="把操作拆成靠近、建立接触与移动，用共享闭环原语连接物体轨迹和灵巧手；仿真、真机及未见手型实验。",
        why="稳定接触应成为跨手型复用的底层能力。",
        mech="仍需可行的靠近规格。",
        metrics="以原文为准。",
        concl="物体中心 + 接触原语是跨手型灵巧操作的可复用层。",
        related=[DEX, MANIP, C + "contact-rich-manipulation.md"],
        abbr=[
            ("GOTT", "Grasp-Object-Then-Transport", "论文原语缩写（推测）"),
            ("Dex", "Dexterous Hand", "灵巧手"),
            ("Cross-emb.", "Cross-Embodiment", "跨本体"),
        ],
    ),
    dict(
        a="a2",
        g="世界模型与仿真",
        slug="simforcing-simulation-motion-priors",
        short="SimForcing",
        arxiv="2610.06598",
        title="SimForcing: Distilling Simulation Motion Priors into Real-Domain Robot World Models",
        code="https://github.com/Wang-Xiaodong1899/SimForcing",
        project=None,
        open=OPEN,
        open_note="仓库可达。",
        tags=["paper", "world-model", "sim2real", "distillation", "video-prediction"],
        one="对齐潜空间时间变化，把仿真运动先验传给真实域世界模型，并控制仿真条件引导强度。",
        why="真实域视频模型缺运动先验，仿真可补。",
        mech="区分视频预测质量与下游策略收益。",
        metrics="以原文为准。",
        concl="仿真→真实世界模型的蒸馏应只传运动先验，不传外观。",
        related=[GWM, S2R, WAM],
        abbr=[
            ("WM", "World Model", "世界模型"),
            ("Sim2Real", "Simulation to Real", "仿真到真实"),
            ("KD", "Knowledge Distillation", "知识蒸馏"),
        ],
    ),
    dict(
        a="a2",
        g="VLA 与动作生成",
        slug="eapn-execution-aligned-progressive-noise",
        short="EAPN",
        arxiv="2610.06090",
        title="Execution-Aligned Progressive Noise for Consistent Asynchronous Replanning in Generative Robot Policies",
        code=None,
        project="https://embodied.magiclab.top/works/eapn/index.html",
        open=NOLINK,
        open_note="项目页入库日未列 GitHub。",
        tags=[
            "paper",
            "asynchronous-replanning",
            "diffusion-policy",
            "noise-scheduling",
            "consistency",
        ],
        one="按实际执行位移对齐跨动作块噪声并建模块内时间相关性，使异步重规划延续同一条随机轨迹。",
        why="异步重规划若噪声不一致会导致行为跳变。",
        mech="高低延迟实验采用各自训练协议。",
        metrics="以原文为准。",
        concl="噪声对齐是异步重规划连续性的另一条路线，可与 FreeBar 的引导约束对照。",
        related=[
            CHUNK,
            DP,
            "./paper-freebar-barrier-guidance-async-vla.md",
            "./paper-streamingwam.md",
        ],
        abbr=[
            ("EAPN", "Execution-Aligned Progressive Noise", "本文方法"),
            ("DP", "Diffusion Policy", "扩散策略"),
            ("RTC", "Real-Time Chunking", "实时动作块执行"),
        ],
    ),
    dict(
        a="a2",
        g="世界模型与仿真",
        slug="sdob-sampling-based-disturbance-observers",
        short="SDOB",
        arxiv="2610.04896",
        title="Tackling Sim-to-Real Mismatch Through Sampling-Based Disturbance Observers: From Analytical Models to Learned World Models",
        code=None,
        project="https://sampling-based-dob.github.io/",
        open=TBD,
        open_note="项目页列出 GitHub 链接（sampling-based-dob/sampling-based-dob），入库日 ls-remote 不可达，按待发布处理。",
        tags=["paper", "sim2real", "disturbance-observer", "sampling-based-control", "world-model"],
        one="通过状态预测或代价查询估计扰动，使传统观测器适用于仿真器与学习模型；前提含可观测性与缓变扰动。",
        why="采样式控制器需要扰动补偿缩小模型与现实差异。",
        mech="仿真与真机验证。",
        metrics="以原文为准。",
        concl="DOB 思想可推广到黑箱仿真器与世界模型，但依赖扰动缓变假设。",
        related=[S2R, MPC, C + "physics-fidelity-sim2real-gap.md"],
        abbr=[
            ("DOB", "Disturbance Observer", "扰动观测器"),
            ("MPC", "Model Predictive Control", "模型预测控制"),
            ("WM", "World Model", "世界模型"),
        ],
    ),
    dict(
        a="a2",
        g="世界模型与仿真",
        slug="tapdreamer-adversarial-patches-wam",
        short="TAPDreamer",
        arxiv="2610.06814",
        title="TAPDreamer: Transferable Adversarial Patches for World Action Models",
        code=None,
        project="https://tapdreamer.github.io",
        open=TBD,
        open_note="项目页含「coming soon」字样，入库日未列 GitHub。",
        tags=["paper", "adversarial", "wam", "robustness", "security"],
        one="仅利用公开视觉编码器构造固定局部扰动，在所测 WAM 间迁移；闭环仿真揭示编码器风险。",
        why="共享编码器是 WAM 的共同脆弱点。",
        mech="不能直接外推为实体攻击结论。",
        metrics="以原文为准。",
        concl="WAM 鲁棒性评估应包含对共享编码器的迁移攻击。",
        related=[WAM, SAFE],
        abbr=[
            ("WAM", "World Action Model", "世界动作模型"),
            ("TAP", "Transferable Adversarial Patch", "可迁移对抗补丁"),
            ("Enc.", "Visual Encoder", "视觉编码器"),
        ],
    ),
    dict(
        a="a2",
        g="VLA 与动作生成",
        slug="cnp-flow-conditional-source-distribution",
        short="CNP-Flow",
        arxiv="2610.05349",
        title="Learning Conditional Source Distribution via Flow Reversal for Temporal Flow Matching",
        code=None,
        project="https://embodiedai-ntu.github.io/cnpflow/index.html",
        open=NOLINK,
        open_note="项目页入库日未列 GitHub。",
        tags=[
            "paper",
            "flow-matching",
            "source-distribution",
            "video-generation",
            "planning",
            "ntu",
        ],
        one="通过流反演监督条件噪声预测器再微调原流模型；视频生成与七自由度机械臂规划表明改进噪声起点可降采样开销。",
        why="改噪声起点是降低采样成本的另一杠杆。",
        mech="与 Δ-Flow 同属「改起点」路线。",
        metrics="以原文为准。",
        concl="条件起点分布可学，且与 Δ-Flow 的历史起点互补。",
        related=[FM, "./paper-delta-flow-matching-robot-foundation-models.md"],
        abbr=[
            ("CNP", "Conditional Noise Predictor", "条件噪声预测器"),
            ("FM", "Flow Matching", "流匹配"),
            ("7-DoF", "7 Degrees of Freedom", "七自由度机械臂"),
        ],
    ),
    dict(
        a="a2",
        g="世界模型与仿真",
        slug="ttt-rm-residual-memory",
        short="TTT-RM",
        arxiv="2610.04701",
        title="Test-Time Training as Residual Memory for Robot Policies",
        code=None,
        project="https://hatchetproject.github.io/tttrm/",
        open=NOLINK,
        open_note="项目页入库日未列 GitHub。",
        tags=["paper", "memory", "test-time-training", "long-horizon", "policy"],
        one="用历史重建残差监督快速权重，只保存当前上下文解释不了的信息，为有限记忆库补充。",
        why="长任务记忆应互补而非无限历史。",
        mech="记忆任务与真机收纳。",
        metrics="以原文为准。",
        concl="残差式记忆与 IMPRINT 的多流记忆可对照：一个补缺、一个汇流。",
        related=[
            IL,
            "./paper-imprint-multi-stream-latent-memory.md",
            "./paper-robottt-test-time-training-vla-context.md",
        ],
        abbr=[
            ("TTT", "Test-Time Training", "测试时训练"),
            ("RM", "Residual Memory", "残差记忆"),
            ("Fast weights", "Fast Weights", "快速权重"),
        ],
    ),
    dict(
        a="a2",
        g="世界模型与仿真",
        slug="depthworld-3d-world-model",
        short="DepthWorld",
        arxiv="2610.08780",
        title="DepthWorld: 3D World Model for Robot Manipulation",
        code="https://github.com/Jai2500/depthworld",
        project="https://www.jaibardhan.com/depthworld",
        open=OPEN,
        open_note="项目页链接的 GitHub 仓库可达。",
        tags=["paper", "world-model", "depth", "3d", "manipulation"],
        one="先联合校准机器人运动学与多视角外参，再以潜变量空间拼接预测颜色和深度；同预算下优于仅颜色基线。",
        why="几何一致性需要深度参与世界模型预测。",
        mech="以几何一致性为重点。",
        metrics="以原文为准。",
        concl="加深度的世界模型值得在同训练预算下对照仅颜色基线。",
        related=[GWM, WAM, "./paper-meco-wam-4d-geometry-cotraining.md"],
        abbr=[
            ("WM", "World Model", "世界模型"),
            ("RGB-D", "Color + Depth", "彩色与深度"),
            ("Extrinsics", "Camera Extrinsics", "相机外参"),
        ],
    ),
    dict(
        a="a2",
        g="策略与后训练",
        slug="verifine-scaling-verification-self-improvement",
        short="VeriFine",
        arxiv="2610.08761",
        title="VeriFine: Scaling Verification for Self-Improvement in Embodied Reasoning",
        code=None,
        project="https://veri-fine.github.io/",
        open=TBD,
        open_note="项目页含「coming soon」字样，入库日未列 GitHub。",
        tags=["paper", "self-improvement", "verifier", "embodied-reasoning", "curriculum"],
        one="策略、课程与评分器共同演进；停滞时针对高信息量失败征求人类指导并校准评判规则。",
        why="策略变强，验证器不更新会成为瓶颈。",
        mech="具身推理训练反馈。",
        metrics="以原文为准。",
        concl="自改进系统要把验证器纳入演进对象。",
        related=[C + "ai-agent-evaluation.md", RL],
        abbr=[
            ("Verifier", "Verifier model", "评分器"),
            ("Curriculum", "Curriculum Learning", "课程学习"),
            ("RL", "Reinforcement Learning", "强化学习"),
        ],
    ),
    dict(
        a="a2",
        g="世界模型与仿真",
        slug="kineworld-action-induced-transport-fields",
        short="KineWorld",
        arxiv="2610.06349",
        title="KineWorld: Action-Induced Transport Fields for Embodied World Modeling",
        code="https://github.com/modaxiansheng/KineWorld",
        project="https://modaxiansheng.github.io/KineWorld/",
        open=OPEN,
        open_note="项目页链接的 GitHub 仓库可达。",
        tags=["paper", "world-model", "kinematics", "loss-weighting", "bimanual"],
        one="把机器人运动学映射为相机对齐的输运场，对未来视频监督加权，让损失关注动作引起的局部变化。",
        why="均匀外观损失稀释了动作后果。",
        mech="双臂仿真数据评测。",
        metrics="以原文为准。",
        concl="世界模型监督应按「动作影响区域」加权。",
        related=[GWM, WAM, BIM],
        abbr=[
            ("WM", "World Model", "世界模型"),
            ("Transport field", "Action-induced Transport Field", "动作输运场"),
            ("FK", "Forward Kinematics", "正运动学"),
        ],
    ),
    dict(
        a="a2",
        g="世界模型与仿真",
        slug="flow-policies-skill-level-world-models",
        short="技能级世界模型的流策略动作",
        arxiv="2610.04767",
        title="Flow Policies as Actions of Skill-Level World Models: Learned and Symbolic Abstractions for Long-Horizon Planning",
        code=None,
        project="https://andreumatoses.github.io/research/flow-skill-wm",
        open=NOLINK,
        open_note="项目页入库日未列 GitHub。",
        tags=["paper", "world-model", "skill", "planning", "symbolic"],
        one="比较流策略噪声、学习编码与符号标签等技能接口；积木仿真显示长序列性能受搜索限制，符号方案优势部分来自适用性规则。",
        why="让一次完整技能成为世界模型的一步，需要选对接口。",
        mech="仿真。",
        metrics="以原文为准。",
        concl="技能接口的比较要剔除搜索与适用性规则的影响。",
        related=[GWM, M + "hierarchical-reinforcement-learning.md", FM],
        abbr=[
            ("WM", "World Model", "世界模型"),
            ("FM", "Flow Matching", "流匹配"),
            ("TAMP", "Task and Motion Planning", "任务与运动规划"),
        ],
    ),
    dict(
        a="a2",
        g="灵巧操作",
        slug="patterndex-interaction-patterns-bimanual-dexterous",
        short="PatternDex",
        arxiv="2610.04765",
        title="PatternDex: Learning Interaction Patterns to Guide Reinforcement Learning of Bimanual Dexterous Manipulation of Articulated Objects",
        code=None,
        project="https://patterndex.github.io/PatternDex/",
        open=TBD,
        open_note="项目页含「coming soon」字样，入库日未列 GitHub。",
        tags=["paper", "dexterous", "bimanual", "articulated-objects", "reinforcement-learning"],
        one="从人类示范学习交互模式，再预测目标手型的腕部运动和接触点以指导 RL；多手型仿真与开微波炉真机。",
        why="先学人手与物体关系，再适配机器人，可跨手型复用。",
        mech="模式复用。",
        metrics="以原文为准。",
        concl="交互模式是比关节轨迹更可迁移的人类先验。",
        related=[DEX, BIM, RL, "./paper-eigendexplore.md"],
        abbr=[
            ("RL", "Reinforcement Learning", "强化学习"),
            ("Dex", "Dexterous Hand", "灵巧手"),
            ("Articulated", "Articulated Objects", "铰接物体"),
        ],
    ),
    dict(
        a="a2",
        g="多机器人与评测",
        slug="queen-chess-language-model-explanations",
        short="QUEEN",
        arxiv="2610.03695",
        title="Language Models that Play Chess and Explain Their Moves",
        code="https://github.com/queen-project/queen",
        project="https://queen-project.github.io/",
        open=OPEN,
        open_note="仓库可达。",
        tags=["paper", "language-model", "chess", "explainability", "distillation"],
        one="把棋类专家编码器接到语言模型并迭代蒸馏落子解释；国际象棋研究，未验证机器人部署。",
        why="专家能力与语言表达结合的跨领域参考。",
        mech="非机器人实验。",
        metrics="以原文为准。",
        concl="作为「专家 + 解释」范式参考阅读，不作机器人结论引用。",
        related=[C + "deep-rl-game-milestones.md"],
        abbr=[
            ("LM", "Language Model", "语言模型"),
            ("KD", "Knowledge Distillation", "知识蒸馏"),
            ("Elo", "Elo rating", "棋力评分"),
        ],
    ),
]


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
        f"- **线索来源：** [{a['title']}](../blogs/{blog_name(p['a'])})（具身智能小站，{a['date']}）",
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

**{p["short"]}**（*{p["title"]}*，[arXiv:{ax}](https://arxiv.org/abs/{ax}){link_txt}）来自 [具身智能小站 {a["date"]} 盘点](../../sources/blogs/{blog_name(p["a"])})，属「{p["g"]}」组。

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


def _rows(a: str) -> list[tuple[str, str, str, str, str, str]]:
    """(group, short, arxiv, open, wiki_rel_from_sources, wiki_rel_from_wiki_overview)"""
    rows = []
    for p in PAPERS:
        if p["a"] != a:
            continue
        rows.append(
            (
                p["g"],
                p["short"],
                p["arxiv"],
                p["open"],
                f"[paper-{p['slug']}](../../wiki/entities/paper-{p['slug']}.md)",
                f"[paper-{p['slug']}](../entities/paper-{p['slug']}.md)",
            )
        )
    if a == "a2":
        for ax, r in REUSE.items():
            rows.append(
                (
                    r["group"],
                    r["label"],
                    ax,
                    r["open"],
                    f"[{r['wiki']}](../../wiki/entities/{r['wiki']}.md)（**复用**）",
                    f"[{r['wiki']}](../entities/{r['wiki']}.md)（复用）",
                )
            )
    return rows


def _blog(a: str) -> str:
    A = ARTICLES[a]
    rows = _rows(a)
    new = sum(1 for r in rows if "复用" not in r[4])
    body = "\n".join(
        f"| {i:02d} | {r[0]} | {r[1]} | [{r[2]}](https://arxiv.org/abs/{r[2]}) | **{r[3]}** | {r[4]} |"
        for i, r in enumerate(rows, 1)
    )
    return f"""# {A["title"]}

> 来源归档（blog / 微信公众号）

- **标题：** {A["title"]}
- **类型：** blog
- **作者：** 具身智能小站（微信公众号）
- **原始链接：** {A["url"]}
- **发表日期：** {A["date"]}
- **入库日期：** {TODAY}
- **抓取方式：** `scripts/fetch_wechat_article.py`（Chromium + 微信 UA）
- **原始抓取落盘：** [`sources/raw/wechat_embodied_station_{A["key"]}.md`](../raw/wechat_embodied_station_{A["key"]}.md)
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
    related = [f"../entities/paper-{p['slug']}.md" for p in PAPERS if p["a"] == a][:3]
    if a == "a2":
        related += [f"../entities/{r['wiki']}.md" for r in list(REUSE.values())[:2]]
    related += [VLA, WAM]
    flow_nodes = "\n".join(
        f'  G{i}["{g}（{len(rs)}）"] --> GOAL' for i, (g, rs) in enumerate(groups.items())
    )
    other = "a2" if a == "a1" else "a1"
    return f"""---
type: overview
tags: [overview, survey, embodied-ai, wam, vla, technology-map, wechat]
status: complete
updated: {TODAY}
related:
{_yaml_list(related, 2)}
  - ./{ARTICLES[other]["map"]}
sources:
  - ../../sources/blogs/{blog_name(a)}
summary: "{A["map_summary"]}"
---

# {A["map_title"]}

> **本页定位**：为 [具身智能小站 · {A["title"]}]({A["url"]})（{A["date"]}）提供按研究方向组织的阅读坐标；{A["n"]} 篇全部有独立详情节点。

## 一句话观点

**{A["view"]}**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| WAM | World Action Model | 世界–动作联合模型 |
| RL | Reinforcement Learning | 强化学习 |
| FM | Flow Matching | 流匹配生成 |

## 为什么单独做这张地图

- 公众号一次列出 {A["n"]} 篇、方向跨度大，需要横切面索引；每篇只保留一个判断，细节在各节点。
- 节点内的数字为公众号转述，未核验原文表格；开源结论为入库日核查结果，见各节点「为什么重要」。
- 与 [{ARTICLES[other]["map_title"]}](./{ARTICLES[other]["map"]}) 为相邻两期，可对照阅读（异步动作块：StreamingWAM / FreeBar / EAPN；世界模型记忆：ReCAP / IMPRINT / TTT-RM）。

## 流程总览

```mermaid
flowchart TB
{flow_nodes}
  GOAL[可部署具身系统]
```

## 分组索引

{chr(10).join(sections)}
## 关联页面

- [VLA](../methods/vla.md)
- [World Action Models](../concepts/world-action-models.md)
- [Flow Matching 具身策略](../concepts/flow-matching-embodied-policy.md)
- [{ARTICLES[other]["map_title"]}](./{ARTICLES[other]["map"]})

## 参考来源

- [{blog_name(a)}](../../sources/blogs/{blog_name(a)})

## 推荐继续阅读

- [{rows[0][1]}]({rows[0][5].split("](")[1].rstrip(")")})
- [{rows[1][1]}]({rows[1][5].split("](")[1].rstrip(")")})
"""


def _patch_reuse() -> None:
    """复用节点：补 10-13 期博客来源与地图关联（幂等）。"""
    blog = f"  - ../../sources/blogs/{blog_name('a2')}\n"
    mapl = f"  - ../overview/{ARTICLES['a2']['map']}\n"
    for r in REUSE.values():
        path = ROOT / "wiki/entities" / f"{r['wiki']}.md"
        text = path.read_text(encoding="utf-8")
        head, sep, body = text.partition("\n---\n")
        if not sep:
            raise SystemExit(f"frontmatter 缺失: {path}")
        lines = head.split("\n")

        def _append(key: str, item: str) -> None:
            nonlocal lines
            if item.strip() in (l.strip() for l in lines):
                return
            try:
                i = next(i for i, l in enumerate(lines) if l.startswith(f"{key}:"))
            except StopIteration:
                lines.append(f"{key}:")
                lines.append(item.rstrip("\n"))
                return
            j = i + 1
            while j < len(lines) and lines[j].startswith("  "):
                j += 1
            lines.insert(j, item.rstrip("\n"))

        _append("sources", blog)
        _append("related", mapl)
        lines = [f"updated: {TODAY}" if l.startswith("updated:") else l for l in lines]
        path.write_text("\n".join(lines) + sep + body, encoding="utf-8")


def main() -> None:
    raw_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    seen = set()
    for p in PAPERS:
        if p["arxiv"] in seen or p["arxiv"] in REUSE:
            raise SystemExit(f"重复 arXiv: {p['arxiv']}")
        seen.add(p["arxiv"])
        ent = ROOT / "wiki/entities" / f"paper-{p['slug']}.md"
        if ent.exists():
            raise SystemExit(f"节点已存在: {ent}")
        for rel in p["related"]:
            tgt = (ROOT / "wiki/entities" / rel).resolve()
            if not tgt.exists() and not rel.startswith("./paper-"):
                raise SystemExit(f"关联页不存在: {rel} ({p['slug']})")
    for a, A in ARTICLES.items():
        if raw_dir:
            src = raw_dir / f"{a}.txt"
            dst = ROOT / "sources/raw" / f"wechat_embodied_station_{A['key']}.md"
            dst.write_text(
                f"# {A['title']}\n\n> 原始抓取（{TODAY}，`scripts/fetch_wechat_article.py`），仅供溯源；来源：{A['url']}\n\n"
                + src.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
        (ROOT / "sources/blogs" / blog_name(a)).write_text(_blog(a), encoding="utf-8")
        (ROOT / "wiki/overview" / A["map"]).write_text(_map(a), encoding="utf-8")
    for p in PAPERS:
        (ROOT / "sources/papers" / _ax_file(p)).write_text(_paper_source(p), encoding="utf-8")
        (ROOT / "wiki/entities" / f"paper-{p['slug']}.md").write_text(_entity(p), encoding="utf-8")
        if p.get("code"):
            (ROOT / "sources/repos" / _repo_file(p)).write_text(_repo(p), encoding="utf-8")
        if p.get("project"):
            (ROOT / "sources/sites" / _site_file(p)).write_text(_site(p), encoding="utf-8")
    _patch_reuse()
    print(f"Wrote 2 blogs, 2 maps, {len(PAPERS)} new entities, patched {len(REUSE)} reuse nodes")


if __name__ == "__main__":
    main()
