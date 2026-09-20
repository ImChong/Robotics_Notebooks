#!/usr/bin/env python3
"""Bootstrap ingest: 自由度FreeDof Sim2Real 四条路线梳理 — 44 篇参考文献独立节点."""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TODAY = date.today().isoformat()
BLOG = "wechat_freedof_sim2real_four_routes_dr_to_residual_2026-08-23.md"
BLOG_PATH = ROOT / "sources/blogs" / BLOG
CATALOG = ROOT / "sources/papers/freedof_sim2real_44_catalog.md"
MAP_PATH = ROOT / "wiki/overview/freedof-sim2real-44-papers-technology-map.md"
COMP_PATH = ROOT / "wiki/comparisons/sim2real-four-routes-identifiability.md"


def _yaml_list(items: list[str], indent: int = 0) -> str:
    pad = " " * indent
    return "\n".join(f"{pad}- {x}" for x in items)


# ref, slug (new only), reuse entity basename, title, arxiv, venue, section, open, one_liner, why, mechanism, conclusion, tags, abbrev, code, project
PAPERS: list[dict] = [
    {
        "ref": 1,
        "slug": "khosla-robot-dynamics-parameter-identification-1985",
        "reuse": None,
        "title": "Parameter identification of robot dynamics",
        "arxiv": None,
        "venue": "CDC 1985",
        "section": "系统辨识",
        "open": "不适用",
        "one_liner": "经典机器人动力学参数辨识起点，奠定从输入–输出数据反推动力学参数的范式。",
        "why": "理解后续基参数、秩亏与实验设计问题的历史源头。",
        "mechanism": "给定关节轨迹与力矩观测，最小化模型预测与实测的动力学残差以估计参数。",
        "conclusion": "作为 SysID 文献起点阅读即可；现代腿足执行器辨识需结合 PACE 等工程约束。",
        "tags": ["paper", "system-identification", "robot-dynamics", "sim2real"],
        "abbrev": [
            ("SysID", "System Identification", "系统辨识"),
            ("CDC", "Conference on Decision and Control", "控制决策会议"),
            ("DOF", "Degrees of Freedom", "自由度"),
        ],
        "code": None,
        "project": None,
    },
    {
        "ref": 2,
        "slug": "gautier-khalil-inertial-parameter-identification-1988",
        "reuse": None,
        "title": "On the identification of the inertial parameters of robots",
        "arxiv": None,
        "venue": "CDC 1988",
        "section": "系统辨识",
        "open": "不适用",
        "one_liner": "提出机器人惯性参数基参数（base parameters）概念：部分参数只能成组辨识。",
        "why": "FreeDof 文内强调 PD 增益与惯量退化方向，其理论根源在基参数与可辨识性分析。",
        "mechanism": "通过 QR 分解等线性代数工具找出动力学回归矩阵的基，避免对不可辨识参数做无意义估计。",
        "conclusion": "读 PACE 退化方向解析前，先读此文可理解「秩亏不是数值 bug 而是结构问题」。",
        "tags": ["paper", "system-identification", "base-parameters", "sim2real"],
        "abbrev": [
            ("SysID", "System Identification", "系统辨识"),
            ("QR", "QR decomposition", "矩阵分解求基参数"),
            ("DOF", "Degrees of Freedom", "自由度"),
        ],
        "code": None,
        "project": None,
    },
    {
        "ref": 3,
        "slug": None,
        "reuse": "paper-pace-sim2real-legged-robots.md",
        "title": "Towards bridging the gap: Systematic sim-to-real transfer for diverse legged robots (PACE)",
        "arxiv": "2509.06342",
        "venue": "IJRR 2026",
        "section": "系统辨识",
        "open": "已开源",
        "one_liner": "悬空 chirp + CMA-ES 辨识紧凑执行器参数，再以 PMSM 物理能量 reward 盲训腿足 RL 并零样本部署。",
        "why": "文内执行器 SysID 主线代表；无需力矩传感器即可显著缩小执行器 gap。",
        "mechanism": "4n+1 维关节参数 + 全局延迟；固定基座辨识后窄 DR 训练。",
        "conclusion": "高敏捷任务应先把名义执行器模型做准；PACE 给出可复现工程管线。",
        "tags": ["paper", "sim2real", "system-identification", "locomotion"],
        "abbrev": [
            ("PACE", "Precise Adaptation through Continuous Evolution", "本文框架"),
            ("CMA-ES", "Covariance Matrix Adaptation Evolution Strategy", "无梯度参数优化"),
            ("CoT", "Cost of Transport", "运输成本"),
        ],
        "code": "https://github.com/leggedrobotics/pace-sim2real",
        "project": "https://pace.filipbjelonic.com/",
    },
    {
        "ref": 4,
        "slug": None,
        "reuse": "paper-sa-2503-01255-impact-of-static-friction-on-sim2real-in-robotic.md",
        "title": "Impact of static friction on Sim2Real in robotic reinforcement learning",
        "arxiv": "2503.01255",
        "venue": "arXiv 2025",
        "section": "系统辨识",
        "open": "待核实",
        "one_liner": "静摩擦缺失会导致平地可行、上楼梯失败；补上静摩擦感知随机化后超过执行器网络与常规 DR。",
        "why": "文内用来说明「参数集要加对方向」——漏掉静摩擦比多加无关参数更致命。",
        "mechanism": "对比执行器网络、控制论关节建模+SysID、静摩擦感知 DR 三组基线。",
        "conclusion": "SysID/DR 的参数集设计必须覆盖任务相关非光滑效应，而非盲目扩维。",
        "tags": ["paper", "sim2real", "friction", "reinforcement-learning"],
        "abbrev": [
            ("Sim2Real", "Simulation to Real", "仿真到真机"),
            ("RL", "Reinforcement Learning", "强化学习"),
            ("DR", "Domain Randomization", "域随机化"),
        ],
        "code": None,
        "project": None,
    },
    {
        "ref": 5,
        "slug": None,
        "reuse": "paper-notebook-sampling-based-system-identification-with-active.md",
        "title": "Sampling-based system identification with active exploration for legged robot sim2real learning (SPI-Active)",
        "arxiv": "2505.14266",
        "venue": "CoRL 2025",
        "section": "系统辨识",
        "open": "已开源",
        "one_liner": "用 Fisher 信息选高信息量主动激励轨迹，提升含接触场景下的动力学参数可辨识性。",
        "why": "文内「主动激励」代表；把实验设计从人工 chirp 推进到优化器选动作。",
        "mechanism": "采样式 SysID + 主动探索；可含接触但实验成本与摔机风险更高。",
        "conclusion": "在固定基座辨识之上，若需足地接触参数，应评估主动激励是否值得额外真机代价。",
        "tags": ["paper", "system-identification", "sim2real", "locomotion"],
        "abbrev": [
            (
                "SPI-Active",
                "Sampling-based System Identification with Active exploration",
                "本文方法",
            ),
            ("FIM", "Fisher Information Matrix", "信息量矩阵"),
            ("SysID", "System Identification", "系统辨识"),
        ],
        "code": "https://github.com/LeCAR-Lab/SPI-Active",
        "project": None,
    },
    {
        "ref": 6,
        "slug": "gevers-identification-information-matrix-2009",
        "reuse": None,
        "title": "Identification and the information matrix: how to get just sufficiently rich?",
        "arxiv": None,
        "venue": "IEEE TAC 2009",
        "section": "系统辨识",
        "open": "不适用",
        "one_liner": "从信息矩阵与实验设计理论回答：激励要多丰富才足以区分待辨参数。",
        "why": "SPI-Active 与主动激励的理论根源；理解「激励不足」的数学含义。",
        "mechanism": "分析输入信号 richness 与 Fisher 信息矩阵可逆性的关系。",
        "conclusion": "动手做辨识实验前，用此文校准对「足够激励」的预期，避免采集无效数据。",
        "tags": ["paper", "system-identification", "experiment-design", "sim2real"],
        "abbrev": [
            ("FIM", "Fisher Information Matrix", "Fisher 信息矩阵"),
            ("SysID", "System Identification", "系统辨识"),
            ("TAC", "Transactions on Automatic Control", "IEEE 控制汇刊"),
        ],
        "code": None,
        "project": None,
    },
    {
        "ref": 7,
        "slug": "kovalev-differentiable-simulation-locomotion-sysid",
        "reuse": None,
        "title": "Achieving precise and reliable locomotion with differentiable simulation-based system identification",
        "arxiv": "2508.04696",
        "venue": "IROS 2025",
        "section": "系统辨识",
        "open": "待核实",
        "one_liner": "用可微仿真梯度替代纯采样优化，在高维参数空间做精确 locomotion SysID。",
        "why": "文内「可微仿真做辨识」代表，与伴随敏感度分析同数学脉络。",
        "mechanism": "可微物理引擎 + 梯度优化拟合真机轨迹。",
        "conclusion": "当参数维度高且仿真可微时，梯度法可显著降低辨识样本与迭代成本。",
        "tags": ["paper", "system-identification", "differentiable-simulation", "locomotion"],
        "abbrev": [
            ("SysID", "System Identification", "系统辨识"),
            ("Sim2Real", "Simulation to Real", "仿真到真机"),
            ("IROS", "Intelligent Robots and Systems", "IEEE 机器人旗舰会"),
        ],
        "code": None,
        "project": None,
    },
    {
        "ref": 8,
        "slug": None,
        "reuse": "paper-notebook-simulator-adaptation-via-proprioceptive-distribu.md",
        "title": "Simulator adaptation for sim-to-real learning of legged locomotion via proprioceptive distribution matching",
        "arxiv": "2604.11090",
        "venue": "arXiv 2026",
        "section": "系统辨识",
        "open": "待核实",
        "one_liner": "不对齐单点参数，而用本体感受各关节 1D Wasserstein 距离对齐仿真器行为分布。",
        "why": "彻底绕开参数可辨识性的一种行为对齐思路，靠近 DR/适应边界。",
        "mechanism": "分布匹配目标替代参数回归；强调策略在仿真与硬件上的行为方式一致。",
        "conclusion": "当参数不可辨或不想维护参数语义时，行为层对齐是可行替代，但可解释性更弱。",
        "tags": ["paper", "sim2real", "locomotion", "distribution-matching"],
        "abbrev": [
            ("Sim2Real", "Simulation to Real", "仿真到真机"),
            ("Wasserstein", "Wasserstein distance", "分布距离度量"),
            ("Proprio", "Proprioception", "本体感受"),
        ],
        "code": None,
        "project": None,
    },
    {
        "ref": 9,
        "slug": None,
        "reuse": "paper-notebook-domain-randomization-for-transferring-deep-neura.md",
        "title": "Domain randomization for transferring deep neural networks from simulation to the real world",
        "arxiv": "1703.06907",
        "venue": "IROS 2017",
        "section": "域随机化",
        "open": "待核实",
        "one_liner": "通过随机化纹理、光照与相机位姿，使真实图像成为仿真变化的子集，开启视觉 DR 传统。",
        "why": "文内推荐四篇打底之首；理解 DR 从视觉到动力学的扩展脉络。",
        "mechanism": "渲染/外观层随机化 + 深度网络训练；真机零微调迁移。",
        "conclusion": "读 DR 文献的默认起点；动力学 DR 是后续扩展而非另起炉灶。",
        "tags": ["paper", "domain-randomization", "sim2real", "computer-vision"],
        "abbrev": [
            ("DR", "Domain Randomization", "域随机化"),
            ("DNN", "Deep Neural Network", "深度神经网络"),
            ("Sim2Real", "Simulation to Real", "仿真到真机"),
        ],
        "code": None,
        "project": None,
    },
    {
        "ref": 10,
        "slug": "peng-dynamics-randomization-sim2real",
        "reuse": None,
        "title": "Sim-to-real transfer of robotic control with dynamics randomization",
        "arxiv": "1710.06537",
        "venue": "ICRA 2018",
        "section": "域随机化",
        "open": "待核实",
        "one_liner": "将域随机化从视觉扩展到质量、摩擦、时延等动力学参数，训练对参数分布鲁棒的策略。",
        "why": "文内 DR 打底第二篇；腿足与操作 Sim2Real 的共同引用起点。",
        "mechanism": "在仿真中对动力学参数采样，优化期望回报下的策略。",
        "conclusion": "理解 DR 保守性之前，先读此文建立「随机化参数空间」直觉。",
        "tags": ["paper", "domain-randomization", "sim2real", "dynamics-randomization"],
        "abbrev": [
            ("DR", "Domain Randomization", "域随机化"),
            ("Sim2Real", "Simulation to Real", "仿真到真机"),
            ("ICRA", "International Conference on Robotics and Automation", "机器人旗舰会"),
        ],
        "code": None,
        "project": None,
    },
    {
        "ref": 11,
        "slug": "tan-quadruped-agile-locomotion-sim2real",
        "reuse": None,
        "title": "Sim-to-real: learning agile locomotion for quadruped robots",
        "arxiv": "1804.10332",
        "venue": "RSS 2018",
        "section": "域随机化",
        "open": "待核实",
        "one_liner": "腿足 Sim2Real 经典流水线：电机辨识、时延补偿、动力学随机化与推力扰动。",
        "why": "文内第三篇打底；展示 DR 与 SysID 组件如何组合成可部署四足系统。",
        "mechanism": "多阶段校准 + 随机化 + RL 训练；强调工程模块顺序。",
        "conclusion": "DR 不是单点技巧，而是与辨识、时延补偿绑定的系统配方。",
        "tags": ["paper", "sim2real", "quadruped", "domain-randomization"],
        "abbrev": [
            ("Sim2Real", "Simulation to Real", "仿真到真机"),
            ("DR", "Domain Randomization", "域随机化"),
            ("RSS", "Robotics: Science and Systems", "机器人科学系统会议"),
        ],
        "code": None,
        "project": None,
    },
    {
        "ref": 12,
        "slug": None,
        "reuse": "paper-pai-1808-00177-learningdexterousinhandmanipulat.md",
        "title": "Learning dexterous in-hand manipulation",
        "arxiv": "1808.00177",
        "venue": "arXiv 2018",
        "section": "域随机化",
        "open": "待核实",
        "one_liner": "OpenAI 灵巧手内操作：大规模 DR + RL 实现 Rubik 级复杂接触操控的 Sim2Real。",
        "why": "文内举例 DR 在高维接触任务上的成功与代价。",
        "mechanism": "域随机化覆盖动力学与观测不确定性 + 大规模并行仿真训练。",
        "conclusion": "DR 能覆盖复杂接触，但训练与仿真成本极高，不宜作为唯一默认方案。",
        "tags": ["paper", "sim2real", "manipulation", "domain-randomization"],
        "abbrev": [
            ("DR", "Domain Randomization", "域随机化"),
            ("RL", "Reinforcement Learning", "强化学习"),
            ("Sim2Real", "Simulation to Real", "仿真到真机"),
        ],
        "code": None,
        "project": None,
    },
    {
        "ref": 13,
        "slug": None,
        "reuse": "paper-pai-1910-13325-simopt.md",
        "title": "Closing the sim-to-real loop: adapting simulation randomization with real world experience (SimOpt)",
        "arxiv": "1810.05687",
        "venue": "ICRA 2019",
        "section": "域随机化",
        "open": "待核实",
        "one_liner": "用少量真机 rollout 闭环更新 DR 分布：训—测—改范围—再训。",
        "why": "文内「用数据把随机范围收窄」的直接代表。",
        "mechanism": "根据真机轨迹误差反馈调整随机化参数分布。",
        "conclusion": "比手工设 DR 范围更数据驱动，但仍需真机迭代预算。",
        "tags": ["paper", "domain-randomization", "sim2real", "simopt"],
        "abbrev": [
            ("SimOpt", "Simulation Optimization", "仿真随机化闭环优化"),
            ("DR", "Domain Randomization", "域随机化"),
            ("Sim2Real", "Simulation to Real", "仿真到真机"),
        ],
        "code": None,
        "project": None,
    },
    {
        "ref": 14,
        "slug": None,
        "reuse": "paper-pai-1906-01728-bayessim.md",
        "title": "BayesSim: adaptive domain randomization via probabilistic inference for robotics simulators",
        "arxiv": "1906.01728",
        "venue": "RSS 2019",
        "section": "域随机化",
        "open": "待核实",
        "one_liner": "估计参数后验分布并从中采样做 DR，把不可辨识性表达为后验不确定性。",
        "why": "文内强调「多组参数都能解释数据」的贝叶斯读法，重新引入辨识信息。",
        "mechanism": "贝叶斯推断更新仿真参数分布，而非单点参数。",
        "conclusion": "当参数不可唯一辨识时，用分布 DR 比假装有单点真值更诚实。",
        "tags": ["paper", "domain-randomization", "sim2real", "bayesian-inference"],
        "abbrev": [
            ("BayesSim", "Bayesian Simulation adaptation", "本文方法"),
            ("DR", "Domain Randomization", "域随机化"),
            ("RSS", "Robotics: Science and Systems", "机器人科学系统会议"),
        ],
        "code": None,
        "project": None,
    },
    {
        "ref": 15,
        "slug": "muratore-bayesian-optimization-domain-randomization",
        "reuse": None,
        "title": "Data-efficient domain randomization with Bayesian optimization",
        "arxiv": "2003.02471",
        "venue": "arXiv 2020",
        "section": "域随机化",
        "open": "待核实",
        "one_liner": "用贝叶斯优化调 DR 分布，只需成功率等稀疏表现信号，无需逐时刻轨迹对齐。",
        "why": "降低真机数据代价的 DR 调参路线。",
        "mechanism": "BO 搜索随机化超参，以任务表现作为黑盒目标。",
        "conclusion": "真机数据贵时，稀疏奖励 BO 比全轨迹拟合更可行，但样本效率仍有限。",
        "tags": ["paper", "domain-randomization", "sim2real", "bayesian-optimization"],
        "abbrev": [
            ("BO", "Bayesian Optimization", "贝叶斯优化"),
            ("DR", "Domain Randomization", "域随机化"),
            ("Sim2Real", "Simulation to Real", "仿真到真机"),
        ],
        "code": None,
        "project": None,
    },
    {
        "ref": 16,
        "slug": None,
        "reuse": "paper-as-1910-07113-solving-rubik-s-cube-with-a-robot-hand.md",
        "title": "Solving Rubik's cube with a robot hand (ADR)",
        "arxiv": "1910.07113",
        "venue": "arXiv 2019",
        "section": "域随机化",
        "open": "待核实",
        "one_liner": "自动域随机化（ADR）课程：从窄范围起步，达标后外扩随机化边界。",
        "why": "文内提醒 ADR 是训练课程，最终范围不保证等于真实分布。",
        "mechanism": "自适应扩大随机化范围 + 大规模 RL 训练。",
        "conclusion": "ADR 降低手工设范围难度，但不能替代真机验收与分布校准。",
        "tags": ["paper", "domain-randomization", "sim2real", "adr"],
        "abbrev": [
            ("ADR", "Automatic Domain Randomization", "自动域随机化"),
            ("DR", "Domain Randomization", "域随机化"),
            ("RL", "Reinforcement Learning", "强化学习"),
        ],
        "code": None,
        "project": None,
    },
    {
        "ref": 17,
        "slug": "epopt-robust-policies-model-ensembles",
        "reuse": None,
        "title": "EPOpt: learning robust neural network policies using model ensembles",
        "arxiv": "1610.01283",
        "venue": "ICLR 2017",
        "section": "域随机化",
        "open": "待核实",
        "one_liner": "用模型集成中回报最差样本更新策略，把优化重点从期望移向尾部。",
        "why": "文内鲁棒/对抗训练分支；区别于期望 DR 的尾部风险处理。",
        "mechanism": "ensemble of simulators + worst-case or CVaR-style policy update。",
        "conclusion": "需要尾部鲁棒时考虑 EPOpt 类方法，但可能更保守。",
        "tags": ["paper", "robust-rl", "sim2real", "domain-randomization"],
        "abbrev": [
            ("EPOpt", "Epistemic Policy Optimization", "集成模型鲁棒策略优化"),
            ("CVaR", "Conditional Value at Risk", "条件风险价值"),
            ("RL", "Reinforcement Learning", "强化学习"),
        ],
        "code": None,
        "project": None,
    },
    {
        "ref": 18,
        "slug": "rarl-robust-adversarial-rl",
        "reuse": None,
        "title": "Robust adversarial reinforcement learning (RARL)",
        "arxiv": "1703.02702",
        "venue": "ICML 2017",
        "section": "域随机化",
        "open": "待核实",
        "one_liner": "训练对手网络施加扰动力，与策略对抗以提升鲁棒性。",
        "why": "文内对抗训练代表；扰动若不符合物理会导致过度保守。",
        "mechanism": "min-max 博弈：策略 vs 扰动生成器。",
        "conclusion": "对抗训练可补 DR 尾部，但扰动物理合理性必须约束。",
        "tags": ["paper", "robust-rl", "sim2real", "adversarial-training"],
        "abbrev": [
            ("RARL", "Robust Adversarial Reinforcement Learning", "本文方法"),
            ("RL", "Reinforcement Learning", "强化学习"),
            ("DR", "Domain Randomization", "域随机化"),
        ],
        "code": None,
        "project": None,
    },
    {
        "ref": 19,
        "slug": "polysim-multi-simulator-humanoid-sim2real",
        "reuse": None,
        "title": "PolySim: bridging the sim-to-real gap for humanoid control via multi-simulator dynamics randomization",
        "arxiv": "2510.01708",
        "venue": "arXiv 2025",
        "section": "域随机化",
        "open": "待核实",
        "one_liner": "并行 IsaacSim/IsaacGym/Genesis 等多引擎训练，把随机化从参数层扩到动力学结构层。",
        "why": "文内 52.8% 跟踪成功率提升与真机人形零样本部署案例。",
        "mechanism": "多异构仿真器并行 rollout + 结构层 DR。",
        "conclusion": "当 gap 来自引擎近似而非仅参数时，多引擎训练比单引擎宽 DR 更对症。",
        "tags": ["paper", "sim2real", "humanoid", "domain-randomization", "polysim"],
        "abbrev": [
            ("PolySim", "Poly Simulator training", "多仿真器训练"),
            ("DR", "Domain Randomization", "域随机化"),
            ("Sim2Real", "Simulation to Real", "仿真到真机"),
        ],
        "code": None,
        "project": None,
    },
    {
        "ref": 20,
        "slug": "erez-simulation-tools-comparison-icra-2015",
        "reuse": None,
        "title": "Simulation tools for model-based robotics: comparison of Bullet, Havok, MuJoCo, ODE and PhysX",
        "arxiv": None,
        "venue": "ICRA 2015",
        "section": "域随机化",
        "open": "不适用",
        "one_liner": "横比主流物理引擎的速度–精度权衡，说明引擎选择本身影响 Sim2Real。",
        "why": "支撑文内「gap 出在引擎本身」论点。",
        "mechanism": "统一任务下 benchmark 多引擎接触与积分行为差异。",
        "conclusion": "换引擎有时比调参更有效；PolySim 类多引擎训练有明确动机。",
        "tags": ["paper", "simulation", "physics-engine", "sim2real"],
        "abbrev": [
            ("ODE", "Open Dynamics Engine", "开源动力学引擎"),
            ("PhysX", "NVIDIA PhysX", "商业物理引擎"),
            ("Sim2Real", "Simulation to Real", "仿真到真机"),
        ],
        "code": None,
        "project": None,
    },
    {
        "ref": 21,
        "slug": "acosta-validating-simulators-real-world-impacts",
        "reuse": None,
        "title": "Validating robotics simulators on real-world impacts",
        "arxiv": "2110.00541",
        "venue": "RA-L 2022",
        "section": "域随机化",
        "open": "待核实",
        "one_liner": "用方块抛落与 Cassie 跳跃落地真机冲击数据检验 Drake/MuJoCo/Bullet。",
        "why": "文内说明接触刚度可辨识性随任务变化——方块 vs 人形落地敏感性不同。",
        "mechanism": "对比三引擎在冲击阶段的轨迹与接触力复现。",
        "conclusion": "接触参数是否可辨取决于实验条件；不能脱离任务谈参数敏感性。",
        "tags": ["paper", "simulation", "contact-model", "sim2real"],
        "abbrev": [
            ("RA-L", "Robotics and Automation Letters", "IEEE 机器人快报"),
            ("Sim2Real", "Simulation to Real", "仿真到真机"),
            ("MuJoCo", "Multi-Joint dynamics with Contact", "接触动力学仿真器"),
        ],
        "code": None,
        "project": None,
    },
    {
        "ref": 22,
        "slug": "le-lidec-contact-models-comparative-analysis",
        "reuse": None,
        "title": "Contact models in robotics: a comparative analysis",
        "arxiv": "2304.06372",
        "venue": "IEEE TRO 2024",
        "section": "域随机化",
        "open": "待核实",
        "one_liner": "从 Signorini、库仑摩擦与最大耗散原理比较各引擎接触近似及其物理松弛。",
        "why": "PolySim 的理论骨架；解释 reality gap 的结构来源。",
        "mechanism": "统一数学框架对比软约束、互补约束等接触实现。",
        "conclusion": "引擎接触近似不是细节，而是 Sim2Real gap 的一级来源。",
        "tags": ["paper", "contact-model", "simulation", "sim2real"],
        "abbrev": [
            ("TRO", "Transactions on Robotics", "IEEE 机器人汇刊"),
            ("Sim2Real", "Simulation to Real", "仿真到真机"),
            ("LCP", "Linear Complementarity Problem", "线性互补接触公式"),
        ],
        "code": None,
        "project": None,
    },
    {
        "ref": 23,
        "slug": "up-osi-universal-policy-online-sysid",
        "reuse": None,
        "title": "Preparing for the unknown: learning a universal policy with online system identification (UP-OSI)",
        "arxiv": "1702.02453",
        "venue": "RSS 2017",
        "section": "在线适应",
        "open": "待核实",
        "one_liner": "通用策略 + 在线辨识器：从历史估计动力学参数并调节动作，是在线适应早期形态。",
        "why": "文内与 RMA 并列的「推迟辨识」代表。",
        "mechanism": "辨识器输出 extrinsics/参数 → 条件策略；部署期在线运行。",
        "conclusion": "在线 SysID 与 DR 组合是成熟路线，但激励不足会静默失效。",
        "tags": ["paper", "online-adaptation", "system-identification", "sim2real"],
        "abbrev": [
            ("UP-OSI", "Universal Policy with Online System Identification", "本文方法"),
            ("SysID", "System Identification", "系统辨识"),
            ("RSS", "Robotics: Science and Systems", "机器人科学系统会议"),
        ],
        "code": None,
        "project": None,
    },
    {
        "ref": 24,
        "slug": None,
        "reuse": "paper-rma-rapid-motor-adaptation.md",
        "title": "RMA: rapid motor adaptation for legged robots",
        "arxiv": "2107.04034",
        "venue": "RSS 2021",
        "section": "在线适应",
        "open": "已开源",
        "one_liner": "特权 extrinsics 训练 base policy，历史编码器在线估计环境上下文并快速适应。",
        "why": "文内在线适应最常被引用的现代代表之一。",
        "mechanism": "Teacher-student 式特权训练 + 部署期 adaptation module。",
        "conclusion": "放松参数可辨识性要求，但不取消对历史信息量的需求。",
        "tags": ["paper", "online-adaptation", "sim2real", "locomotion"],
        "abbrev": [
            ("RMA", "Rapid Motor Adaptation", "本文方法"),
            ("Sim2Real", "Simulation to Real", "仿真到真机"),
            ("RL", "Reinforcement Learning", "强化学习"),
        ],
        "code": "https://github.com/antonilo/rl_locomotion",
        "project": "https://rma-legged-robots.github.io/",
    },
    {
        "ref": 25,
        "slug": None,
        "reuse": "paper-rapid-locomotion-rl.md",
        "title": "Rapid locomotion via reinforcement learning",
        "arxiv": "2205.02824",
        "venue": "RSS 2022",
        "section": "在线适应",
        "open": "待核实",
        "one_liner": "大规模训练 + 速度课程把四足速度上限推高，含适应与 sim2real 组件。",
        "why": "文内在线适应规模扩展代表。",
        "mechanism": "并行仿真 + 课程 + 适应模块。",
        "conclusion": "适应效果与训练规模、课程设计强相关。",
        "tags": ["paper", "locomotion", "sim2real", "online-adaptation"],
        "abbrev": [
            ("RL", "Reinforcement Learning", "强化学习"),
            ("Sim2Real", "Simulation to Real", "仿真到真机"),
            ("RSS", "Robotics: Science and Systems", "机器人科学系统会议"),
        ],
        "code": None,
        "project": None,
    },
    {
        "ref": 26,
        "slug": None,
        "reuse": "paper-digit-humanoid-locomotion-rl.md",
        "title": "Real-world humanoid locomotion with reinforcement learning",
        "arxiv": "2303.03381",
        "venue": "Science Robotics 2024",
        "section": "在线适应",
        "open": "待核实",
        "one_liner": "Transformer 吃长历史隐式完成在线辨识，无需单独辨识器模块。",
        "why": "文内架构演进：从显式辨识器到序列模型隐式适应。",
        "mechanism": "长上下文策略网络 + 大规模 RL 训练 + 真机部署。",
        "conclusion": "隐式适应简化架构，但失败诊断更难。",
        "tags": ["paper", "humanoid", "locomotion", "sim2real", "transformer"],
        "abbrev": [
            ("RL", "Reinforcement Learning", "强化学习"),
            ("Sim2Real", "Simulation to Real", "仿真到真机"),
            ("WBC", "Whole-Body Control", "全身控制"),
        ],
        "code": None,
        "project": None,
    },
    {
        "ref": 27,
        "slug": None,
        "reuse": "paper-notebook-learning-quadrupedal-locomotion-over-challenging.md",
        "title": "Learning quadrupedal locomotion over challenging terrain (Teacher-Student)",
        "arxiv": "2010.11251",
        "venue": "Science Robotics 2020",
        "section": "在线适应",
        "open": "待核实",
        "one_liner": "特权教师–学生：教师读地形/摩擦等特权信息，学生仅用本体历史模仿。",
        "why": "文内 privileged teacher-student 结构的标准引用。",
        "mechanism": "仿真特权训练 + 部署期仅学生前向。",
        "conclusion": "解决信息不可得，不自动解决动力学模型不准。",
        "tags": ["paper", "privileged-learning", "locomotion", "sim2real"],
        "abbrev": [
            ("RL", "Reinforcement Learning", "强化学习"),
            ("Sim2Real", "Simulation to Real", "仿真到真机"),
            ("Teacher-Student", "Privileged Teacher-Student", "特权教师学生训练"),
        ],
        "code": None,
        "project": None,
    },
    {
        "ref": 28,
        "slug": None,
        "reuse": "paper-anymal-walk-minutes-parallel-drl.md",
        "title": "Learning to walk in minutes using massively parallel deep reinforcement learning",
        "arxiv": "2109.11978",
        "venue": "CoRL 2022",
        "section": "在线适应",
        "open": "已开源",
        "one_liner": "大规模并行 DRL 缩短训练时间，ANYmal 上快速获得可部署行走策略。",
        "why": "文内 privileged learning 与大规模仿真结合的代表。",
        "mechanism": "数千并行环境 + PPO/类似 on-policy 算法。",
        "conclusion": "训练算力可换部分辨识工作，但不替代真机验收。",
        "tags": ["paper", "locomotion", "sim2real", "parallel-rl"],
        "abbrev": [
            ("DRL", "Deep Reinforcement Learning", "深度强化学习"),
            ("Sim2Real", "Simulation to Real", "仿真到真机"),
            ("ANYmal", "ANYmal quadruped", "苏黎世理工四足平台"),
        ],
        "code": "https://github.com/leggedrobotics/legged_gym",
        "project": None,
    },
    {
        "ref": 29,
        "slug": None,
        "reuse": "paper-notebook-learning-agile-and-dynamic-motor-skills-for-legg.md",
        "title": "Learning agile and dynamic motor skills for legged robots (Actuator Network)",
        "arxiv": "1901.08652",
        "venue": "Science Robotics 2019",
        "section": "残差学习",
        "open": "待核实",
        "one_liner": "用真机试验数据训练执行器网络，从关节状态历史预测输出转矩并嵌入仿真训练。",
        "why": "文内力矩层残差/执行器网络路线的奠基工作（Hwangbo 2019）。",
        "mechanism": "监督学习执行器模型 → 仿真中替换或增强力矩通道 → RL 训练 → 真机部署不运行网络。",
        "conclusion": "当 URDF 执行器模型过于简化时，执行器网络是力矩层残差的主流起点。",
        "tags": ["paper", "actuator-network", "sim2real", "locomotion"],
        "abbrev": [
            ("Sim2Real", "Simulation to Real", "仿真到真机"),
            ("RL", "Reinforcement Learning", "强化学习"),
            ("PD", "Proportional–Derivative", "比例微分底层控制"),
        ],
        "code": None,
        "project": None,
    },
    {
        "ref": 30,
        "slug": "golemo-neural-augmented-robot-simulation",
        "reuse": None,
        "title": "Sim-to-real transfer with neural-augmented robot simulation",
        "arxiv": None,
        "venue": "CoRL 2018",
        "section": "残差学习",
        "open": "待核实",
        "one_liner": "保留解析物理模型，用 RNN 学习残差修正不可建模的历史相关误差（回差、延迟等）。",
        "why": "文内「替换 vs 叠加」分歧中的叠加/灰盒代表。",
        "mechanism": "物理仿真 + 神经网络残差项；循环结构表达时序误差。",
        "conclusion": "灰盒组合保留可解释性，适合主误差可物理解释、剩余结构复杂的平台。",
        "tags": ["paper", "residual-learning", "sim2real", "neural-augmented-simulation"],
        "abbrev": [
            ("RNN", "Recurrent Neural Network", "循环神经网络"),
            ("Sim2Real", "Simulation to Real", "仿真到真机"),
            ("CoRL", "Conference on Robot Learning", "机器人学习会议"),
        ],
        "code": None,
        "project": None,
    },
    {
        "ref": 31,
        "slug": None,
        "reuse": "paper-notebook-bridging-the-sim-to-real-gap-for-athletic-loco-m.md",
        "title": "Bridging the sim-to-real gap for athletic loco-manipulation (UAN)",
        "arxiv": "2502.10894",
        "venue": "arXiv 2025",
        "section": "残差学习",
        "open": "待核实",
        "one_liner": "把执行器建模当作 RL 问题：学修正力矩使仿真轨迹靠近真机，避开不可靠电流–力矩标签。",
        "why": "文内辨识↔残差边界代表；论证电流不能当关节力矩真值。",
        "mechanism": "轨迹匹配奖励训练执行器残差；抑制 reward hacking。",
        "conclusion": "观测口径受限时，轨迹级残差比回归电流标签更可靠。",
        "tags": ["paper", "residual-learning", "sim2real", "loco-manipulation"],
        "abbrev": [
            ("UAN", "Unified Actuator Network / athletic loco-manip work", "文内执行器残差线"),
            ("Sim2Real", "Simulation to Real", "仿真到真机"),
            ("RL", "Reinforcement Learning", "强化学习"),
        ],
        "code": None,
        "project": None,
    },
    {
        "ref": 32,
        "slug": None,
        "reuse": "paper-residual-rl-robot-control.md",
        "title": "Residual reinforcement learning for robot control",
        "arxiv": "1812.03201",
        "venue": "ICRA 2019",
        "section": "残差学习",
        "open": "待核实",
        "one_liner": "在基础控制器上叠加 RL 学出的残差动作，起源于操作领域。",
        "why": "文内动作层残差的理论起点。",
        "mechanism": "a = a_base + Δa_RL；基础控制器提供先验。",
        "conclusion": "动作层残差工程上更易落地，因不必改仿真器内部动力学。",
        "tags": ["paper", "residual-learning", "sim2real", "manipulation"],
        "abbrev": [
            ("RL", "Reinforcement Learning", "强化学习"),
            ("Sim2Real", "Simulation to Real", "仿真到真机"),
            ("ICRA", "International Conference on Robotics and Automation", "机器人旗舰会"),
        ],
        "code": None,
        "project": None,
    },
    {
        "ref": 33,
        "slug": None,
        "reuse": "paper-hrl-stack-25-asap.md",
        "title": "ASAP: aligning simulation and real-world physics for learning agile humanoid whole-body skills",
        "arxiv": "2502.01143",
        "venue": "RSS 2025",
        "section": "残差学习",
        "open": "待核实",
        "one_liner": "真机 rollout 学 delta action，冻结后嵌入仿真微调策略，部署时去掉修正模型。",
        "why": "文内 52.7% 跟踪误差降低案例；动作层残差完整流程代表。",
        "mechanism": "Sim 预训练 → 真机配对轨迹 → delta action 模型 → 仿真对齐微调 → 真机无 delta 部署。",
        "conclusion": "辨识做到头后，动作层残差常是敏捷人形技能的下一档；基线需充分调优再解读百分比。",
        "tags": ["paper", "humanoid", "residual-learning", "sim2real"],
        "abbrev": [
            ("ASAP", "Aligning Simulation And real-world Physics", "本文方法"),
            ("Sim2Real", "Simulation to Real", "仿真到真机"),
            ("WBC", "Whole-Body Control", "全身控制"),
        ],
        "code": None,
        "project": "https://agile.human2humanoid.com/",
    },
    {
        "ref": 34,
        "slug": None,
        "reuse": "paper-loco-manip-161-014-mosaic.md",
        "title": "MOSAIC: bridging the sim-to-real gap in generalist humanoid motion tracking and teleoperation with rapid residual adaptation",
        "arxiv": "2602.08594",
        "venue": "arXiv 2026",
        "section": "残差学习",
        "open": "待核实",
        "one_liner": "针对遥操作接口延迟/噪声/重定向偏差，用少量接口数据训策略再加性残差模块蒸馏。",
        "why": "文内 loco-manipulation 与接口适配残差；强调离线指标与硬件表现可脱钩。",
        "mechanism": "接口专属预训练 + 残差蒸馏进通用 tracker。",
        "conclusion": "接口 gap 与动力学 gap 需分开处理；仿真 SOTA 分数不等于真机可用。",
        "tags": ["paper", "humanoid", "teleoperation", "residual-learning", "sim2real"],
        "abbrev": [
            ("MOSAIC", "Motion tracking with rapid residual adaptation", "本文框架"),
            ("Sim2Real", "Simulation to Real", "仿真到真机"),
            ("WBC", "Whole-Body Control", "全身控制"),
        ],
        "code": None,
        "project": None,
    },
    {
        "ref": 35,
        "slug": "eysenbach-off-dynamics-rl",
        "reuse": None,
        "title": "Off-dynamics reinforcement learning: training for transfer with domain classifiers",
        "arxiv": "2006.13916",
        "venue": "ICLR 2021",
        "section": "残差学习",
        "open": "待核实",
        "one_liner": "训练分类器判断仿真轨迹在真机上的可信度，无需显式目标动力学模型即可面向迁移训练。",
        "why": "文内真机微调/离策略迁移代表之一。",
        "mechanism": "domain classifier 作为辅助信号 shaping 策略学习。",
        "conclusion": "当显式动力学模型难建时，判别式迁移信号是可行替代。",
        "tags": ["paper", "sim2real", "off-dynamics", "reinforcement-learning"],
        "abbrev": [
            ("RL", "Reinforcement Learning", "强化学习"),
            ("Sim2Real", "Simulation to Real", "仿真到真机"),
            ("OOD", "Out-of-Distribution", "分布外"),
        ],
        "code": None,
        "project": None,
    },
    {
        "ref": 36,
        "slug": "smith-legged-robots-keep-learning",
        "reuse": None,
        "title": "Legged robots that keep on learning: fine-tuning locomotion policies in the real world",
        "arxiv": "2110.05457",
        "venue": "ICRA 2022",
        "section": "残差学习",
        "open": "待核实",
        "one_liner": "真机持续微调 locomotion 策略，机器人可在真实世界中自行恢复与学习。",
        "why": "文内硬件在线微调代表；高动态平台仍受摔机与重置成本限制。",
        "mechanism": "仿真预训练 + 安全约束下的真机 policy fine-tuning。",
        "conclusion": "真机 RL 微调可行但昂贵；常见折中是少量安全校准。",
        "tags": ["paper", "sim2real", "real-world-rl", "locomotion"],
        "abbrev": [
            ("RL", "Reinforcement Learning", "强化学习"),
            ("Sim2Real", "Simulation to Real", "仿真到真机"),
            ("ICRA", "International Conference on Robotics and Automation", "机器人旗舰会"),
        ],
        "code": None,
        "project": None,
    },
    {
        "ref": 37,
        "slug": "rapt-sim2real-ood-detection",
        "reuse": None,
        "title": "RAPT: model-predictive out-of-distribution detection and failure diagnosis for sim-to-real humanoid deployment",
        "arxiv": "2602.01515",
        "venue": "arXiv 2026",
        "section": "监控与评测",
        "open": "待核实",
        "one_liner": "仿真学习标称执行流形，部署时用预测偏差做 OOD 检测与 sim2real 失配诊断。",
        "why": "文内部署期监控完整方案；89% TPR / 87.5% Top-1 诊断等数字出处。",
        "mechanism": "模型预测 + 失配度量 +（可选）LLM 根因推理。",
        "conclusion": "策略无自我评判机制，监控应作为部署标配而非附加项。",
        "tags": ["paper", "humanoid", "deployment", "ood-detection", "sim2real"],
        "abbrev": [
            ("RAPT", "Robust Adaptive Prediction for Transfer", "文内部署监控框架"),
            ("OOD", "Out-of-Distribution", "分布外检测"),
            ("Sim2Real", "Simulation to Real", "仿真到真机"),
        ],
        "code": None,
        "project": None,
    },
    {
        "ref": 38,
        "slug": "kadian-sim2real-predictivity",
        "reuse": None,
        "title": "Sim2Real predictivity: does evaluation in simulation predict real-world performance?",
        "arxiv": None,
        "venue": "RA-L 2020",
        "section": "监控与评测",
        "open": "待核实",
        "one_liner": "系统研究仿真评测指标能否预测真机表现，为 benchmark 设计提供依据。",
        "why": "文内评测侧代表；呼应「离线指标与硬件表现脱钩」。",
        "mechanism": "跨 sim/real 任务对比相关性；提出 predictivity 概念。",
        "conclusion": "仿真 leaderboard 高不等于部署可用；需读 predictivity 文献校准期望。",
        "tags": ["paper", "sim2real", "evaluation", "benchmark"],
        "abbrev": [
            ("Sim2Real", "Simulation to Real", "仿真到真机"),
            ("RA-L", "Robotics and Automation Letters", "IEEE 机器人快报"),
            ("Benchmark", "Benchmark", "标准化评测套件"),
        ],
        "code": None,
        "project": None,
    },
    {
        "ref": 39,
        "slug": "splatsim-gaussian-splatting-sim2real",
        "reuse": None,
        "title": "SplatSim: zero-shot sim2real transfer of RGB manipulation policies using Gaussian splatting",
        "arxiv": "2409.10161",
        "venue": "ICRA 2025",
        "section": "视觉 Sim2Real",
        "open": "待核实",
        "one_liner": "用 3D Gaussian Splatting 从真实场景重建可渲染仿真，实现 RGB 操控策略零样本迁移。",
        "why": "文内视觉 gap 旁支代表，与动力学 Sim2Real 正交。",
        "mechanism": "Real-to-sim 场景重建 + 策略在重建渲染中训练。",
        "conclusion": "观测层 gap 应走视觉/渲染路线，勿用动力学 SysID 硬修。",
        "tags": ["paper", "sim2real", "gaussian-splatting", "manipulation"],
        "abbrev": [
            ("3DGS", "3D Gaussian Splatting", "三维高斯溅射"),
            ("Sim2Real", "Simulation to Real", "仿真到真机"),
            ("RGB", "Red Green Blue", "视觉像素策略"),
        ],
        "code": None,
        "project": None,
    },
    {
        "ref": 40,
        "slug": None,
        "reuse": "paper-notebook-gaussgym-an-open-source-real-to-sim-framework-fo.md",
        "title": "GaussGym: an open-source real-to-sim framework for learning locomotion from pixels",
        "arxiv": "2510.15352",
        "venue": "arXiv 2025",
        "section": "视觉 Sim2Real",
        "open": "待核实",
        "one_liner": "3DGS 作为向量化仿真器 drop-in 渲染器，大规模并行视觉 locomotion 训练并零样本迁移四足。",
        "why": "文内视觉 locomotion 代表；强调渲染吞吐与 sim2real。",
        "mechanism": "Real-to-sim 3DGS 场景 + GPU 向量化 RL 环境。",
        "conclusion": "视觉 Sim2Real 需要渲染器工程，与动力学 gap 分工明确。",
        "tags": ["paper", "sim2real", "gaussian-splatting", "locomotion"],
        "abbrev": [
            ("GaussGym", "Gaussian Splatting Gym", "本文仿真框架"),
            ("3DGS", "3D Gaussian Splatting", "三维高斯溅射"),
            ("Sim2Real", "Simulation to Real", "仿真到真机"),
        ],
        "code": None,
        "project": None,
    },
    {
        "ref": 41,
        "slug": "schwarke-differentiable-simulation-locomotion-corl",
        "reuse": None,
        "title": "Learning deployable locomotion control via differentiable simulation",
        "arxiv": "2404.02887",
        "venue": "CoRL 2025",
        "section": "可微仿真",
        "open": "待核实",
        "one_liner": "可微接触模型突破腿足部署瓶颈，首个完全在可微仿真中训练并零样本上真机的腿足 locomotion。",
        "why": "文内可微仿真横切工具代表；腿足此前卡在接触梯度。",
        "mechanism": "兼顾梯度信息量与物理保真的接触模型 + 端到端策略优化。",
        "conclusion": "可微仿真可用于辨识或直接训策略，但非光滑接触梯度仍需谨慎。",
        "tags": ["paper", "differentiable-simulation", "locomotion", "sim2real"],
        "abbrev": [
            ("Sim2Real", "Simulation to Real", "仿真到真机"),
            ("CoRL", "Conference on Robot Learning", "机器人学习会议"),
            ("Contact", "Contact model", "接触动力学模型"),
        ],
        "code": None,
        "project": None,
    },
    {
        "ref": 42,
        "slug": None,
        "reuse": "paper-notebook-learning-sim-to-real-humanoid-locomotion-in-15-m.md",
        "title": "Learning sim-to-real humanoid locomotion in 15 minutes",
        "arxiv": "2512.01996",
        "venue": "arXiv 2025",
        "section": "训练成本",
        "open": "待核实",
        "one_liner": "单卡 RTX 4090 约 15 分钟训出可 sim2real 的人形 locomotion，改变辨识 vs DR 的性价比计算。",
        "why": "文内「训练成本塌缩」代表；低敏捷任务可能多跑 DR 而非做 SysID。",
        "mechanism": "off-policy 算法 + 数千并行环境 + 极简 reward。",
        "conclusion": "训练便宜不等于验收便宜；高敏捷仍倾向先标定执行器。",
        "tags": ["paper", "humanoid", "sim2real", "locomotion", "fast-training"],
        "abbrev": [
            ("SAC", "Soft Actor-Critic", "离线策略 RL 算法"),
            ("Sim2Real", "Simulation to Real", "仿真到真机"),
            ("TD3", "Twin Delayed DDPG", "离线策略 RL 算法"),
        ],
        "code": None,
        "project": None,
    },
    {
        "ref": 43,
        "slug": None,
        "reuse": "paper-survey-sim2real-rl-foundation-models.md",
        "title": "A survey of sim-to-real methods in RL: progress, prospects and challenges with foundation models",
        "arxiv": "2502.13187",
        "venue": "arXiv 2025",
        "section": "综述",
        "open": "已开源",
        "one_liner": "按 MDP 四要素 taxonomy 梳理 Sim2Real RL，维护 AwesomeSim2Real 仓库。",
        "why": "文内推荐最全面综述，与四条路线对比页互补。",
        "mechanism": "四要素分类 + 开源 benchmark 清单 + FM 展望。",
        "conclusion": "选型时同时读「MDP 哪一环 gap」与「辨识立场」两套坐标。",
        "tags": ["paper", "survey", "sim2real", "reinforcement-learning"],
        "abbrev": [
            ("Sim2Real", "Simulation to Real", "仿真到真机"),
            ("MDP", "Markov Decision Process", "马尔可夫决策过程"),
            ("FM", "Foundation Model", "基础模型"),
        ],
        "code": "https://github.com/LongchaoDa/AwesomeSim2Real",
        "project": None,
    },
    {
        "ref": 44,
        "slug": "awesome-humanoid-robot-learning",
        "reuse": None,
        "title": "Awesome Humanoid Robot Learning",
        "arxiv": None,
        "venue": "GitHub 策展仓库",
        "section": "资源",
        "open": "已开源",
        "one_liner": "Yanjie Ze 维护的人形机器人学习论文与代码精选列表，Sim2Real 与人形 loco-manip 更新频繁。",
        "why": "文内人形方向持续更新资源索引。",
        "mechanism": "社区策展 README + 分类链接。",
        "conclusion": "作扩展阅读索引，与站内 awesome 实体互参而不重复造 survey 页。",
        "tags": ["resource", "awesome-list", "humanoid", "sim2real"],
        "abbrev": [
            ("Sim2Real", "Simulation to Real", "仿真到真机"),
            ("WBC", "Whole-Body Control", "全身控制"),
            ("VLA", "Vision-Language-Action", "视觉–语言–动作模型"),
        ],
        "code": "https://github.com/YanjieZe/awesome-humanoid-robot-learning",
        "project": None,
    },
]


def entity_basename(p: dict) -> str:
    if p.get("reuse"):
        return p["reuse"]
    assert p["slug"]
    return f"paper-{p['slug']}.md"


def entity_rel(p: dict) -> str:
    return f"../entities/{entity_basename(p)}"


def _paper_source(p: dict) -> str:
    ax = p.get("arxiv")
    ax_line = f"- **arXiv：** <https://arxiv.org/abs/{ax}>\n" if ax else ""
    code_line = f"- **代码：** <{p['code']}>\n" if p.get("code") else ""
    proj_line = f"- **项目页：** <{p['project']}>\n" if p.get("project") else ""
    return f"""# {p["title"]}

> 来源归档（paper / 自由度FreeDof Sim2Real 44 篇参考文献 [{p["ref"]:02d}/44]）

- **标题：** {p["title"]}
- **类型：** paper
- **出处：** {p["venue"]}
- **章节：** {p["section"]}（[四条路线梳理](https://mp.weixin.qq.com/s/K_6MibGXWwh9OL9eSZxOMg)）
{ax_line}{code_line}{proj_line}- **入库日期：** {TODAY}
- **开源状态：** {p["open"]}
- **一句话说明：** {p["one_liner"]}
- **沉淀到 wiki：** [`wiki/entities/{entity_basename(p)}`](../../wiki/entities/{entity_basename(p)})

## 核心摘录（归纳）

- {p["why"]}
- {p["mechanism"]}

## 对 wiki 的映射

- [{entity_basename(p).replace(".md", "")}](../../wiki/entities/{entity_basename(p)})
- [freedof-sim2real-44-papers-technology-map](../../wiki/overview/freedof-sim2real-44-papers-technology-map.md)
- [sim2real-four-routes-identifiability](../../wiki/comparisons/sim2real-four-routes-identifiability.md)
"""


def _seq_block(p: dict) -> str:
    if p["open"] == "已开源" and p.get("code"):
        return """
## 源码运行时序图

```mermaid
sequenceDiagram
  autonumber
  participant U as 用户/脚本
  participant R as 官方仓库
  participant T as 训练/辨识/推理
  participant S as 仿真或真机
  U->>R: clone + 依赖安装
  U->>T: 配置/权重/数据
  T->>S: rollout 或辨识实验
  S-->>U: 日志与指标
```
"""
    if p["open"] in ("待核实", "待发布", "部分开源"):
        return f"""
## 源码运行时序图

**不适用（{p["open"]}）** — 截至 {TODAY} 以项目页/论文 Code availability 为准；入库未核验可运行入口。
"""
    return f"""
## 源码运行时序图

**不适用（{p["open"]}）** — 经典文献或策展资源，无可运行官方代码仓。
"""


def _entity_new(p: dict) -> str:
    slug = p["slug"]
    assert slug
    ax = p.get("arxiv")
    ax_yaml = f'arxiv: "{ax}"\n' if ax else ""
    code_yaml = f"code: {p['code']}\n" if p.get("code") else ""
    src_name = f"freedof_sim2real_{p['ref']:02d}_{slug}.md"
    abbrev = "\n".join(f"| {a} | {b} | {c} |" for a, b, c in p["abbrev"])
    related = [
        "../comparisons/sim2real-four-routes-identifiability.md",
        "../overview/freedof-sim2real-44-papers-technology-map.md",
        "../concepts/sim2real.md",
    ] + [r for r in p.get("extra_related", [])]
    ax_link = f"[arXiv:{ax}](https://arxiv.org/abs/{ax})" if ax else p["venue"]
    ax_pdf = f"- [arXiv PDF](https://arxiv.org/pdf/{ax})\n" if ax else ""
    code_read = f"- [{p['code']}]({p['code']})\n" if p.get("code") else ""
    proj_read = f"- [项目页]({p['project']})\n" if p.get("project") else ""
    tag_type = "entity" if p["ref"] != 44 else "entity"
    return f"""---
type: {tag_type}
tags:
{_yaml_list(p["tags"], 2)}
status: complete
updated: {TODAY}
{ax_yaml}{code_yaml}related:
{_yaml_list(related, 2)}
sources:
  - ../../sources/papers/{src_name}
  - ../../sources/papers/freedof_sim2real_44_catalog.md
  - ../../sources/blogs/{BLOG}
summary: "{p["one_liner"][:160]}"
---

# {p["title"].split("(")[0].strip()}

**{p["title"]}**（{ax_link}）收录于 [自由度FreeDof · Sim2Real 四条路线梳理](../../sources/blogs/{BLOG}) 参考文献 **[{p["ref"]:02d}/44]**，归类 **{p["section"]}**。

## 一句话定义

{p["one_liner"]}

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
{abbrev}

## 为什么重要

- {p["why"]}
- 在 [44 篇技术地图](../overview/freedof-sim2real-44-papers-technology-map.md) 与 [四条路线对比](../comparisons/sim2real-four-routes-identifiability.md) 中作为 **{p["section"]}** 节点。
- 开源结论：**{p["open"]}**（步骤 2.5，{TODAY}）。

## 核心机制

| 项 | 内容 |
|----|------|
| **出处** | {p["venue"]} |
| **文内章节** | {p["section"]} |
| **要点** | {p["mechanism"]} |
| **开源** | **{p["open"]}** |

{_seq_block(p)}

## 结论

**{p["conclusion"]}**

1. 文内角色：{p["section"]} 路线上的参考节点，非重复 arXiv 页面。
2. 机制要点：{p["mechanism"][:100]}…
3. 部署/复现前请对照原文与项目页，勿直接外推公众号数字。

## 关联页面

- [Sim2Real 四条路线（可辨识性）](../comparisons/sim2real-four-routes-identifiability.md)
- [44 篇技术地图](../overview/freedof-sim2real-44-papers-technology-map.md)
- [Sim2Real](../concepts/sim2real.md)

## 参考来源

- [freedof_sim2real_{p["ref"]:02d}_{slug}.md](../../sources/papers/freedof_sim2real_{p["ref"]:02d}_{slug}.md)
- [{BLOG}](../../sources/blogs/{BLOG})
- [freedof_sim2real_44_catalog.md](../../sources/papers/freedof_sim2real_44_catalog.md)

## 推荐继续阅读

{ax_pdf}{code_read}{proj_read}- [44 篇 Sim2Real 技术地图](../overview/freedof-sim2real-44-papers-technology-map.md)
"""


def _catalog() -> str:
    rows = []
    for p in PAPERS:
        ax = p.get("arxiv")
        ax_cell = f"[{ax}](https://arxiv.org/abs/{ax})" if ax else "—"
        wiki = (
            f"[{entity_basename(p).replace('.md', '')}](../../wiki/entities/{entity_basename(p)})"
        )
        reuse = "复用" if p.get("reuse") else "新建"
        rows.append(
            f"| {p['ref']:02d} | {p['section']} | {p['title'][:60]} | {ax_cell} | {p['open']} | {reuse} | {wiki} |"
        )
    body = "\n".join(rows)
    return f"""# FreeDof Sim2Real 四条路线 — 44 篇参考文献总索引

> 来源归档（paper catalog）

- **标题：** 从域随机化到残差学习：Sim2Real 技术路线梳理 — 参考文献表
- **类型：** paper catalog
- **原始链接：** https://mp.weixin.qq.com/s/K_6MibGXWwh9OL9eSZxOMg
- **博客归档：** [`wechat_freedof_sim2real_four_routes_dr_to_residual_2026-08-23.md`](../blogs/{BLOG})
- **入库日期：** {TODAY}
- **一句话说明：** 44/44 独立 wiki 详情节点（新建 + 复用已有 canonical 页），**0 重复 arXiv 节点**。

## 44 篇 → 本库节点

| # | 章节 | 论文 | arXiv | 开源 | 节点 | wiki |
|---|------|------|-------|------|------|------|
{body}

## 对 wiki 的映射

- **阅读坐标：** [freedof-sim2real-44-papers-technology-map](../../wiki/overview/freedof-sim2real-44-papers-technology-map.md)
- **对比页：** [sim2real-four-routes-identifiability](../../wiki/comparisons/sim2real-four-routes-identifiability.md)
- **姊妹篇：** [关节动力学辨识实验设计](../../wiki/methods/sim2real-joint-sysid-experiment-design.md)
"""


def _map() -> str:
    sections: dict[str, list[dict]] = {}
    for p in PAPERS:
        sections.setdefault(p["section"], []).append(p)

    def table(ps: list[dict]) -> str:
        lines = ["| # | 论文 | 节点 | 开源 |", "|---|------|------|------|"]
        for p in ps:
            short = p["title"].split(":")[0].split("(")[0].strip()[:48]
            lines.append(
                f"| {p['ref']:02d} | {short} | [{entity_basename(p).replace('.md', '')}]({entity_rel(p)}) | {p['open']} |"
            )
        return "\n".join(lines)

    sec_blocks = "\n\n".join(f"### {name}\n\n{table(ps)}" for name, ps in sections.items())
    new_n = sum(1 for p in PAPERS if not p.get("reuse"))
    reuse_n = sum(1 for p in PAPERS if p.get("reuse"))

    return f"""---
type: overview
tags: [overview, survey, sim2real, system-identification, domain-randomization, technology-map]
status: complete
updated: {TODAY}
related:
  - ../comparisons/sim2real-four-routes-identifiability.md
  - ../concepts/sim2real.md
  - ../methods/sim2real-joint-sysid-experiment-design.md
  - ../overview/hub-sim2real.md
  - ../entities/paper-pace-sim2real-legged-robots.md
  - ../entities/paper-survey-sim2real-rl-foundation-models.md
sources:
  - ../../sources/blogs/{BLOG}
  - ../../sources/papers/freedof_sim2real_44_catalog.md
summary: "自由度FreeDof 四条 Sim2Real 路线梳理：44 篇参考文献独立节点索引（{new_n} 新建 + {reuse_n} 复用）。"
---

# Sim2Real 四条路线：44 篇参考文献阅读坐标

> **本页定位**：为 [自由度FreeDof · 从域随机化到残差学习](https://mp.weixin.qq.com/s/K_6MibGXWwh9OL9eSZxOMg) 文末 **44 篇参考文献** 提供按章节组织的独立详情节点索引。

## 一句话观点

**Sim2Real 选型先看「参数能否辨识、剩余误差如何处理」，再按 SysID → 窄 DR → 残差/适应 分层组合；44 篇文献是四条立场在工程上的证据链，而非时间线摘要堆叠。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Sim2Real | Simulation to Real | 仿真策略迁移真机 |
| SysID | System Identification | 系统辨识 |
| DR | Domain Randomization | 域随机化 |
| RMA | Rapid Motor Adaptation | 在线快速适应 |
| OOD | Out-of-Distribution | 分布外/失配检测 |

## 为什么单独做这张地图

- 原文按 **可辨识性** 串联系统辨识、DR、在线适应、残差学习，参考文献跨四十年与多子领域。
- **44/44 独立节点**：**{new_n} 新建** + **{reuse_n} 复用** 已有 canonical 页；**0 重复 arXiv 节点**。
- 与 [四条路线对比](../comparisons/sim2real-four-routes-identifiability.md) 配合：对比页讲立场与组合，本页讲 **逐篇入口**。

## 流程总览

```mermaid
flowchart LR
  subgraph id["§2 系统辨识"]
    PACE[PACE]
    SPI[SPI-Active]
  end
  subgraph dr["§3 域随机化"]
    Peng[Peng DR]
    Poly[PolySim]
  end
  subgraph ad["§4 在线适应"]
    RMA[RMA]
    UP[UP-OSI]
  end
  subgraph res["§5 残差学习"]
    AN[Actuator Net]
    ASAP[ASAP]
  end
  id --> dr --> ad --> res
```

## 分组索引

{sec_blocks}

## 关联页面

- [Sim2Real 四条路线（可辨识性）](../comparisons/sim2real-four-routes-identifiability.md)
- [Sim2Real](../concepts/sim2real.md)
- [Hub: Sim2Real](../overview/hub-sim2real.md)
- [Sim2Real RL 综述（2502.13187）](../entities/paper-survey-sim2real-rl-foundation-models.md)

## 参考来源

- [{BLOG}](../../sources/blogs/{BLOG})
- [freedof_sim2real_44_catalog.md](../../sources/papers/freedof_sim2real_44_catalog.md)

## 推荐继续阅读

- [PACE](../entities/paper-pace-sim2real-legged-robots.md)
- [ASAP](../entities/paper-hrl-stack-25-asap.md)
- [AwesomeSim2Real 综述](../entities/paper-survey-sim2real-rl-foundation-models.md)
"""


def _patch_reuse(p: dict) -> None:
    path = ROOT / "wiki/entities" / entity_basename(p)
    if not path.exists():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8")
    catalog_src = "  - ../../sources/papers/freedof_sim2real_44_catalog.md\n"
    blog_src = f"  - ../../sources/blogs/{BLOG}\n"
    map_rel = "../overview/freedof-sim2real-44-papers-technology-map.md"
    if catalog_src.strip() not in text:
        text = re.sub(r"(sources:\n)", r"\1" + catalog_src, text, count=1)
    if BLOG not in text:
        text = re.sub(r"(sources:\n(?:  - .+\n)+)", lambda m: m.group(0) + blog_src, text, count=1)
    if "freedof-sim2real-44-papers-technology-map" not in text:
        text = re.sub(
            r"(related:\n(?:  - .+\n)+)", lambda m: m.group(0) + f"  - {map_rel}\n", text, count=1
        )
    if p.get("arxiv") and "arxiv:" not in text.split("---", 2)[1]:
        text = text.replace(f"updated: {TODAY}", f'updated: {TODAY}\narxiv: "{p["arxiv"]}"')
    text = re.sub(r"^updated: \d{4}-\d{2}-\d{2}", f"updated: {TODAY}", text, count=1, flags=re.M)
    path.write_text(text, encoding="utf-8")


def _update_blog() -> None:
    rows = []
    for p in PAPERS:
        ax = p.get("arxiv")
        ax_cell = f"[{ax}](https://arxiv.org/abs/{ax})" if ax else "—"
        wiki = (
            f"[{entity_basename(p).replace('.md', '')}](../../wiki/entities/{entity_basename(p)})"
        )
        reuse = "**复用**" if p.get("reuse") else "**新建**"
        rows.append(
            f"| {p['ref']:02d} | {p['section']} | {p['title'][:55]} | {ax_cell} | {reuse} | {wiki} |"
        )
    table = "\n".join(rows)
    text = BLOG_PATH.read_text(encoding="utf-8")
    block = f"""
### 44 篇参考文献 → 独立详情节点（{TODAY} 补齐）

| # | 章节 | 论文 | arXiv | 节点 | wiki |
|---|------|------|-------|------|------|
{table}

- **44/44 独立节点**；阅读坐标：[freedof-sim2real-44-papers-technology-map](../../wiki/overview/freedof-sim2real-44-papers-technology-map.md)
"""
    if "44 篇参考文献 → 独立详情节点" not in text:
        text = text.replace(
            "## 当前提炼状态\n\n- [x] 公众号正文抓取与 raw 归档\n- [x] 升格对比页（四条路线 + 可辨识性轴）\n- [x] 与姊妹篇 SysID 实验设计页交叉链接",
            "## 当前提炼状态\n\n- [x] 公众号正文抓取与 raw 归档\n- [x] 升格对比页（四条路线 + 可辨识性轴）\n- [x] 与姊妹篇 SysID 实验设计页交叉链接\n- [x] 44 篇参考文献独立详情节点（0 重复 arXiv）",
        )
        text = text.rstrip() + "\n" + block + "\n"
    else:
        # replace table section
        text = re.sub(
            r"\n### 44 篇参考文献 → 独立详情节点.*",
            "\n" + block,
            text,
            flags=re.S,
        )
    # update mapping section
    text = re.sub(
        r"- \*\*新建对比页：\*\*.*",
        "- **对比页：** [sim2real-four-routes-identifiability](../../wiki/comparisons/sim2real-four-routes-identifiability.md)\n"
        "- **44 篇地图：** [freedof-sim2real-44-papers-technology-map](../../wiki/overview/freedof-sim2real-44-papers-technology-map.md)\n"
        "- **44 篇 catalog：** [freedof_sim2real_44_catalog.md](../papers/freedof_sim2real_44_catalog.md)",
        text,
        count=1,
    )
    # fix named table entries that said 暂无独立页
    replacements = {
        "SPI-Active | （文内引用；本库暂无独立页）": "SPI-Active | [paper-notebook-sampling-based-system-identification-with-active](../../wiki/entities/paper-notebook-sampling-based-system-identification-with-active.md)",
        "PolySim | （文内引用）": "PolySim | [paper-polysim-multi-simulator-humanoid-sim2real](../../wiki/entities/paper-polysim-multi-simulator-humanoid-sim2real.md)",
        "RAPT | （文内引用 arXiv:2602.01515）": "RAPT | [paper-rapt-sim2real-ood-detection](../../wiki/entities/paper-rapt-sim2real-ood-detection.md)",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    BLOG_PATH.write_text(text, encoding="utf-8")


def _update_comparison() -> None:
    text = COMP_PATH.read_text(encoding="utf-8")
    if "freedof-sim2real-44-papers-technology-map" not in text:
        text = text.replace(
            "  - ../overview/hub-sim2real.md\n",
            "  - ../overview/hub-sim2real.md\n  - ../overview/freedof-sim2real-44-papers-technology-map.md\n",
        )
        text = text.replace(
            "  - ../../sources/blogs/wechat_freedof_sim2real_dynamics_identification.md\n",
            "  - ../../sources/blogs/wechat_freedof_sim2real_dynamics_identification.md\n  - ../../sources/papers/freedof_sim2real_44_catalog.md\n",
        )
    text = re.sub(r"^updated: \d{4}-\d{2}-\d{2}", f"updated: {TODAY}", text, count=1, flags=re.M)
    if "44 篇参考文献" not in text:
        insert = """

### 44 篇参考文献索引

逐篇独立详情节点见 [Sim2Real 44 篇技术地图](../overview/freedof-sim2real-44-papers-technology-map.md)（44/44，0 重复 arXiv）。代表节点：[PACE](../entities/paper-pace-sim2real-legged-robots.md)、[SPI-Active](../entities/paper-notebook-sampling-based-system-identification-with-active.md)、[RMA](../entities/paper-rma-rapid-motor-adaptation.md)、[ASAP](../entities/paper-hrl-stack-25-asap.md)、[PolySim](../entities/paper-polysim-multi-simulator-humanoid-sim2real.md)、[RAPT](../entities/paper-rapt-sim2real-ood-detection.md)。
"""
        text = text.replace("## 关联页面\n", insert + "\n## 关联页面\n")
    COMP_PATH.write_text(text, encoding="utf-8")


def _log_append() -> None:
    log = ROOT / "log.md"
    entry = f"""
## [{TODAY}] ingest | sources/blogs/wechat_freedof_sim2real_four_routes — 44 篇参考文献独立详情节点 + 技术地图

- **意图：** 补齐 FreeDof Sim2Real 四条路线公众号文末 44 篇文献表，每篇独立非重复 wiki 节点。
- **关键页：** [freedof-sim2real-44-papers-technology-map](wiki/overview/freedof-sim2real-44-papers-technology-map.md)、[freedof_sim2real_44_catalog](sources/papers/freedof_sim2real_44_catalog.md)
"""
    content = log.read_text(encoding="utf-8")
    if "44 篇参考文献独立详情节点" not in content:
        # insert after first line heading
        content = content.replace("\n## [", entry + "\n## [", 1)
        log.write_text(content, encoding="utf-8")


def main() -> None:
    new_count = 0
    reuse_count = 0
    for p in PAPERS:
        slug_for_src = p["slug"] or entity_basename(p).replace("paper-", "").replace(".md", "")
        src_path = ROOT / "sources/papers" / f"freedof_sim2real_{p['ref']:02d}_{slug_for_src}.md"
        src_path.write_text(_paper_source(p), encoding="utf-8")
        if p.get("reuse"):
            _patch_reuse(p)
            reuse_count += 1
        else:
            ent_path = ROOT / "wiki/entities" / entity_basename(p)
            ent_path.write_text(_entity_new(p), encoding="utf-8")
            new_count += 1
            if p.get("code") and p["ref"] != 44:
                repo_slug = p["slug"].replace("-", "_")
                (ROOT / "sources/repos" / f"{repo_slug}.md").write_text(
                    f"# {p['title']}\n\n- **链接：** {p['code']}\n- **入库日期：** {TODAY}\n",
                    encoding="utf-8",
                )
    if p := next(x for x in PAPERS if x["ref"] == 44):
        (ROOT / "sources/repos/awesome-humanoid-robot-learning.md").write_text(
            f"# Awesome Humanoid Robot Learning\n\n- **链接：** {p['code']}\n- **入库日期：** {TODAY}\n- **wiki：** [awesome-humanoid-robot-learning](../../wiki/entities/awesome-humanoid-robot-learning.md)\n",
            encoding="utf-8",
        )
    CATALOG.write_text(_catalog(), encoding="utf-8")
    MAP_PATH.write_text(_map(), encoding="utf-8")
    _update_blog()
    _update_comparison()
    _log_append()
    print(f"Done: {new_count} new entities, {reuse_count} reused, 44 sources, map+catalog")


if __name__ == "__main__":
    main()
