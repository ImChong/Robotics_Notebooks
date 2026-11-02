#!/usr/bin/env python3
"""Bootstrap ingest: senlanke 具身运控lab 10.26-10.30 前沿论文动态（人形/四足 33）.

公众号用 scripts/fetch_wechat_article.py（Chromium + 微信 UA）读取；条目 33 个唯一 arXiv，
其中 10 篇已有详情节点（复用，仅追加来源），23 篇新建独立 paper-* 节点。arXiv 摘要/作者/评论由摘要页抓取后内嵌。
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TODAY = date.today().isoformat()
PUB = "2026-11-01"
WEEK = "10-26–10-30"
BLOGS = {
    "hq": {
        "file": "wechat_senlanke_weekly_humanoid_quadruped_2026-10-26_30.md",
        "title": "【10.26-10.30前沿论文动态】人形／四足33篇",
        "url": "https://mp.weixin.qq.com/s/eNLudfSs4fI_9YEiBojtJQ",
        "overview": "senlanke-weekly-2026-10-26-30-humanoid-quadruped-technology-map",
        "label": "人形/四足",
    },
}

SEC_META = {
    "人形机器人": (
        "humanoid",
        ["paper", "humanoid"],
        ["../tasks/humanoid-locomotion.md", "../methods/reinforcement-learning.md"],
    ),
    "四足机器人": (
        "quadruped",
        ["paper", "quadruped", "locomotion"],
        ["../tasks/locomotion.md", "../methods/reinforcement-learning.md"],
    ),
    "轮足与通用腿式机器人": (
        "legged",
        ["paper", "legged-robot", "locomotion"],
        ["../tasks/locomotion.md"],
    ),
}
MAN_REL = ["../tasks/manipulation.md", "../methods/vla.md"]
MAN_TAGS = {
    "VLA / 通用操作策略": ["vla", "manipulation"],
    "WAM / World Model 与操作预测": ["world-action-model", "manipulation"],
    "在线/离线强化学习与后训练": ["reinforcement-learning", "manipulation"],
    "模仿学习、视觉运动策略与生成式动作策略": ["imitation-learning", "manipulation"],
    "双臂操作": ["bimanual", "manipulation"],
    "灵巧操作": ["dexterous-manipulation", "manipulation"],
    "柔性/可变形物体操作": ["deformable-manipulation", "manipulation"],
    "长程任务、推理与规划": ["long-horizon", "manipulation"],
    "数据集、预训练、跨本体与硬件平台": ["cross-embodiment", "manipulation"],
}
ABBREV = {
    "VLA": ("Vision-Language-Action", "视觉–语言–动作模型"),
    "WAM": ("World Action Model", "世界–动作联合建模"),
    "RL": ("Reinforcement Learning", "强化学习"),
    "IL": ("Imitation Learning", "模仿学习"),
    "WBC": ("Whole-Body Control", "全身控制"),
    "MPC": ("Model Predictive Control", "模型预测控制"),
    "LiDAR": ("Light Detection and Ranging", "激光雷达"),
    "IMU": ("Inertial Measurement Unit", "惯性测量单元"),
}

DATA = json.loads(r"""{
 "papers": {
  "2610.34740": {
   "id": "2610.34740",
   "title": "Networked Humanoid Intelligence: Communication, Coordination, Collaboration, and Social Interaction",
   "zh": "网络化人形智能：通信、协调、协作与社会交互",
   "org": "庆熙大学、悉尼大学",
   "prob": "人形研究通常将机器人视为独立个体，缺少对多机器人共享信息、任务、基础设施和知识的统一分析。",
   "inno": "按通信、协调、协作和社会智能四个功能梳理网络化人形系统，并从交互架构、交换信息、物理耦合、时间尺度、系统规模和机器人形态六个维度建立分类体系。",
   "reuse": null,
   "short": "Networked Humanoid Intelligence",
   "slug": "networked-humanoid-intelligence-survey",
   "abstract": "Humanoid robots are moving from laboratory platforms toward connected fleets that share workspaces, infrastructure, tasks, and knowledge with other robots and humans. As deployments grow, the operation and improvement of each humanoid depend on information exchanged across the network. However, existing research has largely studied humanoids as individual systems. This survey examines networked humanoid intelligence through four functions of information exchange: communication, coordination, collaboration, and social intelligence with human interaction. We first establish the foundations of networked humanoid systems, define the question addressed by each function, and connect them through a common factory scenario. We review the literature along six dimensions: interaction architecture, exchanged information, physical coupling, operating timescale, system scale, and embodiment. We separate results obtained with physical and simulated humanoids from those based on other robotic platforms or disembodied agents, because the latter do not fully capture humanoid-specific constraints such as balance, whole-body motion, contact, and physical interaction. We examine hardware, datasets, and benchmarks, and distinguish risks handled locally from those networking introduces, such as stale information or unsafe knowledge propagation. Finally, we identify open problems in control-aware networking, compact body-centric messages, physically grounded coordination, network- and embodiment-aware knowledge sharing, fleet-level social consistency, and accountable human oversight.",
   "comments": "",
   "authors": "Nguyen, Loc X., Le, Huy Q., Yoon, Ji Su, Tran, Nguyen H., Huh, Eui-Nam, Hong, Choong Seon",
   "links": []
  },
  "2610.35075": {
   "id": "2610.35075",
   "title": "Zero-HLM: Generalizable Humanoid Loco-Manipulation without Real Robot Data",
   "zh": "Zero-HLM：无需真实机器人数据的通用人形移动操作学习",
   "org": "北京理工大学、北京通用人工智能研究院、清华大学、Delta Intelligence",
   "prob": "人形移动操作依赖昂贵的真实示范，而普通仿真场景又存在视觉、物理属性和全身动作不一致。",
   "inno": "从手机采集重建高斯场景，用策略执行反馈联合修正动作参考和交互几何；生成 4800 条物理验证示范，使 GR00T N1.7 在三项 G1 任务上直接 Sim-to-Real。",
   "reuse": "wiki/entities/paper-zero-hlm.md"
  },
  "2610.35862": {
   "id": "2610.35862",
   "title": "Learning Force-Reactive Humanoid Loco-Manipulation",
   "zh": "力反馈响应式人形移动操作学习",
   "org": "Amazon Frontier AI & Robotics、纽约大学、卡内基梅隆大学、芝加哥大学",
   "prob": "推门、擦拭等任务不仅要求手部到达目标位置，还要控制双手接触力并保持全身平衡。",
   "inno": "使用带力传感器的手持设备采集动作和接触扳手；扩散策略同时预测动作与双手目标力，再由混合柔顺全身控制器跟踪力指令并维持平衡。",
   "reuse": "wiki/entities/paper-flore.md"
  },
  "2610.36452": {
   "id": "2610.36452",
   "title": "Whole-Body Humanoid Control via Stochastic Motion Tokenization",
   "zh": "基于随机动作标记化的人形全身控制",
   "org": "达姆施塔特工业大学、德国人工智能研究中心、hessian.AI、Robotics Institute Germany",
   "prob": "速度指令不足以描述舞蹈、格斗等复杂全身动作，而密集动作参考不适合作为高层任务接口。",
   "inno": "用随机动作 Tokenizer 将全身运动压缩成离散 token，并将 token 直接输入跟踪策略；同一接口能够生成并执行行走、格斗、舞蹈以及长时技能转换。",
   "reuse": null,
   "short": "随机动作标记化全身控制",
   "slug": "stochastic-motion-tokenization-humanoid-wbc",
   "abstract": "While velocity-commanded RL policies are effective for controlling quadrupeds, they are a limited interface for humanoids that need to perform complex whole-body motions. Furthermore, RL tracking policies for whole-body motions typically have limited information about the desired future motion, preventing them from performing controlled deviations to achieve long-term task success; such information is particularly relevant when a controller must prepare for transitions between behaviors. We address these challenges using concise motion representations that can be generated by a high-level controller, given a behavior label or text command. These representations can be fed directly into a tracking policy to provide a detailed motion reference. Our method exhibits a high degree of flexibility in performing real-world robot motions such as dancing, fighting, or walking based on conditional information. At matched capacity, our adopted stochastic motion tokenizer reconstructs motions from the Unitree G1 at a level that exceeds state-of-the-art approaches such as finite scalar quantization (FSQ) and the vector-quantized autoencoder (VQ-VAE). The resulting discrete interface conditions whole-body tracking policies that execute generated dance, fight, and walk motions lasting 16.8 s each and a 35 s fight-to-dance sequence on a physical Unitree G1.",
   "comments": "",
   "authors": "Drolet, Michael, Arenz, Oleg, Peters, Jan",
   "links": []
  },
  "2610.36507": {
   "id": "2610.36507",
   "title": "GRAFT: Geometry-Aware Residual Adaptation for Foundation Trackers",
   "zh": "GRAFT：面向基础动作跟踪器的几何感知残差适配",
   "org": "小米机器人实验室、香港科技大学（广州）",
   "prob": "在平地动作上训练的全身跟踪器无法根据楼梯、沟隙和平台调整抬脚高度、落脚位置与接触时序；重新微调完整策略成本较高。",
   "inno": "冻结原始动作跟踪器，只用轻量适配器对下肢执行目标添加地形条件残差；适配器从特权地形观测蒸馏到机载深度输入，并在平地上抑制不必要修正。",
   "reuse": "wiki/entities/paper-graft.md"
  },
  "2610.36546": {
   "id": "2610.36546",
   "title": "REACH: An RL-based Hierarchical Humanoid Crawling Framework for Adaptable Confined Space Traversal",
   "zh": "REACH：面向可适应狭窄空间穿越的分层强化学习人形爬行框架",
   "org": "南方科技大学、香港中文大学",
   "prob": "仅降低双足行走高度仍无法穿越极低空间，而且行走与爬行之间存在较大的姿态和接触模式突变。",
   "inno": "分别训练变高度双足策略和爬行策略，以历史条件 latent 编码接触及地形信息；双向转换策略连接两种模式，激光雷达规划器进一步生成三维路径、速度和主动高度指令。",
   "reuse": null,
   "short": "REACH",
   "slug": "reach-humanoid-crawling-confined-space",
   "abstract": "Humanoid locomotion in confined spaces requires continuous height adaptation across multiple locomotion modes, yet most existing approaches remain limited to height-adjustable bipedal walking. We present REACH, an RL-based hierarchical framework that enables a Unitree G1 humanoid to traverse height-constrained environments through variable-height bipedal walking and crawling. The learned policies regulate body height over complementary command ranges of 1.27--0.50m in bipedal locomotion and 0.50--0.20m in crawling, extending traversal beyond the clearance limits of crouched walking. To address contact-rich crawling dynamics, we employ a history-conditioned latent representation that captures terrain and interaction context from proprioceptive observations. A bidirectional transition policy further connects crouched bipedal locomotion and crawling across large kinematic and contact discontinuities. These skills are integrated with a LiDAR-based planner that converts point clouds into collision-free three-dimensional paths and locomotion commands, including proactive height targets. Real-world experiments demonstrate accurate height and velocity tracking, robust terrain traversal, reliable mode transitions and confined-space traversal with the proposed pipeline. Code and videos are available at this https URL.",
   "comments": "",
   "authors": "Xu, Shengzhang, Guo, Yuning, Hu, Weiyi, Xu, Yifeng, Li, Zhongyu, Kong, He",
   "links": []
  },
  "2610.36615": {
   "id": "2610.36615",
   "title": "HumiMimic: Mimetic Embodiment Transfer of Human-Humanoid Interaction from In-the-Wild Videos",
   "zh": "HumiMimic：从自然场景视频迁移人类—人形交互行为",
   "org": "AIOZ、穆罕默德·本·扎耶德人工智能大学、RMIT 大学、利物浦大学、台湾清华大学、越南国家大学信息技术大学、庆应义塾大学",
   "prob": "将拥抱、握手和双人舞等动作迁移到机器人时，既要保持单人动作，还要保持双方距离、朝向、接触和节奏。",
   "inno": "从自然视频重建双人三维动作，再通过关节相对方向、解剖偏移和交互关键权重完成跨形态转移；构建约 50 万帧、支持两种人形平台的 HumiMimic 数据集。",
   "reuse": null,
   "short": "HumiMimic",
   "slug": "humimic-human-humanoid-interaction-video",
   "abstract": "Enabling a humanoid to interact naturally with human partners requires preserving not only individual motions but also the spatial, semantic, and contact relationships underlying the interaction (i.e., hugging, dancing, hand-shaking, etc.). Furthermore, a practical solution should generalize across various humanoid embodiments with distinct morphologies and kinematic constraints. To tackle this challenge, we propose HumiTransfer, a mimetic embodiment transfer framework that converts in-the-wild human-human interaction videos into physically plausible human-humanoid interactions. HumiTransfer first reconstructs 3D human motion with consistent foot-ground contact, then transfers one participant to a target humanoid embodiment via interaction modeling, using weighted joint-relative directions and anatomical offsets to prioritize interaction-critical inter-body relationships, while post-retargeting optimization enforces balance, contact, and collision avoidance. Subsequently, we apply HumiTransfer to 1,200 in-the-wild videos, producing approximately 500,000 frames of diverse interactions for the 26-DoF Unitree R1 Edu and the 23-DoF HighTorque Mini-hi humanoid robot, resulting in our HumiMimic dataset. Across both robots, HumiTransfer achieves superior motion fidelity, preserves inter-body relationships, and maintains contact accuracy compared with existing baselines. Moreover, we demonstrate that HumiMimic supports motion generation methods for synthesizing interactive humanoid motions following a human partner. Finally, our real robot deployments confirm the physical executability of the transferred motions in real-world settings. Our project is available at this https URL",
   "comments": "",
   "authors": "Le, Nghia, Nguyen, Khang, Tran, Ba Long, Pham, Thi Quynh Nhu, Nguyen, Binh, Do, Tuong, Tjiputra, Erman, Nguyen, Hoan et al.",
   "links": []
  },
  "2610.36825": {
   "id": "2610.36825",
   "title": "HC2Robot: Hyper-Connection Transformers with Predicted Foresight for Multi-Terrain Humanoid Parkour",
   "zh": "HC2Robot：基于超连接 Transformer 与预测前瞻的人形多地形跑酷",
   "org": "香港科技大学（广州）、CL-TECH、华南农业大学、南方科技大学",
   "prob": "非对称 Actor–Critic 中，Critic 能够看到特权接触和地形信息，但部署 Actor 只能使用机载观测，导致复杂地形落脚决策存在信息差。",
   "inno": "前瞻模块从可部署观测预测下一次落脚的时间、位置和支撑质量，将特权监督转化为部署 latent；超连接 Transformer 使用多流路由增强时间推理，在楼梯、沟隙、高台和斜坡上平均成功率达到 97.6%。",
   "reuse": null,
   "short": "HC2Robot",
   "slug": "hc2robot-humanoid-parkour",
   "abstract": "Humanoid parkour requires robots to reason about future footholds and whole-body motions over highly discontinuous terrain. We revisit the information asymmetry in asymmetric actor-critic learning for perceptive humanoid locomotion: the critic exploits privileged simulator signals during training, whereas the deployable actor operates solely from onboard perception. We argue that the key is not exposing privileged information to the actor, but learning a deployable approximation of it to narrow the information gap between training and deployment. We present HC2Robot, a deployable perceptive humanoid parkour framework consisting of a Predicted Foresight Module (PFM) and a Hyper-Connection (HC) Transformer. PFM predicts from deployable observations the policy-conditioned timing, location, and support quality of each foot's next touchdown event, converting privileged supervision into a deployable latent representation that provides the actor with predicted touchdown information without privileged runtime sensing. The HC Transformer further enhances temporal reasoning through learnable multi-stream routing, while a future motion prediction head provides short-horizon motion priors for regularization. Across stairs, gaps, high platforms, and slopes, HC2Robot achieves a mean simulated traversal success rate of 97.60% and consistently higher safe foothold rates than strong perceptive locomotion baselines, supporting improved traversal performance and geometric foothold support under deployable sensing.",
   "comments": "",
   "authors": "Xu, Wenhao, Mai, Yanheng, Fu, Yifei, Zheng, Nan, Zhou, Yanbo, Huang, Zirui, Ling, Mingsheng, Wang, Xinjue et al.",
   "links": []
  },
  "2610.36859": {
   "id": "2610.36859",
   "title": "ActiveAlign: Aligning Egocentric Manipulation via Active Perception",
   "zh": "ActiveAlign：通过主动感知对齐第一视角机器人操作",
   "org": "NVIDIA、加州大学圣迭戈分校",
   "prob": "被动采集的图像难以准确恢复相机外参和物体位置，导致真实场景与仿真场景错位，影响毫米级操作。",
   "inno": "利用主动挥手动作和可微渲染完成手眼标定，再通过主动触碰及触觉反馈细化物体位姿；在 Unitree G1 上完成按钮和滑块等毫米级操作。",
   "reuse": null,
   "short": "ActiveAlign（主动感知）",
   "slug": "activealign-egocentric-active-perception",
   "abstract": "Real-to-Sim-to-Real (Real2Sim2Real) enables structured world representations for planning, interpretability, and safe execution. Most existing approaches rely on passive perception, which creates real2sim gaps due to imperfect observation. Inspired by how humans perceive actively through actions, we propose ActiveAlign, a framework that leverages active perception to align the Real2Sim2Real gap from interactions. ActiveAlign introduces two complementary modules: (i) ActiveAlign-Waving, which performs hand-eye calibration by exploiting active hand motions and differentiable rendering, and (ii) ActiveAlign-Touching, which refines object alignment through tactile feedback to achieve millimeter-level precision. Together, these capabilities enable strong real2sim alignment. We demonstrate that ActiveAlign-Waving can be used to curate simulation environments for open-sourced uncalibrated robot manipulation datasets, and demonstrate on a real Unitree G1 robot that ActiveAlign enables fine manipulation of millimeter-scale objects such as pressing buttons and moving sliders.",
   "comments": "",
   "authors": "Qiu, Ri-Zhao, Wen, Bowen, Jiang, Yifeng, Tessler, Chen, Heiden, Eric, Peng, Xue Bin, Wang, Xiaolong, Birchfield, Stan et al.",
   "links": []
  },
  "2610.37801": {
   "id": "2610.37801",
   "title": "An Empirical Study of Wrist-Target Construction and Joint-Angle Limits in Humanoid Arm Retargeting",
   "zh": "人形手臂重定向中腕部目标构造与关节限位的实证研究",
   "org": "佐治亚理工学院",
   "prob": "人体腕部目标直接映射到机器人时，可能频繁把手臂关节推到限位，并增加跟踪误差和执行力矩。",
   "inno": "对比按人体肢段方向重建的腕部目标与按机器人可达距离归一化的目标；实验证明距离归一化能显著降低 Unitree H1-2 关节触限次数。",
   "reuse": null,
   "short": "腕目标与关节限位实证",
   "slug": "wrist-target-joint-limit-humanoid-arm-retargeting",
   "abstract": "Motion retargeting maps a demonstrated human movement onto a robot to compute its joint motion. This study examines how the construction of the desired wrist position affects whether computed arm joint angles reach their permitted limits on a Unitree H1-2. A reconstructed target is formed from demonstrated arm directions and robot segment lengths, so its shoulder-to-wrist distance depends on the demonstrated angle between the segments even when no posture-matching preference is applied. A reach-normalized target retains that direction while rescaling the distance, using the same processed demonstrations and the same joint-angle method. Across a forward-reach sweep the reconstructed target places a joint angle at a bound in 10 of 20 repetitions and the reach-normalized target in none. Across 25 recordings from one participant the corresponding counts are 8 and 6 on the reaching arm and 14 and 5 on the supporting arm, and a second participant repeats the bricklaying and lifting comparison. In hardware replay a joint is commanded to a bound in 12 of 16 direction-matching arm-records and in none of the 16 normalized records, and within the eligible records both tracking error and executed torque at those frames exceed their values elsewhere in the same run. Cross-body reach and arm-to-arm clearance identify limits of radial adjustment. The findings support treating target construction and posture matching as separate retargeting decisions, and evaluating both alongside the wrist positions and clearance that manipulation requires.",
   "comments": "",
   "authors": "Chang, MinJun",
   "links": []
  },
  "2610.37974": {
   "id": "2610.37974",
   "title": "FORGE: Feasible Online Reduced-Order Reference Generation and Whole-Body Execution for Agile Humanoid Locomotion",
   "zh": "FORGE：面向敏捷人形运动的在线可行降阶参考生成与全身执行",
   "org": "加州理工学院、得克萨斯大学埃尔帕索分校",
   "prob": "降阶动力学模型可快速生成动作，但生成的参考不一定满足完整人形机器人的运动学和执行器约束。",
   "inno": "先在单刚体动力学和完整运动学组成的降阶环境中训练参考生成器，再用其在线输出训练全阶跟踪策略；单一系统覆盖前后、侧向、转向、单腿跳和连续技能转换。",
   "reuse": null,
   "short": "FORGE",
   "slug": "forge-humanoid-reduced-order-reference",
   "abstract": "This paper presents a two-stage learning framework for versatile and agile humanoid locomotion that couples online reduced-order model (ROM) reference generation with whole-body tracking control. In the first stage, we train a command-conditioned whole-body reference generator using single rigid body dynamics combined with full humanoid kinematics. In the second stage, the frozen ROM policy is rolled out online to generate references for training a whole-body tracking policy on the full-order dynamics model. Learning a kino-dynamically consistent reference generator in the simplified ROM space enables us to generate references over a broad command space and over multiple locomotion strategies using a single unified framework. We demonstrate the approach on a Unitree G1 humanoid robot, achieving longitudinal velocities from -2.0 m/s to 2.8 m/s, lateral velocities up to +-1.0 m/s, angular velocities up to +-3.0 rad/s, one-legged hopping, and continuous transitions across the commands, all using one unified reference policy and one whole-body tracking policy.",
   "comments": "",
   "authors": "Ghansah, Adrian B., An, Yujin, Kim, Jeeseop, Chung, Soon-Jo",
   "links": []
  },
  "2610.38027": {
   "id": "2610.38027",
   "title": "METATRON: A Promptable Whole-Body Execution Layer for Humanoids",
   "zh": "METATRON：可提示的人形全身动作执行层",
   "org": "NVIDIA、Dollhouse Robotics",
   "prob": "不同人形任务通常分别训练控制策略，重复学习平衡和全身协调能力。",
   "inno": "冻结一个接收身体部位目标、速度和风格提示的生成式基础策略；高层技能以有状态程序逐周期生成提示，使编码智能体能够通过修改程序而不是重新训练策略来构建新技能。",
   "reuse": null,
   "short": "METATRON",
   "slug": "metatron-promptable-whole-body-execution",
   "abstract": "Humanoid controllers often require task-specific training despite sharing the need for balance and whole-body coordination. We present Metatron, a promptable whole-body execution layer combining a generative base policy with a programmable skill layer. The base policy receives partial goals, collectively called a prompt, that can combine timed body-part targets, velocity commands, and style cues. It produces diverse whole-body motions consistent with these goals. Training combines motion capture, synthetic recovery trajectories, and retargeted scene-interaction motions. After training, the policy is frozen. Stateful programs called skills express task intent as prompts at each control step, drawing rough geometric sketches that the policy turns into coordinated movement. Skills are ordinary code, refined by coding agents that analyze motion data and fit compact rules without task-specific policy training. On the physical Unitree G1, one frozen checkpoint supports walking, recovery, teleoperated object pickup, and sitting on furniture. Comparative evaluations show higher reaching success and push robustness than the tested reference-tracking pipelines. The same recipe produces climbing and acrobatics on a simulated character. Metatron enables coding agents to turn learned motor capabilities into new skills through code, without further policy training.",
   "comments": "",
   "authors": "Tessler, Chen, Jiang, Yifeng, Otto, Alec, Shi, Yi, Iqbal, Umar, Chechik, Gal, Peng, Xue Bin",
   "links": []
  },
  "2610.39044": {
   "id": "2610.39044",
   "title": "Catch It All!: Generalizable Dynamic Catching with a Dexterous Hand",
   "zh": "Catch It All！基于灵巧手的可泛化动态接取",
   "org": "加州大学圣迭戈分校、哈维穆德学院",
   "prob": "人形机器人需要在单目视觉噪声下接住形状、弹性、柔软度和反弹特性均未知的物体。",
   "inno": "通过抓取奖励和安全课程学习实时适应物体姿态及形状的接取策略；仅用一台第一视角相机，在 G1 上零样本接住 9 种未见物体。",
   "reuse": "wiki/entities/paper-catch-it-all.md"
  },
  "2610.39327": {
   "id": "2610.39327",
   "title": "SONIC-K1: Porting a Universal Humanoid Motion Tracker to a Small Humanoid on a Single GPU",
   "zh": "SONIC-K1：使用单 GPU 将通用人形动作跟踪器迁移到小型人形机器人",
   "org": "维尔纽斯大学、肯特州立大学",
   "prob": "SONIC 等通用动作跟踪方法训练成本高，且针对小型人形机器人的公开模型和权重较少。",
   "inno": "用单张共享 GPU 从头训练 Booster K1 全身跟踪器，并公开权重、代码和评测工具；仿真可跟踪大部分动作，实机已实现上肢和慢速动作跟踪。",
   "reuse": "wiki/entities/paper-sonic-k1.md"
  },
  "2610.40003": {
   "id": "2610.40003",
   "title": "Humanoid Dodgeball: Learning Fast Dynamic Obstacle Avoidance Motions for Humanoids",
   "zh": "人形躲避球：快速动态障碍规避动作学习",
   "org": "新加坡国立大学、众擎机器人、大连理工大学",
   "prob": "面对快速飞来的物体，人形机器人必须在很短时间内完成全身闪避，同时保持平衡和动作自然性。",
   "inno": "使用威胁感知奖励训练反应式 RL 策略，并用多个方向特定的对抗动作先验约束闪避动作；在 EngineAI PM01 上验证跨速度、跨发射距离的前跨和后跳闪避。",
   "reuse": null,
   "short": "Humanoid Dodgeball",
   "slug": "humanoid-dodgeball-dynamic-obstacle-avoidance",
   "abstract": "Humanoids have become increasingly agile and capable in real-world environments. To operate safely in dynamic scenarios, they must rapidly evade fast-moving obstacles while maintaining stability and physically feasible whole-body motion. Learning such behaviors remains challenging due to the high-dimensional dynamics of humanoid systems and the need to jointly balance reactive obstacle avoidance, motion naturalness, and control robustness. In this work, we present Humanoid Dodgeball, a reinforcement learning (RL) framework for agile dynamic obstacle avoidance in humanoids. Our method enables highly dynamic whole-body evasive maneuvers against incoming projectiles through threat-aware reward shaping, while multiple direction-specific adversarial motion priors regularize the learned behaviors toward stable and diverse human-like motions. Extensive simulation and hardware experiments demonstrate that the proposed framework substantially improves dynamic obstacle avoidance performance across diverse obstacle speeds and launch distances, while preserving robust and physically plausible humanoid motions.",
   "comments": "",
   "authors": "Zhang, Aoqian, Wang, Chunzheng, Chen, Haoran, Tang, Annan, Xiang, Cheng, Shi, Fan, Ge, Shuzhi Sam",
   "links": []
  },
  "2610.35462": {
   "id": "2610.35462",
   "title": "DEX-PHD: A Dexterous Portable Handheld Device for Multimodal Data Collection",
   "zh": "DEX-PHD：面向多模态数据采集的便携式灵巧手持设备",
   "org": "伊利诺伊大学厄巴纳—香槟分校 KIMLAB",
   "prob": "机器人运动链遥操作能够准确记录状态，但不便携；纯视觉采集便携，却缺少准确的手指状态和接触数据。",
   "inno": "将灵巧机器人手、逐指控制、腕部相机、关节状态和指尖接触传感集成到无系留手持设备；采集的数据可在实体人形机器人上直接开环回放。",
   "reuse": null,
   "short": "DEX-PHD",
   "slug": "dex-phd-handheld-data-collection",
   "abstract": "Existing manipulation data collection methods typically offer either accurate robot-state acquisition through kinematic-chain-based teleoperation or convenient, portable operation through vision-based approaches. We present DEX-PHD, a portable handheld device that combines these advantages for dexterous manipulation by integrating a dexterous robot hand, continuous per-finger control, and multimodal logging. The system records proprioceptive information, fingertip contact data, and wrist-camera video, while the operator directly controls the robot hand through joint-level mapping. Its untethered design enables data collection across diverse environments. We demonstrate DEX-PHD on contact-rich tasks requiring coordinated multi-finger manipulation and deploy the collected demonstrations through open-loop replay on a physical humanoid robot. The results show that open-loop replay preserves the main temporal patterns of finger motion and task-relevant contact. Therefore, DEX-PHD provides a practical platform for portable, multimodal dexterous manipulation data collection.",
   "comments": "",
   "authors": "Hong, Jooyoung, Myers, Noboru, Kwon, Obin, Yang, Yixian, Kim, Joohyung",
   "links": []
  },
  "2610.36789": {
   "id": "2610.36789",
   "title": "A Fully Articulated Humanoid Dexterous Hand with Human-Level Joint Structure Enabled by Linkage-Based Actuation",
   "zh": "基于连杆驱动的全关节人形灵巧手",
   "org": "浙江大学机械工程学院、浙江大学机器人研究院",
   "prob": "现有灵巧手难以同时实现类人关节结构、高负载、机械可靠性和简单可控性。",
   "inno": "Torch-Hand 采用全连杆传动，具有 21 自由度，其中拇指具有五自由度和多轴腕掌关节；所有执行器位于手掌内，并通过机械解耦减少非目标关节联动。",
   "reuse": null,
   "short": "连杆驱动人形灵巧手",
   "slug": "linkage-driven-humanoid-dexterous-hand",
   "abstract": "Humanoid dexterous hands are a core component of embodied intelligence, enabling robots to translate high-level intent into rich and adaptive physical interaction. However, existing designs remain constrained by a persistent trade-off among anthropomorphic kinematic complexity, mechanical robustness, load capacity and control simplicity, limiting dexterous manipulation in unstructured environments. Here we present Torch-Hand, a fully articulated humanoid dexterous hand that replicates the joint structure and motion hierarchy of the human hand with 21 degrees of freedom (DOF), comprising a highly articulated five-DOF thumb and four DOF in each of the remaining fingers. The differentiated thumb integrates multi-axis CMC motion to provide broad opposition and flexible contact orientation relative to the four fingers. Unlike tendon-driven or hybrid actuation approaches, Torch-Hand adopts an entirely linkage-based transmission architecture. By integrating screw-linkage actuation, the hand achieves high grasping force and substantial payload capacity while maintaining a compact anthropomorphic form. Its actuation architecture is optimized for control tractability, with all actuators integrated within the palm and arranged in a serial-robot-like configuration that establishes a direct correspondence between motor motion and joint kinematics. This architecture mechanically decouples functional DOF to reduce unintended inter-joint motion propagation, while allowing independently addressable DOF to be actively coordinated into coupled multi-joint motion patterns during complex manipulation. Comprehensive experiments assess reachable workspace, coordinated fingertip trajectories, thumb opposition, 33-grasp taxonomy coverage, grasping force, payload capacity and task-level manipulation. These results establish Torch-Hand as a robust hardware foundation for dexterous embodied manipulation.",
   "comments": "",
   "authors": "Li, Jihao, Wang, Jianguo, Guo, Haotian, Wang, Yanzhe, Lu, Guodong, Dong, Huixu",
   "links": []
  },
  "2610.35670": {
   "id": "2610.35670",
   "title": "Continuous-Time Safe Planning and Control for Agile Quadrupeds",
   "zh": "面向敏捷四足机器人的连续时间安全规划与控制",
   "org": "Amazon Fulfillment Technologies & Robotics、爱荷华大学、麻省理工学院",
   "prob": "离散时刻避碰不能排除两个规划点之间的碰撞，简化为单轮车模型又会丢失四足机器人的横向运动能力。",
   "inno": "保留带朝向耦合速度边界的完整 SE(2) 模型，以分段 Bernstein 多项式表示轨迹；利用凸包性质在整个连续时间区间上验证避碰和控制边界。",
   "reuse": null,
   "short": "连续时间安全规划控制",
   "slug": "continuous-time-safe-planning-agile-quadrupeds",
   "abstract": "Agile quadruped navigation in dynamic environments demands trajectories that are both dynamically feasible and provably collision-free. Most real-time planners achieve tractability through a convex reduction that requires simplified kinematic models, discarding motion that the robot can readily execute, and certify safety only at discretization points, leaving open possible inter-sample collisions with moving obstacles. We present a real-time motion planning framework that retains the fully-actuated $SE(2)$ model of a quadruped, with its anisotropic, heading-coupled velocity bounds, and accepts the resulting nonconvexity. By parameterizing trajectories as composite Bernstein polynomials, we exploit the convex hull property to certify collision avoidance and the input bounds over the entirety of each planning horizon. Solved at online replanning rates, the optimization runs in closed loop, commanding body velocities to the locomotion controller. We validate the approach in simulation against ablations and external baselines, and in hardware experiments on a quadruped navigating among unknown moving obstacles, recovering both lateral and heading-coupled motion that simplified models exclude.",
   "comments": "",
   "authors": "MacLin, Gage, Peterson, Mason, Tabasso, Camilla, Cichella, Venanzio, Migdal, Joshua, Brady, Tye, Padir, Taskin",
   "links": []
  },
  "2610.36537": {
   "id": "2610.36537",
   "title": "A Trainable CPG Framework for Quadruped Locomotion",
   "zh": "面向四足运动的可训练中央模式发生器框架",
   "org": "浙江大学",
   "prob": "多步态控制通常为每种步态和转换分别训练策略，难以获得可解释且连续的统一控制。",
   "inno": "八神经元对称 CPG 生成节律，共享腿部执行网络将节律与轨迹参数转换为关节参考；一个反馈策略只调节六个物理可解释变量，即可实现五种步态及全部方向转换。",
   "reuse": null,
   "short": "可训练 CPG",
   "slug": "trainable-cpg-quadruped-locomotion",
   "abstract": "We present CPG-ENN, a trainable quadruped locomotion framework that combines explicit neural coordination with learned motion realization. A symmetry-based eight-neuron central pattern generator (CPG) supplies gait rhythms, while a pretrained, leg-shared execution neural network (ENN) maps rhythmic signals and trajectory parameters to joint-angle references without online inverse kinematics. A single feedback policy modulates six physically interpretable trajectory variables. The controller tracks planar velocity and yaw-rate commands across walk, trot, pace, bound, and pronk without gait-specific learned controllers or transitionspecific rewards. Simulation yields 99.39% transition success across all 20 directed gait pairs and all source-target speed combinations from {0.5, 1.0, 1.5, 2.0} m/s. Ablations show high cross-gait transition feasibility without transition exposure; exposure mainly refines gait acquisition and velocity transients. Comparisons with AllGaits and Walk These Ways demonstrate accurate gait-speed regulation, and Unitree Go2 experiments validate five gaits and their transitions.",
   "comments": "",
   "authors": "Liu, Xiyan, Liu, Yide, Mao, Guoyong, Qu, Shaoxing",
   "links": []
  },
  "2610.36599": {
   "id": "2610.36599",
   "title": "Camera-to-Quadruped Robot Pose Estimation from Sequential Images",
   "zh": "基于序列图像的外部相机—四足机器人姿态估计",
   "org": "韩国全南大学",
   "prob": "外部单目相机观察四足机器人时，腿部对称、自遮挡和关节持续运动会造成关键点匹配歧义。",
   "inno": "QuadPose 从视频估计六自由度基座位姿和 12 个关节角，并构建首个外部相机四足姿态基准 QuadSyn；推理速度约 33 FPS。",
   "reuse": null,
   "short": "相机到四足位姿估计",
   "slug": "camera-to-quadruped-pose-estimation",
   "abstract": "Quadruped robots are increasingly deployed in hazardous environments such as disaster response and facility inspection, where external-camera monitoring is essential for safe operation. This setting is challenging because quadrupeds are articulated systems with time-varying leg joints, leg symmetry, and frequent self-occlusions, which make keypoint correspondence ambiguous. Moreover, no public benchmark exists, making quantitative comparison difficult. To address these challenges, we propose QuadPose, which estimates the full-body 18-DoF pose (6-DoF base pose + 12-DoF joint angles) from monocular video sequences, and introduce QuadSyn, the first camera-to-quadruped robot pose benchmark. On QuadSyn, QuadPose achieves 92.18% PCK@2.5px, 95.54% PCK-AUC, 85.94% ADD@20cm, 66.98% ADD-AUC, and 3.65 degree mean joint-angle error at approximately 33 FPS.",
   "comments": "",
   "authors": "Hwang, Seobin, Beom, Ji-Seong, Cho, Yeong-Jun",
   "links": []
  },
  "2610.36852": {
   "id": "2610.36852",
   "title": "When Black-Box Robot Policies Fail: Online Sensor Diagnosis and Repair with a Transparent Surrogate",
   "zh": "黑盒机器人策略失效时：基于透明代理的在线传感器诊断与修复",
   "org": "慕尼黑工业大学、Siemens Mobility",
   "prob": "冻结的黑盒运动策略遇到传感器故障时，无法通过访问策略内部结构定位和修复错误输入。",
   "inno": "用显式分离周期步态与传感器修正项的透明代理近似策略，再利用代理梯度诊断并修正输入；在 Go2 上将故障条件成功数由 259/640 提高到 382/640。",
   "reuse": null,
   "short": "黑箱策略传感器诊断修复",
   "slug": "blackbox-policy-sensor-diagnosis-repair",
   "abstract": "Robot foundation models may generalize across tasks and embodiments, but their actions still depend on sensors that can fail. When the policy is frozen or inaccessible, can an external monitor diagnose and repair corrupted inputs without access to its internals? We test a structured surrogate that separates periodic gait from sensor-dependent corrections to guide input repair around a frozen locomotion policy. On simulated Go2, offline diagnosis raises walking success from 259/640 to 429/640 episodes; online repair reaches 382/640, with no triggers in 64 held-out fault-free episodes. A matched boosted-tree surrogate predicts clean actions similarly well but yields no detected repair improvement over no repair. Separate fits improve walking on Go1 and ANYmal C, although phase-aware gradients perform better. Evidence remains limited to one policy and a partly screened simulated fault suite; robot foundation model and hardware validation remain open.",
   "comments": "",
   "authors": "Ziegler, Marc, Özen, Batu, Mentes, Sami, Waschulzik, Thomas, Knoll, Alois",
   "links": []
  },
  "2610.38388": {
   "id": "2610.38388",
   "title": "ELEVATE: Agentic Controller Acquisition from Task Failure for Quadrupedal Manipulation",
   "zh": "ELEVATE：从任务失败中自主获取四足操作控制器",
   "org": "卡内基梅隆大学、三菱电机研究实验室",
   "prob": "任务规划智能体只能组合现有控制器；当任务需要新的全身物理能力时，修改高层程序无法解决问题。",
   "inno": "根据持续失败识别能力缺口，由编码智能体自动构造奖励、课程和终止条件来训练新 RL 控制器；新技能被加入控制器库并与已有技能组合，且无需实机微调即可部署。",
   "reuse": "wiki/entities/paper-elevate.md"
  },
  "2610.39267": {
   "id": "2610.39267",
   "title": "RAIBO-Wagon: Real-Time Navigation System for a Legged Robot Towing a Wagon",
   "zh": "RAIBO-Wagon：四足机器人牵引货车的实时导航系统",
   "org": "韩国科学技术院机器人与人工智能实验室",
   "prob": "四足机器人牵引欠驱动货车时，组合系统动力学复杂，在狭窄空间中可行路径范围很小。",
   "inno": "条件流匹配模型为 MPPI 提供初始轨迹，再由基于学习前向动力学模型的 MPPI 迭代修正；全板载系统可牵引接近机器人自重的 40 kg 负载。",
   "reuse": null,
   "short": "RAIBO-Wagon",
   "slug": "raibo-wagon-legged-towing-navigation",
   "abstract": "Legged robots have demonstrated advanced capabilities across a wide range of real-world tasks, yet much of this progress concerns the robot controlling only itself. Once the robot is coupled to an external object -- particularly an underactuated one -- the dynamics of the combined assembly become more complex. Towing a wagon is a common practical scenario, where coupled dynamics make reliable navigation challenging. This difficulty is amplified in confined spaces, where collision-free navigation requires finding a narrow set of feasible paths under strict real-time constraints. To address this, we propose a navigation framework that combines Conditional Flow Matching (CFM) with Model Predictive Path Integral (MPPI) control over a learned Forward Dynamics Model (FDM). The CFM model mitigates the initialization problem of MPPI and accelerates its convergence, while MPPI with the FDM recovers feasible solutions beyond the CFM model's training distribution by iterative refinement. We integrate the framework into a full-stack navigation system for RAIBO-Wagon, a quadruped towing a custom wagon, and validate it in real-world outdoor experiments on 15 m-long courses through confined spaces, with all sensing and computation performed fully onboard. The system achieves an 85% success rate across 40 trials, towing a total load of up to 40 kg -- nearly 100% of RAIBO's body weight.",
   "comments": "",
   "authors": "Yang, Insung, Kim, Eunchae, Hur, Jungwoo, Jeon, Seunghun, Choi, Donghyuk, Hwangbo, Jemin",
   "links": []
  },
  "2610.40248": {
   "id": "2610.40248",
   "title": "LLM-guided Planar Loco-Manipulation for Quadruped Robots",
   "zh": "大语言模型引导的四足平面移动操作",
   "org": "米兰理工大学、Robotics and AI Institute、Flexion Robotics",
   "prob": "障碍物可移动时，机器人需要联合选择操作技能、目标物体和几何作用点，而非只规划无碰撞路径。",
   "inno": "LoRA 微调的语言任务规划器将语言指令和结构化场景转换为可变长度参数化动作；底层端到端全身推动策略在仿真中达到 93.3% 成功率，并在实机 45 次实验中完成 34 次端到端任务。",
   "reuse": null,
   "short": "LLM 引导平面移动操作",
   "slug": "llm-guided-planar-loco-manipulation-quadruped",
   "abstract": "Robotic navigation typically assumes a collision-free route to the goal exists; in cluttered environments with movable obstacles, however, reaching the goal may require physical interaction. Classical solutions encode these behaviors through geometric rules and task-specific logic, while learning-based approaches often constrain the decision space to low-dimensional motion commands or predefined action sets. We instead consider a broader decision space, jointly selecting the skill, relevant object, and geometric target. Our language-conditioned task planner, fine-tuned with LoRA on synthetic data, maps natural-language tasks and a structured scene representation to variable-length sequences of parameterized actions, executed by low-level reinforcement-learning policies. The planner learns the geometric relations underlying interactive navigation and outperforms a classical planner and a prompted frontier model on interactive navigation tasks. Comparisons with models trained from scratch indicate that composition does not require pretraining when the representation is structured, whereas pretraining yields more reliable geometric targets. At the low level, we introduce an end-to-end whole-body planar pushing policy that achieves a 93.3% success rate in simulation, versus 83.5% for a hierarchical baseline, while reducing completion time by more than 60%. On hardware, the hierarchy yields the correct skill composition in 44/45 trials and end-to-end success in 34/45. Project website: this https URL",
   "comments": "",
   "authors": "Jetti, Georges, Khayyat, Michael, Sleiman, Jean-Pierre, Arrigoni, Stefano, Braghin, Francesco",
   "links": []
  },
  "2610.40364": {
   "id": "2610.40364",
   "title": "Is It the Curriculum or Just the Easy Start? Benchmarking Unsupervised Environment Design on Quadrupeds",
   "zh": "究竟是课程有效，还是起点简单？四足无监督环境设计基准",
   "org": "弗罗茨瓦夫理工大学、Solvro 学生研究团队",
   "prob": "无监督环境设计方法主要在二维迷宫上比较，而四足地形课程使用不同机器人、预算和指标，结论无法直接对照。",
   "inno": "在五种三维四足机器人上，以相同步数预算比较 14 种环境设计、域随机化和地形课程；结果表明没有一种方法在所有机器人及测试地形上占优，腿式机器人常用的成功率驱动课程进入最优组的次数最多。",
   "reuse": null,
   "short": "无监督环境设计基准",
   "slug": "unsupervised-environment-design-quadruped-benchmark",
   "abstract": "Unsupervised environment design (UED) methods such as PLR, ACCEL, SFL and PAIRED are compared mostly on 2D mazes, grid worlds and BipedalWalker, while terrain curricula for legged robots are hand-designed and evaluated against a few baselines. We present a benchmark that runs fourteen UED, domain-randomisation and robotics curricula under one protocol on five 3D quadrupeds in MuJoCo MJX: the closed-chain W01-TEK and the open-chain Unitree Go1 and Go2, Google Barkour vB and Boston Dynamics Spot. Levels are vectors of eight interpretable terrain parameters, and all curricula drive the same PPO student at matched environment-step budgets. The evaluation reports success on random and held-out structured terrains, interquartile means with stratified bootstrap intervals, capability envelopes and a level-by-level domination comparison. We compare methods within each robot and report, per level set, the top group of methods not significantly below the best. No method is in the top group on every robot and level set. Methods with easy or low-difficulty training levels, curated or not, fill the top groups on in-distribution random levels of Go1 and Barkour, and of W01-TEK apart from PAIRED+BiBC. The success-driven terrain curriculum from legged-robot practice is in a top group most often, domain randomisation on uniform levels in none, and PAIRED in none outside Spot, where the fourteen methods lie within 3.8 points on random levels of difficulty 0.25. On three robots the leader on in-distribution random levels is not the leader on held-out terrain. We collect the results into a per-robot guide to choosing a curriculum.",
   "comments": "",
   "authors": "Gilewicz, Jakub, Galus, Dominik, Wojdas, Piotr",
   "links": []
  },
  "2610.40285": {
   "id": "2610.40285",
   "title": "MUSBRIC: LEGO-Compatible Pneumatic Muscle Bricks for Reconfigurable Musculoskeletal Robots",
   "zh": "MUSBRIC：用于可重构肌肉骨骼机器人的乐高兼容气动肌肉模块",
   "org": "清华大学、清华大学医学院",
   "prob": "肌肉骨骼机器人改变肌肉连接位置或肢体结构时，通常需要重新加工执行器和骨架。",
   "inno": "将纤维增强伸长型气动肌肉封装成乐高兼容模块，并建立对应 MuJoCo 模型；通过 15 种腿部构型、肘关节、灵巧手和四足行走平台验证可重构设计。",
   "reuse": null,
   "short": "MUSBRIC",
   "slug": "musbric-pneumatic-muscle-bricks",
   "abstract": "Musculoskeletal robots combine compliant actuators with articulated skeletons, but exploring alternative configurations often requires repeated fabrication. We present MUSBRIC, a modular hardware platform and accompanying MuJoCo-based simulator for reconfigurable musculoskeletal robots. LEGO-compatible endcaps turn fiber-reinforced extensile pneumatic artificial muscles into reusable, plug-and-play muscle bricks that connect directly to standard LEGO components. Experimentally identified actuator behavior supports corresponding musculoskeletal simulations. An elbow-joint prototype validates load-dependent flexion and extension, showing that pulley-guided routing enables reversible motion and a mechanical stop stabilizes the straight posture. A robotic hand demonstrates compliant grasping and multi-joint construction. Matched hardware and simulation sweeps across 15 leg configurations demonstrate systematic exploration of attachment geometry, while a quadruped assembled from four legs demonstrates walking. Together, these demonstrations span mechanism analysis, configuration exploration, and functional robot construction within a single reconfigurable system. The hardware designs and simulation code will be released as open source soon to facilitate reproduction and further development.",
   "comments": "",
   "authors": "Wang, Qiyao, Zhu, Yinglei, Zhou, Liang, Gong, Wenzhe, Liu, Xin-Jun, Zhao, Huichan",
   "links": []
  },
  "2610.35852": {
   "id": "2610.35852",
   "title": "Differentiable Contact Dynamics for Legged Locomotion: Policy Learning and Contact-Implicit Trajectory Optimization",
   "zh": "面向腿式运动的可微接触动力学：策略学习与接触隐式轨迹优化",
   "org": "韩国科学技术院",
   "prob": "硬接触的精确梯度只描述当前接触模式，无法提供向分离、粘着或滑动等相邻模式变化的有效一阶信息。",
   "inno": "前向过程仍使用硬接触保证轨迹可行，仅在反向传播中以摩擦互补问题的中心路径松弛替换梯度；10 万样本、不到 2 分钟即可训练出能在碎石和湿滑地面实机运行的策略。",
   "reuse": "wiki/entities/paper-differentiable-contact-legged.md"
  },
  "2610.36270": {
   "id": "2610.36270",
   "title": "Learning Multi-Phase Robot Loco-Manipulation via Phase-Conditioned Policy Optimization",
   "zh": "基于阶段条件策略优化的多阶段机器人移动操作学习",
   "org": "苏黎世联邦理工学院机器人系统实验室、Robotics and AI Institute",
   "prob": "下蹲、抓取、移动和放置组成的长时任务探索困难，而且很多方法依赖人形动作数据，难以跨机器人平台。",
   "inno": "扩展 PPO，引入阶段条件 Critic、阶段门控奖励和多阶段重置课程；通过分层控制分离底层平衡与高层阶段执行，同一方法部署到 Unitree G1 和带机械臂 Spot。",
   "reuse": "wiki/entities/paper-phase-conditioned-loco-manipulation.md"
  },
  "2610.37199": {
   "id": "2610.37199",
   "title": "A Bi-level Co-design Framework for Lunar Monoped with 2D Jump and Landing Control",
   "zh": "面向月球单足机器人二维跳跃与着陆的双层协同设计",
   "org": "印度空间科学与技术学院、U R Rao 卫星中心",
   "prob": "月球低重力会延长腾空时间并减小接触力，使腿部尺寸和跳跃控制器高度耦合。",
   "inno": "外层差分进化搜索腿段长度，内层分别使用 QP 任务空间控制和直接配点轨迹优化评估设计；进一步加入角动量调节和 LQR 落地稳定控制。",
   "reuse": null,
   "short": "月球单足协同设计",
   "slug": "lunar-monoped-bilevel-codesign",
   "abstract": "Legged robots for lunar exploration must be designed for low gravity, where reduced contact forces and long flight phases make leg morphology and controller choice interdependent. We study this for a planar monoped in lunar gravity ($1.625$ m/s$^2$) using a bi-level co-design framework: a differential evolution outer loop searches over link lengths, and each candidate is evaluated on a vertical jump by either a quadratic program based task-space controller (QP-TSC) or a direct-collocation based trajectory optimization. The two controllers yield different optimal link lengths, and trajectory optimization outperforms the QP-TSC across most of the design space, with 17% lower Cost of Transport (COT) at their respective optima. We then extend to a 2D jump, where trajectory optimization with centroidal angular momentum regulation produces stable in-place and forward jumps, and an LQR controller absorbs impact and stabilizes the robot after touchdown. This is demonstrated on a baseline morphology with both links 0.3 m long.",
   "comments": "",
   "authors": "Ramkumar, Sanjay, Kumar, Suraj",
   "links": []
  },
  "2610.37203": {
   "id": "2610.37203",
   "title": "Remember Where You Walk: Learning Spatial Memory for End-to-End Legged Robot Exploration",
   "zh": "记住走过的地方：面向端到端腿式探索的空间记忆学习",
   "org": "达姆施塔特工业大学、Robotics Institute Germany、德国人工智能研究中心、hessian.AI",
   "prob": "LSTM 等非结构化记忆难以在长时间后保留已经离开视野的地形与空间布局。",
   "inno": "提出局部空间矩阵记忆，将任务信息写入与机器人位置对应的存储单元；在迷宫探索覆盖率相近时，地图重建误差较循环和有限历史基线降低约三倍。",
   "reuse": "wiki/entities/paper-rememberwhereyouwalk.md"
  },
  "2610.37806": {
   "id": "2610.37806",
   "title": "TACOS: Terrain-Aware COntact Supervision for Visual Footstep Planning",
   "zh": "TACOS：面向视觉落脚规划的地形感知接触监督",
   "org": "拉瓦尔大学 Northern Robotics Laboratory、牛津大学 Mobile Robotics Group",
   "prob": "视觉落脚模型缺少可靠的自监督标签，仅根据滑移、冲击或人工经验定义不稳定性，难以同时描述泥地、植被和几何障碍。",
   "inno": "根据腿式状态估计因子图残差定义连续落脚不稳定分数，并投影到落脚前图像训练稠密预测器；在 50 米森林实机路线中可识别泥地和植被，同时绕开树枝。",
   "reuse": "wiki/entities/paper-tacos.md"
  },
  "2610.37865": {
   "id": "2610.37865",
   "title": "Echo: Scaling Visual Reinforcement Learning with Off-Policy Experience",
   "zh": "Echo：利用离策略经验扩展视觉强化学习",
   "org": "哥伦比亚大学",
   "prob": "直接视觉 RL 需要持续渲染图像并执行视觉网络，计算成本高；Teacher–Student 两阶段蒸馏又存在模仿误差。",
   "inno": "让状态输入辅助策略反向模仿持续更新的视觉策略，并采集无需渲染的廉价经验；这些经验通过共享特权 Critic 参与离策略价值学习，直接更新视觉策略。该工作目前为初步仿真实验。",
   "reuse": null,
   "short": "Echo（离策略视觉 RL）",
   "slug": "echo-visual-rl-off-policy-experience",
   "abstract": "Scaling visual reinforcement learning (RL) with GPU-accelerated simulation is computationally expensive due to rendering and visual-policy inference costs. Consequently, two-stage teacher--student distillation is commonly used to train visual policies, but introduces an imitation gap. In this preliminary study, we introduce \\echomethod{}, \\textbf{a single-stage off-policy RL framework that scales experience collection without proportionally increasing visual computation}. Through \\textbf{inverse distillation}, a state-based auxiliary policy imitates the evolving visual policy and collects inexpensive, unrendered experience. Echo leverages this experience for off-policy value learning, directly improving the visual policy through a shared privileged critic. We instantiate \\echomethod{} with FastTD3 and evaluate it on three visual control tasks and two dexterous manipulation tasks in Genesis. Relative to \\sdpgmethod{} under the evaluated configurations, Echo reduces time to reference returns by an average of \\textbf{60\\% across the three locomotion tasks} and \\textbf{56\\% across both hand tasks}, while improving peak mean returns by \\textbf{62\\% and 12\\%}, respectively. Both hand tasks also achieve higher mean returns at matched simulator-transition budgets. These initial results suggest that scaling off-policy experience is a promising approach to efficient direct visual RL.",
   "comments": "",
   "authors": "Khandate, Gagan",
   "links": []
  },
  "2610.38502": {
   "id": "2610.38502",
   "title": "OmniQuad: A Wheeled-Legged Hybrid Robot with Omnidirectional Wheels",
   "zh": "OmniQuad：采用全向轮的轮腿混合机器人",
   "org": "比萨大学、Enrico Piaggio 研究中心",
   "prob": "普通主动轮腿机器人在狭窄空间内侧移或保持固定朝向移动时受到轮子非完整约束。",
   "inno": "设计安装麦克纳姆轮的轮式足端，并与 EM-Act 腿部执行器组成 OmniQuad；机器人可在不改变机身朝向的情况下完成前后、横向和斜向移动，并保留腿式越障能力。",
   "reuse": null,
   "short": "OmniQuad",
   "slug": "omniquad-wheeled-legged-omnidirectional",
   "abstract": "Combining wheeled and legged locomotion within a single robotic platform offers the potential to harness the advantages of both systems. The development of such hybrid systems remains an active area of research. While legged robots exhibit remarkable adaptability for traversing uneven and rugged terrains, their relatively low speed limits their practicality in indoor environments. Conversely, wheeled robots demonstrate higher efficiency on flat surfaces but often encounter difficulties when navigating obstacles such as steps. This paper introduces OmniQuad, a novel hybrid robot that integrates mecanum wheels with legged locomotion to exploit the benefits of both modalities. Mecanum wheels provide omnidirectional mobility, enhancing the robot's manoeuvrability by enabling agile movement in confined spaces and the ability to maintain a consistent orientation during task execution. In this work, we first present the design of a custom wheeled-foot featuring mecanum wheels. Subsequently, we describe the complete robot assembly, comprising these wheeled feet and EM-Act actuators. Finally, we detail the control scheme developed to govern the OmniQuad and evaluate its performance through experimental trials conducted in both indoor and outdoor environments.",
   "comments": "",
   "authors": "Iotti, Francesco, Ranjan, Alok, Angelini, Franco, Garabini, Manolo",
   "links": []
  }
 },
 "lists": {
  "hq": [
   {
    "id": "2610.34740",
    "n": 1,
    "star": false,
    "sec": "人形机器人"
   },
   {
    "id": "2610.35075",
    "n": 2,
    "star": false,
    "sec": "人形机器人"
   },
   {
    "id": "2610.35862",
    "n": 3,
    "star": false,
    "sec": "人形机器人"
   },
   {
    "id": "2610.36452",
    "n": 4,
    "star": false,
    "sec": "人形机器人"
   },
   {
    "id": "2610.36507",
    "n": 5,
    "star": true,
    "sec": "人形机器人"
   },
   {
    "id": "2610.36546",
    "n": 6,
    "star": true,
    "sec": "人形机器人"
   },
   {
    "id": "2610.36615",
    "n": 7,
    "star": false,
    "sec": "人形机器人"
   },
   {
    "id": "2610.36825",
    "n": 8,
    "star": true,
    "sec": "人形机器人"
   },
   {
    "id": "2610.36859",
    "n": 9,
    "star": false,
    "sec": "人形机器人"
   },
   {
    "id": "2610.37801",
    "n": 10,
    "star": false,
    "sec": "人形机器人"
   },
   {
    "id": "2610.37974",
    "n": 11,
    "star": false,
    "sec": "人形机器人"
   },
   {
    "id": "2610.38027",
    "n": 12,
    "star": false,
    "sec": "人形机器人"
   },
   {
    "id": "2610.39044",
    "n": 13,
    "star": false,
    "sec": "人形机器人"
   },
   {
    "id": "2610.39327",
    "n": 14,
    "star": false,
    "sec": "人形机器人"
   },
   {
    "id": "2610.40003",
    "n": 15,
    "star": false,
    "sec": "人形机器人"
   },
   {
    "id": "2610.35462",
    "n": 16,
    "star": false,
    "sec": "人形机器人"
   },
   {
    "id": "2610.36789",
    "n": 17,
    "star": false,
    "sec": "人形机器人"
   },
   {
    "id": "2610.35670",
    "n": 1,
    "star": false,
    "sec": "四足机器人"
   },
   {
    "id": "2610.36537",
    "n": 2,
    "star": false,
    "sec": "四足机器人"
   },
   {
    "id": "2610.36599",
    "n": 3,
    "star": false,
    "sec": "四足机器人"
   },
   {
    "id": "2610.36852",
    "n": 4,
    "star": false,
    "sec": "四足机器人"
   },
   {
    "id": "2610.38388",
    "n": 5,
    "star": true,
    "sec": "四足机器人"
   },
   {
    "id": "2610.39267",
    "n": 6,
    "star": false,
    "sec": "四足机器人"
   },
   {
    "id": "2610.40248",
    "n": 7,
    "star": false,
    "sec": "四足机器人"
   },
   {
    "id": "2610.40364",
    "n": 8,
    "star": true,
    "sec": "四足机器人"
   },
   {
    "id": "2610.40285",
    "n": 9,
    "star": false,
    "sec": "四足机器人"
   },
   {
    "id": "2610.35852",
    "n": 1,
    "star": true,
    "sec": "轮足与通用腿式机器人"
   },
   {
    "id": "2610.36270",
    "n": 2,
    "star": false,
    "sec": "轮足与通用腿式机器人"
   },
   {
    "id": "2610.37199",
    "n": 3,
    "star": false,
    "sec": "轮足与通用腿式机器人"
   },
   {
    "id": "2610.37203",
    "n": 4,
    "star": false,
    "sec": "轮足与通用腿式机器人"
   },
   {
    "id": "2610.37806",
    "n": 5,
    "star": true,
    "sec": "轮足与通用腿式机器人"
   },
   {
    "id": "2610.37865",
    "n": 6,
    "star": true,
    "sec": "轮足与通用腿式机器人"
   },
   {
    "id": "2610.38502",
    "n": 7,
    "star": true,
    "sec": "轮足与通用腿式机器人"
   }
  ]
 }
}""")
PAPERS: dict[str, dict] = DATA["papers"]
LISTS: dict[str, list[dict]] = DATA["lists"]


def h1_of(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return path.stem


def node_rel(p: dict, base: str) -> str:
    """相对 base 目录给出 wiki 节点链接。"""
    stem = Path(p["reuse"]).stem if p["reuse"] else f"paper-{p['slug']}"
    return f"{base}{stem}.md"


def sec_info(sec: str, blog_key: str) -> tuple[list[str], list[str], list[str]]:
    if blog_key == "hq":
        _, tags, rel = SEC_META[sec]
        return tags, rel, ["RL", "WBC"] if "legged" not in tags else ["RL", "MPC"]
    tags = ["paper", *MAN_TAGS[sec]]
    abbr = ["VLA", "IL"]
    if "world-action-model" in tags:
        abbr = ["WAM", "VLA"]
    elif "reinforcement-learning" in tags:
        abbr = ["RL", "VLA"]
    return tags, MAN_REL, abbr


def open_status(p: dict) -> str:
    links = p.get("links", [])
    gh = [u for u in links if "github.com/" in u and ".github.io" not in u]
    if gh:
        return f"arXiv 摘要页/评论给出 GitHub 链接（{gh[0]}；仅核对到链接存在，未核验仓库内容）"
    if links:
        return f"arXiv 摘要页/评论给出项目页或外链（{links[0]}）；未发现代码仓库链接"
    return "arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布）"


def abstract_excerpt(text: str, n: int = 2) -> str:
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(sents[:n])


def abbrev_table(keys: list[str]) -> str:
    rows = ["| 缩写 | 英文全称 | 简要说明 |", "|------|----------|----------|"]
    rows += [f"| {k} | {ABBREV[k][0]} | {ABBREV[k][1]} |" for k in keys]
    return "\n".join(rows)


def appearances(aid: str) -> list[tuple[str, dict]]:
    return [(k, e) for k, lst in LISTS.items() for e in lst if e["id"] == aid]


def display_name(p: dict) -> str:
    """标题：中文标题已含方法名时不再重复前缀。"""
    zh = p["zh"]
    return zh if p["short"].split()[0].lower() in zh.lower() else f"{p['short']}：{zh}"


def render_entity(p: dict) -> str:
    aid, slug = p["id"], p["slug"]
    name = display_name(p)
    apps = appearances(aid)
    key, first = apps[0]
    tags, rel, abbr = sec_info(first["sec"], key)
    blogs = [BLOGS[k] for k, _ in apps]
    ov = [b["overview"] for b in blogs]
    src_paper = f"sources/papers/{slug}_arxiv_{aid.replace('.', '_')}.md"
    rel_all = [f"../overview/{o}.md" for o in ov] + rel
    ol = p["inno"].replace('"', "'")
    summary = f"{name}（arXiv:{aid}）：{ol[:100]}"
    blog_lines = "\n".join(f"  - ../../sources/blogs/{b['file']}" for b in blogs)
    rel_lines = "\n".join(f"  - {r}" for r in rel_all)
    comment = p["comments"] or "（无）"
    identity = ""
    for u in p.get("links", []):
        if "github.com/" in u and ".github.io" not in u and "code:" not in identity:
            identity += f"code: {u.replace('http://', 'https://')}\n"
        elif ".github.io" in u and "project:" not in identity:
            identity += f"project: {u.replace('http://', 'https://')}\n"
    return f"""---
type: entity
tags:
{chr(10).join(f"  - {t}" for t in tags)}
status: complete
updated: {TODAY}
arxiv: "{aid}"
project_id: {slug}
{identity}related:
{rel_lines}
sources:
  - ../../{src_paper}
{blog_lines}
summary: "{summary}"
---

# {name}

**{p["short"]}**（*{p["title"]}*，[arXiv:{aid}](https://arxiv.org/abs/{aid})）收录于 senlanke「具身运控lab」{WEEK} 前沿论文动态（{" / ".join(b["label"] for b in blogs)}）。本页依据**周报条目 + arXiv 摘要页**整理，未逐页核对正文 PDF；指标与结论以论文为准。

## 一句话定义

**{name}——{p["inno"]}**

## 英文缩写速查

{abbrev_table(abbr)}

## 为什么重要

- 周报所述待解决的问题：{p["prob"]}

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [{aid}](https://arxiv.org/abs/{aid}) · [PDF](https://arxiv.org/pdf/{aid}) |
| **作者（arXiv）** | {p["authors"]} |
| **机构（周报所述，未逐条核实）** | {p["org"]} |
| **arXiv 评论栏** | {comment} |
| **开源** | {open_status(p)}（{TODAY}） |
| **周报所述创新点** | {p["inno"]} |

### arXiv 摘要节选（英文原文）

> {abstract_excerpt(p["abstract"])}

## 源码运行时序图

**不适用**：{open_status(p)}，无可对齐的官方代码流程。

## 实验与评测

- 周报与摘要给出的数字/场景已写入上表「创新点」与摘要节选；其余实验细节、基线与成功率定义请回到论文正文。
- 读法：先对齐任务设定、传感器与成功定义（仿真/真机），再解读 headline 数字，勿跨论文直接比较。

## 与其他工作对比

| 维度 | 读法 |
|------|------|
| **同周对照** | 见 {"、".join(f"[{b['label']}周报技术地图](../overview/{b['overview']}.md)" for b in blogs)}，按方向浏览同期论文 |
| **开源状态** | {open_status(p)} |

## 结论

**{name}。**

1. 本页是入库级摘要，复现前请核对论文正文、附录与代码仓库。
2. 指标须连同实验条件解读。
3. 关注 arXiv 版本更新与代码发布。

## 关联页面

{chr(10).join(f"- [{r.split('/')[-1].removesuffix('.md')}]({r})" for r in rel_all)}

## 参考来源

- [{Path(src_paper).name}](../../{src_paper})
{chr(10).join(f"- [{b['file']}](../../sources/blogs/{b['file']})" for b in blogs)}
- [arXiv:{aid}](https://arxiv.org/abs/{aid})

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/{aid})
"""


def render_source(p: dict) -> str:
    aid = p["id"]
    blogs = [BLOGS[k] for k, _ in appearances(aid)]
    links = "\n".join(f"- **外链：** <{u}>" for u in p.get("links", []))
    return f"""# {p["short"]}（arXiv:{aid}）

> 来源归档（paper）

- **标题：** {p["title"]}
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/{aid}>
- **PDF：** <https://arxiv.org/pdf/{aid}>
- **作者：** {p["authors"]}
- **arXiv 评论栏：** {p["comments"] or "（无）"}
{links}
- **入库日期：** {TODAY}
- **一句话说明：** {p["zh"]}。

## 开源状态

- {open_status(p)}（{TODAY}，仅核对 arXiv 摘要页）

## 核心摘录

1. **策展来源：** {"；".join(f"[{b['title']}](../blogs/{b['file']})" for b in blogs)}
2. **周报所述问题：** {p["prob"]}
3. **周报所述创新点：** {p["inno"]}
4. **arXiv 摘要（英文节选）：** {abstract_excerpt(p["abstract"], 3)}

**对 wiki 的映射**

- [paper-{p["slug"]}](../../wiki/entities/paper-{p["slug"]}.md)
"""


def table_rows(key: str, base: str) -> dict[str, list[str]]:
    by_sec: dict[str, list[str]] = {}
    for e in LISTS[key]:
        p = PAPERS[e["id"]]
        path = ROOT / (p["reuse"] if p["reuse"] else f"wiki/entities/paper-{p['slug']}.md")
        name = h1_of(path) if p["reuse"] else display_name(p)
        star = " ★" if e["star"] else ""
        tag = "复用" if p["reuse"] else "新建"
        by_sec.setdefault(e["sec"], []).append(
            f"| {e['n']} | [{name}]({node_rel(p, base)}){star} | [{e['id']}](https://arxiv.org/abs/{e['id']}) | {tag} |"
        )
    return by_sec


def render_blog(key: str) -> str:
    b = BLOGS[key]
    n_new = sum(1 for e in LISTS[key] if not PAPERS[e["id"]]["reuse"])
    n_all = len(LISTS[key])
    parts = []
    for sec, rows in table_rows(key, "../../wiki/entities/").items():
        parts.append(
            f"### {sec}（{len(rows)}）\n\n| # | 节点 | arXiv | 状态 |\n|---|------|-------|------|\n"
            + "\n".join(rows)
        )
    return f"""# {b["title"]}

> 来源归档（blog / 微信公众号）

- **标题：** {b["title"]}
- **类型：** blog
- **作者：** senlanke / 具身运控lab（微信公众号）
- **原始链接：** {b["url"]}
- **发表日期：** {PUB}
- **入库日期：** {TODAY}
- **抓取方式：** `scripts/fetch_wechat_article.py`（Chromium + 微信内置 UA，AGENTS.md「兜底」流程）
- **一句话说明：** {WEEK} {b["label"]} **{n_all}** 篇；★ 为周报标注的重点论文；**每篇 arXiv 论文唯一详情节点**（新建 {n_new} / 复用 {n_all - n_new}）。
- **说明：** 条目格式为「英文标题 / 中文标题 / 机构 / 论文链接 / 解决的问题 / 创新点」，机构为周报所述、未逐条核实；技术事实以各详情节点引用的 arXiv 原文为准。

## 论文映射（{n_all}）

{chr(10).join(parts)}

## 对 wiki 的映射

- **技术地图：** [`wiki/overview/{b["overview"]}.md`](../../wiki/overview/{b["overview"]}.md)
- 同一 arXiv 论文在其他周报重复出现时共用同一节点，不重复造页。

## 当前提炼状态

- [x] 公众号正文抓取与条目解析（{n_all} 条）
- [x] 唯一 arXiv 节点核查（按 arXiv ID 对仓库去重）
"""


def render_overview(key: str) -> str:
    b = BLOGS[key]
    n_all = len(LISTS[key])
    n_new = sum(1 for e in LISTS[key] if not PAPERS[e["id"]]["reuse"])
    parts = []
    for sec, rows in table_rows(key, "../entities/").items():
        parts.append(
            f"### {sec}（{len(rows)}）\n\n| # | 论文节点 | arXiv | 状态 |\n|---|----------|-------|------|\n"
            + "\n".join(rows)
        )
    if key == "hq":
        tags, related, abbr = (
            "[survey, humanoid, quadruped, legged-robot, senlanke]",
            [
                "../tasks/locomotion.md",
                "../tasks/humanoid-locomotion.md",
                "../methods/reinforcement-learning.md",
            ],
            ["RL", "WBC", "MPC", "LiDAR", "IMU"],
        )
    else:
        tags, related, abbr = (
            "[survey, manipulation, vla, world-action-model, senlanke]",
            ["../tasks/manipulation.md", "../methods/vla.md", "../concepts/world-action-models.md"],
            ["VLA", "WAM", "RL", "IL"],
        )
    return f"""---
type: overview
tags: {tags}
status: complete
updated: {TODAY}
related:
{chr(10).join(f"  - {r}" for r in related)}
sources:
  - ../../sources/blogs/{b["file"]}
summary: "senlanke 具身运控lab {WEEK} {b["label"]}周报：{n_all} 条论文条目，每条对应唯一 arXiv 详情节点（新建 {n_new} / 复用 {n_all - n_new}）。"
---

# senlanke 周报索引：{WEEK} {b["label"]}（{n_all} 篇）

> **本页定位**：为 [{b["title"]}]({b["url"]}) 提供按周报分栏组织的 **{n_all} 个条目 → 唯一 arXiv 详情节点**索引。★ 为周报标注的重点论文。

## 一句话观点

本页只按周报原有分栏（{"；".join(dict.fromkeys(e["sec"] for e in LISTS[key]))}）列出条目；论文间的横向优劣不在此判断，请按 arXiv 进各详情节点，条目说明来自周报「问题/创新点」与 arXiv 摘要页。

## 英文缩写速查

{abbrev_table(abbr)}

## 论文索引

{chr(10).join(parts)}

## 关联页面

{chr(10).join(f"- [{r.split('/')[-1].removesuffix('.md')}]({r})" for r in related)}

## 参考来源

- [{b["file"]}](../../sources/blogs/{b["file"]})
- [原文链接]({b["url"]})
"""


def append_blog_source(path: Path, blog_name: str) -> None:
    text = path.read_text(encoding="utf-8")
    if blog_name in text:
        return
    text = text.replace("sources:\n", f"sources:\n  - ../../sources/blogs/{blog_name}\n", 1)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    n_new = n_reuse = 0
    for aid, p in PAPERS.items():
        if p["reuse"]:
            wiki = ROOT / p["reuse"]
            assert wiki.exists(), wiki
            for k, _ in appearances(aid):
                append_blog_source(wiki, BLOGS[k]["file"])
            n_reuse += 1
            continue
        wiki = ROOT / f"wiki/entities/paper-{p['slug']}.md"
        src = ROOT / f"sources/papers/{p['slug']}_arxiv_{aid.replace('.', '_')}.md"
        assert not wiki.exists(), f"slug collision {wiki}"
        wiki.write_text(render_entity(p), encoding="utf-8")
        src.write_text(render_source(p), encoding="utf-8")
        n_new += 1
    for key, b in BLOGS.items():
        (ROOT / "sources/blogs" / b["file"]).write_text(render_blog(key), encoding="utf-8")
        (ROOT / "wiki/overview" / f"{b['overview']}.md").write_text(
            render_overview(key), encoding="utf-8"
        )
    print(f"Done: {n_new} new + {n_reuse} reuse; {len(PAPERS)} unique arXiv")


if __name__ == "__main__":
    main()
