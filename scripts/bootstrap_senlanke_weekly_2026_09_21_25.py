#!/usr/bin/env python3
"""Bootstrap ingest: senlanke 具身运控lab 9.21-9.25 双周更（人形/四足 35 + Manipulation 15）."""

from __future__ import annotations

from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TODAY = date.today().isoformat()
BLOG_HQ = "wechat_senlanke_weekly_humanoid_quadruped_2026-09-21_25.md"
BLOG_MAN = "wechat_senlanke_weekly_manipulation_2026-09-21_25.md"
WEEK_LABEL = "2026-09-21–25"

REUSE: dict[str, dict[str, str]] = {
    "2609.21447": {"wiki": "wiki/entities/paper-footquery-perceptive-humanoid-locomotion.md", "label": "FootQuery"},
    "2609.21467": {"wiki": "wiki/entities/paper-dcrr-distance-conditioned-humanoid-transport.md", "label": "距离条件运输"},
    "2609.24145": {"wiki": "wiki/entities/paper-mimicagent.md", "label": "MimicAgent"},
    "2609.24840": {"wiki": "wiki/entities/paper-predactor.md", "label": "PredActor"},
    "2609.25363": {"wiki": "wiki/entities/paper-hotice.md", "label": "HOTICE"},
    "2609.25627": {"wiki": "wiki/entities/paper-me-u0.md", "label": "MachEmbodied-U0"},
    "2609.26420": {"wiki": "wiki/entities/paper-sample-simulate-select.md", "label": "Sample, Simulate, Select"},
    "2609.26467": {"wiki": "wiki/entities/paper-routelt.md", "label": "RouteRLT"},
    "2609.26520": {"wiki": "wiki/entities/paper-mate-virtual-teleop.md", "label": "MATE"},
    "2609.28175": {"wiki": "wiki/entities/paper-davis-humanoid-soccer.md", "label": "DAVIS"},
    "2609.28281": {"wiki": "wiki/entities/paper-brickcraft-duo.md", "label": "BrickCraft-Duo"},
    "2609.28378": {"wiki": "wiki/entities/paper-forgetmimic.md", "label": "ForgetMimic"},
    "2609.28393": {"wiki": "wiki/entities/paper-pointcast-point-set-world-model.md", "label": "PointCast"},
    "2609.28807": {"wiki": "wiki/entities/paper-streaming-rl-continual-robotics.md", "label": "Streaming RL 分析"},
    "2609.28959": {"wiki": "wiki/entities/paper-tactilestep.md", "label": "TactileStep"},
    "2609.28960": {"wiki": "wiki/entities/paper-echo-in-the-steps.md", "label": "Echo in the Steps"},
}

REUSE_BLOG: dict[str, str] = {
    "2609.25627": BLOG_MAN,
    "2609.26467": BLOG_MAN,
    "2609.28281": BLOG_MAN,
    "2609.28393": BLOG_MAN,
}

DEFAULT_ABBREV = [
    ("RL", "Reinforcement Learning", "强化学习"),
    ("WBC", "Whole-Body Control", "全身控制"),
    ("MPC", "Model Predictive Control", "模型预测控制"),
]

LOCO_REL = ["../tasks/humanoid-locomotion.md", "../methods/reinforcement-learning.md", "../concepts/sim2real.md"]
MANIP_REL = ["../tasks/manipulation.md", "../methods/vla.md", "../concepts/world-action-models.md"]
QUAD_REL = ["../tasks/locomotion.md", "../methods/reinforcement-learning.md"]


def mk(
    slug: str,
    short: str,
    title: str,
    arxiv: str,
    blog: str,
    one_liner: str,
    why: str,
    mechanism: str,
    metrics: str,
    conclusion: str,
    tags: list[str],
    related: list[str] | None = None,
    open_status: str = "待发布",
) -> dict:
    return {
        "slug": slug,
        "short": short,
        "title": title,
        "arxiv": arxiv,
        "blog": blog,
        "open": open_status,
        "tags": ["paper", *tags],
        "one_liner": one_liner,
        "why": why,
        "mechanism": mechanism,
        "metrics": metrics,
        "conclusion": conclusion,
        "related": related or LOCO_REL,
        "abbrev": DEFAULT_ABBREV,
    }


PAPERS: list[dict] = [
    mk("redact-robust-perceptive-locomotion", "REDACT", "REDACT: Robust Perceptive Locomotion under Unseen Visual Corruption", "2609.25450", BLOG_HQ, "Teacher–Student + 特征遮蔽 + 共识门控：仅用干净仿真深度训练，迁移到未知视觉损坏与森林场景。", "真机深度常遇训练未覆盖的损坏；单纯数据增强无法覆盖未知 corruption。", "Teacher–Student；continual feature masking；conformal-calibrated consensus gating on depth features.", "结构化环境与森林场景迁移（以 PDF 为准）。", "REDACT 把「哪些深度特征仍可信」做成可部署门控，适合未知视觉损坏下的感知 locomotion。", ["humanoid", "teacher-student", "depth", "sim2real"]),
    mk("strider-multi-gait-loco-manip", "STRIDER", "STRIDER: Stepping-Enabled Multi-Gait Hierarchical 3D Loco-Manipulation Framework for Humanoid Robots", "2609.23483", BLOG_HQ, "AMP 行走 + 3D 落脚专家 + 笛卡尔上肢；LD-PPO 在线 RL + DAgger + Teacher latent 对齐蒸馏统一 Student。", "速度指令策略难控三维落点；单独踏步策略难与行走/操作统一。", "Multi-expert + LD-PPO with DAgger and teacher-conditioned latent alignment.", "X-Humanoid 平台分层 loco-manip（以 PDF 为准）。", "STRIDER 用 latent 蒸馏把异构专家合成可踏步的多步态 loco-manip 框架。", ["humanoid", "loco-manipulation", "teacher-student"]),
    mk("plat-sparse-keyframe-tracking", "PLAT", "PLAT: Sparse Timed Keyframe Motion Tracking for Humanoid Control via Privileged Latent Transition Learning", "2609.25754", BLOG_HQ, "稠密动作专家→DAgger 学 latent 转移先验→RL 只修正 latent 转移；部署仅需稀疏关键帧+到达时间。", "稠密逐帧跟踪无法作高层运动控制器。", "Privileged latent transition learning; sparse timed keyframes at deploy time.", "稀疏关键帧跟踪部署（以 PDF 为准）。", "PLAT 把「改 latent 而非动作残差」用于可稀疏部署的 motion tracking。", ["humanoid", "motion-tracking", "teacher-student"]),
    mk("unipoint-sensor-fusion-locomotion", "UniPoint", "UniPoint: Unified Point-Level Sensor Fusion for Humanoid Locomotion Across Challenging Terrains", "2609.23666", BLOG_HQ, "360° LiDAR + 双深度→机身点集体素 token；线性自注意力+本体查询；传感退化注入训练单一全地形策略。", "单前向深度覆盖有限；多相机编码成本高。", "Unified point tokens; linear self-attn + proprio cross-attn; sensor dropout.", "复杂地形单策略（浙大/云深处，以 PDF 为准）。", "UniPoint 用点级融合使计算量不随传感器数量线性爆炸。", ["humanoid", "perception", "lidar"]),
    mk("higennto-noise-space-optimization", "HIGenNTO", "HIGenNTO: Scalable Humanoid Interaction Generation via Noise-Space Trajectory Optimization", "2609.22611", BLOG_HQ, "优化预训练文本动作模型的初始噪声而非动作序列，满足接触/避碰/支撑约束。", "接触丰富交互难获稳定物理可执行参考。", "Noise-space optimization on generative motion prior with physics constraints.", "可训练跟踪器或深度视觉策略（CMU/庆应，以 PDF 为准）。", "HIGenNTO 在噪声空间做约束优化，扩展接触交互参考生成。", ["humanoid", "motion-generation"]),
    mk("limbo-barrier-objectives-wbc", "LIMBO", "LIMBO: Learning and Internalizing Model-Free Barrier Objectives for Agile and Safe Whole-Body Control", "2609.22075", BLOG_HQ, "围绕冻结控制器残差学习状态–动作屏障函数；任务策略训练时内化安全反馈。", "解析安全证书难；在线安全滤波增开销。", "Model-free control barrier on residual actions around frozen controller.", "高 DoF 人形敏捷安全 WBC（Amazon/Caltech 等，以 PDF 为准）。", "LIMBO 把屏障结构内化进策略，减少在线滤波依赖。", ["humanoid", "wbc", "safe-rl"]),
    mk("whole-body-umi-realtime-motion", "Whole-Body UMI", "Whole-Body UMI: Transferring UMI Manipulation Skills to Humanoid Whole-Body Manipulation via Real-Time Motion Generation", "2609.22829", BLOG_HQ, "扩散策略预测 UMI 末端轨迹；独立实时全身生成器转参考；G1 异步层级闭环。", "UMI 末端轨迹无法唯一确定全身协调。", "Decouple task diffusion on EE + real-time whole-body motion generator.", "G1 全身 UMI 迁移（浙大/港中文等，以 PDF 为准）。", "Whole-Body UMI 解耦语义学习与全身协调生成。", ["humanoid", "loco-manipulation", "imitation-learning"]),
    mk("opt2vla-force-aware-humanoid", "Opt2VLA", "Opt2VLA: Force-Aware Vision-Language-Action for Contact-Rich Humanoid Whole-Body Manipulation", "2609.23968", BLOG_HQ, "多任务 VLA 同时输出几何目标与连续接触力参考；RL WBC 跟踪；WTO 自动生成带力标签数据。", "几何相同但接触力需求不同的操作无法仅靠视觉区分。", "VLA predicts motion + force reference; task-specific RL WBC tracking.", "三个 contact-rich 人形任务仿真+真机（Georgia Tech，以 PDF 为准）。", "Opt2VLA 把力参考纳入 VLA 动作接口，连接语义与接触力控制。", ["humanoid", "vla", "force-control"], MANIP_REL),
    mk("smoothness-constraint-locomotion", "DeCap 平滑约束", "Smoothness as a Constraint for Stable Humanoid Locomotion", "2609.24552", BLOG_HQ, "DeCap：上下半身分别设物理运动约束，有界屏障惩罚接近边界前介入；可跨地形迁移。", "平滑奖励与速度跟踪竞争；统一全身约束致迟缓。", "DeCap constrained RL with upper/lower body separate smoothness barriers.", "多地形迁移无需重调平滑奖励（瑞典 Örebro，以 PDF 为准）。", "DeCap 把平滑性从 reward 竞争改为分半身硬约束。", ["humanoid", "locomotion"]),
    mk("frames-failure-recovery-loco-manip", "FRAMES", "FRAMES: Failure Recovery And Monitoring of Embodied Skills for Humanoid Loco-Manipulation", "2609.22538", BLOG_HQ, "VLM 监控多视角时序+状态+接触证据；失败时结构化原因给恢复智能体+记忆复用。", "语言规划选对技能仍可能在抓取/搬运阶段失败。", "VLM monitor + structured failure reason + recovery agent with memory.", "人形 loco-manip 技能监测恢复（Duke，以 PDF 为准）。", "FRAMES 把失败监测与恢复从规划层下沉到技能执行层。", ["humanoid", "loco-manipulation", "vlm"]),
    mk("primo-human-motion-odometry", "PRIMO", "PRIMO: Prior-Informed Odometry from Human-Motion Tracking for Humanoid Robots", "2609.23610", BLOG_HQ, "跟踪大量重定向人体动作扩分布；物理与对称先验约束速度/旋转预测。", "单策略里程计过拟合；无约束网络 Sim2Real 不合理。", "Human-motion tracking data + physics/symmetry priors on odometry.", "武大/智元 G1 里程计（以 PDF 为准）。", "PRIMO 用人体跟踪先验拓宽里程计训练分布。", ["humanoid", "state-estimation"]),
    mk("emopose-emotion-gesture", "EmoPose", "EmoPose: Vision-Language Model Guided Emotion-Aware Gesture Generation for Humanoid Robots", "2609.23414", BLOG_HQ, "VLM 选手势类/版本/强度/语音触发点；本地 14-DoF 动作库生成验证调度。", "VLM 直接出关节难保证可执行。", "VLM semantic planning + local verified gesture library.", "开放式语言交互手势（港科广等，以 PDF 为准）。", "EmoPose 分离语义规划与安全轨迹执行。", ["humanoid", "social-hri", "vlm"]),
    mk("saber-semantic-affordance-legged", "SABER", "SABER: Learning Attention-based Semantic Affordance for Legged Locomotion", "2609.21572", BLOG_HQ, "3D 几何+语义接触代价统一地形图；注意力 logit 加足端距离符号语义偏置。", "纯几何可能把管道/易碎箱当可踩区。", "Semantic affordance map + signed bias in foot placement attention.", "Unitree B2 室内外语义避踩（A*STAR/NTU，以 PDF 为准）。", "SABER 把语义接触代价注入落脚注意力。", ["quadruped", "semantics", "rl"], QUAD_REL),
    mk("duty-factor-quadruped-robustness", "占空比预测鲁棒性", "Duty Factor Predicts Robust Constrained Quadrupedal Locomotion Across Gait Types", "2609.22073", BLOG_HQ, "在 TO+LQR、学习控制、质心 MPC 中统一分析；占空比比步态名更能预测窄梁/扰动稳定性。", "步态类别不足以解释受限环境稳定性。", "Duty factor as cross-framework stability predictor; terrain-width-conditioned DF.", "四足窄梁与扰动（迈阿密/CMU，以 PDF 为准）。", "占空比是比名义步态更通用的鲁棒性旋钮。", ["quadruped", "locomotion"], QUAD_REL),
    mk("sg-cpg-actuator-degradation", "SG-CPG", "SG-CPG: Severity-Gated Central Pattern Generators for Adaptive Quadruped Locomotion under Continuous Actuator Degradation", "2609.25687", BLOG_HQ, "冻结健康 CPG + 严重度门控残差协调 + 弱腿振幅门；Go2 最高 93% 小腿力矩退化仍多数通过。", "容错常把关节二值化为正常/失效。", "Frozen healthy CPG + severity-gated residual and amplitude gates.", "Go2 仿真 95% 强度损失仍 100% survival；真机 28/29 trials（Purdue，以 PDF 为准）。", "SG-CPG 用连续严重度门控延长健康 CPG 到渐进退化。", ["quadruped", "cpg", "rl"], QUAD_REL),
    mk("flycns-connectome-communication", "FlyCNS", "FlyCNS: Connectome-Grounded Information Organization for Communication-Constrained Embodied Control", "2609.28816", BLOG_HQ, "果蝇连接组启发局部模块+上下行路径；RL 联合学信息内容与发送时机；~21% 通信量保跟踪性能。", "分布式腿控不能把所有传感持续传到中央。", "Connectome-inspired comm paths; RL co-learns content and timing.", "约 21% 通信量接近全通信策略（印第安纳大学，以 PDF 为准）。", "FlyCNS 把通信预算作为一等设计变量。", ["locomotion", "rl", "communication"], QUAD_REL),
    mk("online-sim2real-closed-loop-modeling", "在线闭环 Sim2Real", "Online Sim-to-Real Adaptation via Closed-Loop System Modeling", "2609.28878", BLOG_HQ, "把机器人+已部署策略视为闭环系统，学指令–响应关系；在线只改给控制器的参考，不改策略参数。", "迁移后残余动力学致持续跟踪偏差。", "Closed-loop system ID on command–response; reference adaptation only.", "双足速度跟踪与移动操作硬件（Duke，以 PDF 为准）。", "在线改参考而非重训策略，是轻量 Sim2Real 适应路线。", ["sim2real", "locomotion"], LOCO_REL),
    mk("when-to-waddle-biped-friction", "何时摇摆行走", "When to Waddle: A Comparative Study of Bipedal Torso-Stabilization on Low-Friction Surfaces", "2609.21185", BLOG_HQ, "五执行器双足比较直立 vs 企鹅式躯干侧移；系统改变质心高度与摩擦。", "低摩擦下地面反力受限，步态选择不明确。", "Comparative upright vs penguin waddle gaits on low-friction surfaces.", "低摩擦高质心企鹅步态更优；较高摩擦低质心更优（CMU/NYU，以 PDF 为准）。", "低摩擦场景应优先考虑躯干侧移策略而非仅调 PD。", ["biped", "locomotion"], LOCO_REL),
    mk("spiderbot-hexapod-open-source", "Spiderbot", "Spiderbot: An Open-Source Energy-Efficient Hexapod with Passive Gravity Compensation", "2609.26989", BLOG_HQ, "两 DoF 四连杆+被动弹簧支撑；站立 1.5 W；<$400；开源 CAD/mjlab/部署。", "三 DoF 腿增重与支撑力矩。", "Passive gravity-compensated hexapod legs + open mjlab stack.", "斜坡/粗糙/台阶 Sim2Real（印度 BITS 果阿，以 PDF 为准）。", "Spiderbot 是低成本开源六足+RL 训练栈样本。", ["hexapod", "open-source", "rl"], QUAD_REL, open_status="已开源"),
    mk("banana-kick-humanoid-soccer", "Banana Kick", "Banana Kick: Response-Informed Skill Evolution for Humanoid Soccer", "2609.27269", BLOG_HQ, "RISE：按物理响应排序候选目标变化，逐步把普通射门推进到旋转球接触区。", "旋转奖励在普通射门策略附近梯度弱。", "Response-informed skill evolution (RISE) for spin shot exploration.", "人形足球旋转球（CMU/GM 等，以 PDF 为准）。", "RISE 用响应排序解决稀疏旋转技能探索。", ["humanoid", "soccer", "rl"], LOCO_REL),
    mk("humanoid-fly-inspired-rnn", "果蝇启发 RNN 控制器", "Humanoid Locomotion with a Fly-Inspired Recurrent Controller", "2609.27001", BLOG_HQ, "3609 连续神经状态接 G1 仿真；重置/路径替换定位行为来源。", "生物启发控制器机制难分析。", "Fly-inspired recurrent controller with mechanistic ablations on G1 sim.", "持续运动依赖本体–指令与循环 motor 状态（港科/Zenbot 等，以 PDF 为准）。", "该工作提供可追踪机制的生物启发 locomotion 分析范式。", ["humanoid", "locomotion", "neuroscience"], LOCO_REL),
    mk("runway-expressive-locomotion", "走秀表现型行走", "Learning Expressive Humanoid Locomotion from Monocular Runway Videos for Robot Fashion Shows", "2609.27003", BLOG_HQ, "单目视频→重定向→修正→策略→Booster K1 走秀部署全流程。", "常规定位稳定/速度，难复现窄步宽与姿态风格。", "Monocular video to expressive gait on Booster K1.", "时装秀步态真机（维尔纽斯/肯特州立，以 PDF 为准）。", "表现型 locomotion 需要视频到策略的完整风格链。", ["humanoid", "locomotion"], LOCO_REL),
    mk("brace-yourself-environmental-bracing", "Brace Yourself", "Brace Yourself: Task-Conditioned Environmental Bracing for Forceful Humanoid Manipulation", "2609.25486", BLOG_HQ, "Supporting Hand Strategy：任务手操作、支撑手找环境支撑点；双 RL 同步；G1 最大 60 N vs 无支撑 13.5 N。", "单脚支撑限制最大操作力。", "Task-conditioned environmental bracing with dual RL policies.", "Unitree G1 强力操作（QUT 等，以 PDF 为准）。", "环境支撑是把反作用力卸到场景的关键人形操作技巧。", ["humanoid", "loco-manipulation", "rl"], LOCO_REL),
    mk("safeloop-vla-rollback", "SafeLoop", "SafeLoop: Risk-Aware Rollback for Vision-Language-Action Manipulation", "2609.26313", BLOG_MAN, "外部 risk predictor 预测碰撞/物体失败概率与时间；noop/record/rollback；回到安全关节态再查 VLA。", "长程 VLA 小误差累积成不可逆失败。", "External risk predictor + checkpoint rollback without fine-tuning VLA.", "LIBERO 24 任务 + 3 真机；危险事件约降 70%（南大/港科广/北理工，以 PDF 为准）。", "SafeLoop 是不改 VLA 参数的安全回滚包装层。", ["vla", "manipulation", "safety"], MANIP_REL),
    mk("internw0-physical-world-model", "InternW0", "InternW0: A Foundational Physical World Model for Efficient Real-World Interactions", "2609.27656", BLOG_MAN, "非对称 Video Expert 低频预测 + Action Expert 高频动作；缓存 layer-wise K/V + context routing 修正，解耦世界预测与控制频率。", "WAM 每步重生成未来视频难实时。", "Asymmetric video/action experts with cached KV routing.", "7200 h 异构数据；15 阶段实验室+移液真机（上海 AI Lab 等，以 PDF 为准）。", "InternW0 用 KV 缓存把 WAM 推到可部署频率。", ["wam", "manipulation"], MANIP_REL),
    mk("core-wam-tracedelta", "CoRe-WAM", "CoRe-WAM: Correspondence-Aligned Temporal Residuals for World Action Models", "2609.27314", BLOG_MAN, "TraceDelta：tracking 对齐历史特征到当前位置，signed feature difference 作 temporal residual；冻结 Motus 仅训 1.59M。", "相机/物体运动使同像素比较失效。", "Correspondence-aligned TraceDelta adapter on frozen WAM.", "RoboTwin 2.0 50 任务 92.22% clean；可接 StarVLA（HKUST，以 PDF 为准）。", "TraceDelta 是参数高效的 WAM 时序对齐插件。", ["wam", "manipulation"], MANIP_REL),
    mk("world-action-agent-rehearsal", "World Action Agent", "World Action Agent: Harnessing VLMs for Robot Manipulation via World Action Rehearsal", "2609.29964", BLOG_MAN, "Visual action workspace；Action Rehearsal 预览修改候选动作；in-view correction；LIBERO-Pro 75.6%。", "VLM 未在执行前观察动作后果。", "WAA with rehearsal + in-view correction + skill accumulation.", "LIBERO-90 技能→LIBERO-Pro/robosuite（以 PDF 为准）。", "World Action Rehearsal 把 VLM 变成可预演动作的工作空间。", ["vla", "manipulation", "agent"], MANIP_REL),
    mk("cfm-multitask-distillation", "CFM 多任务蒸馏", "Distillation for Efficient Multitask Manipulation Policies via Conditional Flow Matching", "2609.28107", BLOG_MAN, "多 single-task CFM Expert→蒸馏 velocity field 到共享 Multi-Task CFM，保留 FM demo objective。", "每任务单独 CFM 成本高；混合训练易干扰。", "Distill expert velocity fields into shared multitask CFM.", "RLBench 优于直接混合多任务训练（弗莱堡，以 PDF 为准）。", "蒸馏 velocity field 比蒸馏最终动作更适合多任务 CFM。", ["flow-matching", "manipulation"], MANIP_REL),
    mk("jamb-bimanual-diffusion", "JAMB", "JAMB: Joint Action-Motion Diffusion for Bimanual Manipulation", "2609.25322", BLOG_MAN, "同一 Transformer 联合去噪双臂 action + 未来 3D point tracks，denoising 中互相修正。", "双臂 action-only 扩散不显式预测场景如何被改变。", "Joint denoising of bimanual actions and future point tracks.", "RoboTwin 2.0 16 任务 83.4%（CMU/密歇根，以 PDF 为准）。", "JAMB 用 future point tracks 把双臂交互写进扩散联合空间。", ["bimanual", "diffusion", "manipulation"], MANIP_REL),
    mk("cartesian-hand-linear-fingers", "Cartesian Hand", "The Cartesian Hand: In-Hand Manipulation with All-Linear Fingers", "2609.25696", BLOG_MAN, "7-DoF 全线性：双平行夹爪+四平移指尖；35 种物体操作；计划开源软硬件。", "仿人灵巧手复杂；平行夹爪无手内操作。", "All-linear 7-DoF end-effector with composable linear primitives.", "35 种 in-hand 任务；可迁人形双臂（Duke 等，以 PDF 为准）。", "Cartesian Hand 用线性原语组合实现高覆盖手内操作。", ["dexterous-manipulation", "hardware"], MANIP_REL, open_status="宣称将开源"),
    mk("glotouch-haptic-grasping", "GLoTouch", "GLoTouch: Global-to-Local Haptic Perception Using a Parallel Gripper for Object Search, Recognition, and Grasping Without External Vision", "2609.27695", BLOG_MAN, "全局探针搜索+局部 visuotactile 与 3D 模型匹配；无外部视觉。", "平行夹爪触觉范围小，难同时搜索与识别。", "Global probe search then local visuotactile model matching.", "黑暗/低光无视觉抓取（以 PDF 为准）。", "GLoTouch 把全局搜索与局部触觉识别拆成两阶段。", ["haptic", "grasping", "manipulation"], MANIP_REL),
    mk("watch-recall-act-concurrent-streams", "Watch, Recall, Act", "Watch, Recall, Act: Always-On Robots in Concurrent Embodied Streams", "2609.28429", BLOG_MAN, "π0.5 上三轻量模块压缩持续视觉/状态/历史动作为可读 context；感知记忆动作异步。", "真实家庭并发事件需长期记忆而非 reset benchmark。", "Always-on context modules on π0.5 for concurrent embodied streams.", "并发流数据与双臂评测（上交/港中文/南大/Astribot 等，以 PDF 为准）。", "常开机器人需要把 memory 做成策略可读 context 而非 episodic reset。", ["long-horizon", "manipulation", "memory"], MANIP_REL),
]


def abbrev_table(rows: list[tuple[str, str, str]]) -> str:
    lines = ["| 缩写 | 英文全称 | 简要说明 |", "|------|----------|----------|"]
    for a, b, c in rows:
        lines.append(f"| {a} | {b} | {c} |")
    return "\n".join(lines)


def related_lines(related: list[str]) -> str:
    return "\n".join(f"  - {r}" for r in related)


def render_wiki(p: dict) -> str:
    arxiv = p["arxiv"]
    slug = p["slug"]
    src_paper = f"../../sources/papers/{slug}_arxiv_{arxiv.replace('.', '_')}.md"
    src_blog = f"../../sources/blogs/{p['blog']}"
    related_wiki = p["related"]
    ol = p["one_liner"].replace('"', "'")
    sm = p["summary"] if "summary" in p else f"{p['short']}（arXiv:{arxiv}）：{ol[:120]}"
    return f"""---
type: entity
tags:
{chr(10).join(f"  - {t}" for t in p["tags"])}
status: complete
updated: {TODAY}
arxiv: "{arxiv}"
related:
{related_lines(related_wiki)}
sources:
  - {src_paper}
  - {src_blog}
summary: "{sm}"
---

# {p["short"]}（arXiv:{arxiv}）

**{p["short"]}**（*{p["title"]}*，[arXiv:{arxiv}](https://arxiv.org/abs/{arxiv})）来自 [senlanke 具身运控lab 周更盘点]({src_blog})（{WEEK_LABEL}）。

## 一句话定义

**{p["one_liner"]}**

## 英文缩写速查

{abbrev_table(p["abbrev"])}

## 为什么重要

- {p["why"]}

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [{arxiv}](https://arxiv.org/abs/{arxiv}) |
| **开源** | **{p["open"]}**（步骤 2.5，{TODAY}） |
| **方法摘要** | {p["mechanism"]} |

## 源码运行时序图

**不适用**（截至 {TODAY} 未发布可运行官方代码或待核实）。

## 实验与评测

- {p["metrics"]}
- 读法：先对齐任务设定、传感器与成功定义，再解读 headline 数字。

## 与其他工作对比

| 维度 | 读法 |
|------|------|
| **同周对照** | 见对应 [周更盘点]({src_blog}) 映射表，勿跨任务直接比 SR |
| **开源状态** | **{p["open"]}** — 部署前以项目页/arXiv 为准 |

## 结论

**{p["conclusion"]}**

1. 开源：**{p["open"]}**；勿凭公众号摘要臆断可复现性。
2. 指标须连同实验条件解读（仿真/真机、平台、成功阈值）。
3. 关注 arXiv 版本更新与代码发布。

## 关联页面

{chr(10).join(f"- [{r.split('/')[-1].replace('.md', '')}]({r})" for r in related_wiki)}

## 参考来源

- [{slug}_arxiv_{arxiv.replace(".", "_")}.md]({src_paper})
- [{p["blog"]}]({src_blog})
- [arXiv:{arxiv}](https://arxiv.org/abs/{arxiv})

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/{arxiv})
"""


def render_source(p: dict) -> str:
    arxiv = p["arxiv"]
    slug = p["slug"]
    return f"""# {p["short"]}（arXiv:{arxiv}）

> 来源归档（paper）

- **标题：** {p["title"]}
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/{arxiv}>
- **PDF：** <https://arxiv.org/pdf/{arxiv}>
- **入库日期：** {TODAY}
- **一句话说明：** {p["one_liner"]}

## 开源状态

- **{p["open"]}**（步骤 2.5 核查，{TODAY}）

## 核心摘录

1. **策展来源：** [senlanke 周更](../../sources/blogs/{p["blog"]})
2. **机制：** {p["mechanism"]}

**对 wiki 的映射**

- [paper-{slug}](../../wiki/entities/paper-{slug}.md)
"""


def append_blog_source(wiki_path: Path, blog_name: str) -> None:
    text = wiki_path.read_text(encoding="utf-8")
    src_line = f"  - ../../sources/blogs/{blog_name}"
    if blog_name in text:
        return
    if "sources:\n" in text:
        text = text.replace("sources:\n", f"sources:\n{src_line}\n", 1)
    else:
        text = text.replace("summary:", f"sources:\n{src_line}\nsummary:", 1)
    wiki_path.write_text(text, encoding="utf-8")


def main() -> None:
    for p in PAPERS:
        arxiv_id = p["arxiv"].replace(".", "_")
        wiki_path = ROOT / f"wiki/entities/paper-{p['slug']}.md"
        src_path = ROOT / f"sources/papers/{p['slug']}_arxiv_{arxiv_id}.md"
        wiki_path.write_text(render_wiki(p), encoding="utf-8")
        src_path.write_text(render_source(p), encoding="utf-8")
        print(f"created {wiki_path.name}")

    for arxiv, info in REUSE.items():
        wiki_path = ROOT / info["wiki"]
        if not wiki_path.exists():
            print(f"WARN missing reuse target {wiki_path}")
            continue
        blog = REUSE_BLOG.get(arxiv, BLOG_HQ)
        append_blog_source(wiki_path, blog)
        print(f"updated source on reuse {wiki_path.name}")

    print(f"Done: {len(PAPERS)} new + {len(REUSE)} reuse source updates")


if __name__ == "__main__":
    main()
