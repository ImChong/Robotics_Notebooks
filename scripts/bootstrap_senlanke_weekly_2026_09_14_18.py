#!/usr/bin/env python3
"""Bootstrap ingest: senlanke 具身运控lab 9.14-9.18 双周更（人形/四足 28 + Manipulation 14）."""

from __future__ import annotations

from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TODAY = date.today().isoformat()
BLOG_HQ = "wechat_senlanke_weekly_humanoid_quadruped_2026-09-14_18.md"
BLOG_MAN = "wechat_senlanke_weekly_manipulation_2026-09-14_18.md"

REUSE: dict[str, dict[str, str]] = {
    "2609.19340": {"wiki": "wiki/entities/paper-viloman.md", "label": "ViLoMan"},
    "2609.19272": {"wiki": "wiki/entities/paper-rom-nav.md", "label": "ROM-Nav"},
    "2609.18732": {"wiki": "wiki/entities/paper-passage.md", "label": "PASSAGE"},
    "2609.20558": {
        "wiki": "wiki/entities/paper-g1-slope-adaptive-roofing-locomotion.md",
        "label": "G1 屋面斜坡",
    },
    "2609.18869": {"wiki": "wiki/entities/paper-kino.md", "label": "KINO"},
    "2609.16644": {"wiki": "wiki/entities/paper-wholebodywam.md", "label": "WholeBodyWAM"},
    "2609.15213": {"wiki": "wiki/entities/paper-x-wbc.md", "label": "X-WBC"},
    "2609.14432": {"wiki": "wiki/entities/paper-emog.md", "label": "EMoG"},
    "2609.19582": {"wiki": "wiki/entities/paper-omnicalib.md", "label": "OmniCalib"},
    "2609.20566": {"wiki": "wiki/entities/paper-omnimimic.md", "label": "OmniMimic"},
    "2609.15770": {"wiki": "wiki/entities/paper-jeplo.md", "label": "JEPLO"},
    "2609.18207": {
        "wiki": "wiki/entities/paper-real-time-expo-ft.md",
        "label": "Real-Time EXPO-FT",
    },
    "2609.16586": {"wiki": "wiki/entities/paper-proxidex.md", "label": "ProxiDex"},
    "2609.18620": {"wiki": "wiki/entities/paper-deformsmith.md", "label": "DeformSmith"},
    "2609.16683": {"wiki": "wiki/entities/paper-weave.md", "label": "WEAVE"},
    "2609.17210": {"wiki": "wiki/entities/fluxvla-engine.md", "label": "FluxVLA Engine"},
}

PAPERS: list[dict] = [
    {
        "slug": "recal-collision-aware-wbc",
        "short": "RECAL",
        "title": "Collision-Aware Humanoid Whole-Body Control under Imperfect Tracking Targets",
        "arxiv": "2609.16405",
        "blog": BLOG_HQ,
        "open": "待发布",
        "tags": ["paper", "humanoid", "wbc", "teacher-student", "collision-avoidance"],
        "one_liner": "RECAL 交叉注意力层用机器人/物体/环境点云修正冻结 WBC；特权 Teacher→观测 Student，Digit V3 兼顾跟踪、行走与局部避碰。",
        "why": "冻结 WBC 在 clutter 与 imperfect reference 下易碰撞；需几何感知修正层而非重训全身策略。",
        "mechanism": "RECAL cross-attention 融合点云修正 WBC 输出；Teacher–Student 蒸馏部署观测策略。",
        "metrics": "Digit V3：目标跟踪 + 行走 + 局部避碰（以 PDF 为准）。",
        "conclusion": "RECAL 把碰撞感知作为 WBC 之上的可蒸馏修正层，适合 imperfect tracking 的 clutter 场景。",
        "related": [
            "../tasks/humanoid-locomotion.md",
            "../concepts/whole-body-control.md",
            "../methods/reinforcement-learning.md",
        ],
        "abbrev": [
            ("WBC", "Whole-Body Control", "全身控制"),
            ("RL", "Reinforcement Learning", "强化学习"),
            ("TS", "Teacher–Student", "特权–观测蒸馏"),
        ],
    },
    {
        "slug": "gated-residual-body-hand-coordination",
        "short": "门控残差身–手协调",
        "title": "Gated Residual Body-Hand Coordination for Whole-Body Humanoid Teleoperation",
        "arxiv": "2609.18763",
        "blog": BLOG_HQ,
        "open": "待发布",
        "tags": ["paper", "humanoid", "teleoperation", "dexterous-manipulation"],
        "one_liner": "冻结两模块，仅学有界残差；动作门控分配关节组修正权限，几何奖励门控强调当前交互约束；腕/指尖误差降 39–56%。",
        "why": "全身遥操作中身体与手模块独立优化易冲突；残差+门控可在不破坏原模块前提下协调。",
        "mechanism": "Bounded residual on frozen body/hand modules; action gating + geometry reward gating.",
        "metrics": "腕部/指尖误差降低 39.2%–56.3%（作者报告）。",
        "conclusion": "门控残差是全身遥操作的可插拔协调范式，适合已有 body/hand 栈增量部署。",
        "related": [
            "../tasks/teleoperation.md",
            "../tasks/loco-manipulation.md",
            "../concepts/whole-body-control.md",
        ],
        "abbrev": [
            ("WBC", "Whole-Body Control", "全身控制"),
            ("DoF", "Degrees of Freedom", "自由度"),
            ("HRI", "Human-Robot Interaction", "人机交互"),
        ],
    },
    {
        "slug": "decentralized-multi-humanoid-pickup",
        "short": "去中心化多人形搬运",
        "title": "Learning Multi-Humanoid Pickup and Transport via Decentralized Object-Centric Control",
        "arxiv": "2609.17824",
        "blog": BLOG_HQ,
        "open": "待发布",
        "tags": ["paper", "humanoid", "multi-robot", "loco-manipulation"],
        "one_liner": "每机物体局部附着区 + 相同策略 + 局部观测、无直接通信；单机拾取至十机协同与交接，迁移两台 Digit V3。",
        "why": "多机协同常需通信与异构策略；物体中心局部附着可扩展规模。",
        "mechanism": "Object-centric attachment regions; decentralized identical policies; no inter-robot comms.",
        "metrics": "最多十机协同运输；两台 Digit V3 迁移（以 PDF 为准）。",
        "conclusion": "物体中心去中心化控制使多人形 pickup/transport 在统一策略下扩展。",
        "related": [
            "../concepts/humanoid-multi-robot-coordination.md",
            "../tasks/loco-manipulation.md",
            "./paper-recal-collision-aware-wbc.md",
        ],
        "abbrev": [
            ("MRS", "Multi-Robot System", "多机器人系统"),
            ("WBC", "Whole-Body Control", "全身控制"),
            ("RL", "Reinforcement Learning", "强化学习"),
        ],
    },
    {
        "slug": "leap-quadruped-active-perception",
        "short": "LEAP（四足主动感知）",
        "title": "LEAP: Learning Emergent Active Perception for Quadruped Navigation",
        "arxiv": "2609.17628",
        "blog": BLOG_HQ,
        "open": "待发布",
        "tags": ["paper", "quadruped", "navigation", "active-perception", "rl"],
        "one_liner": "连续深度图融合为视线无关 egocentric 信念地图；仅通过任务难度递增使凝视控制涌现；成功率 92.7% 接近 Oracle。",
        "why": "四足危险地形导航需主动视角；显式 gaze 模块不如 curriculum 涌现稳定。",
        "mechanism": "Depth → view-invariant belief map; emergent gaze via progressive task difficulty.",
        "metrics": "92.7% success vs privileged Oracle（作者报告）。",
        "conclusion": "LEAP 表明四足主动感知可从导航课程中涌现，无需手工 gaze 奖励塑形。",
        "related": [
            "../tasks/locomotion.md",
            "../methods/reinforcement-learning.md",
            "../concepts/sim2real.md",
        ],
        "abbrev": [
            ("LEAP", "Learning Emergent Active Perception", "本文框架"),
            ("RL", "Reinforcement Learning", "强化学习"),
            ("Oracle", "Privileged Oracle", "特权上界策略"),
        ],
    },
    {
        "slug": "pose-semantic-legged-exploration",
        "short": "POSE",
        "title": "Pose-aware Legged Robot Semantic Exploration with Omnidirectional Perception in Confined Unknown Environments",
        "arxiv": "2609.19460",
        "blog": BLOG_HQ,
        "open": "待发布",
        "tags": ["paper", "quadruped", "exploration", "semantic-mapping", "vlm"],
        "one_liner": "POSE 规划器把机身 pitch/roll 纳入语义视点选择；VLM 据历史与 BEV 剪枝冗余视点。",
        "why": "狭窄未知环境探索中机身姿态影响传感器覆盖；纯平移视点规划不足。",
        "mechanism": "Pose-aware semantic viewpoint selection + VLM pruning on BEV/history.",
        "metrics": "Confined unknown environments（以 PDF 为准）。",
        "conclusion": "POSE 把腿式机身姿态作为语义探索的一等变量，适合全向感知狭窄场景。",
        "related": [
            "../tasks/locomotion.md",
            "../methods/vla.md",
            "../concepts/embodied-semantic-cognitive-map.md",
        ],
        "abbrev": [
            ("POSE", "Pose-aware Semantic Exploration", "本文规划器"),
            ("BEV", "Bird's-Eye View", "鸟瞰地图"),
            ("VLM", "Vision-Language Model", "视觉语言模型"),
        ],
    },
    {
        "slug": "fmhp-quadruped-harmonic-policies",
        "short": "FMHP",
        "title": "Feedback-Modulated Harmonic Policies for Quadruped Locomotion",
        "arxiv": "2609.17946",
        "blog": BLOG_HQ,
        "open": "待发布",
        "tags": ["paper", "quadruped", "locomotion", "control"],
        "one_liner": "指令条件傅里叶级数生成关节基准轨迹；状态反馈在线调节偏置、谐波增益、频率与相位；Go2 暂定 3.67 m/s。",
        "why": "参数化谐波轨迹 + 闭环反馈可在保持可解释性的同时提高速度鲁棒性。",
        "mechanism": "Command-conditioned Fourier series base + online feedback modulation.",
        "metrics": "Unitree Go2 ~3.67 m/s + load tests（作者报告）。",
        "conclusion": "FMHP 用谐波先验 + 反馈调制平衡四足速度与可调性。",
        "related": [
            "../tasks/locomotion.md",
            "../entities/go2-motion-imitation.md",
            "../methods/reinforcement-learning.md",
        ],
        "abbrev": [
            ("FMHP", "Feedback-Modulated Harmonic Policies", "本文方法"),
            ("CPG", "Central Pattern Generator", "中枢模式发生器"),
            ("DoF", "Degrees of Freedom", "自由度"),
        ],
    },
    {
        "slug": "smelldiffusion",
        "short": "SmellDiffusion",
        "title": "SmellDiffusion: Diffusion-Based Quadruped Navigation with Olfactory Scene Graphs",
        "arxiv": "2609.20624",
        "blog": BLOG_HQ,
        "open": "待发布",
        "tags": ["paper", "quadruped", "navigation", "diffusion"],
        "one_liner": "开放词汇嗅觉场景图保存气体类别与源估计；几何门控修正浓度峰值；扩散模型生成四足导航轨迹。",
        "why": "搜救/巡检等需非视觉化学源定位；嗅觉图 + 扩散规划是新传感模态组合。",
        "mechanism": "Olfactory scene graph + gated peak correction + diffusion trajectory generation.",
        "metrics": "四足导航任务（以 PDF 为准）。",
        "conclusion": "SmellDiffusion 把嗅觉场景图接入腿式扩散导航，拓展感知模态边界。",
        "related": ["../tasks/locomotion.md", "../methods/generative-world-models.md"],
        "abbrev": [
            ("SLAM", "Simultaneous Localization and Mapping", "同步定位与建图"),
            ("RL", "Reinforcement Learning", "强化学习"),
            ("BEV", "Bird's-Eye View", "鸟瞰视图"),
        ],
    },
    {
        "slug": "glamdring",
        "short": "GLAMDRING",
        "title": "GLAMDRING: Gait Learning And Morphology co-Design via Reinforcement LearnING of CPGs",
        "arxiv": "2609.19452",
        "blog": BLOG_HQ,
        "open": "待发布",
        "tags": ["paper", "quadruped", "morphology", "cpg", "rl"],
        "one_liner": "同时选连杆尺寸、关节执行器与 Hopf CPG 策略；跨形态 RL 推断执行器工作包络。",
        "why": "步态与形态耦合；分离设计常导致执行器/结构不匹配。",
        "mechanism": "Co-design linkage, actuators, Hopf CPG via RL under speed/power/load constraints.",
        "metrics": "Morphology–gait co-design cases（以 PDF 为准）。",
        "conclusion": "GLAMDRING 把 CPG 策略学习嵌入四足形态协同设计闭环。",
        "related": ["../tasks/locomotion.md", "../methods/reinforcement-learning.md"],
        "abbrev": [
            ("CPG", "Central Pattern Generator", "中枢模式发生器"),
            ("RL", "Reinforcement Learning", "强化学习"),
            ("Co-design", "Co-design", "形态–控制联合设计"),
        ],
    },
    {
        "slug": "spot-precision-weeding",
        "short": "Spot 精准除草",
        "title": "Mechanical Precision Weeding with a Quadruped Robot",
        "arxiv": "2609.20048",
        "blog": BLOG_HQ,
        "open": "待发布",
        "tags": ["paper", "quadruped", "spot", "agriculture", "manipulation"],
        "one_liner": "Boston Dynamics Spot 刚性安装铣削除草工具；足端不动、用本体 DoF 定位工具；集成检测、规划与室内外流程。",
        "why": "农业精准作业需移动基座 + 工具定位；四足可复用现有 Spot 平台。",
        "mechanism": "Fixed feet + body DoF tool positioning; weed detection + motion planning pipeline.",
        "metrics": "Indoor/outdoor weeding workflow（以 PDF 为准）。",
        "conclusion": "Spot 精准除草展示四足作为农业机械载体的系统级集成样本。",
        "related": [
            "../entities/paper-autonomous-spot-nebula-exploration.md",
            "../tasks/manipulation.md",
            "../tasks/locomotion.md",
        ],
        "abbrev": [
            ("DoF", "Degrees of Freedom", "自由度"),
            ("SLAM", "Simultaneous Localization and Mapping", "同步定位与建图"),
            ("CV", "Computer Vision", "计算机视觉"),
        ],
    },
    {
        "slug": "wave-go",
        "short": "WAVE-Go",
        "title": "WAVE-Go: World-Model Navigation with Adaptive Execution for Wheel-Legged Robots",
        "arxiv": "2609.18193",
        "blog": BLOG_HQ,
        "open": "待发布",
        "tags": ["paper", "wheeled-leg", "navigation", "world-model"],
        "one_liner": "按累计失败风险选 4/8/16 步最长可执行前缀；RGB-D/LiDAR 持续重验证并可中断；模式切换需空间/稳定/任务证据。",
        "why": "轮足导航需在 walk/drive 间切换；世界模型前缀执行降低盲目长 horizon 风险。",
        "mechanism": "World-model navigation + adaptive prefix execution + mode-switch evidence checks.",
        "metrics": "Wheel-legged robot navigation（以 PDF 为准）。",
        "conclusion": "WAVE-Go 用自适应前缀执行把世界模型导航落到轮足异构运动切换。",
        "related": [
            "../methods/generative-world-models.md",
            "../tasks/locomotion.md",
            "../concepts/sim2real.md",
        ],
        "abbrev": [
            ("WM", "World Model", "世界模型"),
            ("LiDAR", "Light Detection and Ranging", "激光雷达"),
            ("RGB-D", "RGB-Depth", "彩色深度传感"),
        ],
    },
    {
        "slug": "force-aware-wheeled-leg-manip",
        "short": "力感知轮足 loco-manip",
        "title": "Force-Aware Reinforcement Learning with Hybrid Sensorless Force Estimation for Wheeled-Legged Loco-Manipulation",
        "arxiv": "2609.13779",
        "blog": BLOG_HQ,
        "open": "待发布",
        "tags": ["paper", "wheeled-leg", "force-control", "rl", "loco-manipulation"],
        "one_liner": "广义动量观测 + 接触约束投影 + 时序残差估计末端力；显式输入带力/位选择器的全身 RL；覆盖自由运动、纯力控与混合力位控。",
        "why": "轮足操作常缺力传感；混合估计 + 力感知 RL 统一多控制模式。",
        "mechanism": "Hybrid sensorless force estimation → force-aware whole-body RL with axial force/position selector.",
        "metrics": "Wheeled-legged loco-manipulation modes（以 PDF 为准）。",
        "conclusion": "本文把无传感器力估计与轮足全身 RL 绑定，实现力位混合单一控制器。",
        "related": [
            "../tasks/loco-manipulation.md",
            "../methods/reinforcement-learning.md",
            "../concepts/contact-dynamics.md",
        ],
        "abbrev": [
            ("RL", "Reinforcement Learning", "强化学习"),
            ("WBC", "Whole-Body Control", "全身控制"),
            ("EE", "End-Effector", "末端执行器"),
        ],
    },
    {
        "slug": "dr-mpc",
        "short": "DR-MPC",
        "title": "DR-MPC: Fast and Feasible Dynamics-Relaxed Model-Predictive Control for Legged Locomotion",
        "arxiv": "2609.20035",
        "blog": BLOG_HQ,
        "open": "待发布",
        "tags": ["paper", "mpc", "locomotion", "quadruped"],
        "one_liner": "动力学等式与仿射输入约束转二次惩罚，保留盒约束；专用内点法；Go1 中位求解 4.4 ms。",
        "why": "腿式 MPC 常因硬动力学约束导致不可行或慢；松弛保可行且实时。",
        "mechanism": "Dynamics-relaxed QP + block-arrow Hessian + contact-aligned control partitioning.",
        "metrics": "Unitree Go1 median solve 4.4 ms（作者报告）。",
        "conclusion": "DR-MPC 在可行性与速度间为腿式 locomotion 提供可部署 MPC 路线。",
        "related": [
            "../methods/model-predictive-control.md",
            "../tasks/locomotion.md",
            "../entities/autonomy-stack-go2.md",
        ],
        "abbrev": [
            ("MPC", "Model Predictive Control", "模型预测控制"),
            ("QP", "Quadratic Program", "二次规划"),
            ("IPM", "Interior Point Method", "内点法"),
        ],
    },
    {
        "slug": "adaptive-mhe",
        "short": "Adaptive-MHE",
        "title": "Adaptive-MHE: A Sampling-Based Adaptive MPC for Legged Loco-Manipulation via Moving Horizon Estimation",
        "arxiv": "2609.17832",
        "blog": BLOG_HQ,
        "open": "待发布",
        "tags": ["paper", "mpc", "system-identification", "loco-manipulation"],
        "one_liner": "滑动窗并行采样物理参数，以预测–实测误差在线辨识，再反馈给采样式 MPC；无需可微仿真或力传感。",
        "why": "loco-manip 参数漂移常见；MHE 与 MPC 闭环可自适应而不重训策略。",
        "mechanism": "Parallel parameter sampling + trajectory error identification → adaptive sampling MPC.",
        "metrics": "Legged loco-manipulation adaptive control（以 PDF 为准）。",
        "conclusion": "Adaptive-MHE 把在线辨识嵌入 MPC，适合参数不确定的腿式操作。",
        "related": [
            "../methods/model-predictive-control.md",
            "../tasks/loco-manipulation.md",
            "../methods/reinforcement-learning.md",
        ],
        "abbrev": [
            ("MHE", "Moving Horizon Estimation", "移动时域估计"),
            ("MPC", "Model Predictive Control", "模型预测控制"),
            ("SysID", "System Identification", "系统辨识"),
        ],
    },
    {
        "slug": "wrench-polytope-stability",
        "short": "力旋多面体稳定",
        "title": "Optimized Wrench Polytope Analysis for Real-Time Stability Control of Legged Robots in Complex Multi-Contact Configurations",
        "arxiv": "2609.17405",
        "blog": BLOG_HQ,
        "open": "待发布",
        "tags": ["paper", "locomotion", "contact", "stability", "mpc"],
        "one_liner": "高效六维多面体交集与原点单纯形扩展；~49 Hz 计算各关节可实现力矩；LAURON VI 斜墙多接触稳定。",
        "why": "复杂多接触下实时力旋分析是稳定控制瓶颈。",
        "mechanism": "Optimized wrench polytope intersection + feasible torque computation at ~49 Hz.",
        "metrics": "LAURON VI multi-contact incl. inclined wall support（作者报告）。",
        "conclusion": "本文把力旋多面体分析推到复杂多接触场景的实时稳定控制。",
        "related": [
            "../concepts/contact-dynamics.md",
            "../tasks/locomotion.md",
            "../methods/model-predictive-control.md",
        ],
        "abbrev": [
            ("Wrench", "Wrench Polytope", "力旋多面体"),
            ("MPC", "Model Predictive Control", "模型预测控制"),
            ("CoM", "Center of Mass", "质心"),
        ],
    },
    {
        "slug": "skill-composition-legged-rl",
        "short": "腿式 RL 技能组合",
        "title": "Skill Composition for Legged Robot Reinforcement Learning",
        "arxiv": "2609.14647",
        "blog": BLOG_HQ,
        "open": "待发布",
        "tags": ["paper", "locomotion", "rl", "skill-composition"],
        "one_liner": "区分连续加权「混合」与过渡控制器「桥接」；冻结已有专家；踢球残差叠加与走–跳过渡初步实验。",
        "why": "多技能腿式系统需明确组合语义；混用 blend/bridge 会导致复现困难。",
        "mechanism": "Mixing vs bridging taxonomy; frozen experts; preliminary kick residual + walk–jump transition.",
        "metrics": "Preliminary experiments（研究立场 + 初步结果，非完整统一算法）。",
        "conclusion": "本文提供腿式 RL 技能组合的概念框架，工程落地仍待更完整算法与基准。",
        "related": [
            "../methods/reinforcement-learning.md",
            "../methods/residual-policy-learning.md",
            "../tasks/locomotion.md",
        ],
        "abbrev": [
            ("RL", "Reinforcement Learning", "强化学习"),
            ("MoE", "Mixture of Experts", "专家混合"),
            ("WBC", "Whole-Body Control", "全身控制"),
        ],
    },
    {
        "slug": "ga-biped-slope-gait",
        "short": "GA 坡面双足",
        "title": "Walking on the Slope: Stable Bipedal Gaits with Genetic-Algorithm-Optimized Trajectories",
        "arxiv": "2609.20570",
        "blog": BLOG_HQ,
        "open": "待发布",
        "tags": ["paper", "biped", "locomotion", "zmp"],
        "one_liner": "8-DoF 运动学 + Newton–Euler 动力学；GA 优化三项轨迹参数 + ZMP 惩罚；最快 0.5 s 步周期、最高 22.5° 坡面仿真稳定。",
        "why": "坡面双足需轨迹级 ZMP 可行优化；GA 可探索非凸步态参数空间。",
        "mechanism": "GA trajectory optimization with ZMP feasibility penalty on 8-DoF biped model.",
        "metrics": "Sim: 0.5 s step period, up to 22.5° slope stability boundary.",
        "conclusion": "GA+ZMP 为坡面双足提供仿真级轨迹优化基线，真机迁移未报告。",
        "related": [
            "../formalizations/zmp-lip.md",
            "../tasks/humanoid-locomotion.md",
            "../tasks/locomotion.md",
        ],
        "abbrev": [
            ("GA", "Genetic Algorithm", "遗传算法"),
            ("ZMP", "Zero Moment Point", "零力矩点"),
            ("DoF", "Degrees of Freedom", "自由度"),
        ],
    },
    {
        "slug": "mechanical-intelligence-info-theory",
        "short": "机械智能信息论",
        "title": "Quantifying Mechanical Intelligence in Legged Robots with Information Theory",
        "arxiv": "2609.19588",
        "blog": BLOG_HQ,
        "open": "待发布",
        "tags": ["paper", "locomotion", "actuator", "information-theory"],
        "one_liner": "把身体动力学视为计算与通信信道；信息论指标量化机械模态与坐标间信息处理；比较 SEA vs 低减速比本体感知执行器及 RL 四足复杂地形。",
        "why": "「机械智能」缺乏可比较度量；信息论提供跨本体/执行器分析语言。",
        "mechanism": "Information-theoretic metrics on body dynamics as computation + communication.",
        "metrics": "SEA vs proprioceptive low-ratio actuators; learned quadruped rough terrain sim.",
        "conclusion": "本文提出用信息论量化腿式机械智能，适合执行器选型与 body design 讨论。",
        "related": [
            "../methods/reinforcement-learning.md",
            "../tasks/locomotion.md",
            "../concepts/humanoid-knee-harmonic-drive-limits.md",
        ],
        "abbrev": [
            ("SEA", "Series Elastic Actuator", "串联弹性执行器"),
            ("RL", "Reinforcement Learning", "强化学习"),
            ("IT", "Information Theory", "信息论"),
        ],
    },
    {
        "slug": "savla",
        "short": "SAVLA",
        "title": "SAVLA: Symmetry-Aware Vision-Language-Action Models for Robotic Manipulation",
        "arxiv": "2609.16641",
        "blog": BLOG_MAN,
        "open": "待发布",
        "tags": ["paper", "vla", "manipulation", "equivariance", "flow-matching"],
        "one_liner": "冻结 VLM backbone，Action Head 用 equivariant Flow Matching + learned canonicalizer；LIBERO 平均 +5.1pp，Goal 旋转测试 41.5%→90.4%。",
        "why": "旋转泛化靠数据增强昂贵；结构等变比堆数据更高效。",
        "mechanism": "Invariant/equivariant channels + canonicalizer + equivariant flow matching action head.",
        "metrics": "LIBERO +5.1pp vs GR00T N1.5 avg; LIBERO-Goal rotation 90.4%（作者报告）。",
        "conclusion": "SAVLA 用几何等变结构替代旋转数据增强，显著提升 Goal 旋转泛化。",
        "related": ["../methods/vla.md", "../tasks/manipulation.md", "../entities/isaac-gr00t.md"],
        "abbrev": [
            ("VLA", "Vision-Language-Action", "视觉–语言–动作"),
            ("FM", "Flow Matching", "流匹配"),
            ("SE(3)", "Special Euclidean Group", "三维刚体变换群"),
        ],
    },
    {
        "slug": "swim-vla",
        "short": "SWIM",
        "title": "SWIM: Vision-Language-Grounded Soft Whole-Body Interactive Manipulation",
        "arxiv": "2609.17035",
        "blog": BLOG_MAN,
        "open": "待发布",
        "tags": ["paper", "vla", "soft-robot", "manipulation"],
        "one_liner": "RGB+语言+tendon state → Diffusion Action Head 整段 tendon chunk；Visual Soft Proprioception 保留身体几何；仿真 packing/reaching/grasping 100%/96%/88% + 真机。",
        "why": "软体操作需 tendon 状态与视觉本体融合；刚性 VLA 假设不适用。",
        "mechanism": "SWIM-VLA unified encoding + diffusion tendon chunks + visual soft proprioception.",
        "metrics": "Sim 100%/96%/88%; real robot validation（作者报告）。",
        "conclusion": "SWIM 把 VLA 扩展到软体全身交互，tendon 与视觉软本体是关键输入。",
        "related": [
            "../methods/vla.md",
            "../tasks/manipulation.md",
            "../concepts/robot-simulation-three-layers.md",
        ],
        "abbrev": [
            ("VLA", "Vision-Language-Action", "视觉–语言–动作"),
            ("DoF", "Degrees of Freedom", "自由度"),
            ("BC", "Behavior Cloning", "行为克隆"),
        ],
    },
    {
        "slug": "xpace",
        "short": "XPACE",
        "title": "XPACE: Joint World and Action Modeling from Heterogeneous Experience",
        "arxiv": "2609.17372",
        "blog": BLOG_MAN,
        "open": "待发布",
        "tags": ["paper", "wam", "humanoid", "xpeng", "self-improvement"],
        "one_liner": "WAM+Simulator 合一：联合预测视频与动作或给定动作预测视觉；无动作视频学动态；Simulator 生成 deviation→recovery 再微调 Policy；小鹏 IRON 真机。",
        "why": "异构经验（人视频+机器人示范）需统一世界–动作模型；自生成恢复数据可闭环提升。",
        "mechanism": "Shared video backbone; world-action joint prediction; simulator-driven recovery data self-improvement.",
        "metrics": "XPENG IRON humanoid; human video skill transfer + recovery data gains（以 PDF 为准）。",
        "conclusion": "XPACE 把 world simulator 用作 policy self-improvement 引擎，连接人视频与机器人示范。",
        "related": [
            "../concepts/world-action-models.md",
            "../methods/generative-world-models.md",
            "../entities/isaac-gr00t.md",
        ],
        "abbrev": [
            ("WAM", "World Action Model", "世界–动作模型"),
            ("VLA", "Vision-Language-Action", "视觉–语言–动作"),
            ("IL", "Imitation Learning", "模仿学习"),
        ],
    },
    {
        "slug": "mpc-scaffolding-dex-rl",
        "short": "MPC 脚手架灵巧 RL",
        "title": "Real-World Reinforcement Learning with MPC Scaffolding for Dexterous Manipulation",
        "arxiv": "2609.14878",
        "blog": BLOG_MAN,
        "open": "待发布",
        "tags": ["paper", "dexterous-manipulation", "rl", "mpc", "allegro"],
        "one_liner": "Sampling MPC 初始化 buffer 并预训练；在线 SAC 与 MPC 共训，逐渐交权；16-DoF Allegro 手内旋转 7 min 达 5/5，20 min 速度超 MPC 5×、1000 次旋转。",
        "why": "真机 dex RL 探索难；MPC scaffolding 提供安全探索与初始化。",
        "mechanism": "MPC trajectories init replay + online SAC with gradual handoff from MPC to policy.",
        "metrics": "Allegro in-hand rotation: 5/5 in 7 min online; 1000 rotations / 110+ min（作者报告）。",
        "conclusion": "MPC scaffolding 使无示范真机 dex RL 在分钟级达到超 MPC 吞吐。",
        "related": [
            "../methods/reinforcement-learning.md",
            "../tasks/manipulation.md",
            "../methods/model-predictive-control.md",
        ],
        "abbrev": [
            ("MPC", "Model Predictive Control", "模型预测控制"),
            ("SAC", "Soft Actor-Critic", "软 actor-critic"),
            ("RL", "Reinforcement Learning", "强化学习"),
        ],
    },
    {
        "slug": "graphpoint",
        "short": "GraphPoint",
        "title": "GraphPoint: Semantic Entity Graphs and Point Trajectories for Compositional Robot Manipulation",
        "arxiv": "2609.18358",
        "blog": BLOG_MAN,
        "open": "待发布",
        "tags": ["paper", "manipulation", "compositional-generalization", "visuomotor"],
        "one_liner": "CoMani Benchmark 测组合泛化；Semantic Entity Graph + gripper point trajectory + progress 预测驱动 subtask transition。",
        "why": "组合泛化需显式语义–几何接口，而非端到端低层动作。",
        "mechanism": "Entity graph → point trajectories → robot geometry actions; progress for subtask transitions.",
        "metrics": "CoMani compositional generalization benchmark（以 PDF 为准）。",
        "conclusion": "GraphPoint 用实体图与点轨迹桥接语言组合泛化与几何控制。",
        "related": [
            "../tasks/manipulation.md",
            "../methods/vla.md",
            "../concepts/behavior-tree-vla-orchestration.md",
        ],
        "abbrev": [
            ("VLA", "Vision-Language-Action", "视觉–语言–动作"),
            ("SR", "Success Rate", "成功率"),
            ("HOI", "Human-Object Interaction", "人–物交互"),
        ],
    },
    {
        "slug": "maniskillformer",
        "short": "ManiSkillFormer",
        "title": "ManiSkillFormer: Demonstration-Free Compositional Manipulation via Task-Conditioned Geometric Contracts",
        "arxiv": "2609.16331",
        "blog": BLOG_MAN,
        "open": "待发布",
        "tags": ["paper", "manipulation", "bimanual", "llm", "compositional"],
        "one_liner": "Task-Conditioned Geometric Contract 声明 keypoint/normal 等 primitive；LLM 生成 contract+模板，视觉定位后实例化；Galaxea R1-Lite 无示范 Pick-and-Place 88.24%。",
        "why": "无示范组合操作需可验证几何契约而非黑盒 policy。",
        "mechanism": "LLM contracts + motion templates + 3D primitive grounding on dual-arm platform.",
        "metrics": "Pick-and-place 88.24% avg without per-object demos（作者报告）。",
        "conclusion": "ManiSkillFormer 用几何契约 + LLM 模板实现无示范双臂组合操作。",
        "related": [
            "../tasks/manipulation.md",
            "../methods/vla.md",
            "../tasks/loco-manipulation.md",
        ],
        "abbrev": [
            ("LLM", "Large Language Model", "大语言模型"),
            ("BC", "Behavior Cloning", "行为克隆"),
            ("EE", "End-Effector", "末端执行器"),
        ],
    },
    {
        "slug": "holistic-biped-loco-manip",
        "short": "双足整体 loco-manip",
        "title": "Learning Holistic Whole-Body Loco-Manipulation with a Bipedal Mobile Manipulator",
        "arxiv": "2609.18930",
        "blog": BLOG_MAN,
        "open": "待发布",
        "tags": ["paper", "loco-manipulation", "biped", "rl", "diffusion-policy"],
        "one_liner": "统一 WBC 仅输入 6-DoF 末端目标；reward gating 平衡跟踪/移动/平衡；Transformer+GRU+dynamics aux；真机可接 VR/Diffusion/scripted 末端命令。",
        "why": "双足移动操作常分离 base 与 arm 命令；整体策略简化上层接口。",
        "mechanism": "Single RL whole-body policy from EE target only; reward gating; history encoder with dynamics prediction.",
        "metrics": "Real bipedal mobile manipulator; multi high-level command sources（以 PDF 为准）。",
        "conclusion": "本文展示双足移动操作可由单一低层 WBC 消化多样上层末端指令。",
        "related": [
            "../tasks/loco-manipulation.md",
            "../methods/reinforcement-learning.md",
            "../methods/diffusion-policy.md",
        ],
        "abbrev": [
            ("WBC", "Whole-Body Control", "全身控制"),
            ("EE", "End-Effector", "末端执行器"),
            ("RL", "Reinforcement Learning", "强化学习"),
        ],
    },
    {
        "slug": "project-kitchen",
        "short": "Project Kitchen",
        "title": "From Gameplay to Policy: Towards Scalable Robot Data Collection via Gamified Robot-Free Interaction",
        "arxiv": "2609.18650",
        "blog": BLOG_MAN,
        "open": "待发布",
        "tags": ["paper", "imitation-learning", "data-collection", "cross-embodiment"],
        "one_liner": "VR 游戏化无机器人数据采集；Game2Policy 提取 embodiment-invariant affordance 预训练，再 few-shot 真机联合微调；仿真 +10pp、真机 +18.3pp。",
        "why": "机器人数据贵；游戏化跨本体数据可放大预训练再 few-shot 落地。",
        "mechanism": "Project Kitchen VR game → affordance pretrain → few-shot real robot co-fine-tune.",
        "metrics": "Sim +10.0pp, real +18.3pp few-shot（作者报告）。",
        "conclusion": "Project Kitchen 把 scalable 人类游戏数据接入 cross-embodiment few-shot 策略学习。",
        "related": [
            "../concepts/data-flywheel.md",
            "../methods/imitation-learning.md",
            "../tasks/manipulation.md",
        ],
        "abbrev": [
            ("VR", "Virtual Reality", "虚拟现实"),
            ("IL", "Imitation Learning", "模仿学习"),
            ("BC", "Behavior Cloning", "行为克隆"),
        ],
    },
    {
        "slug": "bench2dex",
        "short": "Bench2Dex",
        "title": "Bench2Dex: Benchmarking Visuo-Tactile Bimanual Dexterous Manipulation Across Dexterous Hands",
        "arxiv": "2609.15726",
        "blog": BLOG_MAN,
        "open": "待发布",
        "tags": ["paper", "benchmark", "dexterous-manipulation", "bimanual", "vla"],
        "one_liner": "Isaac Lab 统一 visuotactile 双臂基准：12 种灵巧手、26 任务、~1300 遥操作 demo；7 类扰动分 invariance/equivariance；评测 ACT/DP/π0.5/GR00T N1.5。",
        "why": "跨手型灵巧操作缺统一 visuotactile 双臂基准与扰动轴。",
        "mechanism": "Unified Isaac Lab benchmark + human demos + perturbation axes + multi-policy eval.",
        "metrics": "12 hands, 26 tasks, ~1300 demos; ACT/DP/π0.5/GR00T N1.5 comparison.",
        "conclusion": "Bench2Dex 为跨灵巧手 visuotactile 双臂策略提供可复现横评底座。",
        "related": ["../entities/isaac-lab.md", "../methods/vla.md", "../tasks/manipulation.md"],
        "abbrev": [
            ("VLA", "Vision-Language-Action", "视觉–语言–动作"),
            ("BC", "Behavior Cloning", "行为克隆"),
            ("DP", "Diffusion Policy", "扩散策略"),
        ],
    },
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
summary: "{p["short"]}（arXiv:{arxiv}）：{p["one_liner"][:120]}"
---

# {p["short"]}（arXiv:{arxiv}）

**{p["short"]}**（*{p["title"]}*，[arXiv:{arxiv}](https://arxiv.org/abs/{arxiv})）来自 [senlanke 具身运控lab 周更盘点]({src_blog})（2026-09-14–18）。

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

{chr(10).join(f"- [{r.split('/')[-1].replace('.md', '')}]({r})" if r.startswith("../") else f"- {r}" for r in related_wiki)}

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
        # insert before summary if no sources block
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

    REUSE_BLOG = {
        "2609.18207": BLOG_MAN,
        "2609.16586": BLOG_MAN,
        "2609.18620": BLOG_MAN,
        "2609.16683": BLOG_MAN,
        "2609.17210": BLOG_MAN,
    }
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
