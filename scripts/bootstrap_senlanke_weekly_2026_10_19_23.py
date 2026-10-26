#!/usr/bin/env python3
"""Bootstrap ingest: senlanke 具身运控lab 10.19-10.23 前沿论文动态（人形/四足 39 + Manipulation 37）.

两篇公众号用 scripts/fetch_wechat_article.py（Chromium + 微信 UA）读取；条目 72 个唯一 arXiv
（4 篇跨栏重复），其中 24 篇已有详情节点（复用，仅追加来源），48 篇新建独立 paper-* 节点。
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TODAY = date.today().isoformat()
PUB = "2026-10-26"
WEEK = "10-19–10-23"
BLOGS = {
    "hq": {
        "file": "wechat_senlanke_weekly_humanoid_quadruped_2026-10-19_23.md",
        "title": "【10.19-10.23前沿论文动态】人形／四足39篇",
        "url": "https://mp.weixin.qq.com/s/DBARqsgZl-RdP1LYvAS2rA",
        "overview": "senlanke-weekly-2026-10-19-23-humanoid-quadruped-technology-map",
        "label": "人形/四足",
    },
    "man": {
        "file": "wechat_senlanke_weekly_manipulation_2026-10-19_23.md",
        "title": "【10.19-10.23前沿论文动态】Manipulation 37篇",
        "url": "https://mp.weixin.qq.com/s/kn-3vRhkVME86YZupnuSmQ",
        "overview": "senlanke-weekly-2026-10-19-23-manipulation-technology-map",
        "label": "Manipulation",
    },
}

SEC_META = {
    "人形机器人": ("humanoid", ["paper", "humanoid"], ["../tasks/humanoid-locomotion.md", "../methods/reinforcement-learning.md"]),
    "四足机器人": ("quadruped", ["paper", "quadruped", "locomotion"], ["../tasks/locomotion.md", "../methods/reinforcement-learning.md"]),
    "轮足与通用腿式机器人": ("legged", ["paper", "legged-robot", "locomotion"], ["../tasks/locomotion.md"]),
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

DATA = json.loads(r'''{
 "papers": {
  "2610.25711": {
   "id": "2610.25711",
   "title": "SWAP: Test-Time Steerable Humanoid Whole-Body Collision Avoidance",
   "zh": "SWAP：测试时可引导的人形机器人全身避碰",
   "org": "卡内基梅隆大学、Amazon Frontier AI & Robotics",
   "prob": "人形机器人通过狭窄、拥挤空间时，需要同时处理全身碰撞、动态平衡以及多种可能的绕障动作。",
   "inno": "用感知条件扩散策略生成多模态全身运动方案，由底层 RL 控制器负责平衡与跟踪；通过事后标注自动生成无碰撞训练数据，并在测试时用代价函数引导采样，无需重新训练即可调整速度、朝向和姿态。",
   "reuse": "wiki/entities/paper-swap-humanoid-collision-avoidance.md"
  },
  "2610.25701": {
   "id": "2610.25701",
   "title": "Validation of Human Motor Development Scales with Humanoid Robots",
   "zh": "利用人形机器人验证人类运动发育量表",
   "org": "伍斯特理工学院",
   "prob": "人形机器人学习从躺卧、翻身到站立和行走的一系列技能时，缺少有科学依据的课程顺序。",
   "inno": "把四种婴幼儿运动发育量表转化为技能依赖图并用于 RL 课程学习；通过失败分析发现缺失的「跪姿到站立」里程碑，并验证补齐节点后可恢复完整技能学习。",
   "reuse": null,
   "short": "MotorDev-RL",
   "slug": "motor-development-scales-humanoid-curriculum",
   "abstract": "We investigate how humanoid robots can learn a series of motor skills defined by human infant motor development scales, and we evaluate which scales are comprehensive enough to serve as self-contained guides for developing gross motor skills up to standing and walking. We represent each motor development scale as a developmental graph that guides the training of a humanoid robot. We then apply this graph within a reinforcement learning framework so that the robot learns skills following the natural phases of human motor development described by the scale. Our work demonstrates that graphs derived from three out of four popular pediatric motor scales can successfully guide humanoid robots to develop these skills. The fourth scale misses some intermediate milestones that the other scales have, which causes some failures. After adding a missing milestone in its graph, it also enables the robot to develop those skills successfully.",
   "comments": "8 pages, 7 figures. Submitted to ICRA 2027. Video available at this https URL",
   "authors": "Fanxing Meng, Jing Xiao",
   "links": [
    "https://youtu.be/SR3LUVhohV4"
   ]
  },
  "2610.26043": {
   "id": "2610.26043",
   "title": "OmniInteract: Towards Unified Whole-Body Interaction Control for Dual Humanoids",
   "zh": "OmniInteract：面向双人形机器人的统一全身交互控制",
   "org": "中山大学、深圳河套学院、北京理工大学、香港科技大学",
   "prob": "双人形交互中的接触关系容易在人体—机器人动作重定向时被破坏，而且交互种类增加后，单一策略容易丧失已有运动能力。",
   "inno": "通过稠密表面对应保持双方接触界面和邻域几何关系；冻结基础运动先验，由交互专家学习残差，再用条件流匹配将多个交互技能统一到共享残差空间。",
   "reuse": "wiki/entities/paper-omniinteract-dual-humanoids.md"
  },
  "2610.26169": {
   "id": "2610.26169",
   "title": "Compositional Humanoid Manipulation through Isomorphic Teleoperation and Independent Skill Routing",
   "zh": "基于同构遥操作与独立技能路由的组合式人形操作",
   "org": "上海交通大学、香港大学、Xense Robotics、复旦大学",
   "prob": "分别采集的左右臂技能难以自由组合，而为每种双臂技能组合重新示范成本过高。",
   "inno": "提出共享专家池、左右臂独立语义路由的扩散策略；设计 G1Exo 同构外骨骼直接采集关节级接触操作数据，并通过语义调度器交替执行操作与身体重定位。",
   "reuse": null,
   "short": "G1Exo 组合操作",
   "slug": "isomorphic-teleop-skill-routing-humanoid",
   "abstract": "Humanoid manipulation requires reusable arm skills, yet demonstrating every possible skill combination is impractical. We present Humanoid Semantic Mixture-of-Experts Diffusion Policy (HS-MoDP), which composes separately demonstrated arm skills through independent semantic routing over a shared expert pool. This design preserves arm-specific skill control while sharing action-generation parameters across arms. A semantic scheduler integrates the learned policy with an existing locomotion controller through alternating manipulation and repositioning stages, extending the effective workspace and enabling skill reuse across location changes. To acquire demonstrations of whole-body contact manipulation, we develop G1Exo, an isomorphic exoskeleton that provides direct joint-level control of arm configurations. Experiments on a Unitree G1 show stronger skill composition than the baselines. G1Exo enables more reliable contact teleoperation than end-effector VR-IK, and its demonstrations improve contact-skill learning across four policy architectures. For HS-MoDP, switching from VR-IK to G1Exo demonstrations raises full cabinet workflow success from 35% to 65%. All project hardware designs and source code will be released as open source after the review process.",
   "comments": "",
   "authors": "Yiming Wang, Peixian Yang, Lifeng Zhuo, Lai Wei, Xinling Du, Kun Song, Weiming Wang, Daolin Ma",
   "links": []
  },
  "2610.26170": {
   "id": "2610.26170",
   "title": "Data-Efficient Learning from Demonstration for Humanoid Robots via Feature-Level Perception and Prompt-Based Object Detection",
   "zh": "基于特征级感知与提示式目标检测的人形机器人数据高效示范学习",
   "org": "德黑兰大学、奥克兰大学",
   "prob": "从原始图像端到端学习人形操作通常需要大量示范和计算资源。",
   "inno": "从视频中提取人体关节、提示词检测结果和三维物体位置等紧凑特征，再预测 NAO 的关节轨迹；在极少示范条件下完成推物和抓放任务。",
   "reuse": null,
   "short": "特征级示范学习",
   "slug": "feature-level-lfd-prompt-detection-nao",
   "abstract": "Learning from Demonstration (LfD) allows robots to learn new skills from human examples; however, many vision-based methods require large datasets and resource-intensive end-to-end training. This paper introduces a data-efficient LfD framework for humanoid robots that learns manipulation skills from compact, task-specific features instead of raw images. The proposed approach integrates layered visual preprocessing, Kinect-based human joint estimation, prompt-based object detection, and depth information to extract human motion features and three-dimensional object locations from demonstration videos. These features are used to train a temporal deep learning model that predicts robot joint-angle trajectories for execution. The framework is implemented on a NAO humanoid robot, evaluated quantitatively in simulation, and demonstrated on physical hardware. In simulation, the robot achieves a 78% success rate on a pick-and-place task using only 16 demonstrations, which increases to 86% when trained with 100 demonstrations. The framework also achieves a 71% success rate on a pushing task using eight demonstrations, highlighting its effectiveness for data-efficient humanoid skill acquisition.",
   "comments": "6 pages, 4 figures, 2 tables. Presented at IEEE ISAIA 2026",
   "authors": "Keyhan Rayati, AliReza Beigy, Anoosheh Saadati, Pourya Shahverdi, Mehdi Tale Masouleh, Ahmad Kalhor, Wing-Yue Geoffrey Louie",
   "links": []
  },
  "2610.26584": {
   "id": "2610.26584",
   "title": "HAF-VLA: Hierarchical Action Flow Framework for Humanoid Vision-Language-Action Control",
   "zh": "HAF-VLA：面向人形视觉—语言—动作控制的分层动作流框架",
   "org": "北京大学、北京人形机器人创新中心、南开大学、西安交通大学、清华大学",
   "prob": "VLA 一次性生成完整全身动作时，底盘或腰部的小误差会引发上肢激烈补偿并破坏平衡。",
   "inno": "在同一动作空间内分阶段生成运动与头部、腰部和全身操作动作；使用跨阶段 KV 缓存保留前一阶段的动作结构，使后续阶段进行条件化细化。",
   "reuse": "wiki/entities/paper-haf-humanoid-vla-adaptation.md"
  },
  "2610.26855": {
   "id": "2610.26855",
   "title": "Ropedia W0: Scaling Loco-Manipulation World Models across Embodiments with Human Experience",
   "zh": "Ropedia W0：利用人类经验扩展跨形态移动操作世界模型",
   "org": "南洋理工大学、Ropedia、杜克大学、加州大学圣迭戈分校、清华大学",
   "prob": "人类与不同机器人具有异构动作空间，难以训练统一的移动操作世界模型。",
   "inno": "用相机位姿表示移动、用投影骨架表示操作，统一不同形态；构建约 1.3 万小时人类与机器人第一视角数据，蒸馏后支持实时、分钟级世界模型展开。",
   "reuse": "wiki/entities/paper-ropedia-w0.md"
  },
  "2610.27162": {
   "id": "2610.27162",
   "title": "Visual Task Modeling and Automated Planning for Humanoid Robots in Education of Individuals with Autism Spectrum Disorder",
   "zh": "面向孤独症教育人形机器人的视觉任务建模与自动规划",
   "org": "查理大学",
   "prob": "教育机器人需要仅根据教具外观理解任务目标，并规划可向学生演示的操作步骤。",
   "inno": "使用视觉大模型把 Shoebox Task 照片转换为 PDDL 问题，再自动生成计划；NAO 可据此演示下一步、发出语音提示并检测操作失败。",
   "reuse": null,
   "short": "孤独症教育 NAO 规划",
   "slug": "autism-education-humanoid-visual-task-planning",
   "abstract": "In robot-assisted education, a robot must understand how to complete a task in order to effectively guide a student through it. We present a novel method that enables a semi-autonomous, camera-equipped robot to recognise the task goal and plan actions to achieve it. Our work focuses on Shoebox Tasks, an activity commonly used in the education of individuals with Autism Spectrum Disorder (ASD). These tasks are designed so that the goal can be inferred directly from their visual structure. We have developed a solver that uses a vision-based Large Language Model to convert photos of a Shoebox Task into a Planning Domain Definition Language (PDDL) representation and generate action plans for solving it. We have also implemented a robotic interface that enables a Nao robot to autonomously execute various Shoebox Tasks and assist a student by suggesting next steps or signalling detected failures through action demonstrations, voice commands, or gestures. With an adapted robotic interface, the solver can be integrated into a variety of educational robots. No programming expertise is required from teachers, and the solver operates exclusively on captured images as input. Research in special educational needs indicates that students with ASD can benefit substantially from robot-assisted teaching, and our system aims to strengthen these learning opportunities while also fostering interest in robotics among individuals with ASD.",
   "comments": "6 pages, presented at conference AICCONF 2026 in Prague, Czech Republic",
   "authors": "Lada Kudláčková",
   "links": []
  },
  "2610.27242": {
   "id": "2610.27242",
   "title": "EmbodyHOI: Embodiment-Aware Generation of Hand-Object Interactions for Humanoid Dexterous Manipulation",
   "zh": "EmbodyHOI：面向人形灵巧操作的形态感知手物交互生成",
   "org": "卡内基梅隆大学",
   "prob": "在人手空间生成的手物交互轨迹，事后映射到机器人时容易出现不可达、接触脱离和策略训练质量下降。",
   "inno": "在冻结的流匹配生成器采样过程中嵌入可微机器人 IK，以机器人可达性梯度直接修正轨迹和机器人相对位置；同一生成器可适配 G1 和 H1-2 而无需重训。",
   "reuse": "wiki/entities/paper-embodyhoi.md"
  },
  "2610.27688": {
   "id": "2610.27688",
   "title": "Adaptive Traversal Tubes for Posture-Aware Navigation of Bipedal Humanoids",
   "zh": "面向双足人形姿态感知导航的自适应通行管道",
   "org": "作者未注明机构",
   "prob": "传统导航只规划路径和速度，无法主动利用侧身姿态穿越窄通道或为行人让路。",
   "inno": "用沿路径变化的通行宽度统一表示几何净空和社交距离，再将其映射成正向—侧向行走姿态及全向速度；配套训练侧步增强的 AMP 运动先验。当前版本的定量评测仍待补充。",
   "reuse": null,
   "short": "Traversal Tubes",
   "slug": "adaptive-traversal-tubes-bipedal-navigation",
   "abstract": "Bipedal humanoids can change their effective footprint by adjusting body yaw, enabling both narrow-passage traversal and socially aware yielding. This paper presents a posture-aware navigation framework centered on \\emph{Adaptive Traversal Tubes} (ATT), \\emph{Interval Velocity--Heading} (IVH) control, and a sidestep-enhanced AMP locomotion prior. ATT represents a behavior-feasible corridor width field along a reference path, unifying geometric clearance and social comfort into a continuous signal. IVH maps this width to a continuous frontal-to-sidestep posture schedule and omnidirectional velocity commands. A velocity-conditioned AMP policy, trained with GVHMR-retargeted frontal/sidestep motions and coupled \\((v_x,v_y)\\) tracking rewards, is designed to improve transition stability over a generic locomotion controller while remaining interchangeable with the default onboard policy. This preprint presents the full method; a planned evaluation protocol on a Unitree G1 is outlined, with quantitative results to follow. Demo videos of MuJoCo simulation and real-robot deployment are available at: this http URL.",
   "comments": "",
   "authors": "Zijian Ge",
   "links": [
    "http://github.com/lborogzj997/Humanoid-Navigation-in-Crowds-and-Narrow-Spaces"
   ]
  },
  "2610.28470": {
   "id": "2610.28470",
   "title": "Fast Motion Retargeting Pipeline for Scalable Humanoid Robot Learning",
   "zh": "面向规模化人形学习的快速动作重定向流程",
   "org": "马萨诸塞大学阿默斯特分校",
   "prob": "将大型人体动作库转移到不同人形机器人时，传统轨迹优化速度慢，还需要逐动作人工修复。",
   "inno": "结合定制 SQP、实时迭代和分块并行，将速度提升至最高每秒 9156 帧；再用接触隐式 iLQR 筛除或修正动力学不可执行、悬脚和滑脚动作。",
   "reuse": null,
   "short": "Fast Retargeting",
   "slug": "fast-motion-retargeting-pipeline-humanoid",
   "abstract": "While large-scale human motion capture datasets provide a rich source of demonstrations for humanoid robot learning, retargeting the diverse motions to different humanoid robots remains a nontrivial challenge. Existing kinodynamic trajectory optimization pipelines are computationally expensive and often require manual adjustments per motion. We present FRDV (Fast Retargeting with Dynamics Validation), a scalable framework consisting of rapid constrained kinematic retargeting and an additional dynamics validation pipeline. Its kinematic stage integrates a custom Sequential Quadratic Programming solver, Real-Time Iteration, and parallel chunk-based processing, achieving up to 9,156 frames per second, approximately up to 136 times faster than GMR and ProtoMotions, while robot-specific constraints are incorporated into the optimization to reduce common retargeting artifacts, such as ground penetration and self-collision. For motions requiring additional verification, a contact implicit iLQR controller is utilized to test the retargeted reference under full rigid body dynamics. This dynamics-based validation identifies dynamically infeasible motions and refines contact-related artifacts such as foot floating and foot skating. We evaluate retargeting quality by training motion tracking reinforcement learning policies using several state-of-the-art pipelines and comparing their tracking fidelity and success rates. Furthermore, sim-to-real transfer is demonstrated on a Unitree G1 humanoid. Videos and code are available at: this https URL.",
   "comments": "",
   "authors": "Daniel Marew, Sangjun Park, Nisal Perera, Donghyun Kim",
   "links": [
    "https://daroslab.github.io/frdv/"
   ]
  },
  "2610.28480": {
   "id": "2610.28480",
   "title": "SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model",
   "zh": "SuperMimic：利用单一可微世界模型扩展人形全身控制",
   "org": "加州大学伯克利分校",
   "prob": "在十万级动作库上训练人形跟踪器需要巨量仿真采样和 GPU 资源，专家蒸馏又会损失动作覆盖率。",
   "inno": "从刚体仿真日志学习统一可微动力学模型，通过短时想象轨迹反向传播解析梯度；单 GPU 训练 3 小时即可超过大型 PPO 基线，将 GPU 时降低超过 1000 倍，并迁移到 Unitree G1。",
   "reuse": "wiki/entities/paper-supermimic.md"
  },
  "2610.28523": {
   "id": "2610.28523",
   "title": "VLM-Guided Contact Retargeting and Generative Partner Modeling for Humanoid-Human Interaction",
   "zh": "面向人形—人交互的 VLM 引导接触重定向与生成式伙伴建模",
   "org": "卡内基梅隆大学、庆应义塾大学、Keio AI Research Center",
   "prob": "近距离人机接触会造成严重遮挡，使人体姿态估计失效；人类示范数据本身也可能因互相遮挡而产生错误。",
   "inno": "先由 VLM 识别语义接触并进行接触保持重定向，再将可访问伙伴真值的特权教师蒸馏为仅使用机载深度的学生；通过生成式伙伴状态预测处理视野外和遮挡人体。",
   "reuse": null,
   "short": "VLM 接触重定向",
   "slug": "vlm-contact-retargeting-partner-modeling",
   "abstract": "Humanoid-human interaction requires robots to operate under severe partial observability: during close physical contact, the human partner is occluded, truncated, or outside the robot's field of view, causing pose estimation to fail and often necessitating external tracking or privileged human state. At the same time, heavy mutual occlusions introduce errors into the human-human motion-capture data used for learning. We present REACT, a framework for learning deployable humanoid-human interaction policies from human-human demonstrations. REACT treats semantic contact structure as the key supervisory signal connecting demonstration processing, interaction learning, and onboard control. First, a VLM-guided contact modeling and retargeting pipeline identifies semantically meaningful contacts and transfers them onto humanoid-human geometry while preserving interaction structure. Using these annotations, privileged interaction policies learn physically grounded behaviors through interaction-aware contact rewards and are then distilled into deployable egocentric depth policies trained entirely from onboard sensing using photorealistic human depth rendering. To address severe partial observability, we introduce generative partner-state modeling objectives that predict partner motion from occluded egocentric views. We evaluate REACT in simulation and on a real Unitree G1 humanoid across five contact-rich interactions. Semantic contact modeling substantially improves preservation of demonstrated interactions, while generative partner-state modeling improves contact recall on four of five interactions, most under heavy occlusion. On real hardware, REACT performs all five interactions using onboard sensing alone.",
   "comments": "CoRL2026, project page: this https URL",
   "authors": "Yuto Shibata, Kotaro Amaya, Kashu Yamazaki, Lalit Jayanti, Yoshimitsu Aoki, Mariko Isogawa, Katerina Fragkiadaki",
   "links": [
    "https://yutoshibata07.github.io/REACT/"
   ]
  },
  "2610.28712": {
   "id": "2610.28712",
   "title": "Warping the Workspace: Expanding Visual Access with Adjustable Reach Mapping for Humanoid Teleoperation",
   "zh": "工作空间扭曲：通过可调触达映射扩展人形遥操作视觉范围",
   "org": "东京大学、Sony Computer Science Laboratories 京都",
   "prob": "第一视角遥操作时，侧后方目标通常要求操作者和机器人同步大幅转身，降低操作效率。",
   "inno": "联合设计全景方位角视觉扭曲和逆向手臂偏航映射，使侧后方目标被压缩到正面视野，同时保持机器人手臂指向真实目标。",
   "reuse": null,
   "short": "Warping the Workspace",
   "slug": "warping-workspace-adjustable-reach-teleop",
   "abstract": "First-person humanoid teleoperation ties access to side and rear targets to operator rotation. We present coupled visual--motor remapping, pairing a panoramic azimuth warp with its inverse upper-limb yaw transform. A working prototype implements Global Zoom (GZ) and Focus+Context (F+C). In a within-subject study (N=8); 72 trials), GZ had the largest completion count (23/24; Fixed: 20/24; F+C: 21/24), whereas F+C was preferred by 6/8 participants. Therefore, completion and preference favored different mappings in this sample.",
   "comments": "",
   "authors": "Keigo Minamida, Koya Dendo, Yuto Nakamura, Jun Rekimoto",
   "links": []
  },
  "2610.28925": {
   "id": "2610.28925",
   "title": "One Policy, Many Styles: Where to Aggregate Preferences for Steerable Humanoid Control?",
   "zh": "一个策略、多种风格：可引导人形控制应在何处聚合偏好？",
   "org": "佐治亚理工学院",
   "prob": "偏好条件策略可能虽然响应风格指令，却同时改变无关行为或损害任务性能。",
   "inno": "建立偏好响应度、覆盖率和效用联合指标，系统比较奖励、优势、损失及 PCGrad 层面的目标聚合；揭示早期聚合响应范围更大，后期聚合控制更精确。",
   "reuse": null,
   "short": "One Policy Many Styles",
   "slug": "one-policy-many-styles-preference-aggregation",
   "abstract": "Modern robotic agents are increasingly expected to support steerable control, where a single policy can adjust its behavioral style at runtime under changing user preferences. Preference-conditioned reinforcement learning provides this interface, but conditioning alone does not ensure that a requested behavior change is strong, does not leak into unrelated behaviors, and does not compromise task performance. Our evaluation combines Preference Reward Responsiveness (PRR) with coverage and utility metrics. We use these measures to compare reward-level, advantage-level, loss-level, and PCGrad-based aggregation across two humanoid-control frameworks. No aggregation level performs best across all environments and criteria. Experiments show that aggregating objectives early produces a wider range of behavioral responses but greater unintended changes in other objectives, while aggregating them later improves control precision at some cost to responsiveness.",
   "comments": "",
   "authors": "Ziqian Huang, Yuzhuo Chen, Shimin Zhang, Junyang He, Xinyue Annie Yang, Animesh Garg",
   "links": []
  },
  "2610.29169": {
   "id": "2610.29169",
   "title": "SMR: Learning Streaming Motion Retargeting for Humanoids from Human Semantic",
   "zh": "SMR：从人体动作语义学习流式人形动作重定向",
   "org": "复旦大学、上海创智学院、Logos Robotics Group",
   "prob": "几何重定向不保证动力学可执行，而双向全序列网络无法用于实时流式输入。",
   "inno": "先训练物理专家跟踪机器人感知的任务空间语义目标，再将专家实际执行轨迹蒸馏到严格因果 Transformer；部署时只使用当前和历史人体观测生成动作参考。",
   "reuse": null,
   "short": "SMR",
   "slug": "smr-streaming-motion-retargeting",
   "abstract": "Retargeting human motion to humanoids requires preserving motion semantics while accounting for the robot's physical constraints. Kinematic pipelines prescribe a robot trajectory through geometric matching, without establishing whether it can be executed under the robot's dynamics. Prior physics-supervised neural methods train bidirectional full-sequence models on physics-refined kinematic trajectories, remaining tied to prescribed kinematic solutions and requiring noncausal context. We propose SMR, a streaming neural motion retargeter learned from semantic-conditioned physics experts. We convert human motion into robot-aware task-space targets for body landmark positions, orientations, and velocities. Physics experts track these targets directly in simulation under contact, dynamics, and actuator constraints. We train a single strictly causal Transformer on their executed state trajectories that complete the motion and pass semantic and physical validity checks. At each incoming frame, the student uses only current and past human observations to generate a robot motion reference for a downstream whole-body controller. Experiments demonstrate that semantic-conditioned physics experts generate motion with higher semantic fidelity and physical validity than direct optimization-based retargeting, while the resulting streaming retargeter achieves improved closed-loop performance over kinematic and physics-refined baselines under a common whole-body controller.",
   "comments": "17 pages, 6 figures. Project page: this https URL",
   "authors": "Chun Gu, Juntong Zhang, Jingyu Li, Li Zhang",
   "links": [
    "https://logosintelligencelab.github.io/SMR/"
   ]
  },
  "2610.29612": {
   "id": "2610.29612",
   "title": "Watch Your Step: Anticipated Perception for Humanoid Locomotion",
   "zh": "注意脚下：面向人形运动的前瞻式感知",
   "org": "苏黎世联邦理工学院机器人系统实验室、ETH AI Center",
   "prob": "处理完整高程图既浪费计算，又迫使策略同时学习运动和视觉注意区域，复杂地形训练效率较低。",
   "inno": "由轻量落脚预测器先根据本体状态和速度指令预测未来落脚点，策略只接收落脚点周围的局部高程图；训练迭代量约降至完整地图方法的三分之一，并零样本跨越楼梯、沟隙和台阶。",
   "reuse": "wiki/entities/paper-watch-your-step.md"
  },
  "2610.29903": {
   "id": "2610.29903",
   "title": "HuMiT: Low-Latency Whole-Body Humanoid Teleoperation via Compact Reference Tracking",
   "zh": "HuMiT：基于紧凑参考跟踪的低延迟人形全身遥操作",
   "org": "清华大学、香港中文大学、中国人民大学",
   "prob": "密集全身参考、未来帧条件和在线动作重定向会显著增加遥操作延迟。",
   "inno": "控制器只接收根节点高度与速度，以及双腕、双踝四个关键点的当前 SE(3) 位姿；无需在线优化和未来参考，通信到策略输出延迟约 20 毫秒。",
   "reuse": null,
   "short": "HuMiT",
   "slug": "humit-low-latency-whole-body-teleop",
   "abstract": "Real-time whole-body humanoid teleoperation remains challenging due to the computational latency introduced by optimization-based retargeting pipelines and future-frame conditioning. Existing motion tracking policies typically require dense reference targets including full joint positions, body-part transformations, and even future frames, which tightly couples the controller to expensive upstream modules. In this work, we present HuMiT, a whole-body teleoperation system built on a unified tracking policy that requires only a compact reference target, consisting of root height, root velocity, and the SE(3) poses of four sparse keypoints at the current frame, yet achieves competitive tracking fidelity compared to methods relying on more diverse reference states including complete joint positions, dense body-part transformations, and even future frames. Experiments show that our policy achieves a communication-to-inference latency of $\\sim$20 ms measured between headset streaming and policy output and a video-based optical flow latency of 65 ms (median) across different motion types. We validate the system on a Unitree G1 humanoid, showcasing split-second reactive responses such as catching a tossed ball, as well as coordinated loco-manipulation capabilities. Project page: this https URL.",
   "comments": "7 pages, 3 figures, 4 tables. Project page: this https URL",
   "authors": "Liu Cao, Botian Xu, Ruoqu Chen, Jiajun Liu, Mingzhi Pei, Zihao Wang, Zhongyu Li, Mengdi Xu",
   "links": [
    "https://xiaohu-art.github.io/Track/"
   ]
  },
  "2610.30044": {
   "id": "2610.30044",
   "title": "SP-HLM: Sparse-Pose-Driven Motion Generation and Whole-Body Control for Humanoid Loco-Manipulation",
   "zh": "SP-HLM：稀疏姿态驱动的人形移动操作动作生成与全身控制",
   "org": "上海交通大学、摩尔线程、香港大学",
   "prob": "高层策略若直接输出全身动作，学习负担较重；只输出少量任务目标又无法充分指定全身协调方式。",
   "inno": "VLA 预测五个关键身体位姿，经过两阶段蒸馏的动作生成模型补全全身运动，再由通用跟踪器同时利用显式稀疏目标和隐式动作先验执行控制。",
   "reuse": "wiki/entities/paper-sp-hlm.md"
  },
  "2610.30061": {
   "id": "2610.30061",
   "title": "Test-Time Motion Steering for Perceptive Humanoid Fall Recovery",
   "zh": "面向感知式人形起身恢复的测试时动作引导",
   "org": "密歇根大学",
   "prob": "同一跌倒姿态可能有翻滚、直接起身等多种恢复路径，而环境障碍、地形和后续任务会改变最佳选择。",
   "inno": "学习可组合的恢复动作词表，并用视觉扩散规划器预测未来身体状态；测试时根据目标朝向、起身速度或路径代价引导动作 latent，无需重新训练即可选择恢复方式。",
   "reuse": "wiki/entities/paper-rise-humanoid-fall-recovery.md"
  },
  "2610.30080": {
   "id": "2610.30080",
   "title": "Do Earth-Trained Humanoid Controllers Walk on the Moon? Gravity, Terrain, and Soil Transfer",
   "zh": "在地球训练的人形控制器能在月球行走吗？重力、地形与土壤迁移研究",
   "org": "斯坦福大学",
   "prob": "现有研究很少同时考虑月球低重力、粗糙地形和可变形月壤，也不清楚地球策略向月球迁移是否对称。",
   "inno": "在刚性地面和三维阻力理论土壤上系统比较地球训练、月球训练、微调及重力随机化策略；发现重力迁移高度不对称，而粗糙月壤加重力随机化具有最佳跨环境鲁棒性。",
   "reuse": null,
   "short": "Moon Walk Transfer",
   "slug": "earth-trained-humanoid-controllers-moon-transfer",
   "abstract": "The Moon changes gravity, terrain, and soil at once. Learned locomotion has studied reduced gravity on rigid ground and granular terrain mostly at Earth gravity, but not humanoids under both, generalist-controller transfer, or the Earth cost of lunar adaptation. On a simulated Unitree G1, we train 12 policies with one recipe, varying gravity, terrain geometry, and ground contact (rigid or three-dimensional resistive force theory, 3D RFT), and including Earth-to-Moon fine-tuning and gravity randomization. We test them at both gravities on rigid ground and 3D RFT, and at their training gravity on three held-out soil models. We also test the frozen SONIC, a released generalist humanoid controller, at both gravities on rigid ground only. Gravity transfer is asymmetric. Earth-trained policies keep 0.64 to 0.75 survival at the Moon on flat rigid ground, Moon-trained and fine-tuned policies cannot stand at Earth, and SONIC survives at most 0.10 at the Moon. Rough training lifts combined-tile survival from 0.53 to 0.98 at Earth and 0.80 to 0.99 at the Moon. In 3D RFT, soil training helps on Earth sand, on tiles and for rough-trained policies also on flat sand. On the stiffer lunar soil it helps only flat-trained Earth policies at Earth and changes the survival of Moon-trained policies by at most a few hundredths. In the sampled rollouts of the held-out soil models, Moon-trained policies keep most of their rigid-ground progress on lunar regolith at the low G1 foot pressure, with per-seed progress ratios of 0.84 to 1.39 apart from one seed with a short rigid reference. A gravity-randomized policy trained on rough 3D RFT terrain has a seed-mean survival of at least 0.84 at Earth and 0.96 at the Moon on every key test cell up to 20 degrees. Lunar results are simulation predictions conditional on the contact model and soil stiffness.",
   "comments": "11 pages, 4 figures, 5 tables. Supplementary material (13 pages) included as an ancillary file",
   "authors": "Guillem Casadesus Vila, Sri Aditya Deevi, Grace X. Gao",
   "links": []
  },
  "2610.30334": {
   "id": "2610.30334",
   "title": "MPBRS: Multi-Potential-Based Reward Shaping for Unified Reward Configuration in Humanoid Locomotion",
   "zh": "MPBRS：面向人形运动统一奖励配置的多势函数奖励塑形",
   "org": "内华达大学里诺分校",
   "prob": "人形 RL 奖励项容易互相冲突，对权重敏感，而且更换机器人后通常需要重新设计。",
   "inno": "将多个奖励项改写为势函数差分，在理论上不改变最优策略；相同奖励结构和权重可直接用于九种人形机器人，仅需提供关节及默认姿态元数据。",
   "reuse": "wiki/entities/paper-mpbrs-humanoid-reward-shaping.md"
  },
  "2610.30500": {
   "id": "2610.30500",
   "title": "PhysibleMDM: Enforcing Inference-Time Physical Feasibility Constraints on Motion Diffusion Models via Noise Optimization",
   "zh": "PhysibleMDM：通过噪声优化为动作扩散模型施加推理时物理可行性约束",
   "org": "Disney Research、MIT CSAIL",
   "prob": "动作扩散模型生成的参考可能出现穿地、滑脚、自碰撞或动力学不可执行，影响人形跟踪。",
   "inno": "不重训生成模型，直接用 CMA-ES 优化初始扩散噪声；评价函数结合运动学约束和锥规划逆动力学残差，并以自回归方式生成长时物理可行动作。",
   "reuse": "wiki/entities/paper-physiblemdm.md"
  },
  "2610.30540": {
   "id": "2610.30540",
   "title": "Motion Retargeting with Spatiotemporal Preferences for Humanoid Loco-Manipulation",
   "zh": "面向人形移动操作的时空偏好动作重定向",
   "org": "同济大学、上海创智学院",
   "prob": "均匀分配优化计算量会忽视接触建立、放置等关键帧和关键身体部位。",
   "inno": "由 VLM 选择重要信号和部位，再向关键帧分配更多求解迭代；利用近似物体扳手模型细化接触锚点，并以关键阶段采样和接触力奖励训练跟踪策略。",
   "reuse": "wiki/entities/paper-str-spatiotemporal-retargeting.md"
  },
  "2610.30817": {
   "id": "2610.30817",
   "title": "Robot Motion Transition with Human Motion Prior",
   "zh": "基于人体动作先验的机器人动作衔接",
   "org": "高丽大学、ROBOTIS",
   "prob": "两个独立动作片段直接插值时，位置、朝向、支撑脚和过渡时长可能不匹配。",
   "inno": "把机器人动作映射到人体空间，学习模型对下一段动作的位置、朝向和过渡时长候选进行排序，再用人体动作先验生成中间动作，最后映射回不同机器人形态。",
   "reuse": "wiki/entities/paper-stamp.md"
  },
  "2610.30876": {
   "id": "2610.30876",
   "title": "LocoManiLab: A Unified Testbed for Data, Policies, Controllers, and Deployment in Humanoid Loco-Manipulation",
   "zh": "LocoManiLab：统一人形移动操作数据、策略、控制器与部署测试平台",
   "org": "中国电信 TeleAI、中国科学技术大学、香港大学、浙江大学、上海交通大学、上海科技大学、Gamma Robotics",
   "prob": "不同论文使用不同任务、控制器、数据和扰动设置，难以公平比较移动操作策略。",
   "inno": "提供 234 项原子与组合任务、51.5 小时仿真遥操作数据，并对 12 种策略、5 种控制器、数据规模、视觉扰动和仿真—实机排序一致性进行系统评测。",
   "reuse": "wiki/entities/paper-locomanilab.md"
  },
  "2610.31166": {
   "id": "2610.31166",
   "title": "Memory before Goal: Role of Pre-History Memory for Humanoid Loco-Manipulation",
   "zh": "目标之前的记忆：前置历史记忆在人形移动操作中的作用",
   "org": "穆罕默德·本·扎耶德人工智能大学",
   "prob": "任务关键线索可能在机器人开始执行前出现，之后因转身、遮挡或物体移动而消失，单帧 VLA 无法完成非马尔可夫决策。",
   "inno": "用 SAM 跟踪任务物体，以物体事件因果选择关键历史帧，并通过交叉注意力同时读取图像记忆和包含外观、几何、时间的物体 token。",
   "reuse": "wiki/entities/paper-wholebodymem.md"
  },
  "2610.31608": {
   "id": "2610.31608",
   "title": "Humanoid Mobility via Attention-based Contacts",
   "zh": "基于注意力接触规划的人形机器人机动控制",
   "org": "苏黎世联邦理工学院、新加坡国立大学、Apple、ETH AI Center、NVIDIA",
   "prob": "纯 RL 虽能适应新地形，但难以仅靠奖励发现自然的全身协调；动作模仿策略又容易被参考轨迹限制。",
   "inno": "用自监督注意力规划器预测未来身体—环境接触，再由接触条件跟踪器执行全身运动；动作数据只作为训练先验，部署时无需参考输入，并可继续通过 RL 扩展到训练数据外的复杂地形。",
   "reuse": "wiki/entities/paper-acp-humanoid-mobility.md"
  },
  "2610.25628": {
   "id": "2610.25628",
   "title": "Humanoid SRL: Augmenting Humanoids with Supernumerary Limbs for Coordinated Loco-Manipulation",
   "zh": "Humanoid SRL：利用超额附加肢体增强人形协同移动操作",
   "org": "佐治亚理工学院",
   "prob": "当人形机器人的双臂都用于操作时，缺少额外肢体建立支撑接触或完成第三项操作。",
   "inno": "为人形平台集成第三条机械臂，并用轨迹优化生成动力学可行参考、目标条件 RL 控制器完成三臂协同；部分任务还使用由实机示范训练的视觉扩散策略自主执行。",
   "reuse": "wiki/entities/paper-humanoid-srl.md"
  },
  "2610.29030": {
   "id": "2610.29030",
   "title": "PHD-Hand: A Printable Humanoid Dexterous Hand with Linkage-Driven 3-DOA Digits",
   "zh": "PHD-Hand：采用三驱动自由度连杆手指的可打印人形灵巧手",
   "org": "新泽西理工学院、Union County Magnet High School",
   "prob": "腱绳灵巧手成本高、维护复杂，而传统连杆手指难以在紧凑尺寸内实现多驱动自由度。",
   "inno": "结合柔性铰链和弹簧钢弹性推杆，减少微型运动部件；可使用普通 3D 打印机制造，并在接近人手尺寸的结构中实现每指三驱动自由度。",
   "reuse": null,
   "short": "PHD-Hand",
   "slug": "phd-hand-printable-humanoid-dexterous-hand",
   "abstract": "Currently, dexterous humanoid robot hands are widely considered the final hardware bottleneck that hampers generalist robots from physically assisting people at home and work. As conventional tendon-driven designs suffer from high cost, demanding maintenance, poor reliability, and short durability, alternative linkage-driven designs have increasingly gained attention. Despite their potential to avoid the weaknesses of tendon-driven hands, integrating multiple degrees of actuation (DOA) in a single digit with linkage drivetrains is a challenge for design and fabrication. Only a few studies have managed to present functional humanoid hands that are completely linkage-driven and have at least three DOAs per digit without compromising actuator placement and other design preferences. Our work introduces a new linkage-driven hand featuring a unique flexure hinge. Together with elastic drive rods made of spring steel, the linkage drivetrains are significantly simplified. Unlike other linkage-driven designs of similar dexterity, our design does not require many miniature moving parts. As a result, the hand can be fabricated easily with a budget 3D printer. This not only significantly lowers the barrier to entry for using and developing dexterous humanoid robot hand functions, but also allows the hands to be sized much closer to regular human hands. The design offers desirable kinematic advantages as well. A prototype is evaluated for grasping and object manipulation. In addition, this paper presents our latest findings on kinematics and control of the hand.",
   "comments": "Submitted and Accepted at IEEE-RAS International Conference on Humanoid Robots (Humanoids 2026). 8 pages. Supplementary video: this https URL",
   "authors": "Neel Adwani, Muhaiminul Islam Akash, Saffia Ouali, Ethan Liu, Cong Wang",
   "links": [
    "https://lnkd.in/p/er-NRfMn"
   ]
  },
  "2610.27315": {
   "id": "2610.27315",
   "title": "Design and Control of a Multimodal Legged-Aerial Robot Through Separate Control Contraction Metrics",
   "zh": "基于分离控制收缩度量的多模态四足空地机器人设计与控制",
   "org": "东北大学 Silicon Synapse Lab、加州理工学院",
   "prob": "步行和飞行具有不同动力学及控制需求，直接使用统一模型难以兼顾两种模式和推力辅助小跑。",
   "inno": "分别离线求解步行与飞行的控制收缩度量，再通过在线能量型 CLF-QP 统一分配控制；同时研制带四个推进器的四足空地平台并验证推力辅助小跑。",
   "reuse": null,
   "short": "Legged-Aerial SCCM",
   "slug": "multimodal-legged-aerial-robot-sccm",
   "abstract": "The integration of dynamic legged locomotion and aerial mobility offers versatility that is deterred by challenges in payload distribution, actuation, and the contrasting requirements of both domains. In this paper, we present the design and operation of a multimodal quadrupedal aerial-legged robot. The presence of legged aerial systems also allows for thruster-assisted legged locomotion, which we make possible with a unified control framework using Control Contraction Metrics (CCM). We search for separate domain-specific contraction metrics for walking and flight independently offline and unify them using an online energy-based Control Lyapunov Function Quadratic Program (CLF QP). This enables optimal, cross-domain control allocation, which we show using simulations. We also show experimental results that validate the mechanical viability of our hardware and the paradigm of thruster-assisted trotting. The stability of said experiments motivates the deployment of the control framework for multimodal operations.",
   "comments": "Presented at 2026 IEEE International Conference on Intelligent Robots and Systems, Pittsburgh PA",
   "authors": "Kaushik Venkatesh Krishnamurthy, Eric Sihite, Alireza Ramezani, Morteza Gharib",
   "links": []
  },
  "2610.27472": {
   "id": "2610.27472",
   "title": "Towards Autonomous Bricklaying Using Boston Dynamics Spot",
   "zh": "使用 Boston Dynamics Spot 实现自主砌砖",
   "org": "麻省理工学院、德累斯顿工业大学、CeTI",
   "prob": "建筑机器人需要在分离的供料区和施工区之间移动，并准确执行 CAD 定义的砌筑任务。",
   "inno": "将 CAD 砖块位姿直接转换为 Spot 操作任务，以 AprilTag 定位并通过 Autowalk 完成区域间导航；在重复曲面墙体搭建实验中定量测量放置误差，并进行室外楼梯测试。",
   "reuse": null,
   "short": "Spot 砌砖",
   "slug": "autonomous-bricklaying-boston-dynamics-spot",
   "abstract": "Mobile legged robots offer unique advantages for on-site construction, including the ability to traverse unstructured terrain, climb stairs, and operate independently between spatially separated work zones. This paper presents a proof-of-concept system for autonomous brick stacking using a Boston Dynamics Spot quadrupedal robot equipped with a manipulator arm. The system operates between a supply area and an assembly area, using AprilTag fiducials for localization and autowalk missions for navigation. Target brick positions and orientations are specified in CAD software and exported directly for robotic execution. We evaluate placement accuracy through ten repeated assemblies of a curved four-layer wall comprising 14 wooden bricks. Results show mean translation errors below 10 mm and mean yaw errors below 1.5 degrees, while the overall RMSE across all runs is 27.72 mm (translation) and 2.06 degrees (rotation). The system was also tested outdoors with stairs between the two areas. The workflow is modular and easily extendable to different wall geometries.",
   "comments": "8 pages. Published in the Proceedings of the 43rd International Symposium on Automation and Robotics in Construction (ISARC 2026), Singapore",
   "authors": "Karl-Johan Sorensen, Nico vom Hofe, Frank H. P. Fitzek",
   "links": []
  },
  "2610.27998": {
   "id": "2610.27998",
   "title": "LocoPilot: Single-Operator Bilateral Teleoperation for Loco-Manipulation Policy Learning",
   "zh": "LocoPilot：面向移动操作策略学习的单人双向遥操作",
   "org": "意大利技术研究院动态腿式系统实验室",
   "prob": "单人难以同时控制四足底盘、身体姿态、机械臂和夹爪，而无实机示范又缺少平台特有观测和动作。",
   "inno": "用主机械臂与两个手柄实现全系统同步控制，以无力传感器方式估计并反馈接触力；数据采集和策略部署共享同一个 11 维动作接口，避免额外动作重定向。",
   "reuse": null,
   "short": "LocoPilot",
   "slug": "locopilot-bilateral-teleop-loco-manipulation",
   "abstract": "Learning loco-manipulation policies requires demonstrations that capture coordinated base and arm motion on the target embodiment. Robot-free interfaces allow demonstrations to be collected independently of the target robot, increasing flexibility but without directly capturing platform-specific commands, proprioception, and body-mounted observations. On-robot teleoperation allows to capture these signals directly, but requires an interface capable of commanding the complete system concurrently. This paper presents LocoPilot, a single-operator bilateral teleoperation and data-collection framework implemented on a 12-DoF Unitree Go2 quadruped equipped with a 6-DoF AgileX Piper L manipulator, an Intel RealSense D435 wrist camera, and an Intel RealSense D405 head camera. A second 6-DoF Piper L arm, used as the leader and two handheld controllers provide simultaneous control of the base, body posture, manipulator, and gripper. The follower arm interaction forces are estimated without a dedicated force/torque sensor and rendered to the operator through a passivity-constrained haptic channel. A ROS~2 pipeline records synchronized multimodal observations together with a unified 11D action shared by teleoperation and policy deployment. Compared to prior work, LocoPilot uniquely unifies robot-in-the-loop legged data collection, force feedback without motion-capture instrumentation, and the same action interface at deployment. We evaluate the interface across multiple loco-manipulation tasks and validate the complete collection-to-deployment pipeline by fine-tuning MolmoAct2 \\cite{MOLMO} on pick-and-place demonstrations and deploying on the physical robot without action retargeting or additional inverse kinematics. Code available at: this https URL",
   "comments": "",
   "authors": "Antonio Langella, Giovanni Minelli, Giovanni B. Dessy, Lorenzo Amatucci, Giulio Turrisi, Victor Barasuol, Claudio Semini",
   "links": [
    "https://github.com/iit-DLSLab/LocoPilot"
   ]
  },
  "2610.29586": {
   "id": "2610.29586",
   "title": "End-to-End Reinforcement Learning for Quadrupedal Jumping in Lunar Gravity",
   "zh": "面向月球重力四足跳跃的端到端强化学习",
   "org": "苏黎世联邦理工学院机器人系统实验室",
   "prob": "月球崎岖、断裂地形需要跳跃越过间隙，但策略必须依赖任务允许的机载传感器，并满足实际执行器功率限制。",
   "inno": "通过多阶段训练得到单一端到端定点跳跃策略；分析控制冻结等极端故障，并在重力卸载硬件实验中验证垂直跳跃及力矩、功率可行性。",
   "reuse": null,
   "short": "月球重力跳跃",
   "slug": "quadruped-jumping-lunar-gravity-rl",
   "abstract": "Legged robots can extend planetary exploration to rugged and discontinuous terrain that is inaccessible to conventional rovers. In lunar gravity, jumping further expands this mobility by enabling robots to cross obstacles and gaps that cannot be traversed by walking alone. We present a multistage reinforcement-learning approach that produces a single end-to-end control policy for commanded point-to-point jumping under lunar gravity. The resulting policy has a simple architecture and relies exclusively on mission-compatible onboard sensing, making its deployment on a real lunar mission plausible. In simulation, the learned policy performs accurate point-to-point jumps with landing errors smaller than the robot's body size. We analyze the torque and power demands of lunar jumping and characterize performance degradation under extreme outlier events, including control-update freezes of up to \\SI{1}{s} during flight. Hardware transfer demonstrates that the policy can successfully execute a vertical jump under lunar-gravity offloading. Furthermore, the measured torque and power profiles are consistent with the simulation results and remain within the robot's hardware limits.",
   "comments": "",
   "authors": "Levin Müller, Oliver Fischer, Marco Trentini, Yusuke Tanaka, Marco Hutter",
   "links": []
  },
  "2610.25962": {
   "id": "2610.25962",
   "title": "SkateNav: Learning Energy-Efficient Navigation through Adaptive Skating on Passive-Wheel Quadrupedal Robots",
   "zh": "SkateNav：通过自适应滑行学习被动轮四足机器人的高能效导航",
   "org": "NAVER LABS、韩国科学技术院",
   "prob": "速度跟踪奖励会抑制滑行动作固有的速度起伏，使被动轮机器人频繁蹬地而无法充分利用惯性。",
   "inno": "把导航改写为带时间预算的到达目标任务，使策略联合决定转向、蹬地和滑行时长；通过可逐控制周期调频的滑行动作循环适应环境，并零样本迁移到带被动轮的 Mini Cheetah。",
   "reuse": null,
   "short": "SkateNav",
   "slug": "skatenav-passive-wheel-quadruped-skating",
   "abstract": "Passive-wheel quadrupedal robots offer energy-efficient locomotion through leg-driven propulsion and momentum-driven gliding. However, navigation requires coordinating steering with intermittent propulsion and adapting push-glide timing to task demands. Velocity-tracking objectives may suppress the speed variations inherent in skating, discouraging efficient gliding. To address these challenges, we propose SkateNav, a unified skating-navigation framework based on reinforcement learning. A time-budgeted goal-reaching objective allows the policy to jointly learn steering, propulsion, and their temporal allocation. An adaptive skating cycle structures gliding, pushing, and recovery, while frequency modulation at every control step adjusts phase durations to navigation demands. Simulation experiments demonstrate higher average success rates and lower cost of transport than the evaluated planner-controller baselines, with ablations confirming the contributions of cycle structure and frequency adaptation. Real-world experiments demonstrate zero-shot transfer from simulation to a passive-wheel Mini Cheetah across cluttered indoor spaces, slopes, and paved outdoor surfaces. Supplementary videos are available on the project page: this https URL",
   "comments": "8 pages, 12 figures, 3 tables. Project page: this https URL",
   "authors": "Jeil Jeong, Taeyun Kim, Sangjin Kim, Jinyoung Choi, Sung-Eui Yoon",
   "links": [
    "https://ktw1404.github.io/SkateNav/"
   ]
  },
  "2610.26550": {
   "id": "2610.26550",
   "title": "Learning Wheel Motion Uncertainty for Degeneracy-Aware LiDAR-Inertial Odometry on Quadrupedal Wheel-Legged Robots",
   "zh": "面向四足轮腿机器人的轮运动不确定性学习与退化感知激光—惯性里程计",
   "org": "香港中文大学、同济大学、仙工智能",
   "prob": "走廊等几何退化环境需要轮运动提供额外约束，但打滑、悬空会使轮速预测失真，二值接触状态不能反映不同方向上的可靠性。",
   "inno": "从本体历史为每个轮子学习方向相关过程协方差，在可微滤波器中训练；同时抑制激光雷达几何弱方向的约束，使轮运动和激光约束根据各自退化程度自适应加权。",
   "reuse": null,
   "short": "Wheel Uncertainty LIO",
   "slug": "wheel-motion-uncertainty-degeneracy-aware-lio",
   "abstract": "Wheel kinematics can support odometry for quadrupedal wheel-legged robots when LiDAR geometry provides weak motion constraints. However, slip and contact loss make rolling predictions unreliable, and contact state alone cannot determine their reliability across motion directions. We propose a tightly coupled LiDAR-inertial-kinematic odometry framework that retains analytical rolling predictions and learns per-wheel, direction-dependent process covariances from proprioceptive history. The estimator maintains moving wheel-center states, allowing the learned uncertainty to regulate the influence of rolling predictions through forward-kinematic corrections. Degeneracy-aware projection (DAP) complements this mechanism by suppressing LiDAR constraints along geometrically weak directions. The learned covariance network (CovNet) is trained through a differentiable filter using simulated velocity supervision and deployed on the real robot without fine-tuning. On the six room sequences with the highest degeneracy ratios in our indoor dataset, the method reduces mean per-sequence position root mean square error (RMSE) by 66.2% relative to FAST-LIO2 and by 31.0% relative to a fixed-covariance variant. In a severely degenerate corridor run, it achieves an endpoint position error of 0.198 m, compared with 7.531 m for FAST-LIO2. These results demonstrate robust odometry under challenging geometric and wheel-motion conditions. We plan to publicly release our code and multimodal dataset. Project website: this https URL",
   "comments": "",
   "authors": "Hairuo Wei, Yunfan Ren, Renda Yang, Qiangsheng Huang, Zhongyu Li",
   "links": [
    "https://weihairuo.github.io/wheel-motion-uncertainty/"
   ]
  },
  "2610.26414": {
   "id": "2610.26414",
   "title": "RoboMMT: Learning Stable Tracking of Modulated Expressive and Interactive Robot Motion",
   "zh": "RoboMMT：调制式表达与交互机器人动作的稳定跟踪学习",
   "org": "斯图加特大学",
   "prob": "在线改变表达动作的幅度、方向和节奏时，动态不稳定的腿式平台可能接近平衡极限。",
   "inno": "在完整表达参数域上训练参考条件跟踪策略，再由轻量交互策略修改动作生成器的空间目标；在四足和两种人形平台上统一验证，并完成四足零样本实机迁移。",
   "reuse": null,
   "short": "RoboMMT",
   "slug": "robommt-modulated-expressive-motion-tracking",
   "abstract": "Expressive robot motion can improve naturalness, effectiveness, and perceived likability in human-robot interaction. When deployed on dynamically unstable platforms, however, reactive execution of expressive body motion may bring the robot to its stability limits, especially when expressivity is decoupled from physical control. In this paper, we present RoboMMT (Robot Modulated Motion Tracking), a modular control architecture that couples parameterized expressive motion synthesis, physics-based character control, and reactive adaptation to spatial interaction goals. Our approach trains a reference-conditioned tracking policy across a broad domain of expressivity, enabling it to faithfully realize the commanded motions while maintaining dynamic stability. Goal reactivity is achieved by training lightweight, task-specific interaction policies that modulate the reference generator's spatial goal-attraction point, with the stabilizing tracking policy kept in the loop. Evaluations on a quadruped and two humanoid robots show that the learned tracker remains stable and accurate across the entire modulation domain, whereas trackers trained without this coverage lose balance, tracking accuracy, or both. We show that layering closed-loop goal adaptation over this foundation enables interactive facing and pointing behaviors, while maintaining stable execution and comparable reference-tracking fidelity. The robot learns to proactively redistribute its weight and take corrective steps. Zero-shot sim-to-real transfer is demonstrated through real-robot experiments on a quadruped under heavy disturbances.",
   "comments": "",
   "authors": "Till Hielscher, Kai O. Arras",
   "links": []
  },
  "2610.26810": {
   "id": "2610.26810",
   "title": "Bio-inspired Thrust-Assisted Trotting on a Legged-Aerial Robot via Real-Time QP-MPC",
   "zh": "基于实时 QP-MPC 的仿生推力辅助空地双足小跑",
   "org": "东北大学、加州理工学院",
   "prob": "空地双足机器人小跑时需要同时调节地面反力和推进器矢量，并满足摩擦锥和实时计算要求。",
   "inno": "将推力辅助弹簧倒立摆动力学凝聚成 500 Hz QP-MPC，矢状面由 MPC 控制、侧向由级联姿态控制器稳定；平均速度可跟踪指令且单次求解约 0.18 毫秒。",
   "reuse": null,
   "short": "Thrust-Assisted Trotting",
   "slug": "thrust-assisted-trotting-legged-aerial-qp-mpc",
   "abstract": "This paper presents the hardware implementation of a quadratic program-based controller on Harpy: thruster-assisted bipedal robot to perform thruster-assisted trotting. This is inspired by wing-assisted locomotion observed in birds such as the chukar partridge, which employ thrust vectoring to manipulate the ground reaction force. The proposed controller formulates the thruster-assisted spring-loaded inverted pendulum dynamics as a condensed QP-based model predictive control problem, enabling simultaneous regulation of ground contact force and active thrust vectoring at 500 Hz. A decoupled control architecture is used, where the robot's sagittal dynamics are controlled using QP-MPC while the frontal dynamics is regulated using a cascade orientation stabilization controller. Two experiments were conducted with different forward reference velocities of 0 m/s and 0.1 m/s. While the instantaneous velocity oscillates, the mean velocities can track their respective reference. Additionally, joint tracking error remains within 1 deg while tracking the desired leg length given by the leg trajectory controller and desired swing trajectory given by Raibert's heuristic. The QP-MPC problem is consistently solved with 0.18ms, which is well within the 2ms computation budget while satisfying the GRF friction cone constraint.",
   "comments": "To be published in IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2026)",
   "authors": "Shreyansh Pitroda, Eric Sihite, Alireza Ramezani, Morteza Gharib",
   "links": []
  },
  "2610.29438": {
   "id": "2610.29438",
   "title": "A2I-Calib++: A Kinematics-Driven Anti-noise Active Calibration Framework for Whole-Body Multi-IMU Systems on Legged Robots",
   "zh": "A2I-Calib++：面向腿式机器人全身多 IMU 的运动学驱动抗噪主动标定",
   "org": "上海交通大学、Openmind Smart Robotics、江西智联先进技术研究院",
   "prob": "日常腿式运动激励不足，而足端冲击、振动和打滑又会干扰多 IMU 时空外参及偏置标定。",
   "inno": "设计噪声各向同性激励轨迹，以带滚动接触奖励的 AMP 控制器执行；再利用小腿 IMU、关节运动学和接触约束构造虚拟参考 IMU，同时估计时间偏移、旋转、平移和惯性偏置。",
   "reuse": null,
   "short": "A2I-Calib++",
   "slug": "a2i-calib-plus-multi-imu-legged-calibration",
   "abstract": "Recently, multi-node inertial measurement unit (multi-IMU) systems have shown promise for robust odometry in legged robots. Multi-IMU fusion can improve legged robot odometry but requires accurate spatiotemporal calibration among the sensors. Existing multi-IMU calibration methods, designed for aerial or handheld platforms, face challenges on legged robots because routine locomotion may provide insufficient excitation, while foot slips, impacts, and vibrations introduce measurement disturbances. To address these challenges, we propose A2I-Calib++, a kinematics-driven anti-noise active calibration framework for body-mounted IMUs covered by the robot kinematic model. It estimates temporal offsets, extrinsic rotations and translations, and inertial biases through temporal correlation search followed by analytic spatial and bias estimation, without artificial calibration targets or complex parameter tuning for the target IMU. A2I-Calib++ comprises three core modules: 1) a Noise-Isotropy-Inspired Trajectory supported by a finite-horizon conditioning proposition that generates sufficient excitation to limit noise sensitivity; 2) a constraint-aware Adversarial Motion Priors controller that combines the anti-noise excitation with a rolling-contact reward for contact-aware execution; and 3) a contact-constrained virtual IMU that fuses pre-calibrated calf IMUs and joint kinematics under a rolling-contact model to construct the calibration reference. Furthermore, evaluations on both simulation and real-world quadruped platforms demonstrate notably reduced noise sensitivity and simulation calibration errors compared to conventional methods, together with repeatable real-world calibration and improved proprioceptive odometry accuracy for legged robots.",
   "comments": "",
   "authors": "Fangyu Jiang, Chaoran Xiong, Kehui Ma, Zhen Sun, Zeyu Zhang, Ling Pei",
   "links": []
  },
  "2610.26854": {
   "id": "2610.26854",
   "title": "Physical Words: A Vocabulary for Structuring Latent Reasoning in Robot Manipulation",
   "zh": "物理词汇：用于组织机器人操作潜在推理的词汇体系",
   "org": "待核实（作者：Jiaming Liu、Shanghang Zhang、Yan Chang 等）。",
   "prob": "语言推理能够描述机器人任务的语义目标，但难以表达细粒度物理变化；潜在状态推理能够预测未来，却缺少可复用的语义结构。",
   "inno": "提出 PhysWord，以原子动作词汇作为语义锚点，将动作词的隐藏状态与未来观测提取的潜在表示对齐。模型自回归预测未来的 Physical Words，并通过视觉或触觉条件表达不同操作阶段的状态变化。在 UniVTAC 和 LIBERO-Long 上分别达到 90.3% 和 99.6% 成功率，并在真实双臂夹爪与灵巧手任务中验证。",
   "reuse": null,
   "short": "PhysWord",
   "slug": "physical-words-latent-reasoning-manipulation",
   "abstract": "Multimodal reasoning provides robotic foundation models with predictive context needed for reliable action generation. However, language-based reasoning captures task semantics but struggles to express fine-grained physical dynamics, while latent reasoning models future state evolution but offers limited structure for organizing these dynamics into reusable reasoning primitives. To bridge this gap, we introduce PhysWord, a reasoning-before-acting foundation model built on Physical Words, which use atomic action verbs as shared semantic anchors for continuous representations of physical dynamics. Specifically, PhysWord predicts upcoming atomic action words and directly aligns their contextual hidden states with latent targets derived from the corresponding future observations. Building on these primitives, we construct a temporally structured PhysWord space with visual and optional tactile variants of each action word. Autoregressively predicting Physical Words across future timesteps captures transitions between execution stages, providing structured multimodal context for action generation. PhysWord achieves state-of-the-art success rates of 90.3\\% on visuotactile UniVTAC and 99.6\\% on vision-based LIBERO-Long, and outperforms competing methods on real-world tasks using bimanual grippers and dexterous hands. Further evaluations demonstrate that our structured latent reasoning framework generalizes to unseen scenarios and enables long-horizon manipulation. Project page: this https URL",
   "comments": "",
   "authors": "Jiaming Liu, Xinghao Zhu, Yiming Zhang, Shanghang Zhang, Yinxi Wang, Yichu Yang, Zhuoyang Liu, Dantong Niu, Soha Pouya, Juekun Li, Yan Chang",
   "links": [
    "https://nvidia-isaac.github.io/video_to_data/physicalword/"
   ]
  },
  "2610.31423": {
   "id": "2610.31423",
   "title": "FocusVLA: Future-Intent Guided Dynamic View Routing for Vision-Language-Action Models",
   "zh": "FocusVLA：基于未来意图引导的视觉语言动作模型动态视角路由",
   "org": "待核实（作者：Huixin Xu、Ziya Yu、Peng Li）。",
   "prob": "多相机 VLA 在不同操作阶段需要关注不同视角，但统一处理全部相机信息会增加计算开销，也可能引入无关视觉内容。",
   "inno": "将未来操作意图引入视角路由，根据即将执行的动作动态选择视觉信息，使视觉处理与后续操作需求相匹配。论文已被 CoRL 2026 接收。",
   "reuse": null,
   "short": "FocusVLA",
   "slug": "focusvla-future-intent-view-routing",
   "abstract": "Vision-language-action (VLA) models have become an important paradigm for robotic manipulation by unifying visual perception, language understanding, and action generation. Since reliable manipulation requires sufficient observation of objects, end-effectors, and their spatial relationships, multi-view inputs are widely used to provide complementary spatial information. However, our experiments show that existing VLA models do not fully exploit such multi-view complementarity. The policy tends to persistently rely on the wrist view, lacking the ability to dynamically adjust global-local view dependency across manipulation stages, which limits stage adaptability and robustness. To address this issue, we propose FocusVLA, a stage-aware multi-view routing framework. FocusVLA predicts future interaction intent from historical actions and uses it to guide dynamic view routing during action generation. By adaptively coordinating global and local observations across different task stages, FocusVLA improves multi-view utilization. On RoboTwin 2.0, FocusVLA improves the average success rate over the strongest baseline $\\pi_{0.5}$ by 10.10 pp. In real-world evaluation, FocusVLA improves the success rate from 71.7\\% to 85.0\\%, achieving a 13.3\\% gain over $\\pi_{0.5}$. Moreover, the intent prediction and dynamic routing modules of FocusVLA can be integrated into existing VLA frameworks without modifying their core architectures.",
   "comments": "Accepted at CoRL 2026",
   "authors": "Huixin Xu, Ziya Yu, Peng Li",
   "links": []
  },
  "2610.31569": {
   "id": "2610.31569",
   "title": "Echo: Coupled Action and Task Status Signal Denoising for Vision Language Action Models",
   "zh": "Echo：面向视觉语言动作模型的动作与任务状态信号联合去噪",
   "org": "待核实（作者：Yufeng Liu、Yang Gao、Cheng Ma 等）。",
   "prob": "VLA 需要同时决定下一步动作和判断当前任务状态。如果动作生成与任务状态估计分别学习，两者可能出现不一致。",
   "inno": "Echo 将动作与任务状态信号放入耦合去噪框架，使动作生成过程能够利用任务状态信息，并建立动作预测与任务进度估计之间的联系。",
   "reuse": null,
   "short": "Echo-VLA",
   "slug": "echo-coupled-action-task-status-denoising",
   "abstract": "Recognizing task completion and failure enables robots to stop upon success and initiate recovery or replanning after failure. Existing methods augment Vision Language Action (VLA) policies with task status signal prediction to support these execution decisions. However, learning and integrating these signals can require fine-grained supervision, external monitoring models, or complex architectural designs. Moreover, the potential of task status signals to directly guide action generation remains underexplored. We present Echo, a framework that combines task status signal learning without frame-level progress annotations with a compact design for status prediction and action generation within a single VLA. Specifically, Echo represents task status with stop and failure signals, added as two dimensions of each action chunk and denoised together with robot actions. Their combination distinguishes ongoing execution, successful completion, and failure. To learn these signals without fine-grained progress annotations, Echo constructs successful, failed, and recovery trajectories with complementary supervision properties. To enable status signal-guided action generation, Echo controls interactions between the signals and actions and denoises the signals faster than the actions. These allow later action denoising steps to condition on cleaner status estimates while preserving accurate signal prediction. Extensive real-world manipulation experiments show that Echo accurately predicts task status and consistently improves policy performance.",
   "comments": "Project page: this https URL",
   "authors": "Yufeng Liu, Bohao Fan, Juntu Zhao, Xinyi Wang, Hang Yu, Wenzhuan Luo, Yingzhi Wang, Qinghai Li, Yingdong Hu, Junliang Guo, Richard Wang, Dequan Wang, Yang Gao, Cheng Ma",
   "links": [
    "https://lyfeng001.github.io/Echo/"
   ]
  },
  "2610.28814": {
   "id": "2610.28814",
   "title": "T-WLA: Bridging Slow Vision and Fast Touch",
   "zh": "T-WLA：连接低频视觉与高频触觉",
   "org": "待核实（作者：Kaichen Zhou、Wojciech Matusik、Antonio Torralba、Paul Pu Liang 等）。",
   "prob": "视觉信息适合识别物体和规划操作，但接触发生后的状态变化往往需要更高频率的触觉反馈。两种模态的更新频率不同，增加了实时控制的融合难度。",
   "inno": "T-WLA 研究低频视觉与高频触觉的联合建模，使机器人在利用视觉进行任务级决策的同时，能够通过快速触觉信息响应接触变化。",
   "reuse": null,
   "short": "T-WLA",
   "slug": "t-wla-slow-vision-fast-touch",
   "abstract": "In contact-rich manipulation, contact conditions can change before a visual policy updates its action plan. Recomputing the plan through the visual policy adds latency, while tactile feedback alone does not specify how the ongoing motion should be revised for the task. We introduce T-WLA, a world-language-action architecture that uses shared visual memory to guide both action generation and tactile correction. At each visual update, the visual branch integrates the current image, task instruction, and retrieved history into a visual memory that conditions base-action generation. The same memory is cached for a lightweight tactile editor, which combines it with the active action chunk and incoming touch to predict residual corrections to unexecuted actions without rerunning the visual branch. Future-image and future-pressure prediction provide training-only supervision, encouraging the visual memory and tactile features to capture interaction dynamics relevant to action generation and correction. We evaluate T-WLA on tactile-conditioned hand-motion prediction using EgoTouch and Tachin, and on four real-world dexterous manipulation tasks. T-WLA reduces finger-joint MPJPE by 30% relative to the strongest reported baseline and achieves an average success rate of approximately 50.0% across real-robot manipulation tasks. Videos and code are available at this https URL.",
   "comments": "",
   "authors": "Kaichen Zhou, Yuzhen Chen, Hannah Yao, Chonghe Jiang, Rao Fu, Xihang Yu, Wei Dai, Bashar Kabbarah, Haoyu Chen, Ishita Goluguri, Zhiyang Dou, Haonan Chen, Fangneng Zhan, Mengyu Wang, Wojciech Matusik, Antonio Torralba, Paul Pu Liang",
   "links": [
    "https://anonymous-submission-20.github.io/T-WLA.github.io/"
   ]
  },
  "2610.28019": {
   "id": "2610.28019",
   "title": "PACT-VLA: Speculative Action-Chunk Verification with Pre-Commit Observations",
   "zh": "PACT-VLA：利用执行前观测进行推测式动作块验证",
   "org": "待核实（作者：Yixiang Zhu、Ruilong Wu、Yonghao Chen 等）。",
   "prob": "VLA 生成动作块后，连续执行期间的环境变化可能使后续动作失效。固定长度的开环执行难以及时处理状态偏移。",
   "inno": "将执行前观测与动作块验证结合，在提交动作之前检查候选动作与当前环境状态的一致性，研究降低动作块执行风险的方法。",
   "reuse": null,
   "short": "PACT-VLA",
   "slug": "pact-vla-speculative-action-chunk-verification",
   "abstract": "Vision-language-action (VLA) policies are commonly deployed with asynchronous inference in real-world settings, where the next action chunk is generated while the robot is still executing the previous one. By the time the new chunk is committed, the robot and scene have changed, so the chunk is conditioned on a stale observation and task success drops sharply as this inference delay grows. Generating multiple candidate chunks could let newer observations inform the final choice, but existing multi-candidate methods either rank candidates with the stale inference-start observation or repeatedly switch between candidates during execution, which can make the executed action sequence discontinuous. Our key insight is that inference delay can be used as decision time. While candidate generation is still underway, candidate selection can reuse the VLA itself to choose the complete action chunk that best fits a later observation. We propose PACT-VLA, a training-free framework that generates candidates from the inference-start observation and selects one using this pre-commit observation. Its scoring function, FlowRes, measures the residual between each candidate's implied flow direction and the velocity predicted by the frozen policy under the pre-commit observation, requiring no separate verifier. A pipelined implementation overlaps observation processing with candidate generation and batches scoring, so selection adds no control-step delay on the evaluated hardware. At a four-step delay, PACT-VLA improves average success by 15.0 percentage points over RTC on Kinetix and by 5.2 points over single-candidate asynchronous execution with GR00T N1.7 on LIBERO, and it further improves existing delay-robust methods. On three real-robot tasks, it raises average success from 60% to 82% over naive asynchronous execution with the same frozen $\\pi_{0.5}$ checkpoint.",
   "comments": "",
   "authors": "Yixiang Zhu, Ruilong Wu, Yonghao Chen, Jiaxiang Zou, Yusong Hu, Xinyu Chen",
   "links": []
  },
  "2610.31613": {
   "id": "2610.31613",
   "title": "Progress Probes P²: Lightweight Runtime Monitoring for Vision-Language-Action Models",
   "zh": "进度探针 P²：面向视觉语言动作模型的轻量化运行时监测",
   "org": "待核实（作者：Atiksh Bhardwaj、Andrea Bajcsy、Preston Culbertson 等）。",
   "prob": "VLA 执行过程中需要判断任务是否正常推进，但持续调用大型视觉语言模型进行状态评估会增加运行开销。",
   "inno": "引入轻量级进度监测模块，在策略运行时评估任务执行状态，为动作继续执行、重新规划和失败检测提供反馈。",
   "reuse": null,
   "short": "Progress Probes P²",
   "slug": "progress-probes-p2-vla-runtime-monitoring",
   "abstract": "Vision-Language-Action (VLA) models can perform diverse manipulation tasks directly from visual observations and natural language instructions, but they remain prone to mistakes and unpredictable failures. Deploying these policies without continuous human supervision will require robust runtime monitors that detect when the policy has gone off-course. Existing monitors use external foundation models, such as vision-language models (VLMs) to supervise the policy, introducing significant complexity and computational cost for monitoring at runtime. In this work, we investigate an alternate approach: \\textit{can a VLA's own internal activations provide the information needed to monitor its execution?} Our method trains linear probes on policy activations from demonstrations to estimate progress, then flags executions that deviate from expected progress. In experiments with the open-source VLA $\\pi_{0.5}$ and across simulated and real-world manipulation tasks, this approach is competitive with or outperforms VLM-based monitors in task failure detection, as well as methods which require failure data for training. These findings suggest that simple probes of VLA internals can be used to develop reliable runtime monitors, opening a path toward building policies that can act reliably and monitor their execution without external foundation models.",
   "comments": "",
   "authors": "Atiksh Bhardwaj, Edward W. Duan, Junwon Seo, Prithwish Dan, Wei-Chiu Ma, Andrea Bajscy, Preston Culbertson",
   "links": []
  },
  "2610.29692": {
   "id": "2610.29692",
   "title": "FlowVLA: Learning Flow-Aware Physical Completion for Quantitative Robotic Pouring",
   "zh": "FlowVLA：面向定量机器人倾倒操作的流动感知物理完成状态学习",
   "org": "待核实（作者：Liwen Yang、Yutian Zhang、Dibo Hou 等）。",
   "prob": "定量倾倒不仅要求机器人完成倾倒动作，还需要判断流体转移是否达到目标数量。传统操作策略难以仅根据机械臂运动判断任务完成程度。",
   "inno": "将流动状态与物理完成程度引入操作学习，使机器人能够根据物体转移过程调整动作，而不是仅执行预先生成的倾倒轨迹。",
   "reuse": null,
   "short": "FlowVLA 倾倒",
   "slug": "flowvla-quantitative-robotic-pouring",
   "abstract": "During steady pouring, images and robot pose may remain nearly unchanged while transferred mass accumulates, making the return decision ambiguous for a whole-task vision-language-action (VLA) policy. We present FlowVLA, which separates motor generation from physical-completion estimation. A shared $\\pi_{0.5}$ action expert learns approach, pour, and return with boundary-aware supervision. A target-independent estimator fuses visual, joint, wrist-wrench, and recurrent force-belief histories to maintain an estimate of transferred mass. The controller initiates return when this estimate plus predicted residual outflow reaches the requested mass. In coffee-pot/water experiments, FlowVLA achieves 13.1 g goal-balanced mean absolute error (MacroMAE), lower than all tested VLA baselines. Across all attempts, 53.7% both complete the task and finish within +/-10% of the requested mass. The same model interpolates to held-out 150/250-g targets without retraining. Independently trained models further evaluate the formulation in three additional container-fluid settings.",
   "comments": "9 pages, 6 figures",
   "authors": "Liwen Yang, Yutian Zhang, Jiaqi Zhai, Li Yang, Tutian Tang, Qiwei Shan, Yancheng Bi, Qiankuan Shangguan, Yangkun Zhu, Ce Hao, Dong Wei, Qiaojun Yu, Dibo Hou",
   "links": []
  },
  "2610.30136": {
   "id": "2610.30136",
   "title": "Agentic World Action Models",
   "zh": "智能体式世界动作模型",
   "org": "待核实（作者：Zhuoyang Liu、Ken Goldberg、Danfei Xu、Yuke Zhu、Linxi Fan、Trevor Darrell 等）。",
   "prob": "世界动作模型能够学习动作与未来环境状态之间的关系，但复杂操作还需要将世界预测与任务级决策、执行反馈结合起来。",
   "inno": "研究 Agentic World Action Model，将世界预测和动作生成能力用于智能体式机器人决策。其核心研究方向是建立世界模型与任务执行之间更完整的闭环联系。",
   "reuse": "wiki/entities/paper-aw-0-agentic-world-action-models.md"
  },
  "2610.30991": {
   "id": "2610.30991",
   "title": "UniFlowWAM: A Unified World-Action Model with Visual Semantics and 4D Geometry",
   "zh": "UniFlowWAM：融合视觉语义与四维几何的统一世界动作模型",
   "org": "待核实（作者：Jialin Tian、Cewu Lu、Lixin Yang 等）。",
   "prob": "机器人操作需要理解物体语义和三维运动。仅使用二维视觉特征进行世界预测，难以直接表达物体在时间和空间中的几何变化。",
   "inno": "将视觉语义与四维几何信息统一引入世界动作建模，使模型能够同时利用场景语义和时空几何结构生成动作。",
   "reuse": null,
   "short": "UniFlowWAM",
   "slug": "uniflowwam-semantics-4d-geometry-world-action-model",
   "abstract": "Robot demonstrations record what the robot saw and what it did. They do not specify how the 3D world changed in between. Scene-level 3D flow describes it by tracking 3D scene points, providing the world representation with 4D geometry. Prior work has used flow to improve robot policies. Flow served as an intermediate prediction that a separate model converts into actions, as a prior injected at inference time, or as an extra output learned together with the action. In all of these designs, flow stays outside the policy, as a separate module or as an extra prediction head. We introduce UniFlowWAM, a Unified World-Action Model that makes 3D scene flow a third modality on par with vision and action. Its world representation therefore covers both visual semantics, through future visual features, and 4D geometry, through 3D flow, and flow receives its own supervision. UniFlowWAM is a diffusion transformer with one expert per modality, and the three experts exchange information through joint attention. We train UniFlowWAM under a ten-objective role grid, where each objective assigns every modality exactly one of three roles, namely target, condition, or absent. The same model thus serves as a policy, as forward and inverse dynamics, and as a predictor of future visual features or 3D flow. We evaluate UniFlowWAM on RoboCasa-GR1, LIBERO, and LIBERO-Plus in simulation, and on four real-world bimanual manipulation tasks. On RoboCasa-GR1, UniFlowWAM raises the success rate of its base model without flow from 52.1% to 56.4%. It preserves in-distribution success on LIBERO at 98.1% and reaches 87.7% zero-shot on LIBERO-Plus. On the real robot, it achieves 87.0% average success, compared with 68.5% for the strongest baseline.",
   "comments": "14 pages, 5 figures, 4 tables. Project page: this https URL",
   "authors": "Jialin Tian, Zhuofan Li, Xinkai Wang, Bokai Lin, Chenyi Wang, Xinyu Zhan, Cewu Lu, Lixin Yang",
   "links": [
    "https://kksham111.github.io/UniFlowWAM_ProjectPage/"
   ]
  },
  "2610.28626": {
   "id": "2610.28626",
   "title": "UniWALM: Unified World Action Language Model for Embodied Robot Control",
   "zh": "UniWALM：面向具身机器人控制的统一世界—动作—语言模型",
   "org": "待核实（作者：Rahul Sajnani、Diego Romeres、Srinath Sridhar 等）。",
   "prob": "语言任务理解、未来环境预测和机器人动作生成通常使用不同模型，导致高层任务语义与底层物理执行之间存在表示差异。",
   "inno": "UniWALM 研究将语言、世界状态和动作统一建模，使语言任务条件、世界预测与机器人控制共享学习框架。",
   "reuse": "wiki/entities/paper-uniwalm.md"
  },
  "2610.30117": {
   "id": "2610.30117",
   "title": "CW-WAM: A Structured Contact-Wrench World Action Model for Force-Sensitive Manipulation",
   "zh": "CW-WAM：面向力敏感操作的结构化接触力旋量世界动作模型",
   "org": "待核实（作者：Houzhu Ding、Xinda Qi、Luqi Wang、Ming Yang）。",
   "prob": "机器人接触操作不仅取决于物体的视觉位置，还取决于接触力和力矩。缺少接触动力学信息的世界模型难以准确表达力敏感任务。",
   "inno": "CW-WAM 将结构化接触力旋量（Contact Wrench）引入世界动作模型，使世界预测和动作生成能够利用接触力与力矩信息。",
   "reuse": null,
   "short": "CW-WAM",
   "slug": "cw-wam-contact-wrench-world-action-model",
   "abstract": "Force-sensitive manipulation requires contact and load information that is not provided explicitly by visual observations. We present CW-WAM, a world action model that jointly predicts video, motion, and contact wrench. Inspired by hybrid motion-force control, it contains a unified interaction module that treats both motion and contact wrench as modeling targets and predicts time-aligned motion-wrench chunks as control references. A pair-causal attention mechanism structures the motion-wrench dependencies, while a long fixed-length wrench memory retains past interaction information. The predicted wrench chunk acts as a force reference for a 20-Hz outer-loop admittance controller within a slow-fast control system that maintains force feedback and smooth motion between model queries. CW-WAM improves contact prediction and increases the insertion success rate from 25% to 70% for square pegs with 1-mm clearance per side and from 40% to 95% for power adapters, compared with WAM without wrench input. It also improves lifting-height accuracy across different loads. Our framework with an admittance controller reduces force-reference RMSE from 4.8 to 2.7 N for peg insertion and from 2.0 to 1.3 N for wiping, respectively, with smaller force fluctuations during successful execution. Project website: this https URL",
   "comments": "Project website: this https URL",
   "authors": "Houzhu Ding, Xinda Qi, Luqi Wang, Ming Yang",
   "links": [
    "https://cw-wam-project.github.io/"
   ]
  },
  "2610.30320": {
   "id": "2610.30320",
   "title": "Self-Distilling Future Video Representations for World Action Models",
   "zh": "面向世界动作模型的未来视频表示自蒸馏",
   "org": "待核实（作者：Huiwon Jang、Suhyeok Jang、Jinwoo Shin、Sihyun Yu）。",
   "prob": "WAM 需要利用未来视觉变化学习动作，但完整视频预测会带来较高计算成本。如何保留未来信息中的控制相关特征，是提高模型效率的重要问题。",
   "inno": "研究未来视频表示的自蒸馏，使模型通过蒸馏学习未来视觉特征，并将这些表示用于世界动作建模。",
   "reuse": null,
   "short": "Self-Distilling WAM",
   "slug": "self-distilling-future-video-representations-wam",
   "abstract": "World Action Models (WAMs) have emerged as a promising approach to learning generalist robot policies using rich spatiotemporal priors from pre-trained video diffusion models. A recent line of WAMs improves inference efficiency by conditioning a separate action decoder on intermediate future video representations extracted from a single denoising step, avoiding costly iterative denoising required for video generation. This design makes action prediction depend directly on the quality of these future representations. In this paper, we introduce Future REpresentation learning via self-Distillation with temporal context (FRED), a framework that trains WAMs to improve the quality of future video representations from noise in a single denoising step. Specifically, FRED uses self-distillation to predict future representations from noise, using representations extracted from ground-truth future observations as targets. We further condition the video model on historical observations, providing temporal context for predicting future representations better. Across diverse real-world manipulation scenarios, FRED consistently outperforms competing WAMs. It achieves particularly large gains on long-horizon and out-of-domain tasks under zero-shot environment generalization, and substantially improves human-to-robot skill transfer, reaching 50.0% task success on tasks absent from robot training data and demonstrated only in human videos, compared with 19.4% for the best competing WAM.",
   "comments": "27 pages; Project page: this https URL",
   "authors": "Huiwon Jang, Suhyeok Jang, Hyeonmo Kang, Seungku Kim, Jinwoo Shin, Sihyun Yu",
   "links": [
    "https://huiwon-jang.github.io/FRED/"
   ]
  },
  "2610.28613": {
   "id": "2610.28613",
   "title": "Physical Canvas: A Vision Tactile World Model For Robot Policy Evaluation And Learning",
   "zh": "Physical Canvas：面向机器人策略评估与学习的视觉触觉世界模型",
   "org": "待核实（作者：Zebin Xing、Yuhang Zheng、Yichen Zhang 等）。",
   "prob": "仅根据视觉预测操作结果，难以完整表达接触丰富任务中的物理交互。机器人策略评估还需要考虑接触状态及其变化。",
   "inno": "Physical Canvas 将视觉与触觉共同用于世界建模，研究通过多模态物理预测支持机器人策略评估和策略学习。",
   "reuse": null,
   "short": "Physical Canvas",
   "slug": "physical-canvas-vision-tactile-world-model",
   "abstract": "Evaluating and post-training robot policies through physical interaction is costly and difficult to scale, motivating action-conditioned world models as interactive simulators. However, visually plausible predictions can misrepresent contact dynamics, providing unreliable feedback for policy evaluation and learning. We introduce PhysCanvas, a state-consistent visuo-tactile world model for contact-rich robot manipulation. PhysCanvas unifies multi-view RGB, tactile images, marker displacement, task state, and actions within a unified vision space. All modalities share a video VAE and pretrained generative backbone, allowing optical touch to reuse visual and temporal priors. Motion images derived through forward kinematics and camera projection provide spatially aligned action conditioning, while an initial RGB anchor and updated reference observations guide sequential generation. Our consistency-aware post-training uses the frozen Scene Reward Model, Modality Reward Model, and State Reward Model to supervise scene preservation, visual-tactile compatibility, and agreement with real task outcomes. The resulting world model supports closed-loop policy rollouts for evaluation and provides imagined interactions and rewards for GRPO policy updates. Across four real-world manipulation tasks, when policies are rolled out in the world model, State Reward Model supervision increases the arithmetic mean of task-wise Pearson correlations between world-model and real-world success rates from 0.602 to 0.885. Policy post-training in the reward-guided world model further increases $\\pi_{0.5}$'s mean real-world success rate from a 59.1% SFT baseline to 70.8%. These results demonstrate the value of unified multimodal modeling and consistency-aware post-training for robot policy evaluation and learning. Project page: this https URL.",
   "comments": "",
   "authors": "Zebin Xing, Yuhang Zheng, Yupeng Zheng, Yichen Zhang, Zichen Li, Shangqing Zhou, Yujie Zang, Yuchen Liu, Zhongpu Xia, Haoran Li, Qichao Zhang",
   "links": [
    "https://zebinx.github.io/PhysCanvas_HomePage/"
   ]
  },
  "2610.31474": {
   "id": "2610.31474",
   "title": "Skeleton-Grounded Latent Action Model for Robot World Models",
   "zh": "面向机器人世界模型的骨架约束潜在动作模型",
   "org": "待核实（作者：Qi Lv、Jiayi Lin、Jianzong Wang、Mike Shou Zheng 等）。",
   "prob": "从视频学习潜在动作时，模型需要将视觉变化与机器人实际运动联系起来。缺少运动结构约束的潜在动作表示，可能难以表达机器人本体的运动规律。",
   "inno": "将机器人骨架信息引入潜在动作学习，利用运动结构为世界模型提供更明确的动作表示。",
   "reuse": null,
   "short": "Skeleton Latent Action",
   "slug": "skeleton-grounded-latent-action-model",
   "abstract": "Robot world models simulate how environments evolve in response to robot actions. Latent action models enable learning dynamics from videos by encoding visual transitions as proxy actions. However, reconstruction-driven latent actions mix robot motion with object responses and other scene changes, complicating their correspondence with robot controls. We propose SkeLAM, a skeleton-guided latent action model for controllable robot world modeling. SkeLAM decomposes latent actions into a kinematic component grounded by robot skeleton supervision and a residual component retaining complementary visual dynamics. Cross-modal swap reconstruction encourages RGB-derived and skeleton-derived kinematic representations to be interchangeable during video reconstruction. The world model incorporates both components through separate conditioning pathways, while branch dropout reduces its dependence on future-derived residual information. A lightweight adapter maps robot actions to the kinematic space, and subsequent world-model fine-tuning adapts generation to these predicted conditions. At inference, the model generates future observations from the current observation and robot actions. On DROID, SkeLAM reduces end-effector ADE by 19.5% and LPIPS by 11.1% compared with DreamDojo. Motion probing supports kinematic specialization, while qualitative results demonstrate zero-shot human-to-robot motion transfer in generated rollouts. Policy evaluation shows agreement between predicted success rates and those measured in real robot executions.",
   "comments": "Project Page: this https URL",
   "authors": "Qi Lv, Jiayi Lin, Yuxin Jiang, Jianzong Wang, Xiang Deng, Mike Shou Zheng",
   "links": [
    "https://aopolin-lv.github.io/SkeLAM/"
   ]
  },
  "2610.27387": {
   "id": "2610.27387",
   "title": "Training Controllable World Models via Action Influence Maximization",
   "zh": "通过最大化动作影响训练可控世界模型",
   "org": "待核实（作者：Samuel M. Bateman、Benjamin Eysenbach、Jaime Fernández Fisac、Dhruv Shah 等）。",
   "prob": "世界模型即使能够预测未来，也未必充分学习动作对环境状态的影响。机器人规划需要区分由自身动作控制的变化和其他环境变化。",
   "inno": "将动作影响最大化引入世界模型训练，使预测表示更加关注可由动作改变的状态因素，从而提高世界模型对控制与规划的适用性。论文已被 CoRL 2026 接收。",
   "reuse": "wiki/entities/paper-aim-action-influence-maximization.md"
  },
  "2610.30097": {
   "id": "2610.30097",
   "title": "AtomicRL: Atomic Reinforcement Learning for Long-Horizon Robotic Manipulation",
   "zh": "AtomicRL：面向长程机器人操作的原子级强化学习",
   "org": "待核实（作者：Likui Zhang、Xiaodan Liang、Liang Lin 等）。",
   "prob": "长程操作包含多个连续子任务。使用整个任务的最终成功信号进行强化学习，难以判断哪些局部动作真正促进了任务完成。",
   "inno": "AtomicRL 研究原子级强化学习，将长程操作中的局部行为作为学习的重要单位，改善长程任务中的学习信号分配。",
   "reuse": "wiki/entities/paper-atomicrl.md"
  },
  "2610.29161": {
   "id": "2610.29161",
   "title": "SimPT: An Automated Simulation Framework for Intervention-Based VLA Post-Training",
   "zh": "SimPT：面向干预式 VLA 后训练的自动化仿真框架",
   "org": "待核实（作者：Caicheng Wang、Zipeng Ye、Yuchen Xie 等）。",
   "prob": "VLA 在部署过程中需要通过失败纠正提高性能，但真实机器人上的人工干预数据采集成本较高。",
   "inno": "SimPT 将干预式后训练流程引入自动化仿真环境，研究通过仿真中的执行、干预与策略更新降低后训练的数据采集成本。",
   "reuse": null,
   "short": "SimPT",
   "slug": "simpt-intervention-based-vla-post-training",
   "abstract": "Human-in-the-loop (HIL) post-training improves vision-language-action (VLA) policies through expert corrections, but costly human supervision and repeated real-robot evaluation limit algorithm exploration and development. We introduce SimPT (Simulation-based Post-Training), a simulation framework that automates intervention-based VLA post-training by integrating policy rollouts, corrective data collection, policy training, and standalone evaluation. To automatically determine intervention timing and recovery strategies, we develop an Agent-Compiled Progress Graph that tracks task progress using privileged simulator state and retains stable snapshots during policy execution. After a rollout fails, a CuRobo-based expert attempts recovery from these snapshots, starting at more advanced task stages and falling back to earlier ones as needed to generate corrective trajectories. We implement and compare four post-training methods, RECAP, HG-DAgger, DPO+SFT, and FlowPRO, across seven RoboTwin manipulation tasks. Relative to a fixed-interval snapshot baseline, the Progress Graph reduces recovery attempts by 75.0% and increases recovery success from 93.4% to 98.9%. Using SimPT, we develop RECAP-CF-LPF, a refinement of RECAP that combines cross-fitting and low-pass filtering of value predictions. This refinement increases the overall success rate from 76.50% to 79.36% in held-out simulation trials. In real-robot HIL post-training, it increases the success rate from 72% to 96% across 25 scenarios of an Ethernet cable-insertion task. These results demonstrate the utility of SimPT for exploring and developing VLA post-training algorithms.",
   "comments": "11 pages, 6 figures",
   "authors": "Caicheng Wang, Zipeng Ye, Zhexuan Zhou, Zilong Huang, Xi Zhang, Yuchen Xie",
   "links": []
  },
  "2610.27736": {
   "id": "2610.27736",
   "title": "AWLS: Advantage-Weighted Latent Steering for One-Step Generative Robotic Manipulation Policies",
   "zh": "AWLS：面向单步生成式机器人操作策略的优势加权潜在引导",
   "org": "待核实（作者：Jiahong Tang、Jiaju He、Zhiyuan Yang 等）。",
   "prob": "单步生成式策略具有较低推理延迟，但策略优化仍需要区分高质量动作与低质量动作。",
   "inno": "AWLS 将优势加权与潜在空间引导结合，通过动作价值信息调整生成式策略的潜在表示，研究在保留单步生成效率的同时改善操作性能。论文已被 CoRL 2026 接收。",
   "reuse": null,
   "short": "AWLS",
   "slug": "awls-advantage-weighted-latent-steering",
   "abstract": "Human-collected real-world robotic manipulation demonstrations often contain mixed-quality trajectories, including hesitant, indirect, or inefficient action segments. While diffusion- and flow-based policies can model such multimodal data, standard behavior cloning (BC) assigns equal likelihood to all demonstrated actions and may therefore reproduce suboptimal modes. Offline RL value estimates offer a natural signal for biasing policies toward higher-value behavioral modes, but directly fine-tuning the generative policy or using critic gradients for policy steering can amplify critic errors and induce prior drift. We propose Advantage-Weighted Latent Steering (AWLS), an offline post-training method for improving frozen one-step generative manipulation policies without updating their decoders. AWLS learns a residual noise shaper, an observation-conditioned residual correction to the policy's latent noise, and trains it with an advantage-weighted action reconstruction objective. The advantages are detached, scale-normalized, and used only as supervised weights on demonstrated actions, avoiding critic-gradient policy updates while preserving fast single-step inference. We further introduce a rollout-free decoder-sensitivity criterion for selecting pretrained checkpoints that remain responsive to latent-noise perturbations. Experiments on simulated and real-world robotic manipulation tasks with clean and mixed-quality demonstrations show that AWLS improves over the frozen base policy while limiting drift from the pretrained behavior prior. Project page: this https URL",
   "comments": "Accepted at CoRL 2026",
   "authors": "Jiahong Tang, Jiaju He, Zhiyuan Yang, Ruyang Han, Haiyue Zhu",
   "links": [
    "https://awls-corl2026.github.io/"
   ]
  },
  "2610.29117": {
   "id": "2610.29117",
   "title": "Learning to Absorb Corrections: Critic-Weighted Residual Flow for Vision-Based Insertion",
   "zh": "学习吸收纠正：面向视觉插入任务的 Critic 加权残差 Flow",
   "org": "待核实（作者：Yiqing Yuan、Zhi Li、Hui Cheng 等）。",
   "prob": "视觉插入任务需要较高位置精度。基础策略执行过程中出现的误差，需要通过纠正动作进行补偿，但不同纠正动作的价值并不相同。",
   "inno": "将 Critic 加权与残差 Flow 结合，研究利用纠正反馈优化视觉插入策略，使策略能够学习更有效的误差补偿行为。",
   "reuse": null,
   "short": "Critic-Weighted Residual Flow",
   "slug": "critic-weighted-residual-flow-insertion",
   "abstract": "Vision-based insertion requires correcting alignment errors while retaining useful behavior learned from demonstrations. We present ACT--Flow, an online learning framework that couples an adaptive Action Chunking Transformer (ACT) with a conditional residual flow policy. Following anchor-guided warmup, ACT absorbs successful executed actions through behavioral cloning, while residual targets are recomputed relative to the updated prior. A chunk-level critic learns from successful and failed interactions and evaluates final composed actions. Its estimates weight generated residual candidates for flow matching, complemented by supervision from successful real executions. Candidate scoring is confined to training; deployment uses RGB observations and samples one residual per action chunk. We evaluate the framework on power-plug, USB, and key insertion, together with cable routing and needle threading. Results show a five-task mean success rate of 66% at 20,000 post-initialization interaction steps, exceeding the strongest baseline by 18 percentage points at this checkpoint. Evaluations at matched checkpoints with the residual disabled show ACT's mean success increasing from 16% to 76%. Component ablations further support the complementary roles of behavior absorption, value weighting, and successful-action supervision.",
   "comments": "9 pages, 7 figures",
   "authors": "Yiqing Yuan, Zhi Li, Jinrong Yang, Mingshan Sun, Mei Yu Zhi, Yong Zhao, Liangliang Ren, Hui Cheng",
   "links": []
  },
  "2610.31010": {
   "id": "2610.31010",
   "title": "Revisiting Distribution Matching for State-to-Vision Policy Distillation in Laparoscopic Robotic Manipulation",
   "zh": "重新研究腹腔镜机器人操作中从状态到视觉的策略蒸馏分布匹配",
   "org": "待核实（作者：Mingyu Sheng、Jianan Fan、Ron Kikinis、Weidong Cai 等）。",
   "prob": "使用完整状态训练的教师策略能够获得更多环境信息，但视觉学生策略只能根据相机观测执行动作。两者的输入信息和闭环状态分布存在差异。",
   "inno": "重新研究 State-to-Vision Policy Distillation 中的分布匹配问题，重点关注教师与学生之间的状态分布差异对蒸馏效果的影响。",
   "reuse": null,
   "short": "Laparoscopic Distillation",
   "slug": "laparoscopic-state-to-vision-policy-distillation",
   "abstract": "End-to-end control policies trained via Reinforcement Learning (RL) have emerged as an alternative to conventional surgical robotic control approaches, with recent studies exploring state-based policies trained on privileged state observations. However, such approaches rely on precise physical states that are difficult to obtain reliably in surgical environments. Vision-based RL provides an alternative by learning manipulation policies directly from RGB observations, avoiding explicit state estimation and manually engineered perception pipelines while enabling policies to acquire task-relevant visual representations. Nevertheless, obtaining vision-based policies via pure RL methods for surgical manipulation remains challenging. Thus, imitation learning (IL) becomes another solution, yet it requires a large number of expert demonstrations, which are difficult to obtain in minimally invasive surgery. In this work, we propose a framework for state-to-vision policy distillation in laparoscopic robotic manipulation. Specifically, a state-based policy with privileged observations is employed as the teacher, and a vision-based policy learns from RGB observations and robot proprioceptive information, which requires less expert demonstrations than IL and introduces superior performance than RL methods. We conduct extensive experiments on various surgical manipulation tasks in the simulated environment. Experimental results show that, in the context of laparoscopic robotic manipulation, training vision-based policies with general-purpose RL methods is challenging, whereas our proposed state-to-vision policy distillation leads to more consistent and stable performance across tasks.",
   "comments": "8 pages, 3 figures, 2 tables",
   "authors": "Mingyu Sheng, Jianan Fan, Dongnan Liu, Guoyan Zheng, Ron Kikinis, Weidong Cai",
   "links": []
  },
  "2610.30735": {
   "id": "2610.30735",
   "title": "When One Step Is Enough: An Empirical Study of Euler Integration in Flow-Matching Policies",
   "zh": "何时一步就够：Flow Matching 策略中欧拉积分的实证研究",
   "org": "待核实（作者：Shivakanth Sujit、Luca Nunziante、Rousslan Fernand Julien Dossa、Kai Arulkumaran）。",
   "prob": "Flow Matching Policy 通常需要通过数值积分生成动作。增加积分步数会提高推理开销，但更多步数不一定带来相应的控制收益。",
   "inno": "系统研究欧拉积分步数对生成式机器人策略的影响，重点分析单步积分在不同条件下的适用性。论文已被 CoRL 2026 接收。",
   "reuse": null,
   "short": "One-Step Euler FM",
   "slug": "one-step-euler-integration-flow-matching-policies",
   "abstract": "Flow matching is now the dominant approach for vision-language action (VLA) robot policies, thanks to its expressiveness and simple training objective. However, inference for standard flow matching requires iteratively solving an ODE, limiting the control frequency at deployment. While several works have proposed novel methods to allow for single-shot action generation, we ask whether such additional objectives are actually necessary. We find that across six flow matching VLAs, one-step action generation largely matches multi-step sampling at scale: multi-step integration confers little to no consistent advantage at inference. We empirically find that this holds as long as models are sufficiently trained and have enough capacity. Practically, the implication is that for large VLAs, the single-step policy comes essentially for free from the multi-step model, enabling low-latency deployment without retraining, distillation, or auxiliary objectives.",
   "comments": "Accepted at the 10th Conference on Robot Learning (CoRL 2026)",
   "authors": "Shivakanth Sujit, Luca Nunziante, Rousslan Fernand Julien Dossa, Kai Arulkumaran",
   "links": []
  },
  "2610.29997": {
   "id": "2610.29997",
   "title": "DONIC: Generalist Dexterous Bimanual Tracking Control from Human Demonstrations",
   "zh": "DONIC：从人类示范学习通用灵巧双臂运动跟踪控制",
   "org": "待核实（作者：Mingze Sun、Ziang Cao、Zhaoxi Chen、Ziwei Liu 等）。",
   "prob": "双臂灵巧操作需要协调双手和手指运动。人类示范包含自然的双手协作行为，但机器人需要将这些运动转换为可执行的控制策略。",
   "inno": "DONIC 研究从人类示范学习通用双臂灵巧运动跟踪控制，使双臂策略能够利用人类操作中的协调运动信息。",
   "reuse": "wiki/entities/paper-donic.md"
  },
  "2610.29716": {
   "id": "2610.29716",
   "title": "Reconstruction-Augmented Visuomotor Pre-Training of 3D-Grounded Bimanual Dexterity from RGB-Only Human Videos",
   "zh": "利用三维重建增强的视觉运动预训练，从纯 RGB 人类视频学习双臂灵巧操作",
   "org": "待核实（作者：Xingyi He、Zhiwen Qiu、Kuan Fang 等）。",
   "prob": "人类 RGB 视频缺少机器人动作标签和直接可用的三维操作状态，限制了双臂策略从大规模人类视频中学习。",
   "inno": "将三维重建引入视觉运动预训练，使模型能够利用纯 RGB 视频中的三维操作信息学习双臂灵巧能力。",
   "reuse": null,
   "short": "Recon-Augmented Bimanual Pretrain",
   "slug": "reconstruction-augmented-bimanual-pretraining-rgb-human-video",
   "abstract": "We present SpatialDex, a framework that learns 3D-grounded bimanual dexterity from RGB-only human videos through reconstruction-augmented visuomotor pre-training. The central idea is to learn human behavior together with its spatial context. From each video, we reconstruct the scene geometry and estimate the hand motion, align them at a shared video-level scale using the hand-size prior of 3D hand estimates. We retain the reconstructed geometry as both a policy input and a prediction target. The policy grounds the token grid of a pre-trained vision-language model in this geometry through gated residual fusion, is trained to predict future scene geometry alongside wrist and finger actions, and generates finger motion conditioned on the predicted trajectories of both wrists. Pre-trained on 1M human-video clips and fine-tuned with a small number of robot demonstrations, SpatialDex outperforms the state-of-the-art methods by 16.5 points on nine simulated dexterous tasks across two embodiments and by 38.3 points on six real-world bimanual dexterous tasks. Under a matched comparison, pre-training with reconstructed geometry improves success by 14.2 points over pre-training on hand motion alone.",
   "comments": "Project page: this https URL",
   "authors": "Xingyi He, Zhiwen Qiu, Adhitya Polavaram, Yunhao Cao, Cory Fan, Julius Samwer, Saleh Kalantari, Kuan Fang",
   "links": [
    "https://spatial-dex.github.io/"
   ]
  },
  "2610.28628": {
   "id": "2610.28628",
   "title": "Learning Adaptable Dexterous Skills via Hybrid Dexterous Mimicking",
   "zh": "通过混合灵巧模仿学习可适应的灵巧操作技能",
   "org": "待核实（作者：Yuxuan Kuang、Katerina Fragkiadaki、Shubham Tulsiani 等）。",
   "prob": "灵巧手操作需要同时满足手指协调、接触约束和任务目标。直接模仿固定轨迹，难以适应物体位置和操作条件的变化。",
   "inno": "研究混合式灵巧运动模仿，将示范运动学习与技能适应结合，使机器人能够在不同操作条件下调整已学习的灵巧技能。",
   "reuse": "wiki/entities/paper-adadex.md"
  },
  "2610.27995": {
   "id": "2610.27995",
   "title": "Learning Tactile-Aware Hand Representations from Human Hand-Object Interaction Data",
   "zh": "从人手与物体交互数据中学习触觉感知手部表示",
   "org": "待核实（作者：Yeonguk Yu、Jaehong Kim、Byungok Han）。",
   "prob": "人类手部操作数据包含丰富的接触行为，但单纯学习手部姿态，难以完整表示手与物体之间的触觉交互。",
   "inno": "将触觉相关信息引入手部表示学习，使模型能够从人手与物体交互数据中学习更适合接触操作的手部特征。",
   "reuse": null,
   "short": "Tactile-Aware Hand Rep",
   "slug": "tactile-aware-hand-representations-human-hoi",
   "abstract": "Tactile representation learning for dexterous manipulation often requires substantial robot-side data collection, limiting its scalability across different hands and sensors. We explore whether large-scale human hand-object interaction (HOI) data can provide a useful prior for robotic tactile learning. To this end, we propose HuTact, a framework that pretrains tactile-aware hand representations from human HOI data without requiring real tactile signals during pretraining. HuTact transfers the interaction structure learned from human hand motions to robotic tactile observations and can be further adapted with a small amount of unlabeled robot interaction data. We evaluate HuTact on tactile perception and manipulation tasks, including grasp prediction, slip detection, object classification, and tactile-conditioned policy learning. Across these tasks, human HOI pretraining improves data efficiency and provides a strong initialization for downstream learning. Notably, with HOI pretraining followed by midtraining on unlabeled robot play data, HuTact achieves 88.26% Macro-F1 on grasp prediction using only 25\\% of the labeled downstream data, comparable to training from scratch on the full labeled dataset (88.23%). These results suggest that human interaction data can serve as a scalable complementary source for robotic tactile representation learning.",
   "comments": "9 pages, 4 figures",
   "authors": "Yeonguk Yu, Jaehong Kim, Byungok Han",
   "links": []
  },
  "2610.30988": {
   "id": "2610.30988",
   "title": "In-Context Learning World Models for Robot Dual-Arm Manipulation of Soft Garment with Partial Observations",
   "zh": "面向部分可观测柔性衣物双臂操作的上下文学习世界模型",
   "org": "待核实（作者：Hengyi Sim、Quentin Rouxel、Dennis Hong、Darwin Caldwell、Fei Chen）。",
   "prob": "柔性衣物的形状会随接触动作持续变化。相机只能观察部分布料表面，遮挡区域的状态难以直接估计，使双臂操作规划更加复杂。",
   "inno": "将 In-Context Learning 引入柔性衣物世界模型，研究如何利用上下文中的交互信息预测衣物状态变化，并支持部分可观测条件下的双臂操作。",
   "reuse": null,
   "short": "In-Context Garment WM",
   "slug": "in-context-world-model-dual-arm-garment",
   "abstract": "Robust manipulation of soft objects, such as garments, is a critical skill for service and household robots to adopt, but remains challenging due to its high-dimensional state space, and material-dependent deformations. Humans address this challenge by using mental world models that \"imagine\" the shape and material properties from partial observations, which enables them to predict how actions affect future states. As such, embedding such world models in robots allows for better generalization in manipulating a wide variety of garments in the real world. However, learned world models typically require large amounts of real world data that is strenuous to collect, and a mechanism to infer latent properties that govern dynamics. We present a Soft-Object World Model that performs generative full-state estimation and action-conditioned dynamics prediction using a transformer backbone trained on large-scale simulated soft object interactions, with measures to mitigate the sim-to-real gap. By learning to infer latent shape and material cues through in-context learning over a variable-length partial point-cloud history, the model generalizes across robot embodiments, material parameters, garment types and shapes, and manipulation coordination modes without weight updates. The world model is then integrated with a Goal-Conditioned action planner for motion synthesis. We validate the approach in simulation and real world for bimanual garment folding, reporting improvements in reconstruction fidelity, prediction accuracy, and task success.",
   "comments": "",
   "authors": "Hengyi Sim, Quentin Rouxel, Dennis Hong, Darwin Caldwell, Fei Chen",
   "links": []
  },
  "2610.30694": {
   "id": "2610.30694",
   "title": "Live Thread Reconstruction for Autonomous Manipulation",
   "zh": "面向自主操作的实时线状物体重建",
   "org": "待核实（作者：Emma Huang、Omar Hernandez、Zih-Yun Chiu、Michael Yip）。",
   "prob": "线状柔性物体容易发生弯曲、遮挡和形变，机器人需要持续估计其形状才能进行精细操作。",
   "inno": "研究操作过程中的实时线状物体重建，使机器人能够利用更新后的物体几何状态辅助自主操作。",
   "reuse": null,
   "short": "Live Thread Reconstruction",
   "slug": "live-thread-reconstruction-manipulation",
   "abstract": "Autonomous robotic surgery has drawn increased attention in recent years due to its potential to relieve surgeons of time-consuming, mentally straining, and tedious tasks, particularly during prolonged operations. One suitable task for automation is suturing, which involves tracking the thread as it deforms during needle retrieval, de-tangling, knot tightening, and other tasks. Unlike rigid-object manipulation, deformable thread manipulation has no fixed pose to track and effectively infinite degrees of freedom, making its configuration difficult to predict and requiring the thread's state to be re-estimated rather than assumed. At the same time, thread manipulations involve sequential motions that depend on past and current thread contact points, which must be accurately and continuously tracked. This paper presents an image-to-manipulation pipeline that concurrently generates live thread reconstruction while conducting safe and optimized sequential manipulations to achieve long-horizon tasks.",
   "comments": "",
   "authors": "Emma Huang, Omar Hernandez, Zih-Yun Chiu, Michael Yip",
   "links": []
  },
  "2610.28948": {
   "id": "2610.28948",
   "title": "COWBOY: A Scalable Sim-to-Real Framework for Learning Contextual Whole-Body Manipulation",
   "zh": "COWBOY：面向情境化全身操作学习的可扩展仿真到真实框架",
   "org": "待核实（作者：Jiahui Yang、Deepak Pathak、Ruslan Salakhutdinov 等）。",
   "prob": "全身操作需要协调身体姿态、移动和物体交互。不同任务情境对全身运动的要求不同，增加了策略训练与真实部署难度。",
   "inno": "COWBOY 研究可扩展的仿真到真实学习框架，将任务情境信息引入全身操作学习，提高全身控制策略对不同操作条件的适应能力。",
   "reuse": "wiki/entities/paper-cowboy-contextual-whole-body-manipulation.md"
  },
  "2610.25785": {
   "id": "2610.25785",
   "title": "Long-Horizon Multi-Object Rearrangement via Distilled Tree Search over Continuous Actions",
   "zh": "通过连续动作空间树搜索蒸馏实现长程多物体重排",
   "org": "待核实（作者：Kallol Saha、Yining Hong、Yejin Choi、Maxim Likhachev、David Held 等）。",
   "prob": "长程物体重排需要考虑当前动作对后续操作的影响。树搜索能够规划未来，但在连续高维动作空间中直接搜索计算成本很高。",
   "inno": "提出 DISCO（连续动作树搜索蒸馏）。使用 Progressive-Widening MCTS 在物理仿真中生成长程计划，由学习到的策略提出连续动作，价值函数评估任务进度；随后将搜索结果周期性蒸馏到 Policy-Value Model，使下一轮搜索获得更好的动作候选和价值估计。最终部署时直接执行蒸馏后的策略，无需在线仿真搜索。在仿真和真实多物体重排任务中，DISCO 优于论文比较的行为克隆、强化学习及搜索基线。论文获 CoRL 2026 Oral。",
   "reuse": null,
   "short": "Distilled Tree Search Rearrangement",
   "slug": "distilled-tree-search-multi-object-rearrangement",
   "abstract": "Despite recent advances in policy learning, long-horizon manipulation remains difficult because learned policies must avoid compounding errors while preserving future feasibility. While search-based planning can explicitly reason over such future consequences, it quickly becomes expensive in high-dimensional continuous action spaces. This raises a central question: how can long-horizon reasoning from search be transferred into policies in continuous action spaces? To this end, we propose DISCO: Distilled Tree Search over Continuous Actions. DISCO uses progressive-widening MCTS in physics simulation to generate long-horizon plans. During search, a learned policy proposes continuous actions, while a learned value function scores actions by their expected progress toward task completion. Using GPU-parallelized simulation, DISCO periodically distills MCTS-generated plans into the policy-value model, improving the action proposals and value estimates used in subsequent searches. Finally, the learned policy-value model can be directly deployed without the need for simulator-based search at test time. Across simulated and real-world rearrangement tasks, DISCO outperforms behavior cloning, reinforcement learning, and search-based baselines.",
   "comments": "Accepted at CoRL 2026 (Oral). Project page: this https URL",
   "authors": "Kallol Saha, Yining Hong, Amber Li, Yejin Choi, Maxim Likhachev, David Held",
   "links": [
    "https://distilled-search.github.io"
   ]
  },
  "2610.29560": {
   "id": "2610.29560",
   "title": "From VLM Planners to Robotic Agents: Self-Evolving Harnesses for History-Dependent Manipulation",
   "zh": "从视觉语言规划器到机器人智能体：面向历史依赖操作的自演化执行框架",
   "org": "待核实（作者：Tingwen Du、Chunhua Shen、Xiaojun Chang 等）。",
   "prob": "历史依赖操作需要根据过去的观测、动作和执行结果进行决策。单次视觉语言规划难以持续处理任务状态变化。",
   "inno": "研究自演化机器人执行框架，使视觉语言规划能力与历史信息管理、动作执行和反馈更新结合，支持持续的任务决策。",
   "reuse": null,
   "short": "Self-Evolving Harness",
   "slug": "self-evolving-harness-history-dependent-manipulation",
   "abstract": "Vision-language-action (VLA) models achieve strong performance across embodied manipulation tasks by unifying visual understanding, language conditioning, and action generation. However, long-horizon tasks can involve non-Markovian dependencies: successful decisions may require retaining task progress and past interaction outcomes, while partial observability and execution errors compound over time. Existing work has begun to address these challenges by augmenting foundation-model policies with memory and by introducing hierarchical architectures that separate high-level planning from low-level execution. We organize the high-level VLM as a robotic agent through an editable harness to maintain task context and improve planning through execution experience. We instantiate this approach as SERA (Self-Evolving Robotic Agent), with a fixed VLA executor providing low-level control. The harness maintains and updates task context using observations and execution feedback to guide subsequent planning. Self-evolution uses environment rollouts to propose and evaluate harness revisions, with the selected configuration fixed before held-out testing. On RoboMME, SERA achieves 51.38% overall success with autonomous selection, compared with 44.38% without self-evolution. Our exploration suggests that agentic planning is a promising direction for history-dependent robotic manipulation.",
   "comments": "",
   "authors": "Tingwen Du, Haoyu Guo, Hanxue Zhang, Xuanxu Wang, Junting Dong, Chao Gao, Mingfei Han, Tao Lu, Mulin Yu, Chunhua Shen, Xiaojun Chang, Zhihui Li",
   "links": []
  },
  "2610.31427": {
   "id": "2610.31427",
   "title": "FastHarness: Experience-Driven Review Scheduling for Vision-Language-Action Policies",
   "zh": "FastHarness：面向视觉语言动作策略的经验驱动复查调度",
   "org": "待核实（作者：Tao Lin、Yuxin Du、Haoxiao Wang、Bo Zhao 等）。",
   "prob": "长程机器人执行过程中，需要适时检查任务状态。频繁检查会增加计算开销，检查过少又可能导致错误积累。",
   "inno": "FastHarness 将历史执行经验引入复查调度，研究如何根据经验决定何时重新评估 VLA 的执行状态。",
   "reuse": null,
   "short": "FastHarness",
   "slug": "fastharness-experience-driven-review-scheduling",
   "abstract": "Agentic robot systems combine language-model reasoning with robot policies to monitor execution and recover from failures. However, reviewing every action chunk incurs substantial overhead, even in familiar situations. We introduce FastHarness, an experience-driven execution harness that adaptively schedules GPT-6 Astra supervision of a {\\pi}0.5 policy. A fast path uses stage memory and reviewed outcomes to reduce review frequency as experience accumulates, while a complementary recovery path focuses supervision on correcting errors. Once recovery is verified, routine execution resumes, with failure experience retained to guide future interventions. The policy remains frozen and generates fresh actions from current observations. Across four simulated manipulation tasks, reviewer token use decreases by an average of 57.2% per task from the first evaluated episode to the mean of the final five episodes. Mean episode time decreases from 85.5 to 52.0 minutes over the same interval. On the same tasks, mean success increases from 80% for GPT-as-Policy ({\\pi}0.5 + GPT-6 Astra) to 97.19% with FastHarness. These results highlight adaptive supervision as a practical route to more efficient agentic robotics without additional model training.",
   "comments": "Github:this https URL",
   "authors": "Tao Lin, Yuxin Du, Haoxiao Wang, Zile Zhong, Bo Zhao",
   "links": [
    "https://github.com/FastHarness/fast-harness"
   ]
  },
  "2610.27794": {
   "id": "2610.27794",
   "title": "Multi-View Latent Action Pretraining for Cross-Embodiment Robot Learning from Videos",
   "zh": "面向跨本体机器人视频学习的多视角潜在动作预训练",
   "org": "待核实（作者：Oliver Hausdörfer、Felix Minzenmay、Angela P. Schoellig 等）。",
   "prob": "不同机器人动作空间不一致，而视频通常不包含可直接使用的机器人控制标签。单一视角还可能无法完整观察操作运动。",
   "inno": "研究通过多视角视频学习潜在动作表示，使模型能够利用不同视角的运动信息进行预训练，并将学到的动作表示用于跨机器人本体学习。",
   "reuse": null,
   "short": "Multi-View LAPA",
   "slug": "multi-view-latent-action-pretraining-cross-embodiment",
   "abstract": "Action-free videos are an appealing substitute for costly teleoperated data. Latent action models (LAMs) turn videos into a supervision signal by inferring abstract actions between frames. Existing LAMs observe a single camera view, which resolves 6D robot actions only ambiguously missing out-of-plane movements and potential occlusions. We propose a LAM that ingests synchronized multi-camera views using cross-attention to resolve depth and occlusions before compressing them into latent actions. The latent actions are fed into a policy in which a latent expert and an action expert are coupled by sequential flow matching around a frozen VLM backbone. We further evaluate in strict cross-embodiment settings where task-specific demonstrations come from other embodiments. We show that dual-view LAM pretraining improves the decodability of ground-truth 6D actions by 17% compared to single-view, while scaling single-view data by two orders of magnitude is not sufficient to fully recover out-of-plane motions. The downstream policy improves success rate by +13.5% compared to +6.2% for single-view LAM. A single cross-embodiment is sufficient for skill transfer, and adding a second viewpoint outperforms adding up to four further embodiments. The results transfer to hardware where demonstrations come from a human hand improving success rate by +20% over action-only policies in 850 rollouts. We show that video and robot data are complementary, and that benefits of learning from video are task-dependent. Website: this https URL",
   "comments": "Website: this https URL",
   "authors": "Oliver Hausdörfer, Felix Minzenmay, Mani Majd, Maximilian P. Christof, Shiyuan Zhang, Angela P. Schoellig",
   "links": [
    "https://learnsyslab.github.io/multiview_latent_action_pretraining/"
   ]
  },
  "2610.31661": {
   "id": "2610.31661",
   "title": "OpenTac: An Open Ecosystem for Customizable, Scalable, and Transferable 3D Tactile Sensing",
   "zh": "OpenTac：可定制、可扩展、可迁移的三维触觉感知开放生态",
   "org": "待核实（作者：Ariel Slepyan、Gregory Hager、Homanga Bharadhwaj、Nitish V. Thakor 等）。",
   "prob": "触觉传感器的硬件结构、输出格式和标定方法存在差异，增加了触觉操作策略的数据采集和跨平台迁移难度。",
   "inno": "OpenTac 构建开放的三维触觉感知生态，重点支持传感器定制、规模化部署与跨设备迁移，并提供开源软硬件和演示资源。",
   "reuse": null,
   "short": "OpenTac",
   "slug": "opentac-open-3d-tactile-sensing-ecosystem",
   "abstract": "We present OpenTac, a novel sensor and a full-stack ecosystem for tactile skins that are highly customizable, scalable, and readily deployed in downstream robot control and learning applications. Our core contribution is twofold. First, we develop a scalable resistive tactile skin with a structured deformable overlay that converts tangential loading into spatially varying pressure. Differential responses across neighboring \"taxels\" support decoding of local shear vectors, and the common-mode response encodes normal force. Second, we provide a highly customizable sensor design tool that is able to automatically generate fabrication files for these tactile skins from 2D outlines and developable 3D surfaces, including from abstract descriptions such as text (e.g., \"design a sensor for a size 9 shoe insole\"). We develop an ecosystem around the OpenTac sensor, including the CAD design tool, a sensor simulator, cross-sensor transfer, and fast/adaptive data acquisition. We demonstrate the broad applicability of the OpenTac-CAD sensor design tool by designing and fabricating more than 30 sensors across a variety of robot embodiments. We characterize the three-dimensional tactile fields produced by OpenTac with a commercial, high-precision force/torque sensor. We demonstrate the broad applicability of 3D tactile sensing across contact-rich feedback control and manipulation policy learning on four real-world robot embodiments. Additionally, adaptive acquisition enables kilohertz-rate partial readouts, exceeding full-array raster scanning on the same sensor by 19.2X at 16 sampled taxels per update for a 1024-taxel skin. We provide all hardware, software, and result videos are at this http URL",
   "comments": "14 pages, 14 figures. Project website, open-source hardware and software, and demonstration videos: this https URL",
   "authors": "Ariel Slepyan, Geeta Chandra Raju Bethala, Jiafeng Gu, Anagha Rajendra Dangle, Yinzhe Zhou, Hongyuan Fu, Jade Sim, Tuhin Basu, Xinyi Yin, Md Faizal Karim, Gregory Hager, Homanga Bharadhwaj, Anand Bhattad, Krishna Murthy Jatavallabhula, Nitish V. Thakor",
   "links": [
    "https://opentac.github.io",
    "http://opentac.github.io"
   ]
  }
 },
 "lists": {
  "hq": [
   {
    "id": "2610.25711",
    "n": 1,
    "star": true,
    "sec": "人形机器人"
   },
   {
    "id": "2610.25701",
    "n": 2,
    "star": false,
    "sec": "人形机器人"
   },
   {
    "id": "2610.26043",
    "n": 3,
    "star": false,
    "sec": "人形机器人"
   },
   {
    "id": "2610.26169",
    "n": 4,
    "star": false,
    "sec": "人形机器人"
   },
   {
    "id": "2610.26170",
    "n": 5,
    "star": false,
    "sec": "人形机器人"
   },
   {
    "id": "2610.26584",
    "n": 6,
    "star": false,
    "sec": "人形机器人"
   },
   {
    "id": "2610.26855",
    "n": 7,
    "star": false,
    "sec": "人形机器人"
   },
   {
    "id": "2610.27162",
    "n": 8,
    "star": false,
    "sec": "人形机器人"
   },
   {
    "id": "2610.27242",
    "n": 9,
    "star": false,
    "sec": "人形机器人"
   },
   {
    "id": "2610.27688",
    "n": 10,
    "star": false,
    "sec": "人形机器人"
   },
   {
    "id": "2610.28470",
    "n": 11,
    "star": false,
    "sec": "人形机器人"
   },
   {
    "id": "2610.28480",
    "n": 12,
    "star": true,
    "sec": "人形机器人"
   },
   {
    "id": "2610.28523",
    "n": 13,
    "star": true,
    "sec": "人形机器人"
   },
   {
    "id": "2610.28712",
    "n": 14,
    "star": false,
    "sec": "人形机器人"
   },
   {
    "id": "2610.28925",
    "n": 15,
    "star": false,
    "sec": "人形机器人"
   },
   {
    "id": "2610.29169",
    "n": 16,
    "star": false,
    "sec": "人形机器人"
   },
   {
    "id": "2610.29612",
    "n": 17,
    "star": true,
    "sec": "人形机器人"
   },
   {
    "id": "2610.29903",
    "n": 18,
    "star": false,
    "sec": "人形机器人"
   },
   {
    "id": "2610.30044",
    "n": 19,
    "star": true,
    "sec": "人形机器人"
   },
   {
    "id": "2610.30061",
    "n": 20,
    "star": true,
    "sec": "人形机器人"
   },
   {
    "id": "2610.30080",
    "n": 21,
    "star": true,
    "sec": "人形机器人"
   },
   {
    "id": "2610.30334",
    "n": 22,
    "star": false,
    "sec": "人形机器人"
   },
   {
    "id": "2610.30500",
    "n": 23,
    "star": false,
    "sec": "人形机器人"
   },
   {
    "id": "2610.30540",
    "n": 24,
    "star": false,
    "sec": "人形机器人"
   },
   {
    "id": "2610.30817",
    "n": 25,
    "star": false,
    "sec": "人形机器人"
   },
   {
    "id": "2610.30876",
    "n": 26,
    "star": false,
    "sec": "人形机器人"
   },
   {
    "id": "2610.31166",
    "n": 27,
    "star": false,
    "sec": "人形机器人"
   },
   {
    "id": "2610.31608",
    "n": 28,
    "star": true,
    "sec": "人形机器人"
   },
   {
    "id": "2610.25628",
    "n": 29,
    "star": false,
    "sec": "人形机器人"
   },
   {
    "id": "2610.29030",
    "n": 30,
    "star": false,
    "sec": "人形机器人"
   },
   {
    "id": "2610.27315",
    "n": 1,
    "star": false,
    "sec": "四足机器人"
   },
   {
    "id": "2610.27472",
    "n": 2,
    "star": false,
    "sec": "四足机器人"
   },
   {
    "id": "2610.27998",
    "n": 3,
    "star": false,
    "sec": "四足机器人"
   },
   {
    "id": "2610.29586",
    "n": 4,
    "star": true,
    "sec": "四足机器人"
   },
   {
    "id": "2610.25962",
    "n": 1,
    "star": true,
    "sec": "轮足与通用腿式机器人"
   },
   {
    "id": "2610.26550",
    "n": 2,
    "star": true,
    "sec": "轮足与通用腿式机器人"
   },
   {
    "id": "2610.26414",
    "n": 3,
    "star": false,
    "sec": "轮足与通用腿式机器人"
   },
   {
    "id": "2610.26810",
    "n": 4,
    "star": false,
    "sec": "轮足与通用腿式机器人"
   },
   {
    "id": "2610.29438",
    "n": 5,
    "star": false,
    "sec": "轮足与通用腿式机器人"
   }
  ],
  "man": [
   {
    "id": "2610.26854",
    "n": 1,
    "star": false,
    "sec": "VLA / 通用操作策略"
   },
   {
    "id": "2610.31423",
    "n": 2,
    "star": false,
    "sec": "VLA / 通用操作策略"
   },
   {
    "id": "2610.31569",
    "n": 3,
    "star": false,
    "sec": "VLA / 通用操作策略"
   },
   {
    "id": "2610.28814",
    "n": 4,
    "star": false,
    "sec": "VLA / 通用操作策略"
   },
   {
    "id": "2610.28019",
    "n": 5,
    "star": false,
    "sec": "VLA / 通用操作策略"
   },
   {
    "id": "2610.31613",
    "n": 6,
    "star": false,
    "sec": "VLA / 通用操作策略"
   },
   {
    "id": "2610.29692",
    "n": 7,
    "star": false,
    "sec": "VLA / 通用操作策略"
   },
   {
    "id": "2610.30136",
    "n": 1,
    "star": false,
    "sec": "WAM / World Model 与操作预测"
   },
   {
    "id": "2610.26855",
    "n": 2,
    "star": false,
    "sec": "WAM / World Model 与操作预测"
   },
   {
    "id": "2610.30991",
    "n": 3,
    "star": false,
    "sec": "WAM / World Model 与操作预测"
   },
   {
    "id": "2610.28626",
    "n": 4,
    "star": false,
    "sec": "WAM / World Model 与操作预测"
   },
   {
    "id": "2610.30117",
    "n": 5,
    "star": false,
    "sec": "WAM / World Model 与操作预测"
   },
   {
    "id": "2610.30320",
    "n": 6,
    "star": false,
    "sec": "WAM / World Model 与操作预测"
   },
   {
    "id": "2610.28613",
    "n": 7,
    "star": false,
    "sec": "WAM / World Model 与操作预测"
   },
   {
    "id": "2610.31474",
    "n": 8,
    "star": false,
    "sec": "WAM / World Model 与操作预测"
   },
   {
    "id": "2610.27387",
    "n": 9,
    "star": false,
    "sec": "WAM / World Model 与操作预测"
   },
   {
    "id": "2610.30097",
    "n": 1,
    "star": false,
    "sec": "在线/离线强化学习与后训练"
   },
   {
    "id": "2610.29161",
    "n": 2,
    "star": false,
    "sec": "在线/离线强化学习与后训练"
   },
   {
    "id": "2610.27736",
    "n": 3,
    "star": false,
    "sec": "在线/离线强化学习与后训练"
   },
   {
    "id": "2610.29117",
    "n": 1,
    "star": false,
    "sec": "模仿学习、视觉运动策略与生成式动作策略"
   },
   {
    "id": "2610.31010",
    "n": 2,
    "star": false,
    "sec": "模仿学习、视觉运动策略与生成式动作策略"
   },
   {
    "id": "2610.30735",
    "n": 3,
    "star": false,
    "sec": "模仿学习、视觉运动策略与生成式动作策略"
   },
   {
    "id": "2610.29997",
    "n": 1,
    "star": false,
    "sec": "双臂操作"
   },
   {
    "id": "2610.29716",
    "n": 2,
    "star": false,
    "sec": "双臂操作"
   },
   {
    "id": "2610.28628",
    "n": 1,
    "star": false,
    "sec": "灵巧操作"
   },
   {
    "id": "2610.27995",
    "n": 2,
    "star": false,
    "sec": "灵巧操作"
   },
   {
    "id": "2610.30988",
    "n": 1,
    "star": false,
    "sec": "柔性/可变形物体操作"
   },
   {
    "id": "2610.30694",
    "n": 2,
    "star": false,
    "sec": "柔性/可变形物体操作"
   },
   {
    "id": "2610.28948",
    "n": 1,
    "star": false,
    "sec": "人形机器人上半身操作"
   },
   {
    "id": "2610.31166",
    "n": 2,
    "star": false,
    "sec": "人形机器人上半身操作"
   },
   {
    "id": "2610.30044",
    "n": 3,
    "star": false,
    "sec": "人形机器人上半身操作"
   },
   {
    "id": "2610.25785",
    "n": 1,
    "star": false,
    "sec": "长程任务、推理与规划"
   },
   {
    "id": "2610.29560",
    "n": 2,
    "star": false,
    "sec": "长程任务、推理与规划"
   },
   {
    "id": "2610.31427",
    "n": 3,
    "star": false,
    "sec": "长程任务、推理与规划"
   },
   {
    "id": "2610.27794",
    "n": 1,
    "star": false,
    "sec": "数据集、预训练、跨本体与硬件平台"
   },
   {
    "id": "2610.30876",
    "n": 2,
    "star": false,
    "sec": "数据集、预训练、跨本体与硬件平台"
   },
   {
    "id": "2610.31661",
    "n": 3,
    "star": false,
    "sec": "数据集、预训练、跨本体与硬件平台"
   }
  ]
 }
}''')
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
            f"### {sec}（{len(rows)}）\n\n| # | 节点 | arXiv | 状态 |\n|---|------|-------|------|\n" + "\n".join(rows)
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
- 同一 arXiv 论文跨「人形/四足」与「Manipulation」两篇周报时共用同一节点，不重复造页。

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
            f"### {sec}（{len(rows)}）\n\n| # | 论文节点 | arXiv | 状态 |\n|---|----------|-------|------|\n" + "\n".join(rows)
        )
    if key == "hq":
        tags, related, abbr = "[survey, humanoid, quadruped, legged-robot, senlanke]", ["../tasks/locomotion.md", "../tasks/humanoid-locomotion.md", "../methods/reinforcement-learning.md"], ["RL", "WBC", "MPC", "LiDAR", "IMU"]
    else:
        tags, related, abbr = "[survey, manipulation, vla, world-action-model, senlanke]", ["../tasks/manipulation.md", "../methods/vla.md", "../concepts/world-action-models.md"], ["VLA", "WAM", "RL", "IL"]
    other = BLOGS["man" if key == "hq" else "hq"]
    return f"""---
type: overview
tags: {tags}
status: complete
updated: {TODAY}
related:
{chr(10).join(f"  - {r}" for r in related)}
  - {other["overview"]}.md
sources:
  - ../../sources/blogs/{b["file"]}
summary: "senlanke 具身运控lab {WEEK} {b['label']}周报：{n_all} 条论文条目，每条对应唯一 arXiv 详情节点（新建 {n_new} / 复用 {n_all - n_new}）。"
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
- [{other["label"]}周报索引（同期另一篇）]({other["overview"]}.md)

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
        (ROOT / "wiki/overview" / f"{b['overview']}.md").write_text(render_overview(key), encoding="utf-8")
    print(f"Done: {n_new} new + {n_reuse} reuse; {len(PAPERS)} unique arXiv")


if __name__ == "__main__":
    main()
