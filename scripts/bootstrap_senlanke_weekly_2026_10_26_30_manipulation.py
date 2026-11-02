#!/usr/bin/env python3
"""Bootstrap ingest: senlanke 具身运控lab 10.26-10.30 前沿论文动态（Manipulation 57）.

公众号用 scripts/fetch_wechat_article.py（Chromium + 微信 UA）读取；条目 57 个唯一 arXiv，
其中 26 篇已有详情节点（复用，仅追加来源；含与 PR #3293 周报重叠的 Echo / Metatron / DEX-PHD），31 篇新建独立 paper-* 节点。arXiv 摘要/作者/评论由摘要页抓取后内嵌。
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
    "man": {
        "file": "wechat_senlanke_weekly_manipulation_2026-10-26_30.md",
        "title": "【10.26-10.30前沿论文动态】Manipulation 57篇",
        "url": "https://mp.weixin.qq.com/s/uIziIlqQrYGvUp-BivNHPw",
        "overview": "senlanke-weekly-2026-10-26-30-manipulation-technology-map",
        "label": "Manipulation",
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
    "人形机器人上半身操作": ["humanoid", "manipulation"],
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
  "2610.40373": {
   "id": "2610.40373",
   "title": "HiFOR: Hierarchical Foresight with Action-Guided World Grounding for Vision-Language-Action Model",
   "zh": "HiFOR：面向视觉语言动作模型的分层未来预测与动作引导世界信息关联",
   "org": "南京理工大学、新加坡国立大学等。",
   "prob": "VLA 引入未来状态预测后，短期动作变化与长期任务目标通常分别建模。预测得到的大量未来信息也不一定与当前动作有关。",
   "inno": "提出分层未来预测框架，同时预测短期状态、语义里程碑和最终目标。通过结构化潜在世界表示组织不同时间尺度的信息，再由动作生成过程动态检索相关未来特征，减少无关预测信息对动作决策的干扰。",
   "reuse": null,
   "short": "HiFOR",
   "slug": "hifor-hierarchical-foresight-vla",
   "abstract": "Vision-Language-Action (VLA) models have demonstrated promising capabilities in robotic manipulation by integrating visual perception, language understanding, and action generation. Recent advances have further incorporated future prediction to enhance the understanding of environmental dynamics and task progression. However, existing methods often model short-term dynamics and long-term task objectives separately, while directly incorporating predicted futures into action policies without sufficiently considering their relevance to current action requirements. To address these limitations, we propose HiFOR, a hierarchical foresight framework with action-guided world retrieval. Specifically, Hierarchical Foresight Prediction jointly predicts short-term state evolution, semantic milestones, and final goal states through a Hierarchical Dependency Mask (HDM), establishing dependencies across temporal horizons. Building upon these predictions, Structure-Aware Latent World Modeling organizes future latent representations along the horizon, view, and spatial dimensions to facilitate fine-grained retrieval. Furthermore, Action-Guided Factorized World Retrieval adaptively retrieves relevant future information from the structured latent world based on evolving action representations during denoising, providing targeted guidance for action generation. Extensive experiments on two simulation benchmarks and six real-world robotic manipulation tasks demonstrate the effectiveness of our approach.",
   "comments": "",
   "authors": "Ding, Rongxing, Qu, Hongyu, Wu, Kehong, Wang, Xudong, Zhang, Fan, Xiang, Xinguang, Shu, Xiangbo",
   "links": []
  },
  "2610.37381": {
   "id": "2610.37381",
   "title": "PDRVLA: A Vision-Language-Action Model with Physical Dynamics Representations",
   "zh": "PDRVLA：具有物理动力学表示的视觉语言动作模型",
   "org": "北京大学、京东相关研究团队。",
   "prob": "VLA 缺少对环境物理变化的有效建模，而在控制循环中直接生成未来视频会增加推理延迟。",
   "inno": "提出轻量化 Physical Dynamics Module，从视觉语言特征中提取物理 Token，并预测未来 RGB、深度和光流特征。训练时使用冻结的视频世界模型提供辅助监督，推理时无需生成未来视频。在 LIBERO、LIBERO-Plus、RoboTwin 2.0 和 SimplerEnv 上，成功率分别达到 **99.3%、89.9%、93.6% 和 81.5%**。",
   "reuse": null,
   "short": "PDRVLA",
   "slug": "pdrvla-physical-dynamics-representations",
   "abstract": "Vision--Language--Action (VLA) models generate robot actions from visual observations and language instructions, but their understanding of physical dynamics remains limited. Video world models can capture scene dynamics, yet generating future videos introduces high latency into the control loop. We introduce PDRVLA, whose lightweight Physical Dynamics Module extracts physical tokens from vision--language features to condition action generation without online video generation. The module learns through direct prediction of future RGB, depth, and optical-flow features, with auxiliary supervision through a frozen pretrained video world model. Training relies entirely on publicly available data, expanded through human-to-robot synthesis, subtask annotation, and embodied chain-of-thought (ECoT) annotation. Across four simulation benchmarks, PDRVLA achieves success rates of 99.3% on LIBERO, 89.9% on LIBERO-Plus, 93.6% on RoboTwin~2.0, and 81.5% on SimplerEnv. Ablations show improved future prediction with world-model supervision and evaluate how subtask conditioning, future object-pose supervision, and ECoT training affect action success.",
   "comments": "",
   "authors": "Wang, Hanqing, Cai, Zhaotian, An, Ruichuan, Zeng, Bohan, Zhang, Tao, Chen, Yilin, Jiang, Xu, Chen, Kaiquan et al.",
   "links": []
  },
  "2610.38000": {
   "id": "2610.38000",
   "title": "Self-Labeled Preference Transfer for Steerable Robot Manipulation",
   "zh": "面向可控机器人操作的自标注偏好迁移",
   "org": "伊利诺伊大学厄巴纳-香槟分校、加州大学伯克利分校等。",
   "prob": "VLA 可以从标注示范中学习抓取位置、绕障路径等执行偏好，但迁移到新任务后，普通微调容易提高任务成功率，却破坏原有偏好控制能力。",
   "inno": "提出 Self-Labeled Preference Transfer（SPT）。首先联合训练偏好识别器和偏好条件策略，然后冻结识别器，为新任务的未标注示范自动生成偏好标签。在真实机器人任务迁移中，偏好遵循率由 **52.5% 提升至 92.5%**。",
   "reuse": null,
   "short": "SPT 自标注偏好迁移",
   "slug": "self-labeled-preference-transfer-steerable-manipulation",
   "abstract": "Vision-language-action (VLA) policies can learn to follow execution preferences, such as where to grasp an object or how to move around an obstacle, from demonstrations annotated with preference labels. Providing these annotations for every new task, however, is costly. We study whether learned preference control can transfer to new tasks whose demonstrations contain the desired behaviors but lack preference annotations. We find that a policy trained with preference annotations can follow preferences on a new task before adaptation but achieves limited task success. Fine-tuning on target demonstrations without these annotations improves task success while weakening preference control. Motivated by this finding, we introduce Self-Labeled Preference Transfer (SPT). SPT jointly trains a shared vision-language backbone for preference-conditioned action prediction and preference recognition on annotated source demonstrations. It then freezes the recognizer to preserve its cross-task preference recognition capability and uses it to self-label target demonstrations for policy fine-tuning. Across five types of execution preferences in simulation, the recognizer achieves 96.4% accuracy, and SPT reaches 84.8% preference following and 89.2% task success, compared with 59.4% and 88.0% for naive finetuning. We further extend to multiple preferences, achieving 93.8% preference following compared with 42.5% of baselines. In real-world experiments, SPT increases preference following from 52.5% to 92.5% relative to naive finetuning across two task transfers. These results show that SPT can support preference adaptation across tasks and objects while maintaining success, without additional manual preference annotations.",
   "comments": "",
   "authors": "Hong, Kaiwen, Peng, Shaoting, Zeng, Xiangyu, Wang, Pengcheng, Tian, Boyuan, Chen, Haonan, Zhu, Yifan, Wang, Yixiao et al.",
   "links": []
  },
  "2610.37215": {
   "id": "2610.37215",
   "title": "Retrieve to Act: Motion Primitive Graph Retrieval for Robust VLA Execution",
   "zh": "检索后行动：通过运动基元图检索实现鲁棒 VLA 执行",
   "org": "VinUniversity、VinRobotics、德国人工智能研究中心等。",
   "prob": "VLA 在视觉分布发生变化时，容易受到背景、光照和物体外观变化的影响，生成错误的空间运动轨迹。",
   "inno": "提出 Retrieve to Act（R2A），将成功示范组织成时空运动基元图。推理时使用图神经网络检索与当前执行状态匹配的运动基元，再通过门控机制修正冻结 VLA 的动作。无需微调基础策略，仿真成功率最高提升 12 个百分点，真实机器人最高提升 40 个百分点。",
   "reuse": "wiki/entities/paper-retrieve-to-act.md"
  },
  "2610.37378": {
   "id": "2610.37378",
   "title": "VLA-Stem: A Lightweight Safety Mechanism for Vision-Language-Action Models via Reinforcement Learning",
   "zh": "VLA-Stem：基于强化学习的轻量化视觉语言动作模型安全机制",
   "org": "中国科学技术大学、澳门大学等。",
   "prob": "VLA 主要学习完成操作任务，但面对动态障碍物和空间约束时，原始动作块可能发生碰撞。直接重新微调 VLA 成本较高。",
   "inno": "在冻结 VLA 下方增加强化学习安全策略，将 VLA 动作块转换为参考轨迹。安全策略根据本体状态和局部障碍物信息调整动作，同时尽量保留原始任务意图。训练采用 Teacher–Student 蒸馏和课程学习，在六种受约束仿真环境中，平均无碰撞任务成功率达到 **52.8%**，最强安全基线为 **23.7%**。",
   "reuse": null,
   "short": "VLA-Stem",
   "slug": "vla-stem-lightweight-safety-rl",
   "abstract": "Vision-Language-Action (VLA) models integrate high-level reasoning with low-level action execution for end-to-end robotic control. However, most VLA models target static manipulation in unconstrained environments, overlooking spatial constraints and dynamic obstacles. Fine-tuning VLAs on obstacle-avoidance data is a straightforward solution, but collecting such data is costly, especially for dynamic scenarios, and may degrade their original task capabilities. To address these limitations, we propose VLA-Stem, a lightweight, reinforcement-learning-based safety mechanism for VLA systems. Without fine-tuning the underlying VLA policy, VLA-Stem converts its action chunks into motion references and uses trajectory-level guidance to preserve their spatial intent while allowing flexible motion and timing adjustments. Conditioned on these references, robot proprioception, and local obstacle information, the Stem policy adaptively modifies execution at inference time, preserving the original task intent while enabling responsive adaptation to spatial constraints and dynamic obstacles. For robust and sample-efficient training, we combine teacher-student distillation with curriculum learning. We evaluate VLA-Stem on reference following, static spatial constraints, and dynamic obstacle avoidance. Across six constrained simulation settings, VLA-Stem achieves 52.8% average collision-free task success versus 23.7% for the strongest safety baseline, while preserving base-policy performance in obstacle-free settings; real-world experiments further show consistent collision reductions under static and dynamic obstacles. These results demonstrate that VLA-Stem couples high-level VLA task intent with rapid, reflex-like responses to immediate physical risks, providing a brainstem-inspired mechanism for robust robot execution.",
   "comments": "",
   "authors": "Zou, Bojun, Yu, Wenhao, Li, Jiaohao, Lin, Jiaxuan, Ye, Guanting, Geng, Haohan, Yu, Weihao, Zhang, Yanyong et al.",
   "links": []
  },
  "2610.38661": {
   "id": "2610.38661",
   "title": "Quantize by Doing: Task-Guided Mixed-Precision Quantization of Vision-Language-Action Models",
   "zh": "通过任务执行指导视觉语言动作模型的混合精度量化",
   "org": "Colorado School of Mines、东北大学等。",
   "prob": "VLA 模型参数量较大，部署到机器人端侧设备时受到显存和计算资源限制。统一降低模型精度可能损害动作生成能力。",
   "inno": "提出 QbD-VLA，使用短时间闭环任务成功率指导不同网络模块的量化位宽分配，而不是只依赖权重或激活的数值误差。在四种 VLA 模型上研究混合精度量化，权重平均精度约为 4 bit。真实机器人实验同时揭示了仿真量化配置向实际部署迁移时的差异。",
   "reuse": "wiki/entities/paper-qbd-vla.md"
  },
  "2610.38508": {
   "id": "2610.38508",
   "title": "SaiVLA-JEPA: Evaluating Future Latent Prediction in JEPA-Conditioned VLA Control",
   "zh": "SaiVLA-JEPA：评估 JEPA 条件化 VLA 控制中的未来潜在表示预测",
   "org": "Synthoid.ai VLA Research Team。",
   "prob": "JEPA 可以学习未来视觉表示，但增加未来预测监督是否一定能改善 VLA 的闭环操作性能，仍需要直接验证。",
   "inno": "将冻结的 V-JEPA 特征引入 Flow-Based VLA，并增加两个训练期未来特征预测 Head。推理时移除辅助 Head，保持原有动作采样流程。实验发现，当前 JEPA 特征条件化使成功率从 **71.6% 提升至约 82.5%**，但额外的未来预测监督在该训练预算下没有带来可检测的总体收益。",
   "reuse": null,
   "short": "SaiVLA-JEPA",
   "slug": "saivla-jepa-future-latent-prediction",
   "abstract": "Joint-embedding predictive architectures (JEPAs) learn to predict representations rather than pixels, offering a compact way to model how visual scenes evolve. We ask whether this predictive objective improves closed-loop control when a vision-language-action (VLA) policy already uses V-JEPA features. We call our JEPA-conditioned flow policy family SaiVLA-JEPA. It combines a vision-language model with a diffusion transformer action expert and receives features from a frozen V-JEPA encoder. Two lightweight auxiliary heads predict future representations during training and are omitted at inference, leaving the action sampler unchanged. We compare policies without JEPA features, with current JEPA conditioning, and with additional future supervision on 10,030 paired LIBERO-Plus episodes. Current JEPA conditioning reaches about 82.5% success versus 71.6% without JEPA, while future supervision provides no detectable aggregate gain at the evaluated training budget. We separately evaluate SaiVLA-JEPA-TCA, which adapts latent context, and SaiVLA-JEPA-TPA, which adapts selected parameters. The context-retry gain is a system-level result under a larger attempt budget; parameter adaptation shows no consistent advantage over the frozen policy and target controls. These results motivate assessing latent-prediction objectives through paired closed-loop rollouts rather than latent loss alone.",
   "comments": "",
   "authors": "Xiao, Rufeng, Xiao, Ruihao, Chen, Wei, Sun, Xinhai",
   "links": []
  },
  "2610.36011": {
   "id": "2610.36011",
   "title": "E0.5: A Vision-Language-Action World Model",
   "zh": "E0.5：视觉语言动作世界模型",
   "org": "X-Era AI Lab。",
   "prob": "机器人需要将视觉理解、语言指令、环境变化预测和动作生成结合起来，而传统 VLA 主要关注从观测直接预测动作。",
   "inno": "提出 E0.5，将世界状态预测与视觉语言动作建模统一到同一框架，使环境未来变化参与机器人动作学习。",
   "reuse": "wiki/entities/paper-e05-geometry-world-action.md"
  },
  "2610.36791": {
   "id": "2610.36791",
   "title": "RynnWorld-Latent: A Cross-Embodiment World Model Conditioned on Universal Latent Actions",
   "zh": "RynnWorld-Latent：基于通用潜在动作条件的跨本体世界模型",
   "org": "阿里巴巴达摩院相关研究团队。",
   "prob": "不同机器人具有不同动作空间，人类视频又缺少机器人控制标签，导致多来源交互数据难以共同训练可控世界模型。",
   "inno": "使用 RynnLAM 提供的三维潜在动作作为统一条件接口，将人类交互视频、机器人示范和仿真 Rollout 共同用于世界模型预训练。模型使用 23 个数据集、约 5.535 亿帧数据，并支持数字遥操作和潜在动作合成两种数据生成方式，生成数据可用于下游机器人策略训练。",
   "reuse": "wiki/entities/paper-rynnworld-latent.md"
  },
  "2610.36823": {
   "id": "2610.36823",
   "title": "TAPESTRY: Learning Unified Latent Actions from Heterogeneous Robot and Human Videos",
   "zh": "TAPESTRY：从异构机器人与人类视频中学习统一潜在动作",
   "org": "待核实（作者联合研究团队，机构信息以论文首页为准）。",
   "prob": "人类视频和不同机器人视频中的运动形式不一致，直接使用各自原始动作空间难以形成统一表示。",
   "inno": "研究从异构视频中学习共享潜在动作表示，将不同本体的运动信息映射到可共同建模的动作空间，为跨本体世界预测和策略学习提供统一接口。",
   "reuse": "wiki/entities/paper-tapestry.md"
  },
  "2610.40401": {
   "id": "2610.40401",
   "title": "Continuous-Action World Models: Learning Cross-Embodiment Spatiotemporal Dynamics Priors",
   "zh": "连续动作世界模型：学习跨本体时空动力学先验",
   "org": "同济大学。",
   "prob": "不同机器人具有不同动作维度和控制频率。直接学习离散动作序列，会使策略依赖特定本体与采样频率。",
   "inno": "提出 CoAWM，将动作轨迹表示为连续时间函数。首先通过动作重建和跨采样频率一致性学习共享潜在函数空间，再将该空间连接到世界模型。模型根据视觉、语言和机器人状态预测潜在动作函数，最后通过任意时间查询生成末端执行器动作，使动作表示与具体控制频率解耦。",
   "reuse": null,
   "short": "连续动作世界模型",
   "slug": "continuous-action-world-models-cross-embodiment",
   "abstract": "We introduce Continuous-Action World Models (CoAWM), a framework for learning cross-embodiment spatiotemporal dynamics priors that support robot action generation across embodiments and execution frequencies. At its core is a shared latent function space that captures the spatial and temporal structure of robot actions. We learn this space through two stages. First, function pretraining brings trajectories from different embodiments and sampling frequencies into a common representation, learning shared dynamics priors through action reconstruction and consistency across temporal sampling views. By accommodating heterogeneous robot trajectories, this stage broadens the range of data sources available for pretraining and facilitates scaling to larger and more diverse datasets. Second, we connect the pretrained function space to a world-model backbone, enabling observations, language instructions, and the current robot state to condition the prediction of a latent action function. A single decoder takes any query time and returns the corresponding end-effector action. By separating shared dynamics priors from task-conditioned action selection and temporal sampling, our approach provides a unified foundation for transferring robot experience across embodiments and control frequencies.",
   "comments": "",
   "authors": "Li, Liangyu, Liu, Qingwen",
   "links": []
  },
  "2610.39528": {
   "id": "2610.39528",
   "title": "PRIME W0: Physical Priors from World Models for Embodied Manipulation",
   "zh": "PRIME W0：利用世界模型物理先验进行具身操作",
   "org": "待核实（作者：Huabin Peng、Fagang Jin 等）。",
   "prob": "VLA 通常只使用当前观测生成动作，缺少长期历史信息和未来环境变化的有效条件化。",
   "inno": "提出分层条件 Flow Matching 框架，结合 VLM 逐层 KV 注入、自适应 MoE 动作专家和异步世界模型子目标图像预测。同时引入历史视觉特征桥接和协方差结构噪声，使 Flow Matching 的动作生成过程能够利用机器人运动维度之间的耦合关系。",
   "reuse": null,
   "short": "PRIME W0",
   "slug": "prime-w0-physical-priors-world-models",
   "abstract": "Vision-Language-Action (VLA) models excel at instruction following, yet most remain feed-forward: they condition actions only on the current observation and discard both long-horizon temporal context and the hierarchical structure of VLM representations. We present PRIME W0, a hierarchical conditional flow-matching framework that closes this gap through three synergistic designs: (1)per-layer KV injection from a pretrained VLM into an Adaptive MoE action expert; (2)dynamics-aware conditioning via asynchronous world-model subgoal-image prediction and a DINOv2 historical-frame semantic bridge; and (3)Action Dynamic Manifold (ADM) flow matching with covariance-structured noise that respects robot kinematic coupling. PRIME W0 is trained in four stages---VLM pre-training, DAT/NTP visuomotor pre-training, DiT flow-matching decoder assembly with WM injection, and end-to-end SFT---and achieves state-of-the-art results on LIBERO and real-robot benchmarks, with systematic ablations validating each architectural component.",
   "comments": "",
   "authors": "Peng, Huabin, Jin, Fagang, Wei, Yancong, Ma, Tengfei, Liu, Chuhang, Yu, Chao, Chen, Liqing",
   "links": []
  },
  "2610.36840": {
   "id": "2610.36840",
   "title": "Trace-WAM: Learning to Act via 3D Motion Prediction for World Action Models",
   "zh": "Trace-WAM：通过三维运动预测学习世界动作模型",
   "org": "香港理工大学、中国科学院自动化研究所等。",
   "prob": "机器人操作主要关心机器人和目标物体如何运动，而完整视频预测还需要重建背景与外观，增加了学习负担。",
   "inno": "使用未来三维运动预测作为辅助监督，联合学习时间点对应关系、图像平面位移和深度变化。提出三分支 Tri-MoT 架构，分别负责视频预测、运动预测和动作生成；推理时动作分支不需要输入预测轨迹，因此保留直接动作生成路径。",
   "reuse": "wiki/entities/paper-trace-wam.md"
  },
  "2610.37091": {
   "id": "2610.37091",
   "title": "GAWM: A Compact Latent World-Action Model for Multi-Task Robot Manipulation",
   "zh": "GAWM：面向多任务机器人操作的紧凑潜在世界动作模型",
   "org": "Genisom AI。",
   "prob": "大型 WAM 通常依赖数十亿参数的视觉语言或视频生成模型，训练成本和推理延迟较高。",
   "inno": "使用冻结的 DINOv3 Encoder，将每个相机视角压缩成少量 Token。轻量预测器根据任务条件预测两个未来时间尺度的视觉 Latent，再由动作 Decoder 生成动作块。模型参数量低于 100M，无需机器人或视频预训练，在 RoboTwin 2.0 上超过论文比较的部分十亿参数 WAM。",
   "reuse": null,
   "short": "GAWM",
   "slug": "gawm-compact-latent-world-action-model",
   "abstract": "World-action models (WAMs) incorporate the prediction of future observations into robot policy learning. However, most WAMs build on billion-parameter vision-language or video-generation backbones, which incur high training cost and inference latency. In this letter, we present GAWM (Geometry-Aware World Model), a compact multi-task WAM that predicts future observations in the latent space of a frozen DINOv3 encoder. GAWM compresses the features of each camera view into a small set of tokens. A lightweight predictor, conditioned on the task rather than on actions, forecasts these tokens at two future horizons. The forecast is supervised with the dense patch features of future frames, which requires it to encode the spatial layout of the future scene. An action decoder directly regresses a chunk of future actions from the current and predicted tokens. With fewer than 100M parameters and no robot or video pretraining, GAWM outperforms billion-parameter WAMs on RoboTwin 2.0 and performs comparably to billion-parameter VLAs on LIBERO. On a dual-arm robot, it achieves lower action-prediction error on held-out demonstrations than a fine-tuned billion-parameter VLA, with substantially lower inference latency. Ablation studies show that the latent world model substantially improves performance when training data or training time is limited. Code is available at this https URL.",
   "comments": "",
   "authors": "Liu, Yang, Liu, Xiaoli, Li, Liuzhao, Song, Yuduo, Liu, Muyuan, Gao, Xiang",
   "links": []
  },
  "2610.35850": {
   "id": "2610.35850",
   "title": "OSWAM: An Effective World Action Model with One-Step Action Generation",
   "zh": "OSWAM：具有单步动作生成能力的高效世界动作模型",
   "org": "香港理工大学。",
   "prob": "WAM 依赖多步去噪生成动作，推理延迟较高。直接减少去噪步数又容易损害动作质量。",
   "inno": "引入 Imagination Tokens，为扩散过程提供结构化动作先验，并通过特权信息蒸馏学习未来条件信息。模型只需一步扩散即可生成动作，相比直接策略式 WAM 约加速 3 倍；在 RTX 5090 上编译推理延迟最低约 40 ms。",
   "reuse": null,
   "short": "OSWAM",
   "slug": "oswam-one-step-action-generation",
   "abstract": "Direct-policy World Action Models (WAM) have demonstrated the benefits of decoupling action generation from explicit future visual prediction for efficient inference in autonomous driving and embodied control. However, they still require tens of denoising steps to achieve competitive action quality, incurring substantial latency and hindering real-time deployment. Simply reducing the number of denoising steps will collapse the nonlinear multi-step denoising trajectory into coarse updates, substantially degrading action quality. To reduce the denoising steps while enabling real-time action generation, we present an effective world action model with one-step action generation, namely OSWAM, for autonomous driving and embodied control. We argue that it is the absence of structured action priors in random noise initialization that forces direct-policy WAMs to recover coherent action sequences through iterative denoising. Hence, we construct imagination tokens to initialize the diffusion process, providing physical priors and action-conditioned representations. We further introduce privileged information distillation with future conditioning to enhance future-aware action generation. As a result, OSWAM can efficiently and effectively generate high quality actions in just one diffusion step. Our experiments demonstrate that OSWAM achieves comparable or even better results in both autonomous driving and embodied control than direct-policy WAM and Joint-WAM that require tens of steps. It obtains approximately $3\\times$ speedup over direct-policy WAM baselines and supports real-time compiled inference with latency as low as 40ms on an NVIDIA RTX 5090 GPU. The project page is available at this https URL",
   "comments": "",
   "authors": "Zhang, Guowen, Chen, Liyi, Hu, Xu, Wang, Pengfei, Chen, Jingfan, He, Chenhang, Zhang, Lei",
   "links": []
  },
  "2610.39278": {
   "id": "2610.39278",
   "title": "WorldFlow: Joint Video and 3D Motion Prediction via Large-Scale Pretraining",
   "zh": "WorldFlow：通过大规模预训练联合预测视频与三维运动",
   "org": "待核实（作者：Yuhao Lu、Jiongxu Chen、Lin Shao 等）。",
   "prob": "视频世界模型通常隐式表示三维动力学，而独立的三维运动预测模型又缺少视频生成提供的视觉监督。",
   "inno": "将未来视频和三维轨迹统一到 Mixture-of-Transformers 中，使用 Rectified Flow 联合生成。模型在约 144 万个物体级样本上训练，并提出 WorldFlowBench。在 SpatialCog 测试集上，平均位移误差达到 0.060 m，相对最强外部基线降低 **54.2%**。",
   "reuse": "wiki/entities/paper-worldflow.md"
  },
  "2610.39232": {
   "id": "2610.39232",
   "title": "Seeing and Imagining as One: Unified Video Generation and 4D Perception for Embodied World Modeling",
   "zh": "感知与想象统一：面向具身世界建模的视频生成与四维感知",
   "org": "RoboScience、南京大学、香港理工大学、新加坡国立大学等。",
   "prob": "世界模型通常将视频生成与深度、分割和运动感知分开训练，导致不同预测结果之间缺少一致性。",
   "inno": "提出 GenSight，在统一视频 DiT 中同时建模未来视频、深度、机器人本体分割和二维光流。训练使用共享时空表示和统一预测空间，推理支持多步未来生成及单步密集感知。",
   "reuse": "wiki/entities/paper-gensight.md"
  },
  "2610.37865": {
   "id": "2610.37865",
   "title": "Echo: Scaling Visual Reinforcement Learning with Off-policy Experience",
   "zh": "Echo：利用 Off-Policy 经验扩展视觉强化学习",
   "org": "哥伦比亚大学相关研究团队。",
   "prob": "视觉强化学习需要频繁渲染图像并运行视觉策略，计算成本高。传统 Teacher–Student 蒸馏虽然降低训练成本，但会引入模仿误差。",
   "inno": "提出单阶段 Off-Policy 视觉强化学习框架。通过 Inverse Distillation，让状态策略模仿持续更新的视觉策略，并在不渲染图像的环境中采集大量低成本经验。这些经验用于共享特权 Critic 的 Off-Policy 价值学习，再直接更新视觉策略。基于 FastTD3 实现，在两个灵巧手任务中，达到参考回报所需时间平均减少 **56%**。",
   "reuse": "wiki/entities/paper-echo-visual-rl-off-policy-experience.md"
  },
  "2610.39212": {
   "id": "2610.39212",
   "title": "Event2Reward: Learning Robotic Rewards from Progress Relations",
   "zh": "Event2Reward：通过任务进度关系学习机器人奖励",
   "org": "待核实（作者：Junnan Nie、Likui Zhang 等）。",
   "prob": "使用示范时间作为任务进度标签，默认任务持续向前推进。但实际机器人执行可能停滞、倒退或恢复，导致奖励模型错误地奖励失败行为。",
   "inno": "同时利用示范时间标签和执行轨迹中的相对进度关系，学习能够识别进度前进、停滞和倒退的模型。从约 10 万条真实机器人执行记录构建事件标注数据，将预测进度用于 Reward Shaping。在 LIBERO 的 40 个任务中，相比仅使用稀疏奖励的基线，平均成功率提高 18.16 个百分点。",
   "reuse": "wiki/entities/paper-event2reward.md"
  },
  "2610.38513": {
   "id": "2610.38513",
   "title": "Force-Driven Hook Disassembly via Constraint-Aware Object-Centric Reinforcement Learning",
   "zh": "通过约束感知的物体中心强化学习实现力驱动卡扣拆解",
   "org": "大阪大学、日本产业技术综合研究所。",
   "prob": "工业卡扣连接通常被外壳遮挡，机器人难以仅根据视觉判断正确的拆解方向。",
   "inno": "使用物体中心状态表示和粒子化信念状态估计隐藏约束，再通过强化学习选择拆解动作。方法能够处理单轴和多阶段拆解，并在真实废旧空调部件拆解任务中验证。",
   "reuse": null,
   "short": "钩子拆卸力驱动 RL",
   "slug": "force-driven-hook-disassembly-object-centric-rl",
   "abstract": "The growing volume of end-of-life products has intensified the need for automated disassembly, yet hook-type connections remain challengin: their orientations vary, they deform under load, and their geometry is hidden, so vision alone cannot identify a feasible non-destructive disassembly direction. This work proposes a constraint-aware reinforcement learning framework that infers feasible extraction directions online from contact forces. The policy is trained in a robot-agnostic, object-centric simulation on a simplified peg-in-hole pair and deployed zero-shot on a physical robot, where it transfers to real connections with unseen, multi-stage constraint structures. Because instantaneous contact forces are noisy and locally ambiguous, the policy is coupled with a particle filter belief that provides the policy with temporally integrated directional information and reduces redundant exploratory motion. The approach is validated on four real-world tasks, from peg-in-hole pairs to the inner propeller and front cover of an end-of-life air-conditioning unit. The full method achieves success rates of 100%, 100%, 90%, and 75% with the highest path efficiency on every task, and raises the success rate on the air-conditioning cover from 45% to 75% relative to a policy trained without the belief. These results indicate that force-driven reinforcement learning can enable disassembly without product-specific models or demonstrations.",
   "comments": "",
   "authors": "Elhadad, Nadeen, Kiyokawa, Takuya, Harada, Kensuke",
   "links": []
  },
  "2610.35659": {
   "id": "2610.35659",
   "title": "Environment Edits as a Corrective Signal for Reward Learning",
   "zh": "将环境编辑作为奖励学习的纠正信号",
   "org": "待核实（作者：Isaac Ngui、Courtney McBeth、Nancy M. Amato）。",
   "prob": "人类通过纠正机器人动作表达偏好时，一次动作变化可能同时改变多个状态特征，难以确定用户真正关注的因素。",
   "inno": "提出 FERRET，让人类在机器人运动保持不变时直接编辑环境中的物体。通过结合机器人动作纠正和环境编辑轨迹，学习物体属性、空间关系等偏好信息，减少奖励学习中的特征歧义。",
   "reuse": null,
   "short": "环境编辑奖励学习",
   "slug": "environment-edits-corrective-reward-learning",
   "abstract": "Robot corrections allow people to adjust a robot's behavior, but a single correction changes many features. Also, preferences about object attributes and relationships between objects cannot be expressed by moving the robot. We propose environment edits as a complementary form of feedback. The robot's motion is held constant as the human edits the scene, so the only explanation for the edit is the moved object. We introduce Feature Learning from Environmental and Robot-Relational Edit Traces (FERRET), which learns preferences from both robot and environment corrections. We show that environment edits are able to disambiguate features and learn preferences that robot corrections alone cannot.",
   "comments": "",
   "authors": "Ngui, Isaac, McBeth, Courtney, Amato, Nancy M.",
   "links": []
  },
  "2610.36694": {
   "id": "2610.36694",
   "title": "TOPS: Reshaping Flow Source and Transport for Multi-Task Robot Manipulation",
   "zh": "TOPS：通过重构 Flow 源分布与传输路径实现多任务机器人操作",
   "org": "南洋理工大学 MARS Lab。",
   "prob": "多任务 Flow Matching Policy 通常对所有任务使用相同的高斯噪声分布和线性插值路径，但不同操作任务的动作分布差异很大。",
   "inno": "提出 Task-Oriented Priors and Schedules（TOPS）。首先从专家动作反向映射构建任务专用的多模态源分布，再根据源分布与目标动作分布的协方差差异，设计任务专用的 Flow 插值时间调度。在 LIBERO-10 上，单步推理成功率达到 **69.67%**，超过标准 Flow Matching 十步推理的 **53.44%**，推理时间约为基线的六分之一。",
   "reuse": null,
   "short": "TOPS",
   "slug": "tops-flow-source-transport-multitask",
   "abstract": "Flow-based policies for language-conditioned multi-task manipulation commonly share a single $\\textit{Gaussian}$ source prior and a common linear transport path across tasks. However, reliably executing multiple tasks with distinct motion requirements remains challenging for a shared policy. We introduce $\\textbf{TOPS}$ ($\\textbf{T}$ask-$\\textbf{O}$riented $\\textbf{P}$riors and $\\textbf{S}$chedules), which adapts the flow source prior and interpolation schedule to each task without modifying the network architecture. TOPS first constructs multimodal task sources from reverse-mapped expert action sequences, allowing language instructions to select where generation begins. Guided by source-target covariance mismatch, it then slows the transition toward expert actions early in denoising and accelerates it later along directions requiring expansion. Under a $\\textit{Gaussian}$ endpoint model, our analysis shows that this schedule adjustment can lower an upper bound on velocity-prediction risk. Experiments on LIBERO, LIBERO-Plus, and CALVIN demonstrate improved performance across in-distribution, perturbation and cross-environment settings, alongside better data efficiency. Using one integration step, TOPS raises LIBERO-10 success from $53.44\\%$ to $69.67\\%$ compared with ten-step Standard Flow Matching (FM), while reducing inference time to approximately one-sixth of the baseline. TOPS also yields consistent gains when transferred to the large-scale flow-based $\\pi_{0.5}$ policy across all four evaluated settings. Real-world experiments further demonstrate improved task success, including an increase from $20\\%$ with standard FM to $80\\%$ with TOPS on the upper-drawer opening task.",
   "comments": "",
   "authors": "An, Tuo, Zhou, Chuhao, Dong, Jiahua, Li, Jingliang, Bai, Jiaqi, Jia, Jindou, Yang, Jianfei",
   "links": []
  },
  "2610.40271": {
   "id": "2610.40271",
   "title": "Gaussian Distillation of Flow Policies",
   "zh": "Flow 策略的高斯蒸馏",
   "org": "日内瓦大学、蒂宾根大学等。",
   "prob": "Flow Policy 需要多次网络前向传播才能生成动作，推理成本较高，也难以直接计算动作概率密度。",
   "inno": "提出 Gaussian Action Readout（GAR），用预测高斯均值和协方差的输出层替换 Flow Policy 最后一层，冻结其余网络。只使用约 0.02% 的可用训练状态拟合输出层，即可实现单次网络前向采样，并获得闭式动作概率密度。在 RoboCasa365 的 50 个任务中，蒸馏策略保持接近教师的成功率，推理速度最高提升约 2 倍。",
   "reuse": null,
   "short": "流策略高斯蒸馏",
   "slug": "gaussian-distillation-flow-policies",
   "abstract": "Flow policies are among the strongest robot controllers and are well suited to modelling multimodal action distributions. Our analysis shows that the continuous-action distributions of the studied flow policies are well approximated by Gaussians. We propose the Gaussian Action Readout (GAR), which replaces a flow policy's final layer with a Gaussian readout trained on just 0.02% of eligible states, while keeping the rest of the network frozen. It samples with one action-network pass, gives closed-form likelihoods, and achieves near-teacher task success at up to twice the inference speed.",
   "comments": "",
   "authors": "Saied, Youssef, Sheebaelhamd, Ziyad, Vernade, Claire, Fleuret, François",
   "links": []
  },
  "2610.40356": {
   "id": "2610.40356",
   "title": "JEDI: Joint-Embedding Demonstration Compression for In-Context Imitation Learning",
   "zh": "JEDI：面向上下文模仿学习的联合嵌入示范压缩",
   "org": "纽约大学、纽约大学阿布扎比分校。",
   "prob": "上下文模仿学习需要将示范轨迹作为条件输入，但完整示范包含大量图像和动作 Token，限制了可同时使用的示范数量。",
   "inno": "将每条示范压缩为少量 Latent Slots，并使用冻结的 V-JEPA 预测查询轨迹下一阶段的关键帧表示。同时引入任务和执行过程的反事实示范，使压缩表示保留抓取位置、操作顺序等信息，而不只是任务语义。还通过预测误差选择更有用的示范，并支持人手视频作为机器人操作条件。",
   "reuse": null,
   "short": "JEDI",
   "slug": "jedi-demonstration-compression-in-context-il",
   "abstract": "In-context imitation learning (ICIL) adapts robot policies through test-time demonstrations, but long multimodal trajectories constrain how many examples fit in context. We ask what a compressed demonstration should preserve. We propose JEDI, which compresses demonstrations into latent slots trained to predict the query episode's next-stage keyframe in a frozen video-JEPA space. Task and process counterfactuals help the slots preserve execution details, while the predictor selects demonstrations from a bank. Post-training with keypoint tokens enables human-hand conditioning on robots. Experiments on LIBERO and real robots show that JEDI supports many-shot context; LIBERO objective controls show improved success and demonstration following over action-only compression. These findings support query-future prediction as a training objective for preserving execution-relevant information in compressed demonstrations.",
   "comments": "",
   "authors": "Liang, Jiazhao, He, Zhewen, Yuan, Shuaihang, Huang, Hao, Fang, Yi",
   "links": []
  },
  "2610.40075": {
   "id": "2610.40075",
   "title": "Learning Safe Flow-Matching Robot Policies from Binary Feasibility Oracles",
   "zh": "通过二值可行性判定学习安全 Flow Matching 机器人策略",
   "org": "上海交通大学、同济大学。",
   "prob": "Flow Matching 生成的机器人动作可能违反关节限制或碰撞约束，而真实系统中的可行域不一定具有可微解析表达式。",
   "inno": "提出 Oracle-Guided Differentiable Safeguard。前向传播时，通过二值可行性查询修正不安全动作；反向传播时，从局部边界信息构造约束相关梯度，使安全修正能够参与策略训练。在障碍物感知机器人操作实验中，实现 100% 的约束满足率。",
   "reuse": null,
   "short": "二值可行性安全流匹配",
   "slug": "safe-flow-matching-binary-feasibility-oracles",
   "abstract": "Robot foundation models increasingly rely on flow-based action generation for general-purpose manipulation, yet the generated actions are not guaranteed to satisfy physical constraints such as joint limits and collision avoidance. This problem is especially challenging when the feasible set cannot be described analytically and is accessible only through black-box feasibility oracles. We study this practical oracle-only setting and propose an oracle-guided differentiable safeguard for safe flow-matching robot policies. The safeguard uses binary feasibility queries to correct unsafe updates in the forward pass, while extracting local boundary information to provide constraint-aware learning signals in the backward pass. Combined with discrete-time flow matching, the same safeguard is used during training and inference, enabling forward--backward aligned policy learning without explicit or differentiable constraint models. Experiments on obstacle-aware robotic manipulation demonstrate 100% constraint satisfaction while preserving task performance.",
   "comments": "",
   "authors": "Ma, Jianming, Jin, Rongjun, Chen, Yizhi, Gao, Yue",
   "links": []
  },
  "2610.40430": {
   "id": "2610.40430",
   "title": "Improving Fine-Grained Manipulation through Demonstration Expansion and Execution Design",
   "zh": "通过示范扩展与执行设计改善精细机器人操作",
   "org": "待核实（作者：Jingxu Gu、Dongze Lian）。",
   "prob": "精细操作不仅依赖策略预测，还受到示范覆盖范围、相机变化、动作执行周期和夹爪控制映射的影响。",
   "inno": "提出 DEED，对同一物理轨迹生成不同语言指令、相机位置和光照条件下的训练示范。部署时结合逆运动学、任务专用夹爪映射和闭环动作块重规划，在 MetaFine 五项任务评测中取得 85.55% 的总体评分。",
   "reuse": null,
   "short": "示范扩展与执行设计",
   "slug": "demonstration-expansion-execution-design-fine-grained",
   "abstract": "Fine-grained manipulation requires a robot policy to follow instructions, remain robust to visual variations, and translate predicted actions into precise robot control. We present Demonstration Expansion and Execution Design (DEED), a framework for adapting a pretrained vision-language-action policy to the five MetaFine tasks through structured demonstration expansion and execution design. DEED generates demonstrations for different instructions from the same initial scene and renders each physical trajectory under different camera positions and lighting conditions while preserving its original states and action targets. A hierarchical sampling strategy balances instructions, trajectories, and visual conditions during supervised adaptation. At deployment, DEED converts predicted action chunks into robot commands using inverse kinematics, gripper mappings selected for each task, and closed-loop replanning. DEED achieves an official Overall score of 85.55%. Fixed-policy evaluations show that performance varies with execution horizon, denoising budget, and gripper mapping.",
   "comments": "",
   "authors": "Gu, Jingxu, Lian, Dongze",
   "links": []
  },
  "2610.35458": {
   "id": "2610.35458",
   "title": "RoKON: Efficient and Adaptive Data-Enabled Predictive Control for Robotic Manipulation",
   "zh": "RoKON：面向机器人操作的高效自适应数据驱动预测控制",
   "org": "待核实（作者：Akshun Sharma、Zhongqi Wei 等）。",
   "prob": "机器人操作中的摩擦等动力学参数可能随环境变化。依赖固定动力学模型的预测控制难以及时适应这些变化。",
   "inno": "使用高斯核构建非线性数据预测器，并提出阻尼 Gauss–Newton 优化方法提高实时求解效率。通过混合数据缓存同时保留有用的离线经验和新采集的在线数据，使控制器能够适应变化的接触动力学。",
   "reuse": null,
   "short": "RoKON",
   "slug": "rokon-data-enabled-predictive-control",
   "abstract": "Data-enabled predictive control has recently emerged as a promising control paradigm, enabling a transparent optimal-control formulation without the need to learn an explicit dynamics model. This paper presents a number of extensions paving the path towards its use in real-time robotics tasks. Following recent work, we utilize a nonlinear predictor based on Gaussian kernels to represent the relationship between input and output data. We propose a damped Gauss-Newton method that enables real-time control by efficiently exploiting the least-squares structure of the control problem. In addition, we introduce a hybrid buffer management strategy that incorporates informative online data while retaining relevant offline data, allowing the predictor to account for changes in the dynamics. We demonstrate the tracking performance and computational suitability of the proposed method in MuJoCo using a Franka Research 3 arm pushing a block across surface, without prior knowledge of its changing friction values.",
   "comments": "",
   "authors": "Sharma, Akshun, Wei, Zhongqi, Ou, Buqing, Dümbgen, Frederike",
   "links": []
  },
  "2610.36250": {
   "id": "2610.36250",
   "title": "cuArms: GPU-Accelerated Conflict-Based Planning for Multi-Arm Manipulation",
   "zh": "cuArms：面向多机械臂操作的 GPU 加速冲突搜索规划",
   "org": "卡内基梅隆大学、悉尼大学、NVIDIA 相关研究团队。",
   "prob": "多机械臂在共享工作空间中执行任务时，需要同时处理任务分配、运动规划、碰撞冲突和执行顺序。联合搜索的计算复杂度很高。",
   "inno": "提出与 GPU 并行计算协同设计的冲突搜索规划器。使用可复用的多模态路径图分别表示单臂运动，再通过延迟冲突搜索解决机械臂之间的碰撞和死锁。结合高层任务调度，实现任务分配、时间安排与运动规划的联合优化。",
   "reuse": "wiki/entities/paper-cuarms.md"
  },
  "2610.36961": {
   "id": "2610.36961",
   "title": "Keeping Contact: Task-Adaptive Friction Margins for Bimanual Non-Prehensile Manipulation",
   "zh": "保持接触：面向双臂非抓取操作的任务自适应摩擦裕度",
   "org": "南洋理工大学。",
   "prob": "双臂通过推、托、倾斜等非抓取方式操作物体时，即使接触力满足摩擦锥约束，也可能过于接近滑移边界。",
   "inno": "使用有符号摩擦距离量化接触力的安全裕度，并根据物体倾斜过程中的力矩平衡需求动态调整裕度。通过二阶锥规划求解接触力分配，使双臂操作能够适应物体属性变化和模型误差。",
   "reuse": "wiki/entities/paper-keeping-contact.md"
  },
  "2610.40210": {
   "id": "2610.40210",
   "title": "AcrossDEX 1.0: From Human Motion Pretraining to Bimanual Dexterous Manipulation across Hand Embodiments",
   "zh": "AcrossDEX 1.0：从人类运动预训练到跨灵巧手本体的双臂操作",
   "org": "待核实（作者：Xiaoyu Yang、Nan Wu）。",
   "prob": "不同灵巧手的手指自由度和动作空间不同，难以直接复用从人类视频学习的双手协调能力。",
   "inno": "将人类双手运动预训练、仿真专家学习和机器人闭环执行连接起来，使用共享手腕表示、独立手指特征和有效性掩码，同时适配 Inspire Hand 和 Allegro Hand。研究还比较直接回归、Diffusion 和 Flow Matching，并分析动作块连续性对双臂协同执行的影响。",
   "reuse": null,
   "short": "AcrossDEX 1.0",
   "slug": "acrossdex-bimanual-dexterous-human-motion-pretraining",
   "abstract": "Bimanual dexterous manipulation requires continuous coordination of wrist motion, finger contact, and object motion. Human videos provide rich manipulation experience. Using this experience for robot control requires aligning human poses with robot commands and accounting for embodiment-specific kinematics and contact dynamics. We present AcrossDEX 1.0, a learning system that connects human motion pretraining, simulation-based expert learning, robot demonstration collection, and visual closed-loop execution. Pretraining on HOT3D Aria predicts one second of bimanual motion from five images and thirty annotated hand states over the preceding second. Robot adaptation preserves the visual encoder, temporal backbone, and wrist parameters. Shared wrist representations, separate finger fields, and validity masks support both six-channel Inspire hands and sixteen-joint Allegro hands. The robot policy predicts sixty control targets from frontal images, command history, and task conditions. It executes a prefix and replans from updated observations. We compare direct regression, diffusion, and flow matching on lid opening and return and closed-lid object reorientation. Direct regression achieves success rates of 40.6\\% for opening and return, 100.0\\% for Inspire reorientation, and 93.3\\% for Allegro reorientation. Flow matching achieves 90.0\\% on Allegro reorientation. Paired comparisons further show that constraining replanning with the previous plan improves the continuous execution of generative policies. These results demonstrate a common adaptation pipeline for two hand embodiments and highlight the roles of contact coordination and action-chunk continuity.",
   "comments": "",
   "authors": "Yang, Xiaoyu, Wu, Nan",
   "links": []
  },
  "2610.37533": {
   "id": "2610.37533",
   "title": "Which Hand Was Late? Diagnosis for Robot Failure Recovery in Bimanual Tool Use",
   "zh": "哪只手延迟了？双臂工具使用中的失败诊断与恢复",
   "org": "待核实（作者：Baptiste Bonin、Caro Strickland、Audrey Durand）。",
   "prob": "双臂工具操作的失败可能来自左右臂不同步、工具滑移或接触力不足。单一相机或单臂传感器未必能够区分这些原因。",
   "inno": "系统分析不同传感器配置能够识别哪些失败原因，并建立基于诊断结果的恢复策略。实验发现，部分失败原因在被动观测下无法唯一确定，因此恢复策略需要根据可观测性决定直接修正还是进一步获取信息。",
   "reuse": null,
   "short": "Which Hand Was Late",
   "slug": "which-hand-was-late-bimanual-failure-diagnosis",
   "abstract": "Failure diagnosers for robot manipulation are increasingly vision--language models (VLMs), trained and evaluated on single-arm failures produced by perturbing a successful demonstration. Bimanual tool use adds failures of other kinds: a tool that slips out of its socket while still held, two correct gestures made at different times, and attempts that fail for two reasons at once. On a simulated bimanual tool-use task whose failure causes are read from the recorded trace, we measure what each sensing channel lets a diagnoser tell. The tool's slip is read on the tool hand; which hand was late is read on both arms, while a sensor on one hand reads its own hand's delay and at most part of the other's; and whether a late attempt was also pushed too weakly is not identifiable from passive sensing when the tool hand is late. Within two attempts, a repair policy that escalates when the cause is not identifiable repairs as many failures as one told the true cause. A VLM's internal state carries the late hand almost perfectly, along a direction no closer to the one its answer reads than a random one; whether the answer reports it depends on the wording of the question.",
   "comments": "",
   "authors": "Bonin, Baptiste, Strickland, Caro, Durand, Audrey",
   "links": []
  },
  "2610.37652": {
   "id": "2610.37652",
   "title": "Pushing Dexterity and Safety for In-Hand Object Pose Reaching That Works Remarkably Well",
   "zh": "兼顾灵巧性与安全性的手内物体位姿控制",
   "org": "加州大学伯克利分校、加州大学戴维斯分校。",
   "prob": "多指灵巧手不仅需要旋转物体，还需要在不同手腕姿态下，将物体移动到复杂的位置和朝向，同时避免掉落。",
   "inno": "使用仿真强化学习训练 LEAP Hand 的手内位姿控制策略，并通过系统辨识实现零样本真机迁移。在冻结任务策略之上增加可学习残差安全过滤器，降低掉落风险。真实机器人测试覆盖 4,800 个未见目标，状态策略在指定误差阈值下达到 95.2% 的目标到达率，安全过滤器使物体掉落次数减少 **75%**。",
   "reuse": "wiki/entities/paper-dexsafe.md"
  },
  "2610.39582": {
   "id": "2610.39582",
   "title": "Kinematic-Tactile Aligned Dexterous Retargeting with Contact Wrench Space Alignment",
   "zh": "通过接触力旋量空间对齐实现运动学与触觉联合灵巧重定向",
   "org": "待核实（作者：Xinyuan He、Peter B. Shull 等）。",
   "prob": "传统人手到机器人灵巧手的重定向主要匹配关节姿态，无法保留人类操作中的接触力信息。",
   "inno": "提出三阶段运动学—触觉联合重定向框架。先对齐手部运动和接触几何，再构建人手接触力旋量支持空间，最后优化机器人指尖接触力并转换为关节位置修正。真实机器人倾倒实验中的轨迹重放成功率达到 **96.7%**。",
   "reuse": null,
   "short": "运动学-触觉对齐灵巧重定向",
   "slug": "kinematic-tactile-aligned-dexterous-retargeting",
   "abstract": "Human demonstrations provide a natural and scalable source of data for learning dexterous manipulation policies. However, existing retargeting methods primarily transfer human kinematics, while the tactile information underlying physical interaction is rarely preserved across embodiments. To address this limitation, we present a hierarchical three-stage framework for kinematic-tactile aligned dexterous retargeting. We use object-centric contact wrench space as a physical representation for tactile alignment between human and robot hands. The framework first aligns hand kinematics and contact geometry, and then constructs a human wrench support profile from tactile forces and contact geometry. Robot fingertip force targets are subsequently optimized to cover the human wrench support and converted into joint position corrections, yielding directly replayable robot hand trajectories. We evaluate the framework in simulation and real robot experiments across three pouring conditions. In the real robot experiments, our method achieves a 96.7\\% replay success rate and higher wrench transfer fidelity than representative retargeting baselines. The framework enables physically grounded transfer of kinematic and tactile information from human demonstrations to robot hands, providing a path toward scalable learning of contact-rich dexterous manipulation.",
   "comments": "",
   "authors": "He, Xinyuan, Xu, Ziheng, Chen, Yueyuan, Ye, Yufan, Tan, Yuanshuo, Pan, Huiming, Meng, Jianjun, Jiang, Shuo et al.",
   "links": []
  },
  "2610.40349": {
   "id": "2610.40349",
   "title": "TouchBridge: Learning Contact-Aware Dexterous Manipulation from Human Interaction",
   "zh": "TouchBridge：从人类交互中学习接触感知灵巧操作",
   "org": "待核实（作者联合研究团队）。",
   "prob": "人类操作示范能够提供动作信息，但机器人还需要学习接触发生的位置和方式，才能准确执行灵巧操作。",
   "inno": "研究将人类交互中的接触信息与机器人动作学习结合，使策略能够利用接触相关表示进行灵巧控制。",
   "reuse": null,
   "short": "TouchBridge",
   "slug": "touchbridge-contact-aware-dexterous-human-interaction",
   "abstract": "Human demonstrations offer an efficient and scalable source of data for dexterous robot manipulation, but differences in sensing and hand morphology complicate the transfer of contact-rich interactions. We present TouchBridge, a hardware--software co-designed framework for human-to-robot data transformation. A wearable human demonstration interface and a corresponding robot interface establish cross-embodiment correspondence in visual and fingertip tactile sensing. We introduce Tac-Retarget, a tactile-conditioned retargeting method that uses measured contact to align fingertip contact patches and preserve their spatial relations within and across hands. Unlike object-aware or simulation-based transfer pipelines, Tac-Retarget requires no object mesh, object-pose estimate, or physics simulation for retargeting, enabling direct and scalable conversion of human demonstrations into robot-domain training data. Combined with visual synthesis and camera-anchored action conversion, TouchBridge produces robot-domain demonstrations for fine-tuning a pretrained vision-language-action (VLA) model. Across four real-world contact-rich tasks, policies fine-tuned solely on TouchBridge-converted human demonstrations achieve a 72.5% average success rate, compared with 77.5% using an equal number of teleoperated robot demonstrations, while human demonstration collection is 7.0x faster. Under the same policy and training setup, Tac-Retarget outperforms DexPilot and UniDex by 25.0 and 42.5 percentage points, respectively.",
   "comments": "",
   "authors": "Zhang, Haozhe, Zhang, Gu, Zou, Zirui, Chen, Yuzheng, Wu, Jinlong, Hu, Zixuan, Zhang, Jianke, Hu, Yucheng et al.",
   "links": []
  },
  "2610.39044": {
   "id": "2610.39044",
   "title": "Catch It All! Learning Dexterous Catching with Reinforcement Learning",
   "zh": "全部接住：通过强化学习实现灵巧接物",
   "org": "待核实（作者联合研究团队）。",
   "prob": "灵巧手接住运动中的物体，需要在很短时间内预测物体运动、调整手部姿态并建立稳定接触。",
   "inno": "使用强化学习训练动态接物策略，研究多指协调和动态接触控制，使灵巧手能够处理不同物体运动条件。",
   "reuse": "wiki/entities/paper-catch-it-all.md"
  },
  "2610.35058": {
   "id": "2610.35058",
   "title": "3D-Force Estimation for Dual-arm Manipulation of Deformable Linear Objects via Factor Graphs",
   "zh": "基于因子图的双臂可变形线状物体三维受力估计",
   "org": "热那亚大学、喷气推进实验室。",
   "prob": "电缆、绳索等可变形线状物体没有直接驱动关节，难以通过机器人执行器信息推断物体内部或外部受到的力。",
   "inno": "使用因子图建立物体形状、力学约束和传感器测量之间的概率关系。在重建线状物体三维形状的同时，估计外力的大小、方向和作用位置，减少对专用应变传感器的依赖。",
   "reuse": null,
   "short": "可变形线性物体 3D 力估计",
   "slug": "dlo-3d-force-estimation-factor-graphs",
   "abstract": "Estimating forces acting on Deformable Linear Objects (DLOs) is challenging due to their underactuated nature. However, reliable contact estimation is crucial to precisely control them and enable safe manipulation tasks. Differently from the various approaches presented in the literature, this work adopts a probabilistic formulation where the DLO state and sensor data are related through conditional distributions. In particular, factor graphs are used to determine the magnitude, direction, and location of a 3D force acting on a DLO, while simultaneously reconstructing its shape. Since DLOs are not actuated and are typically thin, like cables and wires, strain gauges and actuation readings are usually impractical. Therefore, to overcome this issue, this work is tested using one type of sensing information only, thus developing a generic technique suited for a wider variety of tasks. Realistic tests are performed by introducing noise into the measurements while the DLO follows a prescribed path in the presence of external forces that are accurately estimated.",
   "comments": "",
   "authors": "Delucchi, Annika, Di Paola, Vincenzo, Quadrelli, Bruno Marco, Zoppi, Matteo",
   "links": []
  },
  "2610.35103": {
   "id": "2610.35103",
   "title": "The Language of Deformation: Semantic Failure Analysis as a Prerequisite for Dexterous Manipulation of Deformable Objects",
   "zh": "形变的语言：可变形物体灵巧操作中的语义失败分析",
   "org": "剑桥大学、比萨大学、RAI Institute 等。",
   "prob": "可变形物体操作中的失败不仅包括滑移，还可能出现屈曲、塑性变形、局部压力异常和撕裂。传统刚体操作中的失败判断标准难以直接适用。",
   "inno": "提出针对可变形物体的语义失败分析框架，将不同物理失败现象组织为可识别的事件类型，并通过触觉实验研究其可检测性。",
   "reuse": null,
   "short": "形变语义失败分析",
   "slug": "language-of-deformation-semantic-failure-analysis",
   "abstract": "Dexterous manipulation of rigid objects has reached a level of maturity where grasp stability, control, and planning strategies are successfully applied in industry and logistics. In contrast, manipulating deformable objects (cables, plastic pieces, cloth) remains far more challenging, as rigid-object methods do not account for their additional complexity. This paper identifies semantic failure analysis as a key step toward mature deformable-object manipulation and supports this claim with proof-of-concept experiments. Unlike rigid-object manipulation, where goals are well-defined in terms of criteria such as force balance and slippage, deformable-object manipulation involves a broader variety of failure modes arising from differences in materials, geometry, and behaviour. Typical failures include buckling, plastic deformation, pressure variations, and tearing. These diverse and often hard-to-detect events require distinct corrective strategies, making semantic failure analysis necessary. We define its key components and challenges and present experimental results supporting its potential as a research direction.",
   "comments": "",
   "authors": "Cuiral-Zueco, Ignacio, Hardman, David, Pierallini, Michele, Shahriari, Erfan",
   "links": []
  },
  "2610.40559": {
   "id": "2610.40559",
   "title": "A Tutorial on Shape Servoing of Deformable Objects-Part I: Fundamentals",
   "zh": "可变形物体形状伺服教程（一）：基础原理",
   "org": "萨拉戈萨大学、博洛尼亚大学、Clermont Auvergne INP。",
   "prob": "机器人控制柔性物体达到目标形状时，需要知道机器人运动如何影响物体形变，但该关系依赖材料、几何和抓取边界条件，难以准确建模。",
   "inno": "系统介绍基于形变雅可比矩阵的形状伺服方法，包括形状表示、离线估计、在线 Broyden 更新、闭环控制律及局部稳定性分析。使用双臂操作弹性杆的仿真实例解释完整控制流程。",
   "reuse": null,
   "short": "形状伺服教程 Part I",
   "slug": "shape-servoing-deformable-objects-tutorial-part1",
   "abstract": "In robotics, shape control (or shape servoing) refers to the automatic manipulation of a deformable object to achieve a desired shape. It involves a feedback loop that exploits the relation between robot motion and changes in the measured shape. This relation, known as the interaction Jacobian or, in this context, the deformation Jacobian, depends on the object's mechanical properties, geometry, and boundary conditions (e.g., where the object is grasped). Because these factors are often difficult to model accurately, the deformation Jacobian can be estimated locally from observed shape changes produced by commanded robot motions, without requiring a fine-tuned mechanical model. This article is the first of a two-part tutorial on data-driven Jacobian-based shape control of deformable objects. It introduces the central elements of the approach: the formulation of the shape-control problem, shape representation, estimation and online update of the deformation Jacobian, and a standard control law from the literature. It then assembles these elements into a complete shape-servoing loop, derives a local stability condition for this loop, and illustrates why some target shapes cannot be reached. Part II covers more advanced topics. Throughout both parts, a simulated planar example in which two robot arms manipulate an elastic rod observed by a camera illustrates the concepts.",
   "comments": "",
   "authors": "Cuiral-Zueco, Ignacio, Caporali, Alessio, Aranda, Miguel, Palli, Gianluca, López-Nicolás, Gonzalo, Mezouar, Youcef",
   "links": []
  },
  "2610.35075": {
   "id": "2610.35075",
   "title": "Zero-HLM: Generalizable Humanoid Loco-Manipulation without Real Robot Data",
   "zh": "Zero-HLM：无需真实机器人训练数据的可泛化人形移动操作",
   "org": "待核实（作者：Dianyi Yang、Baoxiong Jia、Siyuan Huang 等）。",
   "prob": "人形移动操作需要大量全身运动与物体交互数据，但真实机器人示范采集成本高，仿真又难以同时保持视觉和物理一致性。",
   "inno": "提出 Real-to-Sim-to-Real 数据生成框架。通过手机拍摄重建视觉场景，再构建可交互的物理数字孪生环境；结合运动与环境协同适应机制，将少量仿真源示范扩展为大量可执行的全身操作轨迹。从 45 条源示范生成 4,800 条物理验证示范。仅使用仿真数据训练的策略能够零样本部署到 Unitree G1，在三个长程任务中完成真实机器人操作。",
   "reuse": "wiki/entities/paper-zero-hlm.md"
  },
  "2610.38027": {
   "id": "2610.38027",
   "title": "Metatron: A Promptable Whole-Body Execution Layer for Humanoids",
   "zh": "Metatron：面向人形机器人的可提示式全身动作执行层",
   "org": "NVIDIA、Dollhouse Robotics。",
   "prob": "人形机器人需要在多种动作任务之间切换，而单独训练的运动跟踪策略通常缺少统一的动作条件接口。",
   "inno": "提出可通过运动提示控制的全身执行层，将密集运动跟踪、动作条件接口和生成式策略结合。训练中使用 Masked On-Policy Distillation 整合不同运动专家，并加入恢复数据，使执行层能够支持不同动作任务及状态恢复。",
   "reuse": "wiki/entities/paper-metatron-promptable-whole-body-execution.md"
  },
  "2610.35862": {
   "id": "2610.35862",
   "title": "FLORE: Learning Generalizable Humanoid Whole-Body Manipulation",
   "zh": "FLORE：学习可泛化的人形全身操作",
   "org": "待核实（作者联合研究团队）。",
   "prob": "人形机器人执行操作任务时，需要同时协调身体姿态、双臂动作和接触约束。单独学习上肢动作难以覆盖全身运动带来的动力学影响。",
   "inno": "研究统一的全身操作学习框架，使人形机器人能够在不同任务条件下协调身体和手臂运动，提高全身操作策略的泛化能力。",
   "reuse": "wiki/entities/paper-flore.md"
  },
  "2610.40016": {
   "id": "2610.40016",
   "title": "LOGIV: Graph-Verified Execution with Causal Local Repair for Long-Horizon Robot Tasks",
   "zh": "LOGIV：通过图验证执行与因果局部修复完成长程机器人任务",
   "org": "待核实（作者：Enqiao Lu、Yuxiang Xie、Ivor Tsang 等）。",
   "prob": "长程操作中的某个步骤即使成功完成，其建立的环境条件也可能被后续扰动破坏，导致剩余计划无法继续执行。",
   "inno": "使用显式任务图表示步骤之间的条件依赖和资源顺序，执行过程中持续验证物理条件。一旦发生失败，通过依赖关系追踪确定需要恢复的局部状态，再重新验证剩余计划。框架能够直接部署在已有 VLA 和 WAM 之上，无需增加可训练模块。真实机器人成功次数从 12/30 提升至 17/30。",
   "reuse": "wiki/entities/paper-logiv.md"
  },
  "2610.36533": {
   "id": "2610.36533",
   "title": "RoboAuto: Autonomous Robot Program Search through Multi-Level Physical Understanding",
   "zh": "RoboAuto：通过多层次物理理解实现自主机器人程序搜索",
   "org": "待核实（作者：Buzhao Liu、Chongjie Zhang 等）。",
   "prob": "LLM 能够生成机器人操作程序，但仅依靠执行成功或失败反馈，难以判断物体可操作性、空间约束和物理交互失败的具体原因。",
   "inno": "建立物体、场景和交互三个层次的物理知识表示。通过分层知识库保存操作技能、物体可供性和动作效果，再使用场景关联与自主实验验证程序中的物理假设。在 LIBERO-PRO 短程任务中达到 89.1% 成功率，并在长程任务中超过论文比较的程序搜索基线。",
   "reuse": "wiki/entities/paper-roboauto.md"
  },
  "2610.39319": {
   "id": "2610.39319",
   "title": "ProLoop: Progress-Aware Closed-Loop Framework via Structured State Relations for Long-Horizon Robotic Task Planning",
   "zh": "ProLoop：基于结构化状态关系的进度感知长程机器人闭环规划",
   "org": "待核实（作者：Runxin Zou、Shiqiang Zhu 等）。",
   "prob": "长程操作通常分解为多个子目标，但 LLM 规划器对任务进度的判断主要依赖历史文本，容易误判某个子任务是否真正完成。",
   "inno": "提出 State-Relation Monitor，根据目标状态和实际执行状态之间的结构化关系，判断任务处于执行中、偏离目标还是已经完成。在 AgentBoard ALFWorld 和真实机器人实验中，整体成功率提升 **12.94%**，环境交互次数减少 **32.3%**。",
   "reuse": null,
   "short": "ProLoop",
   "slug": "proloop-progress-aware-closed-loop",
   "abstract": "Long-horizon robotic tasks are commonly decomposed into semantic subgoals to simplify planning. However, subgoal progress in LLM-based planning is typically inferred implicitly from interaction histories. Inaccurate estimates can misguide execution and allow local errors to propagate. To address this, we propose ProLoop, a Progress-Aware Closed-Loop framework that derives explicit subgoal progress from structured relations between desired and execution states. A learned State-Relation Monitor assesses these relations, which are then aggregated into subgoal progress as ongoing, deviated, or completed. The resulting progress guides execution adaptation, enabling reliable progress-aware control throughout execution. Experiments on AgentBoard ALFWorld and a real-world robotic platform show that ProLoop improves success rate by 12.94% overall and by 15.46% on Hard tasks, while reducing environment interactions by 32.3%.",
   "comments": "",
   "authors": "Zou, Runxin, Liang, Qingmiao, Zhang, Jiatao, Lv, Xinyan, Qie, Liangyu, Bu, Meiyi, Song, Wei, Zhu, Shiqiang",
   "links": []
  },
  "2610.39125": {
   "id": "2610.39125",
   "title": "Tracking Task Progress Through Errors and Recovery: A Benchmark and Recurrent Distributional Model",
   "zh": "通过错误与恢复跟踪任务进度：评测基准与循环分布模型",
   "org": "待核实（作者：Chen Wang、Chang Zhou 等）。",
   "prob": "长程机器人操作中的任务进度可能倒退或恢复，单纯根据当前图像估计进度，难以区分视觉相似但执行阶段不同的状态。",
   "inno": "提出 ReSTEP 数据集，包含 3,156 条操作轨迹和逐帧错误标注。同时提出 REMORA，通过循环视觉记忆、分布式进度预测和因果峰值选择，持续跟踪任务进度，并识别执行错误与恢复过程。",
   "reuse": null,
   "short": "任务进度追踪基准",
   "slug": "task-progress-tracking-errors-recovery-benchmark",
   "abstract": "Operational strategies for long-horizon robotic tasks require reliable evaluation to determine whether execution is progressing, stalling, regressing, or recovering from errors. Yet real- world robot datasets that capture error recovery and provide frame-level error annotations remain scarce. To address this gap, we introduce and open-source ReSTEP, a dataset of 3,156 episodes across two manipulation tasks with semantic stage labels and frame-level er- ror annotations. ReSTEP supports training and benchmarking progress monitors through successful execution, mistakes, and recovery. Existing progress estimators can struggle to distinguish visually similar task stages and maintain temporally consistent predictions while responding to genuine setbacks. We therefore propose REMORA, an online progress esti- mator that combines recurrent visual memory, distributional prediction, and causal peak selection. Memory retains execution history to help resolve visual ambiguity, while distribu- tional prediction represents alternative progress hypotheses and causal peak selection tracks a supported hypothesis over time. Together, these components enable consistent progress tracking that remains responsive to errors and recovery. Experiments on both ReSTEP tasks demonstrate that REMORA outperforms the state-of-the-art SARM baseline in progress ac- curacy and temporal consistency.",
   "comments": "",
   "authors": "Wang, Chen, Niu, Xinze, Zhu, Zihao, Zhou, Keru, Xu, Yicheng, Jin, Shihao, Wang, Wenhao, Jiang, Bowen et al.",
   "links": []
  },
  "2610.35604": {
   "id": "2610.35604",
   "title": "RoboDoctor: Training-Free Recursive Self-Improvement for Robot Policies",
   "zh": "RoboDoctor：无需训练的机器人策略递归自我改进",
   "org": "待核实（作者：Heyang Yu、Tianyu Xin 等）。",
   "prob": "通用机器人策略在执行失败后，通常需要重新规划或训练，但实际失败可能只需要调整控制权分配、局部动作或操作方法。",
   "inno": "提出不更新模型权重的执行时改进框架。分别针对策略调用、人工或智能体干预以及具体操作方法进行修正，并保存失败原因、修正动作和执行结果，用于后续任务复用。",
   "reuse": "wiki/entities/paper-robodoctor.md"
  },
  "2610.38388": {
   "id": "2610.38388",
   "title": "ELEVATE: Agentic Controller Acquisition from Task Failure for Quadrupedal Manipulation",
   "zh": "ELEVATE：从任务失败中自主学习四足机器人操作控制器",
   "org": "待核实（作者：Shiqi Liu、Radu Corcodel、Ding Zhao）。",
   "prob": "四足移动操作机器人可能具备任务规划能力，但现有控制器无法完成某些需要特殊全身运动的操作。",
   "inno": "当任务规划持续失败时，使用 Coding Agent 自动构建新的强化学习任务，包括奖励、课程和终止条件。训练得到的新控制器加入技能库，再与已有控制器组合完成任务。在真实四足机器人上，实现货架取物 18/20 成功和地面到货架放置 16/20 成功。",
   "reuse": "wiki/entities/paper-elevate.md"
  },
  "2610.40537": {
   "id": "2610.40537",
   "title": "DAWN: Diffusion-based Alignment and unfastening Workflow Network",
   "zh": "DAWN：基于扩散策略的对准与螺钉拆卸流程网络",
   "org": "待核实（作者：Hemant Kumar、Piyush Goenka、Nikhil Chopra）。",
   "prob": "自动螺钉拆卸需要完成精确对准、工具插入和持续施力，多个阶段的误差会相互影响。",
   "inno": "将任务分为对准插入和拆卸两个阶段。使用 Diffusion Policy 完成工具对准，利用多模态分类器判断阶段切换，再通过力控制完成螺钉拆卸，在 Franka 机器人上验证了完整流程。",
   "reuse": null,
   "short": "DAWN",
   "slug": "dawn-diffusion-alignment-unfastening",
   "abstract": "We present DAWN (Diffusion-based Alignment and unfastening Workflow Network), a hierarchical robotic framework designed to perform real-world, unfastening tasks through multi-modal sensory integration. Autonomous robotic unfastening is crucial for improving recycling processes, enhancing safety, and reducing operational costs; yet, it remains predominantly manual due to inherent challenges in precise screwdriver engagement with fasteners, detailed force execution, and robustness to environmental variability. To address the complexity of learning long-horizon, contact-rich manipulation tasks, we subdivide the unfastening task procedure into two sequential stages: (1) alignment & insertion, and (2) unfastening. DAWN consists of three key components: AlignNet, a diffusion-based policy ensuring precise screwdriver tip engagement; LatchNet, a multi-modal classifier reliably detecting successful stage transitions; and UnfastNet, which achieves stable, force-controlled screw removal. Experimental validation on a Franka Emika robot demonstrates that DAWN achieves alignment & insertion success rates comparable to trained human operators, robustly transitions between stages, and attains an impressive success rate in unfastening across various unseen scenarios. Ablation studies further elucidate critical input modalities and effective training strategies, highlighting DAWN's scalability, robustness, and data efficiency.",
   "comments": "",
   "authors": "Kumar, Hemant, Goenka, Piyush, Chopra, Nikhil",
   "links": []
  },
  "2610.40207": {
   "id": "2610.40207",
   "title": "RoboAtom: Large-scale Atomic-Action Supervision for Robot Learning",
   "zh": "RoboAtom：面向机器人学习的大规模原子动作监督",
   "org": "上海人工智能实验室、浙江大学、复旦大学、清华大学、上海交通大学。",
   "prob": "大规模机器人数据通常只包含整段任务指令，缺少细粒度动作语义和任务阶段标注，不利于策略学习动作组合与长程执行。",
   "inno": "构建 RoboAtom-Data 和 RoboAtom-Bench，使用原子动作表示组织机器人操作数据。结合人工标注、自动标注和同任务扩展机制生成细粒度监督，并研究这些标注如何用于机器人领域预训练、监督微调和策略执行。",
   "reuse": null,
   "short": "RoboAtom",
   "slug": "roboatom-atomic-action-supervision",
   "abstract": "Fine-grained action annotations can support skill reuse and compositional generalization in robot policies. However, robot datasets typically provide task-level instructions without subtask annotations. Although some include subtask labels, inconsistent formats and conventions hinder cross-source joint training. We propose RoboAtom-Data, which adds unified, time-aligned atomic-action annotations to existing data from multiple sources, covering 1.45 million trajectories, 6.88 million skill-labeled segments, and 16,882.1 hours. Our pipeline combines a unified protocol, large-scale manual annotation, propagation across repeated executions with the same ordered skill sequence, and multi-stage quality control. We evaluate the dataset through automatic atomic-action annotation and robot control. We construct RoboAtom-Bench to standardize evaluation of skill recognition and temporal boundary localization. Trained on the annotated data, RoboAtom-Anno achieves better automatic annotation performance than all evaluated open- and closed-source models on both RoboAtom-Bench and the external WGO-Bench. For robot control, atomic-action supervision paired with hierarchical execution improves task success over task-level baselines on LIBERO-Long, LIBERO-Plus, and RoboTwin. We will release the data, benchmark, model, and construction pipeline to support further research on skill-level robot learning.",
   "comments": "",
   "authors": "Wei, Xueyuan, Wang, Xuanxu, Li, Xudong, He, Yuping, Gan, Wanshui, Peng, Jiaqi, Guo, Wenqi, Zhou, Jianjun et al.",
   "links": []
  },
  "2610.39167": {
   "id": "2610.39167",
   "title": "EgoContact: Evaluating Physical Intelligence with Egocentric Video",
   "zh": "EgoContact：利用第一视角视频评估物理智能",
   "org": "慕尼黑工业大学、苏黎世联邦理工学院、剑桥大学等。",
   "prob": "人类第一视角视频被广泛用于机器人预训练，但模型是否真正理解物体应该在哪里接触，缺少直接评测。",
   "inno": "构建包含 22,500 个人工验证交互事件的数据集，提供接触前观测、操作意图和像素级接触位置。将 EgoContact 作为可供性迁移策略的记忆库后，真实机器人抓取成功率从 **16% 提升至 58%**。",
   "reuse": null,
   "short": "EgoContact",
   "slug": "egocontact-egocentric-physical-intelligence",
   "abstract": "Egocentric video has emerged as one of the most scalable sources of data for learning generalizable manipulation priors. Yet, despite its growing use in pretraining robotic policies, it has been underutilized as a validation tool for physical intelligence, a key component of which is understanding where to make contact with an object to achieve an intended interaction. We present EgoContact, a large-scale egocentric dataset for evaluating the generalizability and precision of learned manipulation priors through future contact-point prediction. EgoContact contains 22.5K manually verified interaction events, each with a pre-contact observation, semantic intent, and pixel-precise contact locations. We find that current methods often struggle to predict precise interaction contacts and sometimes fail to localize the target object in diverse, in-the-wild scenes, even when allowed multiple predictions. Furthermore, we demonstrate the usefulness of EgoContact's annotated interactions for robotic manipulation: simply using EgoContact as the memory bank of an existing affordance-transfer policy improves grasp success from 14.2% to 31.2% in simulation and from 16% to 58% on a real robot.",
   "comments": "",
   "authors": "Deryło, Adam, Gavryushin, Alexey, Cremers, Daniel, Pollefeys, Marc, Wang, Xi",
   "links": []
  },
  "2610.37799": {
   "id": "2610.37799",
   "title": "DexSuite: An All-in-One Framework for Dexterous Manipulation Across Arms, Hands, and Tasks",
   "zh": "DexSuite：跨机械臂、灵巧手与任务的一体化灵巧操作框架",
   "org": "麦吉尔大学、Mila 等。",
   "prob": "灵巧操作研究需要机器人模型、示范采集、策略训练和统一评测，但现有平台通常只覆盖其中部分流程。",
   "inno": "基于 Genesis 构建一体化灵巧操作仿真框架，包含 30 个任务、10 种机械臂、12 种末端执行器，可组合超过 100 种单臂配置和大量双臂配置。同时提供遥操作、模拟触觉、16 组标准评测任务及 4,700 条成功示范轨迹。",
   "reuse": "wiki/entities/paper-dexsuite.md"
  },
  "2610.39385": {
   "id": "2610.39385",
   "title": "UMI-Live for In-The-Wild Bimanual Robotics Data Collection",
   "zh": "UMI-Live：面向自然环境双臂机器人数据采集的开放系统",
   "org": "待核实（作者：Govind Velamoor、Adrian Lin、Brandon Y. Yang 等）。",
   "prob": "UMI 可以在不操作真实机器人的情况下采集人类示范，但设备标定、轨迹重建和数据处理流程仍较复杂。",
   "inno": "构建集成相机、LiDAR、惯性传感器和板载计算的双臂 UMI 系统。通过异步 SLAM 流程自动处理示范数据，并支持双臂遥操作和 VLA 训练。公开 40 小时、三个任务的示范数据，使用该数据微调的 π₀.₅ 和 MolmoAct2 在双臂抓放任务中分别取得 10/10 和 8/10 成功。",
   "reuse": "wiki/entities/paper-umi-live.md"
  },
  "2610.34717": {
   "id": "2610.34717",
   "title": "VLA-HWBench: Where the Time Goes When VLA Models Run on Edge Hardware",
   "zh": "VLA-HWBench：视觉语言动作模型在端侧硬件上的推理耗时分析",
   "org": "待核实（作者：Borui Xu、Bingsheng He、Michael Heinrich 等）。",
   "prob": "VLA 的推理延迟不仅取决于模型参数量，还受到动作生成方式、算子实现、CPU 预处理和内存带宽影响。",
   "inno": "在 NVIDIA Jetson AGX Thor 和 AMD Ryzen AI Max+ PRO 395 上评测 18 个开源 VLA 模型，将推理流程分为七个阶段。实验发现，动作生成方式比参数量更能解释延迟差异；使用较少 Flow Matching 步数生成较长动作块的模型，更容易满足实时控制要求。",
   "reuse": null,
   "short": "VLA-HWBench",
   "slug": "vla-hwbench-edge-hardware-latency",
   "abstract": "Vision-language-action (VLA) models increasingly run on edge devices aboard robots, where inference latency constrains closed-loop robot control. However, it remains unclear how to design VLA models, their inference software and edge hardware so that robots can run VLA inference in real time. Most VLA models report only task success, while prior studies cover limited action-generation strategies, divide inference into at most three stages, and lack comparisons of inference optimizations on the same models. Therefore, we introduce VLA-HWBench, a benchmark that measures 18 open VLA models on NVIDIA Jetson AGX Thor and AMD Ryzen AI Max+ PRO 395, with a seven-stage inference breakdown. Our study reveals three insights. First, action-generation strategy matters more than model size. It dominates 14 of 19 Jetson configurations, latency varies by $82\\times$ on Jetson and $157\\times$ on AMD, and parameter count explains at most 15% of latency variance. Models that generate long action chunks in a few flow-matching steps keep up with control, whereas token-by-token decoding does not. Second, kernel fusion with 4-bit weights helps most, running $\\pi_{0.5}$ and GR00T N1.7 2.6-2.9$\\times$ faster than PyTorch with this http URL, but it exists for only these two models and only on the Jetson. Optimization also moves the time to the stages it shortens least, as CPU preprocessing grows from 12% to 39% of the GR00T N1.7 call. Third, after optimization, memory bandwidth limits only the stages that process few tokens on the Jetson, such as the denoising steps, whereas software limits all other stages, from unfused kernels to serial preprocessing on the host CPU. Overall, low-latency edge VLAs should use few action-generation iterations, inference software should extend kernel fusion and low-bit weights to more models and platforms, and edge hardware needs high memory bandwidth and a fast host CPU.",
   "comments": "",
   "authors": "Xu, Borui, Tan, Hongshi, Jiang, Wenqi, He, Bingsheng, Wu, Ming, Zeng, Jason, Heinrich, Michael",
   "links": []
  },
  "2610.35832": {
   "id": "2610.35832",
   "title": "Profiling a Frontier Agent with EBench: From Direct Control to Harness with Policy",
   "zh": "使用 EBench 评估前沿智能体：从直接控制到策略协同执行",
   "org": "上海人工智能实验室等联合研究团队。",
   "prob": "多模态大模型能够直接生成机器人动作，但其能力与经过后训练的 VLA、WAM 相比如何，以及两类系统能否有效协作，需要统一评测。",
   "inno": "在 EBench 上比较直接控制与智能体调用机器人策略两种模式。结果显示，直接控制在短程任务上具有竞争力，但在长程和精细接触任务上不如专门训练的策略；将智能体与 VLA/WAM 结合后，能够利用两者互补能力改善整体执行表现。",
   "reuse": "wiki/entities/paper-ebench-frontier-agent-profiling.md"
  },
  "2610.38221": {
   "id": "2610.38221",
   "title": "CuraEval: Evaluating Demonstration Curation on Mixed-Skill Robot Data",
   "zh": "CuraEval：评估混合技能水平机器人示范的数据筛选方法",
   "org": "Colorado School of Mines、东北大学等。",
   "prob": "模仿学习数据可能来自不同水平的操作者。复杂的数据质量评分方法是否真的优于随机筛选或简单轨迹长度筛选，缺少统一比较。",
   "inno": "建立统一的数据筛选评测流程，使用 Diffusion Policy 和 BC-RNN 比较多种示范选择方法。研究发现，部分复杂评分指标主要反映轨迹长度，而且筛选结果对训练和评测协议敏感。",
   "reuse": "wiki/entities/paper-curaeval.md"
  },
  "2610.35462": {
   "id": "2610.35462",
   "title": "DEX-PHD: A Dexterous Portable Handheld Device for Multimodal Data Collection",
   "zh": "DEX-PHD：面向多模态数据采集的便携式灵巧手持设备",
   "org": "待核实（作者：Jooyoung Hong、Joohyung Kim 等）。",
   "prob": "灵巧操作示范需要同时记录手指动作、接触状态和视觉信息，但传统遥操作设备通常不便携，纯视觉方案又难以获得准确关节状态。",
   "inno": "设计便携式灵巧手持设备，集成多指关节控制、本体状态记录、指尖接触数据和腕部相机。采集到的示范可以直接用于真实人形机器人灵巧手动作重放。",
   "reuse": "wiki/entities/paper-dex-phd-handheld-data-collection.md"
  },
  "2610.39076": {
   "id": "2610.39076",
   "title": "Tactus: Tactile & Pose Sensing Glove for Natural Dexterity Data Collection",
   "zh": "Tactus：用于自然灵巧操作数据采集的触觉与姿态感知手套",
   "org": "Meta Reality Labs Research。",
   "prob": "人类灵巧操作示范需要准确的手部姿态和接触力信息，但相机在手物遮挡时难以恢复姿态，传统传感手套又可能影响自然抓取动作。",
   "inno": "提出织物集成式传感手套，结合电容触觉阵列与 IMU 姿态跟踪。系统包含 47 个触觉传感位置和 16 个六轴 IMU，并提供跨设备标定流程。在操作过程中，指尖关键点相对误差约 11.0 mm，可用于接触感知机器人策略的数据采集。",
   "reuse": null,
   "short": "Tactus",
   "slug": "tactus-tactile-pose-sensing-glove",
   "abstract": "Collecting naturalistic human manipulation demonstrations with accurate contact force and hand pose is essential for robot dexterous manipulation policy and foundational world model learning - yet no existing wearable platform provides both calibrated fingertip force sensing and occlusion-resilient hand tracking in a form factor that preserves natural manipulation behavior. We present Tactus, a textile-integrated sensing glove that co-optimizes three design pillars: (1) a thin, conformal 3D-knit form factor (1.5 mm fingertip stackup) designed to minimize interference with natural manipulation, (2) strategically placed capacitive tactile sensors at manipulation-relevant regions informed by empirical contact studies, and (3) IMU-fused pose tracking that maintains accuracy through the hand-object occlusions where manipulation data is most valuable. Tactus includes a documented calibration pipeline for cross-glove consistency and per-session calibration for subject-dependent correction. To enable objective comparison, we introduce standardized benchmarking protocols for tactile force regression and hand pose estimation, and apply them to Tactus and two commercially available tactile gloves - to our knowledge the first quantitative cross-platform evaluation of this kind. Tactus resolves a minimum detectable force of 0.59 N at 95% confidence, 1.6 - 2.7x lower than the commercial baselines, and with 0.87 cross-participant consistency, while our pose tracking sustains 11.0 mm fingertip keypoint relative error during manipulation, when occlusion is heaviest. We show that these combined signals support object property estimation, slip detection, and representative recording of hand-object interaction - yielding demonstration data with the fidelity required for contact-aware manipulation policies and world models.",
   "comments": "",
   "authors": "Guan, Li, Granberry, Rachael, Aseeri, Sahar, Rostami, Sahar, Niehues, Taylor, Marri, Sai Dhanooshi, Huang, Mia, Zhang, Dingtian et al.",
   "links": []
  }
 },
 "lists": {
  "man": [
   {
    "id": "2610.40373",
    "n": 1,
    "star": false,
    "sec": "VLA / 通用操作策略"
   },
   {
    "id": "2610.37381",
    "n": 2,
    "star": false,
    "sec": "VLA / 通用操作策略"
   },
   {
    "id": "2610.38000",
    "n": 3,
    "star": false,
    "sec": "VLA / 通用操作策略"
   },
   {
    "id": "2610.37215",
    "n": 4,
    "star": false,
    "sec": "VLA / 通用操作策略"
   },
   {
    "id": "2610.37378",
    "n": 5,
    "star": false,
    "sec": "VLA / 通用操作策略"
   },
   {
    "id": "2610.38661",
    "n": 6,
    "star": false,
    "sec": "VLA / 通用操作策略"
   },
   {
    "id": "2610.38508",
    "n": 7,
    "star": false,
    "sec": "VLA / 通用操作策略"
   },
   {
    "id": "2610.36011",
    "n": 1,
    "star": false,
    "sec": "WAM / World Model 与操作预测"
   },
   {
    "id": "2610.36791",
    "n": 2,
    "star": false,
    "sec": "WAM / World Model 与操作预测"
   },
   {
    "id": "2610.36823",
    "n": 3,
    "star": false,
    "sec": "WAM / World Model 与操作预测"
   },
   {
    "id": "2610.40401",
    "n": 4,
    "star": false,
    "sec": "WAM / World Model 与操作预测"
   },
   {
    "id": "2610.39528",
    "n": 5,
    "star": false,
    "sec": "WAM / World Model 与操作预测"
   },
   {
    "id": "2610.36840",
    "n": 6,
    "star": false,
    "sec": "WAM / World Model 与操作预测"
   },
   {
    "id": "2610.37091",
    "n": 7,
    "star": false,
    "sec": "WAM / World Model 与操作预测"
   },
   {
    "id": "2610.35850",
    "n": 8,
    "star": false,
    "sec": "WAM / World Model 与操作预测"
   },
   {
    "id": "2610.39278",
    "n": 9,
    "star": false,
    "sec": "WAM / World Model 与操作预测"
   },
   {
    "id": "2610.39232",
    "n": 10,
    "star": false,
    "sec": "WAM / World Model 与操作预测"
   },
   {
    "id": "2610.37865",
    "n": 1,
    "star": false,
    "sec": "在线/离线强化学习与后训练"
   },
   {
    "id": "2610.39212",
    "n": 2,
    "star": false,
    "sec": "在线/离线强化学习与后训练"
   },
   {
    "id": "2610.38513",
    "n": 3,
    "star": false,
    "sec": "在线/离线强化学习与后训练"
   },
   {
    "id": "2610.35659",
    "n": 4,
    "star": false,
    "sec": "在线/离线强化学习与后训练"
   },
   {
    "id": "2610.36694",
    "n": 1,
    "star": false,
    "sec": "模仿学习、视觉运动策略与生成式动作策略"
   },
   {
    "id": "2610.40271",
    "n": 2,
    "star": false,
    "sec": "模仿学习、视觉运动策略与生成式动作策略"
   },
   {
    "id": "2610.40356",
    "n": 3,
    "star": false,
    "sec": "模仿学习、视觉运动策略与生成式动作策略"
   },
   {
    "id": "2610.40075",
    "n": 4,
    "star": false,
    "sec": "模仿学习、视觉运动策略与生成式动作策略"
   },
   {
    "id": "2610.40430",
    "n": 5,
    "star": false,
    "sec": "模仿学习、视觉运动策略与生成式动作策略"
   },
   {
    "id": "2610.35458",
    "n": 6,
    "star": false,
    "sec": "模仿学习、视觉运动策略与生成式动作策略"
   },
   {
    "id": "2610.36250",
    "n": 1,
    "star": false,
    "sec": "双臂操作"
   },
   {
    "id": "2610.36961",
    "n": 2,
    "star": false,
    "sec": "双臂操作"
   },
   {
    "id": "2610.40210",
    "n": 3,
    "star": false,
    "sec": "双臂操作"
   },
   {
    "id": "2610.37533",
    "n": 4,
    "star": false,
    "sec": "双臂操作"
   },
   {
    "id": "2610.37652",
    "n": 1,
    "star": false,
    "sec": "灵巧操作"
   },
   {
    "id": "2610.39582",
    "n": 2,
    "star": false,
    "sec": "灵巧操作"
   },
   {
    "id": "2610.40349",
    "n": 3,
    "star": false,
    "sec": "灵巧操作"
   },
   {
    "id": "2610.39044",
    "n": 4,
    "star": false,
    "sec": "灵巧操作"
   },
   {
    "id": "2610.35058",
    "n": 1,
    "star": false,
    "sec": "柔性/可变形物体操作"
   },
   {
    "id": "2610.35103",
    "n": 2,
    "star": false,
    "sec": "柔性/可变形物体操作"
   },
   {
    "id": "2610.40559",
    "n": 3,
    "star": false,
    "sec": "柔性/可变形物体操作"
   },
   {
    "id": "2610.35075",
    "n": 1,
    "star": false,
    "sec": "人形机器人上半身操作"
   },
   {
    "id": "2610.38027",
    "n": 2,
    "star": false,
    "sec": "人形机器人上半身操作"
   },
   {
    "id": "2610.35862",
    "n": 3,
    "star": false,
    "sec": "人形机器人上半身操作"
   },
   {
    "id": "2610.40016",
    "n": 1,
    "star": false,
    "sec": "长程任务、推理与规划"
   },
   {
    "id": "2610.36533",
    "n": 2,
    "star": false,
    "sec": "长程任务、推理与规划"
   },
   {
    "id": "2610.39319",
    "n": 3,
    "star": false,
    "sec": "长程任务、推理与规划"
   },
   {
    "id": "2610.39125",
    "n": 4,
    "star": false,
    "sec": "长程任务、推理与规划"
   },
   {
    "id": "2610.35604",
    "n": 5,
    "star": false,
    "sec": "长程任务、推理与规划"
   },
   {
    "id": "2610.38388",
    "n": 6,
    "star": false,
    "sec": "长程任务、推理与规划"
   },
   {
    "id": "2610.40537",
    "n": 7,
    "star": false,
    "sec": "长程任务、推理与规划"
   },
   {
    "id": "2610.40207",
    "n": 1,
    "star": false,
    "sec": "数据集、预训练、跨本体与硬件平台"
   },
   {
    "id": "2610.39167",
    "n": 2,
    "star": false,
    "sec": "数据集、预训练、跨本体与硬件平台"
   },
   {
    "id": "2610.37799",
    "n": 3,
    "star": false,
    "sec": "数据集、预训练、跨本体与硬件平台"
   },
   {
    "id": "2610.39385",
    "n": 4,
    "star": false,
    "sec": "数据集、预训练、跨本体与硬件平台"
   },
   {
    "id": "2610.34717",
    "n": 5,
    "star": false,
    "sec": "数据集、预训练、跨本体与硬件平台"
   },
   {
    "id": "2610.35832",
    "n": 6,
    "star": false,
    "sec": "数据集、预训练、跨本体与硬件平台"
   },
   {
    "id": "2610.38221",
    "n": 7,
    "star": false,
    "sec": "数据集、预训练、跨本体与硬件平台"
   },
   {
    "id": "2610.35462",
    "n": 8,
    "star": false,
    "sec": "数据集、预训练、跨本体与硬件平台"
   },
   {
    "id": "2610.39076",
    "n": 9,
    "star": false,
    "sec": "数据集、预训练、跨本体与硬件平台"
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
