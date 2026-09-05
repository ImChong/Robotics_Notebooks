---
type: task
tags: [vln, navigation, embodied-ai, vision-language, matterport]
summary: "视觉–语言导航（VLN）要求智能体在三维环境中依据自然语言指令执行一系列离散或连续动作到达目标，是连接语言理解与空间运动规划的基准任务。"
updated: 2026-09-05
status: complete
related:
  - ../entities/paper-abot-n1.md
  - ../entities/paper-roamflow.md
  - ../comparisons/vlm-vln-vla-vlx-world-model-taxonomy.md
  - ../queries/embodied-fm-taxonomy-loop.md
  - ../overview/vln-10-papers-technology-map.md
  - ../overview/vln-open-source-repro-paradigms.md
  - ./zero-shot-object-navigation.md
  - ../entities/sceneverse-pp.md
  - ../entities/esi-bench.md
  - ../entities/paper-worldvln-aerial-vln-wam.md
  - ../entities/paper-uni-lavira.md
  - ../entities/paper-humembr.md
  - ../entities/paper-navwam-goal-conditioned-visual-navigation-wam.md
  - ../entities/paper-green-for-go-vla-nav-grounding.md
  - ../entities/paper-humanoidvln.md
  - ../entities/paper-crosstracer.md
  - ../entities/paper-sru-spatially-enhanced-recurrent-memory.md
  - ../concepts/3d-spatial-vqa.md
  - ../concepts/world-action-models.md
  - ../methods/vla.md
  - ../entities/paper-homeworld-whole-home-scene-generation.md
  - ../entities/paper-vesta-generalist-embodied-reasoning.md
  - ../entities/paper-realm-last-3-meter-vln-grounding.md
  - ../entities/paper-3d-ic-joint-navigation-manipulation-planning.md
  - ../entities/paper-da-nav.md
  - ../entities/paper-fsd-vln.md
  - ../entities/paper-language-to-navigation-goals-rgbd.md
  - ../entities/paper-arcadia.md
  - ../entities/paper-zonda.md
  - ../entities/paper-travexplorer.md
  - ../entities/paper-s-squared-vla.md
  - ../entities/qwen-robot-nav.md
  - ../entities/paper-refertrack.md
  - locomotion.md
sources:
  - ../../sources/blogs/wechat_shenlan_five_embodied_model_taxonomy.md
  - ../../sources/blogs/wechat_shenlan_vln_repro_four_paradigms_2026.md
  - ../../sources/repos/sceneverse-pp.md
  - ../../sources/papers/worldvln_arxiv_2605_15964.md
  - ../../sources/papers/uni_lavira_arxiv_2605_27582.md
  - ../../sources/papers/humembr_arxiv_2606_30404.md
  - ../../sources/papers/realm_last_3_meter_vln_arxiv_2607_03792.md
  - ../../sources/papers/3d_ic_icml_2026.md
  - ../../sources/papers/da_nav_arxiv_2607_11638.md
  - ../../sources/papers/fsd_vln_arxiv_2607_08359.md
  - ../../sources/papers/arcadia_arxiv_2512_00076.md
  - ../../sources/papers/zonda_arxiv_2607_21025.md
  - ../../sources/papers/refertrack_arxiv_2607_20061.md
  - ../../sources/papers/green_for_go_vla_nav_grounding_arxiv_2607_05122.md
  - ../../sources/papers/humanoidvln_arxiv_2608_12860.md
  - ../../sources/papers/crosstracer_arxiv_2608_06688.md
  - ../../sources/papers/abot_n1_arxiv_2607_10383.md
---

# 视觉–语言导航（Vision-and-Language Navigation, VLN）

**VLN**：智能体接收 **自然语言导航指令** 与 **第一人称（egocentric）视觉观测**（渲染视图或真实相机图像），在离散或连续动作空间中决策，最终到达指令描述的目标位置或物体。**语言–视觉接地** 与 **路径效率** 是核心评价维度。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉-语言-动作多模态基础策略方向 |
| VLM | Vision-Language Model | 视觉-语言多模态理解模型，VLA 的上游 |
| WAM | World Action Model | 联合世界模型与动作预测的架构 |
| LLM | Large Language Model | 大语言模型，常作高层任务/语言接口 |
| Locomotion | Robot Locomotion | 足式/人形等无轮移动能力的总称 |

## 为什么重要？

- **机器人场景**：家庭或服务机器人需要理解「穿过客厅，在冰箱左侧停下」这类指令；VLN 提供了可复现的 **语言–几何–动作** 闭环基准。
- **与纯导航的区别**：传统导航多依赖地图与坐标目标；VLN 强调 **语义描述**（地标、相对运动），更贴近人类口头指路。
- **与 VLA 的衔接**：高层策略可将 VLN 视作「语言条件下的路径生成」子问题；仿真基准（如 Matterport3D 上的 **R2R / RxR**）与真实视频蒸馏数据（如室内 tour）常混合使用以缓解 **sim–real** 与 **轨迹分布** 差异。通才 VLA 如 [Qwen-VLA](../entities/qwen-vla.md) 在官方 README 中把 **操作与 VLN 基准** 放进 **同一 checkpoint** 联合评测，可作为「导航是否应并入统一 VLA」的工程参照。
- **与 image-goal 视觉导航的对照**：[NavWAM](../entities/paper-navwam-goal-conditioned-visual-navigation-wam.md) 研究 **目标图像**（非自然语言）条件下的 egocentric 闭环导航，用 **Cosmos Predict 2 系 WAM** 联合预测未来观测、value 与 action，在 **go stanford** 与真机上相对 **OmniVLA** 与 **NWM+CEM** 报告增益——说明「导航」任务族内 **语言接地** 与 **视觉目标接地** 可走不同基础模型路线。另一条 image-goal 生成策略见 [RoamFlow](../entities/paper-roamflow.md)（MeanFlow 一步轨迹 + IL→RL；Habitat/Go2；**未开源**）。
- **冻结导航 VLA 的推理时可通行接地**：[Green for Go](../entities/paper-green-for-go-vla-nav-grounding.md)（arXiv:2607.05122）用 SegFormer **绿/红 overlay** 喂冻结 **OmniVLA**，Grand Tour 上最远航点误差降 **27–44%**，但归一化后显示主要是轨迹缩短约 **30%**；**图像目标几乎无增益**，「stop」修不掉。这是输入侧试验旋钮，不是新 VLA；**未开源**。勿与 [Green-VLA](../entities/paper-greenvla-staged-vla-humanoid.md) 混淆。
- **跨本体像素轨迹残差**：[CrossTracer](../entities/paper-crosstracer.md)（arXiv:2608.06688）把 OmniVLA 改成无本体 **VL-Tracer**，再用 **CE-Adapter** 按机器人 ID 改 8 个归一化航点；NaviTrace 总分 **45.68**（相对 Gemini-2.5-Pro +28.1%），真机相对 OmniVLA 抬轮式/腿式 SR。与 Green for Go 同碰 OmniVLA，但是 **重训两段 + 显式本体**，不是推理时涂色。**项目页截至 2026-09-04 无代码仓。**

- **与坐标目标无地图循环导航的对照**：[SRU](../entities/paper-sru-spatially-enhanced-recurrent-memory.md)（IJRR 2025）用 **相对目标向量 + 单目前向深度 + SRU 隐式空间记忆** 做 **50–120 m 级** 无地图 RL 导航（非语言、非 image-goal），真机 **B2W 零样本**；部署移植见 [SRU-Odin](../entities/sru-odin.md)。与 VLN 共享「部分可观测 + 长程」难点，但 **监督与目标接口** 完全不同，不宜混用 R2R 等语言基准。
- **Agentic 导航基座**：[Qwen-RobotNav](../entities/qwen-robot-nav.md) 以 **可控观测协议 + 任务 mode** 统一 VLN / ObjNav / 跟踪 / NAVSIM 驾驶，并作为 **Qwen3.7-Plus** 等 planner 的导航原语；与 [Qwen-Robot Suite](../entities/qwen-robot-suite.md) 长时程 **EQA / 开放世界寻物** demo 一并阅读。端到端驾驶专用 VLA 对照见 [S²-VLA](../entities/paper-s-squared-vla.md)（语义∥空间双流规划，NAVSIM 纯 SFT）。
- **具身视觉跟踪（EVT）专用 referring**：[ReferTrack](../entities/paper-refertrack.md)（arXiv:2607.20061）把「语言指定行人跟随」拆成索引 bbox Refer-CoT + TVBI 历史几何，再解码航点；EVT-Bench 单前视 STT/DT/AT 上相对 TrackVLA++ 大幅抬升，Go2/G1 真机定性——与通才 RobotNav 的 tracking mode 同榜不同协议，宜分开读。
- **人中心例行记忆 EQA**：[HUMEMBR](../entities/paper-humembr.md)（arXiv:2606.30404）在办公室 Spot 上并行构建 **身份感知多日记忆** 与 LLM 工具检索，服务 PersonEQA 与「找某人」导航——相对 R2R 式路径指令，主轴是 **人例行预测** 而非地标语言接地。
- **通才 planner 统一导航 + 推理**：[Vesta](../entities/paper-vesta-generalist-embodied-reasoning.md) 在同一 **Qwen3-VL-8B** checkpoint 上同时 SFT **VLN-CE（R2R/RxR/ScaleVLN）** 与具身 cognition/localization；R2R-CE **SR 55.5%** 与 InternVLA-N1 specialist 持平，而 **Nav-only finetune 的 generalist 竞品在 R2R 上 SR=0**（灾难性遗忘）——说明 VLN 是否应并入 **更大 planner mix** 时需评估 **域外遗忘** 而不仅是导航榜分数。
- **REVERIE 末段接地鸿沟**：[REALM](../entities/paper-realm-last-3-meter-vln-grounding.md)（arXiv:2607.03792）指出 REVERIE-CE 等任务虽要求框出目标实例，但主流 **3 m SR** 不评 **最终朝向与可见性**——ETPNav-FT SR=34.67% 时 **ONS@0.1m 仅 6.32%**；作者提出 **plug-and-play 末段精修** 与 **REVERIE-AIM** 实例中心评测集。
- **OVMM 导航–操作联合规划**：[3D-IC](../entities/paper-3d-ic-joint-navigation-manipulation-planning.md)（ICML 2026）在 **共享 3D 特征图** 上为开放词汇移动操作生成 **多阶段交互路点链**，用 **VLM 路点可行性 + 转移代价** 选链，避免「导航到了但操作姿态差」的分阶段错配；真机 **Stretch 3** 验证。
- **城市尺度方向感知 VLN**：[DA-Nav](../entities/paper-da-nav.md)（arXiv:2607.11638）用 **商业导航离散方向指令**（非细粒度地标描述），在 **egocentric 图像平面网格** 上做 spatial grounding，并以 **CoT + ReDA recovery** 支撑长程纠偏；CARLA SoTA，**零样本** 迁到 Go2 / 乐聚人形公里级户外——与室内 R2R/REVERIE 栈互补。
- **多楼层动态零样本 ObjectNav**：[ZONDA](../entities/paper-zonda.md)（arXiv:2607.21025）在 Habitat HM3D/MP3D 上用 **高度差可通行图 + 启发式跨楼层规划**、**多视角 VLM 核验** 与 **行人预测避障**，并自建 **HM3D-DYNA**；相对 ASCENT 等不绑平台 RL PointNav，真机部署 Direct Drive Tech TITA。
- **慢–快 VLN 基础模型 + 城市闭环基准**：[ABot-N1](../entities/paper-abot-n1.md)（arXiv:2607.10383）用 **4B 慢推理器（CoT + 像素目标）+ 2B 快动作专家** 统一 Point / Object / POI / 指令 / 跟人五任务；开源 **ABotN-PointBench / POIBench** 与 3DGS 闭环评测栈，POI SR **77.3%**、室内外 Point-Goal **95.4% / 92.9%**；**模型权重截至 2026-08-31 未发布**。

## 核心要素

| 要素 | 说明 |
|------|------|
| 环境 | 常用大规模室内扫描数据集（如 Matterport3D）构建可导航网格 |
| 观测 | 全景图序列或 pinhole 渲染视图；近年也引入真实行走视频 |
| 动作 | 常见为离散前向/转向步长；需与数据集标注一致 |
| 监督 | 专家轨迹模仿、强化学习、或从网页视频重建的伪轨迹 + VLM 生成指令 |

**分布差异**：仿真中最短路径、朝前行走居多；真实 Room-tour 视频存在停顿、回头与冗余旋转，直接用作监督需要 **轨迹清洗与动作离散化**（SceneVerse++ 论文中描述了面向 R2R 的三阶段管线）。

### 空中 / UAV 子域

- **设定差异：** [WorldVLN](../entities/paper-worldvln-aerial-vln-wam.md) 等 **空中 VLN** 工作在 **连续 3D 航点** 与 **大视角 egocentric 变化** 下闭环执行语言指令；相对 Matterport 离散转向，更强调 **因果记忆、短视界世界预测与真机迁移**。
- **范式对照：** 地面开源栈见 [四范式复现路径](../overview/vln-open-source-repro-paradigms.md)；空中路线可将 **自回归 World Action Model**、**导航 VLA** 与 **快慢双系统** 对照阅读（[WAM 概念页](../concepts/world-action-models.md)）。
- **零样本统一 agent：** [Uni-LaViRA](../entities/paper-uni-lavira.md)（arXiv:2605.27582）把 VLN-CE / ObjectNav / EQA / Aerial-VLN 写成同一 **Language→Vision→Robot** 翻译环，**无机器人轨迹训练**；OpenUAV SR 40.0%，并与训练式导航基础模型对照。
- **ROS 2 分层语义导航（待开源）：** [Language-to-Navigation-Goals](../entities/paper-language-to-navigation-goals-rgbd.md)（arXiv:2607.13624）用远程 VLM bbox + RGB-D 投影生成 **Nav2** 目标，TurtleBot3 端到端导航误差约 0.70 m，Go2 真机定位约 0.51 m；与端到端 VLN 策略互补。
- **快慢双系统（仿真、未开源）：** [FSD-VLN](../entities/paper-fsd-vln.md)（arXiv:2607.08359）冻结 VLM 写 **VLSF**，GR00T N1 系 DiT 异步出 8 类离散飞行动作；未见相对自复现 OpenFly SR 5.1%→**13.6%**，单步/任务时长约减半，但 **H=1 最好**、无真机。勿与 WorldVLN 的世界转移或室内 3 m SR 混读。

### 城市尺度 Point / POI 闭环基准

- **设定差异：** [ABot-N1](../entities/paper-abot-n1.md) 的 **ABotN-PointBench / POIBench** 用真实场景 **3DGS 重建 + 社会可通行标注** 做 **闭环连续航点** 评测（室外 SR<sub>&lt;3col</sub>、室内 SR<sub>&lt;1col</sub>、POI SR@2m），相对旧 open-loop waypoint 或粗街区 POI 榜更贴近部署。
- **方法要点：** 慢–快双系统；慢系统输出 **CoT + Target/Affordance Pixel** 作五任务统一接口；快系统追像素生成 SE(2) 航点；GRPO 对齐到达结果。
- **开源边界：** **部分开源**——[`amap-cvlab/ABot-Navigation`](https://github.com/amap-cvlab/ABot-Navigation)（`ABotN-Bench`）数据 + `abotn_evaluator` + `render_server`；**策略权重未发布**。

### 户外 / 城市尺度方向指令

- **设定差异：** [DA-Nav](../entities/paper-da-nav.md) 面向 **地面足式/人形** 的 **城市户外闭环**，指令来自 **Google Maps / 高德** 解析出的 FORWARD / TURN\_\* / STOP，而非 Matterport 细粒度路径描述。
- **方法要点：** 离散 **图像平面网格** + **偏离评估→动作→轨迹** CoT；配套 **ReDA**（含 recovery）；评测额外强调 **DF / CSR**。
- **开源边界：** 截至 2026-07-22 **未发布** 官方代码/权重；选型可读方法，复现仍走 [四范式开源栈](../overview/vln-open-source-repro-paradigms.md)。

### 冻结 VLA 的推理时可通行 overlay

- **设定差异：** [Green for Go](../entities/paper-green-for-go-vla-nav-grounding.md) 不训新策略，只在 OmniVLA 前把 egocentric 图标成 **绿=可通行 / 红=不可通行**；评测是 Grand Tour ETH-2 的 **开环 7 航点误差**，不是 R2R-CE SR。
- **方法要点：** observation-only vs joint（语言后缀或目标图同样 overlay）；长指令远航点收益更大；长度归一化后优势消失。
- **开源边界：** 截至 2026-08-14 **确认未开源**；可跑通导航栈仍走 [四范式开源复现](../overview/vln-open-source-repro-paradigms.md)。

### 跨本体：像素轨迹残差（不是换双足控制器）

- **设定差异：** [CrossTracer](../entities/paper-crosstracer.md)（arXiv:2608.06688）评的是 **同一语义目标下，轮式/腿式该走哪条图像平面路径**，不是 Habitat SR，也不是 [HumanoidVLN](../entities/paper-humanoidvln.md) 的摔倒协议。
- **方法要点：** VL-Tracer 出无本体 8 航点；CE-Adapter 用 FiLM + 残差；CE-RRT* 从分割自动造监督。NaviTrace **45.68**；去 adapter **22.56**；真机相对 OmniVLA 轮式 SR **0.40→0.65**、腿式 **0.45→0.70**。
- **开源边界：** 截至 2026-09-04 **宣称开源 / 待核实**（对照表打勾，项目页无 GitHub）；可跑通栈仍走 [四范式](../overview/vln-open-source-repro-paradigms.md)。

### 人形物理执行：跨本体 Isaac VLN

- **设定差异：** [HumanoidVLN](../entities/paper-humanoidvln.md)（arXiv:2608.12860）不把人形当 Habitat 传送代理，而在 [Isaac Sim](../entities/isaac-sim.md) 上用 **分本体 RL 步态 + PD/MPC 跟踪** 走完语言指令；场景按 **≥100 m² 可通行** 筛选，指令由 MAA + 人工核验，相对 R2R/VLN-CE 把 **摔倒（FR）** 写进协议。
- **方法要点：** 四本体（G1 / H1 / Internal-A / B）零样本评测 NaVILA、DualVLN、StreamVLN、JanusVLN；JanusVLN 平均 SR 43.55%，H1 上部分模型 FR >60%；G1 + DualVLN 20 条 sim–real NE **r=0.935**。
- **开源边界：** 截至 2026-08-14 **宣称录用后开源**，项目页无 GitHub；可跑通栈仍走 [四范式](../overview/vln-open-source-repro-paradigms.md) 与已开源 [NaVILA](../entities/paper-notebook-navila-legged-robot-vision-language-action-model.md)。

### 全生命周期：采集–生成仿真–共享 VLA–反馈

- **设定差异：** [Arcadia](../entities/paper-arcadia.md)（arXiv:2512.00076）不把 VLN 当孤立导航模型，而写成与操作共享 Qwen2.5-VL 的 **Real2Sim2Real 闭环**：部署环境自采 → 3DGS USD → Isaac 训策略 → 真机反馈写回。
- **方法要点：** 同架构下仅换自采数据，相对 NaVILA 平均 SR **+2.7 pp**；加反馈后 VLN-CE-Isaac SR **50.1%**。G1 真机导航 46/100（对照 13）。联合训练几乎不掉点。
- **开源边界：** [EmbodiedKit](../../sources/repos/embodiedkit.md) **部分开源**（数据生成 + Qwen 训练脚本）；探索 / 3DGS / 反馈与权重未发布，**不能**当 [四范式](../overview/vln-open-source-repro-paradigms.md) 新手栈。

### 室内 ObjectNav：跨楼层 + 动态行人

- **设定差异：** [ZONDA](../entities/paper-zonda.md) 目标是 **物体类别到达**（非细粒度语言路径）；评测主线为 Habitat **多楼层** HM3D/MP3D，并扩展 **HM3D-DYNA** 加入移动行人。
- **方法要点：** OST 三图 + 启发式跨楼层（\(H_{\text{agent}}\)）+ 多视角 VLM 确认 + Kalman 行人预测；真机用 MPPI 连续跟踪。
- **开源边界：** 截至 2026-07-24 **未发布** 官方代码/权重；可与已开源 [ASCENT](https://github.com/Zeying-Gong/ascent) / [VLFM](https://github.com/bdaiinstitute/vlfm) 对照读方法。

## 常见误区

- **误区**：「VLN 做得好就等于机器人能走。」仿真离散动作与真实连续控制、动力学约束仍有鸿沟，通常需要低层控制与碰撞规避模块。[HumanoidVLN](../entities/paper-humanoidvln.md) 进一步表明：同一模型在 G1 与 H1 上 SR/FR 可差一个数量级，传送式分数不能外推到双足。
- **误区**：「只用仿真轨迹训练就能覆盖真实室内。」真实视频的引入（含自动指令生成）是为了丰富 **语言风格与行走模式**，但仍需评估在标准基准上的可迁移性。

## 与其他页面的关系

- **技术地图**：[VLN 10 篇论文技术地图](../overview/vln-10-papers-technology-map.md) — 2018–2024 代表性工作按「数据集/平台 → 算法框架」两组入口串读，贯穿减负演进线（R2R → NaVid）。
- **复现路径**：[VLN 四范式开源复现策展](../overview/vln-open-source-repro-paradigms.md) — VLFM / NavGPT / NoMaD / Uni-NaVid 由浅入深（模块化→LLM→扩散 e2e→导航 VLA）。
- **零样本统一导航**：[Uni-LaViRA](../entities/paper-uni-lavira.md) — training-free 三层翻译 + TDM/SCB；四任务 × 四真机（arXiv:2605.27582）。
- **空中 WAM**：[WorldVLN](../entities/paper-worldvln-aerial-vln-wam.md) — 潜自回归世界转移 + 航点解码 + Action-aware GRPO；室内外 UAV 基准与真机部署（arXiv:2605.15964）。
- **空中快慢双系统**：[FSD-VLN](../entities/paper-fsd-vln.md) — 冻结 VLM + VLSF + DiT 异步飞控；仿真未见相对 OpenFly SR 约 2.7×，无真机、未开源（arXiv:2607.08359）。
- **数据**：[SceneVerse++](../entities/sceneverse-pp.md) 将室内漫游视频转为 R2R 兼容的离散导航数据，并报告在相关基准上的增益。
- **全屋仿真场景**：[HomeWorld](../entities/paper-homeworld-whole-home-scene-generation.md) 从文本生成 **sim-ready 多房间家居**（300K **Chinese Style** 矢量平面图 + 5K furnished 全屋 3D 待开源），面向 **跨房间语言导航与家务** 的 **户型本地化** 数据链——与 Matterport 系扫描 benchmark 互补而非直接替代。
- **空间推理**：[3D 空间 VQA](../concepts/3d-spatial-vqa.md) 侧重问答；VLN 侧重 **时序决策**，二者常共享场景表示与 VLM 骨干。[ESI-Bench](../entities/esi-bench.md) 则在 OmniGibson 上评测 **为看见而行动** 的细粒度空间 QA，与 VLN 的 **轨迹到达** 目标互补。
- **运动基础**：[Locomotion](locomotion.md) 提供低层移动能力；VLN 更多占据 **任务规划与语义接地** 层，可与 VLA 分层结合。
- **模型**：[VLA](../methods/vla.md) 可作为统一骨架，在导航子任务上接入离散动作头或目标点输出。
- **通才 embodied planner**：[Vesta](../entities/paper-vesta-generalist-embodied-reasoning.md) — 导航与具身推理、长时程子任务规划 **单模型 SFT**；R2R-CE 与 offline planning / 真机 memory 任务一并报告（arXiv:2606.20905）。
- **REVERIE 实例接地**：[REALM](../entities/paper-realm-last-3-meter-vln-grounding.md) — Last-3-Meter Grounding Gap、ONS/GS/OracleGS 指标与 REVERIE-AIM（arXiv:2607.03792）。
- **城市尺度方向感知**：[DA-Nav](../entities/paper-da-nav.md) — 商业导航指令 + 图像平面离散 grounding + CoT 恢复；CARLA / Go2 / Kuavo-V（arXiv:2607.11638）。
- **多楼层动态 ObjectNav**：[ZONDA](../entities/paper-zonda.md) — 启发式跨楼层 + 多视角核验 + 行人避障；HM3D-DYNA / TITA（arXiv:2607.21025）。
- **冻结 VLA 可通行 overlay**：[Green for Go](../entities/paper-green-for-go-vla-nav-grounding.md) — SegFormer 绿/红接地 OmniVLA；开环航点误差 vs 长度正则（arXiv:2607.05122）。
- **跨本体像素轨迹残差**：[CrossTracer](../entities/paper-crosstracer.md) — VL-Tracer + CE-Adapter；NaviTrace 45.68（arXiv:2608.06688；待核实开源）。
- **人形物理 VLN 平台**：[HumanoidVLN](../entities/paper-humanoidvln.md) — Isaac Sim 四本体 + FR；933 episode 零样本（arXiv:2608.12860；待开源）。
- **终身学习闭环**：[Arcadia](../entities/paper-arcadia.md) — 自采 + 生成式 USD + 共享 VLN/VLA 骨干 + Sim-from-Real；G1 46/100（arXiv:2512.00076；部分开源）。
- **慢–快像素接口 VLN 基础模型**：[ABot-N1](../entities/paper-abot-n1.md) — CoT + 像素目标统一五任务；ABotN-Bench 城市 Point/POI 闭环；基准开源、权重待发布（arXiv:2607.10383）。

## 参考来源

- [深蓝具身智能：VLN 四范式新手复现推荐](../../sources/blogs/wechat_shenlan_vln_repro_four_paradigms_2026.md) — Habitat/R2R 可跑通开源栈策展
- [WorldVLN 论文摘录（arXiv:2605.15964）](../../sources/papers/worldvln_arxiv_2605_15964.md) — 空中 VLN · 自回归 WAM
- [Uni-LaViRA 论文摘录（arXiv:2605.27582）](../../sources/papers/uni_lavira_arxiv_2605_27582.md) — 零样本统一具身导航
- [REALM 论文摘录（arXiv:2607.03792）](../../sources/papers/realm_last_3_meter_vln_arxiv_2607_03792.md) — REVERIE 末段实例接地与评测鸿沟
- [DA-Nav 论文摘录（arXiv:2607.11638）](../../sources/papers/da_nav_arxiv_2607_11638.md) — 城市尺度方向感知 VLN
- [FSD-VLN 论文摘录（arXiv:2607.08359）](../../sources/papers/fsd_vln_arxiv_2607_08359.md) — 空中长程 VLN 快慢双系统
- [Arcadia 论文摘录（arXiv:2512.00076）](../../sources/papers/arcadia_arxiv_2512_00076.md) — 具身终身学习四段闭环
- [ZONDA 论文摘录（arXiv:2607.21025）](../../sources/papers/zonda_arxiv_2607_21025.md) — 多楼层动态零样本 ObjectNav
- [Green for Go 论文摘录（arXiv:2607.05122）](../../sources/papers/green_for_go_vla_nav_grounding_arxiv_2607_05122.md) — 冻结导航 VLA 的绿/红可通行视觉接地
- [CrossTracer 论文摘录（arXiv:2608.06688）](../../sources/papers/crosstracer_arxiv_2608_06688.md) — 像素轨迹残差跨本体导航
- [HumanoidVLN 论文摘录（arXiv:2608.12860）](../../sources/papers/humanoidvln_arxiv_2608_12860.md) — 人形物理 VLN 仿真与基准
- [ABot-N1 论文摘录（arXiv:2607.10383）](../../sources/papers/abot_n1_arxiv_2607_10383.md) — 慢–快 VLN 基础模型与 ABotN-Bench
- [SceneVerse++ 原始资料归档](../../sources/repos/sceneverse-pp.md)
- Chen et al., *Lifting Unlabeled Internet-level Data for 3D Scene Understanding* (arXiv:2604.01907) — VLN 数据生成与 R2R 实验
- Anderson et al., *Vision-and-Language Navigation* — R2R 任务经典定义（如需溯源基准起源可查阅原文）

## 关联页面

- [Query：具身大模型分类学选型闭环知识链](../queries/embodied-fm-taxonomy-loop.md) — VLN 是五层选型闭环的 **② 空间导航层**：在 VLM 语义之上加空间移动分支（`VLN ⊂ VLA`），仅底盘导航、无力控/操作，任务只需移动时比整套 VLA 更省更可解释
- [Habitat-Sim](../entities/habitat-sim.md) — Meta 高速具身仿真宿主（HM3D/MP3D ObjectNav、VLN-CE 等；MIT；v0.3.4 后不再官方主动维护）
- [VLN 10 篇论文技术地图](../overview/vln-10-papers-technology-map.md)
- [VLN 开源复现：四范式学习路径](../overview/vln-open-source-repro-paradigms.md)
- [WorldVLN（空中 VLN · WAM）](../entities/paper-worldvln-aerial-vln-wam.md)
- [FSD-VLN（空中长程 VLN · 快慢双系统）](../entities/paper-fsd-vln.md) — VLSF + DiT；仿真未开源
- [Arcadia（具身终身学习全生命周期）](../entities/paper-arcadia.md) — 共享 VLN/VLA 骨干 + Real2Sim2Real；部分开源
- [World Action Models（WAM）](../concepts/world-action-models.md)
- [SceneVerse++](../entities/sceneverse-pp.md)
- [HomeWorld](../entities/paper-homeworld-whole-home-scene-generation.md) — 文本到 sim-ready 全屋 3D 与中国住宅平面图数据
- [3D 空间 VQA](../concepts/3d-spatial-vqa.md)
- [Locomotion](locomotion.md)
- [VLA](../methods/vla.md)
- [REALM（Last-3-Meter VLN 实例接地）](../entities/paper-realm-last-3-meter-vln-grounding.md) — REVERIE 末段评测鸿沟与 plug-and-play 精修
- [DA-Nav（方向感知城市尺度 VLN）](../entities/paper-da-nav.md) — 商业导航指令 + 图像平面网格 + CoT 恢复
- [ZONDA（多楼层动态零样本 ObjectNav）](../entities/paper-zonda.md) — 跨楼层启发式规划 + 多视角核验 + 行人预测
- [Green for Go（VLA 导航可通行性视觉接地）](../entities/paper-green-for-go-vla-nav-grounding.md) — 冻结 OmniVLA 的绿/红 overlay；未开源
- [CrossTracer](../entities/paper-crosstracer.md) — 像素轨迹残差做跨本体导航；NaviTrace；待核实开源
- [HumanoidVLN](../entities/paper-humanoidvln.md) — 跨人形本体的 Isaac 物理 VLN 基准；待开源
- [RoamFlow](../entities/paper-roamflow.md) — MeanFlow 一步 image-goal 导航（对照；未开源）
- [S²-VLA（驾驶双流 VLA）](../entities/paper-s-squared-vla.md) — NAVSIM 端到端规划；与 VLN / RobotNav 驾驶 mode 对照
- [Qwen-RobotNav](../entities/qwen-robot-nav.md) — 统一 VLN / ObjNav / 跟踪 / NAVSIM 的导航基座
- [HUMEMBR](../entities/paper-humembr.md) — 人中心长时程记忆 + PersonEQA / Spot 例行找人（对照仿真 EQA）
- [ACE-Brain-0.5](../entities/paper-ace-brain-0-5.md) — 统一具身脑内嵌 VLN-CE（R2R/RxR）与操作/进度接口
- [ABot-N1](../entities/paper-abot-n1.md) — 慢–快 CoT+像素接口 VLN 基础模型；ABotN-PointBench / POIBench 城市闭环基准
- [Project Quiver](../entities/project-quiver.md) — 25 kg 开源户外机架；无官方 VLN 绑定，只作重载真机想象

## 推荐继续阅读

- [机器人论文阅读笔记：MolmoSpaces](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/11_Simulation_Benchmark/MolmoSpaces__A_Large-Scale_Open_Ecosystem_for_Robot_Navigation_and_Manipulation/MolmoSpaces__A_Large-Scale_Open_Ecosystem_for_Robot_Navigation_and_Manipulation.html)
- [机器人论文阅读笔记：Thinking in 360°](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/08_Navigation/Thinking_in_360__Humanoid_Visual_Search_in_the_Wild/Thinking_in_360__Humanoid_Visual_Search_in_the_Wild.html)
- [机器人论文阅读笔记：STATE-NAV](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/08_Navigation/STATE-NAV__Stability-Aware_Traversability_Estimation_for_Bipedal_Navigation_on_Rough_Terrain/STATE-NAV__Stability-Aware_Traversability_Estimation_for_Bipedal_Navigation_on_Rough_Terrain.html)
- Matterport3D / R2R、RxR 等官方基准说明
- NaVILA、RoomTour3D 等「真实视频 + 导航」相关工作（与互联网视频蒸馏路线对照）
