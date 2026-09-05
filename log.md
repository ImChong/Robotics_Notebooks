## [2026-09-05] ingest | sources/blogs/wechat_tencent_world_model_questions_2026-09-05.md — 腾讯科技世界模型访谈；点名论文独立节点；关键页 paper-lpwm / paper-phi-wm-acteffect

- **触发：** 用户指定 <https://mp.weixin.qq.com/s/2DEpiexjwh5O6bBJDXk3LA>，并要求文内论文各有独立非重复详情节点。
- **复用：** [Atlas](wiki/entities/atlas-world-model.md)、[LeVJEPA](wiki/entities/paper-levjepa.md)、[World Labs](wiki/entities/world-labs.md)、[功能分类](wiki/concepts/functional-taxonomy-world-models.md)（#1808 已合入）。
- **步骤 2.5：** LeJEPA **已开源** CC BY-NC；LeWM **已开源** MIT+HF；LpWM **已开源** MIT；ActEffect / Atlas **确认未开源**。
- **关键判断：** 过载词要拆岗位——Atlas 生成 3D、LpWM 改隐几何、ActEffect 把 WM 放进训练反馈。
- **新建：** `paper-lejepa`、`paper-lewm`、`paper-lpwm`、`paper-phi-wm-acteffect`；归档访谈 raw + 三仓 + 光象站。
- **交叉：** 生成式 WM / WAM / INTACT / LeVJEPA / Atlas / 功能分类

## [2026-09-05] ingest | sources/papers/world_model_definition_roadmap_arxiv_2607_06401.md + sources/blogs/worldlabs_functional_taxonomy_world_models.md — Fei-Fei 功能分类 + 上智定义/路线图；确认未开源

- **触发：** 用户指定 <https://arxiv.org/abs/2607.06401v1>、<https://drfeifei.substack.com/p/a-functional-taxonomy-of-world-models>
- **复用：** 已有 [生成式世界模型](wiki/methods/generative-world-models.md)、[WAM](wiki/concepts/world-action-models.md)、[World Labs](wiki/entities/world-labs.md)、[Marble](wiki/entities/marble-world-model.md)、[训练闭环三线](wiki/overview/robot-world-models-training-loop-taxonomy.md)、Awesome 索引卡 [paper-sa-2607-06401](wiki/entities/paper-sa-2607-06401-a-definition-and-roadmap-for-world-models.md)。**另建** 功能分类概念页；定义文 **原地升格** Awesome stub（同一 arXiv 不双实体）。
- **步骤 2.5：** 官方博客 <https://www.worldlabs.ai/blog/taxonomy-of-world-models> 与 arXiv HTML **均无** GitHub / 权重。概念文 + 视角文 → **确认未开源**。Marble 仍是部分开源（Spark），不因本篇改变。
- **关键判断：** 功能分类只描述解码（观测/状态/动作）；上智文补压缩定义与表征轴，并声明 WAM 不是第四实现列。仿真是枢纽；倒金字塔把数据天花板钉在互联网视频多样性。
- **新建：** `functional-taxonomy-world-models`；升格 `paper-sa-2607-06401-a-definition-and-roadmap-for-world-models`（不另建第二实体）；归档 `worldlabs_functional_taxonomy_world_models.md`、`world_model_definition_roadmap_arxiv_2607_06401.md`
- **交叉：** 生成式 WM / WAM / VL* 五家族 / 训练闭环三线 / 具身数据金字塔 / Cosmos 3 / Video-as-Simulation

## [2026-09-05] ingest | sources/sites/cosmos-cookbook.md、Transfer1/2.5 仓与文档 — Cosmos Transfer 族 + Cookbook 入库；已开源但 2.x 有限维护；关键页 cosmos-transfer、paper-cosmos-transfer1、cosmos-cookbook

- **触发：** 用户指定 Cookbook、Transfer2.5 文档/仓、NVIDIA/Cosmos、Transfer1 项目页/仓，并要求自动合并 PR
- **复用：** [`nvidia-cosmos`](wiki/entities/nvidia-cosmos.md)、[`cosmos-3`](wiki/entities/cosmos-3.md)、[`paper-sa-2511-00062`](wiki/entities/paper-sa-2511-00062-world-simulation-with-video-foundation-models-fo.md)、[`nvidia_cosmos`](sources/repos/nvidia_cosmos.md) 已存在。**不另造** Cosmos 3 / Predict2.5 论文页。
- **步骤 2.5：** Transfer1 / Transfer2.5 / Cookbook **已开源**（Apache-2.0 + NVIDIA Open Model License）。三仓 README 均写 **有限维护**，新产品走 [NVIDIA/cosmos](https://github.com/NVIDIA/cosmos)。Cosmos 3 Edge **不支持** V2V transfer。
- **关键判断：** Transfer 卖的是「控制图钉几何、prompt 改外观」；Transfer1 均匀四控 Quality 8.54、64×B200 实时 4.2 s；Transfer2.5-2B 比 7B 小 3.5×。配方从 Cookbook 抄，不要从零猜 JSON。
- **新建：** [`cosmos-transfer`](wiki/entities/cosmos-transfer.md)、[`paper-cosmos-transfer1`](wiki/entities/paper-cosmos-transfer1.md)、[`cosmos-cookbook`](wiki/entities/cosmos-cookbook.md)
- **交叉：** nvidia-cosmos / cosmos-3 / Predict2.5 / Sim2Real / GWM / SO-101 / video-as-simulation

## [2026-09-05] ingest | sources/repos/nvidia_isaac_teleop.md + sources/sites/nvidia-isaac-teleop-docs.md — Isaac Teleop 1.5.x 复核；已开源 Apache-2.0；关键页 isaac-teleop

- **触发：** 用户指定 <https://nvidia.github.io/IsaacTeleop/main/index.html>、<https://github.com/NVIDIA/IsaacTeleop>，并要求自动合并 PR
- **复用：** 已有 [isaac-teleop](wiki/entities/isaac-teleop.md)（2026-06 入库）。**不另造页**，按 1.5.x 文档补 Televiz / 无标记手重建 / LeRobot / 开源边界。
- **步骤 2.5：** 文档与 README 均链 GitHub + PyPI。仓 **已开源** Apache-2.0（376★ / 85 forks）；`isaacteleop` wheel 含 Televiz。CloudXR Web Client **首次 EULA**；手重建另需 MANO + BMC + 第三方 Docker 权重。Upcoming（非 XR 主设备、云仿真遥操作、远程相机流）未交付。
- **关键判断：** Lab 3.x XR 主线，不是「所有 Lab 遥操作」。Televiz 是 XR **合成器**，不是 NuRec 训练产品。no-robot 第一人称采数是一等公民。
- **更新：** `isaac-teleop`；新建归档 `nvidia-isaac-teleop-docs.md`；复核 `nvidia_isaac_teleop.md`
- **交叉：** Teleoperation / Isaac Lab / Isaac Sim / Isaac GR00T / GR00T-WBC / LeRobot / NuRec

## [2026-09-05] ingest | sources/blogs/worldlabs_marble_world_model.md — Marble GA + docs/API；部分开源（Spark/示例，模型闭源）；关键页 marble-world-model

- **触发：** 用户指定 <https://docs.worldlabs.ai/>、<https://marble.worldlabs.ai/>、<https://www.worldlabs.ai/blog/marble-world-model>，并要求自动合并 PR
- **复用：** 已有 [world-labs](wiki/entities/world-labs.md)、[Atlas](wiki/entities/atlas-world-model.md)、[Spark](wiki/entities/spark-3dgs-renderer.md)。**另建** [marble-world-model](wiki/entities/marble-world-model.md)，不把文档塞进公司页。
- **步骤 2.5：** 产品站/博客无权重 GitHub。→ **部分开源**：生成模型闭源 SaaS+World API；Spark 与 Interactive World Examples 开源。
- **关键判断：** Marble 是外观/漫游资产源；collider ≠ sim-ready。对照 Instant NuRec / NuRec（忠实重建）与 SimFoundry（操作孪生）。
- **新建：** `marble-world-model`；归档 `worldlabs_marble_world_model.md`、`worldlabs-docs.md`、`marble-worldlabs-ai.md`
- **交叉：** World Labs / Atlas / Spark / 生成式世界模型 / GS-Playground / NuRec


## [2026-09-05] ingest | sources/papers/instant_nurec_arxiv_2607_14203.md — Instant NuRec 前向 3DGS；部分开源静态 CLI；关键页 paper-instant-nurec / nvidia-nurec

- **触发：** 用户指定 <https://docs.nvidia.com/nurec/>、<https://github.com/NVIDIA/instant-nurec>、<https://research.nvidia.com/labs/sil/projects/instant-nurec/>、<https://arxiv.org/abs/2607.14203>，并要求自动合并 PR
- **复用：** 无同名页。挂接已有 [Sim2Real](wiki/concepts/sim2real.md)、[NuRec USDZ 消费方 Flexion 管线](wiki/entities/flexion-niantic-nvidia-rgb-sim2real-pipeline.md)、[Isaac](wiki/entities/isaac-gym-isaac-lab.md)、[SimFoundry](wiki/entities/paper-simfoundry-real2sim-scene-generation.md)、[GS-Playground](wiki/entities/gs-playground.md)。**另建** 产品实体 [nvidia-nurec](wiki/entities/nvidia-nurec.md)，不把文档与论文塞进同一页。
- **步骤 2.5：** 项目页与文档均链到 GitHub + HF 权重。仓 **部分开源**（Apache-2.0）：静态 PLY + `.sky.npz` 可跑；动态层 / 训练 / AlpaSim 评测不在独立 CLI。NuRec 精修是 NGC 容器 `nre-ga:26.04`，不是本仓。NCore 演示数据门控。
- **关键判断：** 闭环价值是 **换重建不改策略排序**（140 场景 AlpaSim），不是追平 34.38 dB；办公室 360° 扫描走 NuRec 机器人 / Niantic 路径，不要套 Instant AV CLI。
- **新建：** [`paper-instant-nurec`](wiki/entities/paper-instant-nurec.md)、[`nvidia-nurec`](wiki/entities/nvidia-nurec.md)；归档 `sources/papers/instant_nurec_arxiv_2607_14203.md`、`sources/repos/nvidia-instant-nurec.md`、`sources/sites/nvidia-research-instant-nurec.md`、`sources/sites/nvidia-nurec-docs.md`
- **交叉：** Sim2Real / 仿真评测基础设施 / Real2Sim 纵深 / Isaac Sim / Flexion 管线 / SimFoundry / GS-Playground / 生成式世界模型

## [2026-09-05] ingest | sources/papers/robot_juggling_arxiv_2608_26800.md — RAI AthenaZero 真机抛接分钟级记忆学习；确认未开源；关键页 paper-robot-juggling-athenazero

- **触发：** 用户指定 <https://arxiv.org/abs/2608.26800>、<https://arxiv.org/html/2608.26800v1>，并要求自动合并 PR
- **步骤 2.5：** arXiv 与 RAI AthenaZero 博客 **无 GitHub / 数据集** → **确认未开源**
- **关键判断：** 先验连 1 cycle 都完不成仍可作学习正则；MRS 排除 89% 不安全规划；五种三球花样 <5 min；与 [SPD](wiki/entities/paper-spd.md) 仿真预训练路线对照
- **新建：** [`paper-robot-juggling-athenazero`](wiki/entities/paper-robot-juggling-athenazero.md)；归档 `sources/papers/robot_juggling_arxiv_2608_26800.md`、`sources/sites/rai-athenazero-blog.md`
- **交叉：** Contact-Rich Manipulation / Sim2Real / Manipulation / Sumo / SMPC-to-RL / SPD

## [2026-09-05] ingest | sources/papers/spd_corl_2026.md — 复核 SPD 并补挂 arXiv:2608.15917；代码/数据仍待发布

- **触发：** 用户指定 ingest *Pre-training Visual Dexterity in Simulation*（<https://spd.bot/>、<https://arxiv.org/abs/2608.15917>），并要求自动合并 PR
- **步骤 2.5：** 项目页 BibTeX 已更新为 arXiv:2608.15917；仍无 GitHub / spd-vr / spd-75h 下载 → **宣称将开源 / 待核实**
- **更新：** [`sources/papers/spd_corl_2026.md`](sources/papers/spd_corl_2026.md)、[`sources/sites/spd-bot.md`](sources/sites/spd-bot.md)、[`wiki/entities/paper-spd.md`](wiki/entities/paper-spd.md)

## [2026-09-05] ingest | sources/papers/dexholdem_arxiv_2605_18727.md — DexHoldem 真机扑克灵巧基准；已开源 Policy/Skills + TexasPokerRobot；关键页 paper-dexholdem

- **触发：** 用户指定 <https://dexholdem.github.io/Dexholdem/>。
- **复用：** 无同名页。挂接已有 [Manipulation](wiki/tasks/manipulation.md)、[DexBench](wiki/entities/dexbench.md)、[评测枢纽](wiki/overview/hub-embodied-eval-benchmark.md)、[π₀](wiki/methods/π0-policy.md)、[VLA](wiki/methods/vla.md)、[Diffusion Policy](wiki/methods/diffusion-policy.md)。**不另造** 方法页。
- **步骤 2.5：** 项目页 Resources 同时链到 Policy 仓、Skills 仓与 HF 数据集。→ **已开源可运行**。两仓截至入库日未附 LICENSE；数据 CC BY 4.0（约 378 GB）。
- **关键判断：** 测的不是扑克智能。报 SPSR 不要只报 TCR；芯片 pull 与筹码字典才是硬项；感知 field-wise 高不等于 Overall。
- **新建：** [`paper-dexholdem`](wiki/entities/paper-dexholdem.md)；归档 `sources/papers/dexholdem_arxiv_2605_18727.md`、`sources/sites/dexholdem-github-io.md`、`sources/repos/dexholdem-policy.md`、`sources/repos/dexholdem-skills.md`、`sources/datasets/texaspokerrobot.md`
- **交叉：** Manipulation / DexBench / 评测选型闭环 / 接触丰富操作 / π₀ / VLA / Diffusion Policy / 灵巧采数指南

## [2026-09-05] ingest | sources/repos/project-quiver.md、sources/sites/arrowair-quiver.md — 接入 Arrow Quiver 25kg 开源 ArduPilot 机架；已开源 CERN-OHL-S；关键页 project-quiver；对照 PX4 / MAVSDK / Crazyflie

- **触发：** 用户指定 <https://github.com/Arrow-air/project-quiver>、<https://arrowair.com/quiver>。
- **复用：** 无同名页。挂接已有 [多旋翼栈总览](wiki/overview/multirotor-simulation-planning-control-stack.md)、[PX4](wiki/entities/px4-autopilot.md)、[MAVSDK](wiki/entities/mavsdk.md)、[Crazyflie](wiki/entities/crazyflie-firmware.md)、[Betaflight](wiki/entities/betaflight.md)。**不另造** ArduPilot 固件实体页。
- **步骤 2.5：** 项目页写明 CAD / KiCad / ArduPilot 配置 / 装配文档在 GitHub。仓 **已开源** CERN-OHL-S-2.0（372★ / 96 forks）。飞控固件在上游 ArduPilot，不在本仓。无学习权重。
- **关键判断：** 这是 **25 kg 作业机架**，不是 PX4 分叉、也不是室内微四轴。补齐总览里「ArduPilot 未列入本批」的机架层。
- **新建：** [`project-quiver`](wiki/entities/project-quiver.md)；归档 `sources/repos/project-quiver.md`、`sources/sites/arrowair-quiver.md`；机构 `arrow-air`。
- **交叉：** 多旋翼栈总览 / PX4 / MAVSDK / Crazyflie / Betaflight / AERIS-10 / 平滑路径 / VLN

## [2026-09-05] ingest | sources/repos/multi-agent-cad.md — 接入清华 IEI Lab MAC；已开源 MIT；关键页 multi-agent-cad；对照 CAD Skills / Text-to-CAD

- **触发：** 用户指定 <https://github.com/Pan-Chera/Multi-Agent-CAD>。
- **复用：** 无同名页。挂接已有 [CAD Skills](wiki/entities/cad-skills.md)、[文字生成 CAD](wiki/concepts/text-to-cad.md)、[FreeCAD MCP](wiki/entities/freecad-mcp.md)、[GenCAD](wiki/entities/gencad.md)。**不另造** 方法页。
- **步骤 2.5：** 无独立项目页。实验室 [IEI Lab](https://maureenzou.github.io/lab.html) 未挂下载。GitHub 仓 **已开源** MIT（930★）。无训练权重；需自备 OpenAI 兼容 API。`packages/cadpy` vendored 自 CAD Skills。
- **关键判断：** 核心是结构化状态传递 + 确定性翻译器，不是「又一个写 build123d 的 agent」。同一套 P1–P10 上自报 116× 少 token。无 URDF/打印链。
- **新建：** [`multi-agent-cad`](wiki/entities/multi-agent-cad.md)；归档 `sources/repos/multi-agent-cad.md`
- **交叉：** CAD Skills / Text-to-CAD / FreeCAD MCP / GenCAD / text-to-cad-tools

## [2026-09-05] ingest | sources/repos/tennis-vision.md — 接入 HarshTomar Tennis-Vision；已开源 MIT；关键页 tennis-vision；对照 Roboflow Sports / 场线检测 / 坐标后处理

- **触发：** 用户指定 <https://github.com/HarshTomar1234/Tennis-Vision>。
- **复用：** 无同名页。挂接已有 [Roboflow Sports](wiki/entities/roboflow-sports.md)、[场线检测](wiki/methods/soccer-field-line-detection.md)、[感知坐标后处理](wiki/concepts/perception-coordinate-postprocessing.md)、[Ultralytics](wiki/entities/ultralytics.md)、[Kalman](wiki/formalizations/kalman-filter.md)、[MediaPipe](wiki/entities/mediapipe.md)、[SAM 3D Body](wiki/entities/sam-3d-body.md)。**不另造** 方法页。
- **步骤 2.5：** 无独立项目页。GitHub 仓 **已开源** MIT（67★）。TrackNet 权重不随仓再分发；可选 SAM 3D Body 为 Meta SAM License，默认关。球场微调权重在 HF `Coddieharsh/tennis-court-keypoints`。
- **关键判断：** 广播网球离线分析，不是机载闭环。教学点：单应只在地板平面有效、重投影自洽 ≠ 场地有效、出点率 ≠ 定位精度、RTS/更大 YOLO 已测过并丢掉。
- **新建：** [`tennis-vision`](wiki/entities/tennis-vision.md)；归档 `sources/repos/tennis-vision.md`
- **交叉：** Roboflow Sports / 场线检测 / 坐标后处理 / 场线定位流水线 / Ultralytics / Kalman / MediaPipe / SAM 3D Body / Humanoid Soccer / 目标检测

## [2026-09-05] ingest | sources/repos/nvidia-warp.md、sources/sites/nvidia-warp-docs.md、sources/repos/mujoco-warp.md — 接入 NVIDIA Warp 与 MuJoCo Warp 并接到 Newton；关键页 nvidia-warp、mujoco-warp

- **触发：** 用户指定 <https://github.com/NVIDIA/warp>、<https://github.com/google-deepmind/mujoco_warp>、<https://nvidia.github.io/warp/stable/>。
- **复用：** 无同名实体（`paper-warp-whole-body-retargeting` 是无关论文）。[Newton](wiki/entities/newton-physics.md) / [mjlab](wiki/entities/mjlab.md) / [MJX](wiki/entities/mujoco-mjx.md) 已存在，**不另造引擎页**。
- **步骤 2.5：** Warp **已开源** Apache-2.0（`warp-lang`，~7.1k★，文档 1.17.0）。MJWarp **已开源** Apache-2.0（`mujoco-warp`，~1.4k★）。二者均可 `pip` 安装；MJWarp 文档在 MuJoCo readthedocs `mjwarp/`。
- **关键判断：** Warp 是 JIT 计算层，`warp.sim` **已弃用**，机器人物理走 Newton。MJWarp 是 GPU drop-in MuJoCo、Newton 主要刚体后端；**Warp AD 尚未接通**（issue #500）。PGS / noslip / PLUGIN / IMPLICITFAST midpoint 有缺口。JAX 训练经 MJX + Playground；PyTorch 经 mjlab 或 Isaac Lab `feature/newton`。
- **新建：** [`nvidia-warp`](wiki/entities/nvidia-warp.md)、[`mujoco-warp`](wiki/entities/mujoco-warp.md)
- **交叉：** Newton / MuJoCo / MJX / mjlab / Brax / Playground / Isaac Lab / Omniverse / 仿真器选型 / 训练栈分层

## [2026-09-05] ingest | sources/repos/newton-physics.md、sources/sites/newton-physics-docs-overview.md、NVIDIA Cosmos 一手族 — Newton 八求解器再核 + Cosmos 1.0/2.5/3 平台对齐；关键页 newton-physics、nvidia-cosmos

- **触发：** 用户指定 <https://github.com/newton-physics/newton>、<https://newton-physics.github.io/newton/stable/guide/overview.html>，并要求一并找到英伟达 Cosmos 模型一手资料。
- **复用：** [`newton-physics`](wiki/entities/newton-physics.md)、[`cosmos-3`](wiki/entities/cosmos-3.md) 已存在（2026-05 / 2026-06）。**不另造** Newton 页。Cosmos 1.0 / Predict2.5 的 Awesome 索引页 **原地升格**（同一 arXiv 只允许一个节点）。
- **步骤 2.5：** Newton **已开源** Apache-2.0（~5.6k★）。Cosmos 3 **已开源**（[NVIDIA/cosmos](https://github.com/NVIDIA/cosmos) ~11.7k★ + [cosmos-framework](https://github.com/NVIDIA/cosmos-framework) + HF Cosmos3）。Predict2.5 **已开源但有限维护**。产品页 FAQ：Omniverse/Newton 仿真视频 → Cosmos Transfer。
- **关键判断：** Newton stable Overview 新增 **Kamino / ImplicitMPM / Style3D**，接触改为 **`CollisionPipeline.collide`**。Cosmos 3 现含 **4B Edge**、4-step 蒸馏、SGLang / TensorRT-LLM、framework SFT。Predict2.5 PAI-Bench I2W Overall **0.810**。解析仿真 ≠ 像素 WFM。
- **新建：** [`nvidia-cosmos`](wiki/entities/nvidia-cosmos.md)；一手源 `cosmos_wfm_arxiv_2501_03575`、`cosmos_predict25_arxiv_2511_00062`、`nvidia_cosmos_framework`、`nvidia_cosmos_predict25`、`sites/nvidia-cosmos`
- **交叉：** Newton / Cosmos 3 / Omniverse / Isaac Lab / generative-world-models / WAM / 仿真器选型

## [2026-09-05] ingest | sources/papers/luna_arxiv_2606_31981.md — LUNA（ECCV 2026）LBS-free 隐式 2D 驱动 3DGS；项目页 GitHub 占位未开源；关键页 paper-luna-universal-3d-human-animation

- **触发：** 用户指定 <https://arxiv.org/abs/2606.31981>、<https://penghtyx.github.io/LUNA/>。
- **复用：** 无同名页。挂接已有 [SMPL-X](wiki/concepts/smpl-x.md)、[GVHMR](wiki/entities/gvhmr.md)、[4DAnyone](wiki/entities/paper-4danyone.md)、[Face Anything](wiki/entities/paper-face-anything-4d-face-reconstruction.md)、[UMA](wiki/entities/paper-uma.md)、[人形训练数据管线](wiki/queries/humanoid-training-data-pipeline.md)。
- **步骤 2.5：** 项目页 GitHub 按钮为注释占位（`href="#"`），无 HF / 权重。→ **宣称将开源 / 截至 2026-09-05 未列可用链接。** 不建 `sources/repos/`。机构 `hkust` / `meta` 已注册。
- **关键判断：** 推理丢掉 LBS，训练仍蒸馏软结构；主场是 Cloth10K PSNR **22.07** 与 MSJ **0.0032**（相对 MV-LHM 约 **4.5×**）。NeuMan 与前馈打平。输出是 splat，**不是**关节指令。
- **新建：** [`paper-luna-universal-3d-human-animation`](wiki/entities/paper-luna-universal-3d-human-animation.md)；归档论文 + 项目页
- **交叉：** SMPL-X / GVHMR / 4DAnyone / Face Anything / UMA / 人形训练数据管线 / 2D→3D Gap

## [2026-09-05] ingest | sources/repos/nvlabs-simfoundry.md — SimFoundry 官方仓再核：部分开源 Apache-2.0；关键页已存在

- **触发：** 用户指定项目页 / [NVlabs/SimFoundry](https://github.com/NVlabs/SimFoundry) / arXiv:2606.28276。
- **复用：** [`paper-simfoundry-real2sim-scene-generation`](wiki/entities/paper-simfoundry-real2sim-scene-generation.md)（2026-07-03 已 complete）。**不另造页**。
- **步骤 2.5（2026-09-05）：** 官方仓已公开（2026-08-14 V0）。→ **部分开源**。A 重建 / B cousins / C OmniGibson 加载可跑；README 写明 **数据生成与策略训练未随仓**。HF `nadunRanawaka1/simfoundry-assets`。导出是 **OmniGibson JSON**，不是论文写的 Isaac Lab。
- **关键判断：** 不要再写「未挂公开仓」。能复现场景孪生，不能复现 Pearson 0.911 训练协议。24 GiB 必须 `low_vram`；全量约 250 GB + 门控权重 + Gemini。
- **新建：** [`nvlabs-simfoundry`](sources/repos/nvlabs-simfoundry.md)
- **交叉：** Sim2Real / Manipulation / 仿真评测基础设施 / GEAR / BEHAVIOR-1K / depth-real2sim

## [2026-09-05] ingest | sources/papers/pointdit_arxiv_2607_02515.md — PointDiT 像素空间点图扩散；已开源 google-research/pointdit；关键页 paper-pointdit

- **触发：** 用户指定 <https://haofeixu.github.io/pointdit/>。
- **复用：** 无同名页。挂接已有 [单目深度综述](wiki/entities/paper-monocular-depth-estimation-survey.md)、[2D→3D 提升 Gap](wiki/concepts/2d-to-3d-semantic-lifting-gap.md)、[感知栈选型闭环](wiki/queries/robot-perception-stack-selection-loop.md)、[Flex-π](wiki/entities/paper-flex-pi.md)、[ADM-BA](wiki/entities/paper-adm-ba.md)。
- **步骤 2.5：** 项目页 + [google-research/pointdit](https://github.com/google-research/pointdit)（Apache-2.0）+ [HF haofeixu/pointdit](https://huggingface.co/haofeixu/pointdit)。仓内 `main.py` / `scripts/demo_*.sh` / `eval_*.sh` / `train_stage*.sh` 可跑。→ **已开源**。DINOv3 权重门控，不随 checkpoint。
- **关键判断：** 去掉 VAE 与混合回归头；x-prediction 是硬门槛（v-pred 崩）。H 型 4 步 Rel\(_d\) **2.75** / BF1 **10.49**，1 步 **72 ms**。室外 KITTI/DIODE/ETH3D 仍落后 MoGe。输出仿射不变，不是 metric。机构注册 `kesai`。
- **新建：** [`paper-pointdit`](wiki/entities/paper-pointdit.md)；归档项目页 + 官方仓
- **交叉：** 单目深度综述 / 2D→3D Gap / 感知选型闭环 / Flex-π / ADM-BA

## [2026-09-05] ingest | sources/papers/hil_hybrid_imitation_learning_arxiv_2505_12619.md — TOG 2026 HIL 再核：heading 任务 + 官方未开源 + 一作非官方 G1 仓；关键页 paper-hil-hybrid-imitation-learning

- **触发：** 用户指定 HIL TOG PDF / Peng 组项目页 / Code(G1) Hybrid-Motion-Imitation。
- **复用：** [`hil-hybrid-imitation-learning`](wiki/methods/hil-hybrid-imitation-learning.md) 与对比页已存在（预印本 *Diverse Parkour Skills from Videos*）。**不另造方法页**，升格 TOG 定稿并补实体页。
- **步骤 2.5：** [jiashunwang.github.io/HIL](https://jiashunwang.github.io/HIL/) 与 [xbpeng.github.io/projects/HIL](https://xbpeng.github.io/projects/HIL/index.html) **无 GitHub**。→ 官方 TOG 角色动画 **确认未开源**。一作 [jiashunwang/Hybrid-Motion-Imitation](https://github.com/jiashunwang/Hybrid-Motion-Imitation)（Apache-2.0）自标 unofficial extension of GfR + HIL，G1 箱攀/搬箱可跑，**无 AMP 判别器**。
- **关键判断：** 统一观测（无相位）；跑酷 skill acc **0.66** / DTW **0.31** / 完成率 0.74；heading facing **0.97**。完成率低于纯任务基线不是失败。机构 `cmu` / `nvidia` / `sfu` 已注册。
- **新建：** [`paper-hil-hybrid-imitation-learning`](wiki/entities/paper-hil-hybrid-imitation-learning.md)；归档项目页 + G1 仓
- **交叉：** AMP / DeepMimic / ASE / MTRG / ZEST / holosoma / Locomotion

## [2026-09-05] ingest | sources/sites/ac5113-milo.md — MILO 项目页再核加深（UT Austin / ECCV 2026）；确认已开源 MIT；关键页已存在

- **触发：** 用户指定 <https://ac5113.github.io/MILO/>
- **复用：** [`paper-milo`](wiki/entities/paper-milo.md)（2026-08-31 已 complete）。**不另造页**。
- **步骤 2.5（2026-09-05 再核）：** 项目页 Code 链到 [ac5113/MILO](https://github.com/ac5113/MILO)；README 与 `docs/PIPELINE.md` 可跑。→ **已开源（MIT）**。SMPL-H / SAM 3 / SAM 3D 权重门控，不随仓。
- **关键判断：** 增益来自 **联合 LRM 脚手架** 不是拟合器。InterCap 有模板联合 PA-CD **7.45 cm**（PICO 10.33，无 GT 接触）。HODome / IMHD **无模板更好**（6.38 / 6.98）。模板对瓶/杯有用，对行李箱/椅/伞会变差。核心 **344 s/图**，瓶颈在 LRM 189 s。输出是网格，**不是**关节指令。
- **加深：** 编译 Table 1–7 + 运行时 + 仓库步骤；挂接 PICO / 重定向 / Manipulation / ECHO。
- **交叉：** PICO / motion-retargeting-pipeline / Manipulation / ECHO / CLAP 九篇地图

## [2026-09-05] ingest | sources/papers/4danyone_arxiv_2608_20335.md — 4DAnyone 单目随意视频→重建级多视角/4DGS；已开源 ant-research/4DAnyone；关键页 paper-4danyone

- **触发：** 用户指定 <https://4danyone.github.io/>。
- **步骤 2.5：** 项目页给出 [ant-research/4DAnyone](https://github.com/ant-research/4DAnyone)（Apache-2.0）与 [HF AntResearch/4DAnyone](https://huggingface.co/AntResearch/4DAnyone)。→ **已开源**。建 `sources/repos/4danyone.md`。机构 `zju` / `robbyant` / `ant-group` / `hkust` / `cuhk` 均已注册。
- **关键判断：** 几何走 GVHMR（depth-buffered 40/308），外观走 Wan2.2-TI2V-5B + RCP/TCR，再抬 4DGS。DNA-Rendering 4DGS PSNR **24.15**。仓内 4090 峰值 **<24 GB**、121 帧约 **27 s**、Turbo **5.58×**。Roadmap 仍写 4DGS 方法未接线；仓文档是 nerfstudio **3DGS**。输出是多视角/点绘，**不是**关节指令。
- **交叉：** GVHMR / SMPL-X / Face Anything / Motion Retargeting / EasyMocap

## [2026-09-05] ingest | sources/papers/fwbc_vla_arxiv_2609_03889.md + MINERVA/BRIDGE 再核 — FWBC-VLA 无传感器接触 WBC 新建实体；MINERVA 补 CPU 5.1ms；BRIDGE CAD 部分开源、规格 80cm

- **触发：** 用户指定 MINERVA（GitHub / arXiv:2609.03715 / HF）、BRIDGE（arXiv:2609.03497 / 项目页）、FWBC-VLA（arXiv:2609.03889）。
- **复用：** [`paper-minerva-libero`](wiki/entities/paper-minerva-libero.md)、[`paper-bridge-humanoid`](wiki/entities/paper-bridge-humanoid.md) 已存在（2026-09-04 九篇盘点）。**不另造页**，再核加深。
- **新建：** [`paper-fwbc-vla`](wiki/entities/paper-fwbc-vla.md)；机构注册 `zust`。
- **步骤 2.5：** MINERVA **已开源**（`k1000dai/MINERVA` + HF）。BRIDGE **部分开源**（项目页 `.stp` CAD；控制/BOM 待录用）。FWBC-VLA **确认未开源**（无项目页/仓；WL&Arm 宣称将公开）。
- **关键判断：** MINERVA 0.54M L1 CPU **5.1 ms/chunk**；BRIDGE 规格以表为准 **80 cm / 12.5 kg / 21 DoF**，跟踪 SR **94.83**；FWBC-VLA 擦白板终段 **64%**、开门 **52%**，FI 与 BC 必须拆开读。
- **交叉：** VLA / Manipulation / Loco-Manipulation / WBC / Contact Estimation / Contact-Rich / Humanoid Locomotion / FM-VLA / 九篇地图

## [2026-09-04] ingest | sources/papers/agile_perceptive_traversal_arxiv_2608_29769.md — Sparse 3D Traversal 再核加深（ETH RSL 猴架）；确认未开源；关键页已存在，补枢纽挂接与电池/热/射线锥

- **触发：** 用户指定 <https://nemantor.github.io/sparse-3d-traversal-website/>
- **复用：** [`paper-agile-perceptive-traversal-sparse-3d`](wiki/entities/paper-agile-perceptive-traversal-sparse-3d.md)（2026-09-02 已 complete）。**不另造页**。
- **步骤 2.5（2026-09-04 再核）：** 项目页 **无** GitHub / HF / Zenodo；作者账号 `nemantor` 无对应仓。→ **确认未开源**。勿建 `sources/repos/`。
- **关键判断：** 14/15 真机成功绑在电池压降 / 热积分 / E1R 射线锥，而不是更大 DR。AME-2 13.8k 打过 1.31M MLP；盲学生不可行。跳上峰值 2.11 kW、电压最低 34.7 V。
- **加深：** 编译 Table IV 全行、电池/热参数、两套 LiDAR 噪声；挂接 [`stair-obstacle-perceptive-locomotion`](wiki/tasks/stair-obstacle-perceptive-locomotion.md) 枢纽与 PHP / ANYmal Parkour 对照。

## [2026-09-04] ingest | sources/papers/lucida_r2s_arxiv_2608_30821.md — Lucida 组合式 Real-to-Sim（ByteDance Seed×PKU×ZJU）；项目页仅 arXiv/HF，确认未开源；关键页 paper-lucida-r2s

- **触发：** 用户指定 <https://lucida-r2s.github.io/>
- **步骤 2.5：** **确认未开源**。项目页 `project.resources` 仅 arXiv + Hugging Face papers；论文未承诺发代码；`github.com/lucida-r2s` 为 Pages 托管。勿建 `sources/repos/`。
- **关键判断：** 精度后移到 GizmoAct 闭环放置；主读数是 R2S-Scene F-Score 0.924 与 CA-1M ADD-SB@0.05 83.4%，不是策略 Pearson。漏检不可事后补。
- **新建：** [`paper-lucida-r2s`](wiki/entities/paper-lucida-r2s.md)；归档论文 + 项目页
- **交叉：** Sim2Real / CRISP / SimFoundry / Agentic Real2Sim / R2S-EGO / Awesome-R2S2R / Manipulation

## [2026-09-04] ingest | sources/repos/easymocap.md — EasyMocap（zju3dv）：无标记多视角/互联网视频 SMPL 工具箱；已开源（非商业科研许可）；ZJU-MoCap 协议申请

- **触发：** 用户指定 <https://github.com/zju3dv/EasyMocap>
- **步骤 2.5：** **已开源（科研许可）**。文档站 <https://chingswy.github.io/easymocap-public-doc/> 与 GitHub 互指；CLI `emc` + `apps/demo/mv1p.py` / `mvmp.py` 可跑。LICENSE 仅教育/研究/非营利，商用须邮件 `xwzhou@zju.edu.cn`。SMPL/MANO 权重不随仓。ZJU-MoCap（LightStage + Mirrored-Human）需协议 PDF + 邮件申请。iMocap 多视频特定动作 README 仍标 Coming soon。
- **关键判断：** 有标定外参走 EasyMocap；野外单目世界轨迹走同组 [GVHMR](wiki/entities/gvhmr.md)。输出 `Rh` ≠ 官方 `global_orient`，接 GMR 前必须转换。不是机器人关节指令。
- **新建：** [`easymocap`](wiki/entities/easymocap.md)；归档 repo + 文档站
- **交叉：** GVHMR / FreeMoCap / MAMMA / motion-retargeting-pipeline / SMPL-X / GMR

## [2026-09-04] ingest | sources/papers/crosstracer_arxiv_2608_06688.md — CrossTracer 像素轨迹残差跨本体导航；NaviTrace 45.68；项目页无代码（宣称开源/待核实）；关键页 paper-crosstracer

- **触发：** 用户指定 CrossTracer（arXiv:2608.06688）；鹏城实验室 / 南科大 / 创新投资研究院 / 苏州大学
- **步骤 2.5：** 项目页 <https://lilduckkk.github.io/CrossTracer-Nav/> **无** GitHub / HF。NaviTrace 对照表把 CrossTracer-8B 标 Open-Source ✓；作者仓 `Lilduckkk` 仅有 `whisper_turtlesim`。→ **宣称开源 / 待核实**。勿建 `sources/repos/`。
- **关键判断：** 语义提案与本体可通行要拆开；去掉 CE-Adapter 掉 23.12 分。NaviTrace 45.68 ≠ VLN-CE SR。goal-pose 63.91 不能与语言-only 横比。
- **新建：** [`paper-crosstracer`](wiki/entities/paper-crosstracer.md)；机构注册 `innovation-investment-research-institute`
- **交叉：** VLA / VLN / 四范式 / hub-cross-embodiment / Green for Go / HumanoidVLN / NaVILA

## [2026-09-04] ingest | sources/blogs/wechat_humanoid_zhiyan_inertia_closedloop_calib_2026-08-26.md — 人形智研院「出厂体检」：惯量标定必须闭环；PRIME 复用并补强（已开源）；KILVO 复用；Calib3R/CAL²M 待升格

- **触发：** 用户指定 <https://mp.weixin.qq.com/s/sl06FnCPmUh6GilJuK-xEQ>
- **步骤 2.5：** 文内主案例 **PRIME 已开源**（[well-robotics/PRIME](https://github.com/well-robotics/PRIME)，BSD-3-Clause；项目页 [PRIME-project](https://jkangkjr.github.io/PRIME-project/)）。KILVO **代码待开放**（既有 complete 页）。Calib3R / CAL²M 为相机外参 / SLAM 轴，**不造页**。
- **关键判断：** 单关节台架覆盖不了分布式质量与终身漂移；量产要运动学 / 惯量 / IMU 零偏 / 足底力四张单子闭环，并绑机身序列号。标定误差是地板，大模型是上限。ISO 13482 **不**写成已核实的出厂惯量强制项。
- **新建：** [`humanoid-closed-loop-inertia-calibration`](wiki/concepts/humanoid-closed-loop-inertia-calibration.md)；PRIME 论文/项目页归档
- **复用：** [`prime-system-id`](wiki/entities/prime-system-id.md)（HMI draft → complete，**不另造** `paper-prime`）；[`paper-kilvo`](wiki/entities/paper-kilvo.md)
- **交叉：** SysID / 关节执行器辨识 / 连杆-转子惯量 / 物理保真度 / Sim2Real 闭环 / 接触力旋量 / Crocoddyl

## [2026-09-04] ingest | sources/papers/levjepa_arxiv_2608_27395.md — LeVJEPA（arXiv:2608.27395）：LeJEPA+SIGReg 视频预训练；已开源 MIT（module.py/权重 CC BY-NC）；关键页 paper-levjepa

- **触发：** 用户指定论文 <https://arxiv.org/abs/2608.27395>、项目页 <https://levjepa.github.io/>
- **步骤 2.5：** **已开源**。项目页互链 [MLO-lab/LeVJEPA](https://github.com/MLO-lab/LeVJEPA)（2026-08-28 建仓）与 HF `galilai-group/LeVJEPA-VideoMix-Large`。主体 MIT；`module.py` 与发布权重 **CC BY-NC 4.0**（改编 Meta V-JEPA）。默认复现走公开 Walking Tours，论文主表 K710 20% 需自备。
- **关键判断：** 坍塌对策从 EMA/predictor 换成 SIGReg；95% 均匀 token drop 是增强不是近似；block-causal 几乎不掉点。本文**不是**规划 WM（无 AC / MPC / 真机）。
- **新建：** [`paper-levjepa`](wiki/entities/paper-levjepa.md)；机构注册 `dkfz` / `goethe` / `mila` / `umontreal` / `ami-labs`
- **交叉：** V-JEPA 2、V-JEPA 2.1、WCM、generative-world-models、world-model-physics-fidelity-outputs、video-as-simulation


## [2026-09-04] ingest | sources/repos/sciencediscovery.md + sources/sites/openjiuwen-com.md — openJiuwen 平台与 ScienceDiscovery 科研工作台；已开源 Apache-2.0；双托管 GitHub/AtomGit

- **触发：** 用户指定 <https://atomgit.com/openJiuwen/sciencediscovery>、<https://github.com/openJiuwen-ai/sciencediscovery>、<https://www.openjiuwen.com>
- **步骤 2.5：** **已开源**。GitHub `openJiuwen-ai/sciencediscovery` 与 AtomGit `openJiuwen/sciencediscovery` 双托管；Apache-2.0；中英文档完整。官网为平台入口（WorkSwarm / 协同工程），ScienceDiscovery 以代码仓为准。模型与文献全文不随仓。
- **关键结构：** Node API `:4310` 进程内 `native-agent`（非 LangChain）+ Runner `:4311` Bubblewrap；科学 MCP（PubMed/arXiv/PDB 等）延迟披露；CAS 溯源。仓内 `skills/` 仅 2 个内置包，README「300+ Skills」按生态叙事读。
- **新建：** [`sciencediscovery`](wiki/entities/sciencediscovery.md)、[`openjiuwen`](wiki/entities/openjiuwen.md)；机构注册 `openjiuwen`
- **未升格：** JiuwenSwarm / JiuwenSymbiosis / agent-core 仅在平台页登记（一次一条）
- **交叉：** AI Auto-Research、Hermes、OpenClaw、DeepSeek Harness、autoresearch、MCP、index.md

## [2026-09-04] ingest | sources/blogs/wechat_embodied_station_9_papers_open_source_2026-09-04.md — 具身智能小站 9 篇开源盘点；9/9 独立节点（GIFT/AdaRoboVLG/IRWOZ 2.0/Network Design/MINERVA/FailBench/XR-2/BRIDGE/ARTiS）；0 复用。MINERVA+NetworkDesign 已开源；IRWOZ/FailBench/XR-2/ARTiS 部分开源；GIFT/AdaRoboVLG 待发布；BRIDGE 仓待核实


## [2026-09-04] ingest | sources/papers/smoothrl_arxiv_2608_29768.md + 深蓝六条路线公众号 — SmoothRL 项目页再核已上线仍未开源；六条窟窿写回 query

- **触发：** 用户指定 <https://www.astribot.com/research/SmoothRL>、arXiv:2608.29768、<https://mp.weixin.qq.com/s/k7CR03ZHaSQRMVvutpSnCg>
- **复用：** [`paper-smoothrl`](wiki/entities/paper-smoothrl.md)（2026-09-02 已 complete）。**不要**把公众号误读成 SmoothRL 通稿——正文是深蓝《拆完这六条技术路线…》。
- **步骤 2.5（SmoothRL，2026-09-04 再核）：** 项目页 **已上线**（Nuxt；2026-09-02 为 404）。Hero + 9 段任务对比视频在 OSS `media/smoothrl/`。页内 **无** GitHub / HF → **确认未开源**。作者：Guang Gao\* / Yuxuan Nong\* / Baifu Huang；Lead：Jianan Wang。
- **关键数字：** S1 三任务 250 ep：投掷 39%→94%、笔帽 8%→83%、开箱 30%→90%；RMS acc/jerk −52%/−47%。
- **新建：** 公众号写回 [`embodied-six-routes-holes`](wiki/queries/embodied-six-routes-holes.md)；raw + blog 归档。文内 Helix / Index / ER 2 / RL Token 等 **映射已有节点，0 重复造页**。
- **交叉：** VLA / action-chunking / Manipulation / RL / Lumo-2 / Philia / Figure / Gemini / 五大范式 / 五层选型闭环

## [2026-09-04] structural | roadmap/depth-icl.md — 新增第 23 条纵深路线「ICL（具身上下文学习）」并接入路线视图/首页/主路线

- **触发：** 维护者要求"增加一个 ICL（in-context learning）的纵深"。仓库已有 [`robot-in-context-learning`](wiki/concepts/robot-in-context-learning.md) 概念页与 [四路线对比](wiki/comparisons/wam-ttt-robottt-stellavla-zero-wam-embodied-icl.md)，但没有对应的学习路线页。
- **新建：** [`roadmap/depth-icl.md`](roadmap/depth-icl.md)（Stage 0–5）：判别边界（映射选择 / 状态估计 / 映射本身，只有第三类是真 ICL）→ 示范表征与 action tokenizer → 遥操作示范线（配对数据 + 防"复制最近 chunk"捷径）→ 人类视频线（无动作标签 + embodiment gap）→ 零梯度上下文 vs 快权重 TTT vs 记忆增强的选型与每步推理预算 → 规模涌现与评测协议。
- **排序位：** 按各方向起点里程碑历史顺序，置于人形拳击（2017）与 BFM（2018）之间，起点取 **One-Shot Imitation Learning，NeurIPS 2017**（综述归档参考文献 1）。
- **接入点：** `docs/depth-filters.js`（DEPTH_ORDER / HUB_IDS / FILTERS / META，22→23 条，emoji 🧩）、`docs/index.html` 首页按钮与纵深路线计数、`docs/main.js` 折叠文案与 hero 兜底值、`README.md` / `index.md` / `roadmap/README.md` / `roadmap/motion-control.md` 四处清单、22 条既有 depth 页的「其它纵深路径」互链，以及 ICL 概念页与四路线对比页的回链。
- **验证：** `tests/test_depth_filters.py` 期望长度 23→24；`make ci-preflight` 12/12 通过、`lint_wiki` 0 errors；全量 `pytest` 429 passed；`eslint docs/main.js` 通过。

## [2026-09-04] structural | docs/roadmap.html + docs/main.js — 路线详情页本库超链接悬停浮窗

- **改动：** [`docs/roadmap.html`](docs/roadmap.html) 增加 `#detail-inline-link-tooltip` 与 `graph-tooltip.js`；[`docs/main.js`](docs/main.js) 在路线页正文 / 知识地图 / 阶段相关项上复用详情页内链 hover 卡片
- **清单：** [`docs/checklists/frontend-optimization-v1.md`](docs/checklists/frontend-optimization-v1.md) Phase 3
- **验证：** [`scripts/verify_roadmap_inline_link_preview.cjs`](scripts/verify_roadmap_inline_link_preview.cjs)

## [2026-09-04] ingest | sources/papers/host_arxiv_2607_20033.md + 再核 Imitator Game / GEN-1.5 / S1 / Zero-WAM — 五条人视频 one-shot / ICL 节点：HOST 新建；其余复用已有 complete 页

- **触发：** 用户指定五条独立不重复详情节点：[1] Imitator Game（imitator-game.github.io）[2] GEN-1.5（generalistai.com/blog/gen-1.5）[3] Skild S1（skild.ai/blogs/s1）[4] arXiv:2607.20033 [5] Zero-WAM（arXiv:2608.26103）
- **复用（同一 arXiv / 已有实体不重造）：**
  - [1] [`paper-imitator-game`](wiki/entities/paper-imitator-game.md)（2608.22301）— 2026-09-04 再核仍 **已开源（MIT）** + HF IG-10K
  - [2] [`generalist-gen15-one-shot`](wiki/entities/generalist-gen15-one-shot.md) — 再核仍 **确认未开源**
  - [3] [`skild-s1`](wiki/entities/skild-s1.md) — 再核 `github.com/skild-ai` 仍 0 公开仓，**确认未开源**
  - [5] [`paper-zero-wam`](wiki/entities/paper-zero-wam.md)（2608.26103）— 仓仍仅 README/`docs`，**仍待发布**（计划 2026-09-15 前）
- **新建：** [4] HOST（*Human-to-robot One-Shot Skill AcquisiTion*）— [`paper-host-one-shot-human-video`](wiki/entities/paper-host-one-shot-human-video.md)
- **步骤 2.5（HOST）：** **已开源** — [CGuangyan-BIT/HOST](https://github.com/CGuangyan-BIT/HOST) 对齐/耦合/策略训练入口 + HF [`Guangyan/HOST`](https://huggingface.co/Guangyan/HOST)（MIT）。仓根无统一 LICENSE；193k 真机轨迹未随仓；`eval_openloop.sh` 不指挥真机。
- **关键数字：** 双臂 ARX R5；单视频平均 **29 s**；八任务 **62%**；相对语言零样本约 **+45 pt**；已掌握保留约 **99%**；相对 50 条 SFT 演示少 50×、快约 507×。
- **交叉：** ICL 概念表、IL / Manipulation / WAM、四路线对比、跨具身枢纽、Foundation Policy；四条复用页回链 HOST
## [2026-09-04] ingest | sources/papers/unified_robot_learning_survey_arxiv_2609_03927.md — TMLR 表征–VLA–世界模型统一综述；独立节点，不造表内空壳

- **触发：** 用户指定 <https://arxiv.org/abs/2609.03927>
- **步骤 2.5：** **确认未开源 / 无配套实现**（arXiv-only，无项目页/GitHub）
- **关键页：** 新建 [`paper-unified-robot-learning-survey`](wiki/entities/paper-unified-robot-learning-survey.md)；交叉 VLA / WM / WAM / 五大分类 / 选型闭环 / 14 篇阅读路线
- **机构注册：** `fujitsu-research-america`、`fujitsu`；CMU 已有

## [2026-09-04] ingest | sources/papers/umr_unified_motion_retargeting_arxiv_2609_02134.md — UMR：学习点云对应的统一人形重定向；独立节点，代码待发布

- **触发：** 用户指定 *Unified Motion Retargeting for Humanoids with Learned Point Cloud Correspondence*（香港科技大学广州校区 / 诺亦腾 / 汉阳大学 / 香港科技大学 / 香港大学；[arXiv:2609.02134](https://arxiv.org/abs/2609.02134)）
- **步骤 2.5：** **待发布**。所谓项目链接即 arXiv；HTML/PDF 未列 GitHub。同团队 [AdaPT 项目页](https://humanoidtennis.github.io/AdaPT/) 写 MoCap「UMR coming soon」，AdaPT 仓不含本实现。
- **关键页：** 新建 [`paper-umr-unified-motion-retargeting`](wiki/entities/paper-umr-unified-motion-retargeting.md)；交叉 [Motion Retargeting](wiki/concepts/motion-retargeting.md)、[hub](wiki/overview/hub-motion-retargeting.md)、[GMR](wiki/methods/motion-retargeting-gmr.md)、[OmniRetarget](wiki/entities/paper-hrl-stack-03-omniretarget.md)、[OmniContact](wiki/entities/paper-omnicontact-humanoid-loco-manipulation.md)、[BeyondMimic](wiki/methods/beyondmimic.md)、[SONIC](wiki/methods/sonic-motion-tracking.md)、[AdaPT](wiki/entities/paper-adapt.md)
- **消歧：** 勿与 AdaMorph（2601.07284）或 PALUM（2601.07272）混页

## [2026-09-04] ingest | sources/papers/{wm_loco,focus,safe_stop} + 两篇公众号 — WM-LOCO/FOCUS 新建独立节点，Safe-Stop 复用；Infra 全景与 SLAM 沙龙写回 overview/query

- **触发：** 用户指定 <https://mp.weixin.qq.com/s/qVqpihnA4GezsE2MIJjKDw>、三篇论文（WM-LOCO / Safe-Stop / FOCUS）须独立非重复详情节点，以及 <https://mp.weixin.qq.com/s/0MUtW7aaPPltT9oO3SUtSg>
- **步骤 2.5：** WM-LOCO **待发布**（`m0puppet.github.io/wm-loco` 仅项目页，Code coming soon）；Safe-Stop **待发布**（2026-09-04 再核项目页仍无 GitHub，**复用**既有 complete 页）；FOCUS **确认未开源**（无项目页/官方仓；勿与 StefanoFerraro/FOCUS 世界模型仓混淆）
- **关键页：** 新建 [`paper-wm-loco`](wiki/entities/paper-wm-loco.md)、[`paper-focus-foot-observation-confidence`](wiki/entities/paper-focus-foot-observation-confidence.md)；复用 [`paper-safe-stop-humanoid`](wiki/entities/paper-safe-stop-humanoid.md)；地图 [`g1-foothold-safe-stop-focus-technology-map`](wiki/overview/g1-foothold-safe-stop-focus-technology-map.md)；Infra 长文写回 [`embodied-infra-2026-panorama`](wiki/overview/embodied-infra-2026-panorama.md)（30 条参考文献映射已有节点，缺页待升格）；SLAM 沙龙写回 [`slam-second-spring-embodied`](wiki/queries/slam-second-spring-embodied.md)
- **机构注册：** `bupt` 北京邮电大学（BUPT）、`soochow` 苏州大学（Soochow University）

## [2026-09-03] lint | wiki/entities/paper-{demomimic,openvla,tapvid-mv,physics-consistent-hrc-benchmark}.md — 清空全量 lint 信息型预警：补 1 篇「对比」段 + 2 条评测枢纽双向回链 + 1 条陈旧声明时效化 + 1 个缺页误报归档

- **触发：** 跑一遍全量知识库 lint（`python3 scripts/lint_wiki.py`），失败项 0，余 **5 条信息型预警**（三段式缺「对比」×1、评测基准枢纽回链缺失×2、陈旧声明×1、缺页概念候选×1）
- **三段式补「对比」：** [`paper-demomimic`](wiki/entities/paper-demomimic.md) 新增 `## 与其他工作对比`（DexMachina\* 95.8→21.72% / HERMES\* 93.7→3.37% 的 sim-to-real 崩塌对照、CHORD 的 wrench 空间 vs 本页接触局部几何、ADEPT 大规模预训练+distill vs 单次示范、VisualMimic 全身 loco-manip vs 桌面多指、多示范 IL 的 per-object 重采集）——均由页内已有事实归并成表，未引入新论断；另注明 \* 基线数字来自项目页复现而非原论文自报
- **评测基准闭环：** [`paper-physics-consistent-hrc-benchmark`](wiki/entities/paper-physics-consistent-hrc-benchmark.md) 落 ③ 层接触安全切面（名义 SR 经区域+力安全筛查 72.9%→56.4%，与 SoftVTBench 同类）；[`paper-tapvid-mv`](wiki/entities/paper-tapvid-mv.md) 落 ② 层前置感知切面（多视角长时 3D 对应，30+ baseline 未接近解决、瓶颈在几何恢复）；枢纽页 [`embodied-eval-benchmark-selection-loop`](wiki/queries/embodied-eval-benchmark-selection-loop.md) 同步补 `related` / `sources` / 「关联页面」条目并 bump `updated`
- **陈旧声明时效化：** [`paper-openvla`](wiki/entities/paper-openvla.md) 的「开源 7B 可达闭源 SOTA 约 85%+」补上 **2024 年原文发表时** 的时间限定与「发表时快照、横比前回榜核协议」的读法，并链到 [`vla-sota-leaderboard`](wiki/entities/vla-sota-leaderboard.md) 作为相对位次的复核入口——该数字是发表时相对 RT-2-X 的对照基线值，不是当前位次
- **缺页误报归档：** `LCM` 属「缩写 slug ≠ 页面 stem」，已由 [`concepts/lcm-basics.md`](wiki/concepts/lcm-basics.md)（UDP 组播 pub/sub + 类型描述语言）+ [`concepts/ipc-inter-process-communication.md`](wiki/concepts/ipc-inter-process-communication.md) + [`comparisons/ros2-vs-lcm.md`](wiki/comparisons/ros2-vs-lcm.md) 覆盖，按 ethercat / ros2 / urdf 惯例登记进 `MISSING_CONCEPT_COVERED_ELSEWHERE`
- **验证：** lint「✅ 所有检查通过！」（0 失败 / 0 信息型，覆盖率 3407/3407）；`make ci-preflight` 全绿（导出质量 12/12）；`eval_search_quality` 通过；`ruff check` + `ruff format --check` 通过；`npm run lint:js` 通过；`pytest` 429 passed / 608 subtests
- **派生物：** 本会话 `git fetch --unshallow` 后重跑 `ci-preflight`，`wiki-activity` 口径为 git（非 log.md 兜底），故 `exports/` `docs/exports/` 统计随本次提交（图谱 30934 → 30940 边）
- **合并 main 后复跑：** 合入 origin/main（PR #1769：新建 `concepts/motion-control-policy-evaluation-metrics.md` + 第 22 条纵深路线 `roadmap/depth-embodied-eval.md`）后二次全量 lint 仍为 **0 失败 / 0 信息型**，新页未带入新预警；冲突仅在 `log.md`（顶部两侧各自 prepend，按倒序时间线保留双方）与徽章/统计派生物（取 main 侧后由 `ci-preflight` 重算，图谱 3434 节点 / 31065 边）

## [2026-09-03] ingest | wiki/concepts/motion-control-policy-evaluation-metrics.md — 新建「运控模型评测指标」概念页：四组指标坐标系（跟踪误差 / 命令跟随 / model-based 求解 / 硬件部署）、可比性四项对齐（参考表示·rollout 记账·终止准则·指标实现）、HumanScore 偏好对齐、MDS/MID/DSJE 难度分层、仿真→半实物→真机三段验收；归纳自 HumanTracker(2608.13555)/MDS(2512.07248)/Barkour(2305.14654)/HumanoidBench/TrackerLab 与 locomotion 任务页

## [2026-09-03] structural | roadmap/depth-embodied-eval.md — 补 Stage 4「运控模型测评」（Succ+MPJPE 联报、参考表示/rollout 记账/终止准则四项对齐、难度分层 MDS/MID/DSJE、locomotion 速度/CoT/摔倒率/硬件安全裕度、MPC-WBC 求解耗时与约束违反、仿真→半实物→真机三段验收）；原 Stage 4/5 顺延为 5/6，路线更新为 Stage 0–6

## [2026-09-03] structural | roadmap/depth-embodied-eval.md — 新增第 22 条纵深路线「具身模型测评」（Stage 0–5：四层边界 → MLLM 认知 → 世界模型保真 → 策略成功率 → sim↔real 校准 → 评测基建/榜单治理）；同步 README/index/motion-control/roadmap README 与 docs 路线视图（depth-filters.js 新增 embodied-eval key、首页按钮与计数 21→22）

## [2026-09-03] fix | wiki/entities/gpufree.md、wiki/comparisons/china-gpu-cloud-platforms.md — 按平台开发者勘误（issue #1767）修正算力自由定位、去除 Docker/星级打分、补容器桌面 3.0

- **触发：** 平台开发者 `@RiskerFalor` 在 [issue #1767](https://github.com/ImChong/Robotics_Notebooks/issues/1767) 指出早期页面（含 AI 生成成分）有误
- **修正点：** ①定位改为「产品重心是机器人仿真」；②「Docker 容器」→「容器实例/容器化平台」；③文档成熟度星级 → 文档风格描述（打分不可跨受众线性比较）；④容器桌面非 VNC，3.0（2026-09-03）GPU 加速 + 手柄；⑤RT Core 由 NVIDIA 卡型决定，不是平台差异点；⑥补赛事/高校合作与价格（标注为厂商自述）；⑦补「公共模型存储未必是加分项」误区
- **核对：** 官网首页已有「合作赛事」入口；`docs/guide/quick_start.html` 仍写 noVNC-vulkan、无 3.0 章节 → 相关条目标注「开发者自述 / 待官方文档更新回填」
- **归档：** [`sources/sites/gpufree.md`](sources/sites/gpufree.md) 新增「开发者勘误」小节原样保留一手说法

## [2026-09-03] ingest | sources/blogs/wechat_embodied_station_8_papers_open_source_2026-09-03.md — 具身智能小站 8 篇开源论文速览；新建 7 实体 + 系统可靠性技术地图；DemoMimic 复用

- **触发：** 用户指定公众号 <https://mp.weixin.qq.com/s/-UqboKHaoG5eu79u9XQU0w>；要求每篇独立非重复详情节点
- **步骤 2.5：** HINT/SA-WAM/Safe-Stop **待发布**（项目页无 GitHub）；Physics HRC **部分/待发布**（匿名预览 benchmark Coming soon）；TAPVid-MV **部分开源**（基准/Perpetua 经项目页）；TTI Survey + PACT **已开源**；DemoMimic **复用** complete 页并补 arXiv:2609.01938
- **关键页：** 新建 `paper-hint-robot-manipulation`、`paper-sa-wam`、`paper-physics-consistent-hrc-benchmark`、`paper-safe-stop-humanoid`、`paper-tapvid-mv`、`paper-test-time-intelligence-survey`、`paper-pact-hrc-action-admission`；地图 [`open-source-system-reliability-8-papers-technology-map`](wiki/overview/open-source-system-reliability-8-papers-technology-map.md)

## [2026-09-03] ingest | sources/sites/ipc_primary_refs.md — POSIX/Linux IPC 一手资料（pipe/shm/mq/unix man-pages、OSTEP、APUE、Beej）；新建 wiki/concepts/ipc-inter-process-communication.md；交叉 OS 基础、运控中间件指南、hub 系统工程/通信

## [2026-09-03] ingest | sources/papers/humanoid_touch_dream.md — Humanoid Touch Dream（IROS 2026）；升格 canonical 论文实体；WBC 已开源、策略/遥操作 on-going

## [2026-09-03] ingest | sources/papers/embodiedskills_arxiv_2609_01281.md — EmbodiedSkills AgentLoop；GitHub 已开源

## [2026-09-03] ingest | sources/papers/demomimic_stanford_2026.md — DemoMimic 单次示范灵巧泛化；开源待发布（Code/arXiv coming soon）

## [2026-09-03] ingest | sources/papers/{parcelstow,adapt_2609_00677,facet_0} — G1 时间鲁棒性 / 文本运动 / 动作–受力联合预测；Facet-0 仓库仍占位

- **触发：** 用户指定三组：ParcelStow（GitHub + arXiv:2609.01453 + HF `cenwerem/parcelstow`）、ADAPT（arXiv:2609.00677 + wuyan01 项目页）、Facet-0（GitHub + arXiv:2609.01596 + 项目页 + ManuFacet-1K）
- **步骤 2.5：**
  - **ParcelStow：已开源** — `coenwerem/parcelstow` Apache-2.0 + HF 示范/权重；Isaac Lab G1 L6；无真机。v1 只报包裹插入。
  - **ADAPT（ETH）：确认未开源** — 项目页无 GitHub/HF。勿与网球 AdaPT（2608.20087 / `noitom-robotics/AdaPT`）混淆。
  - **Facet-0：部分开源** — ManuFacet-1K 已上 HF；`PINE-Lab-NTU/FACET` 仅 README（Code coming soon）；模型卡为空。纠正 2026-09-02 浅入库的「已开源」。
- **关键页：** [`paper-parcelstow`](wiki/entities/paper-parcelstow.md) 深化；新建 [`paper-adapt-text-driven-humanoid`](wiki/entities/paper-adapt-text-driven-humanoid.md)；[`paper-facet-0`](wiki/entities/paper-facet-0.md) 深化并改开源边界
- **主张：** ParcelStow \(r=2\) 专家 84% / ACT 53%；ADAPT 仿真 Success 0.984、真机走慢跑满；Facet-0 五任务 82% vs RECAP 15%，16→38→82 三段互补

## [2026-09-02] ingest | sources/sites/x2robot-twindex.md — 自变量 TwinDEX 三指外骨骼–同构手共设计无本体采数；确认未开源；升格 wiki/entities/twindex.md

- **触发：** 用户指定项目主页 <https://x2robot.com/pages/twindex>；要求自动合并 PR
- **步骤 2.5：** 项目页（Next.js）与英文页 **未列** GitHub / HF / arXiv；BibTeX *Coming soon*。GitHub `twindex` 检索 0 仓。**确认未开源。** 勿把同机构 [XRZero-G0](https://github.com/X-Square-Robot/XRZero-G0)（VR+夹爪，arXiv:2604.13001）或 [sdk_hand](https://github.com/X-Square-Robot/sdk_hand)（五指）写成 TwinDEX 复现入口。
- **关键页：** [`twindex`](wiki/entities/twindex.md)；交叉 teleoperation / 灵巧采数指南 / 手套 vs 视觉 / mimic U1 / DEUX / HandUMI
- **主张（项目页）：** 3 指 9 DoF（7 主动+2 被动）；采数吞吐 **5.3×** 真机遥操作；robot-free 与 on-robot data-efficiency 重叠；化学实验约 24–25 子动作（页内 24/25 不一致）

## [2026-09-02] ingest | sources/papers/rise_adaptive_imagination_arxiv_2608_20430.md — 酷哇 RISE 自适应想象 WAM（arXiv:2608.20430）；代码 MIT + CounterDrive 已开、权重未发；勿与 OpenDriveLab RISE 混淆

- **触发：** 用户指定论文 *RISE: Adaptive Imagination for World Action Models*；机构 COWARobot / 上交 / 河海；项目页 https://cowarobot-ai.github.io/RISE/ ；代码 https://github.com/COOWAI/RISE ；数据集 https://huggingface.co/datasets/COWARobot/CounterDrive
- **步骤 2.5：** 项目页链 GitHub + HF。**已开源（部分）**：`COOWAI/RISE` MIT 训练/评测代码与七段手动链；HF CounterDrive tar 约 31.5 GB（CC-BY-NC-ND-4.0）。**权重未发**。GitHub README（约 2026-08-13）仍写论文/数据未发布，**以项目页 + arXiv:2608.20430 + HF 实际文件为准**。
- **关键页：** [`paper-rise-adaptive-imagination-wam`](wiki/entities/paper-rise-adaptive-imagination-wam.md)；同名对照 [`paper-sa-2602-11075` OpenDriveLab RISE](wiki/entities/paper-sa-2602-11075-rise-self-improving-robot-policy-with-compositio.md)；交叉 WAM / latent-imagination / V-JEPA 2 / X-Foresight
- **机构注册：** `cowarobot` 酷哇科技（COWARobot）、`hohai` 河海大学（Hohai University）；`sjtu` 已有

## [2026-09-02] ingest | sources/papers/imitator_game_arxiv_2608_22301.md — The Imitator Game 再核：MIT 仿真仓 + HF IG-10K/Assets 已开源；深化 L0–L3 / Arena 数字；真机评测仍走申请

- **触发：** 用户指定 arXiv:2608.22301 + 项目页 https://imitator-game.github.io/ ；2026-08-30 已有浅入库
- **步骤 2.5（再核）：** [imitator-game/The-Imitator-Game](https://github.com/imitator-game/The-Imitator-Game) **MIT**；HF `IG-10K-Dataset` / `IG-10K-Assets`；ModelScope `Zhouxunzhe/*`。真机部署无上传接口。
- **关键页：** [`paper-imitator-game`](wiki/entities/paper-imitator-game.md)；交叉 IL / VLA / Manipulation / 评测选型闭环

## [2026-09-02] ingest | sources/blogs/wechat_embodied_ai_lab_vla_wm_reading_roadmap_2026-09-02.md — 14 篇独立节点：新建 10（RT-1/RT-2/OpenVLA/π0/Octo/DP/CLIP/ACT/SPOC/DINOv2）+ 复用 4（LaDi-WM/DreamDojo/RISE/PointWorld）；Octo arxiv 迁出方法页；13/14 已开源，RT-2 官方训练未开源

## [2026-09-02] ingest | sources/blogs/worldlabs_atlas_omni_world_model.md — World Labs Atlas omni 世界模型（相机可控生成、稀疏 3D 重建、Real-to-Sim）；早期访问未开源；新增 wiki/entities/atlas-world-model.md；交叉 world-labs、generative-world-models

- **触发：** 用户指定 <https://www.worldlabs.ai/blog/atlas>
- **步骤 2.5：** 博客与官网 **未列** GitHub/HF → **早期访问（未开源）**；输出 splat 与 Marble/Spark 同表征
- **关键页：** [`atlas-world-model`](wiki/entities/atlas-world-model.md)、[`world-labs`](wiki/entities/world-labs.md)

## [2026-09-02] ingest | sources/blogs/wechat_embodied_station_7_papers_contact_manipulation_2026-09-02.md — 7 篇开源论文独立节点：Facet-0/ParcelStow/Dual-MP/Depth Survey/ADM-BA/Peg-in-Bench/NP-Throw；Peg-in-Bench 仓库待发布

## [2026-09-02] ingest | sources/papers/hydrogym_nature_s41586_026_10917_6.md — HydroGym 主动流控 RL 平台（Nature 2026）；arXiv:2512.17534 开放全文 + GitHub/HF 已开源；通道→翼型零样本减阻

- **触发：** 用户指定 Nature s41586-026-10917-6，并查询 arXiv 等开放获取入口
- **开放获取：** **arXiv:2512.17534**（推荐）；Nature 读者链 rdcu.be/fBvqa；SI PDF 可下
- **步骤 2.5：** `dynamicslab/hydrogym` + `dynamicslab.github.io/hydrogym` → **已开源（MIT）**；HF 环境 checkpoint
- **关键页：** [`paper-hydrogym`](wiki/entities/paper-hydrogym.md)；交叉 [`gymnasium`](wiki/entities/gymnasium.md) / [`sim2real`](wiki/concepts/sim2real.md)

## [2026-09-02] ingest | sources/papers/ame_arxiv_2506_09588.md, sources/repos/ame_locomotion_sii_fusc.md — AME 补 Science Robotics 10(105) eadv3604 正式发表与 SII-FUSC 社区复现；复核 UCAG-P 仍待发布

- **触发：** 用户一次 ingest 含两篇：（1）*One Policy, Many Embodiments* / UCAG-P；（2）*Attention-Based Map Encoding* / AME（arXiv:2506.09588）
- **UCAG-P：** 已于 2026-08-28 入库；**步骤 2.5 复核（2026-09-02）** — [public-bots.github.io/UCAG-P](https://public-bots.github.io/UCAG-P) 与 [Public-BOTs/UCAG-P](https://github.com/Public-BOTs/UCAG-P) 仍 **code release coming soon**，无训练入口
- **AME 更新：** 补 *Science Robotics* **10**(105) **eadv3604**（DOI [10.1126/scirobotics.adv3604](https://doi.org/10.1126/scirobotics.adv3604)）；官方 **无训练代码**，Zenodo [10.5281/zenodo.14499786](https://doi.org/10.5281/zenodo.14499786)；新建社区复现归档 [SII-FUSC/AME_Locomotion](https://github.com/SII-FUSC/AME_Locomotion)（**非官方**，G1 + Isaac Lab）
- **关键页：** [`paper-ame-attention-based-map-encoding`](wiki/entities/paper-ame-attention-based-map-encoding.md)、[`paper-ucag-p`](wiki/entities/paper-ucag-p.md)

## [2026-09-02] ingest | sources/papers/anyworld_arxiv_2608_29242.md — AnyWorld：动作–相机–具身因子化 egocentric WM，无配对人–机数据重组 GR1/IRON rollout；UniT 适配 +4.8pp/+35pp；项目页未开源

- **触发：** 用户指定 arXiv:2608.29242（小鹏机器人 / NTU / A*STAR IAIC / ZJU / CUHK）
- **步骤 2.5：** 项目页 `xpeng-robotics.github.io/anyworld` **未列** GitHub/权重 → **未开源**
- **关键页：** [`paper-anyworld`](wiki/entities/paper-anyworld.md)；交叉 [`paper-unit-unified-physical-language`](wiki/entities/paper-unit-unified-physical-language.md)

## [2026-09-02] ingest | sources/papers/agile_perceptive_traversal + blind_dexterity + smoothrl — 三篇 arXiv:2608.29769/29487/29768：猴架 LiDAR 感知穿越 / G1 纯本体操作 / 异步 VLA 在线 RL；均未开源（Blind Dexterity 待发布）

- **触发：** 用户指定三篇论文 + 项目页
- **步骤 2.5：** Sparse-3D 项目页无代码；BlindDexterity **Code to be released**；SmoothRL 项目页 **404**、arXiv 无仓库
- **关键页：** [`paper-agile-perceptive-traversal-sparse-3d`](wiki/entities/paper-agile-perceptive-traversal-sparse-3d.md)、[`paper-blind-dexterity`](wiki/entities/paper-blind-dexterity.md)、[`paper-smoothrl`](wiki/entities/paper-smoothrl.md)

## [2026-09-01] ingest | sources/papers/fixanything_arxiv_2608_23549.md — FixAnything：Wan2.1 LoRA 统一修复四类 3D 渲染伪影 + Flow-DPO 几何偏好；推理与 HF 权重已开源

- **触发：** 用户指定 arXiv:2608.23549（ECCV 2026，CMU）
- **步骤 2.5：** GitHub `kvuong2711/fix-anything` + HF LoRA **已开源（推理）**；SFT/Flow-DPO 训练脚本未发布
- **关键页：** [`paper-fixanything`](wiki/entities/paper-fixanything.md)；交叉 [`paper-wan-video`](wiki/entities/paper-wan-video.md) / [`paper-r2s-ego`](wiki/entities/paper-r2s-ego.md)

## [2026-09-01] ingest | sources/blogs/wechat_embodied_station_7_papers_open_source_system_loop_2026-09-01.md — 7 篇开源系统闭环盘点；新建 6 实体 + 复用 Motus2；6/7 已开源

## [2026-09-01] ingest | sources/repos/starter_kit_racing.md — mrdoob/starter-kit-racing Kenney Godot→JS 街机赛车；crashcat + GridMap 编辑器；MIT 已开源；升格 starter-kit-racing 实体并补赛车景观

- **触发：** 用户指定 <https://github.com/mrdoob/starter-kit-racing>
- **步骤 2.5：** GitHub + Pages **MIT 已开源**；Kenney CC0 资产在仓内
- **关键页：** [`starter-kit-racing`](wiki/entities/starter-kit-racing.md)；交叉赛车漂移景观 / drive-game / nordschleife-racer

## [2026-09-01] ingest | sources/papers/motus2_arxiv_2608_30237.md — Motus2 自进化 GWM 灵巧操作；项目页未列代码仓；升格 wiki/entities/paper-motus2.md

- **触发：** 用户指定 <https://motus-robotics.github.io/motus2/>
- **步骤 2.5：** 项目页与 `motus-robotics` 组织 **无** 训练/推理仓库 → **未开源**（仅静态站 `motus-robotics.github.io`）
- **关键页：** [`paper-motus2`](wiki/entities/paper-motus2.md)；交叉 Motus 索引 / Motubrain / `world-action-models`

## [2026-09-01] ingest | sources/papers/embodied_robot_manipulation_fm_survey_2512_22983.md — 基础模型时代操作综述（规划×学习双轴）入库；Awesome 列表已开源；升格 wiki/entities/paper-embodied-manipulation-foundation-models-survey.md

## [2026-09-01] ingest | sources/blogs/wechat_zanehub_robot_structural_modal_analysis.md — 接入 Zane Hub 结构模态解读并升格 robot-structural-modal-analysis 概念页

## [2026-09-01] ingest | sources/papers/vlact_arxiv_2608_27550.md + stay_seated + aero_hand_open — VLAct/Stay Seated/Aero Hand Open 三篇 ingest；VLAct 与 Aero 已开源，Stay Seated 未开源

## [2026-09-01] ingest | sources/papers/pamor_arxiv_2608_28213.md — PAMoR：UCL 运动学闭式 V-A + 可组合潜扩散，G1 实时情感全身运动；感知 Top-1 0.384；截至入库日未开源

- **触发：** 用户指定 arXiv:2608.28213（UCL Chengxu Zhou 组）
- **步骤 2.5：** arXiv 为唯一官方入口；**未列 GitHub/权重** → **未开源**
- **关键页：** [`paper-pamor`](wiki/entities/paper-pamor.md)；交叉 [`textop`](wiki/entities/paper-loco-manip-161-022-textop.md) / [`diffusion-motion-generation`](wiki/methods/diffusion-motion-generation.md) / [`hiaer`](wiki/entities/paper-notebook-hierarchical-intention-aware-expressive-motion-g.md)

## [2026-09-01] ingest | sources/papers/ssr_arxiv_2605_30770.md — SSR 复核查补核心信息/工程实践/源码时序图；项目页确认未开源

## [2026-09-01] ingest | sources/papers/robot_in_crib_sensorimotor_contingency_scirobotics_2026.md — iCub 摇篮 mobile paradigm；四策略挑战「动得更多」；仿真代码已开源、真机未列 GitHub

- **触发：** 用户指定 Science Robotics 2026（DOI:10.1126/scirobotics.aed4106）；全文无 arXiv，用 CTU 新闻 + 姊妹 arXiv:2504.17939 + 官方仿真仓核查
- **步骤 2.5：** 仿真 [`mobile-paradigm-model`](https://github.com/ctu-vras/mobile-paradigm-model) **已开源**；真机七条件 **未列代码**；媒体见 CTU Google Drive
- **关键页：** [`paper-robot-in-crib-sensorimotor-contingency`](wiki/entities/paper-robot-in-crib-sensorimotor-contingency.md)

## [2026-08-31] ingest | sources/blogs/wechat_meiri_zhineng_embodied_icl_four_papers_2026-08-31.md — 四篇具身 ICL 纵横向解读；对比页 + StellaVLA 新建 + RoboTTT arXiv 补录

- **触发：** 用户指定 <https://mp.weixin.qq.com/s/vIUalf3vZI3AV-HWSVruew>，与 WAM-TTT / RoboTTT / StellaVLA / Zero-WAM 四论文 ingest 一并合并
- **关键页：** [`wam-ttt-robottt-stellavla-zero-wam-embodied-icl`](wiki/comparisons/wam-ttt-robottt-stellavla-zero-wam-embodied-icl.md)；[`paper-stellavla-structured-icl-vla`](wiki/entities/paper-stellavla-structured-icl-vla.md)

## [2026-08-31] ingest | sources/papers/stellavla_arxiv_2608_11671.md + robottt arXiv 补录 — 四篇部署期适应轴：StellaVLA 结构化 ICL 新建；RoboTTT 补 arXiv:2607.15275；WAM-TTT / Zero-WAM 已入库并交叉链接

- **触发：** 用户指定 WAM-TTT（2607.06988）、RoboTTT（2607.15275）、StellaVLA（2608.11671）、Zero-WAM（2608.26103）；要求自动合并 PR
- **步骤 2.5：** StellaVLA **确认未开源**（仅 VLA-Arena 开源基准）；RoboTTT arXiv 已发布、**仍无公开代码**；Zero-WAM **待发布**（2026-09-15 前）
- **关键页：** [`paper-stellavla-structured-icl-vla`](wiki/entities/paper-stellavla-structured-icl-vla.md)（新建）；[`paper-robottt`](wiki/entities/paper-robottt-test-time-training-vla-context.md)（arXiv 元数据）；交叉 [`robot-in-context-learning`](wiki/concepts/robot-in-context-learning.md) / [`vla`](wiki/methods/vla.md) / [`manipulation`](wiki/tasks/manipulation.md)

## [2026-08-31] ingest | sources/papers/abot_n1_arxiv_2607_10383.md — ABot-N1 慢–快 VLN 基础模型；开源 ABotN-Bench 评测栈，模型权重未发布

## [2026-08-31] ingest | sources/papers/wikiskill_arxiv_2608_27454.md — WikiSkill 持久 wiki 驱动 skill 进化；官方未开源；交叉 LLM Wiki / Superpowers / Darwin

## [2026-08-31] ingest | sources/papers/safeflow_arxiv_2603_23983.md — SafeFlow 物理引导整流流 + 三阶段安全门；G1 相对 TextOp 成功率 98.5%；截至入库日未开源

- **触发：** 用户指定 arXiv:2603.23983（三星 Future Robot AI Group）；要求自动合并 PR
- **步骤 2.5：** [项目页](https://hanbyelcho.info/safeflow/) 方法/视频/量化表齐全；**未列 GitHub/权重** → **未开源**
- **关键页：** [`paper-loco-manip-161-104-safeflow`](wiki/entities/paper-loco-manip-161-104-safeflow.md)（自 161 策展索引升格 arXiv 深读）

## [2026-08-31] ingest | sources/blogs/wechat_embodied_station_clap_9_papers_open_source_2026-08-31.md — 9 篇独立详情节点（新建 6 / 复用 CLAP+FlashVLA+Riemann）；MILO/AlloEgo/MistyPilot 已开源

- **触发：** 用户指定 <https://mp.weixin.qq.com/s/J62q2IVvvBDyT_8OTR9KZQ>；要求每篇论文独立非重复详情节点
- **步骤 2.5：**
  - **MILO** — **已开源** `ac5113/MILO`
  - **CLAP / FlashVLA** — **已开源**（复用既有 complete 页）
  - **ESRP** — **未开源** 项目页
  - **Riemann-1.0** — **确认未开源**（复用既有页）
  - **TrapVLA** — **未开源** Pages 站
  - **ViTaR** — **待发布** Code Coming soon
  - **AlloEgo-VLM / MistyPilot** — **已开源**
- **关键页：** [clap-cross-embodiment-vla-wm-9-papers-technology-map](wiki/overview/clap-cross-embodiment-vla-wm-9-papers-technology-map.md)；新建 `paper-milo`、`paper-esrp`、`paper-trapvla`、`paper-vitar`、`paper-alloego-vlm`、`paper-mistypilot`


## [2026-08-31] ingest | sources/papers/temporalflow_vla_arxiv_2608_26821.md — TemporalFlow-VLA 物理接地执行历史；π₀.₅ 双 query + 离线机器人表面流；截至入库日未列官方代码

## [2026-08-31] ingest | sources/repos/jwasham_coding_interview_university.md — 接入 Coding Interview University 并升格实体页；交叉更新具身面试题库与 qqfly 指南

## [2026-08-31] ingest | sources/repos/hkuds_deeptutor.md — 接入 HKUDS DeepTutor（已开源）并升格 wiki/entities/deeptutor.md；交叉更新 CLI-Anything/Hermes/OpenClaw/Agent Reach


## [2026-08-30] ingest | sources/courses/nvidia_learn_openusd.md — Learn OpenUSD 官方课纲：Stage→Composition→资产结构→Instancing→Data Exchange；GitHub 已开源

- **触发：** 用户指定 <https://docs.nvidia.com/learn-openusd/latest/index.html>；要求自动合并 PR
- **步骤 2.5：** 课纲源码 [LearnOpenUSD](https://github.com/NVIDIA-Omniverse/LearnOpenUSD) **已开源**；在线文档免费；OpenUSD Development Certification 为 **付费认证**
- **关键页：** [`wiki/entities/nvidia-learn-openusd.md`](wiki/entities/nvidia-learn-openusd.md)

## [2026-08-30] ingest | sources/blogs/nvidia_isaac_lab_{ur10e_industrial_assembly,spot_locomotion}_sim2real.md — NVIDIA 官方 Isaac Lab sim2real 博客：UR10e IndustReal 装配 + Spot Researcher Kit 平地 velocity

- **触发：** 用户指定两篇 NVIDIA Developer Blog；要求自动合并 PR
- **步骤 2.5：**
  - **UR10e 装配** — Isaac Lab / Isaac ROS **已开源**；UR Direct Torque **早期访问**；本篇完整训练包 **待发布**
  - **Spot locomotion** — Isaac Lab + [spot-rl-example](https://github.com/boston-dynamics/spot-rl-example) **已开源**；joint SDK / Kit 硬件 **需 BD 渠道**
- **关键页：** [`nvidia-isaac-lab-ur10e-industrial-assembly-sim2real`](wiki/entities/nvidia-isaac-lab-ur10e-industrial-assembly-sim2real.md)、[`nvidia-isaac-lab-spot-locomotion-sim2real`](wiki/entities/nvidia-isaac-lab-spot-locomotion-sim2real.md)

## [2026-08-30] ingest | sources/courses/sutton_barto_rl_book_ch01_sec06_history.md — Sutton & Barto §1.6 强化学习史；wiki/concepts/reinforcement-learning-history.md

- **触发：** 用户指定 <http://incompleteideas.net/book/ebook/node12.html>；要求自动合并 PR
- **来源：** 教材第 1 章 §1.6 *History of Reinforcement Learning*（LaTeX2HTML 在线版；HTTPS 证书异常，HTTP 可访问）
- **关键页：** [`wiki/concepts/reinforcement-learning-history.md`](wiki/concepts/reinforcement-learning-history.md) — 试错 / DP / TD 三线汇合时间线与 Mermaid 总图
- **交叉：** [`sutton-barto-rl-book`](wiki/entities/sutton-barto-rl-book.md)、[`reinforcement-learning`](wiki/methods/reinforcement-learning.md)、[`incompleteideas-net-rich-sutton`](sources/sites/incompleteideas-net-rich-sutton.md)

## [2026-08-30] ingest | sources/blogs/wechat_embodied_station_10_papers_glancewam_vla_crew_2026-08-30.md — 10 篇独立详情节点（新建 8 / 复用 Indi+DreamMimic）；GlanceWAM / Physical Agentic / Bet4Sim2Real 已开源

- **触发：** 用户指定 <https://mp.weixin.qq.com/s/MdCtmijSM_VfYp19f-nZQw>；要求每篇论文独立、非重复详情节点；要求自动合并 PR
- **步骤 2.5：**
  - **GlanceWAM** — **已开源** MIT；训练/评测 sweep + HF `LinhanWang/GlanceWAM`
  - **Indi** — **未开源**（复用 `paper-indi`）
  - **Physical Agentic AI** — **已开源** MIT；mock / Gazebo / G1+Go2
  - **M3 / Meta-Ctrl** — **未开源** 仅项目页
  - **Imitator Game** — **部分开源** 项目页 + Arena
  - **TONAV** — **待发布** 学习代码 Coming Soon
  - **DreamMimic** — **待发布**（复用 `paper-dreammimic`）
  - **Bet4Sim2Real** — **已开源** 无 SPDX
  - **GOLEM** — **待核实** `golem-humanoid` org API 404
- **关键页：** [`wiki/overview/glancewam-vla-crew-10-papers-technology-map.md`](wiki/overview/glancewam-vla-crew-10-papers-technology-map.md)
- **机构：** 新注册 `virginia-tech` / `drexel` / `northeastern-us` / `arizona-state` / `colorado-boulder` / `hunan` / `cityu-hk` / `michigan-state` / `transcengram`

## [2026-08-30] ingest | sources/papers/parkourformer_arxiv_2605_25782.md — 复核 ParkourFormer：加深既有实体，代码改为 Coming Soon

- **触发：** 用户指定 [arXiv:2605.25782](https://arxiv.org/abs/2605.25782)（Mai et al.，HKUST-GZ / CLAI-LAB / SCAU / GDUT；G1 29 DoF 多地形跑酷）
- **排重：** 已有 canonical [`wiki/entities/paper-parkourformer.md`](wiki/entities/paper-parkourformer.md)（2026-08-16 ingest）；**未新建**第二篇 `paper-*`。
- **步骤 2.5：** 项目页现写「Code(Coming Soon)」，但 `MRonaldo-gif/parkourformer.github.io` 仍只有站点页；arXiv 仍为 **v3**；无训练仓 → **宣称将开源 / 待发布**。时序图继续不适用。
- **加深：** 观测拆分、query \(\mathbb{R}^{2\times 128}\) / 记忆 \(8\times 128\) / AMP \(10\times 67\)、Conditional SwiGLU Eq. 5、地形三维加难；交叉页开源措辞同步。
- **机构：** 既有 `hkust-gz` / `clai-lab` / `scau` / `gdut`，未改 `institutions.json`

## [2026-08-30] ingest | sources/repos/archify.md — Archify 可校验系统图 Skill；项目页已开源 MIT；wiki/entities/archify.md

- **触发：** 用户指定 <https://github.com/tt-a1i/archify>
- **步骤 2.5：** **已开源** — 项目页 <https://tt-a1i.github.io/archify/> 链回本仓；MIT；CLI `archify/bin/archify.mjs` + `SKILL.md` + Proof Lab 可运行。无权重。稳定标签 v2.15.0，开发号 v2.16.0-dev.0。
- **关键页：** [`wiki/entities/archify.md`](wiki/entities/archify.md)
- **交叉：** Draw.io Scientific Illustrator、Manim、GSAP Skills、graphify、Agentic Coding 软件工程基础
- **开源结论：** 完整渲染/校验栈已开；适合对外系统图，不替代页内 Mermaid 或 ingest 质量门。
- **机构：** 独立作者 tt-a1i，未改 `institutions.json`

## [2026-08-29] ingest | sources/sites/dexbench-org.md — DexBench 工业灵巧规格；规范已公开，官方仓与 Arena 评测栈待发布；wiki/entities/dexbench.md

- **触发：** 用户指定 <https://dexbench.org/en/>
- **步骤 2.5：** **部分开源** — 项目页完整公开 18 任务 / 55 case / OSC / 采购清单；`RLWRLD/DexBench` 404；Isaac Lab-Arena 将 NVIDIA DexBench 标为 coming soon。HF 数据卡公开检索可见 `dexbench/single-lerobot`、`dexbench/bimanual-lerobot`（本环境 API 401，未能二次核验）。
- **关键页：** [`wiki/entities/dexbench.md`](wiki/entities/dexbench.md)
- **交叉：** All Hands Up、RLDX-1、Isaac Lab、DexVerse（消歧）、评测选型闭环 / 枢纽、Manipulation
- **开源结论：** 规范可搭台；官方评测代码与 Arena 环境待发布。勿与 sail-ucf/dexbench、DexVerse 混名。
- **机构：** 既有 `rlwrld` / `nvidia`，未改 `institutions.json`

## [2026-08-29] ingest | sources/papers/riemann_1_0.md — Riemann-1.0 全因果 WAM；项目页已开、模型代码确认未开源；wiki/entities/paper-riemann-1.md

- **触发：** 用户指定 <https://riemann-dynamics.github.io/Riemann-1.0-Website>
- **步骤 2.5：** **确认未开源** — 页头仅 Paper PDF；GitHub `Riemann-Dynamics` 只有官网静态站与无关的 Matrix-Game 3.5；无权重/数据。无 arXiv。
- **关键页：** [`wiki/entities/paper-riemann-1.md`](wiki/entities/paper-riemann-1.md)
- **开源结论：** 闭源产业技术报告；可作因果 WAM 选型与 RoboCasa365 / 真机对照，不可复现。
- **机构：** 新增 `riemann-dynamics`、`kunlun-wanwei`

## [2026-08-29] ingest | sources/repos/cmoe.md + sources/repos/senlanke_mimic.md — 复核官方 CMoE 部署指针与空占位仓；senlanke/mimic 升为 SMP/CMoE/AME 三线枢纽

- **触发：** 用户指定 <https://github.com/Hoshi-No-Ai/CMoE>、<https://github.com/senlanke/mimic>
- **步骤 2.5：**
  - **CMoE** — **已开源** BSD-3-Clause；可运行 `train.py --task=g1cmoe --alg=cmoe`。无公开 checkpoint。`Fudan-MAGIC-Lab/CMoE` 为空占位。真机指向 `elevation_mapping_humanoid` + `rl_sar`。
  - **senlanke/mimic** — **已开源**（根目录无 SPDX）；SMP Complete、CMoE-G1 Port complete、AME Incomplete/unverified。课程移植，非官方。
- **关键页：** [`wiki/entities/paper-cmoe.md`](wiki/entities/paper-cmoe.md)、[`wiki/entities/smp-g1-mjlab.md`](wiki/entities/smp-g1-mjlab.md)
- **交叉：** AME 论文、楼梯障碍感知、robot_lab/rl_sar、SMP 方法页
- **机构：** 既有 `fudan` / `tsinghua`，未改 `institutions.json`

## [2026-08-29] ingest | sources/blogs/andrew_ng_ai_engineering_skills_se_fundamentals.md — 吴恩达 AI 工程技能图「软件工程基础」；无代码；升格 agentic-coding-software-fundamentals

- **触发：** 用户指定 <https://x.com/andrewyng/status/2093388974194872781>
- **步骤 2.5：** **不适用** — 论述文，无项目页、无可运行代码。同文镜像：X Article / LinkedIn Pulse / DeepLearning.AI *The Batch*
- **关键页：** [`wiki/concepts/agentic-coding-software-fundamentals.md`](wiki/concepts/agentic-coding-software-fundamentals.md)
- **交叉：** mattpocock/skills、Superpowers、ENPIRE、真机 autoresearch harness、AI Auto-Research、Data Flywheel
- **机构：** 未改 `institutions.json`（Stanford 已注册；本页为概念而非论文实体）

## [2026-08-29] ingest | sources/papers/apollo_blender_arxiv_2512_23103.md — APOLLO Blender（Yale）Blender 出图库已开源（apollo-py / apollo-toolbox-py）；论文 import 名 blender_robot_toolbox_py 不在 PyPI

- **触发：** 用户指定 <https://arxiv.org/abs/2512.23103v2>；要求自动合并 PR
- **步骤 2.5：** **已开源** MIT — 实验室研究页点名但不给直链；可核验入口 [apollo-py](https://github.com/Apollo-Lab-Yale/apollo-py) / PyPI `apollo-toolbox-py` 0.0.13（`apollo_py_blender.ChainBlender`、`scripts/test.py`）。论文示例包名 `blender_robot_toolbox_py` **PyPI 404**。
- **关键页：** [`wiki/entities/paper-apollo-blender.md`](wiki/entities/paper-apollo-blender.md)
- **交叉：** Blender / URDD / URDF / 关键帧编辑器 / Robot Viewer / Manim / 描述目录选型
- **机构：** 既有 `yale`，未改 `institutions.json`

## [2026-08-29] ingest | sources/papers/flashvla_arxiv_2608_27384.md + clap_arxiv_2608_27406.md + solo_arxiv_2608_26583.md — FlashVLA 流式异步 VLA 已开源；CLAP 跨本体视频 WM + G1 权重已开源；SOLO 1.5 km 感知行走项目页未列代码

- **触发：** 用户指定 FlashVLA（[论文](https://arxiv.org/abs/2608.27384) / [GitHub](https://github.com/z-lab/flashvla) / [HF](https://huggingface.co/z-lab/flashvla-pi05-libero)）、CLAP（[论文](https://arxiv.org/abs/2608.27406) / [GitHub](https://github.com/omni-CLAP/clap) / [项目页](https://omni-clap.github.io/) / [HF](https://huggingface.co/omni-CLAP/CLAP)）、SOLO（[论文](https://arxiv.org/abs/2608.26583) / [项目页](https://sunpihai-up.github.io/solo/)）；要求自动合并 PR
- **步骤 2.5：**
  - **FlashVLA** — **已开源** Apache-2.0；`sim_eval/libero/eval.sh` / `train/train.sh` / `benchmarks/benchmark_latency.py` 可跑；LIBERO/RoboTwin 权重已发（`flashvla-pi05-libero`、`flashvla-pi05-robotwin`）
  - **CLAP**（omni-CLAP 视频世界模型，勿与音频 CLAP 混淆）— **已开源** MIT；`clap-rollout-replay` / `teleop` / `deploy` / `clap-eval` + G1 适配（`adapt-g1` 26-D）；项目页 BibTeX 仍写 Coming soon，引用用 arXiv
  - **SOLO** — **确认未开源**：项目页无 GitHub/HF；论文无代码 URL。有视频与 2026 世界人形运动会展示（技术同源，非论文主表）
- **关键页：** [`wiki/entities/paper-flashvla.md`](wiki/entities/paper-flashvla.md)、[`wiki/entities/paper-clap-cross-embodiment.md`](wiki/entities/paper-clap-cross-embodiment.md)、[`wiki/entities/paper-solo.md`](wiki/entities/paper-solo.md)
- **交叉：** VLA / action chunking / π0.5 / Ctrl-World / ForeTime-VLA；生成式世界模型 / 虚拟沙盒；DPL / SSR / 特权训练 / 地形适应 / 楼梯障碍感知行走 / 天工
- **机构：** 既有 `ucsd` / `mit` / `princeton` / `ustc` / `x-humanoid` / `hku` / `anu` / `hkust-gz` / `sjtu` / `tsinghua` / `cuhk`，未改 `institutions.json`

## [2026-08-28] ingest | sources/repos/tita_rl.md + wheel_legged_genesis.md + isaac_rl_two_wheel_legged_bot.md — 轮腿双足三仓：TITA Gym / Genesis / Flamingo Lab

- **触发：** 用户指定 <https://github.com/DDTRobot/tita_rl>、<https://github.com/Albusgive/wheel_legged_genesis>、<https://github.com/jaykorea/Isaac-RL-Two-wheel-Legged-Bot>；要求自动合并 PR
- **步骤 2.5：** 三仓均无独立项目页，以 GitHub README 为准
  - **tita_rl** — **已开源** MIT；可运行 `train.py` / `simple_play.py`；部署分仓 `tita_rl_sim2sim2real`（无 SPDX）
  - **wheel_legged_genesis** — **已开源** MIT；Genesis + vendored RSL-RL + MuJoCo sim2sim；无官方真机包
  - **Isaac-RL-Two-wheel-Legged-Bot** — **已开源**（SPDX MIT，`setup.py` 另写 BSD-3-Clause）；`lab.flamingo` 训练可跑；sim2sim 分支迁移中
- **关键页：** [`wiki/concepts/wheel-legged-biped.md`](wiki/concepts/wheel-legged-biped.md)、[`wiki/entities/tita-rl.md`](wiki/entities/tita-rl.md)、[`wiki/entities/wheel-legged-genesis.md`](wiki/entities/wheel-legged-genesis.md)、[`wiki/entities/isaac-rl-two-wheel-legged-bot.md`](wiki/entities/isaac-rl-two-wheel-legged-bot.md)
- **交叉：** DDT_Lab（Gym vs Lab 分代）、轮足四足、Hybrid Locomotion、Genesis、Isaac Gym/Lab、RSL-RL、Webots、ZONDA/TITA
- **机构：** 已有 `direct-drive-tech` / `postech`；新注册 `independent-maintainer`（独立维护者（Independent Maintainer））供 Albusgive 社区仓派生所属机构

## [2026-08-28] ingest | sources/sites/pollen-robotics-microduck.md + sources/repos/microduck.md + sources/repos/microduck_rl.md — Pollen Microduck 产品页/Runtime/RL 已开源；升格 pollen-microduck 与 pollen-microduck-rl

- **触发：** 用户指定 [1] <https://github.com/pollen-robotics/microduck_rl> [2] <https://discord.com/invite/pollen-community-519098054377340948> [3] <https://github.com/pollen-robotics/microduck> [4] <https://pollen-robotics.com/microduck>；要求自动合并 PR
- **步骤 2.5：** **已开源** Apache-2.0 — 产品页 Open source 区链到 Runtime 与 RL 仓；`microduck`（Rust daemon，`main`）+ `microduck_rl`（mjlab + PPO，默认分支 `develop`，可 `uv run train/export`）；无 GPU 可 Hugging Face Jobs。整机是商品预售（$399，宣称 2026 圣诞前发货），不是 DIY BOM。RL README 另写硬件设计 CC BY-SA-NC。
- **关键页：** [`wiki/entities/pollen-microduck.md`](wiki/entities/pollen-microduck.md)、[`wiki/entities/pollen-microduck-rl.md`](wiki/entities/pollen-microduck-rl.md)
- **交叉：** Reachy2 / Open Duck Mini / Open Duck Playground / mjlab / BAM / Reward Design / Sim2Real Gap 缩减
- **机构：** 新注册 `pollen-robotics`（花粉机器人（Pollen Robotics））；既有 `pollen` tag 一并命中

## [2026-08-28] structural | 姊妹仓库更名 Robot_Learning_Paper_Notebooks — 同步 GitHub / Pages URL 与展示名

- **触发：** 姊妹项目已更名为 [Robot_Learning_Paper_Notebooks](https://github.com/ImChong/Robot_Learning_Paper_Notebooks)，站点 [imchong.github.io/Robot_Learning_Paper_Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/)
- **替换：** `Humanoid_Robot_Learning_Paper_Notebooks` → `Robot_Learning_Paper_Notebooks`；展示名 Humanoid Paper Notebooks / 人形论文深读笔记 → Robot Learning Paper Notebooks / 机器人学习论文笔记
- **保留：** 本库内链文件名（`wiki/overview/humanoid-paper-notebooks-index.md`、`sources/sites/humanoid-robot-learning-paper-notebooks.md`、`paper-notebook-*` / `humanoid_pnb_*`）与 tag `humanoid-paper-notebooks` 不变，以免 related 断链
- **关键页：** [`wiki/overview/humanoid-paper-notebooks-index.md`](wiki/overview/humanoid-paper-notebooks-index.md)、[`sources/sites/humanoid-robot-learning-paper-notebooks.md`](sources/sites/humanoid-robot-learning-paper-notebooks.md)
- **脚本：** sync / bootstrap / export / graph.html 识别新旧 Pages 路径；搜索只用新名，不保留旧名别名

## [2026-08-28] ingest | sources/papers/vb_com_arxiv_2502_14814.md — VB-Com（ICRA 2026）视觉/盲策略复合人形运动；项目页 Code coming soon；原地升格 paper-notebook-vb-com

- **触发：** 用户指定 *VB-Com: Learning Vision-Blind Composite Humanoid Locomotion Against Deficient Perception*（Junli Ren 等；ICRA 2026；Unitree G1/H1；论文 <https://arxiv.org/pdf/2502.14814>；项目页 <https://renjunli99.github.io/vbcom.github.io/>）；要求自动合并 PR
- **步骤 2.5：** **宣称将开源 / 截至入库日项目页未列 GitHub** — 按钮文案 *Code (coming soon)*，`href=""`；无训练/推理入口、无权重
- **关键页：** [`wiki/entities/paper-notebook-vb-com-learning-vision-blind-composite-humanoid.md`](wiki/entities/paper-notebook-vb-com-learning-vision-blind-composite-humanoid.md)（原 Paper Notebooks planned 占位，原地升格，未新建重复节点）
- **交叉：** 楼梯与障碍 locomotion 枢纽 / 人形运控观测输入 / Humanoid Locomotion / Terrain Adaptation / RPL / PIM 占位 / Unitree G1
- **机构：** 已有 `shanghai-ai-lab` / `hku` / `sjtu` / `zju` / `cuhk`；未新注册

## [2026-08-28] ingest | sources/courses/nvidia_getting_started_isaac_lab.md — NVIDIA Getting Started With Isaac Lab 全系列四模块；Isaac Lab 已开源；升格 wiki/entities/nvidia-getting-started-isaac-lab.md

- **触发：** 用户指定 <https://docs.nvidia.com/learning/physical-ai/getting-started-with-isaac-lab/latest/index.html> 一整个系列
- **步骤 2.5：** **课程为文档、无可独立课程仓**；可运行代码走 [isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab)（**已开源**）；云端 Brev / Isaac Launchable（课内警告当时钉 Lab 3.0，可能与课测版本不兼容）
- **关键页：** [`wiki/entities/nvidia-getting-started-isaac-lab.md`](wiki/entities/nvidia-getting-started-isaac-lab.md)
- **交叉：** Physical AI 门户 / Isaac Lab / Cartpole / skrl / DR / 特权训练 / Actuator Network / SO-101 对照课
- **机构：** 已有 `nvidia`

## [2026-08-28] ingest | sources/papers/inverse_reinforcement_learning_primary_refs.md — IRL 一手谱系（Ng 2000 / MaxEnt / GCL / GAIL / AIRL）；HumanCompatibleAI/imitation 已开源；升格 wiki/methods/inverse-reinforcement-learning.md

- **触发：** 用户要求找到逆强化学习（IRL）相关一手资料并 ingest；仓库原无独立 IRL 方法页
- **步骤 2.5：**
  - **Ng 2000 / Abbeel 2004 / Ziebart 2008** — **确认未开源**（作者 PDF，无可运行实现）
  - **GCL** — **确认官方 GCL 专仓不存在**；[cbfinn/gps](https://github.com/cbfinn/gps) 是 GPS 不是 GCL
  - **GAIL** — **已开源** MIT，[openai/imitation](https://github.com/openai/imitation) **已归档**（2018）
  - **AIRL** — **已开源** MIT，[justinjfu/inverse_rl](https://github.com/justinjfu/inverse_rl) TF1 时代（2018）
  - **现代入口** — **已开源** MIT，[HumanCompatibleAI/imitation](https://github.com/HumanCompatibleAI/imitation) PyTorch + Gymnasium
- **关键页：** [`wiki/methods/inverse-reinforcement-learning.md`](wiki/methods/inverse-reinforcement-learning.md)
- **交叉：** IL / BC / DAgger / AMP / Reward Design / MDP / RL vs IL / RL Runner
- **机构：** 已有 `berkeley` / `stanford` / `cmu`；未新注册

## [2026-08-28] ingest | sources/sites/anthropic-claude-plays-robotics.md + anthropic-model-hardware-standard.md + anthropic-recursive-self-improvement.md — Anthropic 机器人评测 / MHS / RSI 三件套

- **触发：** 用户指定 [1] <https://www.anthropic.com/research/claude-plays-robotics> [2] <https://www.anthropic.com/news/model-hardware-standard-research-preview> [3] <https://www.anthropic.com/institute/recursive-self-improvement>
- **步骤 2.5：**
  - **Embody** — **宣称将开源 / 截至入库日 404** — `github.com/safety-research/embody`（`EXPERIMENTS.md` / `METRICS.md`）；无可运行入口
  - **MHS** — **宣称将开源 / 研究预览未公开规范仓** — 项目页 <https://modelhardwarestandard.com/> 申请制；无 GitHub / SDK
  - **RSI 论述** — **不适用（论述文）** — 无可运行实现；内部生产率数字不可复现
- **关键页：** [`wiki/concepts/llm-robotics-control-interfaces.md`](wiki/concepts/llm-robotics-control-interfaces.md)、[`wiki/entities/anthropic-embody.md`](wiki/entities/anthropic-embody.md)、[`wiki/concepts/model-hardware-standard.md`](wiki/concepts/model-hardware-standard.md)、[`wiki/concepts/recursive-self-improvement.md`](wiki/concepts/recursive-self-improvement.md)
- **交叉：** MCP / VLA / LeRobot / ASPIRE / Safety Filter / 频率解耦 / locomotion / manipulation / LIBERO / AI Auto-Research / 真机 autoresearch harness
- **机构：** 已有 `anthropic`；新注册 `hhmi-janelia`（珍妮莉亚研究园区（HHMI Janelia））

## [2026-08-28] ingest | sources/papers/wall_ss_x_square_2026.md — 自变量 WALL-SS 下一尺度自回归长程世界模型；训练推理代码待发布；升格 wiki/entities/paper-wall-ss.md

- **触发：** 用户指定 *WALL-SS: Scaling Long-horizon World Models via Next-Scale Autoregression*（自变量机器人；项目页 <http://x2robot.com/pages/ss>）；要求自动合并 PR
- **步骤 2.5：** **宣称将开源 / 待发布训练推理代码** — [X-Square-Robot/wall-ss](https://github.com/X-Square-Robot/wall-ss) MIT 占位仓（PDF + README + assets）；README TODO *Release the training and inference code* 未勾选；无公开 arXiv abs（页眉 `arXiv:submit/7998075`）
- **关键页：** [`wiki/entities/paper-wall-ss.md`](wiki/entities/paper-wall-ss.md)
- **交叉：** 生成式世界模型 / 视频即仿真 / 虚拟沙盒 / 训练闭环 taxonomy / Ctrl-World / OSCAR / SC3-Eval / Cosmos 3 / WorldEcho / GigaWorld-1
- **机构：** 新注册 `x-square-robot`（自变量机器人（X Square Robot））；顺带修正 X2Streaming-TTS 的中文机构名

## [2026-08-28] ingest | sources/repos/omnisim.md — OmniSim（Webots 独立 fork）；已开源 Apache-2.0；Newton 唯一后端；升格 wiki/entities/omnisim.md

- **触发：** 用户指定 <https://github.com/omnilink-tech/omnisim> 与项目页 <https://www.omnilink-agents.com/omnisim>；要求自动合并 PR
- **步骤 2.5：** **已开源** Apache-2.0 — 项目页 CTA 链到同一 GitHub；一等 MCP / HTTP harness / ROS 2 sidecar 在仓内；macOS 物理未验证；**零 sim-to-real**；Twin Shadow 未实现
- **营销页漂移：** 产品页仍写 ODE CPU 回退与更宽机型名单；仓库 README/`AGENTS.md` 写 Newton 唯一后端（ODE 2026-08-08 删除）、世界以 `.omniworld` 为准
- **关键页：** [`wiki/entities/omnisim.md`](wiki/entities/omnisim.md)
- **交叉：** Webots / Newton / 仿真器选型 / MCP / Isaac Sim / Gazebo / mjlab
- **机构：** 注册 `omnilink`（欧姆尼链（OmniLink））

## [2026-08-28] ingest | sources/papers/video2door_traversal_arxiv_2608_20251.md — 加深 Video2DoorTraversal 单视频门孪生穿门；项目页仍 Code Coming soon

- **触发：** 用户指定 <https://video2doortraversal.github.io/> 与 [arXiv:2608.20251v1](https://arxiv.org/abs/2608.20251v1)；要求自动合并 PR
- **步骤 2.5：** **待发布 / 宣称将开源** — 项目页复核（2026-08-28）仍为 **Code Coming soon**，无 GitHub / 权重；不建 `sources/repos/`
- **关键页：** [`wiki/entities/paper-video2door-traversal.md`](wiki/entities/paper-video2door-traversal.md)（由 2026-08-22 盘点摘要加深为论文级编译）
- **交叉：** Articraft / PhysX-Omni / DoorMan / Action Chunking / Agentic Real2Sim / Isaac Gym / loco-manipulation
- **机构：** 已有 `sjtu` / `sdu`；新注册 `neowa`（纽娲机器人（NeoWa Robotics））

## [2026-08-28] ingest | sources/papers/ucag_p_arxiv_2608_26058.md — UCAG-P 相机系锚点几何预训练；宣称将开源；升格 wiki/entities/paper-ucag-p.md

- **触发：** 用户指定 *One Policy, Many Embodiments*（arXiv / 小米具身智能 × 澳门大学）；要求自动合并 PR
- **步骤 2.5：** **宣称将开源 / 待发布** — 项目页 <https://public-bots.github.io/UCAG-P>；[Public-BOTs/UCAG-P](https://github.com/Public-BOTs/UCAG-P) 仅为配图与静态站，README *code will be released soon*
- **关键页：** [`wiki/entities/paper-ucag-p.md`](wiki/entities/paper-ucag-p.md)
- **交叉：** VLA / Qwen-RobotManip / Qwen-VLA / Xiaomi-Robotics-0/1 / DyPES-VLA / Open X-Embodiment / hub-vla / hub-cross-embodiment / manipulation；合入时与 main 上 9 篇盘点 / 五篇 ingest 共用同一实体页并加深编译
- **机构：** `xiaomi` / `umac`

## [2026-08-28] ingest | sources/blogs/wechat_embodied_station_wam_vla_cross_embodiment_9_papers_2026-08-28.md — 具身智能小站 9 篇（WAM/VLA/跨本体）；9/9 独立 paper-* 节点；开源分化

- **触发：** 用户指定 <https://mp.weixin.qq.com/s/FNhRO3KOm8k8CkJEqystQQ>；要求每篇论文独立、非重复详情节点
- **步骤 2.5：** Zero-WAM / StreamPI / UCAG-P / R³ **待发布**；MA-VLA / ConfAL-WM / TARCAT 标注 **已开源**；GaussianDream++ / Super Odometry 2.0 **部分开源**
- **关键页：** [`wiki/overview/wam-vla-cross-embodiment-9-papers-technology-map.md`](wiki/overview/wam-vla-cross-embodiment-9-papers-technology-map.md)；9 个新建 `paper-*`（含 GaussianDream++ 与 Awesome 索引级 2605.20752 **分列**）
- **交叉：** WAM / VLA / 生成式世界模型 / LIO 选型
- **机构：** 注册 `robbyant` / `buffalo` / `uci`；合入 main 时与 RSL-RL 五篇 ingest 共用 Zero-WAM / UCAG-P 规范页

## [2026-08-28] ingest | sources/papers/{anytime_gtmp,lac,zero_wam,ucag_p}_arxiv_2608_*.md, sources/repos/rsl_rl.md — RSL-RL BF16 PPO/蒸馏、Anytime GTMP 批量规划、LAC 线角柔顺、Zero-WAM 人视频提示、UCAG-P 跨本体动作几何

- **触发：** 用户指定五条资料一并 ingest，并要求自动合并 PR
- **步骤 2.5：**
  - **RSL-RL** — **已开源** BSD-3-Clause（[leggedrobotics/rsl_rl](https://github.com/leggedrobotics/rsl_rl)）；PPO + DistillationRunner；BF16 仅 `update()` autocast
  - **Anytime GTMP** — **已开源** MIT（[CoMMALab/anytime_gtmp](https://github.com/CoMMALab/anytime_gtmp)）；需 vamp `benchmark_aorrtc_backend` 子模块
  - **LAC** — **部分开源**（[lac-humanoid/lac-code](https://github.com/lac-humanoid/lac-code) MIT：ckpt / sim2sim / ROS 2；Isaac Lab 训练未发布）；项目页 <https://lac-humanoid.github.io/>
  - **Zero-WAM** — **宣称将开源 / 待发布**（仓占位，预计 2026-09-15 前发代码/模型/数据）；项目页 <https://robbyant-research.github.io/Zero-WAM/>
  - **UCAG-P** — **宣称将开源 / 待发布**（仓为论文图与项目页）；项目页 <https://public-bots.github.io/UCAG-P/>
- **关键页：** [`wiki/entities/rsl-rl.md`](wiki/entities/rsl-rl.md)、[`wiki/entities/paper-anytime-gtmp.md`](wiki/entities/paper-anytime-gtmp.md)、[`wiki/entities/paper-lac.md`](wiki/entities/paper-lac.md)、[`wiki/entities/paper-zero-wam.md`](wiki/entities/paper-zero-wam.md)、[`wiki/entities/paper-ucag-p.md`](wiki/entities/paper-ucag-p.md)
- **交叉：** PPO / RL Runner / 特权训练 / AMP-RSL-RL / Isaac Lab / HMI 开源覆盖；阻抗/导纳 / loco-manip / GentleHumanoid；WAM / ICL / LAWA；manipulation / Qwen-RobotManip / 跨本体迁移 / cuRobo
- **机构：** 注册 `tohoku` / `vinuni` / `robbyant`；UCAG-P 用已有 `umac`

## [2026-08-28] ingest | sources/repos/kimodo-cpp.md — LocalAI kimodo.cpp（C++/GGML）；已开源 Apache-2.0；SOMA/G1 GGUF 已发布、SMPL-X 禁止再分发；升格 wiki/entities/kimodo-cpp.md

- **触发：** 用户指定 <https://github.com/localai-org/kimodo.cpp>；要求自动合并 PR
- **步骤 2.5：** **已开源** — 无独立项目页，以 GitHub README / `PORTING.md` / `docs/IMPLEMENTATION.md` 为入口；代码 Apache-2.0；HF `LocalAI-io` 发布 SOMA/G1 F32 GGUF（NVIDIA Open Model License）；SMPL-X 为 Internal R&D，安装器故意不打包
- **关键页：** [`wiki/entities/kimodo-cpp.md`](wiki/entities/kimodo-cpp.md)
- **交叉：** Kimodo 上游实体、diffusion-motion-generation、HY-Motion vs GENMO vs Kimodo、SAM3DBody-cpp
- **机构：** 注册 `localai`（本地人工智能（LocalAI））

## [2026-08-27] ingest | sources/papers/vgi_white_paper_arxiv_2608_25924.md — CVPR 2026 VGI 白皮书；确认未开源；升格 paper-vgi-white-paper

- **触发：** 用户指定 [arXiv:2608.25924](https://arxiv.org/abs/2608.25924)；要求自动合并 PR
- **步骤 2.5：** **确认未开源** — 工作坊页 <https://cvpr2026-vgi-workshop.limitlab.xyz/> 仅 slides/poster，无 GitHub / 权重 / 数据集；立场白皮书无可运行实现
- **关键页：** [`wiki/entities/paper-vgi-white-paper.md`](wiki/entities/paper-vgi-white-paper.md)
- **交叉：** From AGI to ASI / 生成式世界模型 / WAM / 视频即仿真 / 生成式视觉预训练 / VLM-VLN-VLA-VLX-WM 分类
- **机构：** `aist` / `oxford` / `openai` / `cambridge` / `google-deepmind` / `cmu` / `harvard` / `stanford` / `princeton` / `nyu` / `imperial-college`

## [2026-08-27] ingest | sources/blogs/perceptron_isaac_05.md — Perceptron Isaac 0.5（36B-A2.5B 稀疏具身基础模型）；部分开源（代码 Apache 2.0，Hub 权重 COMING SOON，mHarmony 未进 extra）；关键页 wiki/entities/perceptron-isaac-05.md

- **触发：** 用户指定 <https://www.perceptron.inc/blog/introducing-isaac-0-5>；要求自动合并 PR
- **步骤 2.5：** **部分开源** — [perceptron-ai-inc/isaac](https://github.com/perceptron-ai-inc/isaac) Apache 2.0 + LeRobot 子模块 `perceptron_isaac`；Hub [PerceptronAI/Isaac-0.5](https://huggingface.co/PerceptronAI/Isaac-0.5) 入库日标 **COMING SOON**；mHarmony/TensorStream **未纳入** extra；未来 percept 损失 **专有**
- **关键页：** [`wiki/entities/perceptron-isaac-05.md`](wiki/entities/perceptron-isaac-05.md)
- **交叉：** VLA / foundation-policy / embodied-scaling-laws / LeRobot / Perceptron Egocentric / Isaac GR00T（名称消歧）/ Dyna-2 / π0.7
- **机构：** `perceptron`（感知器（Perceptron））已注册

## [2026-08-27] ingest | sources/blogs/wechat_embodied_ai_lab_scirobotics_three_humanoid_papers_2026.md — SciRob 11(117) 同期 ZEST/SONIC/视觉足球升格三层对比；SONIC DOI aed4592；开源：ZEST 未开源 / SONIC 已开源 / 视觉足球部分开源

- **触发：** 用户指定 <https://mp.weixin.qq.com/s/UC-LTs_E83ssuImnXusQGA>；要求自动合并 PR
- **步骤 2.5：** ZEST **确认未开源**；SONIC **已开源**（[GEAR-SONIC](https://nvlabs.github.io/GEAR-SONIC/) 确认 *Science Robotics* 11(117) / [DOI aed4592](https://doi.org/10.1126/scirobotics.aed4592)，代码 GR00T-WholeBodyControl）；视觉足球 **部分开源**（Zenodo，真机未发布）
- **关键页：** [`wiki/comparisons/zest-vs-sonic-vs-vision-soccer.md`](wiki/comparisons/zest-vs-sonic-vs-vision-soccer.md)
- **交叉：** zest / sonic-motion-tracking / vision soccer 实体、运动跟踪选型、humanoid-soccer、身体系统栈
- **复用：** 三篇均已有实体/方法页，不重复造节点



## [2026-08-27] ingest | sources/papers/{worldecho_worldsync,lawa,arli}_arxiv_2608_*.md — WorldEcho/WorldSync 动作跟随评测、LAWA 潜意图 WAM、ARLI 异步延迟感知 RL；升格三实体

- **触发：** 用户指定 [arXiv:2608.24885](https://arxiv.org/abs/2608.24885)、[arXiv:2608.24882](https://arxiv.org/abs/2608.24882)（项目页 <https://getterupper.github.io/LAWA>）、[arXiv:2608.23831](https://arxiv.org/abs/2608.23831)；要求自动合并 PR
- **步骤 2.5：** WorldEcho/WorldSync **确认未开源**（无项目页/代码）；LAWA **宣称将开源 / 待发布**（项目页 Code coming soon；`getterupper/LAWA` 仅网页）；ARLI **确认未开源**（项目页无代码，GitHub 仅 Pages）
- **关键页：** [`wiki/entities/paper-worldecho-worldsync.md`](wiki/entities/paper-worldecho-worldsync.md)、[`wiki/entities/paper-lawa.md`](wiki/entities/paper-lawa.md)、[`wiki/entities/paper-arli.md`](wiki/entities/paper-arli.md)
- **交叉：** generative-world-models / WAM / 评测选型闭环 / VLA / action-chunking / 部署指南 / manipulation
- **机构：** 注册 `uestc`（电子科技大学（UESTC））；TARS Robotics 正文写出、未注册 tag



- **触发：** 用户指定项目页 <https://pratikkunapuli.github.io/rl-vs-gc/> 与 [arXiv:2506.17832](https://arxiv.org/abs/2506.17832)
- **步骤 2.5：** **已开源** — [PratikKunapuli/rl-vs-gc](https://github.com/PratikKunapuli/rl-vs-gc) 含 DirectRLEnv、`train_rslrl.py`、`gc_tuning.py` 与 PaperModels checkpoint；仓库 **无 LICENSE**。评测纯仿真
- **关键页：** [`wiki/entities/paper-rl-vs-gc.md`](wiki/entities/paper-rl-vs-gc.md)、[`wiki/comparisons/rl-vs-geometric-control.md`](wiki/comparisons/rl-vs-geometric-control.md)
- **交叉：** mpc-vs-rl / wbc-vs-rl / 多旋翼栈 / Isaac Lab / gym-pybullet-drones / Flightmare / sim2real / domain-randomization / reinforcement-learning
- **机构：** `upenn`（GRASP Lab）

## [2026-08-26] ingest | sources/repos/isaaclab_decoupled_wbc.md — HTD 解耦 WBC 全组件入库；Isaac Lab 训练/蒸馏/G1 部署已开源，遥操作与 HTD 策略仍待发布

- **触发：** 用户指定 [IsaacLab-Decoupled-WBC](https://github.com/chrisyrniu/IsaacLab-Decoupled-WBC) 与论文仓 [humanoid-touch-dream](https://github.com/chrisyrniu/humanoid-touch-dream)；要求 ingest 全身控制器全部组件
- **步骤 2.5：** **部分开源** — WBC teacher/student/example checkpoint/G1 部署 **已开源**（BSD-3-Clause）；VR 遥操作与 HTD 策略 README 仍 on-going
- **关键页：** [`wiki/entities/htd-decoupled-wbc.md`](wiki/entities/htd-decoupled-wbc.md)；交叉 HTD 方法页 / WBC / loco-manipulation / Isaac Lab / G1
- **机构：** 注册 `bosch`、`ut-arlington`；实体 tags `cmu` / `bosch` / `ut-arlington`

## [2026-08-26] ingest | sources/blogs/wechat_embodied_station_7_papers_vla_intent_space_2026-08-26.md — 具身智能小站 7 篇开源盘点；新建 6 论文实体 + 技术地图；复用 ROS2SmolVLA；Indi 未开源 / SRB·RAFT·evrgb·PhyFilter 已开源 / MoeCo 部分开源

- **触发：** 用户指定公众号 <https://mp.weixin.qq.com/s/zHxwlUsj22t1oPd9Q2C-dw>；要求每篇论文独立非重复详情节点
- **步骤 2.5：** Indi 仅 Pages 站 **未开源**；SRB / RAFT / simple-evrgb-cal / PhyFilter **已开源**；MoeCo 模型/损失已放、完整训练入口待录用后发布；ROS2SmolVLA 沿用既有开源结论
- **新建实体：** `paper-indi`、`paper-reward-free-continual-adaptation-space`、`paper-raft-thruster-fault`、`paper-moeco`、`paper-simple-evrgb-cal`、`paper-phyfilter`
- **复用实体：** `paper-ros2smolvla`（arXiv:2608.23320，当日先前 ingest）
- **技术地图：** [`wiki/overview/open-source-7-papers-system-structure-technology-map.md`](wiki/overview/open-source-7-papers-system-structure-technology-map.md)

## [2026-08-26] fix | scripts/utils/community_labels.py — 为 VLA 开源复现景观枢纽补 COMMUNITY_NAME_OVERRIDES，修复 Tests community-19 命名断言

- **触发：** PR #1682 合入后 Louvain 把 `wiki/overview/vla-open-source-repro-landscape-2025.md` 选为社区枢纽；H1「VLA 开源复现景观（2025 策展）」以英文缩写开头，不符合 `中文（English） 社区`
- **修复：** override 为「视觉语言动作开源复现景观（VLA Open-Source Reproduction Landscape）」

## [2026-08-26] ingest | sources/papers/{ros2smolvla,dreammimic,ld4wam}_arxiv_2608_*.md — ROS2SmolVLA（ROS 2 本地 SmolVLA×UR10e，已开源）、DreamMimic（视觉全身 RSSM 蒸馏，代码 Coming soon）、LD4WAM（跨本体运动对齐潜动力学 WAM，未开源）

- **ROS2SmolVLA（2608.23320）：** 奥格斯堡大学；项目页+GitHub+HF 核查 **已开源**；实体 `paper-ros2smolvla`；交叉 VLA / LeRobot / ROS 2 / manipulation
- **DreamMimic（2608.22278）：** Independent / 清华；项目页 Code Coming soon，GitHub 仅占位 README；实体 `paper-dreammimic`；交叉 VisualMimic / InterMimic / loco-manipulation
- **LD4WAM（2608.22403）：** 无项目页/代码；通讯作者个人主页为 Berkeley 博士后，论文未列单位；实体 `paper-ld4wam`；交叉 WAM / EgoWAM / Being-H0.7 / manipulation

## [2026-08-26] ingest | sources/blogs/skild_s1_in_context_learning.md — Skild S1 视频 ICL；github.com/skild-ai 0 公开仓确认未开源；升格 wiki/entities/skild-s1.md 与 skild-ai.md

- **触发：** 用户指定 <https://www.skild.ai/blogs/s1>
- **步骤 2.5：** **确认未开源** — 公司 GitHub org 无公开仓；训练配方本篇推迟；LocoFormer 仅有社区非官方实现
- **关键页：** [`wiki/entities/skild-s1.md`](wiki/entities/skild-s1.md)、[`wiki/entities/skild-ai.md`](wiki/entities/skild-ai.md)；交叉 ICL / GEN-1.5 / foundation-policy / scaling / flywheel
- **开源结论：** 数字为内部基准自报（未见 100k h：ICL 66% vs 语言 VLA 9%）

## [2026-08-26] ingest | sources/blogs/seohong_behavioral_cloning_mystery.md — 真机风格 BC 四条 mystery；升格 wiki/concepts/behavioral-cloning-mysteries.md；基准宣称 2026-10 开源、入库日未见代码

- **触发：** 用户指定 <https://seohong.me/blog/behavioral-cloning-mystery/>
- **步骤 2.5：** **宣称将开源** — 作者计划 2026-10 发官方基准；入库日无 GitHub / 数据链
- **四条：** 过拟合有时更好；开环优于闭环；策略须极大；无限数据下特征缩放仍改成功率
- **关键页：** [`wiki/concepts/behavioral-cloning-mysteries.md`](wiki/concepts/behavioral-cloning-mysteries.md)

## [2026-08-25] ingest | sources/repos/fetchman.md — FetchMan 二次核查开源状态：项目页挂 GitHub 占位仓；更新实体页与 loco-manipulation 索引

- **触发：** 用户指定 <https://orayyan.com/fetchman>、arXiv:2608.17027；要求自动合并 PR
- **步骤 2.5：** **部分开源 / 待发布** — [omarrayyann/fetchman](https://github.com/omarrayyann/fetchman) 仅 README；README：**Code will be added by September 1**；FetchMan-Bench 仍无下载链
- **增量：** 新建 `sources/repos/fetchman.md`；更新 `sources/sites/`、`sources/papers/`、`wiki/entities/paper-fetchman.md` 开源结论（2026-08-20「未开源」→ 占位仓待发布）

## [2026-08-25] ingest | sources/blogs/wechat_embodied_heart_robot_icl_gen15_survey_2026-08-25.md — 具身智能之心「机器人上下文学习」万字综述；新建概念页 robot-in-context-learning；交叉 GEN-1.5 / IL / foundation-policy / manipulation

- **触发：** 用户指定公众号 <https://mp.weixin.qq.com/s/V_Dm8kHvB2YxtGY7qScjXA>
- **核心 taxonomy：** 映射选择 vs 状态估计 vs 映射本身（真 ICL）；按遥操作 / 人视频 / 系统辨识分线；对照 π0.7、记忆 VLA、RoboTTT
- **关键页：** [`wiki/concepts/robot-in-context-learning.md`](wiki/concepts/robot-in-context-learning.md)

## [2026-08-25] ingest | sources/papers/smp.md — 补全 SMP 论文实体页、项目页与 senlanke/mimic G1 复现归档；官方 MimicKit + 乘性奖励 mjlab 管线

## [2026-08-25] ingest | sources/repos/dexmal_opendm.md + sources/blogs/dexmal_dm05.md — OpenDM/DM0.5 二次核查（GitHub + HF DM05 + 技术博客）；补 Gemma 权重许可、官方 News 时间线；更新 wiki/entities/dexmal-dm05.md；互链 wiki/overview/vla-open-source-repro-landscape-2025.md

- **触发：** 用户指定 <https://github.com/dexmal/opendm>、<https://huggingface.co/Dexmal/DM05>、<https://www.dexmal.com/blog/dm0.5>；要求自动合并 PR
- **开源核查（步骤 2.5）：** **已开源** — 代码 Apache-2.0；HF `Dexmal/DM05` 权重 **Gemma 许可**；下游 LIBERO / RobotWin2 / SO101 / VLA-Arena / Table30v2 齐全
- **增量：** 补 OpenDM README News（含 2026-08-03 `robot_platforms.md`）；二次核查日期 2026-08-25
- **交叉更新：** [`wiki/entities/dexmal-dm05.md`](wiki/entities/dexmal-dm05.md)、[`wiki/overview/vla-open-source-repro-landscape-2025.md`](wiki/overview/vla-open-source-repro-landscape-2025.md)

## [2026-08-25] ingest | sources/blogs/wechat_embodied_station_8_papers_open_source_2026-08-25.md — 具身智能小站「8 篇开源论文」；新建 5 论文实体 + 技术地图；复用 ViTacPhys/Q-Planning/DreamHand

- **触发：** 用户指定公众号 <https://mp.weixin.qq.com/s/71jZDzvcWZ3SsoHOEA8sgQ>；要求 8/8 独立详情节点、同一 arXiv 不重复造页
- **步骤 2.5：** Space Mining 清单 **已开源**；SRL-MPC 仓 **待发布**；TOSS OSF **已开源**；PhysCaP **未开源**；GhostTac 演示 **已开源**；复用三篇沿用先前开源结论
- **新建实体：** `paper-space-mining-with-robotics`、`paper-srl-mpc`、`paper-toss-framework`、`paper-physcap`、`paper-ghosttac`
- **复用实体：** `paper-vitacphys`、`paper-qplanning`、`paper-dreamhand`
- **技术地图：** [`wiki/overview/open-source-8-papers-technology-map.md`](wiki/overview/open-source-8-papers-technology-map.md)

## [2026-08-25] ingest | sources/papers/{qplanning,foretime_vla,g1_compliant_surface_standup}_arxiv_2608_*.md — Q-Planning（冻结 BC+Q 自改进，已开源）、ForeTime-VLA（WAM 未来 token 蒸馏 π₀.₅，未开源）、G1 软地面起身（IIT Kanpur，评测+权重已开源）

- **Q-Planning（2608.21204）：** 项目页+GitHub 核查 **已开源**；实体 `paper-qplanning`；交叉 VLA / action-chunking / LWD
- **ForeTime-VLA（2608.20735）：** 清华/上海 AI Lab/哈工大/云深处；无官方代码 URL；实体 `paper-foretime-vla`；交叉 π₀.₅ / WAM
- **G1 Compliant-Surface Stand-Up（2608.20852）：** 硬地演示→MuJoCo 软接触两阶段 PPO；`eval.py`+软地 checkpoint **已发布**、完整训练 **未发布**；实体 `paper-g1-compliant-surface-standup`；交叉 balance-recovery / HoST

## [2026-08-24] ingest | sources/papers/vitacphys_arxiv_2608_21355.md — 小米 ViTacPhys 视触觉物理属性自适应抓取；代码/数据集待发布

## [2026-08-24] ingest | sources/papers/sculpt_arxiv_2608_13541.md — SCULPT 减法式 3D 部件生成；项目页未列代码，记未开源

## [2026-08-24] ingest | sources/papers/t_gmp_terrain_conditioned_generative_motion_priors_arxiv_2606_06944.md — T-GMP（arXiv:2606.06944，HIT×乐聚 Kuavo）；地形条件 CVAE+AMP+Foothold；项目页 404、代码未开源；升格运动小脑 #02 实体页

## [2026-08-24] ingest | sources/blogs/kexue_fm_momentum_feature_gradient_descent_11875.md — 科学空间「动量的新理解」；新建 Feature-Space Gradient Descent 概念页；交叉 Muon / SGDM

- **触发：** 用户指定 <https://kexue.fm/archives/11875>
- **核心：** 动量 = 在线回归解 $Z^{-1}M$，逼近特征层梯度下降；统一 Newton-Muon / DeltaMomentum 与 Muon 退化关系
- **关键页：** [`wiki/concepts/feature-space-gradient-descent.md`](wiki/concepts/feature-space-gradient-descent.md)、[`wiki/methods/muon.md`](wiki/methods/muon.md)

## [2026-08-24] ingest | sources/papers/nap_control_arxiv_2605_20209.md — NaP-Control 扩散先验噪声导航；官方代码已开源

## [2026-08-24] ingest | sources/blogs/wechat_embodied_station_9_papers_vla_predict_grasp_2026-08-24.md — 具身智能小站「9 篇 VLA·预测·抓取」；新建 6 论文实体 + 技术地图；复用 PartialBiGrasp/ReflexVLA/DreamX-Phi

- **触发：** 用户指定公众号 <https://mp.weixin.qq.com/s/e0yXB8Rz4ma3CCPX8HN2CQ>；要求 9/9 独立详情节点、同一 arXiv 不重复造页
- **步骤 2.5：** Self-Demonstrated Control 项目页无代码；FlatLab/hint²/Arm-Aware 待发布；4-DoF 笔具分拣 GitHub 已开源；CPS4All 工作坊无算法仓
- **新建实体：** `paper-self-supervised-control`、`paper-cps4all`、`paper-flatlab`、`paper-hint2`、`paper-arm-aware-dexgrasp`、`paper-4dof-pen-sorting`
- **复用既有：** `paper-partialbigrasp`、`paper-reflexvla`、`paper-dreamx-phi`（仅回链博客与技术地图）
- **阅读坐标：** [`wiki/overview/vla-predict-grasp-9-papers-technology-map.md`](wiki/overview/vla-predict-grasp-9-papers-technology-map.md)

## [2026-08-24] ingest | sources/papers/gigabrain_wbc_0_5_arxiv_2608_18234.md — GigaBrain-WBC-0.5 补全 20 作者与 6 机构元数据；2026-08-24 复核项目页 Code 仍 coming soon；注册 gigaai/bjtu/usst 机构

## [2026-08-23] lint | wiki/entities/paper-language-to-navigation-goals-rgbd.md — 全量 lint 清零最后 1 条信息型预警：补「机器人视觉感知栈选型闭环」③/④ 层双向回链

## [2026-08-23] ingest | sources/papers/language_to_navigation_goals_arxiv_2607_13624.md — Language-to-Navigation-Goals（UPO）：ROS 2 VLM+RGB-D→Nav2 语义导航；代码待接收后开源

## [2026-08-23] ingest | sources/papers/eatr_stereo_arxiv_2608_17453.md — EATR-Stereo（arXiv:2608.17453，哈工大/荣耀）：头载双目 CVAT + 分段本体路由；Omega 1.0 全流程 60%；arXiv 未开源

## [2026-08-23] ingest | sources/blogs/wechat_zanehub_embodied_fm_why_self_develop_robot_body.md — Zane Hub「具身大模型为何自研本体」；升格 wiki/concepts/embodied-foundation-model-hardware-codesign.md；交叉 hub-embodied-foundation-model / foundation-policy / 量产 / 策略架构 / 跨具身

## [2026-08-23] ingest | sources/repos/drive_game.md、nordschleife_racer.md — 接入浏览器纽北驾驶引擎（引擎 MIT；nordschleife GLB/Supabase 未入库）

## [2026-08-23] ingest | sources/blogs/wechat_embodied_station_9_papers_open_source_2026-08-23.md — 具身智能小站「9 篇开源论文」；复用 Revisiting Open-Loop / 新建 8 论文实体 + 技术地图

## [2026-08-23] ingest | sources/papers/cmoe_contrastive_mixture_of_experts_icra_2026.md — CMoE（ICRA 2026，Fudan）：对比学习防 MoE 门控塌缩 + G1 高程图感知；官方 Isaac Gym 代码已开源

## [2026-08-23] ingest | sources/papers/racing_drift_rl_open_source_landscape.md — 接入 10 个赛车/漂移 RL·MPC 开源仓库；新建景观页与 F1TENTH/BARC/drift-drl/xcar 实体；10 项均已开源

## [2026-08-22] ingest | sources/papers/humanoid_rl_stack_26_learning_vision_driven_reactive_soccer_skills_fo.md — Vision-Driven Reactive Soccer（Science Robotics 2026 / arXiv:2511.03996）；项目页 Code→Zenodo 部分开源；补强实体页源码时序图与量化指标


## [2026-08-22] ingest | sources/papers/rpl_arxiv_2602_03002.md — 复核 RPL（arXiv:2602.03002，Yuanhang Zhang 等，Amazon FAR×CMU×Stanford×Berkeley）；项目页标 Code (Coming Soon)；补机构 tags 与核心信息表


## [2026-08-22] ingest | sources/blogs/wechat_embodied_station_video_contact_control_10_papers_2026-08-22.md — 具身智能小站「视频/接触/控制」10 篇；复用 AdaPT / 新建 9 论文实体 + 技术地图

## [2026-08-22] ingest | sources/papers/adapt_arxiv_2608_20087.md 等四篇 — AdaPT（G1 网球风格，部分开源）、DECOWAM（腿足 WAM+ARMDOG，未开源）、HiTac-WAM（分层触觉 WAM，未开源）、Revisiting Open-Loop（长上下文 reactive，代码待发布）

## [2026-08-22] ingest | sources/sites/adept-dexterity-github-io.md — 复核 ADEPT 项目页：16 primitive 清单、per-stage 真机累积成功率、Code 仍 Coming soon；补强实体页与 Allegro / in-hand-reorientation 交叉引用

## [2026-08-21] ingest | sources/papers/scheduled_inpainting_arxiv_2607_29133.md — Disney scheduled inpainting / GME；wiki 实体页 + 交叉 GMR/扩散/角色动画；项目页无代码

## [2026-08-21] lint | wiki/entities/paper-{adept-dexterity,roboedit,partialbigrasp,lt-mem,veragmil,dynamic-spectraformer,x2streaming-tts}.md — 合并 main 后二次清零 lint 信息型预警：补 7 篇新入库论文「对比」段 + 1 条感知栈回链 + 2 个缺页误报归档

- **触发：** 分支合并 origin/main（8 篇 ingest 新增实体页）后重跑全量 lint，失败项 0，新增 **10 条信息型预警**（三段式缺「对比」×7、感知栈回链缺失×1、缺页概念候选×2）
- **三段式补「对比」：** [`paper-adept-dexterity`](wiki/entities/paper-adept-dexterity.md)（from-scratch/naive fine-tune/KL 正则/demo-based/FMB 夹爪流水线/PCA 子空间/vision-only）、[`paper-roboedit`](wiki/entities/paper-roboedit.md)（中间表示/Ego2Robot/de novo 生成/VACE-OmniWeaving/单 LoRA/其他 human→robot 数据集/只报 perceptual metric）、[`paper-partialbigrasp`](wiki/entities/paper-partialbigrasp.md)（完整 mesh 重建/DG16M baseline/Real-Bi-Dex/Mango/单臂 top-down/同批补全线）、[`paper-lt-mem`](wiki/entities/paper-lt-mem.md)（覆盖旧图/逐次快照/VLM-Batch/SMA/统一更新策略/Hydra-0）、[`paper-veragmil`](wiki/entities/paper-veragmil.md)（3D 鼠标/BC-BC-RNN/只报 SR/刚体-流体近似/SHRIMP/ADEPT）、[`paper-dynamic-spectraformer`](wiki/entities/paper-dynamic-spectraformer.md)（空间域 CNN-Transformer/固定频带/稠密注意/频域当降算力技巧/Hui360/PartialBiGrasp）、[`paper-x2streaming-tts`](wiki/entities/paper-x2streaming-tts.md)（句级 TTS/伪流式/过早-过晚承诺/段间无状态/只报单请求延迟）——均由页内已有事实归并成表，未引入新论断
- **感知栈闭环：** [`paper-partialbigrasp`](wiki/entities/paper-partialbigrasp.md) 补回链，落 ③ 层 2D→3D 提升（只补力闭合判据需要的接触区几何，不建完整语义地图）；枢纽页 [`robot-perception-stack-selection-loop`](wiki/queries/robot-perception-stack-selection-loop.md) 同步补 `related` 与「关联页面」条目
- **缺页误报归档：** `sim-to-real` → [`concepts/sim2real.md`](wiki/concepts/sim2real.md)（连字符全称写法，与页面 stem 不同名，同 ethercat / urdf 惯例）；`zero-shot` → 迁移/评测的条件状语（zero-shot 迁移 / 0% 任务数据），已由 sim2real（全页 27 处）+ [`hub-cross-embodiment`](wiki/overview/hub-cross-embodiment.md) 覆盖，与 rgb-d / vlm 同为描述性标签而非独立可成页机制
- **合并冲突：** `log.md` 顶部两侧各自 prepend，按倒序时间线保留双方（08-21 ingest 在上、08-20 lint 在下）；`paper-hydra-0.md` 二次合并自动干净（main 补 sources 条目，本分支补「对比」段，互不重叠）
- **验证：** lint「✅ 所有检查通过！」（0 失败 / 0 信息型，覆盖率 3177/3177）；`eval_search_quality` 40/40；`ruff check` + `ruff format --check` 通过；`pytest` 全绿

## [2026-08-21] ingest | sources/blogs/wechat_embodied_station_8_papers_world_model_memory_2026-08-21.md — 8 篇世界模型/长期记忆盘点；新建 5 独立 paper 节点，回链 ADEPT/GigaBrain/Hydra

## [2026-08-21] ingest | sources/papers/{gigabrain_wbc_0_5,adept,roboedit}_arxiv_2608_*.md — GigaBrain-WBC-0.5 BWM 环境交互跟踪（Code coming soon）；ADEPT 灵巧 RL 预训练+后训练（Code Coming soon）；RoboEdit 人类视频→RoboEdit-14M（无官方代码 URL）

## [2026-08-20] lint | wiki/entities/paper-{fetchman,hydra-0,instant-episode-repetition,prism-grpo}.md + robo-orchard-lab.md — 清空全量 lint 信息型预警：补 4 篇 2608 论文「对比」段 + 2 条枢纽双向回链 + 2 个缺页误报归档

- **触发：** 跑一遍全量知识库 lint（`python3 scripts/lint_wiki.py`），失败项 0，余 **8 条信息型预警**（三段式缺「对比」×4、枢纽回链缺失×2、缺页概念候选×2）
- **三段式补「对比」：** [`paper-fetchman`](wiki/entities/paper-fetchman.md)（VIRAL/DoorMan/纯加演示 BC/Temporal GRPO/端到端 WBC/SigLIP+absolute 消融）、[`paper-hydra-0`](wiki/entities/paper-hydra-0.md)（Cosmos native action/ATI-Wan-Move/OSCAR/SC3-Eval/Ctrl-World/常规 WAM）、[`paper-instant-episode-repetition`](wiki/entities/paper-instant-episode-repetition.md)（replay-PER/SIL/SAC-TD3 基线/on-policy/开环 replay/prioritized 多轨迹）、[`paper-prism-grpo`](wiki/entities/paper-prism-grpo.md)（Binary GRPO/task-specific progress reward/RL-ZVP/group-norm/Temporal GRPO/学习式 RM/只对 same-outcome 加 quality）——均由页内已有事实归并成表，未引入新论断
- **评测基准闭环：** [`paper-hydra-0`](wiki/entities/paper-hydra-0.md) 补回链，定位为 ②层（世界模型预测保真度）向 ③层（策略成功率）的桥接：RoboLab 开环 replay r=0.96 不等于闭环 prospective 成功率
- **感知栈闭环：** [`robo-orchard-lab`](wiki/entities/robo-orchard-lab.md) 补回链，`projects/bip3d_grounding` 落 ③层 2D→3D grounding、`finegrasp` 落 ④层下游消费
- **双向：** 两个 Query 枢纽页 [`embodied-eval-benchmark-selection-loop`](wiki/queries/embodied-eval-benchmark-selection-loop.md) / [`robot-perception-stack-selection-loop`](wiki/queries/robot-perception-stack-selection-loop.md) 同步补 `related` 与「关联页面」条目并 bump `updated`
- **缺页误报归档：** `action` / `ros2` 两个候选经核查均属「slug ≠ 页面 stem」或「数据字段名/API 键名」，已按既有惯例登记进 `MISSING_CONCEPT_COVERED_ELSEWHERE` 并附归属说明——`ros2` → [`concepts/ros2-basics.md`](wiki/concepts/ros2-basics.md)（+ rmw-interface / dds-communication / ros2-vs-lcm）；`action` → [`formalizations/mdp.md`](wiki/formalizations/mdp.md) 动作空间 $A$ + [`methods/action-chunking.md`](wiki/methods/action-chunking.md) + [`concepts/world-action-models.md`](wiki/concepts/world-action-models.md)（命中处全是 RobotWin JSONL 帧类型 / UMI 20 维导出 / ROS pick-place 接口等 token）
- **验证：** lint「✅ 所有检查通过！」（0 失败 / 0 信息型）；`eval_search_quality` 40/40；`ruff check` + `ruff format --check` 通过；`pytest` 全绿（生成 `make export graph` 派生物后 3 项 export 依赖测试亦通过）
- **未提交：** `exports/` `docs/exports/` 统计派生物——本会话是 shallow clone（108 commits），`generate_link_graph` 的 git 首次加入日会退化成 log.md 兜底，交由 `export.yml`（`fetch-depth: 0`）重算

## [2026-08-20] ingest | sources/papers/tramp_vision_assisted_bipedal_locomotion_ieee_lra_2026.md — SJTU TRAMP（IEEE RA-L 2026）：单阶段深度+MoE+平地/楼梯地形相关 AMP；ResearchGate 全文入口；确认未开源

## [2026-08-20] ingest | sources/repos/grove-g1.md — 接入 G1 ROS 2 自主栈 Grove-G1（Nav2+MoveIt+BT），已开源可运行，真机目标检测待补

## [2026-08-20] ingest | sources/papers/warp_arxiv_2606_29940.md — WARP 离线人类全身演示闭式 c-SEW 重定向；升格 paper-warp-whole-body-retargeting；开源结论：截至入库日未开源

- **资料：** [WARP: Whole-Body Retargeting for Learning from Offline Human Demonstrations](https://arxiv.org/abs/2606.29940)（Georgia Tech / Danfei Xu，[项目页](https://warp-retargeting.github.io/)）
- **归档：** [`sources/papers/warp_arxiv_2606_29940.md`](sources/papers/warp_arxiv_2606_29940.md)、[`sources/sites/warp-retargeting-github-io.md`](sources/sites/warp-retargeting-github-io.md)
- **开源核查（步骤 2.5）：** **截至 2026-08-20 项目页未列 GitHub / 数据集**
- **升格：** [`wiki/entities/paper-warp-whole-body-retargeting.md`](wiki/entities/paper-warp-whole-body-retargeting.md) — c-SEW、palm 硬约束、lazy mobile-base、Meta Quest 60 Hz 采集、RB-Y1 真机四任务
- **交叉：** [`wiki/concepts/motion-retargeting.md`](wiki/concepts/motion-retargeting.md)、[`wiki/tasks/loco-manipulation.md`](wiki/tasks/loco-manipulation.md)

## [2026-08-20] ingest | sources/papers/grip_arxiv_2603_16233.md — GRIP 稀疏 IMU+鞋垫物理 MoCap 与 PRISM 数据集；代码/数据已开源

## [2026-08-20] ingest | sources/papers/haf_arxiv_2608_16837.md — 接入 HAF 层次 action flow + 频谱潜空间 RL；升格 paper-haf-humanoid-vla-adaptation；开源结论：截至入库日未开源

## [2026-08-20] ingest | sources/blogs/generalist_gen15_one_shot.md — GEN-1.5 one-shot physical prompting；升格 wiki/entities/generalist-gen15-one-shot.md；更新 generalist-ai-robotics / foundation-policy / embodied-scaling-laws / manipulation / hub-cross-embodiment / generalist-gen1-thousand-hands
- **资料：** [GEN-1.5: Embodied Foundation Models are One-Shot Learners](https://generalistai.com/blog/gen-1.5)（Generalist AI，2026-08-19）
- **归档：** [`sources/blogs/generalist_gen15_one_shot.md`](sources/blogs/generalist_gen15_one_shot.md)
- **开源核查：** **确认未开源**（公司站无 GitHub / HF）
- **升格：** [`wiki/entities/generalist-gen15-one-shot.md`](wiki/entities/generalist-gen15-one-shot.md) — physical prompting、组合示范、sim 提示真机、1–10 步微调、即兴工具使用
- **交叉：** [`wiki/entities/generalist-ai-robotics.md`](wiki/entities/generalist-ai-robotics.md)、[`wiki/concepts/foundation-policy.md`](wiki/concepts/foundation-policy.md)、[`wiki/concepts/embodied-scaling-laws.md`](wiki/concepts/embodied-scaling-laws.md)、[`wiki/tasks/manipulation.md`](wiki/tasks/manipulation.md)、[`wiki/overview/hub-cross-embodiment.md`](wiki/overview/hub-cross-embodiment.md)

## [2026-08-20] ingest | 四篇 arXiv:2608 — IER/FetchMan/Hydra-0/Prism-GRPO 实体页；IER 已开源，FetchMan/Hydra-0 未开源，Prism 基于 SimpleVLA-RL

## [2026-08-20] ingest | sources/papers/radmesh_arxiv_2608_17182.md — RADmesh ECCV 2026 Oral；threedle/radmesh 已开源

## [2026-08-20] ingest | sources/{repos/horizon_robotics_holomotion,sites/holomotion-docs,papers/holomotion_arxiv_2605_15336}.md — 再核 HoloMotion 三链（GitHub/Pages/arXiv:2605.15336）：已开源 v1.4.1 Docker、634★、2000+h/0.4B/MPKPE−40%；wiki/entities/holomotion.md 补实验·工程·局限·结论

## [2026-08-20] ingest | sources/repos/horizon_robotics_robo_orchard_lab.md — 接入 RoboOrchardLab 训练框架（已开源 Apache-2.0）；升格 wiki/entities/robo-orchard-lab.md

## [2026-08-20] ingest | sources/blogs/wechat_zanehub_humanoid_career_entry_for_generalists.md — 普通人切入人形赛道（结构/执行器/测试/制造）；升格 wiki/roadmaps/humanoid-practitioner-entry-roadmap.md

## [2026-08-20] ingest | sources/papers/gaussian_lic2_arxiv_2507_04004.md — Gaussian-LIC2 实时 LIC 3DGS-SLAM；代码已开源，自采评测集待发布

## [2026-08-19] lint | wiki/queries/*-selection-loop.md — 清空全量 lint 信息型预警：补齐 3 条知识链枢纽双向回链

- **触发：** 跑一遍全量知识库 lint（`python3 scripts/lint_wiki.py`），失败项 0，余 **3 条信息型预警**（枢纽回链缺失）
- **评测基准闭环：** [`paper-h2r-bench`](wiki/entities/paper-h2r-bench.md) 补 ②层（世界模型预测保真度）回链；[`paper-robosynchallenge`](wiki/entities/paper-robosynchallenge.md) 补 ③/④层（策略成功率 + sim↔real 校准）回链
- **感知栈闭环：** [`paper-sap-nav`](wiki/entities/paper-sap-nav.md) 补 ③层（2D→3D 提升与在线语义建图）回链——QSSR 可查询空间–语义表征 + AVV 主动视点验证
- **双向：** 两个 Query 枢纽页 [`embodied-eval-benchmark-selection-loop`](wiki/queries/embodied-eval-benchmark-selection-loop.md) / [`robot-perception-stack-selection-loop`](wiki/queries/robot-perception-stack-selection-loop.md) 同步补 `related` 与「关联页面」条目并 bump `updated`
- **验证：** lint「✅ 所有检查通过！」（0 失败 / 0 信息型）；`make ci-preflight` 导出质量 12/12 通过

## [2026-08-19] ingest | sources/blogs/wechat_zanehub_humanoid_mass_production_experience.md — 人形量产经验（DFM·三大核心件·良率/CPK·供应链·可靠性）；升格 wiki/concepts/humanoid-mass-production-engineering.md

- **触发：** 用户指定 <https://mp.weixin.qq.com/s/CARW0vvd4doO1htt0Q1bHg>；要求自动合并 PR
- **工具：** Agent Reach + wechat-article-for-ai（Camoufox；`--no-images`）
- **来源：** [`sources/blogs/wechat_zanehub_humanoid_mass_production_experience.md`](sources/blogs/wechat_zanehub_humanoid_mass_production_experience.md)
- **新建概念：** [`wiki/concepts/humanoid-mass-production-engineering.md`](wiki/concepts/humanoid-mass-production-engineering.md) — 量产经验 = 制造可行性；谐波/PRS/无框电机工艺定型、S 曲线良率、CPK/PPAP、跨行业迁移与主流厂商路径对照
- **开源核查（步骤 2.5）：** **不适用** — 公众号工程解读，无项目页 / 代码仓
- **交叉更新：** [`wiki/overview/humanoid-hardware-101-supply-chain-economics.md`](wiki/overview/humanoid-hardware-101-supply-chain-economics.md)、[`wiki/overview/humanoid-hardware-101-actuation-sensing-chain.md`](wiki/overview/humanoid-hardware-101-actuation-sensing-chain.md)、[`wiki/overview/humanoid-hardware-101-technology-map.md`](wiki/overview/humanoid-hardware-101-technology-map.md)、[`wiki/concepts/humanoid-knee-harmonic-drive-limits.md`](wiki/concepts/humanoid-knee-harmonic-drive-limits.md)、[`wiki/concepts/planetary-roller-screw-humanoid-leg-actuation.md`](wiki/concepts/planetary-roller-screw-humanoid-leg-actuation.md)、[`sources/blogs/wechat_zanehub_humanoid_leg_knee_why_not_harmonic.md`](sources/blogs/wechat_zanehub_humanoid_leg_knee_why_not_harmonic.md)

## [2026-08-19] ingest | sources/papers/tau0_vla_arxiv_2608_16885.md — τ₀-VLA 分层基础模型与世界模型引导测试时计算入库

- **触发：** 用户指定 ingest τ₀-VLA（<https://tau0-vla.github.io/>；上海创智学院、智元 Finch、港中文）
- **步骤 2.5：** 项目页链 arXiv、GitHub、HF → **部分开源**：低层 VLA 权重 + 后训练/deploy 已发布；README **[2026.08.19]** 高层 policy + TTC **逐步发布**
- **来源：** [`sources/papers/tau0_vla_arxiv_2608_16885.md`](sources/papers/tau0_vla_arxiv_2608_16885.md)、[`sources/sites/tau0-vla-github-io.md`](sources/sites/tau0-vla-github-io.md)、[`sources/repos/sii_research_tau_0_vla.md`](sources/repos/sii_research_tau_0_vla.md)
- **升格：** [`wiki/entities/paper-tau0-vla.md`](wiki/entities/paper-tau0-vla.md) — P/W/V/F 高层 TTC、可修订记忆、40 维 MoT 低层、长程四任务表
- **交叉：** τ₀-WM、π₀.₅、LingBot-VLA 2.0、VLA、manipulation
- **机构：** `shanghai-innovation-institute`、`agibot`、`cuhk`

## [2026-08-19] ingest | sources/blogs/wechat_embodied_station_world_model_exec_10_papers_2026-08-19.md — 具身智能小站 10 篇世界模型与真实执行盘点；复用 4 / 新建 6 论文实体 + 技术地图

## [2026-08-18] ingest | sources/blogs/wechat_shenlan_realab_14_papers_2026.md — REALab 14 篇技术地图；开源核查 10+ 项已开

## [2026-08-18] ingest | sources/papers/cmp_arxiv_2608_03234.md — 加深 CMP（arXiv:2608.03234）附录超参/任务上下文/G1 全表；复核仍无官方代码；交叉 PFM-HR / PDF-HR / ASE

- **触发：** 用户指定 ingest *Learning Context-Aware Motion Priors for Humanoid Control*（arXiv 预印本；HKUST-GZ；Yunyang Mo / Yi Gu / Yangchen Zhou / Hanyang Cao / Renjing Xu），并要求自动合并 PR
- **已有页：** 不另造页，加深 [`wiki/entities/paper-cmp.md`](wiki/entities/paper-cmp.md)（2026-08-06 首入库）
- **步骤 2.5：** 打开 [arXiv:2608.03234](https://arxiv.org/abs/2608.03234) HTML；无项目页。正文写 clip manifest「included in the code」；GitHub 检索无官方仓 → **宣称有代码、公开入口未列**
- **加深：** 五任务 \(c\) 维数、Table 1 std、失衡 ×2–×100、附录 E G1 全表、AMP/SMP 超参（\(\lambda_{\mathrm{res}},\tau,\alpha\) clip）、Isaac Gym 4096
- **交叉：** AMP / SMP / ASE / CALM / C·ASE / MimicKit / G1 / PFM-HR / PDF-HR（同组相邻 arXiv:2608.03227）
- **开源结论：** 截至 2026-08-18 无可运行官方代码；源码运行时序图不适用
- **机构：** 已有 `hkust-gz`

## [2026-08-18] ingest | sources/sites/ibrics-lar-upatras.md + sources/repos/go2_flip_to.md — 接入 IBRICS 项目页与 Go2 SE(3) 切空间 TO；AHMP / 浮动基参数化对比均已开源

- **触发：** 用户指定 ingest <https://lar.upatras.gr/projects/ibrics.html>、<https://github.com/yusongmin1/go2_flip_TO>，并要求自动合并 PR
- **步骤 2.5：** 项目页无 Code 按钮。AHMP 论文/README → [hucebot/ahmp](https://github.com/hucebot/ahmp)（BSD-2-Clause，Docker + `trajopt_parallel.py`）。对比论文声明 [upatras-lar/se3_trajopt](https://github.com/upatras-lar/se3_trajopt)；用户仓 `go2_flip_TO` 未标 fork，含 Go2 AMP 50 Hz 导出与 MUMPS。**两线均已开源，均无真机脚本。**
- **来源：** [`sources/sites/ibrics-lar-upatras.md`](sources/sites/ibrics-lar-upatras.md)、[`sources/papers/ahmp_humanoids_2025.md`](sources/papers/ahmp_humanoids_2025.md)、[`sources/papers/se3_tangent_to_arxiv_2508_11520.md`](sources/papers/se3_tangent_to_arxiv_2508_11520.md)、[`sources/repos/ahmp.md`](sources/repos/ahmp.md)、[`sources/repos/se3_trajopt.md`](sources/repos/se3_trajopt.md)、[`sources/repos/go2_flip_to.md`](sources/repos/go2_flip_to.md)
- **升格：** [`wiki/entities/paper-ahmp.md`](wiki/entities/paper-ahmp.md) — CEM-MD 接触发现；Talos 扶手 20/20、<200 s；烟囱 1 m ~85%、3 m ~50%。[`wiki/entities/paper-se3-tangent-to.md`](wiki/entities/paper-se3-tangent-to.md) — 五种浮动基；G1/Go2 空翻仅切空间翻成功
- **交叉：** 轨迹优化 / SE(3) / 李群 / FARO / DSMS / Pinocchio / Crocoddyl / TO vs RL / MPC 求解器 / 浮动基动力学
- **开源结论：** AHMP 与 SE3_TrajOpt 可运行；`go2_flip_TO` 为 AMP 导出扩展；源码运行时序图已写
- **机构：** 注册 `patras`（帕特雷大学（University of Patras））、`athena-rc`（雅典娜研究中心（Athena RC））

## [2026-08-18] ingest | sources/papers/cref_arxiv_2603_29452.md — CReF 深度条件人形行走（arXiv:2603.29452）；项目页仅为静态站，训练代码确认未开源

- **触发：** 用户指定 ingest <https://arxiv.org/abs/2603.29452>、<https://arxiv.org/pdf/2603.29452>，并要求自动合并 PR
- **步骤 2.5：** 打开项目页 <https://cometlogic.github.io/cref/>；源仓 [cometlogic/cref](https://github.com/cometlogic/cref) 仅 GitHub Pages 静态站（`index.html` / `scripts.js` / `res/`），无训练/推理入口。论文未承诺即将开源 → **确认未开源**
- **来源：** [`sources/papers/cref_arxiv_2603_29452.md`](sources/papers/cref_arxiv_2603_29452.md)、[`sources/sites/cometlogic-cref-github-io.md`](sources/sites/cometlogic-cref-github-io.md)
- **升格：** [`wiki/entities/paper-cref.md`](wiki/entities/paper-cref.md) — 本体查询交叉注意 + GRF + GRU highway；足端点云可支撑落脚奖励；X2 Ultra 室内楼梯 20/20、40 cm 台、80 cm 沟；仿真总体 SR 90.45% vs 重实现 HPL 74.57%
- **交叉：** 楼梯枢纽 / Humanoid Locomotion / SSR / HPL / Now You See That / DPL / Hiking / PIE / Privileged Training / Footstep Planning / Sim2Real / Terrain Adaptation
- **开源结论：** 训练代码未开源；源码运行时序图不适用
- **机构：** 已有 `zju`、`sdu`、`agibot`（平台标签）

## [2026-08-18] ingest | sources/papers/harnesseval_w_arxiv_2608_16859.md — 接入 MirroS HarnessEval-W 交互式世界 agentic 评测；评测代码已开源、HF 全量案例待发

- **触发：** 用户指定 ingest 项目页 <https://mirros-lab.github.io/HarnessEval-W>、论文 <https://arxiv.org/abs/2608.16859>、代码 <https://github.com/mirros-lab/harnesseval-w>、Blog <https://mirros.ai/blog/harnesseval>，并要求自动合并 PR
- **步骤 2.5：** 打开项目页，Code 指向 GitHub；仓内 `harnesseval eval/plan/generate/verify`、11 个 skill、`benchmark/plans` 与捆绑 demo **已开源**。README TODO：HF 全量/子集案例与托管提交服务 **待发布**；项目页 V1 Leaderboard **Coming Soon**。许可：README 宣称 Apache-2.0，GitHub License 字段未识别
- **来源：** [`sources/papers/harnesseval_w_arxiv_2608_16859.md`](sources/papers/harnesseval_w_arxiv_2608_16859.md)、[`sources/sites/harnesseval-w-github-io.md`](sources/sites/harnesseval-w-github-io.md)、[`sources/repos/harnesseval-w.md`](sources/repos/harnesseval-w.md)、[`sources/blogs/mirros_harnesseval.md`](sources/blogs/mirros_harnesseval.md)
- **升格：** [`wiki/entities/paper-harnesseval-w.md`](wiki/entities/paper-harnesseval-w.md) — 三轴八设定 + 证据树；330 例 × 18 模型；Intentional ρ=0.93；Physical 成对准确率 31.9%→71.7%；微调常抬 Revisit、打掉 Intentional/Physical
- **交叉：** 评测枢纽 / Query ② 层 / WorldScore / EWMBench / 生成式 WM / Video-as-Simulation / ABot-World-0 / Wan / Cosmos 3
- **开源结论：** 评测管线可运行；全量案例未上 HF；源码运行时序图覆盖 eval CLI + demo
- **机构：** 注册 `mirros`（镜界（MirroS））

## [2026-08-18] ingest | sources/blogs/wechat_embodied_station_contact_predict_adapt_10_papers_2026-08-18.md — 接入具身智能小站「接触/预测/适应」10 篇；复用 Seeker

- **触发：** 用户指定 ingest <https://mp.weixin.qq.com/s/IxmKI4_JYy1KBfp_JCZFLw>
- **抓取：** Agent Reach + wechat-article-for-ai（`--no-images`）
- **步骤 2.5：** 逐篇打开项目页/GitHub。可运行：AutoPSO、HUI360 基线、双臂 DDPM、nav-ps-balance、Mind-the-Context notebook、Seeker（复用）。清单：TF-ART Awesome。空仓：顶层布料分割。仅项目页：BooST。占位 README：DreamX-Phi（权重待赛后）
- **已有页：** 复用 [`wiki/entities/paper-seeker.md`](wiki/entities/paper-seeker.md)
- **升格：** 新建 9 个 `paper-*` + [`wiki/overview/contact-predict-adapt-10-papers-technology-map.md`](wiki/overview/contact-predict-adapt-10-papers-technology-map.md)
- **交叉：** 触觉链 / Awesome Touch / 双臂 / IL / LIBERO / PGIF-MPPI / iCrowdNav / Wan / Ctrl-World / 生成式 WM
- **开源结论：** 见上；空仓与占位仓的源码运行时序图不适用
- **机构：** 注册 `cambridge`、`cea`

## [2026-08-17] ingest | sources/papers/reflexvla_arxiv_2608_14379.md — 加深交大 ReflexVLA；复核录用后开源，并与同名流式 Reflex 分流

- **触发：** 用户指定 ingest *Reflex: Enabling Fast and Predictive Vision-Language-Action Models for Reaction-Critical Manipulation*（交大 陈宇轩 / 张婉若 / 李晓，arXiv 2026-08），并要求自动合并 PR
- **步骤 2.5：** 打开 <https://reflexvla.github.io/>；Code 按钮 title「Code will be released after the paper is accepted」，无 GitHub URL → **宣称录用后开源**
- **已有页：** 不另造页，加深 [`wiki/entities/paper-reflexvla.md`](wiki/entities/paper-reflexvla.md)
- **加深：** 六任务全表、LIBERO 分项、作者通讯、同名分流 arXiv:2607.14695
- **交叉：** LIBERO / GSR / WAM 异步部署 / RTCF
- **开源结论：** 宣称录用后开源；源码运行时序图不适用

## [2026-08-17] ingest | sources/papers/spd_corl_2026.md — 接入斯坦福/MIT/Scale AI 的 SPD 仿真灵巧手预训练；宣称将开源、项目页未列代码

- **触发：** 用户指定 ingest *Pre-training Visual Dexterity in Simulation*（<https://spd.bot/>），并要求自动合并 PR
- **步骤 2.5：** 打开项目页仅 PDF；论文宣称释放 spd-75h / spd-vr / 六套场景；GitHub 检索无官方仓 → **宣称将开源 / 待核实**
- **来源：** [`sources/papers/spd_corl_2026.md`](sources/papers/spd_corl_2026.md)、[`sources/sites/spd-bot.md`](sources/sites/spd-bot.md)
- **升格：** [`wiki/entities/paper-spd.md`](wiki/entities/paper-spd.md) — VR 仿真采 75 h → 222M DiT 预训练 → 56-DoF 真机每任务 1–2 h 微调；五项任务胜过从零 BC；历史+短 chunk 收益最大
- **交叉：** 遥操作 / 灵巧采数指南 / Diffusion Policy / Action Chunking / EgoScale / Sim2Real / TeleDexter / π0
- **开源结论：** 宣称将开源；源码运行时序图不适用；注册 Scale AI 机构标签

## [2026-08-17] ingest | sources/papers/fail_passive_gap_arxiv_2608_02809.md — 接入西门子工业人形 Fail-Passive Gap；确认未开源

- **触发：** 用户指定 ingest Ding / Cui / Wang / Wen（西门子基础技术、Siemens Corporation 普林斯顿）arXiv 预印本 *Toward Certified Functional Safety for Industrial Humanoid Robots*，并要求自动合并 PR
- **步骤 2.5：** 无独立项目页，仅 [arXiv:2608.02809](https://arxiv.org/abs/2608.02809)；论文未列 GitHub / 数据集；检索无官方实现 → **确认未开源**
- **来源：** [`sources/papers/fail_passive_gap_arxiv_2608_02809.md`](sources/papers/fail_passive_gap_arxiv_2608_02809.md)
- **升格：** [`wiki/entities/paper-fail-passive-gap.md`](wiki/entities/paper-fail-passive-gap.md) — 主动安全态 vs ISO 13849 fail-passive；认证外部 D–E–R 当量尺；缺口在 SDA↔平衡策略；G1 EDU 最坏约 1.1 s；不宣称端到端 PL e / SIL 3
- **交叉：** 安全状态机 / 整机配电 / Safety Filter / Capture Point / Balance Recovery / 系统工程枢纽 / Unitree G1 / 控制环路延迟
- **开源结论：** 确认未开源；源码运行时序图不适用

## [2026-08-17] fix(ux): 美元价不被 KaTeX 成对 `$...$` 吞掉（entity-all-hands-up）

- **现象：** [`wiki/entities/all-hands-up.md`](wiki/entities/all-hands-up.md) 工程实践段「约 $50k … 约 $7.5k」被 `docs/main.js` 行内公式正则配对，详情页把中间加粗与中文吞成公式
- **修复：** `renderInlineMarkdown` 跳过货币配对（CJK / `**` / `$30,000–$90,000` 价带）；`$O(n)$`、`$0.99$` 仍走 KaTeX

## [2026-08-17] ingest | sources/blogs/current_robotics_currentworld.md — 接入 CurrentWorld-0 跨本体交互世界模拟器；确认未开源

- **触发：** 用户指定 ingest <https://current-robotics.com/blog/currentworld>，并要求自动合并 PR
- **步骤 2.5：** 博客与公司首页 <https://current-robotics.com/> 无 GitHub / Hugging Face / 权重 / 数据集；GitHub 检索 `CurrentWorld-0`、`current-robotics` 无官方仓 → **确认未开源**
- **来源：** [`sources/blogs/current_robotics_currentworld.md`](sources/blogs/current_robotics_currentworld.md)、[`sources/sites/current-robotics-com.md`](sources/sites/current-robotics-com.md)
- **升格：** [`wiki/entities/current-robotics-currentworld.md`](wiki/entities/current-robotics-currentworld.md) — 不统一低层动作空间；多视角 + 力触觉；失败态回滚/分支 + Human-in-the-World-Model 后训练（π0 / π0.5 / DP 自报）
- **交叉：** Curr-0 / 生成式世界模型 / 虚拟沙盒 / 训练闭环 taxonomy / 评估闭环 04 / Ctrl-World / ViTacWorld / GigaWorld-1 / Hi-WM / OSCAR / loco-manipulation / teleoperation / Wuji Hand
- **开源结论：** 确认未开源；源码运行时序图不适用；定量图为官方自报

## [2026-08-17] ingest | sources/repos/robot-descriptions-py.md — 加深 Python 加载器，并接入 Awesome / fiveages-sim / URDF Files Dataset

- **触发：** 用户指定 ingest [robot_descriptions.py](https://github.com/robot-descriptions/robot_descriptions.py)、[awesome-robot-descriptions](https://github.com/robot-descriptions/awesome-robot-descriptions)、[fiveages-sim/robot_descriptions](https://github.com/fiveages-sim/robot_descriptions)、[URDF Files Dataset](https://github.com/Daniella1/urdf_files_dataset)，并要求自动合并 PR
- **步骤 2.5：**
  - robot_descriptions.py：**Apache-2.0，已开源、可运行**（PyPI 3.1.0 / conda-forge / `uvx`）；机型许可证逐条上游，部分 NC / 厂商图形条款
  - Awesome：**CC0-1.0 列表已开源**；无运行时代码；姊妹加载器即上项
  - fiveages-sim：**Apache-2.0 主树已开源**；`arms_ros2_control` / `robot_usds` 亦公开；**Agibot G2 子模块 private（部分开源）**
  - URDF Files Dataset：**MIT，已开源**（322 Bundle + 分析脚本）；配套 RA-L 2024 / arXiv:2308.00514；仓于 2024-04 冻结
- **来源：** [`sources/repos/robot-descriptions-py.md`](sources/repos/robot-descriptions-py.md)、[`sources/repos/awesome-robot-descriptions.md`](sources/repos/awesome-robot-descriptions.md)、[`sources/repos/fiveages-sim-robot-descriptions.md`](sources/repos/fiveages-sim-robot-descriptions.md)、[`sources/repos/urdf_files_dataset.md`](sources/repos/urdf_files_dataset.md)、[`sources/papers/understanding_urdf_dataset_arxiv_2308_00514.md`](sources/papers/understanding_urdf_dataset_arxiv_2308_00514.md)
- **升格：** 加深 [`wiki/entities/robot-descriptions-py.md`](wiki/entities/robot-descriptions-py.md)；新建 Awesome / fiveages-sim / URDF Dataset 实体与 [`wiki/comparisons/robot-description-catalogs.md`](wiki/comparisons/robot-description-catalogs.md)
- **交叉：** URDF / Pinocchio / MuJoCo / Isaac Sim / URDD / URDF-Studio / Robot Viewer / Unitree / ros2_control / Pinocchio 快速上手
- **开源结论：** 四个入口均可公开获取；G2 子模块未公开；Dataset 非日常仿真源

## [2026-08-17] ingest | sources/repos/lw_benchhub_tour.md — 接入 LW BENCHHUB TOUR：SmolVLA 双臂 Piper 闭环、cuRobo 闸门与自过滤飞轮

- **触发：** 用户指定 ingest <https://github.com/GimpelZhang/lw_benchhub_tour>，并要求自动合并 PR
- **步骤 2.5：**
  - Tour 仓 **Apache-2.0，已开源、可运行**（Stage 1/2/4 脚本 + GitHub Wiki）；依赖 Isaac Sim 5.1 / Lab 2.3.2 / Arena 0.1.1 + 大显存 GPU
  - 官方 [LightwheelAI/LW-BenchHub](https://github.com/LightwheelAI/LW-BenchHub) **Apache-2.0，已开源**；项目页 [lightwheel.ai/lightwheel-platform](https://lightwheel.ai/lightwheel-platform) 为企业叙事，代码入口走 GitHub
  - HF：`LightwheelAI/smolvla-double-piper-pnp`、`lw_benchhub_env`、`Lightwheel-Tasks-Double-Piper`
- **来源：** [`sources/repos/lw_benchhub_tour.md`](sources/repos/lw_benchhub_tour.md)、[`sources/repos/lw-benchhub.md`](sources/repos/lw-benchhub.md)、[`sources/sites/lightwheel-platform.md`](sources/sites/lightwheel-platform.md)
- **升格：** [`wiki/entities/lw-benchhub-tour.md`](wiki/entities/lw-benchhub-tour.md) — EnvHub 五层栈；Stage 1 基线 40%；Stage 2 live IK 闸门；Stage 4 scripted PnP 推碗失败 vs 自过滤 10 ep / 6527 帧
- **交叉：** LeRobot / Isaac Lab / cuRobo / VLA / 双臂 / VLA 部署指南 / SO-101 课 / VLA 开源复现景观
- **开源结论：** 已开源可跑；钉版本与 numpy/warp ABI 补丁是复现硬约束

## [2026-08-17] ingest | sources/sites/symbiosis-robotics-dpc.md — 接入 DPC 直接感知控制；确认未开源

- **触发：** 用户指定 ingest <https://symbiosis-robotics.com/research/dpc/en/>，并要求自动合并 PR
- **步骤 2.5：** 项目页页头/页脚无 GitHub、Hugging Face、PDF 或数据集；Citation 为 *Symbiosis Robotics Blog*（2026-08）→ **确认未开源**
- **来源：** [`sources/sites/symbiosis-robotics-dpc.md`](sources/sites/symbiosis-robotics-dpc.md)、[`sources/blogs/symbiosis_dpc_direct_perception_control.md`](sources/blogs/symbiosis_dpc_direct_perception_control.md)
- **升格：** [`wiki/entities/paper-dpc.md`](wiki/entities/paper-dpc.md) — 去掉 SONIC 式 \(Z_t\) 接口，Symbiotic Attention + DriftDistill；自报 15,010 h G1 关节语料
- **交叉：** loco-manipulation / SONIC / VLA+低层控制器 query / MotionWAM / ω-0 / Gemini Robotics / VLA / π0.7
- **开源结论：** 确认未开源；源码运行时序图不适用；无公开成功率表

## [2026-08-17] ingest | sources/papers/nestdex_arxiv_2608_13362.md — 接入 NestDex 嵌套 copilot 灵巧遥操作

- **触发：** 用户指定 ingest NestDex（悉尼大学 ACFR / PAIR Lab、范德堡；Zhao / Tang / Ba / Zhi；2026-08 arXiv 预印本），并要求自动合并 PR
- **步骤 2.5：** 项目页 <https://aus.bot/research/nestdex/> 为 SPA；JS bundle 仅链 arXiv 与 PAIR Lab 研究站，**无 GitHub / Hugging Face**；GitHub 检索无官方训练仓 → **确认未开源**
- **来源：** [`sources/papers/nestdex_arxiv_2608_13362.md`](sources/papers/nestdex_arxiv_2608_13362.md)、[`sources/sites/aus-bot-nestdex.md`](sources/sites/aus-bot-nestdex.md)
- **升格：** [`wiki/entities/paper-nestdex.md`](wiki/entities/paper-nestdex.md) — 内层本体感觉手技能 + 1-DoF clutch 采数，外层 visuomotor 部署卸 copilot；H-VAE 20→10-D；六任务 Copilot 采数 100%
- **交叉：** Teleoperation / 灵巧采数指南 / 手套 vs 视觉 / Action Chunking / BC / AutoIntervene / TeleDexter / 双臂 / Manipulation / 深度遥操作 Stage 4–5
- **开源结论：** 确认未开源；源码运行时序图不适用

## [2026-08-17] ingest | sources/sites/letools-lejurobot.md — 接入 LeTools 门户/文档/双仓与 LET-Base、REAL-I 数据集

- **触发：** 用户指定官方站+AI 助手、docs.html?type=learning、letools_opensource、skills/README、LeTools-Learning、ICRA REAL-I HF、LET-Base-Dataset 数据卡；并要求自动合并 PR
- **步骤 2.5：**
  - 产品站 / 文档 SPA / KuavoChat：**公开门户**；助手后端为托管 Worker（DeepSeek v4-flash），**非完整开源 agent**
  - [LeTools-Learning](https://github.com/LejuRobotics/LeTools-Learning)：**GPL-3.0，已开源、可运行**（rosbag→LeRobot v3→train/eval）
  - [letools_opensource](https://github.com/LejuRobotics/letools_opensource)：**已开源、可运行**；GitHub **无 SPDX**；SkillBase + 行为树；主力 `leju_wheeled`
  - LET-Base HF：**CC-BY-NC-SA-4.0、ungated**；快照 25824 bag / 无 hdf5
  - REAL-I HF：**ungated、卡上无 license**；sim 3×1000 bag；**real 未更新**；另有未文档化 `vienna/`
- **来源：** [`sources/sites/letools-lejurobot.md`](sources/sites/letools-lejurobot.md)、[`sources/sites/letools-docs.md`](sources/sites/letools-docs.md)、[`sources/repos/letools-learning.md`](sources/repos/letools-learning.md)、[`sources/repos/letools_opensource.md`](sources/repos/letools_opensource.md)、[`sources/datasets/let-base-dataset.md`](sources/datasets/let-base-dataset.md)、[`sources/datasets/kuavo-data-challenge-icra.md`](sources/datasets/kuavo-data-challenge-icra.md)
- **升格：** [`wiki/entities/letools.md`](wiki/entities/letools.md)、[`wiki/entities/let-base-dataset.md`](wiki/entities/let-base-dataset.md)、[`wiki/entities/icra-2026-real-i.md`](wiki/entities/icra-2026-real-i.md)
- **交叉：** 乐聚 / OpenLET / LeRobot / unitree_lerobot / LingBot-VLA / GR00T / Cyclo / BT×VLA / IL / VLA / Manipulation / Teleoperation
- **开源结论：** 两训练/技能仓可跑；数据非商业或许可不明；Chat 与整机不开源


## [2026-08-17] ingest | sources/blogs/wechat_robotshub_ppo_locomotion_fundamentals.md — RobotsHub 万字 PPO：加深既有 PPO/GAE，不新建方法页；rsl_rl 已开源

- **触发：** 用户要求用 agent-reach ingest <https://mp.weixin.qq.com/s/MJQYYyOBSLirVr0vH1-AZg>
- **工具：** Agent Reach v1.5.0 + wechat-article-for-ai（Camoufox）；短链直连成功
- **步骤 2.5：** 教学长文，无独立项目页；文末 [rsl_rl](https://github.com/leggedrobotics/rsl_rl) **已开源**；PPO/GAE/TRPO/Rudin 论文与 Isaac Lab 文档均为公开资料
- **来源：** [`sources/blogs/wechat_robotshub_ppo_locomotion_fundamentals.md`](sources/blogs/wechat_robotshub_ppo_locomotion_fundamentals.md)；raw [`sources/raw/wechat_robotshub_ppo_locomotion_2026-07-16.md`](sources/raw/wechat_robotshub_ppo_locomotion_2026-07-16.md)
- **升格：** 不新建方法页。主沉淀 [`wiki/methods/ppo.md`](wiki/methods/ppo.md)（clip 误区、`old_log_prob`、有效视野、高斯 `action_scale`）；配套 [`wiki/methods/gae.md`](wiki/methods/gae.md)（$\gamma$ vs $\lambda$、value loss 陷阱）
- **交叉：** MDP / 五模块训练栈 / RL Runner / 特权训练 / 奖励设计 / RL 超参指南 / Cookbook / PPO vs SAC
- **开源结论：** 本文无代码仓；对照实现走已开源的 rsl_rl

## [2026-08-17] structural | wiki/entities/xpolicylab.md — 合并重复详情节点：删除 paper-xpolicylab，论文机制/时序图/结论并入工具实体

- **触发：** 用户反馈 XPolicyLab 存在两个重复详情节点，要求合并至一个
- **canonical：** [`wiki/entities/xpolicylab.md`](wiki/entities/xpolicylab.md)（`entity-xpolicylab`）
- **删除：** `wiki/entities/paper-xpolicylab.md`；旧 URL `entity-paper-xpolicylab` 经 [`schema/page-aliases.json`](schema/page-aliases.json) 重定向
- **并入：** arXiv:2608.09892 核心信息、O(N+M) 契约、源码运行时序图、实验与结论、对比表
- **交叉：** 入链改指向 canonical（RoboDojo / 仿真评测基础设施 / 具身评测选型闭环 / sources）

## [2026-08-17] ingest | sources/blogs/wechat_embodied_station_9_papers_2026-08-17.md — 具身智能小站 9 篇盘点：全部新建独立论文节点；SG-WAM 与 Self-Guided 同缩写消歧

- **触发：** 用户要求确保 agent-reach 已装并 ingest <https://mp.weixin.qq.com/s/UsgswMgDw4Kdpt5qI9fxnA>；随后要求自动合并
- **工具：** Agent Reach v1.5.0 + wechat-article-for-ai（Camoufox）；短链直连成功
- **步骤 2.5：**
  - SpeedTuning / SHRIMP / PGIF-MPPI / 4D-WAM / V-Simba：**已开源、可运行**
  - PEEL：双盲 **anonymous.4open.science** 三仓 → **部分开源**
  - SG-WAM（语义引导）：项目页 404；**实现未开源**（勿与 arXiv:2608.01397 Self-Guided SG-WAM 合并）
  - LAMDA：论文 GitHub **404**
  - SurgLAT：项目页已发，独立仓未找到
- **来源：** [`sources/blogs/wechat_embodied_station_9_papers_2026-08-17.md`](sources/blogs/wechat_embodied_station_9_papers_2026-08-17.md)；raw [`sources/raw/wechat_embodied_station_9_papers_2026-08-17.md`](sources/raw/wechat_embodied_station_9_papers_2026-08-17.md)
- **升格：** 9 个 `paper-*` 实体（0 复用）；交叉 WAM / MECo-WAM / MPPI / SAC / 模仿学习 / 4D 几何分类
- **开源结论：** 5 篇可跑；PEEL 匿名仓；3 篇未开源或仓未上线

## [2026-08-17] ingest | sources/sites/sonic-transfer-github-io.md — 接入冻结 GEAR-SONIC → AgiBot X2 Ultra 的闭式 codec + LoRA 跨具身迁移


- **触发：** 用户指定项目页 <https://sonic-agibot-x2.github.io/sonic-transfer/>
- **步骤 2.5：** 项目页 Code 指向 [`meetsitaram/sonic-x2`](https://github.com/meetsitaram/sonic-x2)（`./play_v2.sh` + transfer ONNX + codec sidecar）；HF [`tinkerbuggy/sonic-x2`](https://huggingface.co/tinkerbuggy/sonic-x2)。**无 SPDX LICENSE**；LoRA **训练脚本不在 play 仓**；截至入库日 **无 arXiv**。完整部署栈 sibling `GR00T-WholeBodyControl-X2-review` 未深挖。结论：**部分开源、推理可运行**
- **来源：** [`sources/sites/sonic-transfer-github-io.md`](sources/sites/sonic-transfer-github-io.md)、[`sources/papers/sonic_transfer_frozen_wbc_codec_lora.md`](sources/papers/sonic_transfer_frozen_wbc_codec_lora.md)、[`sources/repos/sonic-x2.md`](sources/repos/sonic-x2.md)
- **升格：** [`wiki/entities/paper-sonic-transfer.md`](wiki/entities/paper-sonic-transfer.md) — PHUMA OOD 69.0 vs 原生 incumbent 59.0；含 mermaid 流程与 `play_v2.sh` 时序图
- **开源结论：** MuJoCo 回放可跑；训练不可从 play 仓复现；真机验证文中 ongoing
- **交叉：** Any2Any / SONIC / 跨具身迁移选型 / WBT pipeline / LoRA / PHUMA
- **读数：** 近亲骨架上的更严冻结变体，不替代 Any2Any 的跨形态可学习对齐；companion 冻结规划器页未升格

## [2026-08-17] ingest | sources/papers/smpc2rl_arxiv_2608_12063.md — 加深 SMPC-to-RL：附录超参/采数细节，再核项目页仍未开源

- **触发：** 用户指定 ingest *Learning Loco-Manipulation From SMPC Demonstrations With Sparse Offline-to-Online RL*（RAI / TUM / ETH；Schuck et al.）。2026-08-14 已随三篇批量入库，本次按「一次一条」补附录与开源再核。
- **步骤 2.5：** 项目页 <https://pages.rai-inst.com/smpc2rl/> 截至 **2026-08-17** 仍无 Code/GitHub；judo（<https://github.com/rai-opensource/judo>，MIT）是通用采样 MPC 工具箱，**不是**本文 tiled 采数 + FastTD3 + ReLIC 仓 → **确认未开源**
- **来源：** [`sources/papers/smpc2rl_arxiv_2608_12063.md`](sources/papers/smpc2rl_arxiv_2608_12063.md)、[`sources/sites/rai-inst-smpc2rl.md`](sources/sites/rai-inst-smpc2rl.md)、[`sources/repos/judo.md`](sources/repos/judo.md)
- **升格：** [`wiki/entities/paper-smpc2rl-loco-manipulation.md`](wiki/entities/paper-smpc2rl-loco-manipulation.md) — 样条 SMPC、有界 critic、专家比例/撤出消融、真机规格；时序图仍不适用
- **交叉：** Loco-Manipulation / Sumo / MPC vs RL / Online vs Offline RL / mjlab / MPC / Unitree G1
- **开源结论：** 训练与部署代码未发布；judo 仅作对照入口

## [2026-08-17] ingest | sources/papers/prm_as_a_judge_arxiv_2608_14284.md + reflexvla_arxiv_2608_14379.md + advdex_arxiv_2608_14028.md — 过程评测套件 / 延迟感知动态 VLA / 人手–灵巧手统一动作空间

- **触发：** 用户指定三篇资料入库（PRM-as-a-Judge 机器人执行过程评测；ReflexVLA 动态任务低延迟 VLA；AdvDex 人手与灵巧手统一动作空间）
- **步骤 2.5：**
  - PRM-as-a-Judge：项目页 + GitHub 已上线，Apache-2.0 评测套件 **已开源**；RoboPulse 已上 HF；**RoboPulse++ 数据仍 Coming Soon**
  - ReflexVLA：项目页按钮 **Code After acceptance**，截至 2026-08-17 **无仓**
  - AdvDex：无独立项目页，论文未给代码/权重/OmniShare 链接，**确认未开源**
- **来源：** [`sources/papers/prm_as_a_judge_arxiv_2608_14284.md`](sources/papers/prm_as_a_judge_arxiv_2608_14284.md)、[`sources/sites/prm-as-a-judge-github-io.md`](sources/sites/prm-as-a-judge-github-io.md)、[`sources/repos/prm-as-a-judge.md`](sources/repos/prm-as-a-judge.md)、[`sources/papers/reflexvla_arxiv_2608_14379.md`](sources/papers/reflexvla_arxiv_2608_14379.md)、[`sources/sites/reflexvla-github-io.md`](sources/sites/reflexvla-github-io.md)、[`sources/papers/advdex_arxiv_2608_14028.md`](sources/papers/advdex_arxiv_2608_14028.md)
- **升格：** [`wiki/entities/paper-prm-as-a-judge.md`](wiki/entities/paper-prm-as-a-judge.md)、[`wiki/entities/paper-reflexvla.md`](wiki/entities/paper-reflexvla.md)、[`wiki/entities/paper-advdex.md`](wiki/entities/paper-advdex.md)
- **开源结论：** PRM 工具仓可跑；ReflexVLA 待录用后开放；AdvDex 确认未开源
- **交叉：** 过程奖励 / TOPReward / RoboDojo / 评测选型闭环 / VLA / Action Chunking / UHAS / 跨具身迁移 / Manipulation
- **机构：** 注册 `paxini`（帕西尼）

## [2026-08-17] ingest | sources/repos/isaac_lab_environments.md — 一手核对 Isaac Lab v3.0.0 全部 197 个注册任务，新建 wiki/entities/isaac-lab-default-environments.md 详细节点

## [2026-08-16] structural | roadmap/depth-motion-retargeting.md + depth-bfm.md + depth-vla.md + depth-wam.md + depth-perceptive-locomotion.md + depth-motion-generation.md + depth-imitation-learning.md — 补齐 2026-08-09（roadmap 上次批量触达日）以来入库的 11 篇里程碑级知识节点到七条纵深路线「推荐读什么」区块，覆盖运动重定向、BFM、VLA、WAM、感知越障、动作生成、模仿学习

- **触发：** 计划任务巡检——审计 `roadmap/*.md` 自 2026-08-09 commit `8c699f17`（"补齐 2026-08-06 以来的 15 篇最新知识节点"）以来的全部新增知识节点；期间 SPLC/ACE-Data-0/Dyna-2/FTL/TEMPO/AutoIntervene/深蓝专辑/关节执行器参数辨识/KILVO/重力补偿/零空间控制/ParkourFormer/Cartpole 等已由后续 ingest 提交直接补入对应 roadmap 文件，跳过；一次性 Awesome 清单机械节点化的 838 篇 `paper-sa-*` 索引级页与 56 篇 Transformer-CV-Applications 通用视觉课程页整体排除（非人形/机器人专属里程碑）；另跳过 Rift-WAM/Flex-π（与 Stage 3 已收录的 Fast-WAM/异步部署/DreamWAM 增量重叠）、SMPC2RL/G1-Confined-Space-WBP/Temporal-GRPO/HIL-HARC（未开源且问题面窄或与既有条目主题重叠）、XPolicyLab/Ego-OSCAR/Hand-Visibility-Detector（工具/数据采集页非方法论文）、R2S-Ego/OccAnyScene/LUCID/ODEWorld/Arcadia/LEGO（与既有 Real2Sim/WAM/VLN 条目叙事重叠或非机器人专属控制里程碑）、All-Hands-Up/CanFestival/SOEM/ComfyUI/DeepSeek-Harness（厂商画廊或通用工具库非论文级贡献）、RL-Runner（跨范式通用分类笔记，与 motion-control.md L5.1/L5.3 既有内容功能重叠）等应用面较窄或主题重叠的单点研究；仅在既有 Stage 内追加 bullet，未新建 Stage、未改动其余既有内容
- **[`roadmap/depth-motion-retargeting.md`](roadmap/depth-motion-retargeting.md)：** Stage 3 方法谱系补 [CoRe](wiki/entities/paper-core.md) 与 [RMR](wiki/entities/paper-rmr.md)（几何映射→接触感知精炼→RL 跟踪三段分工，实现并入 [core-retarget v0.1.0](wiki/entities/core-retarget.md)，Apache-2.0 部分开源）
- **[`roadmap/depth-bfm.md`](roadmap/depth-bfm.md)：** Stage 2 跟踪主线补 [ZEST](wiki/entities/paper-zest.md)（Science Robotics 2026，ViCap 当天出参考跨 Atlas/G1/Spot 复用，全尺寸 Atlas 首次稳定多接触技能）与 [HumanTracker](wiki/entities/paper-humantracker.md)（四族光学跟踪基准 + HumanScore，统一评测 GMT/TWIST2/SONIC/Humanoid-GPT）
- **[`roadmap/depth-vla.md`](roadmap/depth-vla.md)：** Stage 2 VLA 主线补 [Galaxea G0.5](wiki/entities/paper-galaxea-g05.md)（VLM-as-Actor + 学出来的 ActionCodec，GalaxeaVLA 已开源）；Stage 4 部署与系统整合补 [Neural Introspection Gating](wiki/entities/paper-neural-introspection-gating.md)（训练无关可插拔 VLA 推理调度层，动作 token logit margin 做免费不确定性信号）
- **[`roadmap/depth-wam.md`](roadmap/depth-wam.md)：** Stage 2 Cascaded WAM 补 [SLIM-0.5B](wiki/entities/paper-slim-05b.md)（IDM+FDM 解耦预训练 + MoT flow 联合微调，已开源）；Stage 3 Joint WAM 补 [FACT](wiki/entities/paper-fact.md)（失败演示后果监督打破 success bias，RoboTwin 管线已开源）
- **[`roadmap/depth-perceptive-locomotion.md`](roadmap/depth-perceptive-locomotion.md)：** Stage 2 感知策略训练补 [P³](wiki/entities/paper-p3.md)（多样本边缘化修正 VAE-PPO 感知策略单样本似然误判，官方仓开源 Isaac Lab + rl_p3）
- **[`roadmap/depth-motion-generation.md`](roadmap/depth-motion-generation.md)：** Stage 3 物理化落地补 [GenTrack](wiki/entities/paper-gentrack.md)（AAAI 2027，生成器–跟踪器在线互训打破单向管线冻结耦合瓶颈）
- **[`roadmap/depth-imitation-learning.md`](roadmap/depth-imitation-learning.md)：** Stage 3 Diffusion Policy 与生成式动作补 [Seeker](wiki/entities/paper-seeker.md)（冻结 DINOv3 + 动作监督 ROI 逼近特权 Oracle，MIT 已开源）
- **口径：** 每条路线仅在既有 Stage 结构内追加 bullet，未新建 Stage、未改动其余既有内容；`python3 scripts/lint_wiki.py` 与 `python3 scripts/eval_search_quality.py` 均通过（0 阻塞项，另含 2 条既有信息型陈旧声明预警，与本次改动无关）

## [2026-08-16] ingest | sources/papers/barto_sutton_anderson_1983_cartpole.md — 接入 Cartpole 问题独立节点（Gym CartPole-v1 + Isaac-Cartpole-v0）

- **触发：** 用户要求收集 Cartpole 一手资料，结合 Isaac-Cartpole-v0，升格为独立详情节点并配足够 mermaid
- **步骤 2.5：** Gymnasium 环境页与源码 MIT 已开源；Isaac Lab 文档站 + [isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab) BSD-3-Clause，`Isaac-Cartpole-v0` 注册与 cfg 可运行。1983 IEEE 文无独立项目页，历史 C/`pole.c` 与 CODECHECK 仓可复现
- **来源：** [`sources/papers/barto_sutton_anderson_1983_cartpole.md`](sources/papers/barto_sutton_anderson_1983_cartpole.md)、[`sources/sites/gymnasium-cartpole.md`](sources/sites/gymnasium-cartpole.md)、[`sources/sites/isaac-lab-cartpole.md`](sources/sites/isaac-lab-cartpole.md)
- **升格：** [`wiki/concepts/cartpole.md`](wiki/concepts/cartpole.md) — 欠驱动平衡、观测顺序陷阱、Gym 12° 失败 vs Isaac shaping、manager/direct 时序；多 mermaid
- **开源结论：** Gymnasium CartPole 与 Isaac Lab Cartpole 任务 **已开源、可运行**；1983 学习器为历史实现
- **交叉：** Gymnasium / Isaac Lab / MDP / 最小闭环 / Reward Design / Sutton-Barto / dm_control

## [2026-08-16] ingest | sources/personal/rl_runner_types.md — 接入 RL Runner 十类训练循环编排（独立概念页）

- **触发：** 用户提供 Runner 分类图（On-policy / Off-policy / Offline / Distillation / Imitation / Multi-agent / Self-play / Distributed / Model-based / Evaluation）
- **步骤 2.5：** 无项目页；教学谱系整理，非论文仓库 → 不建 `sources/repos/`
- **来源：** [`sources/personal/rl_runner_types.md`](sources/personal/rl_runner_types.md)
- **升格：** [`wiki/concepts/rl-runner.md`](wiki/concepts/rl-runner.md) — Runner ≠ 算法 ≠ 环境闭环；十类循环 + 选型口诀 + mermaid
- **交叉：** RL / PPO / SAC / GAE / Online vs Offline / 模仿 / 蒸馏 / MARL / MBRL / 五模块训练栈 / hub-learning
- **读数：** Distributed / Self-play 是拓扑或数据源，常叠在 on/off-policy 上；Imitation ≠ Distillation

## [2026-08-16] query | wiki/entities/gymnasium.md — 深化 Gymnasium 实体页 mermaid 图

- **触发：** 用户指定深化 [entity-gymnasium](https://imchong.github.io/Robotics_Notebooks/detail.html?id=entity-gymnasium)，增加 mermaid 以促进理解
- **来源：** [`sources/repos/gymnasium.md`](sources/repos/gymnasium.md) 对照官方 Basic Usage / Custom Env / Migration Guide / Vectorize API
- **关键页：** [`wiki/entities/gymnasium.md`](wiki/entities/gymnasium.md) — 补选型树、MDP↔Env 映射、单回合时序、terminated/truncated bootstrap、Wrapper 洋葱栈、spaces、类型关系、Farama 生态、API 向量化 vs GPU 物理并行、自定义环境注册流程
- **开源结论：** 官方仓 [Farama-Foundation/Gymnasium](https://github.com/Farama-Foundation/Gymnasium) 已开源（MIT）


## [2026-08-16] ingest | sources/papers/lego_leveled_language_gs_arxiv_2608_10057.md — 接入 LEGO 层级语言高斯溅射（已开源、可运行）

- **触发：** 用户指定 [arXiv:2608.10057](https://arxiv.org/abs/2608.10057) 与项目页 <https://pz0826.github.io/LEGO-Webpage/>
- **步骤 2.5：** 项目页 Code 指向 [`WHU-USI3DV/LEGO`](https://github.com/WHU-USI3DV/LEGO)；`lego run` / `eval` / `viewer` 与完整 `src/lego` 管线可辨识；`checkpoints/` 仅占位 → **已开源、可运行训练/评测**。许可 CC BY-NC-SA 4.0
- **来源：** [`sources/papers/lego_leveled_language_gs_arxiv_2608_10057.md`](sources/papers/lego_leveled_language_gs_arxiv_2608_10057.md)、[`sources/sites/pz0826-lego-webpage.md`](sources/sites/pz0826-lego-webpage.md)、[`sources/repos/lego.md`](sources/repos/lego.md)
- **升格：** [`wiki/entities/paper-lego-leveled-language-gaussian-splatting.md`](wiki/entities/paper-lego-leveled-language-gaussian-splatting.md) — 结构层级 vs SAM 粒度/绝对尺度；NVOS 94.2 mIoU；CoR 51.6；含 mermaid 时序图
- **机构：** 复用 `whu` / `hkust`
- **交叉：** 2D→3D Gap / 感知栈选型 / SAM / OV-SAM3D
- **读数：** 勿与斯坦福 LEGS 或 LEGO-SLAM 混名；单词查询不走 LLM

## [2026-08-16] ingest | sources/papers/parkourformer_arxiv_2605_25782.md — 接入 ParkourFormer 未来监督人形跑酷（确认未开源）

- **触发：** 用户指定 [arXiv:2605.25782](https://arxiv.org/abs/2605.25782) 与项目页 <https://mronaldo-gif.github.io/parkourformer.github.io/>
- **步骤 2.5：** 项目页与作者 GitHub [`MRonaldo-gif/parkourformer.github.io`](https://github.com/MRonaldo-gif/parkourformer.github.io) 仅为站点仓；无 Code / 权重 / 训练入口；论文未承诺开源 → **确认未开源**。未建 `sources/repos/`
- **来源：** [`sources/papers/parkourformer_arxiv_2605_25782.md`](sources/papers/parkourformer_arxiv_2605_25782.md)、[`sources/sites/parkourformer-github-io.md`](sources/sites/parkourformer-github-io.md)
- **升格：** [`wiki/entities/paper-parkourformer.md`](wiki/entities/paper-parkourformer.md) — query 历史 + 未来两步 AMP 监督；G1 九类地形单策略 **93.85%**；去 MSE 下楼 **9.50%**；时序图不适用
- **机构：** 复用 `hkust-gz`；`schema/institutions.json` 新增 `scau` / `gdut` / `clai-lab`
- **交叉：** Hiking / PHP / LightLP / SSR / Next Token Prediction / Humanoid Parkour Learning / AMP / 人形与楼梯任务页 / 感知越障路线
- **读数：** 摘要「+47.12%」是相对 1-MLP，不是相对 vanilla Transformer（已 90.49%）

## [2026-08-16] ingest | sources/papers/odeworld_arxiv_2607_27924.md — 接入 ODEWorld 物理时间流连续预测（推理+权重已开源，无训练脚本/LICENSE）

- **触发：** 用户指定项目页 <https://dstate.github.io/odeworld_website/> 与论文 [arXiv:2607.27924](https://arxiv.org/abs/2607.27924)
- **步骤 2.5：** 项目页链到 [`Dstate/ODEWorld`](https://github.com/Dstate/ODEWorld) + HF [`ldxxx/odeworld`](https://huggingface.co/collections/ldxxx/odeworld)；`demo_infer.py` + 五套权重可跑 → **已开源、可运行推理**。GitHub **未挂 LICENSE**；README **无训练入口**；策略实验未随仓
- **来源：** [`sources/papers/odeworld_arxiv_2607_27924.md`](sources/papers/odeworld_arxiv_2607_27924.md)、[`sources/sites/odeworld-website.md`](sources/sites/odeworld-website.md)、[`sources/repos/odeworld.md`](sources/repos/odeworld.md)
- **升格：** [`wiki/entities/paper-odeworld.md`](wiki/entities/paper-odeworld.md) — PT-Flow + JVP 一阶监督；LIBERO-LONG 序列子目标 83.6%；AgileX+X-VLA 55%→80%；含 mermaid 时序图
- **机构：** 复用 `tsinghua` / `berkeley`
- **交叉：** Generative WM / Latent Imagination / V-JEPA 2 / 物理保真输出轴 / 级联路线 01 / Video-as-Simulation / LIBERO / Manipulation / Ctrl-World / PlaNet
- **读数：** 当前版本无动作条件；不要当像素沙盒评估器

## [2026-08-16] ingest | sources/papers/arcadia_arxiv_2512_00076.md — 接入 Arcadia 具身终身学习四段闭环（EmbodiedKit 部分开源）

- **触发：** 用户指定 <https://arxiv.org/abs/2512.00076>
- **步骤 2.5：** 论文挂 [Embodied-Arcadia/EmbodiedKit](https://github.com/Embodied-Arcadia/EmbodiedKit)；无独立项目页。根 README 为 TODO；子目录有 VLN/VLA 数据生成与 Qwen 训练入口；探索 / 3DGS / 反馈写回与权重未发布 → **部分开源**
- **来源：** [`sources/papers/arcadia_arxiv_2512_00076.md`](sources/papers/arcadia_arxiv_2512_00076.md)、[`sources/repos/embodiedkit.md`](sources/repos/embodiedkit.md)
- **升格：** [`wiki/entities/paper-arcadia.md`](wiki/entities/paper-arcadia.md) — 四段不可拆闭环；VLN-CE-Isaac SR 50.1%；G1+Dex-3 46/27；时序图对齐子目录入口
- **机构：** `schema/institutions.json` 新增 `manycore`；复用 `zju` / `unitree` / `pku` / `nju` / `bytedance` / `adelaide`
- **交叉：** VLN / VLA / Sim2Real / 数据飞轮 / NaVILA / OpenVLA / 四范式复现 / 操作 / Awesome-R2S2R / DA-Nav
- **读数：** w/o feedback 已 +2.7 pp SR；联合训练几乎不掉；真机组合指令约 17%；仓 ≠ 论文闭环

## [2026-08-16] ingest | sources/papers/fsd_vln_arxiv_2607_08359.md — 接入 FSD-VLN 空中长程 VLN 快慢双系统（确认未开源、无真机）

- **触发：** 用户指定 arXiv 论文 *FSD-VLN: Fast-Slow Dual-System Modeling for Aerial Long-Horizon Vision-Language Navigation*
- **步骤 2.5：** 无项目页；abs/HTML/PDF 无 GitHub/HF；论文未承诺开源 → **确认未开源**。未建 `sources/repos/` / `sources/sites/`
- **来源：** [`sources/papers/fsd_vln_arxiv_2607_08359.md`](sources/papers/fsd_vln_arxiv_2607_08359.md)
- **升格：** [`wiki/entities/paper-fsd-vln.md`](wiki/entities/paper-fsd-vln.md) — VLSF + GR00T N1 系 DiT；未见相对自复现 OpenFly SR 5.1%→13.6%；H=1 优于长 chunk；时序图不适用
- **机构：** `schema/institutions.json` 新增 `siat`；复用 `peng-cheng-lab` / `pku`
- **交叉：** VLN 任务页空中子域 / WorldVLN / Uni-LaViRA / DA-Nav / VLA / 实时性取舍 / GR00T N1 / 四范式复现 / 多旋翼栈
- **读数：** 摘要 2× 主要对 OpenFly，不是对 CityNavAgent；成功半径 20 m；无真机

## [2026-08-16] ingest | sources/repos/unitree_ros2.md — v0.3.0：G1 双臂 / Dex3 / 手部 msg 对齐 SDK2（已开源 BSD-3-Clause）

- **触发：** 用户指定 <https://github.com/unitreerobotics/unitree_ros2> 与 [v0.3.0 Release](https://github.com/unitreerobotics/unitree_ros2/releases/tag/v0.3.0)；G1 双臂、Dex3 与 SDK2 DDS 更新
- **步骤 2.5：** **已开源、可运行**（BSD-3-Clause；`cyclonedds_ws` + `example` 可 `colcon build`）。项目页即 GitHub Release，无独立站点
- **来源：** [`sources/repos/unitree_ros2.md`](sources/repos/unitree_ros2.md)（深度补全，不另建页）
- **升格：** 改写 [`wiki/entities/unitree-ros2.md`](wiki/entities/unitree-ros2.md) — v0.3.0 版本锚点、G1 示例表、手部 msg 破坏性变更
- **交叉：** SDK2 / G1 软件栈 / 灵巧手服务（Dex3 原生 DDS ≠ Serial 桥）/ G1 硬件页 / 组织枢纽
- **破坏性：** `HandCmd`/`HandState`/`PressSensorState` 字段重排以对齐 SDK2；上游无迁移脚本

## [2026-08-15] ingest | sources/blogs/wechat_embodied_station_ego2robot_mango_grasp_2026-08-11.md — 具身智能小站 9 篇盘点：复用 PFM-HR / Why-AC，新建 7 个论文节点；GSR 与 go2_rescue_eval 已开源，其余项目页或 Coming Soon

- **触发：** 用户指定 <https://mp.weixin.qq.com/s/nKF7rxH-OuJz68galP3Xpg>；要求每篇论文独立详情节点、不重复造页
- **抓取：** Agent Reach v1.5.0 + wechat-article-for-ai（Camoufox，`--no-images`）
- **步骤 2.5：** GSR-ParaVLA、go2_rescue_eval **已开源可运行**；PFM-HR / EmbodiedVAE Coming Soon；OnOff / Ego2Robot 仅项目页；DigitCode HandTok 待审稿后挂；MANGO 宣称出版后开源；Why-AC 代码 Coming soon（已有节点）
- **复用：** [`wiki/entities/paper-pfm-hr.md`](wiki/entities/paper-pfm-hr.md)、[`wiki/entities/paper-why-action-chunking-improves-bc.md`](wiki/entities/paper-why-action-chunking-improves-bc.md)
- **新建：** OnOff / DigitCode / EmbodiedVAE / Ego2Robot / SA-Frontier / GSR / MANGO-Grasp
- **机构：** 新增 `gist` / `kentech` / `utec`；复用 sjtu、buaa、zgca、ruc、alibaba、astar、ntu、nus 等
- **交叉：** VLA / LIBERO / RoboTwin / EgoScale / WiLoR / UHAS / 抓取 / WAM / Go2 自主栈

## [2026-08-15] ingest | sources/repos/core_retarget.md — 接入 CoRe v0.1.0 接触感知重定向（已开源精炼；论文 T2M/RL 未随仓）与 RMR/CoRe 项目页

- **触发：** 用户指定 CoRe 仓 / v0.1.0 / HF Space / architecture·robots·licenses、Humanoids 2025 项目页与 RMR 项目页
- **步骤 2.5：** 代码 **已开源、可运行**（Apache-2.0，`core-retarget` + HF Space）；论文 T2M 与 contact-aware RL **未随 v0.1.0 发布**。RMR **无独立仓**，算法并入 DMR
- **来源：** [`sources/repos/core_retarget.md`](sources/repos/core_retarget.md)、[`sources/sites/core-page.md`](sources/sites/core-page.md)、[`sources/sites/rmr-page.md`](sources/sites/rmr-page.md)、[`sources/sites/huggingface-robotaemoon-core.md`](sources/sites/huggingface-robotaemoon-core.md)、[`sources/papers/core_humanoids_2025.md`](sources/papers/core_humanoids_2025.md)、[`sources/papers/rmr_iros_2025.md`](sources/papers/rmr_iros_2025.md)
- **升格：** [`wiki/entities/core-retarget.md`](wiki/entities/core-retarget.md)、[`wiki/entities/paper-core.md`](wiki/entities/paper-core.md)、[`wiki/entities/paper-rmr.md`](wiki/entities/paper-rmr.md)
- **机构：** `schema/institutions.json` 新增 `kist` / `naver-labs` / `rainbow-robotics`；复用 `korea-university` / `uiuc`
- **交叉：** Motion Retargeting / Pipeline / GMR / SOMA Retargeter / robot_retargeter / Kimodo / SOMA-X / hub / PhysCoRe 同名消歧
- **勿混名：** 本 CoRe ≠ [PhysCoRe](wiki/entities/paper-physcore.md)

## [2026-08-15] ingest | sources/papers/gentrack_arxiv_2608_01410.md — 接入 GenTrack 生成器–跟踪器在线物理对齐（确认未开源）

- **触发：** 用户指定 *GenTrack: Physical Alignment for Robot-Native Motion Generation and Zero-Shot Humanoid Tracking*（AAAI 2027；浙大 / 北大 / 腾讯 / 之江实验室）
- **步骤 2.5：** 无项目页、无 GitHub/HF；论文未承诺开源 → **确认未开源**。未建 `sources/repos/` / `sources/sites/`。勿与视觉 MOT GenTrack（arXiv:2510.24399）混名
- **来源：** [`sources/papers/gentrack_arxiv_2608_01410.md`](sources/papers/gentrack_arxiv_2608_01410.md)
- **升格：** [`wiki/entities/paper-gentrack.md`](wiki/entities/paper-gentrack.md) — 滞后闭环执行 + FlowGRPO；SONIC 支 LAFAN1 SR 85→90；时序图不适用
- **机构：** `schema/institutions.json` 新增 `zhejiang-lab`；复用 `zju` / `pku` / `tencent`
- **交叉：** SONIC / ProtoMotions / HY-Motion / PhyGile / PARC / RLPF / Humanoid-GPT / Kimodo / Gen2Humanoid / 跟踪选型

## [2026-08-15] ingest | sources/sites/allhandsup-org.md — 接入 RLWRLD All Hands Up 灵巧手画廊（16 手 URDF + 仿真 Kapandji；无独立 GitHub，部分开源）

- **触发：** 用户指定 <https://allhandsup.org/zh/#gallery>
- **步骤 2.5：** 站点与 `hands_urdf/` 可 HTTP 下载；`RLWRLD/allhandsup` 404 → **部分开源**。未建 `sources/repos/`。DexBench 在独立站，本次不另建页
- **来源：** [`sources/sites/allhandsup-org.md`](sources/sites/allhandsup-org.md)
- **升格：** [`wiki/entities/all-hands-up.md`](wiki/entities/all-hands-up.md) — Type 1/Type 2 双硬件策略 + 画廊 16 手 Kapandji 表
- **机构：** 复用 `rlwrld`
- **交叉：** Allegro / Orca / Wuji / RLDX-1 / MIDAS / 灵巧手运动学 / 灵巧采数指南 / Manipulation / 商业平台纵览

## [2026-08-15] ingest | sources/papers/hand_visibility_detector_arxiv_2608_11574.md — 接入 Hand Visibility Detector 逐关节手部可见性（已开源、研究/非商用）

- **触发：** 用户指定 *Hand Visibility Detector: Per-Keypoint Visibility Estimation for Hands*（arXiv:2608.11574）与代码 <https://github.com/ryhara/hand_visibility_detector>
- **步骤 2.5：** 无独立项目页；GitHub + HF 模型/Space；`HandVisibilityPipeline` + `demo.py` + `training.train` + `best.pt` → **已开源、可运行**。许可为研究/非商用，叠加 WiLoR / HaMeR / MANO
- **来源：** [`sources/papers/hand_visibility_detector_arxiv_2608_11574.md`](sources/papers/hand_visibility_detector_arxiv_2608_11574.md)、[`sources/repos/hand_visibility_detector.md`](sources/repos/hand_visibility_detector.md)
- **升格：** [`wiki/entities/paper-hand-visibility-detector.md`](wiki/entities/paper-hand-visibility-detector.md) — HInt mAP 0.931；微调骨干掉到 0.622；HO3D 重投影最多 −10.1%；含 mermaid 时序图
- **机构：** `schema/institutions.json` 新增 `keio` / `aist` / `omron-sinic-x`；复用 `u-tokyo`
- **交叉：** WiLoR / 灵巧数据管线 / 灵巧采集指南 / MediaPipe / 自动标注 / ViDiHand / Macrodata Hand-Action / 感知栈选型

## [2026-08-15] ingest | sources/papers/zest.md — 接入 Science Robotics ZEST 具身技能迁移（确认未开源）

- **触发：** 用户指定 *Embodied skill transfer for locomotion control*（Sleiman / He Li / Adu-Bredu 等，RAI × 波士顿动力，*Science Robotics* 11(117)，DOI 10.1126/scirobotics.aec7695）
- **可读全文：** 期刊页超时/付费；直接可读 PDF 为 [arXiv:2602.00401](https://arxiv.org/pdf/2602.00401)（预印本标题 ZEST）。未使用微信备份
- **步骤 2.5：** 无项目页、无 GitHub/HF；期刊数据可用性只指向正文/附录 → **确认未开源**。未建 `sources/repos/` / `sources/sites/`
- **来源：** [`sources/papers/zest.md`](sources/papers/zest.md)（由 arXiv 预印本归档升为期刊版）
- **升格：** [`wiki/entities/paper-zest.md`](wiki/entities/paper-zest.md)；方法页 [`wiki/methods/zest.md`](wiki/methods/zest.md) 对齐卷期与评测
- **交叉：** MTRG / HIL / 跑酷对比 / Curriculum / Sim2Real / VideoMimic / Boston Dynamics / G1 / 运动跟踪选型
- **机构：** 复用 `rai-institute`、`boston-dynamics`

## [2026-08-15] ingest | sources/papers/humantracker_arxiv_2608_13555.md — 接入 HumanTracker 四族光学基准与 HumanScore（评测代码已开，153h 数据待发布）

- **触发：** 用户指定论文 *HumanTracker: Towards Comprehensive and Human-Aligned Motion Tracking Benchmark* 与项目页 <https://dairuliu.github.io/humantracker/>
- **步骤 2.5：** 项目页 Paper/Code/Dataset 仍写 Coming Soon；arXiv:2608.13555 已上线；GitHub [GalaxyGeneralRobotics/HumanTracker](https://github.com/GalaxyGeneralRobotics/HumanTracker) 含评测 harness、HumanScore 训练与 `best.pt` → **部分开源**。153 h / 25K 数据集未发布
- **来源：** [`sources/papers/humantracker_arxiv_2608_13555.md`](sources/papers/humantracker_arxiv_2608_13555.md)、[`sources/sites/humantracker-dairuliu-github-io.md`](sources/sites/humantracker-dairuliu-github-io.md)、[`sources/repos/humantracker.md`](sources/repos/humantracker.md)
- **升格：** [`wiki/entities/paper-humantracker.md`](wiki/entities/paper-humantracker.md) — HumanScore 90.83%；零样本表 Humanoid-GPT 整体领先、SONIC Ground HS 更高；含 mermaid 时序图
- **机构：** `schema/institutions.json` 新增 `nankai`（南开大学）；复用 `tsinghua` / `galbot` / `sjtu` / `pku` / `shanghai-pil`
- **交叉：** Humanoid-GPT / SONIC / GMT / TWIST2 / 跟踪选型 / 参考运动数据集 / AMASS / PHUMA / GMR

## [2026-08-15] ingest | sources/papers/temporal_grpo_arxiv_2608_13026.md — 接入 Temporal GRPO 阶段条件 VLA-RL 信用（确认未开源）

- **触发：** 用户指定 arXiv 论文 *Temporal GRPO: Beyond Trajectory-Level Credit in Vision-Language-Action Reinforcement Learning*
- **步骤 2.5：** 无项目页、无 GitHub/HF；论文未承诺开源 → **确认未开源**。未建 `sources/repos/`。勿与 TGRPO（arXiv:2506.08440）混名
- **来源：** [`sources/papers/temporal_grpo_arxiv_2608_13026.md`](sources/papers/temporal_grpo_arxiv_2608_13026.md)
- **升格：** [`wiki/entities/paper-temporal-grpo.md`](wiki/entities/paper-temporal-grpo.md) — 入场门控 + 同阶段编组 + 区间写回；RoboTwin 宏平均 75.8%；LIBERO-Long 99.1%；时序图不适用
- **机构：** `schema/institutions.json` 新增 `iscas`（中国科学院软件研究所）
- **交叉：** VLA 后训练 / TEMPO / Green-VLA / RoboTwin / LIBERO / OpenVLA / WCM / RynnBrain / 四范式复现（不入可跑通清单）

## [2026-08-15] ingest | sources/papers/seeker_arxiv_2608_13422.md — 接入 Seeker 动作监督视觉注意力（MIT 已开源）

- **触发：** 用户指定 GitHub <https://github.com/zheyu-zhuang/seeker> 与论文 <https://arxiv.org/abs/2608.13422>
- **步骤 2.5：** 无独立项目页；仓默认分支 `open_source`（`main` 无 README）；`seeker` CLI + `seeker.mimicgen.pth` → **已开源、可运行**（MIT）
- **来源：** [`sources/papers/seeker_arxiv_2608_13422.md`](sources/papers/seeker_arxiv_2608_13422.md)、[`sources/repos/seeker.md`](sources/repos/seeker.md)
- **升格：** [`wiki/entities/paper-seeker.md`](wiki/entities/paper-seeker.md) — 冻结 DINOv3 + 动作监督 ROI；MimicGen 62.6%；xArm 域内 76.7% / OOD 60.0%；含 mermaid 时序图
- **机构：** `schema/institutions.json` 新增 `uni-freiburg`（弗莱堡大学）；复用 `kth` / `uni-hamburg`
- **交叉：** Diffusion Policy / IL / Manipulation / MimicGen / ActFovea / VLA

## [2026-08-15] ingest | HumanoidVLN 复核 — 已于 2026-08-14 入库，项目页仍无代码

- **触发：** 用户再次指定 <https://arxiv.org/abs/2608.12860> 与项目页 <https://humanoid-vln.github.io/>
- **步骤 2.5 复核：** 项目页仍无 GitHub/HF；论文仍写 *will be released upon acceptance* → **宣称将开源 / 待发布**（与 08-14 结论相同）
- **处理：** 不新建页；沿用 [`wiki/entities/paper-humanoidvln.md`](wiki/entities/paper-humanoidvln.md)

## [2026-08-14] ingest | sources/papers/roboreact_arxiv_2608_03387.md — 接入 RoboReact 生成 egocentric 视频蒸馏全身操作（确认未开源）

- **触发：** 用户指定 arXiv 预印本 *RoboReact: Agentic Skill Distillation from Generated Egocentric Videos for Generalizable Whole-Body Manipulation*（港中深 / 京东科技 / 清华）
- **步骤 2.5：** 项目页 <https://roboreact.github.io/> 无 Code/HF；GitHub 用户 `RoboReact` 仅落地页仓 `RoboReact.github.io`；论文未承诺放代码 → **确认未开源**。未建 `sources/repos/`
- **来源：** [`sources/papers/roboreact_arxiv_2608_03387.md`](sources/papers/roboreact_arxiv_2608_03387.md)、[`sources/sites/roboreact-github-io.md`](sources/sites/roboreact-github-io.md)
- **升格：** [`wiki/entities/paper-roboreact.md`](wiki/entities/paper-roboreact.md) — 单帧 RGB-D 生成视频 → 物体中心关键帧 → 冻结 VLM 标定编辑 → 测试时再接地 + HOMIE；G1 四任务均值 SR 81.3%；时序图不适用
- **机构：** `schema/institutions.json` 新增 `cuhk-sz`（香港中文大学深圳）
- **交叉：** Loco-Manipulation 视频生成路线 / ExoActor / Video-as-Simulation / G1 / HOMIE / 合成视频人形任务 / OKAMI

## [2026-08-14] ingest | sources/papers/humanoidvln_arxiv_2608_12860.md — 接入 HumanoidVLN 人形物理 VLN 仿真与基准（宣称录用后开源）

- **触发：** 用户指定 arXiv 论文 *HumanoidVLN: A Physics-Grounded Simulator and Benchmark for Vision-Language Navigation Across Diverse Humanoid Embodiments*
- **步骤 2.5：** 项目页 <https://humanoid-vln.github.io/> 无 GitHub/HF；论文写 *will be released upon acceptance* → **宣称将开源 / 待发布**。未建 `sources/repos/`
- **来源：** [`sources/papers/humanoidvln_arxiv_2608_12860.md`](sources/papers/humanoidvln_arxiv_2608_12860.md)、[`sources/sites/humanoid-vln-github-io.md`](sources/sites/humanoid-vln-github-io.md)
- **升格：** [`wiki/entities/paper-humanoidvln.md`](wiki/entities/paper-humanoidvln.md) — 四本体 RL+PD/MPC、87 场景、933 episode MAA；JanusVLN 平均 SR 43.55%；G1 DualVLN 20 条 r=0.935；时序图不适用
- **机构：** `schema/institutions.json` 新增 `vinmotion`（越南人形机器人）
- **交叉：** VLN 任务页 / 分类 01 / 四范式（不入可跑通清单）/ NaVILA / VLN-CE / Isaac Sim / G1 / Sim2Real / Locomotion

## [2026-08-14] ingest | sources/papers/green_for_go_vla_nav_grounding_arxiv_2607_05122.md — 接入 UCL Green for Go 导航 VLA 绿/红视觉接地（确认未开源）

- **触发：** 用户指定 arXiv 论文 *Green for Go, Red for No: Visual Grounding via Semantic Segmentation for VLA Navigation Policies*
- **步骤 2.5：** 无项目页、无 GitHub/HF；论文未承诺开源 → **确认未开源**。评测依赖 OmniVLA 与 Grand Tour，SegFormer 不是本文管线仓
- **来源：** [`sources/papers/green_for_go_vla_nav_grounding_arxiv_2607_05122.md`](sources/papers/green_for_go_vla_nav_grounding_arxiv_2607_05122.md)
- **升格：** [`wiki/entities/paper-green-for-go-vla-nav-grounding.md`](wiki/entities/paper-green-for-go-vla-nav-grounding.md) — 冻结 OmniVLA + 绿/红 overlay；WP7 −27–44% 但是长度正则；stop 失败；时序图不适用
- **机构：** `schema/institutions.json` 新增 `ucl`（伦敦大学学院）
- **交叉：** VLN / VLA / NaVILA / NavWAM / DA-Nav / REALM / ActFovea / Green-VLA 消歧 / 四范式复现（不入可跑通清单）

## [2026-08-14] ingest | sources/blogs/wechat_freedof_sim2real_dynamics_identification.md — 接入 Sim2Real 关节动力学辨识实验设计（无项目页/代码仓）

- **触发：** 用户要求确保 agent-reach 已装并 ingest <https://mp.weixin.qq.com/s/B_sH9VNRxB6GCTJwnx6esQ>；随后要求自动合并
- **工具：** Agent Reach v1.5.0（`~/.local/bin/agent-reach`）+ wechat-article-for-ai（Camoufox；`playwright==1.49.1`；`--no-images`）。直连一次成功
- **开源结论：** 工程方法文，无项目页/代码仓；步骤 2.5 不适用
- **来源：** [`sources/blogs/wechat_freedof_sim2real_dynamics_identification.md`](sources/blogs/wechat_freedof_sim2real_dynamics_identification.md)；原文 [`sources/raw/wechat_freedof_sim2real_dynamics_identification_2026-08-12.md`](sources/raw/wechat_freedof_sim2real_dynamics_identification_2026-08-12.md)
- **升格：** [`wiki/methods/sim2real-joint-sysid-experiment-design.md`](wiki/methods/sim2real-joint-sysid-experiment-design.md) — 可辨识性三原则、延迟→摩擦→惯量→柔性分级实验；与算法页 [关节执行器参数辨识](wiki/methods/joint-actuator-parameter-identification.md) 分工
- **交叉：** SysID / armature / 关节摩擦 / 连杆-转子惯量 / 并联关节 / Sim2Real 闭环 / 驱动链枢纽
## [2026-08-14] ingest | sources/papers/smpc2rl_arxiv_2608_12063.md — 接入 SMPC-to-RL 稀疏奖励全身 loco-manipulation（项目页确认未开源）

- **触发：** 用户指定论文 <https://arxiv.org/abs/2608.12063>、项目页 <https://pages.rai-inst.com/smpc2rl/>；G1 稀疏奖励全身操作
- **步骤 2.5：** 项目页无 Code/GitHub；文中 judo 是通用采样 MPC 工具箱，不是本文管线 → **确认未开源**
- **来源：** [`sources/papers/smpc2rl_arxiv_2608_12063.md`](sources/papers/smpc2rl_arxiv_2608_12063.md)、[`sources/sites/rai-inst-smpc2rl.md`](sources/sites/rai-inst-smpc2rl.md)
- **升格：** [`wiki/entities/paper-smpc2rl-loco-manipulation.md`](wiki/entities/paper-smpc2rl-loco-manipulation.md) — SMPC 离线专家 + 稀疏 FastTD3 + 冻结 ReLIC；Spot/G1；时序图不适用
- **交叉：** Loco-Manipulation / Sumo / MPC vs RL / Unitree G1 / 15 分钟行走 FastTD3 / MPC-RL

## [2026-08-14] ingest | sources/papers/galaxea_g05_arxiv_2608_11739.md — 接入 G0.5 统一推理与动作 VLA（GalaxeaVLA + HF 已开源，Community License）

- **触发：** 用户指定 GitHub <https://github.com/OpenGalaxea/GalaxeaVLA>、论文 <https://arxiv.org/abs/2608.11739>、项目页 <https://opengalaxea.github.io/G05/>、权重 <https://huggingface.co/OpenGalaxea/G05>
- **步骤 2.5：** `OpenGalaxea/GalaxeaVLA` 含 `finetune.sh` / `serve_policy.py` / `experiments/`；HF `OpenGalaxea/G05` 权重约 55 GB → **已开源、可运行**。许可证 **G0.5 Community License**（学术/评估，非 Apache/MIT）
- **来源：** 论文摘录 + [`sources/sites/opengalaxea-g05.md`](sources/sites/opengalaxea-g05.md) + [`sources/repos/galaxea-vla.md`](sources/repos/galaxea-vla.md)
- **升格：** [`wiki/entities/paper-galaxea-g05.md`](wiki/entities/paper-galaxea-g05.md) — VLM-as-Actor、ActionCodec 27 维、原生 CoT；含 mermaid 时序图
- **机构：** `schema/institutions.json` 新增 `galaxea`（星海图 / OpenGalaxea）
- **交叉：** VLA / 开源复现景观 / π0.5 / InternVLA-A1.5 / 操作选型 / LIBERO / RoboTwin

## [2026-08-14] ingest | sources/papers/rift_wam_arxiv_2608_11521.md — 接入 Rift 免视频 rollout 低延迟 WAM（确认未开源）

- **触发：** 用户指定论文 <https://arxiv.org/abs/2608.11521>；免视频 Rollout 的低延迟 WAM
- **步骤 2.5：** 无项目页、无官方仓；FastWAM 仓是骨干对照，不能当成 Rift 实现 → **确认未开源**
- **来源：** [`sources/papers/rift_wam_arxiv_2608_11521.md`](sources/papers/rift_wam_arxiv_2608_11521.md)
- **升格：** [`wiki/entities/paper-rift-wam.md`](wiki/entities/paper-rift-wam.md) — 干预 Finding + anticipation prefill；LIBERO 98.8%/247.9 ms；时序图不适用
- **交叉：** World Action Models / 动作后果 01 / DreamWAM / Flex-π / DSWAM / WAM 异步部署 / VLA 部署指南

## [2026-08-13] ingest | sources/papers/gravity_compensation.md — 接入重力补偿算法簇（模型基 RNEA / PD+$g$ / De Luca 迭代学习 / Tomei 自适应）；PAL 教程部分开源、生产控制器未开源

- **触发：** 用户要求找到并 ingest 重力补偿相关算法
- **步骤 2.5：** Takegaki 1981 / De Luca 1993 / Tomei 1991 **确认未开源**。计算核 [Pinocchio](wiki/entities/pinocchio.md) `computeGeneralizedGravity` / [Dynibo](wiki/entities/dynibo.md) `gravity()` → **已开源**。[PAL 教程仓](https://github.com/pal-robotics/gravity_compensation_controller_tutorial) 可运行但许可未声明；PAL OS `pal_controllers/GravityCompensationController` **未开源** → **部分开源**
- **来源：** 论文簇 + De Luca 1993；仓库 gravity-compensation-controller-tutorial；文档站 pal-robotics-gravity-compensation
- **升格：** [`wiki/concepts/gravity-compensation.md`](wiki/concepts/gravity-compensation.md)、[`wiki/entities/paper-learning-gravity-compensation.md`](wiki/entities/paper-learning-gravity-compensation.md)
- **机构：** `schema/institutions.json` 新增 `pal-robotics`、`osaka`
- **交叉：** 摩擦补偿 / ABA·RNEA / CTC / IDC / PID / SysID / 连杆惯量 / 阻抗 / Pinocchio / Dynibo / WBC 实现指南 / 运动控制 L2.1 / 传统控制 Stage 4 / 力矩电机 Stage 5

## [2026-08-13] ingest | sources/papers/null_space_control.md — 接入零空间控制一手论文簇与 7 轴开源实现（Dietrich 2015 综述未开源；Mayr JOSS / libfranka / TSID 已开源）

- **触发：** 用户要求找到并 ingest 机器人零空间控制一手资料与源码（含 7 轴）
- **步骤 2.5：** Dietrich/Ott/Albu-Schäffer IJRR 2015 无官方仓（**确认未开源**，真机 DLR LWR-III 7 轴）。Mayr JOSS 项目页指向 [Cartesian-Impedance-Controller](https://github.com/matthias-mayr/Cartesian-Impedance-Controller)（BSD-3-Clause，FR3/Panda/iiwa7）→ **已开源**。[libfranka](https://github.com/frankarobotics/libfranka) `cartesian_impedance_control` + elbow 运动 → **已开源**。[stack-of-tasks/tsid](https://github.com/stack-of-tasks/tsid) → **已开源**（HQP 替代显式 $N$）
- **来源：** 论文簇 + Dietrich 综述 + Mayr JOSS；仓库 cartesian-impedance-controller / libfranka / tsid；项目页 cartesian-impedance-controller-github-io
- **升格：** [`wiki/concepts/null-space-control.md`](wiki/concepts/null-space-control.md)、[`wiki/entities/paper-null-space-projections-survey.md`](wiki/entities/paper-null-space-projections-survey.md)、[`wiki/entities/paper-cartesian-impedance-controller.md`](wiki/entities/paper-cartesian-impedance-controller.md)
- **机构：** `schema/institutions.json` 新增 `dlr`、`lund`、`kuka`
- **交叉：** IK / Jacobian / HQP / TSID / WBC / 阻抗 / 控制分配 / Franka FR3 / Pink / Pinocchio / 传统控制纵深 Stage 4 / 运动控制 L3

## [2026-08-13] ingest | sources/papers/joint_actuator_parameter_identification.md — 接入关节转子惯量与摩擦参数辨识算法（Swevers/Ayusawa；FloBaRoID 已开源；BAM/PACE 既有实体）

- **触发：** 用户要求找到并 ingest 机器人关节转子惯量与摩擦力等参数辨识算法的一手资料与源码
- **步骤 2.5：** Swevers 1997 / Ayusawa 2014 **确认未开源**；[kjyv/FloBaRoID](https://github.com/kjyv/FloBaRoID) **已开源**（LGPL-3.0，iDynTree+IPOPT）；BAM / PACE **既有开源实体**，补文档站 [bam.readthedocs.io](https://bam.readthedocs.io/)
- **来源：** [`sources/papers/joint_actuator_parameter_identification.md`](sources/papers/joint_actuator_parameter_identification.md)、[`sources/repos/flobaroid.md`](sources/repos/flobaroid.md)、[`sources/sites/bam-readthedocs.md`](sources/sites/bam-readthedocs.md)
- **升格：** [`wiki/methods/joint-actuator-parameter-identification.md`](wiki/methods/joint-actuator-parameter-identification.md)（Fourier+OLS vs CMA-ES 选型）；[`wiki/entities/flobaroid.md`](wiki/entities/flobaroid.md)（两步摩擦流水线）
- **交叉：** SysID / 关节摩擦 / armature / Pinocchio `computeJointTorqueRegressor` / CMA-ES / BAM / PACE / SPI-Active / 执行器驱动链 ③ 层

## [2026-08-13] ingest | sources/blogs/wechat_shenlan_{rl_motion_control_pipeline,rl_inverse_kinematics,forward_kinematics,inverse_kinematics,robot_jacobian}.md — 更新深蓝《具身智能基础》专辑 5→10 篇

- **触发：** 用户指定专辑 <https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzkwMDcyNDUzMQ==&action=getalbum&album_id=4525948187102363653>；要求确保 agent-reach 已装并更新 ingest；随后要求自动合并
- **工具：** Agent Reach v1.5.0（`~/.local/bin/agent-reach`）+ wechat-article-for-ai（Camoufox；`playwright==1.49.1`）。专辑 JSON API 直取 10 篇列表；文章直连 CAPTCHA，**专辑页同会话跳转**抓取第 6–10 篇正文
- **开源结论：** 科普专栏，无项目页/代码仓；步骤 2.5 不适用。工程入口交叉到已有 [Pinocchio](wiki/entities/pinocchio.md) / [Pink](wiki/entities/pink-ik.md) / [Mink](wiki/entities/mink-ik.md)
- **新增 blogs / raw：** 专栏 06–10；专辑清单 [`sources/raw/wechat_shenlan_embodied_ai_fundamentals_album_2026.json`](sources/raw/wechat_shenlan_embodied_ai_fundamentals_album_2026.json)
- **升格：** [`wiki/overview/robot-rl-motion-control-pipeline.md`](wiki/overview/robot-rl-motion-control-pipeline.md)、[`wiki/comparisons/rl-inverse-kinematics-five-approaches.md`](wiki/comparisons/rl-inverse-kinematics-five-approaches.md)、[`wiki/formalizations/forward-kinematics.md`](wiki/formalizations/forward-kinematics.md)、[`wiki/formalizations/inverse-kinematics.md`](wiki/formalizations/inverse-kinematics.md)、[`wiki/formalizations/robot-jacobian.md`](wiki/formalizations/robot-jacobian.md)
- **父节点：** [`wiki/overview/shenlan-embodied-ai-fundamentals-series.md`](wiki/overview/shenlan-embodied-ai-fundamentals-series.md) 5→10 篇
- **交叉：** 齐次坐标 / 最小闭环 / PPO / privileged-training / domain-randomization / WBC / MPC / TSID / Modern Robotics / Pinocchio / Mink / Pink / 运动控制路线 L1.2–L1.3

## [2026-08-13] ingest | sources/papers/wam_realtime_async_arxiv_2608_01880.md — Motubrain 异步部署实证 + WAM 模型入库；官方仓占位

- **触发：** 用户指定论文 <https://arxiv.org/pdf/2608.01880>、博客 <https://www.motubrain.com/zh/research>、GitHub <https://github.com/shengshu-ai/Motubrain>；要求自动合并
- **步骤 2.5：** 官网/博客齐全；[`shengshu-ai/Motubrain`](https://github.com/shengshu-ai/Motubrain) 仅 LICENSE + PDF + README + figures（Modified MIT）→ **部分开源（占位仓）**。2608.01880 **无单独实验脚本**
- **来源：** 异步论文/博客 + Motubrain 报告 + 官网 + 占位仓
- **升格：** [`wiki/entities/paper-wam-realtime-async.md`](wiki/entities/paper-wam-realtime-async.md)（六策略；train 综合最好）；[`wiki/entities/paper-motubrain.md`](wiki/entities/paper-motubrain.md)（RoboTwin 95.8/96.1）
- **机构：** [`schema/institutions.json`](schema/institutions.json) 新增 `shengshu`（生数科技）
- **交叉：** WAM 概念、Action Chunking、VLA 部署指南、动作后果 01、Motus 索引、RoboTwin、Manipulation

## [2026-08-13] ingest | sources/papers/occanyscene_arxiv_2608_08696.md — OccAnyScene 跨室内外 3D 语义占据入库；官方仓占位

- **触发：** 用户指定论文 *OccAnyScene: Towards Unified Indoor-Outdoor 3D Occupancy Prediction*（arXiv:2608.08696）与项目页 <https://roboperception.github.io/OccAnyScene/>；要求自动合并
- **步骤 2.5：** 项目页 Code 指向 [RoboPerception/OccAnyScene](https://github.com/RoboPerception/OccAnyScene)；`main` 仅 README + `assets/`，徽章 **release upon acceptance** → **部分开源（占位仓）/ 训练与推理待录用后发布**
- **来源：** [`sources/papers/occanyscene_arxiv_2608_08696.md`](sources/papers/occanyscene_arxiv_2608_08696.md)、[`sources/sites/roboperception-occanyscene-github-io.md`](sources/sites/roboperception-occanyscene-github-io.md)、[`sources/repos/occanyscene.md`](sources/repos/occanyscene.md)
- **升格：** [`wiki/entities/paper-occanyscene.md`](wiki/entities/paper-occanyscene.md) — Cross-Scene 任务、PFFA+FPGC、DAv3 联合训练 -0.41/-0.19 mIoU；时序图不适用
- **机构：** [`schema/institutions.json`](schema/institutions.json) 新增 `hitsz`（哈尔滨工业大学深圳）
- **交叉：** 六种空间表征、2D→3D 提升 Gap、导航·SLAM 栈、感知选型闭环、Humanoid Occupancy、Nvblox、GaussianWorld

## [2026-08-13] structural | wiki/entities/deepseek-harness.md — 修复源码运行时序图 Mermaid：参与者名 Loop 被解析为保留字

- **触发：** 用户反馈 DeepSeek Harness 详情页 Mermaid 时序图格式问题；要求自动合并
- **根因：** `participant Loop` / `CLI->>Loop` 中的 `Loop` 被 Mermaid sequenceDiagram 当成保留字 `loop`，报 `Expecting ACTOR, got 'loop'`
- **修复：** 参与者改为 `Driver`（对齐官方 agent-lifecycle 的 driver）；去掉消息里的 `/`、`*`、`→`；流程图 `subgraph loop` 改名为 `agentRing`，链式边拆开
- **页面：** [`wiki/entities/deepseek-harness.md`](wiki/entities/deepseek-harness.md)

## [2026-08-13] ingest | sources/repos/deepseek-harness.md — 接入 DeepSeek Harness（dsh）官方插件化 agent 运行时

- **触发：** 用户指定 <https://github.com/deepseek-ai/deepseek-harness>；要求自动合并
- **步骤 2.5：** 无独立项目页（`homepage` 空）；GitHub API **MIT**、TypeScript monorepo + Python SDK + 可运行 `npx @deepseek-ai/dsh web` / headless → **已开源**。Issues/PRs 关闭，反馈走 Discussions；根包 **0.1.0-rc.5** 开发者预览
- **来源：** [`sources/repos/deepseek-harness.md`](sources/repos/deepseek-harness.md)
- **升格：** [`wiki/entities/deepseek-harness.md`](wiki/entities/deepseek-harness.md) — Cordis 一切皆插件、turn/step 日志不变量、Web/headless/Python SDK；与 Harness VLA / RoboHarness **同名不同物**
- **机构：** [`schema/institutions.json`](schema/institutions.json) 新增 `deepseek`（`deepseek` / `deepseek-ai`）
- **交叉：** [`wiki/entities/hermes-agent.md`](wiki/entities/hermes-agent.md)、[`wiki/entities/openclaw.md`](wiki/entities/openclaw.md)、[`wiki/entities/kimi-k3.md`](wiki/entities/kimi-k3.md)、[`wiki/queries/real-robot-policy-autoresearch-harness.md`](wiki/queries/real-robot-policy-autoresearch-harness.md)、[`wiki/entities/paper-harness-vla.md`](wiki/entities/paper-harness-vla.md)

## [2026-08-13] ingest | sources/papers/p3_arxiv_2607_25541.md — P³（arXiv:2607.25541）VAE-PPO 边缘策略传播入库；已开源 ylyem9x/P3_Open

- **触发：** 用户指定论文 *P³: Probabilistic Policy Propagation for Stable VAE-Based Robot Learning*（Yan / Ma / Zhang / Fu / Cao / Zhu / Chen / Gao；SJTU / 同济 / ZJU / 上海创智学院；arXiv:2607.25541）；要求自动合并
- **步骤 2.5：** 无独立项目页；[ylyem9x/P3_Open](https://github.com/ylyem9x/P3_Open) 含 `run_train.sh` / `run_finetune.sh` / `run_play.sh` 与 `rl_p3` 概率 Actor → **已开源**。无 LICENSE 元数据；权重不随仓分发；真机 FAST-LIO 桥未在 README 单列
- **来源：** [`sources/papers/p3_arxiv_2607_25541.md`](sources/papers/p3_arxiv_2607_25541.md)、[`sources/repos/p3-open.md`](sources/repos/p3-open.md)
- **升格：** [`wiki/entities/paper-p3.md`](wiki/entities/paper-p3.md) — 单样本 $r_\theta$ 失配、$D_{\mathrm{eff}}$ 64.6%→100%、MM+LSFT、G1 真机 8/9/10
- **交叉：** [`wiki/methods/ppo.md`](wiki/methods/ppo.md)、[`wiki/methods/dreamwaq.md`](wiki/methods/dreamwaq.md)、[`wiki/methods/pie-perceptive-locomotion.md`](wiki/methods/pie-perceptive-locomotion.md)、[`wiki/tasks/stair-obstacle-perceptive-locomotion.md`](wiki/tasks/stair-obstacle-perceptive-locomotion.md)、[`wiki/entities/isaac-lab.md`](wiki/entities/isaac-lab.md)、[`wiki/entities/unitree-g1.md`](wiki/entities/unitree-g1.md)


## [2026-08-13] ingest | sources/repos/comfyui.md — 接入 ComfyUI（Comfy-Org/ComfyUI + comfy.org）节点式生成引擎

- **触发：** 用户指定 <https://github.com/Comfy-Org/ComfyUI> 与 <https://comfy.org/>；要求自动合并
- **步骤 2.5：** 官网 Desktop/Cloud/API/Enterprise 均指向本仓与 [docs.comfy.org](https://docs.comfy.org/)；GitHub API **GPL-3.0**、完整 `main.py` / `server.py` / `execution.py` / `nodes.py` → **已开源**。权重不随仓分发；Partner/API nodes 为可选付费闭源入口（`--disable-api-nodes` 可强制离线）
- **来源：** [`sources/repos/comfyui.md`](sources/repos/comfyui.md)、[`sources/sites/comfy-org.md`](sources/sites/comfy-org.md)
- **升格：** [`wiki/entities/comfyui.md`](wiki/entities/comfyui.md) — v0.32.0；节点图执行、Desktop/Cloud/API 分工、MCP beta、与 3D Gen Studio / Wan / 生成式数据增强的选型边界
- **机构：** [`schema/institutions.json`](schema/institutions.json) 新增 `comfy-org`（标签仅 `comfy-org` / `comfyanonymous`，避免误贴带 `comfyui` tag 的编排层页面）
- **交叉：** [`wiki/entities/3dgenstudio.md`](wiki/entities/3dgenstudio.md)、[`wiki/entities/blender.md`](wiki/entities/blender.md)、[`wiki/concepts/diffusion-model.md`](wiki/concepts/diffusion-model.md)、[`wiki/methods/generative-data-augmentation.md`](wiki/methods/generative-data-augmentation.md)、[`wiki/concepts/text-to-cad.md`](wiki/concepts/text-to-cad.md)、[`wiki/entities/paper-wan-video.md`](wiki/entities/paper-wan-video.md)、[`wiki/entities/img2threejs.md`](wiki/entities/img2threejs.md)、[`wiki/entities/gpufree.md`](wiki/entities/gpufree.md)、[`wiki/entities/articraft.md`](wiki/entities/articraft.md)

## [2026-08-13] structural | docs/graph.html — 路线视图 chip 保持紧凑布局

- **改动：** 筛选浮窗「路线视图」按钮不再随手风琴吃满剩余高度，按内容紧凑换行；「按社区 / 研究机构」仍吃满
- **验证：** `scripts/verify_graph_filter_accordion.cjs`

## [2026-08-13] structural | docs/graph.html — 更新时间 Top N 默认仅计新增，可显示维护更新

- **改动：** 图谱「更新时间 Top N」默认按 git 首次加入日筛新增；「显示维护更新」打开后改用最近触达日（含维护）
- **数据：** `link-graph.json` 节点增 `added` 字段（与 `activity` 并列）
- **验证：** `scripts/verify_graph_recency_date_mode.cjs`

## [2026-08-13] structural | docs/graph.html — 更新时间 Top N 可切换按日期整日保留

- **改动：** 筛选浮窗「更新时间 Top N」增加「按节点 / 按日期」按钮；按日期时滑块按最近活跃日计数，同一天节点一并留下
- **验证：** `scripts/verify_graph_recency_date_mode.cjs`

## [2026-08-13] ingest | sources/papers/flex_pi_arxiv_2608_10860.md — 独立深挖 Flex-π（arXiv:2608.10860）多流算力柔性 WAM

- **触发：** 用户指定论文 Flex-π（UW / AI2；Yan / Liu / Fan / Cai 等；2026-08 arXiv 预印本）；要求自动合并
- **步骤 2.5 复核：** 项目页 <https://flex-pi.github.io/> Code → [geyan21/flex-pi](https://github.com/geyan21/flex-pi)；GitHub API `size=1`、contents 仅 README「The code is ready soon」→ **代码待发布**
- **来源：** [`sources/papers/flex_pi_arxiv_2608_10860.md`](sources/papers/flex_pi_arxiv_2608_10860.md)（arXiv:2608.10860）；[`sources/sites/flex-pi-github-io.md`](sources/sites/flex-pi-github-io.md)；[`sources/repos/flex-pi.md`](sources/repos/flex-pi.md)
- **实体页深化：** [`wiki/entities/paper-flex-pi.md`](wiki/entities/paper-flex-pi.md) — 共享 VAE 吃 pointmap（PSNR 31.1 dB）、56 流组合、CMF、真机/RoboTwin/LIBERO-Plus 表、源码时序图不适用
- **交叉更新：** [`wiki/concepts/world-action-models.md`](wiki/concepts/world-action-models.md)、[`wiki/methods/vla.md`](wiki/methods/vla.md)、[`wiki/methods/generative-world-models.md`](wiki/methods/generative-world-models.md)、[`wiki/tasks/manipulation.md`](wiki/tasks/manipulation.md)、[`wiki/entities/paper-meco-wam-4d-geometry-cotraining.md`](wiki/entities/paper-meco-wam-4d-geometry-cotraining.md)、[`wiki/entities/paper-kairos-native-world-model-stack.md`](wiki/entities/paper-kairos-native-world-model-stack.md)、[`wiki/entities/libero-benchmark.md`](wiki/entities/libero-benchmark.md)、[`wiki/overview/wm-action-consequence-category-01-wam-action-prediction.md`](wiki/overview/wm-action-consequence-category-01-wam-action-prediction.md)

## [2026-08-13] structural | 站点活动改由 git 驱动，log.md 降为叙事层

- **站点：** 首页「最新知识节点」、更新记录热力图、图谱更新明度改为解析 `wiki/` / `roadmap/` 的 git 历史（`A`=新增，`M`/`R`=维护）
- **log.md：** 不再作为站点数据源；仍用于 ingest / query 的意图、开源结论与问答
- **浅克隆：** git 历史不可用时回退读取 log.md
- **规范：** [`schema/ingest-workflow.md`](schema/ingest-workflow.md)、[`schema/log-format.md`](schema/log-format.md)

## [2026-08-13] ingest | sources/papers/vidihand_arxiv_2606_30308.md — 复核 ViDiHand（vidihand.github.io / arXiv:2606.30308）开源状态并补齐站点/仓库归档

- **触发：** 用户指定 <https://vidihand.github.io/> 与 <https://arxiv.org/abs/2606.30308>
- **步骤 2.5：** 项目页 Code → [NTUYWANG103/ViDiHand](https://github.com/NTUYWANG103/ViDiHand)；仓内仅 README「Code will be released soon」→ **代码待发布**
- **新建：** [`sources/sites/vidihand-github-io.md`](sources/sites/vidihand-github-io.md)、[`sources/repos/vidihand.md`](sources/repos/vidihand.md)
- **更新：** [`sources/papers/vidihand_arxiv_2606_30308.md`](sources/papers/vidihand_arxiv_2606_30308.md)、[`wiki/entities/paper-vidihand.md`](wiki/entities/paper-vidihand.md)
- **交叉：** [`wiki/methods/wilor.md`](wiki/methods/wilor.md)


## [2026-08-13] ingest | sources/papers/fact_arxiv_2608_10232.md — FACT / G1 Confined-Space WBP / Flex-π / Neural Introspection Gating 四篇入库

- **来源：**
  - [`sources/papers/fact_arxiv_2608_10232.md`](sources/papers/fact_arxiv_2608_10232.md)（arXiv:2608.10232；UCSD；**已开源** + HF 权重）
  - [`sources/papers/g1_confined_space_wbp_arxiv_2608_10220.md`](sources/papers/g1_confined_space_wbp_arxiv_2608_10220.md)（arXiv:2608.10220；UT Austin；**未开源**）
  - [`sources/papers/flex_pi_arxiv_2608_10860.md`](sources/papers/flex_pi_arxiv_2608_10860.md)（arXiv:2608.10860；UW / AI2；**代码待发布**）
  - [`sources/papers/neural_introspection_gating_arxiv_2608_10824.md`](sources/papers/neural_introspection_gating_arxiv_2608_10824.md)（arXiv:2608.10824；东京大学；IROS 2026；**未开源**）
- **站点 / 仓库：** [`sources/sites/fact-wam-github-io.md`](sources/sites/fact-wam-github-io.md)、[`sources/repos/fact.md`](sources/repos/fact.md)、[`sources/sites/confined-space-wbp-humanoid-github-io.md`](sources/sites/confined-space-wbp-humanoid-github-io.md)、[`sources/sites/flex-pi-github-io.md`](sources/sites/flex-pi-github-io.md)、[`sources/repos/flex-pi.md`](sources/repos/flex-pi.md)、[`sources/sites/neural-introspection-gating-github-io.md`](sources/sites/neural-introspection-gating-github-io.md)
- **新建：**
  - [`wiki/entities/paper-fact.md`](wiki/entities/paper-fact.md)
  - [`wiki/entities/paper-g1-confined-space-wbp.md`](wiki/entities/paper-g1-confined-space-wbp.md)
  - [`wiki/entities/paper-flex-pi.md`](wiki/entities/paper-flex-pi.md)
  - [`wiki/entities/paper-neural-introspection-gating.md`](wiki/entities/paper-neural-introspection-gating.md)
- **交叉更新：** [`wiki/concepts/world-action-models.md`](wiki/concepts/world-action-models.md)、[`wiki/methods/vla.md`](wiki/methods/vla.md)、[`wiki/concepts/whole-body-control.md`](wiki/concepts/whole-body-control.md)、[`wiki/overview/wm-action-consequence-category-01-wam-action-prediction.md`](wiki/overview/wm-action-consequence-category-01-wam-action-prediction.md)
- **机构注册：** [`schema/institutions.json`](schema/institutions.json) 新增 `u-tokyo`

## [2026-08-12] ingest | sources/papers/lucid_arxiv_2608_07746.md — LUCID（arXiv:2608.07746）分层技能级世界模型想象控制入库

- **来源：** [`sources/papers/lucid_arxiv_2608_07746.md`](sources/papers/lucid_arxiv_2608_07746.md)（arXiv:2608.07746；曼彻斯特大学 / IIT；确认未开源）
- **新建：** [`wiki/entities/paper-lucid.md`](wiki/entities/paper-lucid.md)
- **交叉更新：** [`wiki/tasks/loco-manipulation.md`](wiki/tasks/loco-manipulation.md)、[`wiki/methods/ase.md`](wiki/methods/ase.md)、[`wiki/methods/model-based-rl.md`](wiki/methods/model-based-rl.md)、[`wiki/concepts/latent-imagination.md`](wiki/concepts/latent-imagination.md)、[`wiki/entities/paper-shenlan-wm-13-dreamerv3.md`](wiki/entities/paper-shenlan-wm-13-dreamerv3.md)、[`wiki/entities/paper-bfm-38-tokenhsi.md`](wiki/entities/paper-bfm-38-tokenhsi.md)、[`wiki/entities/paper-bfm-15-intermimic.md`](wiki/entities/paper-bfm-15-intermimic.md)

## [2026-08-12] structural | log.md + schema/ingest-workflow.md — 补全 Transformer CV ingest 日志完整路径，修复「更新记录」新增仅 12 条（实为 63）

- **触发：** 用户反馈今日新增 50+ 节点但更新记录只显示 12 个
- **根因：** 批量建页日志用了文件名缩写 / 斜杠串联 / 无反引号通配，`WIKI_PATH_IN_LOG` 无法解析；更新记录默认仅展示 `action=added`
- **修复：** 将当日 Transformer CV ingest 条目改为显式列出全部 56 个 `wiki/...md` 路径；补充 ingest-workflow 路径写法约束
- **验证：** `wiki_activity_from_log` 2026-08-12 `added_count` **12 → 63**（与 git 当日 A 一致）
- **页面：** [`schema/ingest-workflow.md`](schema/ingest-workflow.md)

## [2026-08-12] ingest | sources/courses/transformer_cv_applications_syllabus.md — Transformer 视觉应用八章大纲入库；策展 wiki/entities/transformer-cv-curriculum.md；补齐 CNN/MHA/检测分割/VLM/Mamba/SAM-SEEM 等缺失独立节点（共 56 页，完整路径见下方）；交叉 wiki/overview/hub-vision-backbone.md 与既有 ViT/检测/SAM/BLIP-2 页

- **触发：** Transformer 视觉应用课程大纲 ingest；批量补齐独立详情节点
- **来源：** [`sources/courses/transformer_cv_applications_syllabus.md`](sources/courses/transformer_cv_applications_syllabus.md)
- **新建节点（须写全路径，供更新记录 / latest_wiki_nodes 解析）：**
  - [`wiki/entities/transformer-cv-curriculum.md`](wiki/entities/transformer-cv-curriculum.md)
  - [`wiki/concepts/convolutional-neural-network.md`](wiki/concepts/convolutional-neural-network.md)
  - [`wiki/concepts/multi-head-attention.md`](wiki/concepts/multi-head-attention.md)
  - [`wiki/concepts/image-segmentation-taxonomy.md`](wiki/concepts/image-segmentation-taxonomy.md)
  - [`wiki/concepts/state-space-model-ssm.md`](wiki/concepts/state-space-model-ssm.md)
  - [`wiki/concepts/visual-foundation-model-trends.md`](wiki/concepts/visual-foundation-model-trends.md)
  - [`wiki/concepts/multimodality-basics.md`](wiki/concepts/multimodality-basics.md)
  - [`wiki/concepts/object-detection-metrics.md`](wiki/concepts/object-detection-metrics.md)
  - [`wiki/methods/channel-spatial-attention.md`](wiki/methods/channel-spatial-attention.md)
  - [`wiki/methods/rcnn-family.md`](wiki/methods/rcnn-family.md)
  - [`wiki/methods/fcn-semantic-segmentation.md`](wiki/methods/fcn-semantic-segmentation.md)
  - [`wiki/methods/unet.md`](wiki/methods/unet.md)
  - [`wiki/methods/segnet.md`](wiki/methods/segnet.md)
  - [`wiki/methods/pspnet.md`](wiki/methods/pspnet.md)
  - [`wiki/methods/mask-rcnn.md`](wiki/methods/mask-rcnn.md)
  - [`wiki/entities/dataset-mnist.md`](wiki/entities/dataset-mnist.md)
  - [`wiki/entities/dataset-cifar.md`](wiki/entities/dataset-cifar.md)
  - [`wiki/entities/dataset-imagenet.md`](wiki/entities/dataset-imagenet.md)
  - [`wiki/entities/dataset-jft-300m.md`](wiki/entities/dataset-jft-300m.md)
  - [`wiki/entities/dataset-coco.md`](wiki/entities/dataset-coco.md)
  - [`wiki/entities/dataset-objects365.md`](wiki/entities/dataset-objects365.md)
  - [`wiki/entities/dataset-pascal-voc.md`](wiki/entities/dataset-pascal-voc.md)
  - [`wiki/entities/dataset-cityscapes.md`](wiki/entities/dataset-cityscapes.md)
  - [`wiki/entities/dataset-ade20k.md`](wiki/entities/dataset-ade20k.md)
  - [`wiki/entities/dataset-mapillary.md`](wiki/entities/dataset-mapillary.md)
  - [`wiki/entities/dataset-flickr30k-entities.md`](wiki/entities/dataset-flickr30k-entities.md)
  - [`wiki/entities/dataset-wit.md`](wiki/entities/dataset-wit.md)
  - [`wiki/entities/dataset-vatex.md`](wiki/entities/dataset-vatex.md)
  - [`wiki/entities/lenet5.md`](wiki/entities/lenet5.md)
  - [`wiki/entities/alexnet.md`](wiki/entities/alexnet.md)
  - [`wiki/entities/vggnet.md`](wiki/entities/vggnet.md)
  - [`wiki/entities/tnt.md`](wiki/entities/tnt.md)
  - [`wiki/entities/cvt.md`](wiki/entities/cvt.md)
  - [`wiki/entities/detr.md`](wiki/entities/detr.md)
  - [`wiki/entities/deformable-detr.md`](wiki/entities/deformable-detr.md)
  - [`wiki/entities/retinanet.md`](wiki/entities/retinanet.md)
  - [`wiki/entities/setr.md`](wiki/entities/setr.md)
  - [`wiki/entities/segformer.md`](wiki/entities/segformer.md)
  - [`wiki/entities/clip.md`](wiki/entities/clip.md)
  - [`wiki/entities/blip.md`](wiki/entities/blip.md)
  - [`wiki/entities/llava.md`](wiki/entities/llava.md)
  - [`wiki/entities/minigpt4.md`](wiki/entities/minigpt4.md)
  - [`wiki/entities/instructblip.md`](wiki/entities/instructblip.md)
  - [`wiki/entities/lisa.md`](wiki/entities/lisa.md)
  - [`wiki/entities/sa2va.md`](wiki/entities/sa2va.md)
  - [`wiki/entities/sida.md`](wiki/entities/sida.md)
  - [`wiki/entities/vision-mamba-vim.md`](wiki/entities/vision-mamba-vim.md)
  - [`wiki/entities/vmamba.md`](wiki/entities/vmamba.md)
  - [`wiki/entities/mambair.md`](wiki/entities/mambair.md)
  - [`wiki/entities/rs-mamba.md`](wiki/entities/rs-mamba.md)
  - [`wiki/entities/changemamba.md`](wiki/entities/changemamba.md)
  - [`wiki/entities/videomamba.md`](wiki/entities/videomamba.md)
  - [`wiki/entities/u-mamba.md`](wiki/entities/u-mamba.md)
  - [`wiki/entities/seem.md`](wiki/entities/seem.md)
  - [`wiki/overview/multimodal-llm-development.md`](wiki/overview/multimodal-llm-development.md)
  - [`wiki/comparisons/rnn-cnn-transformer-mamba.md`](wiki/comparisons/rnn-cnn-transformer-mamba.md)
- **交叉：** [`wiki/overview/hub-vision-backbone.md`](wiki/overview/hub-vision-backbone.md)

## [2026-08-12] ingest | sources/blogs/wechat_ai_explore_yao_can_canfd_humanoid_bus.md — 接入 CAN/CAN FD 人形分层总线科普；补充 wiki/concepts/can-fd.md、can-bus-protocol.md、ethercat-protocol.md、wiki/comparisons/can-vs-ethercat-joint-bus.md、wiki/overview/motor-drive-firmware-bus-protocols.md

- **触发：** 用户指定 <https://mp.weixin.qq.com/s/UvjlH1bCsZwNHC2_z12cBg>；要求确保 agent-reach 已装并自动合并
- **工具：** Agent Reach v1.5.0 + wechat-article-for-ai（Camoufox）；短链直连成功
- **来源：** [`sources/blogs/wechat_ai_explore_yao_can_canfd_humanoid_bus.md`](sources/blogs/wechat_ai_explore_yao_can_canfd_humanoid_bus.md)；原始落盘 [`sources/raw/wechat_ai_explore_yao_can_canfd_humanoid_bus_2026-07-20.md`](sources/raw/wechat_ai_explore_yao_can_canfd_humanoid_bus_2026-07-20.md)
- **开源核查（步骤 2.5）：** 无独立项目页 / 配套仓（科普综述）
- **交叉：** [`wiki/concepts/can-fd.md`](wiki/concepts/can-fd.md)（主补充：人形末端分支场景 + 分层图）、[`wiki/concepts/can-bus-protocol.md`](wiki/concepts/can-bus-protocol.md)、[`wiki/concepts/ethercat-protocol.md`](wiki/concepts/ethercat-protocol.md)、[`wiki/comparisons/can-vs-ethercat-joint-bus.md`](wiki/comparisons/can-vs-ethercat-joint-bus.md)、[`wiki/overview/motor-drive-firmware-bus-protocols.md`](wiki/overview/motor-drive-firmware-bus-protocols.md)


## [2026-08-12] structural | wiki/entities/paper-xpolicylab.md — 修复源码运行时序图 Mermaid：消息内 ASCII 分号导致 sequenceDiagram 解析失败

- **触发：** 用户反馈 <https://imchong.github.io/Robotics_Notebooks/detail.html?id=entity-paper-xpolicylab> Mermaid 格式问题；要求自动合并
- **根因：** `U->>XP: pip install -e . ; create_policy.sh NAME` 中 ASCII `;` 被 Mermaid 当作语句分隔符，后续箭头解析失败
- **修复：** 改为 `then` 连接；流程图链式边拆成 `S --- Proto` / `Proto --- C`
- **页面：** [`wiki/entities/paper-xpolicylab.md`](wiki/entities/paper-xpolicylab.md)

## [2026-08-12] ingest | sources/repos/soem.md — 接入 SOEM（OpenEtherCATsociety）；升格 wiki/entities/soem.md；配套 sources/sites/openethercatsociety-github-io.md；交叉更新 wiki/queries/ethercat-master-optimization.md、wiki/concepts/ethercat-protocol.md、wiki/comparisons/can-vs-ethercat-joint-bus.md、wiki/comparisons/ethercat-vs-ethernet-ip.md、wiki/overview/motor-drive-firmware-bus-protocols.md、wiki/entities/canfestival.md

- **触发：** 用户指定仓库 <https://github.com/OpenEtherCATsociety/SOEM>；要求自动合并
- **来源：** [`sources/repos/soem.md`](sources/repos/soem.md)、[`sources/sites/openethercatsociety-github-io.md`](sources/sites/openethercatsociety-github-io.md)
- **新建实体：** [`wiki/entities/soem.md`](wiki/entities/soem.md) — 用户态 EtherCAT 主站 C 库；GPLv3 + 商业双许可；含 `ec_sample` 源码运行时序图
- **开源核查（步骤 2.5）：** **已开源** — [OpenEtherCATsociety/SOEM](https://github.com/OpenEtherCATsociety/SOEM)（v2.0.0）；项目页 <https://openethercatsociety.github.io/>；文档站需登录
- **机构：** 新注册 `rt-labs`（实时实验室（RT-Labs））
- **交叉：** [`wiki/queries/ethercat-master-optimization.md`](wiki/queries/ethercat-master-optimization.md)、[`wiki/concepts/ethercat-protocol.md`](wiki/concepts/ethercat-protocol.md)、[`wiki/comparisons/can-vs-ethercat-joint-bus.md`](wiki/comparisons/can-vs-ethercat-joint-bus.md)、[`wiki/comparisons/ethercat-vs-ethernet-ip.md`](wiki/comparisons/ethercat-vs-ethernet-ip.md)、[`wiki/overview/motor-drive-firmware-bus-protocols.md`](wiki/overview/motor-drive-firmware-bus-protocols.md)、[`wiki/entities/canfestival.md`](wiki/entities/canfestival.md)

## [2026-08-12] structural | references/papers/system-identification.md — 删除与 wiki 概念页同名的论文索引，搜索仅保留 System Identification 概念节点

- **触发：** 用户反馈搜索「System Identification」出现两个节点
- **根因：** `references/papers/system-identification.md` 与 [`wiki/concepts/system-identification.md`](wiki/concepts/system-identification.md) 标题完全相同，均进入搜索/图谱索引
- **处理：** 删除冗余 reference 页；论文列表已由 [`sources/papers/system_identification.md`](sources/papers/system_identification.md) 与概念页「参考来源」覆盖；无其它正文入链
- **保留：** [`wiki/concepts/system-identification.md`](wiki/concepts/system-identification.md)

## [2026-08-12] structural | docs/main.js + docs/graph-tooltip.js — 修复首页「最新知识节点」回填方向与空白日期

- **触发：** 用户反馈首页最新知识节点显示异常
- **根因：** PR#1537 紧凑模式在 `latest_wiki_nodes` 新增不足时，按 `wiki-activity.days` **升序**回填，灌入最早历史节点且缺 `recency`
- **修复：** `collectHomeCompactAddedNodes` 自新到旧回填并补齐日期；`shortenCommunityLabel` 对齐 `community_short_label`（去掉英文括注）
- **测试：** `tests/test_home_compact_added_backfill.py`
- **清单：** [`docs/checklists/frontend-optimization-v1.md`](docs/checklists/frontend-optimization-v1.md)

## [2026-08-12] ingest | sources/sites/canfestival-org.md — 接入 CanFestival 官网；升格 wiki/entities/canfestival.md；配套 sources/repos/canfestival.md；交叉更新 wiki/overview/motor-drive-firmware-bus-protocols.md、wiki/concepts/can-bus-protocol.md、wiki/comparisons/can-vs-ethercat-joint-bus.md

- **触发：** 用户指定站点 <https://canfestival.org/>
- **来源：** [`sources/sites/canfestival-org.md`](sources/sites/canfestival-org.md)、[`sources/repos/canfestival.md`](sources/repos/canfestival.md)
- **新建实体：** [`wiki/entities/canfestival.md`](wiki/entities/canfestival.md) — ANSI-C CANopen Master/Slave；运行时 LGPLv2 / 工具 GPLv2；含源码运行时序图
- **开源核查（步骤 2.5）：** **已开源** — 官网 Code 列 `hg.beremiz.org/canfestival` 等多源；现代 CMake 入口 [beremiz/canfestival](https://github.com/beremiz/canfestival)
- **交叉：** [`wiki/overview/motor-drive-firmware-bus-protocols.md`](wiki/overview/motor-drive-firmware-bus-protocols.md)、[`wiki/concepts/can-bus-protocol.md`](wiki/concepts/can-bus-protocol.md)、[`wiki/comparisons/can-vs-ethercat-joint-bus.md`](wiki/comparisons/can-vs-ethercat-joint-bus.md)、[`wiki/concepts/can-fd.md`](wiki/concepts/can-fd.md)、[`wiki/concepts/ethercat-protocol.md`](wiki/concepts/ethercat-protocol.md)

## [2026-08-12] ingest | sources/papers/effective_degree_arxiv_2605_29823.md — 复核 Effective Degree（arXiv:2605.29823，ICML 2026，清华）：确认已开源可运行；刷新 wiki/entities/paper-effective-degree.md；交叉 wiki/comparisons/deep-learning-optimizers.md、wiki/concepts/transformer.md、wiki/concepts/deep-learning-foundations.md、wiki/methods/reinforcement-learning.md、wiki/methods/ppo.md、wiki/methods/adamw.md；sources/repos/effective-degree.md

- **触发：** 用户指定论文 <https://arxiv.org/abs/2605.29823>；要求自动合并
- **来源：** [`sources/papers/effective_degree_arxiv_2605_29823.md`](sources/papers/effective_degree_arxiv_2605_29823.md)、[`sources/repos/effective-degree.md`](sources/repos/effective-degree.md)
- **实体页：** [`wiki/entities/paper-effective-degree.md`](wiki/entities/paper-effective-degree.md) — 初入库 2026-08-06；本次复核开源状态、补 LA 缩写与交叉引用
- **开源核查（步骤 2.5）：** **已开源** — [xinzaixinzai/Effective-Degree](https://github.com/xinzaixinzai/Effective-Degree)；无独立项目页；未声明 SPDX；默认分支 tip 仍为 2026-05-11
- **机构：** `tsinghua`（已注册）
- **交叉：** [`wiki/comparisons/deep-learning-optimizers.md`](wiki/comparisons/deep-learning-optimizers.md)、[`wiki/concepts/transformer.md`](wiki/concepts/transformer.md)、[`wiki/concepts/deep-learning-foundations.md`](wiki/concepts/deep-learning-foundations.md)、[`wiki/methods/reinforcement-learning.md`](wiki/methods/reinforcement-learning.md)、[`wiki/methods/ppo.md`](wiki/methods/ppo.md)、[`wiki/methods/adamw.md`](wiki/methods/adamw.md)

## [2026-08-12] ingest | sources/papers/ego_oscar_arxiv_2608_08285.md — 接入 Ego-OSCAR / Stereo-550；升格 wiki/entities/paper-ego-oscar.md；交叉更新 wiki/overview/ego-category-01-data-collection.md、wiki/entities/paper-ego4d.md、wiki/entities/paper-hifi-umi.md、wiki/tasks/teleoperation.md；配套 sources/sites/fpvlabs-ego-oscar.md、sources/datasets/stereo-550.md

- **触发：** 用户指定论文 <https://arxiv.org/abs/2608.08285>；要求自动合并
- **来源：** [`sources/papers/ego_oscar_arxiv_2608_08285.md`](sources/papers/ego_oscar_arxiv_2608_08285.md)、[`sources/sites/fpvlabs-ego-oscar.md`](sources/sites/fpvlabs-ego-oscar.md)、[`sources/datasets/stereo-550.md`](sources/datasets/stereo-550.md)
- **新建实体：** [`wiki/entities/paper-ego-oscar.md`](wiki/entities/paper-ego-oscar.md) — ~USD 200 硬同步立体+IMU 头戴；Stereo-550 ~550 h/相机；部署可用会话 **96%**；源码运行时序图「不适用」（采集仓未列）
- **开源核查（步骤 2.5）：** **部分开源** — HF [fpvlabs/stereo-550](https://huggingface.co/datasets/fpvlabs/stereo-550)（gated + 定制许可）+ Hardware Spec PDF + [3D 页](https://www.fpvlabs.ai/ego-oscar/cap)；`fpv-labs` org 无独立 Ego-OSCAR 采集/CAD 仓
- **机构：** 新注册 `fpv-labs`（第一人称视觉实验室（FPV Labs））
- **交叉：** [`wiki/overview/ego-category-01-data-collection.md`](wiki/overview/ego-category-01-data-collection.md)、[`wiki/entities/paper-ego4d.md`](wiki/entities/paper-ego4d.md)、[`wiki/entities/paper-hifi-umi.md`](wiki/entities/paper-hifi-umi.md)、[`wiki/tasks/teleoperation.md`](wiki/tasks/teleoperation.md)

## [2026-08-12] ingest | sources/papers/hil_harc_arxiv_2608_09762.md — 接入 HIL-HARC（arXiv:2608.09762，IIT×UniGe×TU Delft）；升格 wiki/entities/paper-hil-harc.md；交叉 wiki/methods/reinforcement-learning.md、wiki/comparisons/online-vs-offline-rl.md、wiki/concepts/safe-real-world-rl-fine-tuning.md；确认未开源

- **触发：** 用户指定论文 <https://arxiv.org/abs/2608.09762> 与项目页 <https://hil-harc.github.io/>；要求自动合并
- **来源：** [`sources/papers/hil_harc_arxiv_2608_09762.md`](sources/papers/hil_harc_arxiv_2608_09762.md)、[`sources/sites/hil-harc-github-io.md`](sources/sites/hil-harc-github-io.md)
- **新建实体：** [`wiki/entities/paper-hil-harc.md`](wiki/entities/paper-hil-harc.md) — CTDE 连续臂+离散夹爪 + HRA task/grasp 多头 critic；真机均值 **40%→75%**，干预率 **0%**；源码运行时序图「不适用」（确认未开源）
- **开源核查（步骤 2.5）：** **确认未开源** — 项目页 Resources 无 Code；`HIL-HARC/HIL-HARC.github.io` 为静态页
- **机构：** `iit`（已注册）；新注册 `unige` / `tudelft`
- **交叉：** [`wiki/methods/reinforcement-learning.md`](wiki/methods/reinforcement-learning.md)、[`wiki/comparisons/online-vs-offline-rl.md`](wiki/comparisons/online-vs-offline-rl.md)、[`wiki/concepts/safe-real-world-rl-fine-tuning.md`](wiki/concepts/safe-real-world-rl-fine-tuning.md)、[`wiki/tasks/manipulation.md`](wiki/tasks/manipulation.md)、[`wiki/concepts/sim2real.md`](wiki/concepts/sim2real.md)

## [2026-08-12] ingest | sources/papers/slim_05b_arxiv_2608_09771.md — 接入 SLIM-0.5B（arXiv:2608.09771，Fudan×BAAI×THU×RUC）；升格 wiki/entities/paper-slim-05b.md；交叉 wiki/concepts/world-action-models.md、wiki/methods/defi-decoupled-dynamics-vla.md、wiki/entities/libero-benchmark.md、wiki/entities/calvin-benchmark.md；已开源+HF 权重

- **触发：** 用户指定论文 <https://arxiv.org/abs/2608.09771>、项目页 <https://kzz1031.github.io/slim-project-page/>、GitHub / HF；要求自动合并
- **来源：** [`sources/papers/slim_05b_arxiv_2608_09771.md`](sources/papers/slim_05b_arxiv_2608_09771.md)、[`sources/sites/kzz1031-slim-project-page.md`](sources/sites/kzz1031-slim-project-page.md)、[`sources/repos/slim.md`](sources/repos/slim.md)
- **新建实体：** [`wiki/entities/paper-slim-05b.md`](wiki/entities/paper-slim-05b.md) — Stage-1 IDM+FDM 掩码轨迹 + Stage-2 MoT flow；LIBERO **97.5%** / LIBERO-Plus **77.45%** / CALVIN **4.556**；真机 **77.3 ms · 2.01 GiB**
- **开源核查（步骤 2.5）：** **已开源** — [kzz1031/SLIM](https://github.com/kzz1031/SLIM) + [kzzwang/SLIM-LIBERO](https://huggingface.co/kzzwang/SLIM-LIBERO) / SLIM-CALVIN
- **机构：** `fudan` / `baai` / `tsinghua`；新注册 `ruc`
- **交叉：** [`wiki/concepts/world-action-models.md`](wiki/concepts/world-action-models.md)、[`wiki/methods/defi-decoupled-dynamics-vla.md`](wiki/methods/defi-decoupled-dynamics-vla.md)、[`wiki/entities/libero-benchmark.md`](wiki/entities/libero-benchmark.md)、[`wiki/entities/calvin-benchmark.md`](wiki/entities/calvin-benchmark.md)、[`wiki/methods/vla.md`](wiki/methods/vla.md)

## [2026-08-12] ingest | sources/papers/xpolicylab_arxiv_2608_09892.md — 接入 XPolicyLab 技术报告（arXiv:2608.09892，HKU MMLab×THU）；升格 wiki/entities/paper-xpolicylab.md；刷新 wiki/entities/xpolicylab.md；交叉 wiki/entities/robodojo.md、wiki/concepts/simulation-evaluation-infrastructure.md；已开源（GitHub + 项目页）

- **触发：** 用户指定论文 <https://arxiv.org/abs/2608.09892>、GitHub、项目页 <https://xpolicylab.github.io/>；要求自动合并
- **来源：** [`sources/papers/xpolicylab_arxiv_2608_09892.md`](sources/papers/xpolicylab_arxiv_2608_09892.md)、[`sources/sites/xpolicylab-github-io.md`](sources/sites/xpolicylab-github-io.md)、[`sources/repos/xpolicylab.md`](sources/repos/xpolicylab.md)
- **新建/刷新实体：** [`wiki/entities/paper-xpolicylab.md`](wiki/entities/paper-xpolicylab.md)、[`wiki/entities/xpolicylab.md`](wiki/entities/xpolicylab.md) — \(O(NM)\to O(N{+}M)\)；**42** 策略；集成 **>5 h→2 h**（agent skills≈30 min）
- **开源核查（步骤 2.5）：** **已开源** — [XPolicyLab/XPolicyLab](https://github.com/XPolicyLab/XPolicyLab)（Apache-2.0）
- **机构：** `hku` / `tsinghua`（已注册）
- **交叉：** [`wiki/entities/robodojo.md`](wiki/entities/robodojo.md)、[`wiki/concepts/simulation-evaluation-infrastructure.md`](wiki/concepts/simulation-evaluation-infrastructure.md)、[`wiki/queries/embodied-eval-benchmark-selection-loop.md`](wiki/queries/embodied-eval-benchmark-selection-loop.md)

## [2026-08-12] ingest | sources/papers/refertrack_arxiv_2607_20061.md — ReferTrack（arXiv:2607.20061）升格 wiki/entities/paper-refertrack.md；归档 sources/sites/medlartea-refertrack.md、sources/repos/refertrack.md（占位仓）；交叉 wiki/tasks/vision-language-navigation.md、wiki/entities/qwen-robot-nav.md；注册 schema/institutions.json futian-laboratory

- **触发：** 用户指定论文 <https://arxiv.org/abs/2607.20061>、项目页 <https://medlartea.github.io/referTrack/>、代码 <https://github.com/MedlarTea/referTrack>；要求自动合并
- **来源：** [`sources/papers/refertrack_arxiv_2607_20061.md`](sources/papers/refertrack_arxiv_2607_20061.md)、[`sources/sites/medlartea-refertrack.md`](sources/sites/medlartea-refertrack.md)、[`sources/repos/refertrack.md`](sources/repos/refertrack.md)
- **新建实体：** [`wiki/entities/paper-refertrack.md`](wiki/entities/paper-refertrack.md) — referring-then-tracking EVT VLA（Refer-CoT + TVBI + Refer-QA）；EVT-Bench 单视角 STT/DT/AT SR **89.4 / 73.3 / 74.1**；Go2/G1 真机定性；源码运行时序图「不适用」（占位仓）
- **开源核查（步骤 2.5）：** **宣称将开源 / 占位仓** — README TODO 未勾选 checkpoint、数据集、训练与 data engine；目录仅 README / assets / method.pdf
- **机构：** `sustech` / `tencent` / `pku` / `futian-laboratory`（新注册）
- **交叉：** [`wiki/tasks/vision-language-navigation.md`](wiki/tasks/vision-language-navigation.md)、[`wiki/entities/qwen-robot-nav.md`](wiki/entities/qwen-robot-nav.md)

## [2026-08-11] structural | media/site-demo.gif — 按最新图谱重录 README 演示 GIF（3000 节点）

- **触发：** 用户要求更新 README GIF 并自动合并
- **脚本：** [`scripts/record_readme_demo.cjs`](scripts/record_readme_demo.cjs)（78 frames / 2.86 MB；图谱 **3000** 节点 / **26062** 边）
- **流程：** `make export graph` → `docs/` 本地 `http.server 8765` → 重录并写回 `media/site-demo.gif`
- **引用：** [`README.md`](README.md)「在线演示」仍指向 `media/site-demo.gif`
- **清单：** [`docs/checklists/frontend-optimization-v1.md`](docs/checklists/frontend-optimization-v1.md)

## [2026-08-11] ingest | sources/papers/online_mbrl_robot_control_arxiv_2510_18518.md — 接入 Online MBRL via Online Optimization（arXiv:2510.18518，ETH×MPI-IS×EPFL）；升格 wiki/entities/paper-online-mbrl-robot-control.md；交叉 wiki/methods/model-based-rl.md、wiki/concepts/sim2real.md、wiki/entities/paper-td-mpc2.md、wiki/entities/paper-shenlan-wm-13-dreamerv3.md、wiki/entities/robotic-world-model-eth-rsl.md；确认未开源

- **触发：** 用户指定论文 <https://arxiv.org/abs/2510.18518>；要求自动合并
- **来源：** [`sources/papers/online_mbrl_robot_control_arxiv_2510_18518.md`](sources/papers/online_mbrl_robot_control_arxiv_2510_18518.md)
- **新建实体：** [`wiki/entities/paper-online-mbrl-robot-control.md`](wiki/entities/paper-online-mbrl-robot-control.md) — 真机缓冲学动力学 + 真实轨迹闭环一阶策略更新；HEAP **2.5 h / 2.7 cm**，软臂约 **30 episode / 2.95 cm**；源码运行时序图「不适用」（确认未开源）
- **开源核查（步骤 2.5）：** **确认未开源** — arXiv 无项目页/代码链接；公开检索无官方仓
- **机构：** `eth` / `max-planck` / `epfl`（已注册）
- **交叉：** [`wiki/methods/model-based-rl.md`](wiki/methods/model-based-rl.md)、[`wiki/concepts/sim2real.md`](wiki/concepts/sim2real.md)、[`wiki/concepts/latent-imagination.md`](wiki/concepts/latent-imagination.md)、[`wiki/entities/paper-td-mpc2.md`](wiki/entities/paper-td-mpc2.md)、[`wiki/entities/paper-shenlan-wm-13-dreamerv3.md`](wiki/entities/paper-shenlan-wm-13-dreamerv3.md)、[`wiki/entities/robotic-world-model-eth-rsl.md`](wiki/entities/robotic-world-model-eth-rsl.md)

## [2026-08-11] ingest | sources/papers/sc3_eval_arxiv_2606_18610.md — 升格 SC3-Eval（arXiv:2606.18610）自一致视频策略评估器；canonical wiki/entities/paper-sc3-eval.md（合并原 paper-sa-2606-18610-*）；交叉 generative-world-models / virtual-sandbox / eval-loop / Ctrl-World / IRASim / GigaWorld-1；归档 sources/sites/weichengtseng-sc3-eval.md；确认未开源；注册 vector-institute

- **触发：** 用户指定项目页 <https://weichengtseng.github.io/sc3-eval/> 与论文 <https://arxiv.org/abs/2606.18610>；要求自动合并
- **来源：** [`sources/papers/sc3_eval_arxiv_2606_18610.md`](sources/papers/sc3_eval_arxiv_2606_18610.md)、[`sources/sites/weichengtseng-sc3-eval.md`](sources/sites/weichengtseng-sc3-eval.md)；策展摘录刷新 [`sources/papers/sun_awesome_wm_2606_18610_sc3-eval-evaluating-robot-foundation-mod.md`](sources/papers/sun_awesome_wm_2606_18610_sc3-eval-evaluating-robot-foundation-mod.md)
- **新建/升格实体：** [`wiki/entities/paper-sc3-eval.md`](wiki/entities/paper-sc3-eval.md) — 前向–逆向 + 跨视角 + 测试时早停；七 π₀.₅ 闭环 Pearson **0.929** / MMRV **0.119**；源码运行时序图「不适用」（确认未开源）
- **节点合并：** 删除索引级 `wiki/entities/paper-sa-2606-18610-sc3-eval-evaluating-robot-foundation-models-via.md`；[`schema/page-aliases.json`](schema/page-aliases.json) 重定向至 `entity-paper-sc3-eval`；技术地图 / catalog 指向 canonical
- **开源核查（步骤 2.5）：** **确认未开源** — 项目页仅 Paper；`WeiChengTseng/sc3-eval` 为静态 Pages 仓
- **机构：** `utoronto` / `vector-institute`（新注册）/ `nvidia` / `physical-intelligence` / `stanford` / `berkeley`
- **交叉：** [`wiki/methods/generative-world-models.md`](wiki/methods/generative-world-models.md)、[`wiki/overview/world-models-route-03-virtual-sandbox.md`](wiki/overview/world-models-route-03-virtual-sandbox.md)、[`wiki/overview/wm-action-consequence-category-04-eval-posttrain.md`](wiki/overview/wm-action-consequence-category-04-eval-posttrain.md)、[`wiki/queries/embodied-eval-benchmark-selection-loop.md`](wiki/queries/embodied-eval-benchmark-selection-loop.md)、[`wiki/entities/paper-ctrl-world.md`](wiki/entities/paper-ctrl-world.md)、[`wiki/entities/paper-irasim.md`](wiki/entities/paper-irasim.md)、[`wiki/entities/paper-gigaworld-1-policy-evaluation.md`](wiki/entities/paper-gigaworld-1-policy-evaluation.md)、[`wiki/tasks/manipulation.md`](wiki/tasks/manipulation.md)、[`wiki/overview/sun-awesome-wm-technology-map.md`](wiki/overview/sun-awesome-wm-technology-map.md)

## [2026-08-11] ingest | sources/papers/r2s_ego_arxiv_2608_06827.md — 接入 R2S-EGO（XPENG×PolyU 稀疏捕获双代理 Real-to-Sim）；wiki/entities/paper-r2s-ego.md；交叉更新 wiki/concepts/sim2real.md、wiki/entities/paper-agentic-real2sim.md、wiki/methods/crisp-real2sim.md、wiki/entities/awesome-real2sim2real.md、wiki/entities/paper-notebook-gaussgym-an-open-source-real-to-sim-framework-fo.md

- **触发：** 用户指定 R2S-EGO（Shuai Fang 等；XPENG Robotics × 香港理工大学；arXiv:2608.06827）；要求自动合并
- **来源：** [`sources/papers/r2s_ego_arxiv_2608_06827.md`](sources/papers/r2s_ego_arxiv_2608_06827.md)
- **新建实体：** [`wiki/entities/paper-r2s-ego.md`](wiki/entities/paper-r2s-ego.md) — robot/geometry 双代理稀疏捕获场景细化；六视角 19.062 dB PSNR；真机 G1 坐姿 82.5% vs GaussGym 10%；源码运行时序图「不适用」（确认未开源）
- **开源核查（步骤 2.5）：** **确认未开源** — arXiv/PDF 无项目页与 GitHub；公开检索无仓
- **机构：** `xpeng` / `polyu`（已注册）
- **交叉：** [`wiki/concepts/sim2real.md`](wiki/concepts/sim2real.md)、[`wiki/entities/paper-agentic-real2sim.md`](wiki/entities/paper-agentic-real2sim.md)、[`wiki/methods/crisp-real2sim.md`](wiki/methods/crisp-real2sim.md)、[`wiki/entities/awesome-real2sim2real.md`](wiki/entities/awesome-real2sim2real.md)、[`wiki/entities/paper-notebook-gaussgym-an-open-source-real-to-sim-framework-fo.md`](wiki/entities/paper-notebook-gaussgym-an-open-source-real-to-sim-framework-fo.md)

## [2026-08-11] ingest | sources/papers/kilvo_arxiv_2608_05647.md — 复核 KILVO（TMECH / HIT 机器人技术与系统全国重点实验室）；刷新 wiki/entities/paper-kilvo.md 作者与机构名；交叉 wiki/methods/lidar-odometry-fusion.md、wiki/comparisons/lidar-slam-lio-vio-selection.md、roadmap/depth-navigation.md；仓 sources/repos/kilvo.md 仍占位待开放

- **触发：** 用户指定标题 KILVO（IEEE/ASME TMECH；HIT 机器人技术与系统全国重点实验室；作者 Jixin Gao / Fucheng Liu / Teng Zhang / Fusheng Zha）；要求自动合并
- **来源：** [`sources/papers/kilvo_arxiv_2608_05647.md`](sources/papers/kilvo_arxiv_2608_05647.md)、[`sources/repos/kilvo.md`](sources/repos/kilvo.md)
- **实体刷新：** [`wiki/entities/paper-kilvo.md`](wiki/entities/paper-kilvo.md) — 机构名对齐「全国重点实验室」；补作者行；开源复核日 2026-08-11（仍占位）；源码运行时序图「不适用」
- **开源核查（步骤 2.5）：** **代码待开放** — [JixinGao/KILVO](https://github.com/JixinGao/KILVO) 仅 README「available soon」
- **交叉：** [`wiki/methods/lidar-odometry-fusion.md`](wiki/methods/lidar-odometry-fusion.md)、[`wiki/comparisons/lidar-slam-lio-vio-selection.md`](wiki/comparisons/lidar-slam-lio-vio-selection.md)、[`roadmap/depth-navigation.md`](roadmap/depth-navigation.md)、[`wiki/entities/fast-lio.md`](wiki/entities/fast-lio.md)

## [2026-08-11] ingest | sources/papers/autointervene_arxiv_2608_07065.md — 接入 AutoIntervene（arXiv:2608.07065）Action Chunk 校准自动接管；升格 wiki/entities/paper-autointervene.md；交叉 wiki/methods/action-chunking.md、wiki/methods/dagger.md、wiki/tasks/bimanual-manipulation.md、wiki/entities/paper-why-action-chunking-improves-bc.md、roadmap/depth-vla.md；归档 sources/sites/aus-bot-autointervene.md；确认未开源

- **触发：** 用户指定论文 <https://arxiv.org/abs/2608.07065> 与项目页 <https://aus.bot/AutoIntervene/>；要求自动合并
- **来源：** [`sources/papers/autointervene_arxiv_2608_07065.md`](sources/papers/autointervene_arxiv_2608_07065.md)、[`sources/sites/aus-bot-autointervene.md`](sources/sites/aus-bot-autointervene.md)
- **新建实体：** [`wiki/entities/paper-autointervene.md`](wiki/entities/paper-autointervene.md) — phase-local/global visual-action 支持 + 分位数校准双向接管；九任务双臂真机 R2 avg 80%；源码运行时序图「不适用」（确认未开源）
- **开源核查（步骤 2.5）：** **确认未开源** — 项目页无训练仓；`123qwedsa123/AutoIntervene` 仅为静态页镜像
- **机构：** 注册 `usyd` / `pair-lab` / `vanderbilt`
- **交叉：** [`wiki/methods/action-chunking.md`](wiki/methods/action-chunking.md)、[`wiki/methods/dagger.md`](wiki/methods/dagger.md)、[`wiki/tasks/bimanual-manipulation.md`](wiki/tasks/bimanual-manipulation.md)、[`wiki/entities/paper-why-action-chunking-improves-bc.md`](wiki/entities/paper-why-action-chunking-improves-bc.md)、[`roadmap/depth-vla.md`](roadmap/depth-vla.md)、[`wiki/methods/vla.md`](wiki/methods/vla.md)

## [2026-08-11] ingest | sources/papers/tempo_arxiv_2608_07314.md — 接入 TEMPO（arXiv:2608.07314）VLA 语义–动作双频 RL 后训练；升格 wiki/entities/paper-tempo.md；交叉 wiki/methods/vla.md、wiki/entities/calvin-benchmark.md、roadmap/depth-vla.md、wiki/methods/defi-decoupled-dynamics-vla.md；归档 sources/sites/tempo-anonymous-4open.md；确认未开源

- **触发：** 用户指定论文 <https://arxiv.org/abs/2608.07314> 与项目页 <https://anonymous.4open.science/w/tempo-page/>；要求自动合并
- **来源：** [`sources/papers/tempo_arxiv_2608_07314.md`](sources/papers/tempo_arxiv_2608_07314.md)、[`sources/sites/tempo-anonymous-4open.md`](sources/sites/tempo-anonymous-4open.md)
- **新建实体：** [`wiki/entities/paper-tempo.md`](wiki/entities/paper-tempo.md) — 冻结 VLM + 双 TD3 环（projection 低频 / expert 高频）；CALVIN ABC→D SR5 81.7% / Avg.Len. 4.59；源码运行时序图「不适用」（确认未开源）
- **开源核查（步骤 2.5）：** **确认未开源** — 匿名页 Cloudflare 拦截且论文无 GitHub
- **机构：** 注册 `zjgsu` / `kth`
- **交叉：** [`wiki/methods/vla.md`](wiki/methods/vla.md)、[`wiki/entities/calvin-benchmark.md`](wiki/entities/calvin-benchmark.md)、[`roadmap/depth-vla.md`](roadmap/depth-vla.md)、[`wiki/methods/defi-decoupled-dynamics-vla.md`](wiki/methods/defi-decoupled-dynamics-vla.md)、[`wiki/methods/action-chunking.md`](wiki/methods/action-chunking.md)

## [2026-08-11] ingest | sources/papers/fault_tolerant_locomotion_arxiv_2608_07328.md — 接入 Fault-Tolerant Locomotion（arXiv:2608.07328）执行器功率损失自适应步态；升格 wiki/entities/paper-fault-tolerant-locomotion.md；交叉 wiki/tasks/locomotion.md、roadmap/depth-rl-locomotion.md、wiki/entities/paper-actuator-constrained-rl-high-speed-quadruped-locomotion.md；归档 sources/sites/fault-tolerant-locomotion-github-io.md；确认未开源

- **触发：** 用户指定论文 <https://arxiv.org/abs/2608.07328> 与项目页 <https://gianni0907.github.io/fault_tolerant_locomotion/>；要求自动合并
- **来源：** [`sources/papers/fault_tolerant_locomotion_arxiv_2608_07328.md`](sources/papers/fault_tolerant_locomotion_arxiv_2608_07328.md)、[`sources/sites/fault-tolerant-locomotion-github-io.md`](sources/sites/fault-tolerant-locomotion-github-io.md)
- **新建实体：** [`wiki/entities/paper-fault-tolerant-locomotion.md`](wiki/entities/paper-fault-tolerant-locomotion.md) — 非对称 actor–critic + latent-alignment + 可学习步态频率；KYON 68 kg 崎岖仿真与平地真机；源码运行时序图「不适用」（确认未开源）
- **开源核查（步骤 2.5）：** **确认未开源** — 项目页仅视频/架构，无 GitHub
- **机构：** `iit`（已注册）
- **交叉：** [`wiki/tasks/locomotion.md`](wiki/tasks/locomotion.md)、[`roadmap/depth-rl-locomotion.md`](roadmap/depth-rl-locomotion.md)、[`wiki/entities/paper-actuator-constrained-rl-high-speed-quadruped-locomotion.md`](wiki/entities/paper-actuator-constrained-rl-high-speed-quadruped-locomotion.md)、[`wiki/concepts/sim2real.md`](wiki/concepts/sim2real.md)、[`wiki/methods/reinforcement-learning.md`](wiki/methods/reinforcement-learning.md)

## [2026-08-11] ingest | sources/blogs/dyna_2_million_hour_wam.md — 接入 Dyna-2（dyna.co/dyna-2）百万小时 WAM 跨具身缩放律；升格 wiki/entities/dyna-2.md；交叉 wiki/concepts/world-action-models.md、wiki/concepts/embodied-scaling-laws.md、wiki/methods/egoscale.md、roadmap/depth-wam.md；归档 sources/sites/dyna-co-dyna-2.md、sources/sites/dyna-co.md；注册机构 dyna-robotics；确认未开源

- **触发：** 用户指定 <https://www.dyna.co/dyna-2>；要求自动合并
- **来源：** [`sources/blogs/dyna_2_million_hour_wam.md`](sources/blogs/dyna_2_million_hour_wam.md)、[`sources/sites/dyna-co-dyna-2.md`](sources/sites/dyna-co-dyna-2.md)、[`sources/sites/dyna-co.md`](sources/sites/dyna-co.md)
- **新建实体：** [`wiki/entities/dyna-2.md`](wiki/entities/dyna-2.md) — MoT–DiT flow-matching WAM；1k–1M h 嵌套梯子；人 held-out + 人→机零样本幂律；video co-train 消融；后训练 14 任务 20%→53%；源码运行时序图「不适用」（闭源）
- **开源核查（步骤 2.5）：** **确认未开源** — 研究页与公司站无 GitHub / HF / 数据集入口
- **机构：** 注册 `dyna-robotics` → 戴纳机器人（Dyna Robotics）
- **交叉：** [`wiki/concepts/world-action-models.md`](wiki/concepts/world-action-models.md)、[`wiki/concepts/embodied-scaling-laws.md`](wiki/concepts/embodied-scaling-laws.md)、[`wiki/methods/egoscale.md`](wiki/methods/egoscale.md)、[`roadmap/depth-wam.md`](roadmap/depth-wam.md)

## [2026-08-10] structural | .github/workflows/export.yml + weekly-lint.yml + pages.yml — 修复 Auto Export/Weekly Lint 与 main 并发推送的 git push 竞态（rebase 重试 + 串行 concurrency）；Pages deploy 超时放宽至 30 分钟

- **触发：** 用户要求修复全部 GitHub Actions 问题。tip `main`（Harness VLA #1524）六项 workflow 已全绿；历史失败主要为 (1) 社区命名（已由 #1523 修复）(2) Auto Export `git push` 被并发 main 推送拒绝 (3) Pages `deployment_queued` 超时。
- **改动：** [`export.yml`](.github/workflows/export.yml) / [`weekly-lint.yml`](.github/workflows/weekly-lint.yml) 改为串行 concurrency + push 失败时 `fetch`/`rebase` 最多 5 次；[`pages.yml`](.github/workflows/pages.yml) deploy `timeout-minutes: 30`。

## [2026-08-10] ingest | sources/papers/harness_vla_arxiv_2607_08448.md — 复核 Harness VLA（arXiv:2607.08448v3）+ 项目页 harnessvla.github.io；刷新 wiki/entities/paper-harness-vla.md；交叉 sources/sites/harnessvla-github-io.md、sources/repos/rpent.md、wiki/methods/vla.md、wiki/overview/vla-open-source-repro-landscape-2025.md；代码仍开源 RLinf/RPent

- **触发：** 用户指定论文 <https://arxiv.org/pdf/2607.08448v3> 与项目页 <https://harnessvla.github.io/>；要求自动合并。本库 2026-07-22 已首轮入库，本轮按 v3 PDF / 项目页 / RPent README 复核深化。
- **来源刷新：** [`sources/papers/harness_vla_arxiv_2607_08448.md`](sources/papers/harness_vla_arxiv_2607_08448.md)、[`sources/sites/harnessvla-github-io.md`](sources/sites/harnessvla-github-io.md)、[`sources/repos/rpent.md`](sources/repos/rpent.md)
- **实体刷新：** [`wiki/entities/paper-harness-vla.md`](wiki/entities/paper-harness-vla.md) — 钉死 PDF v3；补附录原语用量（LIBERO `VLA_ACT` 15.8% / RoboCasa 35.3% / C2R 47.4%）；开源边界与 §5 局限对齐 2026-08-10 Feature Matrix
- **交叉：** [`wiki/methods/vla.md`](wiki/methods/vla.md)、[`wiki/overview/vla-open-source-repro-landscape-2025.md`](wiki/overview/vla-open-source-repro-landscape-2025.md)
- **开源核查（步骤 2.5）：** **已开源** — 项目页 Code → `RLinf/RPent`；Pi0.5+LIBERO-PRO 完整路径可用；RoboCasa/真机等矩阵项仍未打勾

## [2026-08-10] ingest | sources/repos/awesome-world-models.md + awesome-egocentric-vision.md + awesome-touch.md + awesome-real2sim2real.md — 接入 sun254667 四份 Awesome 策展清单，并将清单内论文节点化为独立详情页

- **触发：** 用户指定四仓并要求列表内论文均有独立详情节点；要求自动合并
- **来源：** [`sources/repos/awesome-world-models.md`](sources/repos/awesome-world-models.md)、[`sources/repos/awesome-egocentric-vision.md`](sources/repos/awesome-egocentric-vision.md)、[`sources/repos/awesome-touch.md`](sources/repos/awesome-touch.md)、[`sources/repos/awesome-real2sim2real.md`](sources/repos/awesome-real2sim2real.md)
- **列表实体：** [`wiki/entities/awesome-world-models.md`](wiki/entities/awesome-world-models.md)、[`wiki/entities/awesome-egocentric-vision.md`](wiki/entities/awesome-egocentric-vision.md)、[`wiki/entities/awesome-touch.md`](wiki/entities/awesome-touch.md)、[`wiki/entities/awesome-real2sim2real.md`](wiki/entities/awesome-real2sim2real.md)
- **技术地图：** [`wiki/overview/sun-awesome-wm-technology-map.md`](wiki/overview/sun-awesome-wm-technology-map.md)、[`wiki/overview/sun-awesome-ego-technology-map.md`](wiki/overview/sun-awesome-ego-technology-map.md)、[`wiki/overview/sun-awesome-touch-technology-map.md`](wiki/overview/sun-awesome-touch-technology-map.md)、[`wiki/overview/sun-awesome-r2s2r-technology-map.md`](wiki/overview/sun-awesome-r2s2r-technology-map.md)
- **论文节点化：** 生成器 [`scripts/generate_sun254667_awesome_paper_entities.py`](scripts/generate_sun254667_awesome_paper_entities.py)；目录 [`sources/papers/sun_awesome_*_catalog.md`](sources/papers/sun_awesome_wm_catalog.md)；新建索引级 [`wiki/entities/paper-sa-*.md`](wiki/entities/)（约 838 页），同 arXiv 已有 `paper-*` canonical 则复用；948 条清单条目均可点入详情节点
- **开源核查（步骤 2.5）：** 四仓均为 **已开源策展清单**（Markdown；非训练栈）；条目级代码以各论文项目页为准
- **交叉：** [`wiki/concepts/world-action-models.md`](wiki/concepts/world-action-models.md)、[`wiki/overview/ego-9-papers-technology-map.md`](wiki/overview/ego-9-papers-technology-map.md)、[`wiki/overview/hub-tactile.md`](wiki/overview/hub-tactile.md)、[`wiki/overview/hub-sim2real.md`](wiki/overview/hub-sim2real.md)、[`wiki/concepts/tactile-sensing.md`](wiki/concepts/tactile-sensing.md)、[`wiki/concepts/sim2real.md`](wiki/concepts/sim2real.md)

## [2026-08-10] ingest | sources/repos/plotjuggler.md — 复核 PlotJuggler：接入官方站 plotjuggler.io 与 PJ4 beta，刷新实体页并交叉 Foxglove/rerun/MCAP

- **触发：** 用户指定仓库 <https://github.com/PlotJuggler/PlotJuggler>；要求自动合并。本库 2026-06 已有条目，本轮按上游 README / Releases / 官方站复核深化。
- **来源：** [`sources/repos/plotjuggler.md`](sources/repos/plotjuggler.md)、新建 [`sources/sites/plotjuggler-io.md`](sources/sites/plotjuggler-io.md)
- **实体刷新：** [`wiki/entities/plotjuggler.md`](wiki/entities/plotjuggler.md) — 3.17.x 稳定 vs PJ4 beta（`3.999.x`）、MCAP/Python 变换/Foxglove WebSocket bridge、正牌域名 `plotjuggler.io` vs 仿冒 `plotjuggler.com`
- **开源核查（步骤 2.5）：** **已开源** — 主仓 MPL-2.0 + 独立 ROS/MQTT/LSL 插件仓；站内 Download → GitHub Releases
- **交叉：** [`wiki/entities/foxglove-studio.md`](wiki/entities/foxglove-studio.md)、[`wiki/entities/rerun-io.md`](wiki/entities/rerun-io.md)、[`wiki/entities/mcap-log-format.md`](wiki/entities/mcap-log-format.md)、[`wiki/queries/robot-policy-debug-playbook.md`](wiki/queries/robot-policy-debug-playbook.md)、[`wiki/queries/hmi-opensource-projects-coverage.md`](wiki/queries/hmi-opensource-projects-coverage.md)（修正组织 URL）

## [2026-08-10] structural | scripts/generate_link_graph.py + pages/tests/export workflows — 新增/维护完全以 git 首次加入日为准；Pages 等 CI 改为 fetch-depth: 0（浅克隆会把全库标成 tip 日 Added，导致「只有今天有新增」）

- **现象：** 线上更新记录几乎只有 2026-08-10 有「新增」，历史日 added_count 全 0
- **根因：** `actions/checkout` 默认浅克隆下 `git log --name-status` 把几乎所有 wiki/roadmap 标成 tip 日 `A`
- **修复：** workflows `fetch-depth: 0`；`wiki_git_added_dates` 检测浅克隆则跳过并回退日志首日；新建日仍优先 git

## [2026-08-10] structural | scripts/generate_link_graph.py — 修复更新页「新增 / 维护」误判与历史通配幽灵活动日（例：当日新建 wiki/entities/paper-ace-data-0.md 应标「新增」，且不得出现在 7/24·7/28 活动列表）

- **触发：** 线上更新页将当日新建的 [`wiki/entities/paper-ace-data-0.md`](wiki/entities/paper-ace-data-0.md) 标成「维护」
- **根因：**
  - `wiki_first_log_dates` 对 structural 块按「当前仓库树」展开 paper 实体通配，把 2026-08-10 新建页误标为 2026-07-24 首次出现
  - `wiki_activity_from_log` / 最新节点同样展开历史通配，把尚未入库的页灌进 7/28 等旧活动日（7/24 后仍有 depth/paper 通配 structural）
- **修复：** 首次出现日只认显式路径；**新建日以 git 加入日为准**（无 git 才回退日志）；通配展开跳过 `git_added > log_date` 的幽灵路径；同日 →「新增」，跨日 →「维护」
- **测试：** `tests/test_wiki_first_log_dates.py`、`tests/test_generate_link_graph_wiki_activity.py` 增补通配/幽灵/新建日用例

## [2026-08-10] ingest | sources/papers/ace_data_0_arxiv_2607_28625.md — 接入 ACE-Data-0（arXiv:2607.28625）Ambient Capture 家居多模态数据集；升格 wiki/entities/paper-ace-data-0.md；交叉 wiki/entities/paper-data-pyramid-embodied-manipulation.md、wiki/entities/paper-ego4d.md、wiki/entities/rekadaily-10k-dataset.md、wiki/methods/egoscale.md、wiki/entities/paper-ace-brain-0-5.md、roadmap/depth-vla.md；归档 sources/sites/ace-data-0-github-io.md、sources/datasets/ace-data-0.md（HF gated，训练代码未见）

- **触发：** 用户指定论文 *ACE-Data-0: Human-Centric Ambient Capture as Embodied Data Engine*；要求自动合并
- **来源：** [`sources/papers/ace_data_0_arxiv_2607_28625.md`](sources/papers/ace_data_0_arxiv_2607_28625.md)、[`sources/sites/ace-data-0-github-io.md`](sources/sites/ace-data-0-github-io.md)、[`sources/datasets/ace-data-0.md`](sources/datasets/ace-data-0.md)
- **新建实体：** [`wiki/entities/paper-ace-data-0.md`](wiki/entities/paper-ace-data-0.md) — 双尺度 ACE 采集 + 150h/17M/75k episodes + 三层 benchmark；源码运行时序图「不适用」（训练/评测代码未见，2026-08-10）
- **开源核查（步骤 2.5）：** **部分开源** — HF `ACERobotics/ACE-Data-0` gated 研究许可已上线；项目页未列 GitHub
- **机构：** `ntu` / `ace-robotics`（已注册）
- **交叉：** [`wiki/entities/paper-data-pyramid-embodied-manipulation.md`](wiki/entities/paper-data-pyramid-embodied-manipulation.md)、[`wiki/entities/paper-ego4d.md`](wiki/entities/paper-ego4d.md)、[`wiki/entities/rekadaily-10k-dataset.md`](wiki/entities/rekadaily-10k-dataset.md)、[`wiki/methods/egoscale.md`](wiki/methods/egoscale.md)、[`wiki/entities/paper-ace-brain-0-5.md`](wiki/entities/paper-ace-brain-0-5.md)、[`roadmap/depth-vla.md`](roadmap/depth-vla.md)

## [2026-08-10] ingest | sources/papers/splc_arxiv_2607_01925.md — 接入 SPLC（arXiv:2607.01925）社交偏好学习人群导航；升格 wiki/entities/paper-splc.md；交叉 wiki/entities/paper-icrowdnav.md、wiki/entities/paper-notebook-learning-social-navigation-from-positive-and-neg.md、wiki/comparisons/online-vs-offline-rl.md、wiki/overview/navigation-slam-autonomy-stack.md、roadmap/depth-navigation.md；仓库 sources/repos/splc.md（代码 coming soon）

- **触发：** 用户指定论文 *SPLC: Social Preference Learning for Crowd Robot Navigation*（Zixuan Chen、Hao Fu、Haiwen Hu、Shiquan Zheng）；要求自动合并
- **来源：** [`sources/papers/splc_arxiv_2607_01925.md`](sources/papers/splc_arxiv_2607_01925.md)、[`sources/repos/splc.md`](sources/repos/splc.md)
- **新建实体：** [`wiki/entities/paper-splc.md`](wiki/entities/paper-splc.md) — 社交偏好准则自动标注 + Preference Transformer 奖励 + IQL/CQL/TD3BC；源码运行时序图「不适用」（README coming soon，2026-08-10）
- **开源核查（步骤 2.5）：** **宣称将开源 / 截至入库日无可运行实现** — GitHub 仅 README + Graphical Abstract；演示 [YouTube](https://youtu.be/vkWjg4Qcybg)
- **机构：** 注册 `wust` → 武汉科技大学（Wuhan University of Science and Technology）
- **交叉：** [`wiki/entities/paper-icrowdnav.md`](wiki/entities/paper-icrowdnav.md)、[`wiki/entities/paper-notebook-learning-social-navigation-from-positive-and-neg.md`](wiki/entities/paper-notebook-learning-social-navigation-from-positive-and-neg.md)、[`wiki/comparisons/online-vs-offline-rl.md`](wiki/comparisons/online-vs-offline-rl.md)、[`wiki/overview/navigation-slam-autonomy-stack.md`](wiki/overview/navigation-slam-autonomy-stack.md)、[`roadmap/depth-navigation.md`](roadmap/depth-navigation.md)

## [2026-08-10] ingest | sources/papers/omega0_arxiv_2608_06375.md — 深化 ω-0（arXiv:2608.06375）：复核 Code/Dataset 仍 WIP；补 FAST/Qwen3-VL/V-JEPA/Wan/RTC 与消融；交叉 wiki/entities/paper-omega-0.md、wiki/entities/paper-being-m07-humanoid-latent-wam.md、wiki/methods/sonic-motion-tracking.md、wiki/tasks/teleoperation.md、wiki/concepts/world-action-models.md、wiki/tasks/loco-manipulation.md、wiki/entities/paper-motionwam-humanoid-loco-manipulation-wam.md

- **触发：** 用户指定 <https://arxiv.org/abs/2608.06375> 与项目页 <https://gentlefress.github.io/OMEGA-0_page/>；要求自动合并。该篇已于 2026-08-08 首轮入库，本轮做开源复核与方法深化。
- **来源：** [`sources/papers/omega0_arxiv_2608_06375.md`](sources/papers/omega0_arxiv_2608_06375.md)、[`sources/sites/omega0-github-io.md`](sources/sites/omega0-github-io.md)
- **实体深化：** [`wiki/entities/paper-omega-0.md`](wiki/entities/paper-omega-0.md) — 三阶段骨干（FAST / Qwen3-VL / V-JEPA / Wan / RTC）、消融表、Being-M0.7 对照；源码时序图仍「不适用」（Code/Dataset WIP，2026-08-10 复核）
- **交叉：** [`wiki/entities/paper-being-m07-humanoid-latent-wam.md`](wiki/entities/paper-being-m07-humanoid-latent-wam.md)、[`wiki/methods/sonic-motion-tracking.md`](wiki/methods/sonic-motion-tracking.md)、[`wiki/tasks/teleoperation.md`](wiki/tasks/teleoperation.md)；并 bump [`wiki/concepts/world-action-models.md`](wiki/concepts/world-action-models.md)、[`wiki/tasks/loco-manipulation.md`](wiki/tasks/loco-manipulation.md)、[`wiki/entities/paper-motionwam-humanoid-loco-manipulation-wam.md`](wiki/entities/paper-motionwam-humanoid-loco-manipulation-wam.md)

## [2026-08-09] structural | roadmap/depth-motion-retargeting.md + depth-navigation.md + depth-bfm.md + depth-imitation-learning.md + depth-loco-manipulation.md + depth-wam.md + depth-vla.md + depth-humanoid-hardware-design.md — 补齐 2026-08-06（roadmap 上次触达日）以来入库的 15 篇里程碑级知识节点到八条纵深路线「推荐读什么」/关键词区块，覆盖运动重定向、导航、BFM、模仿学习、Loco-Manipulation、WAM、VLA、人形硬件设计

- **触发：** 计划任务巡检——审计 `roadmap/*.md` 自 2026-08-06 02:12 UTC（commit f863707）以来的全部新增知识节点（CommNav 与 HUMEMBR 已由后续 ingest 提交补入 `depth-navigation.md`、CMU Optimal Control 已补入 `motion-control.md`，跳过），确认 15 篇为对应纵深路线的里程碑级或教学级读物；跳过 Immersive Social VR+LLM Humanoids（未开源、小样本，与 Teleopit 叙事重叠）、HumanTouch（数据代码均未发布）、Macrodata 博客（工程配方非里程碑论文，与 ViDiHand 数据源重叠）、UMA（avatar 重建非机器人控制范畴）、DyPES-VLA（未开源，与同期 JoyAI-RA 0.5 主题重叠）、Dynibo/CLI-Anything/Codex Security/embodied-interview-qa（工具类非机器人纵深）、Ego4D/RekaCS2-10k/RekaDaily-10k（通用数据集非机器人专属）、HarnessBank/SkillCorpus/DASH-OPSD（编码/推理 agent 研究非机器人方向）、ROBOTIS 系列与 DDT_Lab/云深处 RL 训练仓（厂商工具生态非论文级贡献）、Effective Degree（通用深度学习论文与人形路线无关）；仅在既有 Stage 内追加条目，未新建 Stage
- **[`roadmap/depth-motion-retargeting.md`](roadmap/depth-motion-retargeting.md)：** Stage 3 方法谱系补 [KDMR](wiki/entities/paper-kdmr.md)（GRF 锚定多接触全身轨迹优化）、[SPARK](wiki/entities/paper-spark-skeleton-aligned-retargeting.md)（骨架校准 + KTO→ID→KDTO 渐进优化）、[Shooting for Contact / DSMS](wiki/entities/paper-shooting-for-contact.md)（接触隐式多重打靶，已开源，G1 零样本爬行/180° 跳转）；Stage 6 方向 A 关键词补 [X-Morph](wiki/entities/paper-xmorph.md)（人体运动跨形态重定向到四足/六足）
- **[`roadmap/depth-navigation.md`](roadmap/depth-navigation.md)：** Stage 1 补 [SLAMFormer-∞](wiki/entities/paper-slamformer-infinity.md)（无界长程单目稠密 SLAM）与 [KILVO](wiki/entities/paper-kilvo.md)（人形多传感器 ESIKF 里程计，真机端到端均值 0.0145 m）
- **[`roadmap/depth-bfm.md`](roadmap/depth-bfm.md)：** Stage 2 跟踪主线补 [PFM-HR](wiki/entities/paper-pfm-hr.md)（无序姿态 Flow Matching 先验，冻结挂载 ADD/BeyondMimic，高动态技能样本效率 +14–29%）
- **[`roadmap/depth-imitation-learning.md`](roadmap/depth-imitation-learning.md)：** Stage 4 对抗式学习补 [CMP](wiki/entities/paper-cmp.md)（上下文条件重权 AMP/SMP，参考失衡场景显著更稳）
- **[`roadmap/depth-loco-manipulation.md`](roadmap/depth-loco-manipulation.md)：** Stage 1 全身控制基座补 [AGILE](wiki/entities/paper-agile-humanoid-loco-manipulation.md)（NVIDIA 全生命周期 RL 工作流，G1+T1 双机验证，已开源）；Stage 4 统一模型补 [ω-0](wiki/entities/paper-omega-0.md)（人形并发 loco-manip Joint WAM，G1 11 任务 Omni SR 81.8%）
- **[`roadmap/depth-wam.md`](roadmap/depth-wam.md)：** Stage 3 Joint WAM 补 [ω-0](wiki/entities/paper-omega-0.md) 与 [DreamWAM](wiki/entities/paper-dreamwam.md)（beyond-RGB 联合去噪，LIBERO-Plus 69.16%→75.47%，已开源）
- **[`roadmap/depth-vla.md`](roadmap/depth-vla.md)：** Stage 2 VLA 主线补 [BridgeVLA++](wiki/entities/paper-bridgevla-plusplus.md)（时空记忆 3D VLA，已开源）；Stage 3 数据与 Scaling 补 [JoyAI-RA 0.5](wiki/entities/paper-joyai-ra-05.md)（人视频缩放主轴，AgiBot G1 seen 92.0/unseen 75.5）；Stage 4 部署与整合补 [RTCF](wiki/entities/paper-rtcf.md)（免训练测试时动作纠偏）
- **[`roadmap/depth-humanoid-hardware-design.md`](roadmap/depth-humanoid-hardware-design.md)：** Stage 1 构型与机械布局补 [人形膝/腿主承力链为何通常避开谐波减速器](wiki/concepts/humanoid-knee-harmonic-drive-limits.md)（PRS/摆线 RV/QDD 三条替代路线对照）
- **口径：** 每条路线仅在既有 Stage 结构内追加 bullet / 关键词，未新建 Stage、未改动其余既有内容

## [2026-08-09] lint | wiki/entities/paper-dypes-vla.md + wiki/entities/paper-slamformer-infinity.md + scripts/lint_wiki.py — 全量健康检查：阻塞型 0、信息型 3 → 0

- **触发：** 全量知识库 lint 巡检，要求把信息型预警一并收敛
- **基线：** `python3 scripts/lint_wiki.py` 阻塞型 **0**、信息型 **3**（陈旧声明巡检 2 条 + 缺页概念巡检 1 条）
- **复核与改写（陈旧声明 V1）：** 两条命中均为无主体、无时限的裸「SOTA」——
  - [`wiki/entities/paper-dypes-vla.md`](wiki/entities/paper-dypes-vla.md)「为什么重要」原作「仿真三榜 SOTA 级数字」，改写为「仿真三榜达到论文（2026-08）所列最强对照的同级数字」；
  - [`wiki/entities/paper-slamformer-infinity.md`](wiki/entities/paper-slamformer-infinity.md)「实验与评测（论文报告摘要）」表 TUM / Replica 行原作「与 SOTA 竞争」，改写为「与论文（2026-08）所列最强基线竞争」。
  两处均保留原数字与结论，只把绝对化措辞降级为**有主体（论文口径）、有时点（2026-08）**的相对表述；同步 `updated` 2026-08-08 → 2026-08-09 记录本次复核。
- **复核与登记（缺页概念巡检 V1）：** 候选 `URDF`（被 6 个页面加粗/反引号引用）经核查**已由 [`wiki/concepts/urdf-robot-description.md`](wiki/concepts/urdf-robot-description.md) 完整覆盖**（含一句话定义、英文缩写速查、`n_q` vs `n_v` 等），仅因 slug 与页面 stem 不同名而误报；按既有 `ethercat` / `wbc` / `wam` 先例登记进 [`scripts/lint_wiki.py`](scripts/lint_wiki.py) 的 `MISSING_CONCEPT_COVERED_ELSEWHERE`（附映射注释），**不新建重复概念页**
- **其余检查：** 阻塞型 22 项与其余信息型 14 项本轮基线即为 0；Sources 覆盖率 **2123/2123 (100%)**；`eval_search_quality` 通过率 **37/37 (100%)**；`check_export_quality` **12/12**；`pytest` 372 passed + 486 subtests
- **门禁：** `make ci-preflight` 通过（lint 阻塞型 0 + 信息型 0）；派生 `exports/lint-report.md`、`catalog.md`、统计与 `docs/exports/` 随本次提交同步

## [2026-08-09] ingest | sources/repos/dynibo.md — 接入 Dynibo（Rust 运动学/动力学库）；升格 wiki/entities/dynibo.md；交叉 wiki/entities/pinocchio.md、wiki/formalizations/articulated-body-algorithms.md、wiki/concepts/urdf-robot-description.md、wiki/entities/ssik.md、wiki/queries/pinocchio-quick-start.md

- **触发：** 用户指定 <https://github.com/xiaojie-xue/dynibo>；要求自动合并
- **来源：** [`sources/repos/dynibo.md`](sources/repos/dynibo.md)；顺带交叉 [`sources/repos/pinocchio.md`](sources/repos/pinocchio.md)
- **新建实体：** [`wiki/entities/dynibo.md`](wiki/entities/dynibo.md) — 树状 URDF + Workspace 零分配；FK/Jacobian/DLS-IK/重力/RNEA；含 `## 源码运行时序图`
- **开源核查（步骤 2.5）：** **已开源（MIT）** — GitHub + PyPI `dynibo` 0.1.0 + CMake C/C++；Release `v0.1.0`（2026-08-05）；无独立项目页
- **交叉更新：** [`wiki/entities/pinocchio.md`](wiki/entities/pinocchio.md)、[`wiki/formalizations/articulated-body-algorithms.md`](wiki/formalizations/articulated-body-algorithms.md)、[`wiki/concepts/urdf-robot-description.md`](wiki/concepts/urdf-robot-description.md)、[`wiki/entities/ssik.md`](wiki/entities/ssik.md)、[`wiki/queries/pinocchio-quick-start.md`](wiki/queries/pinocchio-quick-start.md)

## [2026-08-09] ingest | sources/papers/immersive_social_vr_llm_humanoids_arxiv_2607_07430.md — Immersive Social VR+LLM Humanoids（arXiv:2607.07430，NYUAD）；升格 wiki/entities/paper-immersive-social-vr-llm-humanoids.md；注册 nyuad；交叉 wiki/tasks/teleoperation.md、wiki/tasks/loco-manipulation.md、wiki/entities/paper-loco-manip-161-131-open-television.md、wiki/entities/paper-loco-manip-161-012-humanplus.md、wiki/entities/paper-hrl-stack-07-learning_human_to_humanoid_real_time.md、wiki/entities/paper-teleopit.md

- **触发：** 用户指定标题 *Immersive Social Interaction with VR and LLM-Assisted Humanoids*（arXiv）、单位纽约大学阿布扎比分校、作者 Niraj Pudasaini / Geeta Chandra Raju Bethala / Pranav Doma / Anthony Tzes / Yi Fang；要求自动合并
- **来源：** [`sources/papers/immersive_social_vr_llm_humanoids_arxiv_2607_07430.md`](sources/papers/immersive_social_vr_llm_humanoids_arxiv_2607_07430.md)
- **新建实体：** [`wiki/entities/paper-immersive-social-vr-llm-humanoids.md`](wiki/entities/paper-immersive-social-vr-llm-humanoids.md) — AVP + LLM 语音高层 locomotion + VR 腕/指 + 双向音频；H1 新手 80%/70%；`## 源码运行时序图` 不适用
- **开源核查（步骤 2.5）：** **确认未开源** — 无项目页 / 无官方仓；仅第三方 VisionProTeleop / LiveKit / Silero 组件
- **交叉更新：** [`wiki/tasks/teleoperation.md`](wiki/tasks/teleoperation.md)、[`wiki/tasks/loco-manipulation.md`](wiki/tasks/loco-manipulation.md)、[`wiki/entities/paper-loco-manip-161-131-open-television.md`](wiki/entities/paper-loco-manip-161-131-open-television.md)、[`wiki/entities/paper-loco-manip-161-012-humanplus.md`](wiki/entities/paper-loco-manip-161-012-humanplus.md)、[`wiki/entities/paper-hrl-stack-07-learning_human_to_humanoid_real_time.md`](wiki/entities/paper-hrl-stack-07-learning_human_to_humanoid_real_time.md)、[`wiki/entities/paper-teleopit.md`](wiki/entities/paper-teleopit.md)
- **机构：** 注册 [`schema/institutions.json`](schema/institutions.json) → `nyuad`（纽约大学阿布扎比分校）

## [2026-08-09] ingest | sources/courses/cmu_optimal_control_16_745_2025_youtube.md — CMU 16-745 Optimal Control 2025 全播放列表；wiki/entities/cmu-optimal-control-curriculum.md；交叉 optimal-control / lqr-ilqr / MPC / TrajOpt / numerical-optimization-curriculum / motion-control

- **触发：** 用户要求找到 YouTube *Optimal Control 2025* playlist 并 ingest 整个 playlist
- **定位：** Playlist ID `PLZnJoM76RM6IAJfMXd1PgGNXn3dxhkVgI`；官方课站 [optimalcontrol.ri.cmu.edu/lectures](https://optimalcontrol.ri.cmu.edu/lectures/) 各讲 video 链至此列表；主讲 Zachary Manchester（频道标注 MIT Robotic Exploration Lab）
- **工具：** `yt-dlp --flat-playlist` 枚举 24 讲（约 29.3 h）；单视频字幕因 bot 校验未抽，目录以 flat-playlist + 课程站为准
- **来源：** [`sources/courses/cmu_optimal_control_16_745_2025_youtube.md`](sources/courses/cmu_optimal_control_16_745_2025_youtube.md)、[`sources/sites/cmu_optimal_control_16_745.md`](sources/sites/cmu_optimal_control_16_745.md)、[`sources/repos/optimal_control_16_745.md`](sources/repos/optimal_control_16_745.md)
- **新建实体：** [`wiki/entities/cmu-optimal-control-curriculum.md`](wiki/entities/cmu-optimal-control-curriculum.md) — 讲次→wiki 映射与 L3–L4 学习路径
- **开源核查（步骤 2.5）：** **已开源（教材侧）** — 公开录像 + [`Optimal-Control-16-745/lecture-notebooks`](https://github.com/Optimal-Control-16-745/lecture-notebooks)；作业仓多为课内
- **纠正：** 旧归档误将本 playlist 归为 Tedrake；已改 [`sources/courses/mit_underactuated_kalman_lqr.md`](sources/courses/mit_underactuated_kalman_lqr.md)、[`sources/papers/lqr_ilqr_primary_refs.md`](sources/papers/lqr_ilqr_primary_refs.md)
- **交叉更新：** [`wiki/concepts/optimal-control.md`](wiki/concepts/optimal-control.md)、[`wiki/methods/lqr-ilqr.md`](wiki/methods/lqr-ilqr.md)、[`wiki/methods/model-predictive-control.md`](wiki/methods/model-predictive-control.md)、[`wiki/methods/trajectory-optimization.md`](wiki/methods/trajectory-optimization.md)、[`wiki/entities/numerical-optimization-curriculum.md`](wiki/entities/numerical-optimization-curriculum.md)、[`roadmap/motion-control.md`](roadmap/motion-control.md)

## [2026-08-09] ingest | sources/courses/sergey_levine_diffusion_rl_robotics_simons_youtube.md — Levine Simons 扩散连续动作策略；wiki/overview/sergey-levine-diffusion-expressive-policies.md；交叉 diffusion-policy / action-chunking / imitation-learning / diffusion-model / online-vs-offline-rl / lwd

- **触发：** 用户指定 <https://m.youtube.com/watch?v=agi3xLTGyaU>；要求确保 agent-reach 已装并 ingest
- **工具：** Agent Reach v1.5.0（`~/.local/bin/agent-reach`）；配置 `yt-dlp --js-runtimes node` 后 `doctor` 标 YouTube 可用；本机 yt-dlp 仍触发 bot 校验，正文以 Simons talk 页 abstract + oEmbed/Jina 元数据为准
- **来源：** [`sources/courses/sergey_levine_diffusion_rl_robotics_simons_youtube.md`](sources/courses/sergey_levine_diffusion_rl_robotics_simons_youtube.md)、[`sources/sites/simons_sergey_levine_diffusion_rl_robotics_2026.md`](sources/sites/simons_sergey_levine_diffusion_rl_robotics_2026.md)
- **新建 overview：** [`wiki/overview/sergey-levine-diffusion-expressive-policies.md`](wiki/overview/sergey-levine-diffusion-expressive-policies.md) — 扩散/flow → 长 action chunk → IL 与 offline / O2O RL
- **开源核查（步骤 2.5）：** **不适用** — 学术报告录像 + talk 页，无独立项目代码仓
- **交叉更新：** [`wiki/methods/diffusion-policy.md`](wiki/methods/diffusion-policy.md)、[`wiki/methods/action-chunking.md`](wiki/methods/action-chunking.md)、[`wiki/methods/imitation-learning.md`](wiki/methods/imitation-learning.md)、[`wiki/concepts/diffusion-model.md`](wiki/concepts/diffusion-model.md)、[`wiki/comparisons/online-vs-offline-rl.md`](wiki/comparisons/online-vs-offline-rl.md)、[`wiki/methods/lwd.md`](wiki/methods/lwd.md)

## [2026-08-09] ingest | sources/papers/commnav_arxiv_2607_01044.md — CommNav（arXiv:2607.01044）入库；升格 wiki/entities/paper-commnav.md；归档 sources/repos/commnav.md；注册 sapienza；交叉 wiki/entities/habitat-sim.md、wiki/entities/paper-icrowdnav.md、wiki/entities/paper-notebook-learning-social-navigation-from-positive-and-neg.md、wiki/entities/paper-humembr.md、wiki/overview/navigation-slam-autonomy-stack.md、roadmap/depth-navigation.md

- **触发：** 用户指定 <https://arxiv.org/abs/2607.01044>；要求自动合并
- **来源：** [`sources/papers/commnav_arxiv_2607_01044.md`](sources/papers/commnav_arxiv_2607_01044.md)、[`sources/repos/commnav.md`](sources/repos/commnav.md)
- **新建实体：** [`wiki/entities/paper-commnav.md`](wiki/entities/paper-commnav.md) — Habitat 3.0c + COMM；ES +10 pp；口语与结构化接近；`## 源码运行时序图` 不适用
- **开源核查（步骤 2.5）：** **宣称将开源 / 占位仓** — [S4b3/CommNav](https://github.com/S4b3/CommNav) 仅 README（under preparation）；无训练/评测入口
- **交叉更新：** [`wiki/entities/habitat-sim.md`](wiki/entities/habitat-sim.md)、[`wiki/entities/paper-icrowdnav.md`](wiki/entities/paper-icrowdnav.md)、[`wiki/entities/paper-notebook-learning-social-navigation-from-positive-and-neg.md`](wiki/entities/paper-notebook-learning-social-navigation-from-positive-and-neg.md)、[`wiki/entities/paper-humembr.md`](wiki/entities/paper-humembr.md)、[`wiki/overview/navigation-slam-autonomy-stack.md`](wiki/overview/navigation-slam-autonomy-stack.md)、[`roadmap/depth-navigation.md`](roadmap/depth-navigation.md)
- **机构：** 注册 [`schema/institutions.json`](schema/institutions.json) → `sapienza`（罗马第一大学）

## [2026-08-09] ingest | sources/blogs/wechat_zanehub_humanoid_leg_knee_why_not_harmonic.md — 人形膝/腿主承力链为何通常避开谐波；升格 wiki/concepts/humanoid-knee-harmonic-drive-limits.md；交叉 PRS / Actuator 102 / Hardware 101 / 机械布局 / locomotion

- **触发：** 用户指定 <https://mp.weixin.qq.com/s/GowJUzbDjWQMcujtUezLGA>；要求确保 agent-reach 已装并 ingest
- **工具：** Agent Reach v1.5.0 + wechat-article-for-ai（Camoufox；`--no-images`）
- **来源：** [`sources/blogs/wechat_zanehub_humanoid_leg_knee_why_not_harmonic.md`](sources/blogs/wechat_zanehub_humanoid_leg_knee_why_not_harmonic.md)
- **新建概念：** [`wiki/concepts/humanoid-knee-harmonic-drive-limits.md`](wiki/concepts/humanoid-knee-harmonic-drive-limits.md) — 主承力链避开谐波的边界、冲击谱载/柔轮疲劳/远端惯量判据；PRS / 摆线·RV / QDD 三条替代
- **开源核查（步骤 2.5）：** **不适用** — 公众号工程解读，无项目页 / 代码仓
- **交叉更新：** [`wiki/concepts/planetary-roller-screw-humanoid-leg-actuation.md`](wiki/concepts/planetary-roller-screw-humanoid-leg-actuation.md)、[`wiki/overview/humanoid-actuator-102-split-architecture.md`](wiki/overview/humanoid-actuator-102-split-architecture.md)、[`wiki/overview/humanoid-hardware-101-actuation-sensing-chain.md`](wiki/overview/humanoid-hardware-101-actuation-sensing-chain.md)、[`wiki/concepts/humanoid-mechanical-layout-design.md`](wiki/concepts/humanoid-mechanical-layout-design.md)、[`wiki/tasks/locomotion.md`](wiki/tasks/locomotion.md)、[`sources/blogs/wechat_zanezhang_tesla_optimus_leg_planetary_roller_screw.md`](sources/blogs/wechat_zanezhang_tesla_optimus_leg_planetary_roller_screw.md)

## [2026-08-08] lint | wiki/entities/paper-bridgevla-plusplus.md — 全量健康检查：阻塞型 0、信息型 1 → 0

- **触发：** 全量知识库 lint 巡检，要求把信息型预警一并收敛
- **基线：** `python3 scripts/lint_wiki.py` 阻塞型 **0**、信息型 **1**（陈旧声明巡检：[`wiki/entities/paper-bridgevla-plusplus.md`](wiki/entities/paper-bridgevla-plusplus.md) 正文含绝对化措辞「SOTA」，`updated=2026-08-07` 早于同 tag 更新页 [`wiki/concepts/visual-representation-for-policy.md`](wiki/concepts/visual-representation-for-policy.md) `updated=2026-08-08`）
- **复核与改写（陈旧声明 V1）：** 命中句为「实验与评测」表 RLBench 行备注「相对前 SOTA +6.9 pt（项目页）」——该 +6.9 pt 只见于项目页横向对比，`sources/papers/bridgevla_plusplus_arxiv_2608_05042.md` 归档仅记录 RLBench **93.7%** 绝对值，故不是本库可自证的「当前最强」断言；改写为「相对项目页所列先前最好方法 +6.9 pt（项目页口径，2026-08 复核）」——保留数字与出处、把无时限的绝对化措辞降级为**有主体、有口径、有复核时点**的相对表述，同步 `updated` 2026-08-07 → 2026-08-08 记录本次复核
- **其余检查：** 阻塞型 22 项与其余信息型 14 项本轮基线即为 0；Sources 覆盖率 **2117/2117 (100%)**；`eval_search_quality` 通过率 **37/37 (100%)**；`check_export_quality` **12/12**
- **门禁：** `make ci-preflight` 通过（lint 阻塞型 0 + 信息型 0）；派生 `exports/lint-report.md`、`catalog.md`、统计与 `docs/exports/` 随本次提交同步

## [2026-08-08] ingest | sources/repos/hkuds_cli_anything.md — 接入 CLI-Anything / CLI-Hub；wiki/entities/cli-anything.md；交叉更新 hermes-agent / openclaw / agent-reach / freecad-mcp；sources/sites/cli-anything-hub.md；sources/papers/cli_anything_arxiv_2606_03854.md

- **触发：** 用户指定 <https://github.com/HKUDS/CLI-Anything>；要求自动合并
- **来源：** [`sources/repos/hkuds_cli_anything.md`](sources/repos/hkuds_cli_anything.md)、[`sources/sites/cli-anything-hub.md`](sources/sites/cli-anything-hub.md)、[`sources/papers/cli_anything_arxiv_2606_03854.md`](sources/papers/cli_anything_arxiv_2606_03854.md)
- **新建实体：** [`wiki/entities/cli-anything.md`](wiki/entities/cli-anything.md) — 7 阶段 harness 生成 + CLI-Hub + SKILL；与 GUI agent / MCP 对照
- **开源核查（步骤 2.5）：** **已开源（Apache-2.0）** — 生成器、多应用 harness、Hub 与 registry 公开；技术报告 [arXiv:2606.03854](https://arxiv.org/abs/2606.03854)；上游桌面/引擎软件仍须本机安装
- **交叉更新：** [`wiki/entities/hermes-agent.md`](wiki/entities/hermes-agent.md)、[`wiki/entities/openclaw.md`](wiki/entities/openclaw.md)、[`wiki/entities/agent-reach.md`](wiki/entities/agent-reach.md)、[`wiki/entities/freecad-mcp.md`](wiki/entities/freecad-mcp.md)

## [2026-08-08] ingest | sources/papers/slamformer_infinity_arxiv_2608_03429.md — SLAMFormer-∞（arXiv:2608.03429）入库；升格 wiki/entities/paper-slamformer-infinity.md；归档 sources/sites/tsinghua-mars-lab-slamformer-infinity.md、sources/repos/slamformer_infinity.md；交叉 wiki/overview/navigation-slam-autonomy-stack.md、wiki/overview/hub-state-estimation.md、wiki/concepts/state-estimation.md、wiki/entities/paper-glob3r.md、wiki/methods/lingbot-map.md

- **触发：** 用户指定 <https://arxiv.org/pdf/2608.03429>、项目页 <https://tsinghua-mars-lab.github.io/SLAMFormer-Infinity>；要求自动合并
- **来源：** [`sources/papers/slamformer_infinity_arxiv_2608_03429.md`](sources/papers/slamformer_infinity_arxiv_2608_03429.md)、[`sources/sites/tsinghua-mars-lab-slamformer-infinity.md`](sources/sites/tsinghua-mars-lab-slamformer-infinity.md)、[`sources/repos/slamformer_infinity.md`](sources/repos/slamformer_infinity.md)
- **新建实体：** [`wiki/entities/paper-slamformer-infinity.md`](wiki/entities/paper-slamformer-infinity.md) — memory condition + PGGO；KITTI/Waymo 长程；`## 源码运行时序图` 不适用（占位仓）
- **开源核查（步骤 2.5）：** **部分开源（项目页 + 占位仓）** — GitHub `main` 仅 README，`gh-pages` 为站点/demo；无可运行训练/推理。前作 [SLAM-Former](https://github.com/Tsinghua-MARS-Lab/SLAM-Former) 已开源，勿混
- **交叉更新：** [`wiki/overview/navigation-slam-autonomy-stack.md`](wiki/overview/navigation-slam-autonomy-stack.md)、[`wiki/overview/hub-state-estimation.md`](wiki/overview/hub-state-estimation.md)、[`wiki/concepts/state-estimation.md`](wiki/concepts/state-estimation.md)、[`wiki/entities/paper-glob3r.md`](wiki/entities/paper-glob3r.md)、[`wiki/methods/lingbot-map.md`](wiki/methods/lingbot-map.md)

## [2026-08-08] ingest | sources/papers/kdmr_arxiv_2603_09956.md + sources/papers/spark_skeleton_aligned_retargeting_arxiv_2603_11480.md — KDMR / SPARK kinodynamic 重定向入库；升格 wiki/entities/paper-kdmr.md、wiki/entities/paper-spark-skeleton-aligned-retargeting.md；站点 sources/sites/spark-leggedai.md；交叉 wiki/concepts/motion-retargeting.md、wiki/concepts/motion-retargeting-pipeline.md、wiki/overview/hub-motion-retargeting.md、wiki/methods/motion-retargeting-gmr.md、wiki/methods/beyondmimic.md、wiki/entities/paper-notebook-spark.md

- **触发：** 用户指定 KDMR <https://arxiv.org/abs/2603.09956> 与 SPARK <https://arxiv.org/abs/2603.11480> / <https://www.leggedai.com/publication/2026_spark/>；要求自动合并
- **来源：** [`sources/papers/kdmr_arxiv_2603_09956.md`](sources/papers/kdmr_arxiv_2603_09956.md)、[`sources/papers/spark_skeleton_aligned_retargeting_arxiv_2603_11480.md`](sources/papers/spark_skeleton_aligned_retargeting_arxiv_2603_11480.md)、[`sources/sites/spark-leggedai.md`](sources/sites/spark-leggedai.md)
- **新建实体：** [`wiki/entities/paper-kdmr.md`](wiki/entities/paper-kdmr.md)、[`wiki/entities/paper-spark-skeleton-aligned-retargeting.md`](wiki/entities/paper-spark-skeleton-aligned-retargeting.md)
- **开源核查（步骤 2.5）：** KDMR — **宣称正式发表时开源**，无独立项目页 / 官方 GitHub；SPARK — 项目页有 PDF/Video，**未列代码**（未开源）。二者 `## 源码运行时序图` 均标不适用
- **交叉更新：** [`wiki/concepts/motion-retargeting.md`](wiki/concepts/motion-retargeting.md)、[`wiki/concepts/motion-retargeting-pipeline.md`](wiki/concepts/motion-retargeting-pipeline.md)、[`wiki/overview/hub-motion-retargeting.md`](wiki/overview/hub-motion-retargeting.md)、[`wiki/methods/motion-retargeting-gmr.md`](wiki/methods/motion-retargeting-gmr.md)、[`wiki/methods/beyondmimic.md`](wiki/methods/beyondmimic.md)、[`wiki/entities/paper-notebook-spark.md`](wiki/entities/paper-notebook-spark.md)（同名消歧）

## [2026-08-08] ingest | sources/papers/ego4d_arxiv_2110_07058.md — Ego4D（arXiv:2110.07058 / ego4d-data.org）入库；升格 wiki/entities/paper-ego4d.md；归档 sources/sites/ego4d-data-org.md、sources/repos/ego4d.md；交叉 wiki/overview/ego-category-01-data-collection.md、wiki/comparisons/humannet-table1-human-video-corpora.md、wiki/entities/paper-egoverse.md、wiki/entities/rekadaily-10k-dataset.md

- **触发：** 用户指定 <https://arxiv.org/abs/2110.07058>、<https://ego4d-data.org/>；要求自动合并
- **来源：** [`sources/papers/ego4d_arxiv_2110_07058.md`](sources/papers/ego4d_arxiv_2110_07058.md)、[`sources/sites/ego4d-data-org.md`](sources/sites/ego4d-data-org.md)、[`sources/repos/ego4d.md`](sources/repos/ego4d.md)
- **新建实体：** [`wiki/entities/paper-ego4d.md`](wiki/entities/paper-ego4d.md) — ~3,670 h 全球 egocentric 日常视频 + 五大 benchmark；含流程总览与 `## 源码运行时序图`（CLI 拉数）
- **开源核查（步骤 2.5）：** **数据受控开放**（Ego4D license → AWS）；**代码已开源（MIT）** — [facebookresearch/Ego4D](https://github.com/facebookresearch/Ego4D)；挑战基线见 [EGO4D org](https://github.com/EGO4D/)
- **交叉更新：** [`wiki/overview/ego-category-01-data-collection.md`](wiki/overview/ego-category-01-data-collection.md)、[`wiki/comparisons/humannet-table1-human-video-corpora.md`](wiki/comparisons/humannet-table1-human-video-corpora.md)、[`wiki/entities/paper-egoverse.md`](wiki/entities/paper-egoverse.md)、[`wiki/entities/rekadaily-10k-dataset.md`](wiki/entities/rekadaily-10k-dataset.md)、[`sources/papers/humannet_table1_benchmark_corpora.md`](sources/papers/humannet_table1_benchmark_corpora.md)

## [2026-08-08] ingest | sources/repos/codex-security.md — OpenAI Codex Security CLI/SDK 入库；升格 wiki/entities/codex-security.md；文档 sources/sites/openai-codex-security-docs.md；交叉 wiki/concepts/software-security-basics.md、wiki/concepts/container-orchestration-cicd.md、wiki/overview/hub-systems-engineering.md

- **触发：** 用户指定 <https://github.com/openai/codex-security>；要求自动合并
- **来源：** [`sources/repos/codex-security.md`](sources/repos/codex-security.md)、[`sources/sites/openai-codex-security-docs.md`](sources/sites/openai-codex-security-docs.md)
- **新建实体：** [`wiki/entities/codex-security.md`](wiki/entities/codex-security.md) — CLI/SDK、deep scan、SARIF、容器 bulk-scan、CI 严重度门禁；含 `## 源码运行时序图`
- **开源核查（步骤 2.5）：** **已开源（Apache-2.0）** — GitHub + npm `@openai/codex-security`（入库日 0.1.8）；需推理 API / ChatGPT；部分 finding 需 Trusted Access for Cyber
- **交叉更新：** [`wiki/concepts/software-security-basics.md`](wiki/concepts/software-security-basics.md)、[`wiki/concepts/container-orchestration-cicd.md`](wiki/concepts/container-orchestration-cicd.md)、[`wiki/overview/hub-systems-engineering.md`](wiki/overview/hub-systems-engineering.md)、[`sources/sites/systems_engineering_deploy_obs_security_primary_refs.md`](sources/sites/systems_engineering_deploy_obs_security_primary_refs.md)

## [2026-08-08] ingest | sources/sites/embodied-interview-qa-github-io.md — 具身智能高频面试题库（WinstonJQ）入库；升格 wiki/entities/embodied-interview-qa.md；仓 sources/repos/embodied-interview-qa.md；交叉 VLA/RL/IL/Sim2Real/WBC/深度学习基础与 Lumina/qqfly 指南

- **触发：** 用户指定 <https://winstonjq.github.io/embodied-interview-qa/index.html>；要求自动合并
- **来源：** [`sources/sites/embodied-interview-qa-github-io.md`](sources/sites/embodied-interview-qa-github-io.md)、[`sources/repos/embodied-interview-qa.md`](sources/repos/embodied-interview-qa.md)
- **新建实体：** [`wiki/entities/embodied-interview-qa.md`](wiki/entities/embodied-interview-qa.md) — 八卷 ↔ 本库方法/概念/路线映射；面试速查与 wiki 深读分工
- **开源核查（步骤 2.5）：** **已开源（MIT）** — Pages + `docs/interviews/*.md`；非算法仓，`## 源码运行时序图` 不适用
- **要点：** 公开面经频次合并；L1–L3；答案 ≤350 字；约 425 题（README）/ 入库日 Markdown `<summary>` ≈438
- **交叉更新：** [`wiki/methods/vla.md`](wiki/methods/vla.md)、[`wiki/methods/reinforcement-learning.md`](wiki/methods/reinforcement-learning.md)、[`wiki/methods/imitation-learning.md`](wiki/methods/imitation-learning.md)、[`wiki/concepts/sim2real.md`](wiki/concepts/sim2real.md)、[`wiki/concepts/whole-body-control.md`](wiki/concepts/whole-body-control.md)、[`wiki/concepts/deep-learning-foundations.md`](wiki/concepts/deep-learning-foundations.md)、[`wiki/entities/lumina-embodied.md`](wiki/entities/lumina-embodied.md)、[`wiki/entities/learn-robotics-qqfly-guide.md`](wiki/entities/learn-robotics-qqfly-guide.md)

## [2026-08-08] ingest | sources/papers/kilvo_arxiv_2608_05647.md — KILVO 人形多传感器 ESIKF 里程计（代码待开放）；升格 wiki/entities/paper-kilvo.md；仓 sources/repos/kilvo.md；交叉 wiki/methods/lidar-odometry-fusion.md、wiki/comparisons/lidar-slam-lio-vio-selection.md

- **触发：** 用户指定 GitHub <https://github.com/JixinGao/KILVO> + 论文 <https://arxiv.org/abs/2608.05647>；人形多传感器里程计，代码待开放；要求自动合并
- **来源：** [`sources/papers/kilvo_arxiv_2608_05647.md`](sources/papers/kilvo_arxiv_2608_05647.md)、[`sources/repos/kilvo.md`](sources/repos/kilvo.md)
- **新建实体：** [`wiki/entities/paper-kilvo.md`](wiki/entities/paper-kilvo.md)
- **开源核查（步骤 2.5）：** **代码待开放 / 占位仓** — README「available soon」；根目录无可运行实现；`## 源码运行时序图` 写不适用
- **方法要点：** 异步–顺序混合 ESIKF；接触估计无额外传感器；模态失效自适应；真机端到端均值 **0.0145 m**、输出 **1 kHz**
- **交叉更新：** [`wiki/methods/lidar-odometry-fusion.md`](wiki/methods/lidar-odometry-fusion.md)、[`wiki/comparisons/lidar-slam-lio-vio-selection.md`](wiki/comparisons/lidar-slam-lio-vio-selection.md)

## [2026-08-08] ingest | sources/papers/dypes_vla_arxiv_2608_06374.md — DyPES-VLA 跨本体动力学先验 + MoE 动作头；升格 wiki/entities/paper-dypes-vla.md；项目页 sources/sites/dypes-vla-github-io.md；注册 coco-matrix；交叉 wiki/methods/vla.md

- **触发：** 用户指定 <https://arxiv.org/abs/2608.06374>；跨本体动力学 VLA；要求自动合并
- **来源：** [`sources/papers/dypes_vla_arxiv_2608_06374.md`](sources/papers/dypes_vla_arxiv_2608_06374.md)、[`sources/sites/dypes-vla-github-io.md`](sources/sites/dypes-vla-github-io.md)
- **新建实体：** [`wiki/entities/paper-dypes-vla.md`](wiki/entities/paper-dypes-vla.md)
- **开源核查（步骤 2.5）：** **宣称将开源 / coming soon** — 项目页 Code 按钮禁用；`## 源码运行时序图` 写不适用
- **方法要点：** 未来预测学共享 query 动力学先验 + 本体特化 MoE 原生动作；LIBERO **98.0%** / RoboCasa-GR1 **59.25%** / RoboTwin **89.02%**；真机三本体 **75.6%**
- **机构注册：** [`schema/institutions.json`](schema/institutions.json) 新增 `coco-matrix`（可可矩阵）
- **交叉更新：** [`wiki/methods/vla.md`](wiki/methods/vla.md)

## [2026-08-08] ingest | sources/papers/omega0_arxiv_2608_06375.md — ω-0 潜空间 foresight 人形并发 loco-manipulation WAM；升格 wiki/entities/paper-omega-0.md；项目页 sources/sites/omega0-github-io.md；交叉 wiki/tasks/loco-manipulation.md、wiki/concepts/world-action-models.md、wiki/entities/paper-motionwam-humanoid-loco-manipulation-wam.md

- **触发：** 用户指定 <https://arxiv.org/abs/2608.06375>；人形全身移动操作世界模型；要求自动合并
- **来源：** [`sources/papers/omega0_arxiv_2608_06375.md`](sources/papers/omega0_arxiv_2608_06375.md)、[`sources/sites/omega0-github-io.md`](sources/sites/omega0-github-io.md)
- **新建实体：** [`wiki/entities/paper-omega-0.md`](wiki/entities/paper-omega-0.md)
- **开源核查（步骤 2.5）：** **宣称将开源 / WIP** — 项目页 Code & Dataset 按钮 WIP；`## 源码运行时序图` 写不适用
- **方法要点：** 潜空间未来观测 embedding + 扩散全身动作 latent + SONIC；ω-HOME 40h+；G1 11 任务 Omni **SR 81.8%**
- **交叉更新：** [`wiki/tasks/loco-manipulation.md`](wiki/tasks/loco-manipulation.md)、[`wiki/concepts/world-action-models.md`](wiki/concepts/world-action-models.md)、[`wiki/entities/paper-motionwam-humanoid-loco-manipulation-wam.md`](wiki/entities/paper-motionwam-humanoid-loco-manipulation-wam.md)

## [2026-08-08] ingest | sources/blogs/wechat_shenlan_humanoid_rl_policy_training_system.md — 深蓝具身智能《人形机器人运动控制：强化学习与策略训练体系详解》入库；升格 wiki/overview/humanoid-rl-policy-training-five-modules.md；交叉 wiki/concepts/embodied-rl-minimal-closed-loop.md、wiki/methods/reinforcement-learning.md、wiki/methods/ppo.md、wiki/concepts/privileged-training.md、wiki/queries/humanoid-rl-cookbook.md、wiki/comparisons/wbc-vs-rl.md、wiki/overview/humanoid-rl-motion-control-body-system-stack.md、wiki/overview/shenlan-embodied-ai-fundamentals-series.md

- **触发：** 用户指定 <https://mp.weixin.qq.com/s/mxesB0pGI_NLSkSf-cZYug>；要求确认 Agent Reach 可用并自动合并
- **抓取：** Agent Reach v1.5.0 + `wechat-article-for-ai`（Camoufox；`--no-images`）；原始落盘 [`sources/raw/wechat_shenlan_humanoid_rl_policy_training_2026-08-08.md`](sources/raw/wechat_shenlan_humanoid_rl_policy_training_2026-08-08.md)
- **来源：** [`sources/blogs/wechat_shenlan_humanoid_rl_policy_training_system.md`](sources/blogs/wechat_shenlan_humanoid_rl_policy_training_system.md)（深蓝具身智能；2026-08-08）
- **主升格：** [`wiki/overview/humanoid-rl-policy-training-five-modules.md`](wiki/overview/humanoid-rl-policy-training-five-modules.md) — MDP → Actor-Critic → PPO → 多维奖励 → Teacher-Student；与 WBC/MPC 混合落地
- **交叉更新：** [`wiki/concepts/embodied-rl-minimal-closed-loop.md`](wiki/concepts/embodied-rl-minimal-closed-loop.md)、[`wiki/methods/reinforcement-learning.md`](wiki/methods/reinforcement-learning.md)、[`wiki/methods/ppo.md`](wiki/methods/ppo.md)、[`wiki/concepts/privileged-training.md`](wiki/concepts/privileged-training.md)、[`wiki/queries/humanoid-rl-cookbook.md`](wiki/queries/humanoid-rl-cookbook.md)、[`wiki/comparisons/wbc-vs-rl.md`](wiki/comparisons/wbc-vs-rl.md)、[`wiki/overview/humanoid-rl-motion-control-body-system-stack.md`](wiki/overview/humanoid-rl-motion-control-body-system-stack.md)、[`wiki/overview/shenlan-embodied-ai-fundamentals-series.md`](wiki/overview/shenlan-embodied-ai-fundamentals-series.md)
- **项目页核查：** 无独立项目页 / 代码仓（公众号科普）；跳过步骤 2.5 源码开放项

## [2026-08-08] ingest | sources/repos/dexmal_opendm.md + sources/blogs/dexmal_dm05.md — 接入 OpenDM/DM0.5 开源栈；更新 wiki/entities/dexmal-dm05.md；互链 wiki/methods/vla.md、wiki/entities/dexmal-dw05.md、wiki/tasks/manipulation.md

- **触发：** 用户指定开源仓 <https://github.com/dexmal/opendm> 与技术博客 <https://www.dexmal.com/blog/dm0.5>；要求自动合并
- **来源：** [`sources/repos/dexmal_opendm.md`](sources/repos/dexmal_opendm.md)、修订 [`sources/blogs/dexmal_dm05.md`](sources/blogs/dexmal_dm05.md)
- **主升格 / 更新：** [`wiki/entities/dexmal-dm05.md`](wiki/entities/dexmal-dm05.md) — 补开源状态、DM05 权重表、工程实践、`## 源码运行时序图`
- **开源核查（步骤 2.5）：** **已开源** — Apache-2.0；HF/ModelScope **DM05** 及 LIBERO / RobotWin2 / SO101 / VLA-Arena / Table30v2；`script/dm05_launcher.sh` 统一 train/inference
- **交叉更新：** [`wiki/methods/vla.md`](wiki/methods/vla.md)、[`wiki/entities/dexmal-dw05.md`](wiki/entities/dexmal-dw05.md)、[`wiki/tasks/manipulation.md`](wiki/tasks/manipulation.md)、[`sources/repos/dexmal_opendw.md`](sources/repos/dexmal_opendw.md)

## [2026-08-08] ingest | sources/papers/dash_opsd_arxiv_2608_06243.md — DASH（CASIA / EverMind / 盛大等，arXiv:2608.06243）OPSD 分歧自适应监督视界入库；升格 wiki/entities/paper-dash-opsd.md；归档 sources/repos/dash-opsd.md；交叉 wiki/entities/paper-shenlan-wm-14-rlvr-world.md、wiki/concepts/ai-auto-research.md、wiki/entities/paper-harnessbank.md、wiki/entities/paper-skillcorpus.md；已开源 DBtxy/DASH-OPSD

- **触发：** 用户指定 <https://arxiv.org/pdf/2608.06243v1>；与 HarnessBank / SkillCorpus 一并入库并要求自动合并
- **来源：** [`sources/papers/dash_opsd_arxiv_2608_06243.md`](sources/papers/dash_opsd_arxiv_2608_06243.md)、[`sources/repos/dash-opsd.md`](sources/repos/dash-opsd.md)
- **新建实体：** [`wiki/entities/paper-dash-opsd.md`](wiki/entities/paper-dash-opsd.md)
- **开源核查（步骤 2.5）：** **已开源** — <https://github.com/DBtxy/DASH-OPSD>（`opsd_train.py` / `scripts/run_dash_*.sh`）+ HF LoRA；`## 源码运行时序图` 已写
- **方法要点：** 局部 KL 相对序列均值 → 自适应传播门 → 反向多步聚合；相对匹配 OPSD macro +3.20 / +1.40 / +1.60（1.7B/4B/8B）
- **机构注册：** [`schema/institutions.json`](schema/institutions.json) 新增 `evermind` / `shanda` / `wuhan-ai-research`
- **交叉更新：** [`wiki/entities/paper-shenlan-wm-14-rlvr-world.md`](wiki/entities/paper-shenlan-wm-14-rlvr-world.md)、[`wiki/concepts/ai-auto-research.md`](wiki/concepts/ai-auto-research.md)、[`wiki/entities/paper-harnessbank.md`](wiki/entities/paper-harnessbank.md)、[`wiki/entities/paper-skillcorpus.md`](wiki/entities/paper-skillcorpus.md)

## [2026-08-08] ingest | sources/papers/skillcorpus_arxiv_2607_15557.md — SkillCorpus（EverMind / 盛大 / 北大，arXiv:2607.15557）社区 SKILL.md 策展语料入库；升格 wiki/entities/paper-skillcorpus.md；交叉 wiki/entities/openclaw.md、wiki/entities/darwin-skill.md、wiki/entities/mattpocock-skills.md、wiki/entities/hermes-agent.md、wiki/entities/paper-harnessbank.md；语料与代码 acceptance 后开源

- **触发：** 用户指定 <https://arxiv.org/abs/2607.15557>；要求自动合并
- **来源：** [`sources/papers/skillcorpus_arxiv_2607_15557.md`](sources/papers/skillcorpus_arxiv_2607_15557.md)
- **新建实体：** [`wiki/entities/paper-skillcorpus.md`](wiki/entities/paper-skillcorpus.md)
- **开源核查（步骤 2.5）：** **宣称将开源 / 尚未发布** — acceptance 后释放 96,401 语料 + 检索栈 + 代码；`## 源码运行时序图` 写不适用
- **方法要点：** ~821k→96,401；三面质量 + OSI 许可；SkillsBench 池化 **+7.5 pp**；覆盖边界与 harness 边界
- **交叉更新：** [`wiki/entities/openclaw.md`](wiki/entities/openclaw.md)、[`wiki/entities/darwin-skill.md`](wiki/entities/darwin-skill.md)、[`wiki/entities/mattpocock-skills.md`](wiki/entities/mattpocock-skills.md)、[`wiki/entities/hermes-agent.md`](wiki/entities/hermes-agent.md)、[`wiki/entities/paper-harnessbank.md`](wiki/entities/paper-harnessbank.md)

## [2026-08-08] ingest | sources/papers/harnessbank_arxiv_2607_13683.md — HarnessBank（EverMind / 盛大，arXiv:2607.13683）可信 agent-harness 自进化入库；升格 wiki/entities/paper-harnessbank.md；交叉 wiki/entities/openclaw.md、wiki/entities/darwin-skill.md、wiki/concepts/ai-auto-research.md、wiki/entities/paper-skillcorpus.md；代码 acceptance 后开源

- **触发：** 用户指定 <https://arxiv.org/abs/2607.13683>；要求自动合并
- **来源：** [`sources/papers/harnessbank_arxiv_2607_13683.md`](sources/papers/harnessbank_arxiv_2607_13683.md)
- **新建实体：** [`wiki/entities/paper-harnessbank.md`](wiki/entities/paper-harnessbank.md)
- **开源核查（步骤 2.5）：** **宣称将开源 / 尚未发布** — acceptance 后公开；`## 源码运行时序图` 写不适用
- **方法要点：** Harness Gene Bank + Gated Harness Screening；七基准 Test Pass@1 **+5.1%–15.4%**；跨模型匹配律
- **交叉更新：** [`wiki/entities/openclaw.md`](wiki/entities/openclaw.md)、[`wiki/entities/darwin-skill.md`](wiki/entities/darwin-skill.md)、[`wiki/concepts/ai-auto-research.md`](wiki/concepts/ai-auto-research.md)、[`wiki/entities/paper-skillcorpus.md`](wiki/entities/paper-skillcorpus.md)

## [2026-08-08] ingest | sources/papers/pfm_hr_arxiv_2608_03227.md — PFM-HR（HKUST-GZ / 诺亦腾 / 清华 SIGS / Google，arXiv:2608.03227）姿态 Flow Matching 先验入库；升格 wiki/entities/paper-pfm-hr.md；交叉 wiki/methods/beyondmimic.md、wiki/methods/add.md、wiki/methods/smp.md、wiki/entities/paper-notebook-pdf-hr.md、wiki/entities/mimickit.md、wiki/queries/humanoid-motion-tracking-method-selection.md、wiki/comparisons/amp-add-smp-motion-prior-variants.md；归档 sources/sites/pfm-hr-web.md、sources/repos/pfm-hr.md

- **触发：** 用户指定标题 *PFM-HR: Pose Flow Matching for Humanoid Robots*、项目页 <https://gaoyukang33.github.io/PFM-HR.web/>、机构香港科技大学（广州）/ 诺亦腾机器人 / 清华大学深圳国际研究生院 / Google；要求自动合并
- **来源：** [`sources/papers/pfm_hr_arxiv_2608_03227.md`](sources/papers/pfm_hr_arxiv_2608_03227.md)、[`sources/sites/pfm-hr-web.md`](sources/sites/pfm-hr-web.md)、[`sources/repos/pfm-hr.md`](sources/repos/pfm-hr.md)
- **新建实体：** [`wiki/entities/paper-pfm-hr.md`](wiki/entities/paper-pfm-hr.md)
- **开源核查（步骤 2.5）：** **宣称开源 / 实现待发布** — 项目页 Code → <https://github.com/gaoyukang33/PFM-HR>，tip 仅 MIT + README「Coming Soon」；`## 源码运行时序图` 写不适用
- **方法要点：** 无序姿态 Flow Matching 先验；Pose Geometry Score（PGS / JVP）调制跟踪奖励；冻结挂 ADD / BeyondMimic；高动态样本效率↑
- **交叉更新：** [`wiki/methods/beyondmimic.md`](wiki/methods/beyondmimic.md)、[`wiki/methods/add.md`](wiki/methods/add.md)、[`wiki/methods/smp.md`](wiki/methods/smp.md)、[`wiki/entities/paper-notebook-pdf-hr.md`](wiki/entities/paper-notebook-pdf-hr.md)、[`wiki/entities/mimickit.md`](wiki/entities/mimickit.md)、[`wiki/queries/humanoid-motion-tracking-method-selection.md`](wiki/queries/humanoid-motion-tracking-method-selection.md)、[`wiki/comparisons/amp-add-smp-motion-prior-variants.md`](wiki/comparisons/amp-add-smp-motion-prior-variants.md)

## [2026-08-07] lint | wiki/queries/humanoid-motion-tracking-method-selection.md + wiki/methods/dsms-contact-implicit-multiple-shooting.md + scripts/lint_wiki.py — 全量健康检查：阻塞型 0、信息型 2 → 0

- **触发：** 全量知识库 lint 巡检，要求把信息型预警一并收敛
- **基线：** `python3 scripts/lint_wiki.py` 阻塞型 **0**、信息型 **2**（高频引用 methods 缺 queries/comparisons 落地 1 + 缺页概念候选 1）
- **内容补链（方法-Query 闭环）：** [`wiki/methods/dsms-contact-implicit-multiple-shooting.md`](wiki/methods/dsms-contact-implicit-multiple-shooting.md) 被 4 页引用却无操作落地 → [`wiki/queries/humanoid-motion-tracking-method-selection.md`](wiki/queries/humanoid-motion-tracking-method-selection.md) §4 新增「参考层动力学可行化」段：**失败不在跟踪策略而在参考本身动力学不可行**（爬行/搬箱等接触丰富片段违反作动极限、接触时刻表对不上）时，先用 DSMS 把可微仿真器离散转移嵌进多重打靶 NLP（接触隐式：无 contact force 决策变量 / 无互补松弛 / 无预设时刻表）产出动力学可行参考，再喂下游 mjlab PPO imitation；选型轴 one-shot（周期步态）vs receding-horizon MPC（高动态拼接），并声明与 [GMR](wiki/methods/motion-retargeting-gmr.md) 等运动学前端**串联而非替代**、采样式对照 [DynaRetarget / SBTO](wiki/methods/dynaretarget-sbto-motion-retargeting.md)；同步补 frontmatter `sources` + 参考来源 + 关联页面，DSMS 页回链该 Query
- **误报收敛（缺页概念巡检 V1）：** `vlm` 入 `MISSING_CONCEPT_COVERED_ELSEWHERE` — 6 处加粗/反引号引用中 5 处是「VLM 作底座/规划器」的组件标签，canonical 定义与 I/O 边界已由 [`wiki/comparisons/vlm-vln-vla-vlx-world-model-taxonomy.md`](wiki/comparisons/vlm-vln-vla-vlx-world-model-taxonomy.md) 承担（库内唯一 VLM 家族来源 `sources/blogs/wechat_shenlan_five_embodied_model_taxonomy.md` 即入库于此），机制侧另有 [`wiki/concepts/vision-language-feature-fusion.md`](wiki/concepts/vision-language-feature-fusion.md)、底座维度见 [`wiki/methods/vla.md`](wiki/methods/vla.md)、选型闭环见 [`wiki/overview/hub-embodied-foundation-model.md`](wiki/overview/hub-embodied-foundation-model.md)；再建 concepts/vlm.md 只会与分类学对比页重复同一来源，故按「已有恰当归属」登记而非新建
- **门禁：** `make ci-preflight` 通过（lint 阻塞型 0 + 信息型 0）；`pytest` / `ruff check` / `ruff format --check` / `mypy scripts` 见下方验证记录

## [2026-08-07] ingest | sources/papers/agile_arxiv_2603_20147.md — AGILE（NVIDIA，arXiv:2603.20147）人形 loco-manipulation RL 工作流入库；升格 wiki/entities/paper-agile-humanoid-loco-manipulation.md；交叉 wiki/entities/isaac-lab.md、wiki/tasks/loco-manipulation.md、wiki/concepts/sim2real.md、wiki/entities/unitree-g1.md、wiki/methods/beyondmimic.md；归档 sources/repos/wbc_agile.md、sources/sites/wbc-agile-docs.md

- **触发：** 用户指定标题 *AGILE: A Comprehensive Workflow for Humanoid Loco-Manipulation Learning*（arXiv）、单位 NVIDIA、作者 Huihua Zhao / Rafael Cathomen 等；要求自动合并
- **来源：** [`sources/papers/agile_arxiv_2603_20147.md`](sources/papers/agile_arxiv_2603_20147.md)、[`sources/repos/wbc_agile.md`](sources/repos/wbc_agile.md)、[`sources/sites/wbc-agile-docs.md`](sources/sites/wbc-agile-docs.md)
- **新建实体：** [`wiki/entities/paper-agile-humanoid-loco-manipulation.md`](wiki/entities/paper-agile-humanoid-loco-manipulation.md)
- **开源核查（步骤 2.5）：** **已开源** — 文档站与论文指向 <https://github.com/nvidia-isaac/WBC-AGILE>（`scripts/train.py` / `eval.py`；Isaac Lab v2.3.2）；完整真机驱动管线论文称将另行发布；`## 源码运行时序图` 已写
- **方法要点：** Prepare→Train→Evaluate→Deploy；YAML I/O 描述符；L2C2 / harness / 奖励归一化等可开关增强；G1+T1 五技能；解耦下肢 + GR00T VLA 仿真 90%
- **交叉更新：** [`wiki/entities/isaac-lab.md`](wiki/entities/isaac-lab.md)、[`wiki/tasks/loco-manipulation.md`](wiki/tasks/loco-manipulation.md)、[`wiki/concepts/sim2real.md`](wiki/concepts/sim2real.md)、[`wiki/entities/unitree-g1.md`](wiki/entities/unitree-g1.md)、[`wiki/methods/beyondmimic.md`](wiki/methods/beyondmimic.md)

## [2026-08-07] ingest | sources/sites/humantouch-xsparkai.md — HumanTouch（Xspark SparkLAB）人手全掌触觉采集入库；升格 wiki/entities/humantouch.md；交叉 tactile-sensing / visuo-tactile-fusion / contact-rich-manipulation / hub-tactile / dexterous-data-collection-guide / data-gloves-vs-vision-teleop / manipulation / OSMO；数据待 HF、代码未列

- **触发：** 用户指定项目页 <https://xsparkai.com/sparklab/humantouch/>；团队 Xspark AI · SparkLAB；负责人 Chuqiao Lyu；核心 Chenze Yu / Eric J Chen / Wenxuan Zhu；通讯 Wenbo Ding / Tianxing Chen / Qi Xiong；要求自动合并
- **项目页：** [`sources/sites/humantouch-xsparkai.md`](sources/sites/humantouch-xsparkai.md)
- **主升格：** [`wiki/entities/humantouch.md`](wiki/entities/humantouch.md) — 压阻手套 ~360 pts/hand + MANUS EMF + 头/腕 RGB；初版 ~100 h / 13469 ep；DcSNR；开源=数据待发、代码未列；源码运行时序图不适用
- **交叉：** [`wiki/concepts/tactile-sensing.md`](wiki/concepts/tactile-sensing.md)、[`wiki/concepts/visuo-tactile-fusion.md`](wiki/concepts/visuo-tactile-fusion.md)、[`wiki/concepts/contact-rich-manipulation.md`](wiki/concepts/contact-rich-manipulation.md)、[`wiki/overview/hub-tactile.md`](wiki/overview/hub-tactile.md)、[`wiki/queries/dexterous-data-collection-guide.md`](wiki/queries/dexterous-data-collection-guide.md)、[`wiki/comparisons/data-gloves-vs-vision-teleop.md`](wiki/comparisons/data-gloves-vs-vision-teleop.md)、[`wiki/tasks/manipulation.md`](wiki/tasks/manipulation.md)、[`wiki/entities/paper-notebook-osmo-open-source-tactile-glove-for-human-to-robo.md`](wiki/entities/paper-notebook-osmo-open-source-tactile-glove-for-human-to-robo.md)
- **机构：** [`schema/institutions.json`](schema/institutions.json) 注册 `xspark-ai` → 星火人工智能（Xspark AI）

## [2026-08-07] ingest | sources/papers/joyai_ra_05_arxiv_2608_05674.md — JoyAI-RA 0.5（arXiv:2608.05674）双动作对齐 VLWA 入库；升格 wiki/entities/paper-joyai-ra-05.md；互链 wiki/methods/vla.md、wiki/concepts/world-action-models.md、wiki/tasks/manipulation.md、wiki/entities/paper-ego-02-egolive.md、wiki/overview/ego-category-02-human-to-robot.md、wiki/entities/paper-data-pyramid-embodied-manipulation.md、wiki/entities/paper-internvla-a15-unified-vla.md；sources/sites/joyai-ra-05-github-io.md

- **触发：** 用户指定 <https://arxiv.org/abs/2608.05674>、<https://joyai-ra-05.github.io/>；要求自动合并
- **来源：** [`sources/papers/joyai_ra_05_arxiv_2608_05674.md`](sources/papers/joyai_ra_05_arxiv_2608_05674.md)、[`sources/sites/joyai-ra-05-github-io.md`](sources/sites/joyai-ra-05-github-io.md)
- **新建实体：** [`wiki/entities/paper-joyai-ra-05.md`](wiki/entities/paper-joyai-ra-05.md)
- **开源核查（步骤 2.5）：** **确认未开源** — 项目页仅 arXiv，未列 GitHub/HF/权重；`## 源码运行时序图` 写不适用
- **方法要点：** VLWA = VLM + LAC-WM + Flow Expert；隐式 latent-action + 显式 130-D；内–外环 RL；AgiBot G1 seen **92.0** / unseen **75.5**；人视频缩放未见饱和
- **交叉更新：** [`wiki/methods/vla.md`](wiki/methods/vla.md)、[`wiki/concepts/world-action-models.md`](wiki/concepts/world-action-models.md)、[`wiki/tasks/manipulation.md`](wiki/tasks/manipulation.md)、[`wiki/entities/paper-ego-02-egolive.md`](wiki/entities/paper-ego-02-egolive.md)、[`wiki/overview/ego-category-02-human-to-robot.md`](wiki/overview/ego-category-02-human-to-robot.md)、[`wiki/entities/paper-data-pyramid-embodied-manipulation.md`](wiki/entities/paper-data-pyramid-embodied-manipulation.md)、[`wiki/entities/paper-internvla-a15-unified-vla.md`](wiki/entities/paper-internvla-a15-unified-vla.md)

## [2026-08-07] ingest | sources/blogs/macrodata_egocentric_video_3d_hand_actions.md — Macrodata egocentric→度量手轨迹博客入库；升格 wiki/methods/macrodata-egocentric-hand-action.md；互链 wiki/methods/wilor.md、wiki/methods/egoscale.md、wiki/methods/auto-labeling-pipelines.md、wiki/entities/perceptron-egocentric.md、wiki/entities/paper-vidihand.md、wiki/overview/ego-category-01-data-collection.md、wiki/overview/ego-category-02-human-to-robot.md、wiki/queries/dexterous-manipulation-data-pipeline.md；sources/sites/macrodata-co.md、sources/repos/hawor.md

- **触发：** 用户指定 <https://macrodata.co/blog/turning-egocentric-video-into-3d-hand-actions>；要求确保 agent-reach 已装并自动合并
- **来源：** [`sources/blogs/macrodata_egocentric_video_3d_hand_actions.md`](sources/blogs/macrodata_egocentric_video_3d_hand_actions.md)、[`sources/sites/macrodata-co.md`](sources/sites/macrodata-co.md)、[`sources/repos/hawor.md`](sources/repos/hawor.md)
- **新建方法页：** [`wiki/methods/macrodata-egocentric-hand-action.md`](wiki/methods/macrodata-egocentric-hand-action.md)
- **开源核查（步骤 2.5）：** **部分可复现 / 产品未开源** — 博客配方为 WiLoR+HaWoR+VGGT-Omega 开源组件；Macrodata 专有检测/编排 **确认未开源**（Contact / 免费样例）
- **要点：** HOT3D Action MPJPE **52.04 mm**、覆盖 **81.23%**、**15.53 FPS@H100**；相对 HaWoR 误差 −12%、吞吐 3.34→15.53 FPS；误差主导在相机系腕深
- **机构注册：** [`schema/institutions.json`](schema/institutions.json) 新增 `macrodata` → 宏数据实验室（Macrodata Labs）
- **交叉更新：** [`wiki/methods/wilor.md`](wiki/methods/wilor.md)、[`wiki/methods/egoscale.md`](wiki/methods/egoscale.md)、[`wiki/methods/auto-labeling-pipelines.md`](wiki/methods/auto-labeling-pipelines.md)、[`wiki/entities/perceptron-egocentric.md`](wiki/entities/perceptron-egocentric.md)、[`wiki/entities/paper-vidihand.md`](wiki/entities/paper-vidihand.md)、[`wiki/overview/ego-category-01-data-collection.md`](wiki/overview/ego-category-01-data-collection.md)、[`wiki/overview/ego-category-02-human-to-robot.md`](wiki/overview/ego-category-02-human-to-robot.md)、[`wiki/queries/dexterous-manipulation-data-pipeline.md`](wiki/queries/dexterous-manipulation-data-pipeline.md)


## [2026-08-07] ingest | sources/papers/shooting_for_contact_arxiv_2608_03116.md — Shooting for Contact / DSMS（Caltech/DePaul，arXiv:2608.03116）：接触隐式多重打靶动力学重定向；升格 wiki/entities/paper-shooting-for-contact.md 与 wiki/methods/dsms-contact-implicit-multiple-shooting.md；交叉 wiki/concepts/motion-retargeting.md、wiki/overview/hub-motion-retargeting.md、wiki/methods/dynaretarget-sbto-motion-retargeting.md、wiki/entities/paper-hrl-stack-03-omniretarget.md；注册 depaul；代码 sources/repos/shooting-for-contact.md + sources/sites/shooting-for-contact-github-io.md

- **触发：** 用户指定标题 *Shooting for Contact*、项目页 <https://shooting-for-contact.github.io/>；机构 Caltech / DePaul；要求自动合并
- **来源：** [`sources/papers/shooting_for_contact_arxiv_2608_03116.md`](sources/papers/shooting_for_contact_arxiv_2608_03116.md)、[`sources/sites/shooting-for-contact-github-io.md`](sources/sites/shooting-for-contact-github-io.md)、[`sources/repos/shooting-for-contact.md`](sources/repos/shooting-for-contact.md)
- **新建实体 / 方法：** [`wiki/entities/paper-shooting-for-contact.md`](wiki/entities/paper-shooting-for-contact.md)、[`wiki/methods/dsms-contact-implicit-multiple-shooting.md`](wiki/methods/dsms-contact-implicit-multiple-shooting.md)
- **开源核查（步骤 2.5）：** **已开源** — 项目页 Code → <https://github.com/sesteban951/shooting-for-contact>（MuJoCo+IPOPT DSMS / MPC 与 G1·Go2 示例）；RL/真机栈未随仓；`## 源码运行时序图` 已写
- **方法要点：** 接触隐式多重打靶；Table I SRB→DSMS 落地 **100%**；Table II super-hero backflip 与 DynaRetarget 同档 **98.7%**、远超 OmniRetarget **9.3%**；G1 零样本爬行与 180° 跳转
- **机构注册：** [`schema/institutions.json`](schema/institutions.json) 新增 `depaul` → 德保罗大学（DePaul University）
- **交叉更新：** [`wiki/concepts/motion-retargeting.md`](wiki/concepts/motion-retargeting.md)、[`wiki/overview/hub-motion-retargeting.md`](wiki/overview/hub-motion-retargeting.md)、[`wiki/methods/dynaretarget-sbto-motion-retargeting.md`](wiki/methods/dynaretarget-sbto-motion-retargeting.md)、[`wiki/entities/paper-hrl-stack-03-omniretarget.md`](wiki/entities/paper-hrl-stack-03-omniretarget.md)

## [2026-08-07] ingest | sources/papers/rtcf_arxiv_2608_04527.md — RTCF（arXiv:2608.04527）免训练 VLA 动作纠偏入库；升格 wiki/entities/paper-rtcf.md；互链 wiki/methods/vla.md、wiki/entities/paper-dynawm-vla-online-correction.md、wiki/entities/paper-dreamsteer-vla-deployment-steering.md、wiki/entities/paper-bridgevla-plusplus.md、wiki/tasks/manipulation.md

- **触发：** 用户指定 <https://arxiv.org/abs/2608.04527>；要求与 DreamWAM / BridgeVLA++ 一并入库并自动合并
- **来源：** [`sources/papers/rtcf_arxiv_2608_04527.md`](sources/papers/rtcf_arxiv_2608_04527.md)
- **新建实体：** [`wiki/entities/paper-rtcf.md`](wiki/entities/paper-rtcf.md)
- **开源核查（步骤 2.5）：** **确认未开源** — 无项目页 / GitHub / 权重；`## 源码运行时序图` 写不适用
- **方法要点：** PMA 历史对齐成功轨迹 + 低频运动残差纠偏冻结 PI-FAST；LIBERO All **86.4→88.4**、Long **61.6→68.6**；+11 ms CPU
- **交叉更新：** [`wiki/methods/vla.md`](wiki/methods/vla.md)、[`wiki/entities/paper-dynawm-vla-online-correction.md`](wiki/entities/paper-dynawm-vla-online-correction.md)、[`wiki/entities/paper-dreamsteer-vla-deployment-steering.md`](wiki/entities/paper-dreamsteer-vla-deployment-steering.md)、[`wiki/entities/paper-bridgevla-plusplus.md`](wiki/entities/paper-bridgevla-plusplus.md)、[`wiki/tasks/manipulation.md`](wiki/tasks/manipulation.md)、[`wiki/overview/wm-action-consequence-category-01-wam-action-prediction.md`](wiki/overview/wm-action-consequence-category-01-wam-action-prediction.md)

## [2026-08-07] ingest | sources/papers/bridgevla_plusplus_arxiv_2608_05042.md — BridgeVLA++（arXiv:2608.05042）时空记忆 3D VLA 入库；升格 wiki/entities/paper-bridgevla-plusplus.md；互链 wiki/methods/vla.md、wiki/entities/paper-kemo-event-driven-keyframe-memory-vla.md、wiki/entities/paper-eventvla-visual-evidence-memory.md、wiki/entities/paper-chronos.md、wiki/tasks/manipulation.md；sources/repos/bridgevla.md、sources/sites/bridgevla-plus-github-io.md

- **触发：** 用户指定 <https://github.com/BridgeVLA/BridgeVLA>、<https://arxiv.org/abs/2608.05042>、<https://bridgevla-plus.github.io/>、<https://huggingface.co/datasets/LPY/BridgeVLA>
- **来源：** [`sources/papers/bridgevla_plusplus_arxiv_2608_05042.md`](sources/papers/bridgevla_plusplus_arxiv_2608_05042.md)、[`sources/repos/bridgevla.md`](sources/repos/bridgevla.md)、[`sources/sites/bridgevla-plus-github-io.md`](sources/sites/bridgevla-plus-github-io.md)
- **新建实体：** [`wiki/entities/paper-bridgevla-plusplus.md`](wiki/entities/paper-bridgevla-plusplus.md)
- **开源核查（步骤 2.5）：** **已开源** — Apache-2.0；五仿真基准 + HF/ModelScope 权重；真机数据未发；`## 源码运行时序图` 已写
- **方法要点：** 多视图 heatmap 对齐 + 𝒯/𝒮 时空记忆；RMBench **96.0%**、RLBench **93.7%**；+9.2% 参数
- **交叉更新：** [`wiki/methods/vla.md`](wiki/methods/vla.md)、[`wiki/entities/paper-kemo-event-driven-keyframe-memory-vla.md`](wiki/entities/paper-kemo-event-driven-keyframe-memory-vla.md)、[`wiki/entities/paper-eventvla-visual-evidence-memory.md`](wiki/entities/paper-eventvla-visual-evidence-memory.md)、[`wiki/entities/paper-chronos.md`](wiki/entities/paper-chronos.md)、[`wiki/tasks/manipulation.md`](wiki/tasks/manipulation.md)、[`wiki/entities/paper-rtcf.md`](wiki/entities/paper-rtcf.md)

## [2026-08-07] ingest | sources/papers/dreamwam_arxiv_2608_04996.md — DreamWAM（arXiv:2608.04996）beyond-RGB Joint WAM 入库；升格 wiki/entities/paper-dreamwam.md；互链 wiki/concepts/world-action-models.md、wiki/methods/vla.md、wiki/entities/paper-dynawm-vla-online-correction.md、wiki/overview/wm-action-consequence-category-01-wam-action-prediction.md、wiki/tasks/manipulation.md；sources/repos/dreamwam.md、sources/sites/hustvl-dreamwam-github-io.md

- **触发：** 用户指定 <https://github.com/hustvl/DreamWAM>、<https://arxiv.org/abs/2608.04996>、<https://hustvl.github.io/DreamWAM/>、<https://huggingface.co/hustvl/DreamWAM>
- **来源：** [`sources/papers/dreamwam_arxiv_2608_04996.md`](sources/papers/dreamwam_arxiv_2608_04996.md)、[`sources/repos/dreamwam.md`](sources/repos/dreamwam.md)、[`sources/sites/hustvl-dreamwam-github-io.md`](sources/sites/hustvl-dreamwam-github-io.md)
- **新建实体：** [`wiki/entities/paper-dreamwam.md`](wiki/entities/paper-dreamwam.md)
- **开源核查（步骤 2.5）：** **已开源** — GitHub 训练/评测入口 + HF MIT 权重；`## 源码运行时序图` 已写
- **方法要点：** RGB+Flow 联合去噪 + Depth/DINO 门控残差；部署 RGB-only；LIBERO **98.90%**、LIBERO-Plus **75.47%**、真机扰动 **74.4%**
- **机构注册：** [`schema/institutions.json`](schema/institutions.json) 新增 `d-robotics` → 地瓜机器人（D-Robotics）
- **交叉更新：** [`wiki/concepts/world-action-models.md`](wiki/concepts/world-action-models.md)、[`wiki/methods/vla.md`](wiki/methods/vla.md)、[`wiki/entities/paper-dynawm-vla-online-correction.md`](wiki/entities/paper-dynawm-vla-online-correction.md)、[`wiki/overview/wm-action-consequence-category-01-wam-action-prediction.md`](wiki/overview/wm-action-consequence-category-01-wam-action-prediction.md)、[`wiki/tasks/manipulation.md`](wiki/tasks/manipulation.md)

## [2026-08-07] ingest | sources/repos/robotis-git.md — ROBOTIS-GIT 组织总览入库；升格 wiki/entities/robotis.md 与 Physical AI 主线实体；互链厂商 Lab 与既有 TurtleBot3/OpenMANIPULATOR/Cyclo 页

- **触发：** 用户指定 <https://github.com/ROBOTIS-GIT>；要求自动合入
- **来源：** [`sources/repos/robotis-git.md`](sources/repos/robotis-git.md)、[`sources/repos/cyclo.md`](sources/repos/cyclo.md)、[`sources/repos/ai_worker.md`](sources/repos/ai_worker.md)、[`sources/repos/ai_sapiens.md`](sources/repos/ai_sapiens.md)、[`sources/repos/cyclo_lab.md`](sources/repos/cyclo_lab.md)、[`sources/repos/physical_ai_tools.md`](sources/repos/physical_ai_tools.md)、[`sources/repos/dynamixel_sdk.md`](sources/repos/dynamixel_sdk.md)、[`sources/repos/robotis_mujoco_menagerie.md`](sources/repos/robotis_mujoco_menagerie.md)
- **新建实体：** [`wiki/entities/robotis.md`](wiki/entities/robotis.md)、[`wiki/entities/robotis-ai-worker.md`](wiki/entities/robotis-ai-worker.md)、[`wiki/entities/robotis-ai-sapiens.md`](wiki/entities/robotis-ai-sapiens.md)、[`wiki/entities/cyclo-lab.md`](wiki/entities/cyclo-lab.md)、[`wiki/entities/robotis-physical-ai-tools.md`](wiki/entities/robotis-physical-ai-tools.md)、[`wiki/entities/dynamixel-sdk.md`](wiki/entities/dynamixel-sdk.md)、[`wiki/entities/robotis-mujoco-menagerie.md`](wiki/entities/robotis-mujoco-menagerie.md)
- **开源核查（步骤 2.5）：** **已开源（主线）** — 组织公开；主线仓多为 Apache-2.0；Cyclo README 标明 Supervisor/Hub 等 **私有栈**不在公开组织
- **要点：** Dynamixel → 教育平台 → Cyclo Physical AI（Lab / Tools / Intelligence / AI Worker / AI Sapiens）；`cyclo_lab` 补齐与宇树/云深处/直驱厂商 Lab 对照
- **交叉更新：** [`wiki/entities/cyclo-intelligence.md`](wiki/entities/cyclo-intelligence.md)、[`wiki/entities/turtlebot3.md`](wiki/entities/turtlebot3.md)、[`wiki/entities/robotis-open-manipulator-line.md`](wiki/entities/robotis-open-manipulator-line.md)、[`wiki/entities/robot-lab.md`](wiki/entities/robot-lab.md)、[`wiki/entities/unitree-rl-lab.md`](wiki/entities/unitree-rl-lab.md)、[`wiki/entities/deeprobotics-rl-training.md`](wiki/entities/deeprobotics-rl-training.md)、[`wiki/entities/ddt-lab.md`](wiki/entities/ddt-lab.md)

## [2026-08-07] ingest | sources/repos/rl_training.md — 云深处官方 Isaac Lab RL 训练仓入库；wiki/entities/deeprobotics-rl-training.md wiki/entities/robot-lab.md wiki/concepts/wheel-legged-quadruped.md wiki/entities/ddt-lab.md wiki/entities/unitree-rl-lab.md

- **触发：** 用户指定 <https://github.com/DeepRoboticsLab/rl_training>、<https://github.com/unitreerobotics/unitree_rl_lab>、<https://github.com/DDTRobot/DDT_Lab>；要求自动合入
- **来源：** [`sources/repos/rl_training.md`](sources/repos/rl_training.md)
- **新建实体：** [`wiki/entities/deeprobotics-rl-training.md`](wiki/entities/deeprobotics-rl-training.md)
- **开源核查（步骤 2.5）：** **已开源** — BSD-3-Clause；Lite3 / M20 / DR02（AMP）；部署指向同组织 `sdk_deploy`
- **交叉更新：** [`wiki/entities/robot-lab.md`](wiki/entities/robot-lab.md)、[`wiki/concepts/wheel-legged-quadruped.md`](wiki/concepts/wheel-legged-quadruped.md)、[`wiki/entities/unitree-rl-lab.md`](wiki/entities/unitree-rl-lab.md)、[`wiki/entities/ddt-lab.md`](wiki/entities/ddt-lab.md)

## [2026-08-07] ingest | sources/repos/ddt_lab.md — 直驱科技 DDT_Lab（NP3O 轮足）入库；wiki/entities/ddt-lab.md wiki/concepts/wheel-legged-quadruped.md wiki/entities/robot-lab.md wiki/entities/deeprobotics-rl-training.md

- **触发：** 用户指定 <https://github.com/DDTRobot/DDT_Lab>（与云深处 / 宇树厂商 Lab 一并入库）
- **来源：** [`sources/repos/ddt_lab.md`](sources/repos/ddt_lab.md)
- **新建实体：** [`wiki/entities/ddt-lab.md`](wiki/entities/ddt-lab.md)
- **开源核查（步骤 2.5）：** **已开源** — D1 / Tita + NP3O；URDF 依赖同组织 `ddt_ros2_control`
- **交叉更新：** [`wiki/concepts/wheel-legged-quadruped.md`](wiki/concepts/wheel-legged-quadruped.md)、[`wiki/entities/robot-lab.md`](wiki/entities/robot-lab.md)、[`wiki/entities/deeprobotics-rl-training.md`](wiki/entities/deeprobotics-rl-training.md)

## [2026-08-07] ingest | sources/repos/unitree_rl_lab.md — 复核官方 Isaac Lab RL 仓并互链厂商 Lab；wiki/entities/unitree-rl-lab.md wiki/entities/deeprobotics-rl-training.md wiki/entities/ddt-lab.md

- **触发：** 用户指定复核 <https://github.com/unitreerobotics/unitree_rl_lab>（已有条目，刷新元数据并互链新厂商 Lab）
- **来源：** [`sources/repos/unitree_rl_lab.md`](sources/repos/unitree_rl_lab.md)
- **更新实体：** [`wiki/entities/unitree-rl-lab.md`](wiki/entities/unitree-rl-lab.md)
- **开源核查（步骤 2.5）：** **已开源** — Apache-2.0；Go2 / H1 / G1-29dof；含 C++ deploy
- **交叉更新：** [`wiki/entities/deeprobotics-rl-training.md`](wiki/entities/deeprobotics-rl-training.md)、[`wiki/entities/ddt-lab.md`](wiki/entities/ddt-lab.md)、[`wiki/entities/robot-lab.md`](wiki/entities/robot-lab.md)

## [2026-08-07] ingest | sources/sites/rekacs2-10k.md + sources/datasets/rekacs2-10k.md + sources/repos/cs2-dem-renderer.md — RekaCS2-10k（CS2 ego 视频+逐帧控制）合并入库；升格 wiki/entities/rekacs2-10k-dataset.md；互链 wiki/concepts/world-action-models.md、wiki/concepts/video-as-simulation.md、wiki/overview/ego-category-01-data-collection.md

- **触发：** 用户指定 <https://huggingface.co/datasets/RekaAI/CS2-10k> 与 <https://reka.ai/news/cs2-10k-a-large-scale-egocentric-counter-strike-2-dataset>；要求自动合并
- **来源：** [`sources/sites/rekacs2-10k.md`](sources/sites/rekacs2-10k.md)、[`sources/datasets/rekacs2-10k.md`](sources/datasets/rekacs2-10k.md)、[`sources/repos/cs2-dem-renderer.md`](sources/repos/cs2-dem-renderer.md)
- **新建实体：** [`wiki/entities/rekacs2-10k-dataset.md`](wiki/entities/rekacs2-10k-dataset.md)
- **开源核查（步骤 2.5）：** **已开源** — HF 数据 ungated / **CC BY-NC 4.0**；渲染器 [reka-ai/cs2-dem-renderer](https://github.com/reka-ai/cs2-dem-renderer) **MIT**；Viewer Space 可用；`## 源码运行时序图` 已写
- **要点：** 600k+ 回合 / 10k+ 小时 / 720p@48fps；逐帧键鼠+3D 轨迹；职业 demo 确定性重渲染；非商用许可
- **机构注册：** [`schema/institutions.json`](schema/institutions.json) 新增 `reka` → 瑞卡人工智能（Reka AI）
- **交叉更新：** [`wiki/concepts/world-action-models.md`](wiki/concepts/world-action-models.md)、[`wiki/concepts/video-as-simulation.md`](wiki/concepts/video-as-simulation.md)、[`wiki/overview/ego-category-01-data-collection.md`](wiki/overview/ego-category-01-data-collection.md)

## [2026-08-07] ingest | sources/sites/rekadaily-10k.md + sources/datasets/rekadaily-10k-raw.md — RekaDaily-10k 家务 egocentric 视频（研究页+HF raw）合并入库；升格 wiki/entities/rekadaily-10k-dataset.md；互链 wiki/overview/ego-category-01-data-collection.md、wiki/methods/egoscale.md、wiki/entities/egoworld-100w.md、wiki/entities/hiw-500-dataset.md、wiki/queries/humanoid-training-data-pipeline.md、wiki/entities/paper-data-pyramid-embodied-manipulation.md

- **触发：** 用户指定 <https://huggingface.co/datasets/RekaAI/RekaDaily-10k-raw> 与 <https://reka.ai/labs/research/rekadaily-10k-egocentric-household-manipulation-data>；要求自动合并
- **来源：** [`sources/sites/rekadaily-10k.md`](sources/sites/rekadaily-10k.md)、[`sources/datasets/rekadaily-10k-raw.md`](sources/datasets/rekadaily-10k-raw.md)
- **新建实体：** [`wiki/entities/rekadaily-10k-dataset.md`](wiki/entities/rekadaily-10k-dataset.md)
- **开源核查（步骤 2.5）：** **已开源（数据，增量）** — HF raw ungated / Apache 2.0；入库日 README 约 **886 h / 39,643 视频**；全量目标 **10,312 h**；processed+captioned 另档宣称；无训练代码仓
- **要点：** Claru 付费无剧本家务 ego；raw vs processed 双档；约 1,670 h 原生 4K；PII 三道门；WebDataset shards
- **机构注册：** [`schema/institutions.json`](schema/institutions.json) 新增 `reka` → 瑞卡人工智能（Reka AI）
- **交叉更新：** [`wiki/overview/ego-category-01-data-collection.md`](wiki/overview/ego-category-01-data-collection.md)、[`wiki/methods/egoscale.md`](wiki/methods/egoscale.md)、[`wiki/entities/egoworld-100w.md`](wiki/entities/egoworld-100w.md)、[`wiki/entities/hiw-500-dataset.md`](wiki/entities/hiw-500-dataset.md)、[`wiki/queries/humanoid-training-data-pipeline.md`](wiki/queries/humanoid-training-data-pipeline.md)、[`wiki/entities/paper-data-pyramid-embodied-manipulation.md`](wiki/entities/paper-data-pyramid-embodied-manipulation.md)

## [2026-08-06] ingest | sources/papers/effective_degree_arxiv_2605_29823.md — Effective Degree（arXiv:2605.29823，ICML 2026，清华）多项式代理量化简洁性；升格 wiki/entities/paper-effective-degree.md；互链 wiki/concepts/deep-learning-foundations.md、wiki/methods/reinforcement-learning.md、wiki/methods/ppo.md、wiki/methods/adamw.md；sources/repos/effective-degree.md

- **触发：** 用户指定 *Quantifying and Optimizing Simplicity via Polynomial Representations*（arXiv:2605.29823；章天任 / 李向欣 / 肖明昊 / 陈冠宇 / 陈峰；清华；代码 https://github.com/xinzaixinzai/Effective-Degree）；要求自动合入
- **来源：** [`sources/papers/effective_degree_arxiv_2605_29823.md`](sources/papers/effective_degree_arxiv_2605_29823.md)、[`sources/repos/effective-degree.md`](sources/repos/effective-degree.md)
- **新建实体：** [`wiki/entities/paper-effective-degree.md`](wiki/entities/paper-effective-degree.md)
- **开源核查（步骤 2.5）：** **已开源** — 官方仓含 `train_wd_regular_torch.py` / `poly/` / `rl/ppo_procgen.sh` 等；`## 源码运行时序图` 已写；许可证未声明
- **方法要点：** 数据插值路径 + Chebyshev 代理 → Effective Degree；可微正则；CIFAR ViT-Tiny 87.80→90.82；CLIP/GLUE/Procgen 增益；相关强于 sharpness
- **交叉更新：** [`wiki/concepts/deep-learning-foundations.md`](wiki/concepts/deep-learning-foundations.md)、[`wiki/methods/reinforcement-learning.md`](wiki/methods/reinforcement-learning.md)、[`wiki/methods/ppo.md`](wiki/methods/ppo.md)、[`wiki/methods/adamw.md`](wiki/methods/adamw.md)


## [2026-08-06] ingest | sources/papers/xmorph_arxiv_2606_30290.md — X-Morph（arXiv:2606.30290，NUS）人体→非人形腿式跨形态运动先验；升格 wiki/entities/paper-xmorph.md；互链 wiki/overview/hub-motion-retargeting.md、wiki/overview/hub-cross-embodiment.md、wiki/methods/motion-retargeting-gmr.md、wiki/methods/reactor-physics-aware-motion-retargeting.md、wiki/methods/zest.md、wiki/entities/unitree-g1.md、wiki/entities/unitree.md；sources/sites/maker-rat-morph-github-io.md

- **触发：** 用户指定 *X-Morph: Human Motion Priors for Scalable Robot Learning Across Morphologies*（arXiv；新加坡国立大学；Ritwik Sharma / Shivam Sood / Arhaan Jain / Shyam Charan Kesavamoorthi / Chengyang He / Guillaume Sartoretti）；要求自动合入
- **来源：** [`sources/papers/xmorph_arxiv_2606_30290.md`](sources/papers/xmorph_arxiv_2606_30290.md)、[`sources/sites/maker-rat-morph-github-io.md`](sources/sites/maker-rat-morph-github-io.md)
- **新建实体：** [`wiki/entities/paper-xmorph.md`](wiki/entities/paper-xmorph.md)
- **开源核查（步骤 2.5）：** **宣称将开源 / 未列链接** — 项目页 Code/Video 按钮 disabled，无 GitHub URL；`## 源码运行时序图` 写不适用
- **方法要点：** G1 源表示 → PAN 式跨形态重定向 → 物理感知校正 → 特权 teacher + 因果 student；Go2/Yuna/B2-Z1；视频遥操作 ≤28.9 Hz；corrector 使 Go2 slip −27.2%、penetration −46.9%，Yuna Joint MAE −17.4%
- **交叉更新：** [`wiki/overview/hub-motion-retargeting.md`](wiki/overview/hub-motion-retargeting.md)、[`wiki/overview/hub-cross-embodiment.md`](wiki/overview/hub-cross-embodiment.md)、[`wiki/methods/motion-retargeting-gmr.md`](wiki/methods/motion-retargeting-gmr.md)、[`wiki/methods/reactor-physics-aware-motion-retargeting.md`](wiki/methods/reactor-physics-aware-motion-retargeting.md)、[`wiki/methods/zest.md`](wiki/methods/zest.md)


## [2026-08-06] ingest | sources/papers/uma_arxiv_2506_01802.md — UMA（arXiv:2506.01802，ACM TOG 2026）多级表面对齐超精细可驱动着装人体 avatar；升格 wiki/entities/paper-uma.md；互链 wiki/tasks/teleoperation.md、wiki/queries/humanoid-training-data-pipeline.md、wiki/concepts/smpl-x.md、wiki/entities/paper-face-anything-4d-face-reconstruction.md、wiki/entities/paper-shells-layered-surface-sampling.md；sources/repos/uma.md、sources/sites/vcai-mpi-inf-uma.md

- **触发：** 用户指定 <https://vcai.mpi-inf.mpg.de/projects/UMA/> 与 <https://arxiv.org/abs/2506.01802>；要求自动合并
- **来源：** [`sources/papers/uma_arxiv_2506_01802.md`](sources/papers/uma_arxiv_2506_01802.md)、[`sources/sites/vcai-mpi-inf-uma.md`](sources/sites/vcai-mpi-inf-uma.md)、[`sources/repos/uma.md`](sources/repos/uma.md)（MPI-INF × VIA）
- **新建实体：** [`wiki/entities/paper-uma.md`](wiki/entities/paper-uma.md)
- **开源核查（步骤 2.5）：** **部分开源** — 项目页 Paper/GitHub/Dataset/Demo 齐全；数据集 + 推理 + Viewer demo + checkpoint 已发；README TODO 仍勾选 training utilities；`## 源码运行时序图` 已写
- **方法要点：** per-frame latent 解衣物随机动力学 + CoTracker 引导顶点/纹素多级对齐 + Gaussian 纹理超分；40×6K×5 被试；Training PSNR **37.15** / LPIPS **35.02**
- **交叉更新：** [`wiki/tasks/teleoperation.md`](wiki/tasks/teleoperation.md)、[`wiki/queries/humanoid-training-data-pipeline.md`](wiki/queries/humanoid-training-data-pipeline.md)、[`wiki/concepts/smpl-x.md`](wiki/concepts/smpl-x.md)、[`wiki/entities/paper-face-anything-4d-face-reconstruction.md`](wiki/entities/paper-face-anything-4d-face-reconstruction.md)、[`wiki/entities/paper-shells-layered-surface-sampling.md`](wiki/entities/paper-shells-layered-surface-sampling.md)

## [2026-08-06] structural | roadmap/depth-teleoperation.md + depth-navigation.md + depth-imitation-learning.md + depth-rl-locomotion.md + depth-perceptive-locomotion.md + depth-wam.md + depth-motion-generation.md + depth-vla.md — 补齐 2026-08-02 以来（roadmap 上次触达日）入库的 12 篇最新论文节点到对应纵深路线「推荐读什么」/ 进阶方向关键词，覆盖遥操作、导航、模仿学习、RL locomotion、感知越障、WAM、动作生成、VLA 八条路线

- **触发：** 批量结构性回填——`roadmap/*.md` 自 2026-08-02/03 起未再更新，其间 8 篇用户已列出候选（Teleopit / RoamFlow / kai0 / FDDC / Light-Loco-Parkour / WorldScape Policy 2.0 / Why Action Chunking Improves BC）与向下扫描 log.md 发现的 5 篇额外候选（HiFi-UMI / WCM / 合成视频人形任务生成 / ActFovea / RoboHarness）经逐页复核后确认为对应路线的里程碑级或教学级读物；TransGraspNet（透明器皿抓取，应用较窄）、机器人学习五大范式对比页（通用综述，不专属单一纵深）、CLIFT（与同 Stage 已有 STEAM 主题重叠，判定为增量较小）、SenseNova-U1.5（非机器人动作模型）判定不够纵深特定或增量不足，未收录；浙大人形下肢衍生式设计论文已在 `depth-humanoid-hardware-design.md` Stage 2 存在，无需补
- **[`roadmap/depth-teleoperation.md`](roadmap/depth-teleoperation.md)：** Stage 1 输入接口与硬件通道补 [HiFi-UMI](wiki/entities/paper-hifi-umi.md)（高保真无机器人双臂 UMI，~3 mm/<40 µs，HiFi-UMI-2K 2000 h 开源）；Stage 3 人形全身遥操作补 [Teleopit](wiki/entities/paper-teleopit.md)（PICO VR 全身+连续跨手+主动视觉单一传感源，持出 SR 91.7%/100.0% 超 TWIST2/SONIC/HoloMotion，五仓开源）
- **[`roadmap/depth-navigation.md`](roadmap/depth-navigation.md)：** Stage 3 学习型导航补 [RoamFlow](wiki/entities/paper-roamflow.md)（MeanFlow 一步轨迹 + IL→RL，Habitat SR 68.7%/19.6 ms，Go2 真机 SR 1.00，未开源）
- **[`roadmap/depth-imitation-learning.md`](roadmap/depth-imitation-learning.md)：** Stage 3 Diffusion Policy 补 [Why Action Chunking Improves BC](wiki/entities/paper-why-action-chunking-improves-bc.md)（CoRL 2026，拆穿时序一致性/短 horizon 假说，Delay/RDE 部署可匹配 AC）；Stage 6 方向 C 关键词补 [χ₀/kai0](wiki/entities/paper-kai0.md)（Model Arithmetic + Stage Advantage + TDA，相对 π₀.₅ 成功率约 +250%，已开源）
- **[`roadmap/depth-rl-locomotion.md`](roadmap/depth-rl-locomotion.md)：** Stage 2 RL+Locomotion 核心方法补 [FDDC](wiki/entities/paper-fddc.md)（支撑足相对动态 CoM 可部署观测 + 人体科学奖励，单腿平衡基准 Perfect 95.6% vs 八个通用 SOTA 全 0%）
- **[`roadmap/depth-perceptive-locomotion.md`](roadmap/depth-perceptive-locomotion.md)：** Stage 3 越障进阶补 [Light-Loco-Parkour](wiki/entities/paper-light-loco-parkour.md)（稀疏种子 Real2Sim2Real + 多专家蒸馏无技能标签深度跑酷，攀爬至 0.83H，代码未开源）
- **[`roadmap/depth-wam.md`](roadmap/depth-wam.md)：** Stage 3 Joint WAM 补 [WorldScape Policy 2.0](wiki/entities/paper-worldscape-policy-2.md)（事件/视觉双记忆通路 + semantic forcing，RoboTwin 2.0 94.3%，C2R 47.9% 大幅超 Fast-WAM）
- **[`roadmap/depth-motion-generation.md`](roadmap/depth-motion-generation.md)：** Stage 3 物理化落地补 [合成视频人形任务生成（NCKU）](wiki/entities/paper-synthetic-video-humanoid-tasks.md)（文本→Veo 视频→SMPL-X/GMR 重定向→DeepMimic 式 RL 跟踪，无真机/无 MoCap 示范）
- **[`roadmap/depth-vla.md`](roadmap/depth-vla.md)：** Stage 4 部署与整合补 [ActFovea](wiki/entities/paper-actfovea.md)（不重训 VLA 运行时防护层，LIBERO 视觉攻击成功率 49.3%→90.3%）与 [RoboHarness](wiki/entities/paper-robo-harness.md)（异构策略 agentic skills 编排，LIBERO-LoHo 95.2% vs π₀.₅ 6.4%）；Stage 5 方向 A 关键词补 [WCM](wiki/entities/paper-wcm-world-critic-model.md)（世界模型 critic 修正 VLA RL 单帧价值错配，4 基准 149 任务提升 π₀/π₀.₅/OpenVLA-OFT）
- **口径：** 每条路线仅在既有 Stage 结构内追加 bullet / 关键词，未新建 Stage、未改动其余既有内容

## [2026-08-06] ingest | sources/papers/humembr_arxiv_2606_30404.md — HUMEMBR（arXiv:2606.30404）人中心长时程记忆 + PersonEQA；升格 wiki/entities/paper-humembr.md；互链 wiki/tasks/vision-language-navigation.md、wiki/entities/paper-uni-lavira.md、wiki/entities/qwen-robot-nav.md、wiki/overview/navigation-slam-autonomy-stack.md、wiki/entities/paper-icrowdnav.md、roadmap/depth-navigation.md；sources/repos/humembr.md、sources/sites/samirahuber-humembr-github-io.md；注册 kiel 机构

- **触发：** 用户指定 <https://arxiv.org/abs/2606.30404>（HUMEMBR；IROS 2026）；要求自动合并
- **来源：** [`sources/papers/humembr_arxiv_2606_30404.md`](sources/papers/humembr_arxiv_2606_30404.md)、[`sources/sites/samirahuber-humembr-github-io.md`](sources/sites/samirahuber-humembr-github-io.md)、[`sources/repos/humembr.md`](sources/repos/humembr.md)（Kiel University × George Mason University）
- **新建实体：** [`wiki/entities/paper-humembr.md`](wiki/entities/paper-humembr.md)
- **机构注册：** `schema/institutions.json` 新增 `kiel` → 基尔大学（Kiel University）
- **开源核查（步骤 2.5）：** **代码已开源** — 项目页 Code → [`samirahuber/humembr`](https://github.com/samirahuber/humembr)（robot/server/agent/processing + pgvector）；**COBD 数据集 README 标明 private**；`## 源码运行时序图` 已写
- **方法要点：** 并行记忆构建（Qwen caption + 脸 DBSCAN + KPR ReID）与 LLM 五类检索工具；PersonEQA 上 Gemini **75.41%** vs 全上下文 **67.33%**（~17% token）；Spot + GraphNav 真机六任务
- **交叉更新：** [`wiki/tasks/vision-language-navigation.md`](wiki/tasks/vision-language-navigation.md)、[`wiki/entities/paper-uni-lavira.md`](wiki/entities/paper-uni-lavira.md)、[`wiki/entities/qwen-robot-nav.md`](wiki/entities/qwen-robot-nav.md)、[`wiki/overview/navigation-slam-autonomy-stack.md`](wiki/overview/navigation-slam-autonomy-stack.md)、[`wiki/entities/paper-icrowdnav.md`](wiki/entities/paper-icrowdnav.md)、[`roadmap/depth-navigation.md`](roadmap/depth-navigation.md)

## [2026-08-06] ingest | sources/papers/cmp_arxiv_2608_03234.md — CMP（arXiv:2608.03234）上下文感知运动先验入库；升格 wiki/entities/paper-cmp.md；交叉 wiki/methods/amp-reward.md、wiki/methods/smp.md、wiki/comparisons/amp-add-smp-motion-prior-variants.md、wiki/overview/humanoid-amp-motion-prior-survey.md、wiki/entities/mimickit.md、wiki/entities/paper-amp-survey-01-amp.md、wiki/queries/humanoid-motion-tracking-method-selection.md、wiki/entities/unitree-g1.md

- **触发：** 用户指定 *Learning Context-Aware Motion Priors for Humanoid Control*（arXiv；HKUST-GZ；Yunyang Mo / Yi Gu / Yangchen Zhou / Hanyang Cao / Renjing Xu）；要求自动合并
- **来源：** [`sources/papers/cmp_arxiv_2608_03234.md`](sources/papers/cmp_arxiv_2608_03234.md)
- **新建实体：** [`wiki/entities/paper-cmp.md`](wiki/entities/paper-cmp.md)
- **开源核查（步骤 2.5）：** **确认未开源** — abs/HTML 无项目页 / GitHub；`## 源码运行时序图` 写不适用
- **方法要点：** 高优势 rollout + \(\mathcal{L}_{\mathrm{demo}}\) 对比相关度 → 软重权参考监督 → AMP/SMP 轻量残差适配器；五任务回报与样本效率提升；行走 ×100 失衡 AMP −11.5% / CMP-AMP −2.8%；附录 E 模拟 G1 同趋势
- **交叉更新：** [`wiki/methods/amp-reward.md`](wiki/methods/amp-reward.md)、[`wiki/methods/smp.md`](wiki/methods/smp.md)、[`wiki/comparisons/amp-add-smp-motion-prior-variants.md`](wiki/comparisons/amp-add-smp-motion-prior-variants.md)、[`wiki/overview/humanoid-amp-motion-prior-survey.md`](wiki/overview/humanoid-amp-motion-prior-survey.md)、[`wiki/entities/mimickit.md`](wiki/entities/mimickit.md)、[`wiki/entities/paper-amp-survey-01-amp.md`](wiki/entities/paper-amp-survey-01-amp.md)、[`wiki/queries/humanoid-motion-tracking-method-selection.md`](wiki/queries/humanoid-motion-tracking-method-selection.md)、[`wiki/entities/unitree-g1.md`](wiki/entities/unitree-g1.md)

## [2026-08-06] ingest | sources/papers/dpl_arxiv_2510_07152.md — DPL（arXiv:2510.07152，IEEE RA-L）升格 wiki/entities/paper-notebook-dpl-depth-only-perceptive-humanoid-locomotion-vi.md；交叉 wiki/tasks/stair-obstacle-perceptive-locomotion.md、wiki/concepts/terrain-adaptation.md、wiki/tasks/locomotion.md、wiki/entities/paper-rpl-robust-humanoid-perceptive-locomotion.md、wiki/entities/x-humanoid.md、wiki/overview/paper-notebook-category-05-locomotion.md、sources/papers/humanoid_pnb_dpl-depth-only-perceptive-humanoid-locomotion-vi.md

- **触发：** 用户指定 <https://arxiv.org/abs/2510.07152>（v1 2025-10-08 → v3 2026-08-03；IEEE RA-L 接收 2026-06-09）；要求自动合并
- **来源：** [`sources/papers/dpl_arxiv_2510_07152.md`](sources/papers/dpl_arxiv_2510_07152.md)（X-Humanoid / HKU / USTC / HKUST）
- **升格实体：** [`wiki/entities/paper-notebook-dpl-depth-only-perceptive-humanoid-locomotion-vi.md`](wiki/entities/paper-notebook-dpl-depth-only-perceptive-humanoid-locomotion-vi.md)（原 Paper Notebooks stub → complete）
- **开源核查（步骤 2.5）：** **确认未开源** — 无项目页 / 无官方仓；`## 源码运行时序图` 写不适用
- **方法要点：** 自遮挡射线深度合成 + 跨模态交叉注意力高程重建 + 盲骨干多教师蒸馏与端到端微调；TienKung Ultra + Orbbec；真机重建 MAE **3.25 cm**；感知 ~**20 ms @ 30 Hz**；楼梯绊脚 4/10 vs 无 e2e 8/10
- **交叉更新：** [`wiki/tasks/stair-obstacle-perceptive-locomotion.md`](wiki/tasks/stair-obstacle-perceptive-locomotion.md)、[`wiki/concepts/terrain-adaptation.md`](wiki/concepts/terrain-adaptation.md)、[`wiki/tasks/locomotion.md`](wiki/tasks/locomotion.md)、[`wiki/entities/paper-rpl-robust-humanoid-perceptive-locomotion.md`](wiki/entities/paper-rpl-robust-humanoid-perceptive-locomotion.md)、[`wiki/entities/x-humanoid.md`](wiki/entities/x-humanoid.md)、[`wiki/overview/paper-notebook-category-05-locomotion.md`](wiki/overview/paper-notebook-category-05-locomotion.md)、[`sources/papers/humanoid_pnb_dpl-depth-only-perceptive-humanoid-locomotion-vi.md`](sources/papers/humanoid_pnb_dpl-depth-only-perceptive-humanoid-locomotion-vi.md)

## [2026-08-05] ingest | sources/papers/fddc_arxiv_2608_00500.md — FDDC（arXiv:2608.00500）可部署动态 CoM 单腿平衡入库；升格 wiki/entities/paper-fddc.md；交叉 wiki/concepts/capture-point-dcm.md、reward-design.md、wiki/tasks/balance-recovery.md、wiki/methods/ams.md、sonic-motion-tracking.md、wiki/entities/unitree-g1.md、paper-notebook-hub.md、paper-notebook-learning-sim-to-real-humanoid-locomotion-in-15-m.md

- **触发：** 用户指定 <https://arxiv.org/abs/2608.00500>
- **来源：** [`sources/papers/fddc_arxiv_2608_00500.md`](sources/papers/fddc_arxiv_2608_00500.md)（北京大学；2026-08-01）
- **新建实体：** [`wiki/entities/paper-fddc.md`](wiki/entities/paper-fddc.md)
- **开源核查（步骤 2.5）：** **宣称全栈开源 / 未列 URL** — abs/TeX 写 release data/code/policy/benchmark，公开材料无 GitHub/项目页；`## 源码运行时序图` 写不适用
- **方法要点：** 支撑相对动态 CoM \(o_{\mathrm{bal}}\) 进可部署 actor + 人体科学奖励（MoS/TTB/踝→膝 rate/jerk）+ asymmetric FastSAC；单腿基准 Perfect **95.6%**（86/90），八个通用 SOTA **0/90**；动态 CoM 消融 −40 pt；G1 ONNX 无蒸馏
- **交叉更新：** [`wiki/concepts/capture-point-dcm.md`](wiki/concepts/capture-point-dcm.md)、[`wiki/concepts/reward-design.md`](wiki/concepts/reward-design.md)、[`wiki/tasks/balance-recovery.md`](wiki/tasks/balance-recovery.md)、[`wiki/methods/ams.md`](wiki/methods/ams.md)、[`wiki/methods/sonic-motion-tracking.md`](wiki/methods/sonic-motion-tracking.md)、[`wiki/entities/unitree-g1.md`](wiki/entities/unitree-g1.md)、[`wiki/entities/paper-notebook-hub.md`](wiki/entities/paper-notebook-hub.md)、[`wiki/entities/paper-notebook-learning-sim-to-real-humanoid-locomotion-in-15-m.md`](wiki/entities/paper-notebook-learning-sim-to-real-humanoid-locomotion-in-15-m.md)

## [2026-08-05] ingest | sources/papers/chi0_kai0_arxiv_2602_09021.md + sites/mmlab-kai0 + repos/kai0 — χ₀/kai0（arXiv:2602.09021）入库；升格 wiki/entities/paper-kai0.md；交叉 wiki/methods/dagger.md、awr.md、π0-policy.md、wiki/entities/paper-pi05-open-world-vla.md、paper-lehome-learning-to-fold.md、paper-steam-advantage-modeling.md、aloha.md、wiki/tasks/manipulation.md

- **触发：** 用户指定 <https://arxiv.org/abs/2602.09021>
- **来源：** [`sources/papers/chi0_kai0_arxiv_2602_09021.md`](sources/papers/chi0_kai0_arxiv_2602_09021.md)、[`sources/sites/mmlab-kai0.md`](sources/sites/mmlab-kai0.md)、[`sources/repos/kai0.md`](sources/repos/kai0.md)（Kinetix AI / HKU MMLab / OpenDriveLab；v3 2026-03-17）
- **新建实体：** [`wiki/entities/paper-kai0.md`](wiki/entities/paper-kai0.md)
- **机构注册：** `schema/institutions.json` 新增 `kinetix-ai` → 凯涅克斯人工智能（Kinetix AI）
- **开源核查（步骤 2.5）：** **已开源** — PDF/博客列出 [`OpenDriveLab/kai0`](https://github.com/OpenDriveLab/kai0)；MA/SA/TDA 均 Released；HF/ModelScope 数据 + 每任务 best ckpt；`## 源码运行时序图` 已写
- **方法要点：** Model Arithmetic（OOD DAgger 选 α）+ Stage Advantage（阶段条件成对 advantage）+ TDA（Heuristic DAgger + chunk 平滑）；相对 π₀.₅ SR 约 +250%；双臂协同展平/折叠/挂衣；24 h 连续运行
- **交叉更新：** [`wiki/methods/dagger.md`](wiki/methods/dagger.md)、[`wiki/methods/awr.md`](wiki/methods/awr.md)、[`wiki/methods/π0-policy.md`](wiki/methods/π0-policy.md)、[`wiki/entities/paper-pi05-open-world-vla.md`](wiki/entities/paper-pi05-open-world-vla.md)、[`wiki/entities/paper-lehome-learning-to-fold.md`](wiki/entities/paper-lehome-learning-to-fold.md)、[`wiki/entities/paper-steam-advantage-modeling.md`](wiki/entities/paper-steam-advantage-modeling.md)、[`wiki/entities/aloha.md`](wiki/entities/aloha.md)、[`wiki/tasks/manipulation.md`](wiki/tasks/manipulation.md)、[`sources/repos/openpi.md`](sources/repos/openpi.md)

## [2026-08-05] ingest | sources/papers/robotdancing_arxiv_2509_20717.md — RobotDancing（arXiv:2509.20717，IEEE RA-L）升格 wiki/entities/paper-notebook-robotdancing-residual-action-rl-enables-robust-l.md；交叉更新 wiki/methods/residual-policy-learning.md、wiki/overview/paper-notebook-category-13-physics-based-animation.md、sources/papers/humanoid_pnb_robotdancing.md

- **触发：** 用户指定 <https://arxiv.org/abs/2509.20717>（v1 2025-09-25 → v2 2026-08-03；IEEE RA-L 接收）
- **来源：** [`sources/papers/robotdancing_arxiv_2509_20717.md`](sources/papers/robotdancing_arxiv_2509_20717.md)（TUM / BAAI / XYZ Embodied AI / 清华 / 南大）
- **升格实体：** [`wiki/entities/paper-notebook-robotdancing-residual-action-rl-enables-robust-l.md`](wiki/entities/paper-notebook-robotdancing-residual-action-rl-enables-robust-l.md)（原 Paper Notebooks stub → complete）
- **开源核查（步骤 2.5）：** **确认未开源** — 无项目页 / 无官方仓；附录称 released config 但无 URL；`## 源码运行时序图` 写不适用
- **方法要点：** $q^{\mathrm{tar}}=q^{\mathrm{ref}}+a$（选择性髋/膝 pitch）+ 分布均衡/失败优先采样；G1 八舞蹈 21/24 真机成功；跨平台 H1/H1-2
- **交叉更新：** [`wiki/methods/residual-policy-learning.md`](wiki/methods/residual-policy-learning.md)、[`wiki/overview/paper-notebook-category-13-physics-based-animation.md`](wiki/overview/paper-notebook-category-13-physics-based-animation.md)、[`sources/papers/humanoid_pnb_robotdancing.md`](sources/papers/humanoid_pnb_robotdancing.md)、[`wiki/entities/paper-notebook-asap-aligning-simulation-and-real-world-physics.md`](wiki/entities/paper-notebook-asap-aligning-simulation-and-real-world-physics.md)、[`wiki/entities/paper-notebook-kungfubot-physics-based-humanoid-whole-body-cont.md`](wiki/entities/paper-notebook-kungfubot-physics-based-humanoid-whole-body-cont.md)

## [2026-08-05] ingest | sources/papers/transgraspnet_arxiv_2607_29567.md — TransGraspNet（arXiv:2607.29567）透明含液实验器皿几何–物理一致抓取入库；升格 wiki/entities/paper-transgraspnet.md；交叉 grasp-pose-estimation / grasp-policy-selection / hub-grasp / manipulation / anygrasp / anygrasp-vs-graspnet

- **触发：** 用户指定 <https://arxiv.org/abs/2607.29567>
- **来源：** [`sources/papers/transgraspnet_arxiv_2607_29567.md`](sources/papers/transgraspnet_arxiv_2607_29567.md)（PKU / SJTU / SUSTech；2026-07-31）
- **新建实体：** [`wiki/entities/paper-transgraspnet.md`](wiki/entities/paper-transgraspnet.md)
- **开源核查（步骤 2.5）：** **确认未开源** — 无项目页 / 无官方仓 / RobotSci-Glass 未公开下载；`## 源码运行时序图` 写不适用
- **方法要点：** E-CBAM+Edge Branch 分割 → EGAG 深度补全 → GraspNet 候选 + 质心/主轴/wrench 重打分；AUBO i5 真机 Simple 96% / Clutter 86%；0.5 m/s 运液零洒出
- **交叉更新：** [`wiki/methods/grasp-pose-estimation.md`](wiki/methods/grasp-pose-estimation.md)、[`wiki/queries/grasp-policy-selection.md`](wiki/queries/grasp-policy-selection.md)、[`wiki/overview/hub-grasp.md`](wiki/overview/hub-grasp.md)、[`wiki/tasks/manipulation.md`](wiki/tasks/manipulation.md)、[`wiki/entities/anygrasp.md`](wiki/entities/anygrasp.md)、[`wiki/comparisons/anygrasp-vs-graspnet.md`](wiki/comparisons/anygrasp-vs-graspnet.md)

## [2026-08-05] ingest | sources/blogs/wechat_shenlan_robot_learning_five_paradigms.md — 深蓝具身智能《机器人学习算法五大体系》入库；升格 wiki/comparisons/robot-learning-five-paradigms-taxonomy.md；交叉 wiki/methods/imitation-learning.md、reinforcement-learning.md、vla.md、wiki/comparisons/rl-vs-il.md、vlm-vln-vla-vlx-world-model-taxonomy.md、wiki/concepts/sim2real.md、wiki/overview/robot-learning-overview.md

- **触发：** 用户指定 <https://mp.weixin.qq.com/s/r2zUtQfwH_r0WHrnY4CHuA>；要求确认 Agent Reach 可用
- **抓取：** Agent Reach v1.5.0 + `wechat-article-for-ai`（Camoufox；`--no-images`）；原始落盘 [`sources/raw/wechat_shenlan_robot_learning_five_paradigms_2026-08-05.md`](sources/raw/wechat_shenlan_robot_learning_five_paradigms_2026-08-05.md)
- **来源：** [`sources/blogs/wechat_shenlan_robot_learning_five_paradigms.md`](sources/blogs/wechat_shenlan_robot_learning_five_paradigms.md)（深蓝具身智能；2026-08-05）
- **新建对比页：** [`wiki/comparisons/robot-learning-five-paradigms-taxonomy.md`](wiki/comparisons/robot-learning-five-paradigms-taxonomy.md)
- **核心判断：** 按学习信号划分 IL / RL / LfV / VLA / 持续学习；工程上应组合选型，而非寻找万能算法；与五大具身模型 taxonomy（模型族 I/O）正交
- **开源核查（步骤 2.5）：** 科普综述文，无独立项目页；文内代表作沿用既有归档（ALOHA/ACT、DAgger、Isaac Gym、VideoDex、RT-2、Open X-Embodiment）
- **交叉更新：** [`wiki/methods/imitation-learning.md`](wiki/methods/imitation-learning.md)、[`wiki/methods/reinforcement-learning.md`](wiki/methods/reinforcement-learning.md)、[`wiki/methods/vla.md`](wiki/methods/vla.md)、[`wiki/comparisons/rl-vs-il.md`](wiki/comparisons/rl-vs-il.md)、[`wiki/comparisons/vlm-vln-vla-vlx-world-model-taxonomy.md`](wiki/comparisons/vlm-vln-vla-vlx-world-model-taxonomy.md)、[`wiki/concepts/sim2real.md`](wiki/concepts/sim2real.md)、[`wiki/overview/robot-learning-overview.md`](wiki/overview/robot-learning-overview.md)、[`sources/repos/panniantong_agent_reach.md`](sources/repos/panniantong_agent_reach.md)

## [2026-08-05] ingest | sources/papers/roamflow_arxiv_2606_29934.md — RoamFlow（arXiv:2606.29934）MeanFlow 一步 image-goal 导航入库；升格 wiki/entities/paper-roamflow.md；交叉 NoMaD/NavDP/NavWAM/Habitat/VLN/Orin NX

- **触发：** 用户指定 <https://arxiv.org/abs/2606.29934>
- **来源：** [`sources/papers/roamflow_arxiv_2606_29934.md`](sources/papers/roamflow_arxiv_2606_29934.md)（NTU；2026-06-29）
- **新建实体：** [`wiki/entities/paper-roamflow.md`](wiki/entities/paper-roamflow.md)
- **开源核查（步骤 2.5）：** **确认未开源** — 无项目页 / 无官方仓；`## 源码运行时序图` 不适用
- **方法要点：** MeanFlow 一步轨迹；IL→Habitat PPO；轨迹评估器；Gibson SR 68.7%/19.6 ms；Go2+Orin NX 真机 SR 1.00 / 37.2 ms
- **交叉更新：** [`wiki/entities/paper-notebook-nomad-goal-masked-diffusion-policies-for-navigat.md`](wiki/entities/paper-notebook-nomad-goal-masked-diffusion-policies-for-navigat.md)、[`wiki/entities/paper-notebook-navdp-learning-sim-to-real-navigation-diffusion.md`](wiki/entities/paper-notebook-navdp-learning-sim-to-real-navigation-diffusion.md)、[`wiki/entities/paper-navwam-goal-conditioned-visual-navigation-wam.md`](wiki/entities/paper-navwam-goal-conditioned-visual-navigation-wam.md)、[`wiki/entities/habitat-sim.md`](wiki/entities/habitat-sim.md)、[`wiki/entities/jetson-orin-nx.md`](wiki/entities/jetson-orin-nx.md)、[`wiki/tasks/vision-language-navigation.md`](wiki/tasks/vision-language-navigation.md)、[`wiki/overview/vln-open-source-repro-paradigms.md`](wiki/overview/vln-open-source-repro-paradigms.md)

## [2026-08-05] ingest | sources/papers/synthetic_video_humanoid_tasks_arxiv_2607_21648.md — NCKU 合成视频人形任务（arXiv:2607.21648）入库；升格 wiki/entities/paper-synthetic-video-humanoid-tasks.md；交叉 GenHOI/OASIS/Imagine2Real/loco-manip-02/DeepMimic/GMR/unitree-g1

- **触发：** 用户指定论文 *Learning Diverse Humanoid Tasks via Synthetic Video Scenarios without Real World Data*（NCKU；Yun-Hao Tsai / Cong-Thanh Vu / Yen-Chen Liu）
- **来源：** [`sources/papers/synthetic_video_humanoid_tasks_arxiv_2607_21648.md`](sources/papers/synthetic_video_humanoid_tasks_arxiv_2607_21648.md)（arXiv:2607.21648，2026-07-22）
- **新建实体：** [`wiki/entities/paper-synthetic-video-humanoid-tasks.md`](wiki/entities/paper-synthetic-video-humanoid-tasks.md)
- **机构注册：** `schema/institutions.json` 新增 `ncku` → 国立成功大学（National Cheng Kung University）
- **开源核查（步骤 2.5）：** **确认未开源** — 无项目页；arXiv Code 区无官方仓；作者 GitHub 未见对应实现；`## 源码运行时序图` 写不适用
- **方法要点：** Veo 3/3.1 提示生成视频 → SMPL-X → GMR → motion stitching → Isaac Lab DeepMimic 式 PPO；G1 仿真；关节 MAE 0.04–0.07 m；0.5 kg 负载
- **交叉更新：** [`wiki/entities/paper-loco-manip-03-genhoi.md`](wiki/entities/paper-loco-manip-03-genhoi.md)、[`wiki/entities/paper-loco-manip-04-oasis.md`](wiki/entities/paper-loco-manip-04-oasis.md)、[`wiki/entities/paper-imagine2real-zero-shot-hoi.md`](wiki/entities/paper-imagine2real-zero-shot-hoi.md)、[`wiki/overview/loco-manip-category-02-synthetic-data.md`](wiki/overview/loco-manip-category-02-synthetic-data.md)、[`wiki/overview/loco-manip-contact-category-03-generative-data.md`](wiki/overview/loco-manip-contact-category-03-generative-data.md)、[`wiki/methods/deepmimic.md`](wiki/methods/deepmimic.md)、[`wiki/methods/motion-retargeting-gmr.md`](wiki/methods/motion-retargeting-gmr.md)、[`wiki/entities/unitree-g1.md`](wiki/entities/unitree-g1.md)

## [2026-08-05] ingest | sources/papers/teleopit_arxiv_2608_01834.md + sites/teleopit-project + repos/teleopit/somehand — Teleopit（arXiv:2608.01834）全身体人形遥操作入库；升格 wiki/entities/paper-teleopit.md；交叉 wiki/tasks/teleoperation.md、wiki/entities/paper-twist2.md、paper-heft.md、paper-loco-manip-04-oasis.md、mimiclite.md

- **触发：** 用户指定论文 <https://arxiv.org/abs/2608.01834>、项目页 <https://botrunner64.github.io/teleopit-page/>
- **来源：** [`sources/papers/teleopit_arxiv_2608_01834.md`](sources/papers/teleopit_arxiv_2608_01834.md)、[`sources/sites/teleopit-project.md`](sources/sites/teleopit-project.md)、[`sources/repos/teleopit.md`](sources/repos/teleopit.md)、[`sources/repos/somehand.md`](sources/repos/somehand.md)
- **新建实体：** [`wiki/entities/paper-teleopit.md`](wiki/entities/paper-teleopit.md)（西湖大学 / 上海创智学院；PICO VR 全身+连续灵巧手+主动视觉；History Encoder + failure-aware rewind；五仓开源）
- **开源核查（步骤 2.5）：** **已开源** — 项目页列出 Teleopit / somehand / pico-bridge / OpenNeck / lerobot-teleopit；主仓含 `ckpt/track_g1*.onnx`
- **方法要点：** mjlab+PPO 全身跟踪；归一化指方向跨手重定向；OpenNeck 2-DoF；异步录制→ACT/GR00T（96 demos → 90%/95% SR）
- **交叉更新：** [`wiki/tasks/teleoperation.md`](wiki/tasks/teleoperation.md)、[`wiki/entities/paper-twist2.md`](wiki/entities/paper-twist2.md)、[`wiki/entities/paper-heft.md`](wiki/entities/paper-heft.md)、[`wiki/entities/paper-loco-manip-04-oasis.md`](wiki/entities/paper-loco-manip-04-oasis.md)、[`wiki/entities/mimiclite.md`](wiki/entities/mimiclite.md)、[`wiki/methods/sonic-motion-tracking.md`](wiki/methods/sonic-motion-tracking.md)、[`wiki/entities/unitree-g1.md`](wiki/entities/unitree-g1.md)、[`wiki/concepts/motion-retargeting.md`](wiki/concepts/motion-retargeting.md)

## [2026-08-04] lint | scripts/lint_wiki.py + wiki/entities/paper-actfovea.md — 全量健康检查：阻塞型 0、信息型 3 → 0

- **触发：** 全量知识库 lint 巡检，要求把信息型预警一并收敛
- **基线：** `python3 scripts/lint_wiki.py` 阻塞型 **0**、信息型 **3**（缺页概念候选 2 + 感知栈回链缺失 1）
- **内容补链：** [`wiki/entities/paper-actfovea.md`](wiki/entities/paper-actfovea.md) 回链 [`wiki/queries/robot-perception-stack-selection-loop.md`](wiki/queries/robot-perception-stack-selection-loop.md)（frontmatter `related` + 关联页面），定位为感知栈**第④层下游策略消费**的运行时侧：感知栈选完后，「感知输出 = 策略可信输入」这一抽象被遮挡 / 延迟 / 冻结重放破坏时如何检出与兜底
- **误报收敛（缺页概念巡检 V1）：** `sequenceDiagram` 入 `MISSING_CONCEPT_STOPWORDS` — 7 处行内引用均为 Mermaid 图类型关键字（论文实体页「源码运行时序图」写作约定，如「代码发布后应补 `sequenceDiagram`」），属文档语法 token，与 `md` / `http` 同类；`qwen3-vl` 入 `MISSING_CONCEPT_COVERED_ELSEWHERE` — 外部通用 VLM 产品型号，与 DINOv2 / SigLIP / Wan2.2 / Qwen2.5-VL 同类，本库一律在 [`wiki/methods/vla.md`](wiki/methods/vla.md)、[`wiki/methods/star-vla.md`](wiki/methods/star-vla.md)、[`wiki/concepts/foundation-policy.md`](wiki/concepts/foundation-policy.md) 按「底座」维度记述，不单建概念页
- **门禁：** `make ci-preflight` 通过（lint 阻塞型 0 + 信息型 0，导出质量 12/12）；`pytest` 全绿（覆盖率 62.44% ≥ 52%）、`ruff check` / `ruff format --check` / `mypy scripts` 干净；图谱 19041 → **19042 边**（本次新增回链）

## [2026-08-04] ingest | sources/blogs/zhihu_jagger_task_space_fb_bfm_intact_mimic_vla.md — 知乎专栏：FB/BFM-Zero/INTACT/Mimic/VLA 任务空间表征对照；升格 wiki/comparisons/fb-bfm-zero-intact-mimic-vla-task-space.md

- **触发：** 用户指定 <https://zhuanlan.zhihu.com/p/2066468645300180732>
- **来源：** [`sources/blogs/zhihu_jagger_task_space_fb_bfm_intact_mimic_vla.md`](sources/blogs/zhihu_jagger_task_space_fb_bfm_intact_mimic_vla.md)（作者 Jagger；2026-08-02/04；Jina 403 → Camoufox 抓取）
- **新建对比页：** [`wiki/comparisons/fb-bfm-zero-intact-mimic-vla-task-space.md`](wiki/comparisons/fb-bfm-zero-intact-mimic-vla-task-space.md)
- **核心判断：** 数据之外，目标函数定义域决定学到的是正交任务球（FB/BFM-Zero）、Goal-Reach 子空间（INTACT）、欧氏跟踪曲线（Mimic）或稀疏语义点（VLA）；并附 RL 相对 MPC 的接触平滑读法
- **开源核查（步骤 2.5）：** 博客对照文；所涉工作沿用既有归档——BFM-Zero / MimicLite / UFO **已开源**；INTACT 规范仓+镜像仍为文档仓（训练 Coming Soon）
- **交叉更新：** [`wiki/entities/paper-bfm-zero.md`](wiki/entities/paper-bfm-zero.md)、[`wiki/entities/paper-intact.md`](wiki/entities/paper-intact.md)、[`wiki/entities/mimiclite.md`](wiki/entities/mimiclite.md)、[`wiki/entities/roboparty-ufo.md`](wiki/entities/roboparty-ufo.md)、[`wiki/overview/roboparty-lab-party-os-technology-map.md`](wiki/overview/roboparty-lab-party-os-technology-map.md)、[`wiki/concepts/behavior-foundation-model.md`](wiki/concepts/behavior-foundation-model.md)、[`wiki/overview/bfm-category-01-forward-backward-representation.md`](wiki/overview/bfm-category-01-forward-backward-representation.md)、[`wiki/comparisons/mpc-vs-rl.md`](wiki/comparisons/mpc-vs-rl.md)、[`wiki/methods/vla.md`](wiki/methods/vla.md)、[`wiki/methods/sonic-motion-tracking.md`](wiki/methods/sonic-motion-tracking.md)

## [2026-08-04] ingest | sources/papers/clift_arxiv_2607_29172.md + sites/thomaschen98-clift — CLIFT（arXiv:2607.29172）复检加深：补每轮从基础模型从头微调 / 1.6s chunk / 100-trial 评测=训练数据 / GPT-5.5→Qwen3-VL；wiki/entities/paper-clift-closed-loop-iterative-finetuning.md

- **触发：** 用户指定论文 <https://arxiv.org/abs/2607.29172>、项目页 <https://thomaschen98.github.io/clift/>
- **既有页：** 2026-08-04 已完整 ingest（PR #1448）；本次**不新建实体**，做开源复检 + 工程细节加深
- **更新：** [`sources/papers/clift_arxiv_2607_29172.md`](sources/papers/clift_arxiv_2607_29172.md)、[`sources/sites/thomaschen98-clift.md`](sources/sites/thomaschen98-clift.md)、[`wiki/entities/paper-clift-closed-loop-iterative-finetuning.md`](wiki/entities/paper-clift-closed-loop-iterative-finetuning.md)
- **开源核查（复检 2026-08-04，步骤 2.5）：** 项目页 / 论文仍标 `coming_soon`，**无 GitHub** → 维持「宣称将开源」；`## 源码运行时序图` 仍写不适用
- **加深要点：** 托管 API **每轮从基础模型从头微调**；策略 **1.6 s** action chunk、优势前瞻 **1.8 s**；每任务 **100** 次 rollout 即评测亦为下一轮训练数据；候选奖励 **GPT-5.5** K=12 → 人类偏好筛选 → **Qwen3-VL** LoRA 蒸馏；DINOv3 ViT-S/16
- **交叉（既有，未改）：** [`wiki/entities/gemini-robotics.md`](wiki/entities/gemini-robotics.md)、[`wiki/entities/unitree-g1.md`](wiki/entities/unitree-g1.md)、[`wiki/concepts/safe-real-world-rl-fine-tuning.md`](wiki/concepts/safe-real-world-rl-fine-tuning.md)、[`wiki/overview/hub-safe-fine-tuning.md`](wiki/overview/hub-safe-fine-tuning.md)、[`wiki/tasks/bimanual-manipulation.md`](wiki/tasks/bimanual-manipulation.md)、[`wiki/concepts/reward-design.md`](wiki/concepts/reward-design.md)、[`wiki/methods/vla.md`](wiki/methods/vla.md)

## [2026-08-04] ingest | sources/papers/hifi_umi_arxiv_2607_25895.md + sites/hifi-umi-project + datasets/hifi-umi-2k — HiFi-UMI（arXiv:2607.25895）复检与接口补强：开源仍为数据-only；补 HF state/action/valid.frame；交叉 paper-data-pyramid；wiki/entities/paper-hifi-umi.md + paper-data-pyramid-embodied-manipulation.md

- **触发：** 用户指定论文 <https://arxiv.org/pdf/2607.25895>、项目页 <https://cloud.simpleai.tech/simple-world-lab/hifi-umi/>、数据集 <https://huggingface.co/datasets/simple-world-lab/HiFi-UMI-2K>
- **既有页：** 2026-07-30 已完整 ingest（PR #1367）；本次**不新建实体**，做开源复检 + HF 训练接口补强 + 金字塔交叉
- **更新：** [`sources/papers/hifi_umi_arxiv_2607_25895.md`](sources/papers/hifi_umi_arxiv_2607_25895.md)、[`sources/sites/hifi-umi-project.md`](sources/sites/hifi-umi-project.md)、[`sources/datasets/hifi-umi-2k.md`](sources/datasets/hifi-umi-2k.md)、[`wiki/entities/paper-hifi-umi.md`](wiki/entities/paper-hifi-umi.md)、[`wiki/entities/paper-data-pyramid-embodied-manipulation.md`](wiki/entities/paper-data-pyramid-embodied-manipulation.md)
- **开源核查（复检 2026-08-04，步骤 2.5）：** **部分开源（数据）** — HF 仍公开；项目页 / GitHub 检索仍无采数或训练仓
- **接口补强：** 20 维绝对 next-state、`valid.frame`、六视角 key、世界系约定、丢帧/夹爪误差；VLA 协议约 3200 UMI vs 300 teleop
- **交叉：** 数据金字塔 UMI 层表与「UMI-only 后训练」读法

## [2026-08-04] ingest | sources/repos/habitat-sim.md + sites/aihabitat-org + sites/aihabitat-habitat-sim-docs — Habitat-Sim 官方仓/门户/文档入库：加深 wiki/entities/habitat-sim.md

- **来源：** [`sources/repos/habitat-sim.md`](sources/repos/habitat-sim.md)、[`sources/sites/aihabitat-org.md`](sources/sites/aihabitat-org.md)、[`sources/sites/aihabitat-habitat-sim-docs.md`](sources/sites/aihabitat-habitat-sim-docs.md)
- **加深实体：** [`wiki/entities/habitat-sim.md`](wiki/entities/habitat-sim.md)（MIT；Sim↔Lab 分层；>10k FPS / >8k SPS；conda `withbullet`；源码运行时序图）
- **开源核查（步骤 2.5）：** GitHub MIT **已开源**；场景数据集需单独许可；README 声明 **Beyond v0.3.4 Meta 不再官方主动维护**
- **纠错：** Habitat 1.0 arXiv 由误链 `1904.11121` 更正为 [`1904.01201`](https://arxiv.org/abs/1904.01201)
- **交叉：** [`wiki/overview/sim-platforms-decade-technology-map.md`](wiki/overview/sim-platforms-decade-technology-map.md)、[`wiki/entities/matterport3d-simulator.md`](wiki/entities/matterport3d-simulator.md)、[`wiki/entities/pybullet.md`](wiki/entities/pybullet.md)、[`wiki/tasks/vision-language-navigation.md`](wiki/tasks/vision-language-navigation.md)、[`wiki/concepts/sim2real.md`](wiki/concepts/sim2real.md)

## [2026-08-04] ingest | sources/papers/light_loco_parkour_light_origins_2026.md + sites/light-loco-parkour-github-io — Light-Loco-Parkour（Light Origins, 2026-08-03）入库：稀疏种子 Real2Sim2Real + 多专家蒸馏无技能标签深度跑酷；升格 wiki/entities/paper-light-loco-parkour.md

- **来源：** [`sources/papers/light_loco_parkour_light_origins_2026.md`](sources/papers/light_loco_parkour_light_origins_2026.md)、[`sources/sites/light-loco-parkour-github-io.md`](sources/sites/light-loco-parkour-github-io.md)
- **新建实体：** [`wiki/entities/paper-light-loco-parkour.md`](wiki/entities/paper-light-loco-parkour.md)（Light Origins；Lightbot 0 / 90 cm / 21 DoF；攀爬至 0.83H；踏石 99.9%；无 arXiv）
- **方法要点：** Object-Interaction Mimic 物理修复 → 课程抬障（climb 45→75 cm）→ 多专家 DAgger → transition-group RL（去则 0%）→ GRU 深度蒸馏 + FT（IsaacLab / Orin Nano 50 Hz）
- **开源核查（步骤 2.5）：** 项目页仅 PDF/视频；GitHub 组织仅为 github.io → **确认未开源**；`## 源码运行时序图` 写明不适用
- **机构注册：** [`schema/institutions.json`](schema/institutions.json) 新增 `light-origins`（光原点（Light Origins））
- **交叉：** [`wiki/entities/paper-hrl-stack-22-perceptive_humanoid_parkour.md`](wiki/entities/paper-hrl-stack-22-perceptive_humanoid_parkour.md)、[`wiki/tasks/humanoid-locomotion.md`](wiki/tasks/humanoid-locomotion.md)、[`wiki/tasks/locomotion.md`](wiki/tasks/locomotion.md)、[`wiki/tasks/stair-obstacle-perceptive-locomotion.md`](wiki/tasks/stair-obstacle-perceptive-locomotion.md)、[`wiki/entities/paper-deep-whole-body-parkour.md`](wiki/entities/paper-deep-whole-body-parkour.md)、[`wiki/methods/dagger.md`](wiki/methods/dagger.md)、[`wiki/comparisons/hil-vs-mtrg-vs-zest-parkour-imitation.md`](wiki/comparisons/hil-vs-mtrg-vs-zest-parkour-imitation.md)

## [2026-08-04] ingest | sources/papers/legged_robots_advances_challenges_scirobotics_2026.md — 补 arXiv:2607.28952 作者版；加深 wiki/entities/paper-legged-robots-advances-challenges.md

- **触发：** 用户指定 <https://arxiv.org/abs/2607.28952>（Science Robotics 腿式综述；ETH / Stanford / Berkeley / Edinburgh / KAIST / NVIDIA / Tübingen / MPI-IS / Oxford / Monash / RAI）
- **既有页：** 2026-07-31 已 ingest，当时无合法全文；本次**不新建实体**，改为补 arXiv 作者版并核验数字
- **更新：** [`sources/papers/legged_robots_advances_challenges_scirobotics_2026.md`](sources/papers/legged_robots_advances_challenges_scirobotics_2026.md)、[`wiki/entities/paper-legged-robots-advances-challenges.md`](wiki/entities/paper-legged-robots-advances-challenges.md)
- **开源核查：** 仍无代码/项目页（综述）；全文改判 **作者版 OA（arXiv）**
- **机构注册：** [`schema/institutions.json`](schema/institutions.json) 新增 `rai-institute`（机器人与人工智能研究所（RAI Institute））
- **交叉（既有）：** [`wiki/tasks/locomotion.md`](wiki/tasks/locomotion.md)、[`wiki/entities/quadruped-robot.md`](wiki/entities/quadruped-robot.md)、[`wiki/concepts/sim2real.md`](wiki/concepts/sim2real.md)

## [2026-08-04] ingest | sources/courses/quadruped_vln_embodied_workshop_2day.md — 四足×VLN×具身实战营技术点/项目独立节点补齐

- **触发：** 用户提供课程日程截图，要求技术点与对应项目在本库均有独立详情节点
- **Sources：** [`sources/courses/quadruped_vln_embodied_workshop_2day.md`](sources/courses/quadruped_vln_embodied_workshop_2day.md)；TravExplorer [`sources/papers/travexplorer_arxiv_2605_19958.md`](sources/papers/travexplorer_arxiv_2605_19958.md) / [`sources/sites/wuyi2121-travexplorer.md`](sources/sites/wuyi2121-travexplorer.md) / [`sources/repos/travexplorer.md`](sources/repos/travexplorer.md)；SAM3 [`sources/papers/sam3_arxiv_2511_16719.md`](sources/papers/sam3_arxiv_2511_16719.md) / [`sources/repos/sam3.md`](sources/repos/sam3.md)；BLIP-2 [`sources/papers/blip2_arxiv_2301_12597.md`](sources/papers/blip2_arxiv_2301_12597.md) / [`sources/repos/lavis-blip2.md`](sources/repos/lavis-blip2.md)；OpenClaw [`sources/sites/openclaw-ai.md`](sources/sites/openclaw-ai.md) / [`sources/repos/openclaw.md`](sources/repos/openclaw.md)；Orin NX [`sources/sites/nvidia-jetson-orin-nx.md`](sources/sites/nvidia-jetson-orin-nx.md)
- **新建 wiki：**
  - 总览 [`wiki/overview/quadruped-vln-embodied-workshop.md`](wiki/overview/quadruped-vln-embodied-workshop.md)（日程→节点覆盖表）
  - 实体 [`wiki/entities/paper-travexplorer.md`](wiki/entities/paper-travexplorer.md)、[`wiki/entities/openclaw.md`](wiki/entities/openclaw.md)、[`wiki/entities/paper-sam3.md`](wiki/entities/paper-sam3.md)、[`wiki/entities/paper-blip2.md`](wiki/entities/paper-blip2.md)、[`wiki/entities/jetson-orin-nx.md`](wiki/entities/jetson-orin-nx.md)
  - 任务 [`wiki/tasks/zero-shot-object-navigation.md`](wiki/tasks/zero-shot-object-navigation.md)
  - 概念 [`wiki/concepts/embodied-semantic-cognitive-map.md`](wiki/concepts/embodied-semantic-cognitive-map.md)、[`wiki/concepts/vision-language-feature-fusion.md`](wiki/concepts/vision-language-feature-fusion.md)、[`wiki/concepts/lidar-sensing.md`](wiki/concepts/lidar-sensing.md)
- **已有复用：** VLN、Habitat、四足、分层导航栈、DWA/动态避障、状态估计、Sim2Real、SAM/SAM2 等
- **开源核查：** TravExplorer **占位待发布**；SAM3 / BLIP-2 / OpenClaw **已开源**；Orin NX 为硬件产品页
- **机构：** 注册 `salesforce`（赛富时（Salesforce））；`sjtu` / `meta` / `nvidia` 已有
- **交叉：** Philia↔OpenClaw、SAM2↔SAM3、VLN↔ObjectNav/TravExplorer、Habitat↔语义地图/实战营总览

## [2026-08-04] ingest | sources/papers/why_action_chunking_improves_bc_corl2026.md + sites/action-chunking-github-io — Why Action Chunking Improves BC（CoRL 2026）入库：Delay / RDE 机制消融；升格 wiki/entities/paper-why-action-chunking-improves-bc.md

- **触发：** 用户指定项目页 <https://action-chunking.github.io/>
- **来源：** [`sources/papers/why_action_chunking_improves_bc_corl2026.md`](sources/papers/why_action_chunking_improves_bc_corl2026.md)、[`sources/sites/action-chunking-github-io.md`](sources/sites/action-chunking-github-io.md)
- **新建实体：** [`wiki/entities/paper-why-action-chunking-improves-bc.md`](wiki/entities/paper-why-action-chunking-improves-bc.md)（Polimi / UC Berkeley；CoRL 2026；暂无 arXiv）
- **方法要点：** 否定 temporal consistency / horizon reduction / representation learning 作为充分解释；主因是 **delayed policy**（\(a_t\mid o_{t-n}\)）与 **隐式集成**；**RDE** 部署在多数设定匹配标准 AC；显式集成进一步抬升（Transport 12.6%→41.5%）
- **开源核查（步骤 2.5）：** PDF / presentation 已发布；Code 与 arXiv 均 Coming soon → 判「宣称将开源」；`## 源码运行时序图` 写明**不适用**
- **机构注册：** [`schema/institutions.json`](schema/institutions.json) 新增 `polimi`（米兰理工大学（Politecnico di Milano））
- **交叉：** [`wiki/methods/action-chunking.md`](wiki/methods/action-chunking.md)、[`wiki/methods/behavior-cloning.md`](wiki/methods/behavior-cloning.md)、[`wiki/methods/diffusion-policy.md`](wiki/methods/diffusion-policy.md)、[`wiki/entities/libero-benchmark.md`](wiki/entities/libero-benchmark.md)

## [2026-08-04] ingest | sources/papers/clift_arxiv_2607_29172.md + sites/thomaschen98-clift — CLIFT（arXiv:2607.29172）入库：托管 SFT API 下的非侵入闭环迭代微调；升格 wiki/entities/paper-clift-closed-loop-iterative-finetuning.md

- **来源：** [`sources/papers/clift_arxiv_2607_29172.md`](sources/papers/clift_arxiv_2607_29172.md)、[`sources/sites/thomaschen98-clift.md`](sources/sites/thomaschen98-clift.md)
- **新建实体：** [`wiki/entities/paper-clift-closed-loop-iterative-finetuning.md`](wiki/entities/paper-clift-closed-loop-iterative-finetuning.md)（UC Berkeley / Google DeepMind / NVIDIA；Unitree G1 三任务两轮飞轮 100% / 98% / 96%）
- **方法要点：** 奖励模型 select-then-distill（VLM K=12 候选 → ~100 组人类成对偏好筛选 → Qwen3-VL 蒸馏）+ DINOv3 检索式 chunk 级优势 token（1.8 s 前瞻、同侪前 30% 标正）+ 累积数据集回投托管 SFT API
- **开源核查（步骤 2.5）：** 项目页列 Code 但**无可用链接**（`coming_soon`）→ 判「宣称将开源」；`## 源码运行时序图` 写明**不适用**及原因（真实门槛是 GROD API 访问权与 G1 真机）
- **交叉：** [`wiki/entities/gemini-robotics.md`](wiki/entities/gemini-robotics.md)、[`wiki/entities/unitree-g1.md`](wiki/entities/unitree-g1.md)、[`wiki/concepts/safe-real-world-rl-fine-tuning.md`](wiki/concepts/safe-real-world-rl-fine-tuning.md)、[`wiki/overview/hub-safe-fine-tuning.md`](wiki/overview/hub-safe-fine-tuning.md)、[`wiki/tasks/bimanual-manipulation.md`](wiki/tasks/bimanual-manipulation.md)、[`wiki/concepts/reward-design.md`](wiki/concepts/reward-design.md)、[`wiki/methods/vla.md`](wiki/methods/vla.md)、[`wiki/entities/paper-pi05-open-world-vla.md`](wiki/entities/paper-pi05-open-world-vla.md)

## [2026-08-04] ingest | sources/papers/wcm_world_critic_arxiv_2607_29613.md + repos/wcm-world-critic-model + sites/sylvestf-wcm-homepage — WCM（arXiv:2607.29613）入库：世界模型 Critic 修 VLA RL 的单帧价值估计错配；升格 wiki/entities/paper-wcm-world-critic-model.md

- **来源：** [`sources/papers/wcm_world_critic_arxiv_2607_29613.md`](sources/papers/wcm_world_critic_arxiv_2607_29613.md)、[`sources/repos/wcm-world-critic-model.md`](sources/repos/wcm-world-critic-model.md)、[`sources/sites/sylvestf-wcm-homepage.md`](sources/sites/sylvestf-wcm-homepage.md)
- **新建实体：** [`wiki/entities/paper-wcm-world-critic-model.md`](wiki/entities/paper-wcm-world-critic-model.md)（同济 / 上海创智 / 复旦；4 基准 149 任务；ManiSkill IND π₀ 38.4→84.4、π₀.₅ 47.0→91.9、OpenVLA-OFT 28.1→99.0）
- **方法要点：** LeJEPA critic 联合 `L_value + λ·L_pred + η·L_SIGReg`；K=3 历史最优、λ∈[0.3,0.5]；消融显示 λ=0 的 ViT critic **仍无效**——起作用的是世界建模目标本身；零价值消融 OOD 反超 Flow-SDE
- **开源核查（步骤 2.5）：** 代码 MIT 完整（四步 shell 脚本），权重/数据部分上 HF、其余「逐步开源」→ 判 **部分开源**；已画 `## 源码运行时序图`
- **机构注册：** [`schema/institutions.json`](schema/institutions.json) 新增 `tongji`（同济大学（Tongji））；`make test` 全绿
- **交叉：** [`wiki/methods/model-based-rl.md`](wiki/methods/model-based-rl.md)、[`wiki/methods/generative-world-models.md`](wiki/methods/generative-world-models.md)、[`wiki/comparisons/online-vs-offline-rl.md`](wiki/comparisons/online-vs-offline-rl.md)、[`wiki/entities/openvla.md`](wiki/entities/openvla.md)、[`wiki/methods/vla.md`](wiki/methods/vla.md)

## [2026-08-04] ingest | sources/papers/actfovea_arxiv_2607_29169.md + repos/actfovea — ActFovea（arXiv:2607.29169）入库：VLA 运行时安全防护；升格 wiki/entities/paper-actfovea.md

- **来源：** [`sources/papers/actfovea_arxiv_2607_29169.md`](sources/papers/actfovea_arxiv_2607_29169.md)、[`sources/repos/actfovea.md`](sources/repos/actfovea.md)
- **新建实体：** [`wiki/entities/paper-actfovea.md`](wiki/entities/paper-actfovea.md)（冻结 π₀ + LIBERO 四套件 40 任务 / 2000 episodes；视觉叠加 49.3%→90.3%，NRR 93.7%；延迟 +9.8 pp；动作漂移 +7.0 pp；冻结重放 100% 及时安全失败）
- **方法要点：** 动作条件中央凹（接触圆盘 + 运动走廊）+ 一致性风险分（几何 / 动态 / 时间 / 动作–本体）+ 候选库与动作块验证 + 两级仲裁受限执行 / hold latch 安全失败
- **反例读点：** 纯时间戳 hold 在 3 帧视觉延迟下成功率 **0%**；固定裁剪/平滑无扰动即掉 10.8 pp；消融显示空间恢复靠候选构造、时间与动作侧恢复靠动作块验证
- **开源核查（步骤 2.5）：** [SunnyYWD/ActFovea](https://github.com/SunnyYWD/ActFovea) Apache-2.0 完整实现（openpi 改造）→ 判 **已开源**；权重复用 π₀ 官方 checkpoint；已画 `## 源码运行时序图`
- **交叉：** [`wiki/concepts/safety-filter.md`](wiki/concepts/safety-filter.md)、[`wiki/concepts/robot-safety-state-machine.md`](wiki/concepts/robot-safety-state-machine.md)、[`wiki/queries/vla-deployment-guide.md`](wiki/queries/vla-deployment-guide.md)、[`wiki/entities/libero-benchmark.md`](wiki/entities/libero-benchmark.md)、[`wiki/entities/paper-pi05-open-world-vla.md`](wiki/entities/paper-pi05-open-world-vla.md)、[`wiki/methods/vla.md`](wiki/methods/vla.md)

## [2026-08-03] ingest | sources/repos/sensenova-u1.md + sites/huggingface-sensenova-u1-5-8b-mot-preview.md + sites/modelscope-sensenova-u1-5-8b-mot-preview.md — SenseNova-U1.5 Preview 原生统一多模态模型；升格 wiki/entities/sensenova-u1-5.md

- **触发：** 用户指定三处一手入口 <https://github.com/OpenSenseNova/SenseNova-U1/blob/main/docs/u1.5_preview.md>、<https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT-Preview>、<https://modelscope.cn/models/SenseNova/SenseNova-U1.5-8B-MoT-Preview>
- **Sources：** [`sources/repos/sensenova-u1.md`](sources/repos/sensenova-u1.md)、[`sources/sites/huggingface-sensenova-u1-5-8b-mot-preview.md`](sources/sites/huggingface-sensenova-u1-5-8b-mot-preview.md)、[`sources/sites/modelscope-sensenova-u1-5-8b-mot-preview.md`](sources/sites/modelscope-sensenova-u1-5-8b-mot-preview.md)
- **Wiki：** [`wiki/entities/sensenova-u1-5.md`](wiki/entities/sensenova-u1-5.md)
- **交叉：** [`wiki/entities/sensenova-skills.md`](wiki/entities/sensenova-skills.md)、[`wiki/concepts/generative-vision-pretraining.md`](wiki/concepts/generative-vision-pretraining.md)（新增谱系 E：原生统一预训练）、[`wiki/methods/unified-multimodal-tokens.md`](wiki/methods/unified-multimodal-tokens.md)
- **机构：** 已有 `sensenova`（商汤科技（SenseNova）），无需新注册
- **开源：** **已开源（Apache-2.0）** — 推理 `examples/{t2i,editing}/inference.py`、U1.5 生成侧预训练 `training/shell/train_u1/U1.5_8B.sh`、全参微调代码与权重全部公开；**训练数据未开源**，**U1.5 技术报告未发布**（现有 arXiv 2605.12500 对应 U1）
- **核对要点：** `8B-MoT` 指「≈8B 理解 + ≈8B 生成」，官方脚本实测总参 **17.552B**（bf16 载入 ~35.1 GB，盘上 ~50.2 GB）；HF/ModelScope 分片后缀 `-of-00016` 但实际仅 13 个文件（`00002`–`00004` 缺号），`model.safetensors.index.json` 未引用缺号分片 → 权重完整，非下载失败；`max_pixels 16777216` 恰为 4096×4096，对应「原生 4K」口径
- **口径提醒：** 评测带 † 的 55.17/55.22 含模型外部 PE 改写器，裸分为 49.93/50.25；编辑侧 WeEdit BP 子项反而低于 U1（6.752 vs 7.157）

## [2026-08-03] ingest | sources/papers/worldscape_policy_2_arxiv_2607_18840.md + sites/manifoldai-research-worldscape-policy.md + repos/worldscape-policy.md — WorldScape Policy 2.0 推理增强长短期记忆 WAM；升格 wiki/entities/paper-worldscape-policy-2.md

- **触发：** 用户指定论文 <https://arxiv.org/abs/2607.18840> 与仓库 <https://github.com/manifoldai-research/WorldScape-Policy>
- **Sources：** [`sources/papers/worldscape_policy_2_arxiv_2607_18840.md`](sources/papers/worldscape_policy_2_arxiv_2607_18840.md)、[`sources/sites/manifoldai-research-worldscape-policy.md`](sources/sites/manifoldai-research-worldscape-policy.md)、[`sources/repos/worldscape-policy.md`](sources/repos/worldscape-policy.md)
- **Wiki：** [`wiki/entities/paper-worldscape-policy-2.md`](wiki/entities/paper-worldscape-policy-2.md)
- **交叉：** [`wiki/concepts/world-action-models.md`](wiki/concepts/world-action-models.md)、[`wiki/overview/wm-action-consequence-category-01-wam-action-prediction.md`](wiki/overview/wm-action-consequence-category-01-wam-action-prediction.md)、[`wiki/entities/paper-worldscape-moe-heterogeneous-action.md`](wiki/entities/paper-worldscape-moe-heterogeneous-action.md)
- **机构：** 已有 `tsinghua`、`sjtu`；`manifold` 沿用 Worldscape-MoE 页既有 tag（`schema/institutions.json` 未注册，本次未改注册表）
- **开源：** **宣称将开源 / 待发布** — GitHub 仅 `README.md` + `.gitignore`（"Code is coming soon"），HF `WorldScape-Policy-2` 仅模型卡（"Model is coming soon"，声明 Apache-2.0）；ManipEvent-5M 自采部分无发布计划 → 实体页 `## 源码运行时序图` 标注 **不适用**
- **要点：** 事件记忆（VLM，global-history / local-active / event-boundary 三视图 + 门控）与视觉记忆（DiT，近 4 chunk causal prefill）分层；semantic forcing（\(\lambda_s=0.001\)、T5 靶 stop-grad）把事件字幕语义蒸馏进隐式子目标；RoboTwin 2.0 标准榜 **94.3%** 已饱和（同档仅 +0.2~+0.7），差异在 **C2R 47.9%**（Fast-WAM 39.1）与真机视觉提示任务（叠积木目标图/演示 **60%/70%** vs π₀.₅ 10%/20%）

## [2026-08-03] ingest | sources/papers/humanoid_leg_generative_design_hust_j_260645.md — 动力学仿真驱动人形下肢衍生式设计；升格 wiki/entities/paper-humanoid-leg-generative-design-dynamics.md

- **触发：** 用户指定学报页 <http://xb.hust.edu.cn/thesisDetails#10.13245/j.hust.260645&lang=zh>（DOI `10.13245/j.hust.260645`）
- **Sources：** [`sources/papers/humanoid_leg_generative_design_hust_j_260645.md`](sources/papers/humanoid_leg_generative_design_hust_j_260645.md)
- **Wiki：** [`wiki/entities/paper-humanoid-leg-generative-design-dynamics.md`](wiki/entities/paper-humanoid-leg-generative-design-dynamics.md)
- **交叉：** [`wiki/concepts/humanoid-mechanical-layout-design.md`](wiki/concepts/humanoid-mechanical-layout-design.md)、[`wiki/overview/humanoid-hardware-101-chassis-materials.md`](wiki/overview/humanoid-hardware-101-chassis-materials.md)、[`wiki/overview/humanoid-actuator-102-load-and-mass-spiral.md`](wiki/overview/humanoid-actuator-102-load-and-mass-spiral.md)、[`roadmap/depth-humanoid-hardware-design.md`](roadmap/depth-humanoid-hardware-design.md)、[`wiki/concepts/planetary-roller-screw-humanoid-leg-actuation.md`](wiki/concepts/planetary-roller-screw-humanoid-leg-actuation.md)、[`wiki/queries/humanoid-hardware-selection.md`](wiki/queries/humanoid-hardware-selection.md)
- **机构：** 已有 `zju`；新注册 `avic-facri`（中航工业西安飞行自动控制研究所（AVIC FACRI））
- **开源：** **确认未开源** — 学报 PDF 可下载，无 GitHub/CAD/数据集

## [2026-08-03] ingest | sources/papers/robo_harness_arxiv_2607_18060.md + sites/robo-harness-com.md + repos/robo-harness.md — RoboHarness 异构策略编排；升格 wiki/entities/paper-robo-harness.md

- **触发：** 用户指定项目主页 <https://robo-harness.com> 与论文 <https://arxiv.org/abs/2607.18060>
- **Sources：** [`sources/papers/robo_harness_arxiv_2607_18060.md`](sources/papers/robo_harness_arxiv_2607_18060.md)、[`sources/sites/robo-harness-com.md`](sources/sites/robo-harness-com.md)、[`sources/repos/robo-harness.md`](sources/repos/robo-harness.md)
- **Wiki：** [`wiki/entities/paper-robo-harness.md`](wiki/entities/paper-robo-harness.md)
- **交叉：** [`wiki/methods/vla.md`](wiki/methods/vla.md)、[`wiki/entities/paper-harness-vla.md`](wiki/entities/paper-harness-vla.md)、[`wiki/concepts/behavior-tree-vla-orchestration.md`](wiki/concepts/behavior-tree-vla-orchestration.md)、[`wiki/entities/paper-gap-graph-as-policy.md`](wiki/entities/paper-gap-graph-as-policy.md)、[`wiki/overview/vla-open-source-repro-landscape-2025.md`](wiki/overview/vla-open-source-repro-landscape-2025.md)、[`wiki/entities/paper-pi05-open-world-vla.md`](wiki/entities/paper-pi05-open-world-vla.md)
- **机构：** 注册 `mcgill`、`utoronto`、`2012-labs`；已有 `huawei`、`ubc`
- **开源：** **部分开源（占位仓）** — 项目页 Code → `markli1hoshipu/RoboHarness`，仓内为静态站镜像，无可运行 harness；底层 π₀.₅ / OpenVLA-OFT 权重另开源

## [2026-08-03] ingest | sources/sites/anthropic-model-context-protocol.md + modelcontextprotocol-io.md — MCP 协议一手资料；升格 wiki/concepts/model-context-protocol.md

- **触发：** 用户要求查找并 ingest MCP 相关一手资料（此前库内仅有 FreeCAD/UE 等应用桥，缺协议层）
- **Sources：** [`sources/sites/anthropic-model-context-protocol.md`](sources/sites/anthropic-model-context-protocol.md)（2024-11-25 Anthropic 开源公告）、[`sources/sites/modelcontextprotocol-io.md`](sources/sites/modelcontextprotocol-io.md)（官方文档 / Spec `2026-07-28`）、[`sources/repos/modelcontextprotocol.md`](sources/repos/modelcontextprotocol.md)（规范仓 + org SDK/Inspector/servers 索引）
- **Wiki：** [`wiki/concepts/model-context-protocol.md`](wiki/concepts/model-context-protocol.md)
- **交叉：** [`wiki/entities/freecad-mcp.md`](wiki/entities/freecad-mcp.md)、[`wiki/entities/drawio-scientific-illustrator.md`](wiki/entities/drawio-scientific-illustrator.md)、[`wiki/entities/dimensionalos-dimos.md`](wiki/entities/dimensionalos-dimos.md)、[`wiki/entities/hermes-agent.md`](wiki/entities/hermes-agent.md)、[`wiki/entities/unreal-engine-5.md`](wiki/entities/unreal-engine-5.md)、[`wiki/entities/unreal-mcp.md`](wiki/entities/unreal-mcp.md)、[`wiki/concepts/remote-procedure-call.md`](wiki/concepts/remote-procedure-call.md)
- **机构：** 注册 [`schema/institutions.json`](schema/institutions.json) `anthropic`（人类智能（Anthropic））
- **开源：** **已开源** — 规范/SDK/参考 servers 公开；规范仓许可正从 MIT 过渡到 Apache-2.0

## [2026-08-03] ingest | sources/sites/unreal-mcp-in-unreal-editor.md — Unreal MCP（UE 5.8 Experimental）编辑器内嵌 MCP server

- **触发：** 用户指定 <https://dev.epicgames.com/documentation/unreal-engine/unreal-mcp-in-unreal-editor>
- **Sources：** [`sources/sites/unreal-mcp-in-unreal-editor.md`](sources/sites/unreal-mcp-in-unreal-editor.md)、[`sources/repos/unreal-engine-skills-for-claude-code-plugin.md`](sources/repos/unreal-engine-skills-for-claude-code-plugin.md)
- **Wiki：** [`wiki/entities/unreal-mcp.md`](wiki/entities/unreal-mcp.md)
- **交叉：** [`wiki/entities/unreal-engine-5.md`](wiki/entities/unreal-engine-5.md)、[`wiki/entities/freecad-mcp.md`](wiki/entities/freecad-mcp.md)、[`sources/sites/unreal-engine-5-8-docs.md`](sources/sites/unreal-engine-5-8-docs.md)、[`sources/repos/epicgames-github-org.md`](sources/repos/epicgames-github-org.md)
- **机构：** `epic-games` 已注册
- **开源：** **部分开源** — 引擎内 `ModelContextProtocol*` / Toolset Registry 随 UE 许可与私有 UnrealEngine 源码；Claude Code 技能插件 **MIT 已开源**（`EpicGames/unreal-engine-skills-for-claude-code-plugin`）


## [2026-08-03] ingest | sources/sites/humanoid-robot-learning-paper-notebooks.md — 姊妹论文笔记站归档；补齐 26 个 wiki/overview 汇总页的 sources ingest 锚点

- **触发：** `make lint` 的 Sources 覆盖率长期停在 2022/2048 (99%)，缺口全部落在 `wiki/overview/` 汇总页
- **Sources：** [`sources/sites/humanoid-robot-learning-paper-notebooks.md`](sources/sites/humanoid-robot-learning-paper-notebooks.md) — 姊妹站 Humanoid Robot Learning Paper Notebooks（14 分类 / 站点自述 305 篇），此前是本库唯一未建归档的重要外部输入
- **开源：** **已开源** — GitHub 仓库公开，BSD-3-Clause；含 `progress.json` 与 `papers/PROGRESS.md` 两份进度清单（2026-08-03 核对）
- **分类页（14）：** [`wiki/overview/paper-notebook-category-01-foundational-rl.md`](wiki/overview/paper-notebook-category-01-foundational-rl.md)、[`wiki/overview/paper-notebook-category-02-motion-retargeting.md`](wiki/overview/paper-notebook-category-02-motion-retargeting.md)、[`wiki/overview/paper-notebook-category-03-high-impact-selection.md`](wiki/overview/paper-notebook-category-03-high-impact-selection.md)、[`wiki/overview/paper-notebook-category-04-loco-manipulation-and-wbc.md`](wiki/overview/paper-notebook-category-04-loco-manipulation-and-wbc.md)、[`wiki/overview/paper-notebook-category-05-locomotion.md`](wiki/overview/paper-notebook-category-05-locomotion.md)、[`wiki/overview/paper-notebook-category-06-manipulation.md`](wiki/overview/paper-notebook-category-06-manipulation.md)、[`wiki/overview/paper-notebook-category-07-teleoperation.md`](wiki/overview/paper-notebook-category-07-teleoperation.md)、[`wiki/overview/paper-notebook-category-08-navigation.md`](wiki/overview/paper-notebook-category-08-navigation.md)、[`wiki/overview/paper-notebook-category-09-state-estimation.md`](wiki/overview/paper-notebook-category-09-state-estimation.md)、[`wiki/overview/paper-notebook-category-10-sim-to-real.md`](wiki/overview/paper-notebook-category-10-sim-to-real.md)、[`wiki/overview/paper-notebook-category-11-simulation-benchmark.md`](wiki/overview/paper-notebook-category-11-simulation-benchmark.md)、[`wiki/overview/paper-notebook-category-12-hardware-design.md`](wiki/overview/paper-notebook-category-12-hardware-design.md)、[`wiki/overview/paper-notebook-category-13-physics-based-animation.md`](wiki/overview/paper-notebook-category-13-physics-based-animation.md)、[`wiki/overview/paper-notebook-category-14-human-motion.md`](wiki/overview/paper-notebook-category-14-human-motion.md) — `参考来源` 增加站点归档 + 本类论文 ingest 归档（节选 3 条）
- **知识链汇总页（12）：** [`wiki/overview/hub-contact-force-control.md`](wiki/overview/hub-contact-force-control.md)、[`wiki/overview/hub-data-pipeline.md`](wiki/overview/hub-data-pipeline.md)、[`wiki/overview/hub-learning.md`](wiki/overview/hub-learning.md)、[`wiki/overview/hub-locomotion.md`](wiki/overview/hub-locomotion.md)、[`wiki/overview/hub-motion-retargeting.md`](wiki/overview/hub-motion-retargeting.md)、[`wiki/overview/hub-physics-fidelity.md`](wiki/overview/hub-physics-fidelity.md)、[`wiki/overview/hub-safe-fine-tuning.md`](wiki/overview/hub-safe-fine-tuning.md)、[`wiki/overview/hub-state-estimation.md`](wiki/overview/hub-state-estimation.md)、[`wiki/overview/hub-vision-backbone.md`](wiki/overview/hub-vision-backbone.md)、[`wiki/overview/hub-vla.md`](wiki/overview/hub-vla.md)、[`wiki/overview/hub-wbc.md`](wiki/overview/hub-wbc.md)、[`wiki/overview/hub-wbt.md`](wiki/overview/hub-wbt.md) — `参考来源` 增加「上游原始资料」，链到本链概念页共同的 ingest 来源
- **口径说明：** 站点首页自述 305 篇、`schema/paper-notebook-index.json` 收录 289 条已成稿笔记、`schema/paper-notebook-categories.json` 合计 518 条清单项（含待深读与跨类重复计入），三者不等已在归档中写明
- **结果：** Sources 覆盖率 2022/2048 (99%) → **2048/2048 (100%)**；lint 阻塞型 0、信息型 0
## [2026-08-02] structural | docs/main.js — 详情页正文内链悬停浮窗，并与关联知识图谱迷你图双向联动

- **改动：** [`docs/main.js`](docs/main.js) — 正文中指向站内知识页的内链标记为 `.detail-inline-link`，悬停弹出与图谱同款 hover 卡片（类型徽标 + 标题 + 摘要 + 「打开详情页 →」）；类型与社区色优先取 `link-graph.json` 的细类型（concept / task / paper…），与迷你图浮窗同口径
- **联动：** 悬停正文内链 → 迷你图同一节点点亮（`.mini-node-linked`）；悬停迷你图节点 → 正文中指向它的内链点亮（`.detail-inline-link-linked`）；触屏（`hover: none`）不绑定，点击内链仍直接跳转
- **涉及路径：** [`docs/detail.html`](docs/detail.html)（新增 `#detail-inline-link-tooltip`）、[`docs/style.css`](docs/style.css)
- **清单：** [`docs/checklists/frontend-optimization-v1.md`](docs/checklists/frontend-optimization-v1.md) Phase 3「详情页浮窗联动」
- **验证：** [`scripts/verify_detail_inline_link_preview.cjs`](scripts/verify_detail_inline_link_preview.cjs)（`wiki-concepts-sim2real` → 浮窗可见 / 迷你图点亮 1 节点；反向 → 正文点亮 1 条内链）；`make ci-preflight` 通过

## [2026-08-02] structural | docs/graph.html — 筛选浮窗三区手风琴（按社区 / 路线视图 / 研究机构）

- **改动：** [`docs/graph.html`](docs/graph.html) — 「按社区（当前维度）/ 路线视图 / 研究机构」同时仅展开一个，默认展开「按社区」；展开区吃满剩余高度，收起项靠上或靠下；`?depth=` / `?community=` / `?institution=` 深链同步打开对应区
- **跟进：** Chromium 下对 `<details>` 设 flex/grid 会导致列表不被限高、溢出被裁切且无滑块；展开态改为 absolute 铺满 summary 下方，并以 `.filter-scroll-chrome` 常驻滑块替代不可见的原生 overlay 滚动条
- **清单：** [`docs/checklists/frontend-optimization-v1.md`](docs/checklists/frontend-optimization-v1.md)
- **验证：** [`scripts/verify_graph_filter_accordion.cjs`](scripts/verify_graph_filter_accordion.cjs)

## [2026-08-02] ingest | sources/sites/tinkercad-com.md — Autodesk Tinkercad；wiki/entities/tinkercad.md；交叉 wiki/entities/wokwi.md、wiki/entities/freecad.md、wiki/entities/kicad.md、wiki/entities/simplefoc.md、wiki/overview/motor-drive-firmware-bus-protocols.md

- **触发：** 用户指定 <https://www.tinkercad.com/>
- **Sources：** [`sources/sites/tinkercad-com.md`](sources/sites/tinkercad-com.md)
- **Wiki：** [`wiki/entities/tinkercad.md`](wiki/entities/tinkercad.md)
- **交叉：** [`wiki/entities/wokwi.md`](wiki/entities/wokwi.md)、[`wiki/entities/freecad.md`](wiki/entities/freecad.md)、[`wiki/entities/kicad.md`](wiki/entities/kicad.md)、[`wiki/entities/simplefoc.md`](wiki/entities/simplefoc.md)、[`wiki/overview/motor-drive-firmware-bus-protocols.md`](wiki/overview/motor-drive-firmware-bus-protocols.md)
- **机构：** `autodesk` 已注册
- **开源：** **确认未开源** — Autodesk 闭源 SaaS；免费可用，设计可导出 STL/OBJ 等

## [2026-08-02] ingest | sources/blogs/wechat_shenlan_six_spatial_representations_embodied_perception.md — 具身感知六种空间表征；wiki/concepts/embodied-perception-six-spatial-representations.md

- **触发：** 用户指定微信 <https://mp.weixin.qq.com/s/lWvdz9cjuurS7ikBkZk0vQ>（深蓝具身智能 · 六种空间表征分层）
- **抓取：** Agent Reach v1.5.0 + `wechat-article-for-ai`（已确认安装）；`--no-images`
- **Sources：** [`sources/blogs/wechat_shenlan_six_spatial_representations_embodied_perception.md`](sources/blogs/wechat_shenlan_six_spatial_representations_embodied_perception.md)、[`sources/raw/wechat_shenlan_six_spatial_representations_2026-08-02/`](sources/raw/wechat_shenlan_six_spatial_representations_2026-08-02/)
- **Wiki：** [`wiki/concepts/embodied-perception-six-spatial-representations.md`](wiki/concepts/embodied-perception-six-spatial-representations.md)
- **交叉：** [`2d-to-3d-semantic-lifting-gap`](wiki/concepts/2d-to-3d-semantic-lifting-gap.md)、[`robot-perception-stack-selection-loop`](wiki/queries/robot-perception-stack-selection-loop.md)、[`navigation-slam-autonomy-stack`](wiki/overview/navigation-slam-autonomy-stack.md)、[`isaac-ros-nvblox`](wiki/entities/isaac-ros-nvblox.md)、[`go2-3d-semantic-mapping-sam-pipeline`](wiki/queries/go2-3d-semantic-mapping-sam-pipeline.md)

## [2026-08-02] structural | docs/depth-filters.js — 修复路线视图多 token 片段漏匹配并补齐各纵深命中集

- **问题：** `nodeSegments` 按 `/._-` 切词后，`loco-manip` / `motion-generation` / `sim-to-real` 等带连字符的 segments 几乎永不命中；Loco-Manip 路线视图仅剩枢纽+任务页（约 2 节点），动作生成等路线同样偏空
- **修复：** `segmentHits` 对多 token segment 改为归一路径子串命中（`loco-manip` 可命中 `.../loco-manipulation...`）；单 token 仍精确匹配；并补齐 boxing / soccer / motion-generation / real2sim / wam 等路线的 segments 与显式 `ids`
- **涉及路径：** [`docs/depth-filters.js`](docs/depth-filters.js)、[`tests/test_depth_filters.py`](tests/test_depth_filters.py)
- **验证：** `pytest tests/test_depth_filters.py`；loco-manip 命名语料 129/129 命中

## [2026-08-02] ingest | sources/papers/learning_quiet_walking_aibo_arxiv_2502_10983.md — Sony aibo QuietWalk（ICRA 2025）低噪行走

- **触发：** 用户指定 *Learning Quiet Walking for a Small Home Robot*（arXiv:2502.10983；ETH / Sony / NUS 等；Watanabe / Miki / Shi / Hutter 等；ICRA 2025）
- **Sources：** [`sources/papers/learning_quiet_walking_aibo_arxiv_2502_10983.md`](sources/papers/learning_quiet_walking_aibo_arxiv_2502_10983.md)、[`sources/sites/sony-quietwalk-github-io.md`](sources/sites/sony-quietwalk-github-io.md)
- **Wiki：** [`wiki/entities/paper-learning-quiet-walking-aibo.md`](wiki/entities/paper-learning-quiet-walking-aibo.md)
- **交叉：** [`wiki/entities/paper-quietwalk-humanoid-locomotion.md`](wiki/entities/paper-quietwalk-humanoid-locomotion.md)、[`wiki/tasks/locomotion.md`](wiki/tasks/locomotion.md)、[`wiki/entities/quadruped-robot.md`](wiki/entities/quadruped-robot.md)、[`wiki/queries/locomotion-reward-design-guide.md`](wiki/queries/locomotion-reward-design-guide.md)、[`wiki/queries/legged-humanoid-rl-pd-gain-setting.md`](wiki/queries/legged-humanoid-rl-pd-gain-setting.md)、[`wiki/concepts/humanoid-policy-reward-functions.md`](wiki/concepts/humanoid-policy-reward-functions.md)、[`wiki/tasks/humanoid-locomotion.md`](wiki/tasks/humanoid-locomotion.md)、[`wiki/methods/disney-olaf-character-robot.md`](wiki/methods/disney-olaf-character-robot.md)、[`wiki/entities/paper-variable-impedance-contact-rl.md`](wiki/entities/paper-variable-impedance-contact-rl.md)
- **机构：** `eth` / `sony` / `nus` 已注册
- **开源：** **确认未开源可运行实现** — 项目页与 `sony/QuietWalk` 仅为静态展示仓

## [2026-08-02] ingest | sources/papers/legged_load_adapt_arxiv_2507_07825.md — Legged Load Adapt（arXiv:2507.07825，ZJU-UIUC）未知动态载荷四足适应

- **触发：** 用户指定项目页 <https://leixinjonaschang.github.io/leggedloadadapt.github.io/>；同时给出的 `arxiv.org/abs/2109.12343` 为同名起首 *Beyond Robustness* 多机器人韧性综述（Prorok et al.），与项目页无关——以项目页 BibTeX 校正为 **arXiv:2507.07825**（Chang / Nai / Chen / Yang · ZJU-UIUC）
- **Sources：** [`sources/papers/legged_load_adapt_arxiv_2507_07825.md`](sources/papers/legged_load_adapt_arxiv_2507_07825.md)、[`sources/sites/leggedloadadapt-github-io.md`](sources/sites/leggedloadadapt-github-io.md)
- **Wiki：** [`wiki/entities/paper-legged-load-adapt-unknown-dynamic-load.md`](wiki/entities/paper-legged-load-adapt-unknown-dynamic-load.md)
- **交叉：** [`wiki/concepts/privileged-training.md`](wiki/concepts/privileged-training.md)、[`wiki/tasks/locomotion.md`](wiki/tasks/locomotion.md)、[`wiki/concepts/terrain-adaptation.md`](wiki/concepts/terrain-adaptation.md)、[`wiki/entities/paper-rma-rapid-motor-adaptation.md`](wiki/entities/paper-rma-rapid-motor-adaptation.md)、[`wiki/entities/paper-splitadapter-load-aware-loco-manipulation.md`](wiki/entities/paper-splitadapter-load-aware-loco-manipulation.md)
- **机构：** `zju` / `uiuc` / `unitree` 已注册
- **开源：** **宣称将开源 / 待发布** — 项目页 Code (coming soon)，无独立仓库

## [2026-08-02] ingest | sources/papers/mmhu_arxiv_2507_12463.md + sites/mmhu-benchmark-github-io.md — MMHU（arXiv:2507.12463，TAMU / Brown / JHU / UT Austin）驾驶人体行为多模态基准

- **触发：** 用户指定 *MMHU: A Massive-Scale Multimodal Benchmark for Human Behavior Understanding*（arXiv:2507.12463；Texas A&M / Brown / Johns Hopkins / UT Austin；Li / Ye / Wu / Yang / Fan / Hu / Tu）
- **Sources：** [`sources/papers/mmhu_arxiv_2507_12463.md`](sources/papers/mmhu_arxiv_2507_12463.md)、[`sources/sites/mmhu-benchmark-github-io.md`](sources/sites/mmhu-benchmark-github-io.md)
- **Wiki：** [`wiki/entities/paper-mmhu.md`](wiki/entities/paper-mmhu.md)
- **交叉：** [`wiki/overview/autonomous-driving-core-algorithms-series.md`](wiki/overview/autonomous-driving-core-algorithms-series.md)、[`wiki/overview/e2e-autonomous-driving-top10-algorithms.md`](wiki/overview/e2e-autonomous-driving-top10-algorithms.md)、[`wiki/overview/hub-embodied-eval-benchmark.md`](wiki/overview/hub-embodied-eval-benchmark.md)、[`wiki/methods/diffusion-motion-generation.md`](wiki/methods/diffusion-motion-generation.md)、[`wiki/concepts/3d-spatial-vqa.md`](wiki/concepts/3d-spatial-vqa.md)
- **机构：** 注册 [`schema/institutions.json`](schema/institutions.json) `texas-am` / `brown`；`jhu` / `ut-austin` 已有
- **开源：** **部分开源** — HF Dataset [`jerryye0110/MMHU`](https://huggingface.co/datasets/jerryye0110/MMHU) 已发；项目页截至入库日 **未列 GitHub / 训练代码**


## [2026-08-02] ingest | sources/papers/emergent_transfer_cross_config_arxiv_2607_25593.md — Emergent Transfer（arXiv:2607.25593，HUST / Spirit AI / PKU / SJTU / HIT / 清华）跨配置遗留示教三相迁移

- **触发：** 用户指定 *When Does Legacy Data Start to Help? Emergent Transfer in Cross-Configuration Robot Learning*（arXiv:2607.25593；华中科技大学 / 千寻智能 / 北京大学 / 上海交通大学 / 哈尔滨工业大学 / 清华大学；Wang / Hou / Hu / Gao 等）
- **Sources：** [`sources/papers/emergent_transfer_cross_config_arxiv_2607_25593.md`](sources/papers/emergent_transfer_cross_config_arxiv_2607_25593.md)
- **Wiki：** [`wiki/entities/paper-emergent-transfer-cross-config.md`](wiki/entities/paper-emergent-transfer-cross-config.md)
- **交叉：** [`wiki/overview/hub-cross-embodiment.md`](wiki/overview/hub-cross-embodiment.md)、[`wiki/queries/cross-embodiment-transfer-strategy.md`](wiki/queries/cross-embodiment-transfer-strategy.md)、[`wiki/queries/humanoid-training-data-pipeline.md`](wiki/queries/humanoid-training-data-pipeline.md)、[`wiki/entities/paper-pi05-open-world-vla.md`](wiki/entities/paper-pi05-open-world-vla.md)、[`wiki/methods/behavior-cloning.md`](wiki/methods/behavior-cloning.md)
- **机构：** `hust` / `spirit-ai` / `pku` / `sjtu` / `hit` / `tsinghua` 已注册
- **开源：** **确认未开源** — 项目入口即为 arXiv；无独立项目页 / GitHub

## [2026-08-02] ingest | sources/papers/fa_rdp_arxiv_2607_28596.md + sites/fa-rdp-github-io.md — FA-RDP（arXiv:2607.28596，SJTU / 创智 / Noematrix）频率自适应反应扩散

- **触发：** 用户指定 *FA-RDP: A Frequency-Adaptive Reactive Diffusion Policy for Contact-Rich Manipulation*（arXiv:2607.28596；SJTU / Shanghai Innovation Institute / Noematrix；Zhuo / Chen / Xue / Tang / Lv / Lu / Wen）+ 项目页
- **Sources：** [`sources/papers/fa_rdp_arxiv_2607_28596.md`](sources/papers/fa_rdp_arxiv_2607_28596.md)、[`sources/sites/fa-rdp-github-io.md`](sources/sites/fa-rdp-github-io.md)
- **Wiki：** [`wiki/entities/paper-fa-rdp.md`](wiki/entities/paper-fa-rdp.md)
- **交叉：** [`wiki/methods/diffusion-policy.md`](wiki/methods/diffusion-policy.md)、[`wiki/concepts/contact-rich-manipulation.md`](wiki/concepts/contact-rich-manipulation.md)、[`wiki/tasks/manipulation.md`](wiki/tasks/manipulation.md)
- **机构：** 注册 [`schema/institutions.json`](schema/institutions.json) `noematrix`（诺玛矩阵（Noematrix））；`sjtu` / `shanghai-innovation-institute` 已有
- **开源：** **未开源（coming soon）** — 项目页 Code 为 `href="#"`；`zhuolifeng/FA-RDP` 仅为站点源 + Releases 视频

## [2026-08-02] ingest | sources/papers/world_action_planner_arxiv_2607_27599.md + sites/worldactionplanner-github-io.md + repos/world-action-planner.md — World Action Planner（arXiv:2607.27599，Harvard）深度升格

- **触发：** 用户指定 *World Action Planner: Generalizable Decision-Making with Action-Conditioned World Models*（arXiv:2607.27599；Harvard；Zhang / Du）+ GitHub / 项目页 / Hugging Face
- **Sources：** [`sources/papers/world_action_planner_arxiv_2607_27599.md`](sources/papers/world_action_planner_arxiv_2607_27599.md)、[`sources/sites/worldactionplanner-github-io.md`](sources/sites/worldactionplanner-github-io.md)、[`sources/repos/world-action-planner.md`](sources/repos/world-action-planner.md)、[`sources/sites/huggingface-xiangchengzhang-world-action-planner.md`](sources/sites/huggingface-xiangchengzhang-world-action-planner.md)
- **Wiki：** [`wiki/entities/paper-world-action-planner.md`](wiki/entities/paper-world-action-planner.md)
- **交叉：** [`wiki/concepts/world-action-models.md`](wiki/concepts/world-action-models.md)、[`wiki/methods/generative-world-models.md`](wiki/methods/generative-world-models.md)、[`wiki/methods/vla.md`](wiki/methods/vla.md)、[`wiki/methods/diffusion-policy.md`](wiki/methods/diffusion-policy.md)、[`wiki/entities/libero-benchmark.md`](wiki/entities/libero-benchmark.md)、[`wiki/tasks/manipulation.md`](wiki/tasks/manipulation.md)
- **机构：** `harvard` 已注册
- **开源：** **已开源** — GitHub 训练/服务 + HF 世界模型与 DP/IDM 权重

## [2026-08-02] ingest | sources/papers/refine_dp_arxiv_2603_13707.md + sites/refine-dp-github-io.md — REFINE-DP（arXiv:2603.13707，Georgia Tech / IEEE RA-L）深度升格

- **触发：** 用户指定 *REFINE-DP: Diffusion Policy Fine-tuning for Humanoid Loco-manipulation via Reinforcement Learning*（IEEE RA-L；Georgia Tech；Gu / Chen / Chai / Zhao 等）
- **Sources：** [`sources/papers/refine_dp_arxiv_2603_13707.md`](sources/papers/refine_dp_arxiv_2603_13707.md)、[`sources/sites/refine-dp-github-io.md`](sources/sites/refine-dp-github-io.md)；同步策展槽位 [`loco_manip_161_survey_157_refine-dp.md`](sources/papers/loco_manip_161_survey_157_refine-dp.md)
- **Wiki：** [`wiki/entities/paper-loco-manip-161-157-refine-dp.md`](wiki/entities/paper-loco-manip-161-157-refine-dp.md)（161 #157 原地升格完整实体）
- **交叉：** [`wiki/tasks/loco-manipulation.md`](wiki/tasks/loco-manipulation.md)、[`wiki/methods/diffusion-policy.md`](wiki/methods/diffusion-policy.md)、[`wiki/methods/reinforcement-learning.md`](wiki/methods/reinforcement-learning.md)、[`wiki/methods/residual-policy-learning.md`](wiki/methods/residual-policy-learning.md)、[`wiki/overview/loco-manip-161-category-09-vla-world-models.md`](wiki/overview/loco-manip-161-category-09-vla-world-models.md)、[`wiki/entities/paper-doorman-opening-sim2real-door.md`](wiki/entities/paper-doorman-opening-sim2real-door.md)
- **机构：** `georgia-tech` 已注册
- **开源：** **未开源** — 项目页 Code 为无链接占位；`REFINE-DP/REFINE-DP` 仅为站点源

## [2026-08-01] ingest | sources/blogs/wechat_shenlan_overseas_embodied_labs_43_2026.md — 海外 43 所具身实验室地图；wiki/overview/overseas-embodied-ai-labs-landscape-2026.md

- **触发：** 用户指定微信 <https://mp.weixin.qq.com/s/_zoU9Q-KXHJAUZ041iBuCw>（深蓝具身智能 · 2026 海外 43 所，姊妹国内篇）
- **抓取：** Agent Reach v1.5.0 + `wechat-article-for-ai`（已确认安装）；`--no-images`
- **Sources：** [`sources/blogs/wechat_shenlan_overseas_embodied_labs_43_2026.md`](sources/blogs/wechat_shenlan_overseas_embodied_labs_43_2026.md)、[`sources/raw/wechat_shenlan_overseas_embodied_labs_43_2026-08-01/`](sources/raw/wechat_shenlan_overseas_embodied_labs_43_2026-08-01/)
- **Wiki：** [`wiki/overview/overseas-embodied-ai-labs-landscape-2026.md`](wiki/overview/overseas-embodied-ai-labs-landscape-2026.md)
- **交叉：** [`china-embodied-ai-labs-landscape-2026`](wiki/overview/china-embodied-ai-labs-landscape-2026.md)、[`notable-commercial-robot-platforms`](wiki/overview/notable-commercial-robot-platforms.md)、[`diffusion-policy`](wiki/methods/diffusion-policy.md)、[`aloha`](wiki/entities/aloha.md)、[`π0-policy`](wiki/methods/π0-policy.md)、[`anymal`](wiki/entities/anymal.md)、[`vla-open-source-repro-landscape-2025`](wiki/overview/vla-open-source-repro-landscape-2025.md)

## [2026-08-01] ingest | sources/blogs/wechat_shenlan_china_embodied_labs_50_2026.md — 国内 50 所具身实验室三层地图；wiki/overview/china-embodied-ai-labs-landscape-2026.md

- **触发：** 用户指定微信 <https://mp.weixin.qq.com/s/58c4CgN9XVmtS_RMKbqeKw>（深蓝具身智能 · 国内 50 所盘点，附全景图）
- **抓取：** Agent Reach v1.5.0 + `wechat-article-for-ai`（已确认安装）；`--no-images`
- **Sources：** [`sources/blogs/wechat_shenlan_china_embodied_labs_50_2026.md`](sources/blogs/wechat_shenlan_china_embodied_labs_50_2026.md)、[`sources/raw/wechat_shenlan_china_embodied_labs_50_2026-07-26/`](sources/raw/wechat_shenlan_china_embodied_labs_50_2026-07-26/)
- **Wiki：** [`wiki/overview/china-embodied-ai-labs-landscape-2026.md`](wiki/overview/china-embodied-ai-labs-landscape-2026.md)
- **交叉：** [`overseas-embodied-ai-labs-landscape-2026`](wiki/overview/overseas-embodied-ai-labs-landscape-2026.md)、[`notable-commercial-robot-platforms`](wiki/overview/notable-commercial-robot-platforms.md)、[`agibot-world-2026`](wiki/entities/agibot-world-2026.md)、[`x-humanoid`](wiki/entities/x-humanoid.md)、[`limx-cosa`](wiki/entities/limx-cosa.md)、[`robot-learning-overview`](wiki/overview/robot-learning-overview.md)

## [2026-08-01] ingest | sources/papers/pac_man_perceptive_cbf_rl_arxiv_2607_28623.md — PAC-MAN 感知感知 CBF-RL 人形躲避球

- **触发：** 用户指定论文 *PAC-MAN: Perception-Aware CBF-RL for Whole-Body Safety in Humanoid Dodgeball*（arXiv:2607.28623；Caltech AMBER；Yang / Li / Ames）+ 项目页 / 浏览器 Demo / GitHub
- **Sources：** [`sources/papers/pac_man_perceptive_cbf_rl_arxiv_2607_28623.md`](sources/papers/pac_man_perceptive_cbf_rl_arxiv_2607_28623.md)、[`sources/sites/perceptive-cbf-rl-github-io.md`](sources/sites/perceptive-cbf-rl-github-io.md)、[`sources/repos/perceptive_cbf_rl.md`](sources/repos/perceptive_cbf_rl.md)
- **Wiki：** [`wiki/entities/paper-pac-man-perceptive-cbf-rl.md`](wiki/entities/paper-pac-man-perceptive-cbf-rl.md)
- **交叉：** [`wiki/concepts/control-barrier-function.md`](wiki/concepts/control-barrier-function.md)、[`wiki/methods/safe-rl.md`](wiki/methods/safe-rl.md)、[`wiki/concepts/safety-filter.md`](wiki/concepts/safety-filter.md)、[`wiki/concepts/privileged-training.md`](wiki/concepts/privileged-training.md)、[`wiki/methods/amp-reward.md`](wiki/methods/amp-reward.md)、[`wiki/entities/unitree-g1.md`](wiki/entities/unitree-g1.md)、[`wiki/entities/amp-mjlab.md`](wiki/entities/amp-mjlab.md)、[`wiki/entities/mjlab.md`](wiki/entities/mjlab.md)
- **机构：** `caltech` 已注册
- **开源：** **已开源（MIT）** — 项目页列 Code；训练 + benchmark + `deploy/` 硬件栈与 ONNX；浏览器 Demo 可玩

## [2026-08-01] ingest | sources/papers/importance_sampling_pca_av_failures_arxiv_2607_18106.md — Stanford×Torc AST/DiFS+PCA 商业卡车 AV 稀有失败挖掘

- **触发：** 用户指定论文 *Importance Sampling and PCA for Finding Failures in Commercial Autonomous Vehicles*（arXiv:2607.18106；IEEE ICVES 2026 submitted；Stanford × Torc；Kochenderfer 等）
- **Sources：** [`sources/papers/importance_sampling_pca_av_failures_arxiv_2607_18106.md`](sources/papers/importance_sampling_pca_av_failures_arxiv_2607_18106.md)
- **Wiki：** [`wiki/entities/paper-importance-sampling-pca-av-failures.md`](wiki/entities/paper-importance-sampling-pca-av-failures.md)
- **交叉：** [`wiki/methods/safe-rl.md`](wiki/methods/safe-rl.md)、[`wiki/methods/sac.md`](wiki/methods/sac.md)、[`wiki/concepts/safety-filter.md`](wiki/concepts/safety-filter.md)、[`wiki/concepts/diffusion-model.md`](wiki/concepts/diffusion-model.md)、[`wiki/overview/autonomous-driving-core-algorithms-series.md`](wiki/overview/autonomous-driving-core-algorithms-series.md)、[`wiki/concepts/robot-safety-state-machine.md`](wiki/concepts/robot-safety-state-machine.md)
- **机构：** 注册 [`schema/institutions.json`](schema/institutions.json) `torc`（托克机器人（Torc Robotics））；`stanford` 已有
- **开源：** **确认未开源**（无项目页/GitHub；商业卡车规划栈 + Object Sim 黑盒）

## [2026-08-01] structural | media/site-demo.gif — 重录 README 演示 GIF，纳入入口卡顺时针描边

- **脚本：** [`scripts/record_readme_demo.cjs`](scripts/record_readme_demo.cjs)（88 frames / 3.48 MB；图谱 **2060** 节点 / **18500** 边）
- **变更：** 首页段改为真实点击「项目查询 / 知识图谱」入口卡，录制滚动落点后的顺时针描框特效；字幕同步为 ①–⑩
- **流程：** `make export graph` → `docs/` 本地 `http.server 8765` → 重录并写回 `media/site-demo.gif`
- **清单：** [`docs/checklists/frontend-optimization-v1.md`](docs/checklists/frontend-optimization-v1.md)

## [2026-08-01] fix | docs/mini-graph.js — 首页预览剔除 Top-N 诱导子图孤儿并回填

- **问题：** 预览取全站度数 Top-50 后只保留诱导边；如「机器人视觉感知栈选型闭环」全局度数高但邻居都不在 Top-50，会以孤立点漂浮
- **修复：** 剔除诱导度数为 0 的节点，再按度数序回填能连上当前集合的候选，保持约 50 且无孤儿
- **清单：** [`docs/checklists/frontend-optimization-v1.md`](docs/checklists/frontend-optimization-v1.md)

## [2026-08-01] fix | docs/main.js — 详情页关联迷你图：按规模优先邻居 + 近=重要弹簧

- **问题：** 1-hop 邻居按中文 label 字母序截断 12，重要大节点可能被裁掉；弹簧距离/强度对所有邻居均一，无法表达层级
- **修复：** 按全图度数降序取 Top-16（≤16 全显示）；`forceLink` 距离/强度随邻居半径变化（大邻居更近、吸力更强）；meta 文案标明「近=重要」；完整列表仍走 `graph.html?focus=`
- **清单：** [`docs/checklists/frontend-optimization-v1.md`](docs/checklists/frontend-optimization-v1.md)

## [2026-08-01] fix | docs/main.js — 首页「项目查询 / 知识图谱」入口卡改回窗口顶端对齐

- **问题：** 与 Hero 路线数字共用 `block: 'center'` 后，点击入口卡会把搜索区 / 图谱预览滚到视口中心，不再像旧锚点那样顶对齐
- **修复：** 入口卡滚动锚到 `#wiki-search` / `#mini-graph-section`，`scrollIntoView({ block: 'start' })`；描边仍画在 `#wiki-search-panel` / `#mini-graph-wrap`；Hero「主路线 / 纵深路线」保持居中
- **清单：** [`docs/checklists/frontend-optimization-v1.md`](docs/checklists/frontend-optimization-v1.md)

## [2026-08-01] ingest | sources/sites/disney-research-la.md + sources/sites/disney-research-la-holotile.md — Disney Research LA 门户与 Holotile；升格 wiki/entities/disney-research-la.md、wiki/entities/disney-holotile.md

- **触发：** 用户给出 <https://la.disneyresearch.com/holotile/>、<https://la.disneyresearch.com/research/>
- **Sources：** [`sources/sites/disney-research-la.md`](sources/sites/disney-research-la.md)、[`sources/sites/disney-research-la-holotile.md`](sources/sites/disney-research-la-holotile.md)
- **Wiki：** [`wiki/entities/disney-research-la.md`](wiki/entities/disney-research-la.md)（三大方向 + 出版物映射枢纽）、[`wiki/entities/disney-holotile.md`](wiki/entities/disney-holotile.md)（全向地板；专利 US10416754B2；确认未开源）
- **交叉：** [`disney-olaf-character-robot`](wiki/methods/disney-olaf-character-robot.md)、[`reactor-physics-aware-motion-retargeting`](wiki/methods/reactor-physics-aware-motion-retargeting.md)、[`character-animation-vs-robotics`](wiki/concepts/character-animation-vs-robotics.md)、[`open-duck-mini`](wiki/entities/open-duck-mini.md)、[`locomotion`](wiki/tasks/locomotion.md)
- **开源：** Holotile / 门户级角色硬件 **未开源**；复现对照走 Open Duck；多数 publication 仅 PDF

## [2026-08-01] structural | media/site-demo.gif — 按最新图谱重录 README 演示 GIF

- **脚本：** [`scripts/record_readme_demo.cjs`](scripts/record_readme_demo.cjs)（70 frames / 3.12 MB；图谱 **2058** 节点 / **18475** 边）
- **流程：** `make export graph` → `docs/` 本地 `http.server 8765` → 录制首页入口 / 搜索 / 迷你图谱 / 全图悬停·缩放·侧栏 / 3D 切换
- **引用：** [`README.md`](README.md)「在线演示」仍指向 `media/site-demo.gif`

## [2026-08-01] fix | scripts/utils/community_labels.py — 为 llm-wiki-karpathy 枢纽补 COMMUNITY_NAME_OVERRIDES，修复 Tests community-17 命名断言

- **原因：** video-shotcraft ingest 后 Agent Skills 簇枢纽落在 `wiki/references/llm-wiki-karpathy.md`，H1「LLM Wiki」不符合 `中文（English） 社区` 正则
- **修复：** override → `大语言模型维基（LLM Wiki）`；同步 `graph-stats` / `link-graph` 导出
- **验证：** `pytest tests/test_community_naming.py`

## [2026-07-31] ingest | sources/repos/video-shotcraft.md — 接入 Vincentwei1021/video-shotcraft（Trendshift 上榜 AI 动效技能库）；升格 wiki/entities/video-shotcraft.md

- **触发：** 用户给出 `trendshift/video-shotcraft`（**404**）；核实为 [Trendshift](https://trendshift.io/repositories/88911) 榜单徽章，官方仓为 [`Vincentwei1021/video-shotcraft`](https://github.com/Vincentwei1021/video-shotcraft)
- **Sources：** [`sources/repos/video-shotcraft.md`](sources/repos/video-shotcraft.md)、[`sources/sites/video-shotcraft-gallery.md`](sources/sites/video-shotcraft-gallery.md)
- **Wiki：** [`wiki/entities/video-shotcraft.md`](wiki/entities/video-shotcraft.md)（104 镜头卡 / 161 样片 / Ink Press 模板、三种模式、八阶段流水线、开源与 Remotion 许可边界）
- **交叉：** [`gsap-skills`](wiki/entities/gsap-skills.md)、[`mattpocock-skills`](wiki/entities/mattpocock-skills.md)、[`img2threejs`](wiki/entities/img2threejs.md)、[`manim`](wiki/entities/manim.md)、[`sensenova-skills`](wiki/entities/sensenova-skills.md)
- **开源：** Apache-2.0 技能库 + Gallery Pages；Remotion / Mixkit 另有条款


## [2026-07-31] fix | docs/main.js + docs/style.css — 首页「项目查询」点击卡顿：立刻聚焦、近中心即描边、静默预取搜索索引

- **问题：** 点击「项目查询」后需等 `scrollend`（或 700ms fallback）才聚焦/播描边；同时 focus 同步写「加载中…」并拉取解析大体积 `search-index.json`，主线程卡一下再跳
- **修复：** 点击立刻 `focus({preventScroll})`；rAF 检测接近视口中心即 `playCardBorderTrace`；空查询静默预取；`pointerdown`/`mouseenter`/idle 预取索引；描边布局推迟到下一帧；去掉描边 `drop-shadow`
- **清单：** [`docs/checklists/frontend-optimization-v1.md`](docs/checklists/frontend-optimization-v1.md)

## [2026-07-31] fix | docs/graph-3d.js — 3D 社区漂浮标签按画布分辨率连续缩放，避免过小/过大

- **问题：** 旧逻辑在移动端/粗指针上字号×0.55 且相机 zoom 再×0.55，手机/平板有效字号可落到 ~3–5px；桌面端字号固定 8–16px，900–2560 宽度几乎不变，大屏相对过小
- **修复：** 以画布短边相对 ~800px 做 √ 连续缩放（约 0.78–1.28），绝对字号钳制 7–22px；zoom 钳制随短边略调；窄屏仅收紧内边距
- **验证：** [`scripts/verify_graph_community_labels_3d_responsive.cjs`](scripts/verify_graph_community_labels_3d_responsive.cjs)
- **清单：** [`docs/checklists/frontend-optimization-v1.md`](docs/checklists/frontend-optimization-v1.md)

## [2026-07-31] ingest | sources/blogs/wechat_robot_lecture_legged_robots_survey_2026-07-31.md — Agent Reach 抓取机器人大讲堂导读加深 Frey et al. SciRobotics 腿式综述；更新 wiki/entities/paper-legged-robots-advances-challenges.md（DSL/可反驱/价格/四项政策）；raw sources/raw/wechat_robot_lecture_legged_robots_survey_2026-07-31/

## [2026-07-31] ingest | sources/papers/legged_robots_advances_challenges_scirobotics_2026.md — OA/PDF 全路径复查仍 closed（Unpaywall/OpenAlex/PMC/arXiv/Zenodo/IA/作者站/ETH RC）；更新 wiki/entities/paper-legged-robots-advances-challenges.md 局限与核查表

## [2026-07-31] ingest | sources/papers/legged_robots_advances_challenges_scirobotics_2026.md — Frey et al. Science Robotics 2026 腿式机器人五柱综述；升格 wiki/entities/paper-legged-robots-advances-challenges.md；交叉 wiki/tasks/locomotion.md、wiki/entities/quadruped-robot.md、wiki/entities/paper-bioinspired-multimodal-robotics.md、wiki/concepts/sim2real.md；通稿 sources/blogs/techxplore_legged_robots_ethics_monash_2026-07-30.md；注册 edinburgh/monash

## [2026-07-31] ingest | sources/papers/humoslope_arxiv_2607_07830.md — HumoSlope（arXiv:2607.07830）NTU/A*STAR 两阶段坡面 locomotion；升格 wiki/entities/paper-humoslope-physics-guided-slope-locomotion.md；交叉更新 wiki/tasks/humanoid-locomotion.md、wiki/concepts/lip-zmp.md、wiki/concepts/terrain-adaptation.md、wiki/concepts/privileged-training.md、wiki/entities/unitree-g1.md、wiki/entities/unitree-rl-lab.md；注册 schema/institutions.json astar

## [2026-07-31] structural | docs/index.html + docs/main.js — 「项目查询 / 知识图谱」入口卡跳转目标模块顺时针描边

- **项目查询：** 滚到 `#wiki-search-panel`（搜索栏）中心 → 边框顺时针描边一圈 → 聚焦搜索输入框
- **知识图谱：** 滚到 `#mini-graph-wrap`（图谱预览）中心 → 边框顺时针描边一圈（描边期间临时放开 `overflow:hidden` 以免裁切）
- **复用：** 与 Hero「主路线 / 纵深路线」同一套 `scrollEntryCardToCenter` + `playCardBorderTrace`；入口卡用 `data-trace-target`
- **清单：** [`docs/checklists/frontend-optimization-v1.md`](docs/checklists/frontend-optimization-v1.md)

## [2026-07-31] ingest | sources/repos/diffsheg.md — 接入 JeremyCJM/DiffSHEG（CVPR 2024）语音驱动整体 3D 表情+手势联合扩散；升格 wiki/entities/paper-diffsheg.md；交叉更新 wiki/methods/diffusion-motion-generation.md、wiki/concepts/diffusion-model.md、wiki/entities/paper-notebook-semantic-co-speech-gesture-synthesis-and-real-ti.md；sources/papers/diffsheg_arxiv_2401_04747.md、sources/sites/diffsheg.md

## [2026-07-31] ingest | sources/repos/spi-active.md — 接入 LeCAR-Lab/SPI-Active（CoRL 2025 Oral）采样式 SysID+主动探索；升格 wiki/entities/paper-notebook-sampling-based-system-identification-with-active.md；交叉更新 wiki/concepts/system-identification.md、wiki/methods/cma-es.md、wiki/queries/sim2real-gap-reduction.md、wiki/entities/paper-pace-sim2real-legged-robots.md；sources/papers/spi_active_arxiv_2505_14266.md、sources/sites/spi-active.md

## [2026-07-31] structural | docs/main.js + docs/style.css — 首页入口卡边框描边按 border-box 像素对齐，修复多分辨率错位

- **问题：** 描边 SVG 用 `offsetWidth`（border-box）写 viewBox，CSS 用 padding-box 的 `inset`/`%` 定尺寸，亚像素宽度下左右/上下外扩不对称，且 viewBox 被缩放导致圆角偏离卡片边框
- **修复：** JS 以 `getBoundingClientRect` + 边框宽度写入 `left/top/width/height`，对称外扩于 border-box；圆角取自 computed `border-radius`；动画期间 `ResizeObserver` 跟随卡片尺寸变化
- **清单：** [`docs/checklists/frontend-optimization-v1.md`](docs/checklists/frontend-optimization-v1.md)

## [2026-07-31] ingest | sources/repos/wan-dancer.md + sources/papers/wan_dancer_arxiv_2607_09581.md — Wan-Dancer 分钟级 music-to-dance；升格 wiki/entities/paper-wan-dancer.md；交叉 Wan / Wan-Move / generative-world-models / hub-wbt

- **触发：** 用户给出 `https://github.com/Wan-AI/Wan-Dancer-14B`（**404**）；核实为 HF/ModelScope 权重 ID，代码仓为 [`Wan-Video/Wan-Dancer`](https://github.com/Wan-Video/Wan-Dancer)
- **Sources：** [`sources/repos/wan-dancer.md`](sources/repos/wan-dancer.md)、[`sources/papers/wan_dancer_arxiv_2607_09581.md`](sources/papers/wan_dancer_arxiv_2607_09581.md)、[`sources/sites/wan-dancer-project.md`](sources/sites/wan-dancer-project.md)
- **Wiki：** [`wiki/entities/paper-wan-dancer.md`](wiki/entities/paper-wan-dancer.md)（分层 Global→Local、开源状态、源码运行时序图、结论）
- **交叉：** [`paper-wan-video`](wiki/entities/paper-wan-video.md)、[`paper-wan-move`](wiki/entities/paper-wan-move.md)、[`generative-world-models`](wiki/methods/generative-world-models.md)、[`hub-wbt`](wiki/overview/hub-wbt.md)、[`sources/repos/wan2.1.md`](sources/repos/wan2.1.md)
- **开源：** Apache-2.0 推理仓 + HF `Wan-AI/Wan-Dancer-14B`（global/local）+ 项目页；论文 arXiv:2607.09581

## [2026-07-31] ingest | sources/blogs/gemini_robotics_2_whole_body.md — Gemini Robotics 2 全身智能；升格 wiki/entities/gemini-robotics.md；交叉 foundation-policy / WBC / loco-manip / hub-cross-embodiment / vla

## [2026-07-31] structural | docs/style.css — 纵深路线描边后「展开全部…」文案高亮改为两次

## [2026-07-31] structural | docs/main.js — Hero 路线数字描边时长×2；跳转改为 scrollIntoView 居中

- **时长：** 边框顺时针描边 `1.05s → 2.1s`（`BORDER_TRACE_MS` 1200→2400）
- **滚动：** 主路线/纵深路线数字 `preventDefault` 后 `scrollIntoView({ block: 'center' })`，卡片落在视口垂直中心再播特效

## [2026-07-31] structural | docs/index.html — Hero 主路线/纵深路线改为锚到入口卡+顺时针描边；纵深不再自动展开，描边后高亮展开按钮

- **主路线数字：** `#home-start-main-route`（从零开始卡）+ 边框顺时针高亮一圈后移除
- **纵深路线数字：** `#home-more-routes` 同样描边，**不**自动展开；描边结束后短时高亮 `#homeRouteToggle` 文案
- **实现：** `docs/main.js` 注入 SVG `rect[pathLength=100]` + `stroke-dashoffset` 顺时针扫边；`pulseRouteToggleHint` 提醒展开按钮

## [2026-07-31] structural | docs/index.html — Hero 盘点新增主路线(1)，四项数字可点跳转图谱/主路线/展开纵深路线

- **改动：** `docs/index.html` Hero 盘点由三项改为四项：知识节点 / 互链关系 / **主路线(1)** / 纵深路线(21)；主路线插在互链关系与纵深路线之间
- **跳转：** 知识节点与互链关系数字 → `graph.html`；主路线数字 → `roadmap.html?id=roadmap-motion-control`；纵深路线数字 → `#home-more-routes` 并展开全部纵深路线（`docs/main.js` `setHomeRoutesExpanded`）
- **样式：** `docs/style.css` 为可点数字补 hover/focus，并放宽 `.hero-stats` 宽度以容纳四项
- **清单：** [`docs/checklists/frontend-optimization-v1.md`](docs/checklists/frontend-optimization-v1.md)
## [2026-07-31] query | wiki/queries/hmi-papers-coverage.md — 确保 HMI 论文与项目目录 145 篇论文均有本库独立详情节点；新建 23 个 sources+entities，复用 122 个已有页；修正开源主表 Robot Parkour / ASAP 挂接

- **触发：** 用户要求覆盖 [论文与项目](https://github.com/RealXiaoze/humanoid-motion-intelligence/tree/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE) 列出的论文与项目详情独立节点（不重复造页）
- **Query 产物：** [`wiki/queries/hmi-papers-coverage.md`](wiki/queries/hmi-papers-coverage.md)（145/145 映射；注册 [`wiki/queries/README.md`](wiki/queries/README.md)）
- **新建详情节点（21）+ 升格占位（2）：** 新建含 OSF、Stack of Tasks、ExBody、XHugWBC、AnyBody、PaLM-E、Open X、π0.5、Dreamer、GR00T-Dreams、Robot Parkour 等；P017/P018 升格原 Paper Notebooks planned 占位（避免 arXiv 双节点）— 对应 `sources/papers/hmi_p*.md`
- **复用与升格：** 含 PULSE（`pulse-physics.md`）、Challenging Terrain、Contact-Aided InEKF、Octo、Gemini Robotics 等；开源主表 Robot Parkour 改挂新页、ASAP 改挂 `paper-hrl-stack-25-asap`
- **交叉：** [`wiki/entities/humanoid-motion-intelligence.md`](wiki/entities/humanoid-motion-intelligence.md)、[`wiki/queries/hmi-opensource-projects-coverage.md`](wiki/queries/hmi-opensource-projects-coverage.md)、[`scripts/utils/community_labels.py`](scripts/utils/community_labels.py)
- **说明：** 开源项目主表 166 项此前已覆盖；本轮补齐论文侧缺口并修正两处错误挂接

## [2026-07-31] structural | 图谱「路线视图」纳入主路线 + 21 条策展纵深；原专题枢纽改为 hub-* 知识链

- **产品概念：** 图谱侧栏与详情徽标统一称「**路线视图** / **所属路线**」；路线集 = 主路线 [`roadmap/motion-control.md`](roadmap/motion-control.md) + 21 条纵深 [`roadmap/depth-*.md`](roadmap/)；取消独立「专题」概念
- **图谱：** [`docs/depth-filters.js`](docs/depth-filters.js) 与 `graph.html` chips：主路线置顶（命中集=主路线正文链出的 wiki 节点），其后 21 条纵深；详情页「所属路线」徽标同步
- **原专题枢纽：** `wiki/overview/topic-*` → `wiki/overview/hub-*`（称「知识链汇总」，不再叫纵深）；旧 `topic-*` / 误改的 `overview/depth-*` 详情 ID 写入 [`schema/page-aliases.json`](schema/page-aliases.json)
- **涉及路径：** `docs/depth-filters.js`、`docs/graph.html`、`docs/main.js`、`wiki/overview/hub-*.md`、`roadmap/depth-*.md`、`roadmap/motion-control.md`

## [2026-07-30] structural | schema/canonical-facts.json — V31 P2 事实库扩展 250 → 260 条，补 10 条感知栈选型矛盾检测规则

- **新增 10 条：** 单阶段检测快 vs 两阶段精度高、闭集检测准 vs 开放词汇泛化、实时机载算力受限 vs 服务器侧精度、2D 框够用 vs 必须 3D 语义几何、稠密语义建图信息全 vs 内存/时延、SAM 零样本分割强 vs 类别语义缺失、深度传感精度 vs 成本、在线建图实时 vs 离线建图完整、感知帧率高 ≠ 控制闭环带宽高、DETR 端到端简洁 vs 收敛慢/小目标弱
- **校验方式：** 每条 `terms`/`pos_claims` 逐条对现存感知栈页（[`robot-perception-stack-selection-loop`](wiki/queries/robot-perception-stack-selection-loop.md) / [`2d-to-3d-semantic-lifting-gap`](wiki/concepts/2d-to-3d-semantic-lifting-gap.md) / `object-detection-model-selection` 等）有 pos 命中；`neg_claims` 取朴素错误全句、不命中任何页（含未被 `strip_misconception_sections` 剥离的「误判速查」表），保证 0 误报
- **门禁：** `make lint` 0 errors、潜在矛盾 **0 个**；`make ci-preflight` **12/12**（`graph-stats.json` 0 orphans、`community_quality_warning: false`）
- 勾选 v31 P2「事实库扩展」及 DoD「事实库扩展至 260 条」；v31 P3（图谱路线视图 / 详情页徽标）待后续推进

## [2026-07-30] structural | media/site-demo.gif — 按最新图谱重录 README 演示 GIF

- **脚本：** [`scripts/record_readme_demo.cjs`](scripts/record_readme_demo.cjs)（70 frames / 3.21 MB；图谱 **2031** 节点 / **17987** 边）
- **流程：** `make export graph` → `docs/` 本地 `http.server 8765` → 录制首页入口 / 搜索 / 迷你图谱 / 全图悬停·缩放·侧栏 / 3D 切换
- **引用：** [`README.md`](README.md)「在线演示」仍指向 `media/site-demo.gif`

## [2026-07-30] fix | scripts/utils/community_labels.py — 为 HMI 本库导读页补 COMMUNITY_NAME_OVERRIDES，修复 Tests 社区命名断言

- **原因：** 枢纽 H1「HMI 开源项目主表 · 本库导读」不符合 `中文（English）` 社区命名正则（须以中文开头）
- **修复：** override → `开源项目本库导读（HMI Open-Source Projects Guide）`；同步 `graph-stats.json`

## [2026-07-30] structural | wiki/queries/hmi-opensource-projects-coverage.md — 覆盖索引改写为读者向「本库导读」

- **改动：** 去掉维护者黑话（新建/合并/ci-preflight/曾误建）；改为怎么用、规模说明、可读标题链接、同主题共用提示
- **交叉：** [`wiki/entities/humanoid-motion-intelligence.md`](wiki/entities/humanoid-motion-intelligence.md)、[`wiki/queries/README.md`](wiki/queries/README.md)

## [2026-07-30] structural | wiki — 续：撤销 genmimic 重复实体并修正 DreamWaQ 社区实现挂链

- **删除：** [`wiki/entities/genmimic.md`](wiki/entities/genmimic.md) → 合并入 [`wiki/entities/paper-hrl-stack-04-from_generated_human_videos_to_physi.md`](wiki/entities/paper-hrl-stack-04-from_generated_human_videos_to_physi.md)
- **修正：** 主表「DreamWaQ（社区实现）」由误挂 [`dreamwaq-plus`](wiki/entities/dreamwaq-plus.md) 改回 [`wiki/methods/dreamwaq.md`](wiki/methods/dreamwaq.md)
- **覆盖索引：** [`wiki/queries/hmi-opensource-projects-coverage.md`](wiki/queries/hmi-opensource-projects-coverage.md)（复用 120 / 合并 18 / 新建 46）

## [2026-07-30] structural | wiki — 撤销 13 个与已有节点重复的 HMI 新建实体，合并回 action-chunking / π0-policy / isaac-gym / behavior-1k / project-instinct / dreamwaq / roboparty / beyondmimic / agibot-bfm-2 / mujoco / ALMI 论文页；更新覆盖索引

- **原则：** 库内已有详情节点则只更新原页并挂 HMI 主表入口，禁止平行造页
- **覆盖索引：** [`wiki/queries/hmi-opensource-projects-coverage.md`](wiki/queries/hmi-opensource-projects-coverage.md)
- **删除实体：** act-aloha、openpi、isaac-gym-envs、omnigibson、instinctlab、instinct-rl、instinct-onboard、legged-lab-dwaq、roboparty-train、beyondmimic-reproduction、motion-between-bfm-2、mujoco-menagerie、almi-open（sources 保留并改指 canonical）

## [2026-07-30] query | wiki/queries/hmi-opensource-projects-coverage.md — 确保 HMI 开源项目主表 166 项均有本库独立详情节点；新建 60 个 sources+entities，复用 106 个已有页

- **触发：** 用户要求覆盖 [开源项目主表](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E5%BC%80%E6%BA%90%E9%A1%B9%E7%9B%AE%E4%B8%BB%E8%A1%A8.md) 全部项目
- **Query 产物：** [`wiki/queries/hmi-opensource-projects-coverage.md`](wiki/queries/hmi-opensource-projects-coverage.md)（166/166 映射；注册 [`wiki/queries/README.md`](wiki/queries/README.md)）
- **新建详情节点（60）：** 含 ACT、openpi、WHAM、PULSE、Mink、OmniGibson、InstinctLab / instinct_rl / instinct_onboard、工程中间件（CleanRL、LIBERO、ros2_control、Webots 等）— 见覆盖索引「新建」列；对应 `sources/repos|sites|papers/*`
- **交叉：** [`wiki/entities/humanoid-motion-intelligence.md`](wiki/entities/humanoid-motion-intelligence.md) 导航挂接覆盖索引
- **说明：** 新建页为 draft 级工程入口节点（主表定位 + 官方链接），方法深读仍优先复用已有 `paper-*` / `methods/*`

## [2026-07-30] ingest | sources/repos/daily-omni.md + papers/daily_omni_arxiv_2505_17862 + sites/daily-omni-github-io — Daily-Omni（arXiv:2505.17862）入库：日常 AV 时序对齐基准；升格 wiki/entities/paper-daily-omni.md；交叉 wiki/queries/embodied-eval-benchmark-selection-loop.md、wiki/overview/topic-embodied-eval-benchmark.md、wiki/entities/robo-bench.md、wiki/entities/agibot-lingxi-x1.md；代码 GPL-3.0 + HF 数据已开源

- **触发：** 用户提供 <https://github.com/Lliar-liar/Daily-Omni>（及项目页 Leaderboard）
- **开源核查（步骤 2.5）：** **已开源** — 项目页徽章 → GitHub（GPL-3.0）+ HF [`liarliar/Daily-Omni`](https://huggingface.co/datasets/liarliar/Daily-Omni)（CC BY-NC-SA 4.0，`Videos.tar` + `qa.json`）；仓含 `run_pipeline.py` / `test_model*` / `baseline/`
- **新增归档：** [`sources/papers/daily_omni_arxiv_2505_17862.md`](sources/papers/daily_omni_arxiv_2505_17862.md)、[`sources/sites/daily-omni-github-io.md`](sources/sites/daily-omni-github-io.md)、[`sources/repos/daily-omni.md`](sources/repos/daily-omni.md)；索引 [`sources/README.md`](sources/README.md)
- **主升格：** [`wiki/entities/paper-daily-omni.md`](wiki/entities/paper-daily-omni.md) — 六任务族、半自动管线、Agent 基线、榜单快照（WITA-Omni 85.21%）、流程与源码时序图、结论六点
- **交叉：** [`wiki/queries/embodied-eval-benchmark-selection-loop.md`](wiki/queries/embodied-eval-benchmark-selection-loop.md)、[`wiki/overview/topic-embodied-eval-benchmark.md`](wiki/overview/topic-embodied-eval-benchmark.md)、[`wiki/entities/robo-bench.md`](wiki/entities/robo-bench.md)、[`wiki/entities/agibot-lingxi-x1.md`](wiki/entities/agibot-lingxi-x1.md)
- **机构 tags：** `fudan` / `agibot`

## [2026-07-30] ingest | sources/repos/roboparty-intact-jepa.md + papers/intact_arxiv_2607_26056 — 补充 RoboParty 组织镜像与 Lab 交叉：wiki/entities/paper-intact.md、wiki/entities/roboparty.md、wiki/overview/roboparty-lab-party-os-technology-map.md；规范仓仍为 zju3dv/INTACT-JEPA，训练代码 Coming Soon

- **触发：** 用户提供 Roboparty/INTACT-JEPA + lab.roboparty.com 口径（LeWM 学动作效果 / INTACT 补意图→动作；Direct 宏 ~95.33%，2.9–5.5 ms，约 300×）
- **开源复核（步骤 2.5）：** `Roboparty/INTACT-JEPA` 为 `zju3dv/INTACT-JEPA` 的 **fork 镜像**（文档同构；训练/权重仍 Coming Soon）
- **新增归档：** [`sources/repos/roboparty-intact-jepa.md`](sources/repos/roboparty-intact-jepa.md)；同步 [`intact-jepa.md`](sources/repos/intact-jepa.md)、[`intact_arxiv_2607_26056.md`](sources/papers/intact_arxiv_2607_26056.md)、[`intact-jepa-github-io.md`](sources/sites/intact-jepa-github-io.md)、[`lab_roboparty_com.md`](sources/sites/lab_roboparty_com.md)、[`sources/README.md`](sources/README.md)
- **wiki：** [`wiki/entities/paper-intact.md`](wiki/entities/paper-intact.md) 强化 LeWM↔INTACT 叙事与双仓链接；[`wiki/entities/roboparty.md`](wiki/entities/roboparty.md)、[`wiki/overview/roboparty-lab-party-os-technology-map.md`](wiki/overview/roboparty-lab-party-os-technology-map.md) 挂接 Lab 成果

## [2026-07-30] structural | scripts/generate_link_graph.py + docs/graph.html — 命名社区上限 16→21（显示总数目标约 20，含其他）；社区漂浮标签 2D/3D 字号分离，排除「其他」漂浮

- **社区上限：** `PRIMARY_COMMUNITY_CAP` / `MAX_COMMUNITIES` **16 → 21**（命名席位不含兜底桶；含 `community-other` 时图例总数 = 命名 + 1，目标约 20）
- **漂浮标签：** 2D 保持 8–28px；3D 专用 8–16px + 相机 scale 钳制收窄至约 0.4–1.85；`community-other` 不漂浮
- **页面：** [`docs/graph.html`](docs/graph.html)、[`docs/graph-3d.js`](docs/graph-3d.js)

## [2026-07-30] ingest | sources/papers/intact_arxiv_2607_26056.md + sites/intact-jepa-github-io + repos/intact-jepa — INTACT（arXiv:2607.26056）入库：无搜索意图→动作 WM；升格 wiki/entities/paper-intact.md；交叉 paper-dwm-separating / vjepa2 / world-action-models / latency-tradeoff；代码 Coming Soon

- **归档：** [`sources/papers/intact_arxiv_2607_26056.md`](sources/papers/intact_arxiv_2607_26056.md)、[`sources/sites/intact-jepa-github-io.md`](sources/sites/intact-jepa-github-io.md)、[`sources/repos/intact-jepa.md`](sources/repos/intact-jepa.md)；索引 [`sources/README.md`](sources/README.md)
- **开源核查（2026-07-30，步骤 2.5）：** **部分开源** — 项目页 <https://zju3dv.github.io/INTACT-JEPA/> + [`zju3dv/INTACT-JEPA`](https://github.com/zju3dv/INTACT-JEPA)（MIT）文档齐全；训练/权重徽章 **Coming Soon**（`docs/RELEASE.md` Stage 0–2）
- **主升格：** [`wiki/entities/paper-intact.md`](wiki/entities/paper-intact.md) — 四槽语法、Direct/Guarded、LeWM 四任务数字、结论六要点；源码运行时序图标注不适用
- **交叉：** [`wiki/entities/paper-dwm-separating-world-effects.md`](wiki/entities/paper-dwm-separating-world-effects.md)、[`wiki/entities/paper-vjepa2.md`](wiki/entities/paper-vjepa2.md)、[`wiki/concepts/world-action-models.md`](wiki/concepts/world-action-models.md)、[`wiki/concepts/embodied-fm-latency-generalization-tradeoff.md`](wiki/concepts/embodied-fm-latency-generalization-tradeoff.md)、[`wiki/overview/world-model-physics-fidelity-outputs.md`](wiki/overview/world-model-physics-fidelity-outputs.md)
- **机构 tags：** `zju` / `tsinghua` / `roboparty`

## [2026-07-30] ingest | sources/papers/hifi_umi_arxiv_2607_25895.md + sites/hifi-umi-project + datasets/hifi-umi-2k — HiFi-UMI（arXiv:2607.25895）入库：2000h 双臂 UMI；升格 wiki/entities/paper-hifi-umi.md；交叉 teleoperation / handumi / bimanual / wam；数据已开、系统代码未列

- **归档：** [`sources/papers/hifi_umi_arxiv_2607_25895.md`](sources/papers/hifi_umi_arxiv_2607_25895.md)、[`sources/sites/hifi-umi-project.md`](sources/sites/hifi-umi-project.md)、[`sources/datasets/hifi-umi-2k.md`](sources/datasets/hifi-umi-2k.md)；索引 [`sources/README.md`](sources/README.md)
- **开源核查（2026-07-30，步骤 2.5）：** **部分开源（数据）** — HF [`simple-world-lab/HiFi-UMI-2K`](https://huggingface.co/datasets/simple-world-lab/HiFi-UMI-2K)（CC BY 4.0）；项目页 <https://cloud.simpleai.tech/simple-world-lab/hifi-umi/> 未列采数/训练 GitHub
- **主升格：** [`wiki/entities/paper-hifi-umi.md`](wiki/entities/paper-hifi-umi.md) — 采数共设计、zero-robot 后训练三骨干、2k h 数据集、结论六点；源码运行时序图标注不适用
- **交叉：** [`wiki/entities/handumi.md`](wiki/entities/handumi.md)、[`wiki/tasks/teleoperation.md`](wiki/tasks/teleoperation.md)、[`wiki/tasks/bimanual-manipulation.md`](wiki/tasks/bimanual-manipulation.md)、[`wiki/tasks/manipulation.md`](wiki/tasks/manipulation.md)、[`wiki/methods/vla.md`](wiki/methods/vla.md)、[`wiki/concepts/world-action-models.md`](wiki/concepts/world-action-models.md)
- **机构 tags：** `simple-ai`

## [2026-07-30] ingest | sources/papers/pi_r2_arxiv_2607_26055.md + sites/pi-r2-flow-github-io + repos/pi-r2-flow — πR²（arXiv:2607.26055）入库：GR00T 反应式实时 flow；升格 wiki/entities/paper-pi-r2.md；交叉 action-chunking / vla / latency-tradeoff / manipulation；训练+部署已开源

- **归档：** [`sources/papers/pi_r2_arxiv_2607_26055.md`](sources/papers/pi_r2_arxiv_2607_26055.md)、[`sources/sites/pi-r2-flow-github-io.md`](sources/sites/pi-r2-flow-github-io.md)、[`sources/repos/pi-r2-flow.md`](sources/repos/pi-r2-flow.md)；索引 [`sources/README.md`](sources/README.md)
- **开源核查（2026-07-30，步骤 2.5）：** **已开源** — 项目页 → [`pi-r2-flow/pi-r2-flow`](https://github.com/pi-r2-flow/pi-r2-flow)；`deployment/apps/run_policy.py` + `learning/Isaac-GR00T` 微调入口
- **主升格：** [`wiki/entities/paper-pi-r2.md`](wiki/entities/paper-pi-r2.md) — 快/慢通道、staircase、真机 25 Hz、流程与源码运行时序图、结论六要点
- **交叉：** [`wiki/methods/action-chunking.md`](wiki/methods/action-chunking.md)、[`wiki/methods/vla.md`](wiki/methods/vla.md)、[`wiki/concepts/embodied-fm-latency-generalization-tradeoff.md`](wiki/concepts/embodied-fm-latency-generalization-tradeoff.md)、[`wiki/tasks/manipulation.md`](wiki/tasks/manipulation.md)、[`wiki/entities/paper-hrl-stack-34-gr00t_n1.md`](wiki/entities/paper-hrl-stack-34-gr00t_n1.md)、[`wiki/queries/vla-deployment-guide.md`](wiki/queries/vla-deployment-guide.md)
- **机构 tags：** `cmu`

## [2026-07-30] ingest | sources/papers/shells_arxiv_2605_31283.md + sites/shells-project — SHELLS（arXiv:2605.31283）入库：粗引导分层采样前馈多视角人头；升格 wiki/entities/paper-shells-layered-surface-sampling.md；交叉 Face Anything / GNM Head / visual-representation / humanoid-training-data-pipeline / teleoperation；截至入库日未开源

- **归档：** [`sources/papers/shells_arxiv_2605_31283.md`](sources/papers/shells_arxiv_2605_31283.md)、[`sources/sites/shells-project.md`](sources/sites/shells-project.md)；索引 [`sources/README.md`](sources/README.md)
- **开源核查（2026-07-30，步骤 2.5）：** **未开源** — 项目页 <https://syntec-research.github.io/SHELLS/> 仅 arXiv / PDF / BibTeX，无 GitHub / HF；论文未列代码 URL
- **主升格：** [`wiki/entities/paper-shells-layered-surface-sampling.md`](wiki/entities/paper-shells-layered-surface-sampling.md) — 分层壳方法、0.08s / −88% 显存、少视角与遮挡、结论七要点；源码运行时序图标注不适用
- **交叉：** [`wiki/entities/paper-face-anything-4d-face-reconstruction.md`](wiki/entities/paper-face-anything-4d-face-reconstruction.md)、[`wiki/entities/gnm-head.md`](wiki/entities/gnm-head.md)、[`wiki/concepts/visual-representation-for-policy.md`](wiki/concepts/visual-representation-for-policy.md)、[`wiki/concepts/sim2real.md`](wiki/concepts/sim2real.md)、[`wiki/queries/humanoid-training-data-pipeline.md`](wiki/queries/humanoid-training-data-pipeline.md)、[`wiki/tasks/teleoperation.md`](wiki/tasks/teleoperation.md)
- **机构 tags：** `google`（已在 institutions.json）

## [2026-07-30] ingest | sources/papers/prism_arxiv_2607_23473.md + sites/lsh3163-prism-github-io + repos/prism — PRISM（arXiv:2607.23473）入库：因式分解多项式本体表征；升格 wiki/entities/paper-prism.md；交叉 BFM-Zero / Humanoid-Gym / Diffusion Policy；已开源 conditioner+补丁

- **归档：** [`sources/papers/prism_arxiv_2607_23473.md`](sources/papers/prism_arxiv_2607_23473.md)、[`sources/sites/lsh3163-prism-github-io.md`](sources/sites/lsh3163-prism-github-io.md)、[`sources/repos/prism.md`](sources/repos/prism.md)；索引 [`sources/README.md`](sources/README.md)
- **开源核查（2026-07-30，步骤 2.5）：** **已开源** — 项目页 Code → [`lsh3163/prism`](https://github.com/lsh3163/prism)；`PRISMConditioner` + BFM-Zero / SmolVLA 补丁；上游数据/权重不随仓分发；LICENSE finalize 中
- **主升格：** [`wiki/entities/paper-prism.md`](wiki/entities/paper-prism.md) — 方法、Humanoid-Gym/LIBERO/BFM-Zero/SmolVLA 结果、流程与源码运行时序图、结论六要点
- **交叉：** [`wiki/entities/paper-bfm-zero.md`](wiki/entities/paper-bfm-zero.md)、[`wiki/entities/humanoid-gym.md`](wiki/entities/humanoid-gym.md)、[`wiki/methods/diffusion-policy.md`](wiki/methods/diffusion-policy.md)、[`wiki/tasks/locomotion.md`](wiki/tasks/locomotion.md)、[`wiki/tasks/manipulation.md`](wiki/tasks/manipulation.md)、[`wiki/concepts/contact-rich-manipulation.md`](wiki/concepts/contact-rich-manipulation.md)
- **机构 tags：** `umich`（已在 institutions.json）

## [2026-07-30] ingest | sources/papers/transformer_transformer_arxiv_2607_25798.md + sites/transformer-transformer-github-io + repos/transformer-transformer — Transformer Transformer（arXiv:2607.25798）入库：RoboTokens+DiT 运动条件共设计与跨具身控制；升格 wiki/entities/paper-transformer-transformer.md；交叉 Shape Your Body / ALOHA / diffusion-model / cross-embodiment / bimanual / teleoperation；已开源全栈+ckpt

- **归档：** [`sources/papers/transformer_transformer_arxiv_2607_25798.md`](sources/papers/transformer_transformer_arxiv_2607_25798.md)、[`sources/sites/transformer-transformer-github-io.md`](sources/sites/transformer-transformer-github-io.md)、[`sources/repos/transformer-transformer.md`](sources/repos/transformer-transformer.md)；索引 [`sources/README.md`](sources/README.md)
- **开源核查（2026-07-30，步骤 2.5）：** **已开源** — 代码 [real-stanford/transformer-transformer](https://github.com/real-stanford/transformer-transformer)（MIT+上游例外）+ lab ckpt/data + HF 训练 Zarr
- **主升格：** [`wiki/entities/paper-transformer-transformer.md`](wiki/entities/paper-transformer-transformer.md) — RoboTokens、DSG、三设计空间、ALOHA 真机、流程与源码运行时序图、结论六要点
- **交叉：** [`wiki/entities/paper-shape-your-body-value-gradient-design.md`](wiki/entities/paper-shape-your-body-value-gradient-design.md)、[`wiki/entities/aloha.md`](wiki/entities/aloha.md)、[`wiki/concepts/diffusion-model.md`](wiki/concepts/diffusion-model.md)、[`wiki/queries/cross-embodiment-transfer-strategy.md`](wiki/queries/cross-embodiment-transfer-strategy.md)、[`wiki/tasks/bimanual-manipulation.md`](wiki/tasks/bimanual-manipulation.md)、[`wiki/tasks/teleoperation.md`](wiki/tasks/teleoperation.md)
- **机构 tags：** `stanford` / `columbia`（已在 institutions.json）

## [2026-07-29] lint | scripts/lint_wiki.py + tests/test_lint_wiki_perception_stack_crosslink.py — V31 P0 感知栈页交叉链路巡检 V1（信息型，不阻塞 CI）

- **新增检查：** [`scripts/lint_wiki.py`](scripts/lint_wiki.py) `_check_perception_stack_crosslink` — 对 `tags` 含 `detection` / `segmentation` / `perception` / `semantic`(-mapping)（连字符 token 前缀匹配，覆盖 `object-detection` / `instance-segmentation` / `promptable-segmentation` / `semantic-mapping`，规避 `reception` / `impedance` 裸子串误判）的 `entities/` / `comparisons/` / `concepts/` / `methods/` 页，检查正文是否回链「机器人视觉感知栈选型闭环」专题枢纽（`robot-perception-stack-selection-loop` / `topic-perception-stack`），枢纽页豁免；缺失记 INFO 级 `perception_stack_crosslink`
- **登记：** `INFO_ONLY_KEYS` / `_empty_results` / `format_report` / `lint()` 四处对齐 V30 `_check_actuator_drive_chain_crosslink` 模式
- **基线快照：** [`exports/lint-report.md`](exports/lint-report.md) — 当前 **72 项**待补感知栈回链页（信息型，供后续 P1/P3 交叉补强收敛）
- **测试：** [`tests/test_lint_wiki_perception_stack_crosslink.py`](tests/test_lint_wiki_perception_stack_crosslink.py) 13 例全绿（四类目录、内联/列表 tag、query/topic/双枢纽回链、枢纽豁免、裸子串/无关标签不误判、复数与派生 tag、INFO 不计失败）
- **门禁：** `make ci-preflight` 通过（lint 0 errors、导出质量 12/12）
- **清单：** [`docs/checklists/tech-stack-next-phase-checklist-v31.md`](docs/checklists/tech-stack-next-phase-checklist-v31.md) V31 P0 打勾

## [2026-07-29] ingest | sources/papers/softvtbench_arxiv_2607_04234.md + sites/softvtbench-github-io + repos/softvtbench — SoftVTBench（arXiv:2607.04234）入库：Goal/Safety 视触觉可变形基准；升格 wiki/entities/paper-softvtbench.md；交叉 tactile / visuo-tactile / contact-rich / topic-tactile / manipulation / embodied-eval；开源代码+数据，参考权重待发

- **归档：** [`sources/papers/softvtbench_arxiv_2607_04234.md`](sources/papers/softvtbench_arxiv_2607_04234.md)、[`sources/sites/softvtbench-github-io.md`](sources/sites/softvtbench-github-io.md)、[`sources/repos/softvtbench.md`](sources/repos/softvtbench.md)；索引 [`sources/README.md`](sources/README.md)
- **开源核查（2026-07-29，步骤 2.5）：** **已开源** — 代码 [TuojingAI/SoftVTBench](https://github.com/TuojingAI/SoftVTBench)（Apache-2.0）+ HF/ModelScope 数据；项目页 Dataset 按钮文案滞后；**参考 SoftVTBench checkpoint 待发**
- **主升格：** [`wiki/entities/paper-softvtbench.md`](wiki/entities/paper-softvtbench.md) — Goal/Safety 协议、2×2 套件、π₀.₅ VO/VT 结果、流程与源码运行时序图、结论六要点
- **交叉：** [`wiki/concepts/visuo-tactile-fusion.md`](wiki/concepts/visuo-tactile-fusion.md)、[`wiki/concepts/tactile-sensing.md`](wiki/concepts/tactile-sensing.md)、[`wiki/concepts/contact-rich-manipulation.md`](wiki/concepts/contact-rich-manipulation.md)、[`wiki/overview/topic-tactile.md`](wiki/overview/topic-tactile.md)、[`wiki/overview/topic-embodied-eval-benchmark.md`](wiki/overview/topic-embodied-eval-benchmark.md)、[`wiki/tasks/manipulation.md`](wiki/tasks/manipulation.md)、[`wiki/queries/embodied-eval-benchmark-selection-loop.md`](wiki/queries/embodied-eval-benchmark-selection-loop.md)、[`wiki/entities/paper-taco-tactile-sensor-benchmark.md`](wiki/entities/paper-taco-tactile-sensor-benchmark.md)
- **机构注册：** `seu` / `kcl` / `stevens` / `tuojing` / `simple-ai`（写入 `schema/institutions.json`）

## [2026-07-29] structural | docs/graph-3d.js — 修复 3D 社区标签旋转/平移卡顿：质心缓存 + rAF 合并 + translate3d 定位

- **根因：** 相机 `controls.change` 每事件同步调用 `computeCommunityCentroids3D()`（O(节点数)）并写 `left/top`，拖拽时同帧多次触发 → 标签一卡一卡
- **修复：** [`docs/graph-3d.js`](docs/graph-3d.js) — 世界坐标质心缓存（仅引擎 tick / 全量 sync 失效）；`change` 经 rAF 合并后只做屏幕重投影；[`docs/graph.html`](docs/graph.html) 标签改 `translate3d` + `will-change: transform`
- **验证：** [`scripts/verify_graph_3d_label_smooth.cjs`](scripts/verify_graph_3d_label_smooth.cjs)；同步更新 3D/跟随验证脚本读 transform 坐标

## [2026-07-29] structural | docs/graph.html — 图谱「显示社区标签」改为默认开启，用户可在参数浮窗自行关闭

- **改动：** [`docs/graph.html`](docs/graph.html) — `showCommunityLabels` 默认 `true`，参数面板勾选框默认 `checked`；非「按社区」模式仍置灰不可选
- **附带修复：** 将 `searchQuery` 提前声明，避免默认开启后 simulation tick → `hasActiveGraphFilter` 踩 `let` TDZ 导致图谱加载卡住
- **验证脚本：** [`scripts/verify_graph_community_labels.cjs`](scripts/verify_graph_community_labels.cjs)、[`scripts/verify_graph_community_labels_3d.cjs`](scripts/verify_graph_community_labels_3d.cjs)、[`scripts/verify_graph_community_labels_follow.cjs`](scripts/verify_graph_community_labels_follow.cjs) — 对齐默认开启口径
- **清单：** [`docs/checklists/frontend-optimization-v1.md`](docs/checklists/frontend-optimization-v1.md)

## [2026-07-29] ingest | sources/papers/pot_vla_arxiv_2607_18016.md — 复核查 arXiv:2607.18016v2（科学内容无实质变更，仅 DeepCybo 实习脚注）；开源仍未发布；刷新 wiki/entities/paper-pot-vla.md 与交叉页状态日

- **来源归档：** [`sources/papers/pot_vla_arxiv_2607_18016.md`](sources/papers/pot_vla_arxiv_2607_18016.md) — 标注当前 **v2**、v1→v2 差分（仅实习脚注）、开源复核查日 **2026-07-29**
- **主实体刷新：** [`wiki/entities/paper-pot-vla.md`](wiki/entities/paper-pot-vla.md) — 已有完整升格（2026-07-22/26）；本次不重造页，更新版本/开源状态与 HTML 链至 v2
- **开源核查（步骤 2.5）：** **确认未开源** — arXiv Code 区无入口；GitHub / DeepCybo 无官方仓；源码运行时序图仍 **不适用**
- **既有交叉（未改内容）：** [`wiki/methods/vla.md`](wiki/methods/vla.md)、[`wiki/tasks/loco-manipulation.md`](wiki/tasks/loco-manipulation.md)、[`wiki/entities/unitree-g1.md`](wiki/entities/unitree-g1.md)、[`wiki/entities/isaac-gr00t.md`](wiki/entities/isaac-gr00t.md)、[`wiki/entities/paper-hrl-stack-34-gr00t_n1.md`](wiki/entities/paper-hrl-stack-34-gr00t_n1.md)、[`wiki/entities/paper-loco-manip-161-057-being-0.md`](wiki/entities/paper-loco-manip-161-057-being-0.md)

## [2026-07-29] ingest | sources/papers/data_pyramid_embodied_manipulation_arxiv_2607_24744.md + sites/embodied-data-pyramid + repos/awesome-embodied-data-pyramid — 具身数据金字塔综述（arXiv:2607.24744）入库：五层数据生态 × 六维属性 × 基础模型数据配方；升格 wiki/entities/paper-data-pyramid-embodied-manipulation.md；交叉 OXE / GR00T N1 / scaling-laws / WAM / vla / hub-vla

- **归档：** [`sources/papers/data_pyramid_embodied_manipulation_arxiv_2607_24744.md`](sources/papers/data_pyramid_embodied_manipulation_arxiv_2607_24744.md)、[`sources/sites/embodied-data-pyramid.md`](sources/sites/embodied-data-pyramid.md)、[`sources/repos/awesome-embodied-data-pyramid.md`](sources/repos/awesome-embodied-data-pyramid.md)；索引 [`sources/README.md`](sources/README.md)
- **开源核查（2026-07-29，步骤 2.5）：** **资源型开源**——Awesome 策展清单 `worldbench/awesome-embodied-data-pyramid` + 项目页五层数据集检索表；综述无训练/推理代码（源码运行时序图不适用，已在实体页注明）
- **主升格：** [`wiki/entities/paper-data-pyramid-embodied-manipulation.md`](wiki/entities/paper-data-pyramid-embodied-manipulation.md) — 两主轴+四辅维组织原则、五层金字塔 Mermaid、各层采集管线归纳、70+ 模型数据配方三趋势、动作空间对齐三策略、结论六要点与六大开放挑战
- **交叉：** [`wiki/concepts/open-x-embodiment.md`](wiki/concepts/open-x-embodiment.md)、[`wiki/entities/paper-hrl-stack-34-gr00t_n1.md`](wiki/entities/paper-hrl-stack-34-gr00t_n1.md)（模型专属金字塔 vs 类目级系统化）、[`wiki/concepts/embodied-scaling-laws.md`](wiki/concepts/embodied-scaling-laws.md)、[`wiki/concepts/world-action-models.md`](wiki/concepts/world-action-models.md)、[`wiki/methods/vla.md`](wiki/methods/vla.md)、[`roadmap/depth-vla.md`](roadmap/depth-vla.md) Stage 3
- **机构 tags：** `pku`/`ntu`/`hkust`/`nus`/`cuhk`/`hku`/`duke`/`ucb`/`gbu`/`nju`/`sjtu`（均已在 institutions.json，无需新增注册）

## [2026-07-28] structural | V31 P1 感知栈层级专题交叉补强 — 7 个感知实体页与「机器人视觉感知栈选型闭环」Query 页双向回链，标注各页所在感知栈层（②2D 检测/分割 · ③2D→3D 语义建图）

- **执行清单：** [`docs/checklists/tech-stack-next-phase-checklist-v31.md`](docs/checklists/tech-stack-next-phase-checklist-v31.md) P1 第二项打勾
- **补回链的实体页（各页 `related` frontmatter + 「关联页面」正文均补入 Query 页并标注层级）：** [`wiki/entities/ultralytics.md`](wiki/entities/ultralytics.md)（②单阶段实时检测）、[`wiki/entities/rf-detr.md`](wiki/entities/rf-detr.md)（②端到端 DETR）、[`wiki/entities/paper-yolo-unified-realtime-detection.md`](wiki/entities/paper-yolo-unified-realtime-detection.md)（②单阶段奠基）、[`wiki/entities/paper-segment-anything.md`](wiki/entities/paper-segment-anything.md)（②可提示分割）、[`wiki/entities/paper-sam2.md`](wiki/entities/paper-sam2.md)（②视频可提示分割）、[`wiki/entities/findanything.md`](wiki/entities/findanything.md)（③对象级开放词汇 3D 语义建图）、[`wiki/entities/cmu-mscv-semantic-3d-mapping.md`](wiki/entities/cmu-mscv-semantic-3d-mapping.md)（③DETR+SAM 投影建图）
- **枢纽 Query 页：** [`wiki/queries/robot-perception-stack-selection-loop.md`](wiki/queries/robot-perception-stack-selection-loop.md)（`related` 此前已含全部感知页，本次补齐反向边，双向闭合）
- **门禁：** `make ci-preflight` 通过；`graph-stats.json` 0 orphans、边数 17456 → 17463（+7）、`community_quality_warning: false`（`largest_community_ratio: 0.145`）

## [2026-07-28] ingest | sources/personal/residual-policy-reading-list.md — Residual Policy / Residual RL 九篇谱系入库：新增方法枢纽页 + 7 个论文实体页，RuN 原地升格完整详情页（非 stub、不建重复节点），ResMimic 交叉补强；机构注册 +4（siemens/qut/csiro/ttic）

- **归档：** [`sources/personal/residual-policy-reading-list.md`](sources/personal/residual-policy-reading-list.md)；sites ×7（residualrl / rpl / rfc / learning-to-jump / multimodal / reskill / rsa）、repos ×5（residual-policy-learning / rfc / cheetah-trainer / reskill / rsa-shared-autonomy）
- **开源核查（2026-07-28，步骤 2.5）：** 已开源 5（RPL、RFC 非商用、ARRL 三仓、ReSkill MIT、RSA）；未开源/未见 3（Johannink Residual RL、Versatile Jumping、RuN）；ResMimic 此前已归档
- **方法枢纽：** [`wiki/methods/residual-policy-learning.md`](wiki/methods/residual-policy-learning.md) — a=a_base+Δa 统一形式、九篇谱系表、工程三件套与选型建议
- **论文实体（新建 ×7）：** [`wiki/entities/paper-residual-rl-robot-control.md`](wiki/entities/paper-residual-rl-robot-control.md)（Johannink，ICRA 2019）、[`wiki/entities/paper-residual-policy-learning.md`](wiki/entities/paper-residual-policy-learning.md)（Silver RPL）、[`wiki/entities/paper-rfc-residual-force-control.md`](wiki/entities/paper-rfc-residual-force-control.md)（NeurIPS 2020）、[`wiki/entities/paper-versatile-jumping-action-residuals.md`](wiki/entities/paper-versatile-jumping-action-residuals.md)（L4DC 2022）、[`wiki/entities/paper-multimodal-legged-arrl.md`](wiki/entities/paper-multimodal-legged-arrl.md)（RA-L/IROS 2022）、[`wiki/entities/paper-reskill-residual-skill-policies.md`](wiki/entities/paper-reskill-residual-skill-policies.md)（CoRL 2022）、[`wiki/entities/paper-residual-policy-shared-autonomy.md`](wiki/entities/paper-residual-policy-shared-autonomy.md)（ICRA 2020）
- **原地升格（planned→complete，非重复节点）：** [`wiki/entities/paper-notebook-run-residual-policy-for-natural-humanoid-locomot.md`](wiki/entities/paper-notebook-run-residual-policy-for-natural-humanoid-locomot.md)（RuN，G1 0–2.5 m/s 走跑；保留 paper-notebook 索引关系）
- **交叉补强：** [`wiki/entities/paper-resmimic.md`](wiki/entities/paper-resmimic.md)、[`wiki/methods/reinforcement-learning.md`](wiki/methods/reinforcement-learning.md)、[`wiki/methods/deepmimic.md`](wiki/methods/deepmimic.md)、[`wiki/entities/unitree-g1.md`](wiki/entities/unitree-g1.md)、[`wiki/overview/paper-notebook-category-05-locomotion.md`](wiki/overview/paper-notebook-category-05-locomotion.md)、[`index.md`](index.md)
- **机构注册：** `siemens`（西门子）、`qut`（昆士兰科技大学）、`csiro`（联邦科学与工业研究组织）、`ttic`（丰田工业大学芝加哥分校）

## [2026-07-28] ingest | sources/personal/humanoid-loco-policy-reward-functions-faq.md — 新增概念页 wiki/concepts/humanoid-policy-reward-functions.md（人形运控常见奖励函数六类划分与权重量级，与观测输入页互为对偶）；交叉 reward-design / humanoid-policy-observation-inputs / queries/locomotion-reward-design-guide / queries/humanoid-rl-cookbook / tasks/humanoid-locomotion

## [2026-07-28] ingest | sources/repos/mimickit.md — 复核 xbpeng/MimicKit（Apache-2.0，已开源）并充实归档：引擎版本 pin（Isaac Lab 2ed331a / Newton v1.0.0）、分布式训练、logger/预训练模型、GMR+SMPL 重定向工具链；补充 wiki/entities/mimickit.md 工程信息速查与 GMR 交叉

## [2026-07-28] ingest | sources/papers/teledexter_arxiv_2607_11481.md + sites/teledexter-project.md — TeleDexter（arXiv:2607.11481）手–物 co-tracking 灵巧遥操作入库；升格 wiki/entities/paper-teledexter.md；交叉 teleoperation / contact-rich / in-hand-reorientation / dexterous-data-collection / depth-teleoperation / HDMI；未开源

- **一手入口：** 项目页 [bigai-dex.github.io/blog/teledexter](https://bigai-dex.github.io/blog/teledexter/) · 论文 [arXiv:2607.11481](https://arxiv.org/abs/2607.11481) · PDF [paper_teledexter.pdf](https://bigai-dex.github.io/blog/teledexter/paper_teledexter.pdf)
- **开源核查（2026-07-28）：** **未开源** — 项目页 metalinks 仅 arXiv；GitHub `teledexter` 搜索 0；无 HF/数据集链接
- **归档：** [`sources/papers/teledexter_arxiv_2607_11481.md`](sources/papers/teledexter_arxiv_2607_11481.md)、[`sources/sites/teledexter-project.md`](sources/sites/teledexter-project.md)；索引 [`sources/README.md`](sources/README.md)
- **主升格：** [`wiki/entities/paper-teledexter.md`](wiki/entities/paper-teledexter.md) — 连续子目标 co-tracking、hybrid reward、action masking、七任务 75.2% SR、DP 数据飞轮、结论与开源边界
- **交叉：** [`wiki/tasks/teleoperation.md`](wiki/tasks/teleoperation.md)、[`wiki/concepts/contact-rich-manipulation.md`](wiki/concepts/contact-rich-manipulation.md)、[`wiki/methods/in-hand-reorientation.md`](wiki/methods/in-hand-reorientation.md)、[`wiki/methods/diffusion-policy.md`](wiki/methods/diffusion-policy.md)、[`wiki/queries/dexterous-data-collection-guide.md`](wiki/queries/dexterous-data-collection-guide.md)、[`roadmap/depth-teleoperation.md`](roadmap/depth-teleoperation.md)、[`wiki/entities/paper-hrl-stack-06-hdmi.md`](wiki/entities/paper-hrl-stack-06-hdmi.md)
- **机构 tags：** `bigai` / `tsinghua` / `pku`（已在 institutions.json）

## [2026-07-28] ingest | sources/personal/humanoid-loco-policy-observation-inputs-faq.md — 新增概念页 wiki/concepts/humanoid-policy-observation-inputs.md（人形运控策略观测输入五类划分与获取链路）；交叉 state-estimation / privileged-training / terrain-latent-representation / humanoid-policy-network-architecture / tasks/humanoid-locomotion / queries/humanoid-rl-cookbook

## [2026-07-28] ingest | sources/papers/raven_rl_adaptive_visibility_graph_arxiv_2607_15701.md — UCLA RoMeLa RAVEN（arXiv:2607.15701）复核升格：补结论/开源再核；wiki/entities/paper-raven-rl-adaptive-visibility-graph-mpc.md；交叉 far-planner / mpc-vs-rl / humanoid-locomotion；roadmap/depth-navigation.md Stage 3

## [2026-07-28] structural | 主路线与全部纵深路线链接的论文节点 stub/planned → 详细 complete — roadmap/motion-control + depth-*（30 篇升格）

- **触发：** 确保所有纵深路线与主路线中链接的论文节点均为详细页，而非 stub/planned/draft
- **验收：** 扫描 `roadmap/motion-control.md`、全部 `roadmap/depth-*.md` 与 `wiki/roadmaps/` 内 `wiki/entities/paper-*.md` 链接；`status ∈ {stub,planned,draft}` 或 `paper-notebook-stub/planned` 标签剩余 **0**
- **本批升格（30）：** 遥操作 12 · 导航 7 · 动作生成 5 · 拳击 2 · 感知越障 2 · 力矩电机 2（足球纵深此前已合入）
- **明细见同日后续 ingest 条目**（teleoperation / navigation / motion-generation / boxing·parkour·torque）
- **路线入口：** [`roadmap/README.md`](roadmap/README.md)

## [2026-07-28] ingest | roadmap/depth-navigation 7 篇论文 stub/planned 升格详细实体 — EgoNav / FocusNav / PioneeR / LookOut / NavDP / NaVILA / NoMaD

- **主升格（status → complete）：** [`wiki/entities/paper-notebook-egonav.md`](wiki/entities/paper-notebook-egonav.md)、[`wiki/entities/paper-notebook-focusnav.md`](wiki/entities/paper-notebook-focusnav.md)、[`wiki/entities/paper-notebook-learning-social-navigation-from-positive-and-neg.md`](wiki/entities/paper-notebook-learning-social-navigation-from-positive-and-neg.md)、[`wiki/entities/paper-notebook-lookout.md`](wiki/entities/paper-notebook-lookout.md)、[`wiki/entities/paper-notebook-navdp-learning-sim-to-real-navigation-diffusion.md`](wiki/entities/paper-notebook-navdp-learning-sim-to-real-navigation-diffusion.md)、[`wiki/entities/paper-notebook-navila-legged-robot-vision-language-action-model.md`](wiki/entities/paper-notebook-navila-legged-robot-vision-language-action-model.md)、[`wiki/entities/paper-notebook-nomad-goal-masked-diffusion-policies-for-navigat.md`](wiki/entities/paper-notebook-nomad-goal-masked-diffusion-policies-for-navigat.md)
- **开源核查：** NavDP / NaVILA / NoMaD **已开源**（新增 `sources/repos/`）；LookOut **仅数据开放**；EgoNav **待发布**；FocusNav / PioneeR **未开源**。
- **项目页归档：** `sources/sites/egonav.md`、`lookout.md`、`navdp.md`、`navila.md`、`nomad.md`、`pioneer-social-navigation.md`。
- **路线入口：** [`roadmap/depth-navigation.md`](roadmap/depth-navigation.md) Stage 3–5；交叉 [`wiki/tasks/vision-language-navigation.md`](wiki/tasks/vision-language-navigation.md) 与 [`wiki/entities/paper-da-nav.md`](wiki/entities/paper-da-nav.md)。

## [2026-07-28] ingest | roadmap/depth-teleoperation 六篇 stub/planned 升格为详细论文实体 — ACE、Bunny-VisionPro、CHILD、DexterCap、ByteDexter、DexUMI

- **主升格（status → complete）：** [`wiki/entities/paper-notebook-ace-a-cross-platform-visual-exoskeletons-system.md`](wiki/entities/paper-notebook-ace-a-cross-platform-visual-exoskeletons-system.md)、[`wiki/entities/paper-notebook-bunny-visionpro-real-time-bimanual-dexterous-tel.md`](wiki/entities/paper-notebook-bunny-visionpro-real-time-bimanual-dexterous-tel.md)、[`wiki/entities/paper-notebook-child-a-whole-body-humanoid-teleoperation-system.md`](wiki/entities/paper-notebook-child-a-whole-body-humanoid-teleoperation-system.md)、[`wiki/entities/paper-notebook-dextercap.md`](wiki/entities/paper-notebook-dextercap.md)、[`wiki/entities/paper-notebook-dexterous-teleoperation-of-20-dof-bytedexter-han.md`](wiki/entities/paper-notebook-dexterous-teleoperation-of-20-dof-bytedexter-han.md)、[`wiki/entities/paper-notebook-dexumi-using-human-hand-as-the-universal-manipul.md`](wiki/entities/paper-notebook-dexumi-using-human-hand-as-the-universal-manipul.md)
- **开源核查（2026-07-28）：** ACE 源码/硬件公开但许可未明确；Bunny-VisionPro 部分开源；CHILD 软硬件公开但许可未明确；DexterCap 部分开源且缺原始/中间数据；ByteDexter 未开源；DexUMI 已按 MIT 完整开源。
- **路线入口：** [`roadmap/depth-teleoperation.md`](roadmap/depth-teleoperation.md)

## [2026-07-28] ingest | roadmap/depth-motion-generation 五篇 stub/planned 升格为详细论文实体 — CondMDI、Go to Zero、GMD、OmniControl、PhysDiff

- **主升格（status → complete）：** [`wiki/entities/paper-notebook-flexible-motion-in-betweening-with-diffusion-mod.md`](wiki/entities/paper-notebook-flexible-motion-in-betweening-with-diffusion-mod.md)、[`wiki/entities/paper-notebook-go-to-zero-towards-zero-shot-motion-generation-w.md`](wiki/entities/paper-notebook-go-to-zero-towards-zero-shot-motion-generation-w.md)、[`wiki/entities/paper-notebook-guided-motion-diffusion-for-controllable-human-m.md`](wiki/entities/paper-notebook-guided-motion-diffusion-for-controllable-human-m.md)、[`wiki/entities/paper-notebook-omnicontrol-control-any-joint-at-any-time-for-hu.md`](wiki/entities/paper-notebook-omnicontrol-control-any-joint-at-any-time-for-hu.md)、[`wiki/entities/paper-notebook-physdiff-physics-guided-human-motion-diffusion-m.md`](wiki/entities/paper-notebook-physdiff-physics-guided-human-motion-diffusion-m.md)
- **方法补全：** 补统一阅读骨架、方法数据流、定量评测、结论、工程边界，并交叉 [Diffusion-based Motion Generation](wiki/methods/diffusion-motion-generation.md)、[PhyGile](wiki/entities/paper-phygile.md) 与 [GPC](wiki/entities/paper-gpc-generative-pretrained-controllers.md)
- **开源核查（2026-07-28）：** CondMDI / MotionMillion / GMD / OmniControl **已开源**，新增对应 `sources/sites/` 与 `sources/repos/` 归档及源码运行时序图；PhysDiff 项目页未列官方仓库，标记 **未开源**、时序图不适用
- **路线入口：** [`roadmap/depth-motion-generation.md`](roadmap/depth-motion-generation.md)

## [2026-07-28] ingest | boxing / perceptive-locomotion / torque-motor 六篇论文 stub/planned 升格 complete — wiki/entities/paper-notebook-robostriker.md、wiki/entities/paper-notebook-towards-motion-turing-test.md、wiki/entities/paper-notebook-anymal-parkour-robust-perceptive-locomotion.md、wiki/entities/paper-notebook-humanoid-parkour-learning.md、wiki/entities/paper-notebook-human-level-actuation-for-humanoids.md、wiki/entities/paper-notebook-quasi-direct-drive-for-low-cost-compliant-roboti.md

- **开源核查：** RoboStriker / Humanoid Parkour / Human-Level Actuation 未开源；Motion Turing Test 宣称待发布；ANYmal Parkour 仅图表数据与绘图脚本部分开放；Blue/QDD 的 ROS 核心软件部分开源。
- **归档：** `sources/sites/robostriker-project.md`、`sources/sites/motion-turing-test.md`、`sources/sites/anymal-parkour.md`、`sources/repos/anymal-parkour-plotting-artifact.md`、`sources/sites/humanoid-parkour-learning.md`、`sources/sites/berkeley-open-arms-blue.md`、`sources/repos/blue-core.md`。
- **机构注册：** `schema/institutions.json` → `shanghai-innovation-institute`、`xmu`、`oppo`、`teragon-research`。

## [2026-07-28] ingest | roadmap/depth-teleoperation 关联六篇论文 stub/planned 升格 complete — wiki/entities/paper-notebook-egodex-learning-dexterous-manipulation-from-larg.md、wiki/entities/paper-notebook-nuexo-a-wearable-exoskeleton-covering-all-upper.md、wiki/entities/paper-notebook-osmo-open-source-tactile-glove-for-human-to-robo.md、wiki/entities/paper-notebook-teleopbench-a-simulator-centric-benchmark-for-du.md、wiki/entities/paper-notebook-teleoperation-of-humanoid-robots-a-survey.md、wiki/entities/paper-notebook-whole-body-bilateral-teleoperation-with-multi-st.md

## [2026-07-28] structural | COMMUNITY_NAME_OVERRIDES 补八大机器人控制体系分类（Eight Robot Control Paradigms Taxonomy）— 修复 community-9 命名不符合「中文（English） 社区」导致 Tests/pytest 失败；scripts/utils/community_labels.py

## [2026-07-28] ingest | roadmap/depth-humanoid-soccer 链接论文 stub/planned/draft 升格为详细节点 — wiki/entities/paper-notebook-learning-soccer-skills-for-humanoid-robots.md 等；roadmap/depth-humanoid-soccer.md

- **触发：** 确保 [人形足球纵深](roadmap/depth-humanoid-soccer.md)（`roadmap-depth-humanoid-soccer`）链接的论文节点均为详细页，而非 stub/planned/draft
- **主升格（status → complete）：**
  - [`wiki/entities/paper-notebook-learning-soccer-skills-for-humanoid-robots.md`](wiki/entities/paper-notebook-learning-soccer-skills-for-humanoid-robots.md) — PAiD（原 stub）
  - [`wiki/entities/paper-notebook-learning-agile-striker-skills-for-humanoid-socce.md`](wiki/entities/paper-notebook-learning-agile-striker-skills-for-humanoid-socce.md) — Agile Striker（原 stub）
  - [`wiki/entities/paper-notebook-soccerdiffusion-toward-learning-end-to-end-human.md`](wiki/entities/paper-notebook-soccerdiffusion-toward-learning-end-to-end-human.md) — SoccerDiffusion（原 planned）
  - [`wiki/entities/paper-notebook-humanoid-whole-body-badminton-via-multi-stage-re.md`](wiki/entities/paper-notebook-humanoid-whole-body-badminton-via-multi-stage-re.md) — 全身羽毛球（原 stub）
  - [`wiki/entities/paper-notebook-a-hierarchical-model-based-system-for-high-perfo.md`](wiki/entities/paper-notebook-a-hierarchical-model-based-system-for-high-perfo.md) — ARTEMIS（原 draft，补结论）
  - [`wiki/entities/paper-hrl-stack-26-learning_vision_driven_reactive_socc.md`](wiki/entities/paper-hrl-stack-26-learning_vision_driven_reactive_socc.md) — Vision-Driven Reactive Soccer（索引级→详细）
  - [`wiki/entities/paper-humanoid-soccer-swarm-intelligence.md`](wiki/entities/paper-humanoid-soccer-swarm-intelligence.md) — 补 `## 结论`
- **开源核查归档：**
  - PAiD：[soccer-humanoid-paid.md](sources/sites/soccer-humanoid-paid.md) + 更新 [humanoid_soccer.md](sources/repos/humanoid_soccer.md)（**已开源**）
  - Agile Striker：[humanoidsoccer-agile-striker.md](sources/sites/humanoidsoccer-agile-striker.md) + [humanoid-soccer-agile-striker.md](sources/repos/humanoid-soccer-agile-striker.md)（**已开源**）
  - SoccerDiffusion：[bit-bots-soccerdiffusion.md](sources/sites/bit-bots-soccerdiffusion.md) + [soccerdiffusion.md](sources/repos/soccerdiffusion.md)（**已开源** MIT）
  - Badminton：[humanoid-badminton-multi-stage-rl.md](sources/sites/humanoid-badminton-multi-stage-rl.md)（**待发布**）
  - Vision kick：[humanoid-kick-vision-driven-soccer.md](sources/sites/humanoid-kick-vision-driven-soccer.md)（**暂无 Code**）
- **机构注册：** `schema/institutions.json` → `uni-hamburg`、`bit-bots`
- **路线入口：** [`roadmap/depth-humanoid-soccer.md`](roadmap/depth-humanoid-soccer.md)

## [2026-07-28] ingest | sources/sites/opendrivelab-robonaldo.md + repos/robonaldo.md + repos/robonaldo-deploy.md — 项目页再核：Code 已挂 OpenDriveLab/RoboNaldo（训练 MIT）与 RoboNaldo_Deploy；升格更新 wiki/entities/paper-robonaldo-humanoid-soccer-shooting.md（开源状态/工程实践/源码运行时序图/结论）；交叉 wiki/tasks/humanoid-soccer.md、wiki/methods/paid-framework.md、wiki/queries/humanoid-soccer-skill-learning-method-selection.md；注册 schema/institutions.json → archon-robotics（源策未来）

- **一手入口：** 项目页 [opendrivelab.com/RoboNaldo](https://opendrivelab.com/RoboNaldo/) · 论文 [arXiv:2606.11092](https://arxiv.org/abs/2606.11092) · Video [youtu.be/BuHNzqebIqc](https://youtu.be/BuHNzqebIqc)
- **开源核查（2026-07-28）：** **已开源** — 项目页 Code → [OpenDriveLab/RoboNaldo](https://github.com/OpenDriveLab/RoboNaldo)（训练，MIT，~32★）；配套 [RoboNaldo_Deploy](https://github.com/OpenDriveLab/RoboNaldo_Deploy)（真机/MuJoCo FSM；根目录未声明 LICENSE）
- **归档：** [`sources/sites/opendrivelab-robonaldo.md`](sources/sites/opendrivelab-robonaldo.md)、[`sources/repos/robonaldo.md`](sources/repos/robonaldo.md)、[`sources/repos/robonaldo-deploy.md`](sources/repos/robonaldo-deploy.md)；更新 [`sources/papers/robonaldo_arxiv_2606_11092.md`](sources/papers/robonaldo_arxiv_2606_11092.md)、[`sources/README.md`](sources/README.md)、[`references/repos/humanoid-projects.md`](references/repos/humanoid-projects.md)
- **主升格：** [`wiki/entities/paper-robonaldo-humanoid-soccer-shooting.md`](wiki/entities/paper-robonaldo-humanoid-soccer-shooting.md) — 补齐开源状态、工程实践、源码运行时序图、结论；机构 tags `hku`/`cuhk`/`opendrivelab`/`archon-robotics`
- **交叉：** [`wiki/tasks/humanoid-soccer.md`](wiki/tasks/humanoid-soccer.md)、[`wiki/methods/paid-framework.md`](wiki/methods/paid-framework.md)、[`wiki/queries/humanoid-soccer-skill-learning-method-selection.md`](wiki/queries/humanoid-soccer-skill-learning-method-selection.md)
- **机构注册：** `schema/institutions.json` → `archon-robotics`（源策未来（Archon Robotics））

## [2026-07-28] ingest | sources/repos/drawio-scientific-illustrator.md — Draw.io Scientific Illustrator（Codex 插件 + MCP 可见操控 draw.io）；升格 wiki/entities/drawio-scientific-illustrator.md；交叉 freecad-mcp / cad-skills / img2threejs / gsap-skills / manim

- **一手入口：** [icebird1998/drawio-scientific-illustrator](https://github.com/icebird1998/drawio-scientific-illustrator)（~1.0k★，MIT，v1.0.0；Codex plugin = `drawio-live`/`drawio-file-utils` MCP + Skill `recreate-scientific-figure-in-drawio`）
- **归档：** [`sources/repos/drawio-scientific-illustrator.md`](sources/repos/drawio-scientific-illustrator.md)；索引 [`sources/README.md`](sources/README.md)、[`references/repos/utilities.md`](references/repos/utilities.md)
- **主升格：** [`wiki/entities/drawio-scientific-illustrator.md`](wiki/entities/drawio-scientific-illustrator.md) — 可见逐步绘制、硬边界（禁 OS 自动化 / XML-first）、流程与源码运行时序图
- **交叉：** [`wiki/entities/freecad-mcp.md`](wiki/entities/freecad-mcp.md)、[`wiki/entities/cad-skills.md`](wiki/entities/cad-skills.md)、[`wiki/entities/img2threejs.md`](wiki/entities/img2threejs.md)、[`wiki/entities/gsap-skills.md`](wiki/entities/gsap-skills.md)、[`wiki/entities/manim.md`](wiki/entities/manim.md)
- **开源核查（2026-07-28）：** **已开源**（MIT）；无独立项目页，以 GitHub README / Release 为准；Windows 充分测试，macOS/Linux 尽力支持

## [2026-07-28] ingest | sources/papers/birrell_nelson_implementing_rpc_tocs_1984.md + sites/rfc-5531-onc-rpc.md + grpc-io-docs.md + sources/repos/grpc.md — Remote Procedure Call 一手入库；升格 wiki/concepts/remote-procedure-call.md、wiki/entities/grpc.md

- **一手入口：**
  - Birrell & Nelson, *Implementing Remote Procedure Calls*（ACM TOCS 1984；[PDF](http://birrell.org/andrew/papers/ImplementingRPC.pdf)）
  - IETF [RFC 5531](https://www.rfc-editor.org/rfc/rfc5531) ONC RPC v2
  - gRPC 文档 [Introduction](https://grpc.io/docs/what-is-grpc/introduction/) · [Core concepts](https://grpc.io/docs/what-is-grpc/core-concepts/) · 仓 [grpc/grpc](https://github.com/grpc/grpc)（~45.2k★，Apache-2.0，v1.83.0）
- **归档：** [`sources/papers/birrell_nelson_implementing_rpc_tocs_1984.md`](sources/papers/birrell_nelson_implementing_rpc_tocs_1984.md)、[`sources/sites/rfc-5531-onc-rpc.md`](sources/sites/rfc-5531-onc-rpc.md)、[`sources/sites/grpc-io-docs.md`](sources/sites/grpc-io-docs.md)、[`sources/repos/grpc.md`](sources/repos/grpc.md)；交叉 [`sources/sites/ros2-official-documentation.md`](sources/sites/ros2-official-documentation.md)、[`sources/README.md`](sources/README.md)
- **主升格：** [`wiki/concepts/remote-procedure-call.md`](wiki/concepts/remote-procedure-call.md)、[`wiki/entities/grpc.md`](wiki/entities/grpc.md)
- **交叉：** [`wiki/concepts/ros2-basics.md`](wiki/concepts/ros2-basics.md)、[`wiki/concepts/dds-communication.md`](wiki/concepts/dds-communication.md)、[`wiki/concepts/network-protocol-stack.md`](wiki/concepts/network-protocol-stack.md)、[`wiki/overview/topic-communication.md`](wiki/overview/topic-communication.md)、[`wiki/queries/real-time-control-middleware-guide.md`](wiki/queries/real-time-control-middleware-guide.md)、[`wiki/entities/freecad-mcp.md`](wiki/entities/freecad-mcp.md)
- **机构注册：** `schema/institutions.json` → `ietf`、`cncf`、`xerox-parc`（`google` 已有）
- **开源核查（2026-07-28）：** Birrell 论文 / RFC **公开可读**；gRPC **已开源**（Apache-2.0）
- **缩写提示：** 本库另有 Regularized Predictive Control（RPC）腿足控制线；本条专指 Remote Procedure Call

## [2026-07-28] ingest | sources/sites/ros2-design-rmw-interface.md + ros2-rmw-middleware-vendors.md + sources/repos/rmw.md — RMW（ROS Middleware Interface）一手入库；升格 wiki/concepts/rmw-interface.md

- **一手入口：**
  - Design [ROS 2 middleware interface](https://design.ros2.org/articles/ros_middleware_interface.html)（Dirk Thomas；源 `ros2/design`）
  - 文档 [About Different Middleware Vendors](https://docs.ros.org/en/humble/Concepts/Intermediate/About-Different-Middleware-Vendors.html) · [Working with multiple RMW](https://docs.ros.org/en/humble/How-To-Guides/Working-with-multiple-RMW-implementations.html)（源 RST：`ros2/ros2_documentation`）
  - 仓 [ros2/rmw](https://github.com/ros2/rmw)（~119★，Apache-2.0，Quality Level 1）
- **归档：** [`sources/sites/ros2-design-rmw-interface.md`](sources/sites/ros2-design-rmw-interface.md)、[`sources/sites/ros2-rmw-middleware-vendors.md`](sources/sites/ros2-rmw-middleware-vendors.md)、[`sources/repos/rmw.md`](sources/repos/rmw.md)；交叉 [`sources/sites/ros2-official-documentation.md`](sources/sites/ros2-official-documentation.md)、[`sources/repos/ros2.md`](sources/repos/ros2.md)、[`sources/sites/dds_omg_rtos_edge_ota_safety_primary_refs.md`](sources/sites/dds_omg_rtos_edge_ota_safety_primary_refs.md)、[`sources/README.md`](sources/README.md)
- **主升格：** [`wiki/concepts/rmw-interface.md`](wiki/concepts/rmw-interface.md) — 分层、vendor 表、`RMW_IMPLEMENTATION`、开源核查
- **交叉：** [`wiki/concepts/ros2-basics.md`](wiki/concepts/ros2-basics.md)、[`wiki/concepts/dds-communication.md`](wiki/concepts/dds-communication.md)、[`wiki/entities/fast-dds.md`](wiki/entities/fast-dds.md)、[`wiki/entities/cyclone-dds.md`](wiki/entities/cyclone-dds.md)
- **开源核查（2026-07-28）：** `ros2/rmw` **已开源**（Apache-2.0）；Design / 文档源 **公开**；Fast/Cyclone RMW 适配开源；Connext/Gurum 等商业 vendor 需另装

## [2026-07-28] ingest | sources/sites/omg-dds-spec.md + fast-dds-docs.md + cyclonedds-io.md + sources/repos/fast-dds.md + cyclonedds.md — DDS 标准与 Fast/Cyclone 一手入库；升格 wiki/entities/fast-dds.md、cyclone-dds.md；深化 wiki/concepts/dds-communication.md

- **一手入口：**
  - OMG [DDS 1.4](https://www.omg.org/spec/DDS/1.4) · [DDSI-RTPS 2.5](https://www.omg.org/spec/DDSI-RTPS/2.5)
  - Fast DDS 文档 [fast-dds.docs.eprosima.com](https://fast-dds.docs.eprosima.com/) · 仓 [eProsima/Fast-DDS](https://github.com/eProsima/Fast-DDS)（~2.9k★，Apache-2.0，v3.6.2）
  - Cyclone DDS [cyclonedds.io](https://cyclonedds.io/) · 仓 [eclipse-cyclonedds/cyclonedds](https://github.com/eclipse-cyclonedds/cyclonedds)（~1.3k★，EPL-2.0/EDL-1.0，11.0.1）
- **归档：** [`sources/sites/omg-dds-spec.md`](sources/sites/omg-dds-spec.md)、[`sources/sites/fast-dds-docs.md`](sources/sites/fast-dds-docs.md)、[`sources/sites/cyclonedds-io.md`](sources/sites/cyclonedds-io.md)、[`sources/repos/fast-dds.md`](sources/repos/fast-dds.md)、[`sources/repos/cyclonedds.md`](sources/repos/cyclonedds.md)；更新合集 [`sources/sites/dds_omg_rtos_edge_ota_safety_primary_refs.md`](sources/sites/dds_omg_rtos_edge_ota_safety_primary_refs.md)、[`sources/repos/ros2.md`](sources/repos/ros2.md)
- **主升格：** [`wiki/entities/fast-dds.md`](wiki/entities/fast-dds.md)、[`wiki/entities/cyclone-dds.md`](wiki/entities/cyclone-dds.md)
- **主深化：** [`wiki/concepts/dds-communication.md`](wiki/concepts/dds-communication.md) — 标准两层 + vendor 表 + 一手来源
- **交叉：** [`wiki/concepts/ros2-basics.md`](wiki/concepts/ros2-basics.md)、[`wiki/comparisons/ros2-vs-lcm.md`](wiki/comparisons/ros2-vs-lcm.md)、[`wiki/entities/unitree-ros2.md`](wiki/entities/unitree-ros2.md)、[`wiki/queries/real-time-control-middleware-guide.md`](wiki/queries/real-time-control-middleware-guide.md)
- **机构注册：** `schema/institutions.json` → `eprosima`、`eclipse`、`omg`
- **开源核查（2026-07-28）：** Fast DDS 社区版 **已开源**（Apache-2.0；Pro 为商业扩展）；Cyclone DDS **已开源**（EPL-2.0/EDL-1.0）；OMG 规范 **公开可读**（非软件许可）

## [2026-07-28] ingest | sources/sites/lcm-proj-github-io.md + sources/repos/lcm.md + sources/sites/ros2-github-org.md + sources/repos/ros2.md — LCM 官方文档/仓与 ROS 2 组织/元仓一手入库；深化 wiki/concepts/lcm-basics.md、ros2-basics.md、comparisons/ros2-vs-lcm.md

- **一手入口：**
  - LCM 文档 [lcm-proj.github.io/lcm](https://lcm-proj.github.io/lcm/) · 仓 [lcm-proj/lcm](https://github.com/lcm-proj/lcm)（~1.2k★，LGPL-2.1，v1.5.2）
  - ROS 2 组织 [github.com/ros2](https://github.com/ros2)（~146 公开仓）· 元仓 [ros2/ros2](https://github.com/ros2/ros2)（~5.8k★，`ros2.repos`）
- **归档：** [`sources/sites/lcm-proj-github-io.md`](sources/sites/lcm-proj-github-io.md)、[`sources/repos/lcm.md`](sources/repos/lcm.md)、[`sources/sites/ros2-github-org.md`](sources/sites/ros2-github-org.md)、[`sources/repos/ros2.md`](sources/repos/ros2.md)；交叉更新 [`sources/sites/ros2-official-documentation.md`](sources/sites/ros2-official-documentation.md)
- **主深化：** [`wiki/concepts/lcm-basics.md`](wiki/concepts/lcm-basics.md)、[`wiki/concepts/ros2-basics.md`](wiki/concepts/ros2-basics.md)、[`wiki/comparisons/ros2-vs-lcm.md`](wiki/comparisons/ros2-vs-lcm.md) — 补一手特性/安装/上游拓扑、开源核查、统一骨架与选型结论
- **交叉：** [`wiki/concepts/dds-communication.md`](wiki/concepts/dds-communication.md)、[`wiki/queries/real-time-control-middleware-guide.md`](wiki/queries/real-time-control-middleware-guide.md)、[`tech-map/modules/system/ros2.md`](tech-map/modules/system/ros2.md)
- **机构注册：** `schema/institutions.json` → `open-robotics`（OSRF）；`mit` 已有
- **开源核查（2026-07-28）：** LCM **已开源**；ROS 2 组织与元仓 **已开源**（日常部署仍推荐发行版二进制）

## [2026-07-28] ingest | sources/papers/learning_to_ball_arxiv_2509_22442.md + sources/repos/learning-to-ball.md + sources/sites/pei-xu-basketball-github-io.md — Learning to Ball（arXiv:2509.22442，SIGGRAPH Asia 2025）；升格 wiki/entities/paper-notebook-learning-to-ball.md；交叉 wiki/overview/paper-notebook-category-13-physics-based-animation.md、wiki/methods/hierarchical-reinforcement-learning.md、wiki/methods/imitation-learning.md、wiki/entities/paper-notebook-skillmimic-learning-basketball-interaction-skill.md、wiki/entities/paper-notebook-composite-motion-learning-with-task-control.md

- **论文：** *Learning to Ball: Composing Policies for Long-Horizon Basketball Moves*（Stanford × UC Riverside × Roblox × Clemson；ACM TOG / SIGGRAPH Asia 2025）
- **一手入口：** [arXiv:2509.22442](https://arxiv.org/abs/2509.22442) · 项目页 [pei-xu.github.io/basketball](https://pei-xu.github.io/basketball) · 仓 [xupei0610/basketball](https://github.com/xupei0610/basketball)
- **归档：** [`sources/papers/learning_to_ball_arxiv_2509_22442.md`](sources/papers/learning_to_ball_arxiv_2509_22442.md)、[`sources/repos/learning-to-ball.md`](sources/repos/learning-to-ball.md)、[`sources/sites/pei-xu-basketball-github-io.md`](sources/sites/pei-xu-basketball-github-io.md)；更新 [`sources/papers/humanoid_pnb_learning-to-ball.md`](sources/papers/humanoid_pnb_learning-to-ball.md)
- **主升格：** [`wiki/entities/paper-notebook-learning-to-ball.md`](wiki/entities/paper-notebook-learning-to-ball.md) — Type A/B/C 过渡 + soft router；结论 / 源码运行时序图
- **交叉：** [`wiki/overview/paper-notebook-category-13-physics-based-animation.md`](wiki/overview/paper-notebook-category-13-physics-based-animation.md)、[`wiki/methods/hierarchical-reinforcement-learning.md`](wiki/methods/hierarchical-reinforcement-learning.md)、[`wiki/methods/imitation-learning.md`](wiki/methods/imitation-learning.md)、[`wiki/entities/paper-notebook-skillmimic-learning-basketball-interaction-skill.md`](wiki/entities/paper-notebook-skillmimic-learning-basketball-interaction-skill.md)、[`wiki/entities/paper-notebook-composite-motion-learning-with-task-control.md`](wiki/entities/paper-notebook-composite-motion-learning-with-task-control.md)
- **机构注册：** `schema/institutions.json` → `uc-riverside`、`roblox`、`clemson`（`stanford` 已有）
- **开源核查（2026-07-28）：** 项目页链 GitHub；仓 **已开源**（MIT）训练/评测与子技能预训练；高层 soft router / gather 独立 cfg 与预训练 **未列于公开清单**


## [2026-07-28] ingest | sources/papers/skillmimic_arxiv_2408_15270.md + sources/repos/skillmimic.md + sources/sites/ingrid789-skillmimic-github-io.md — SkillMimic（arXiv:2408.15270，CVPR 2025 Highlight）；升格 wiki/entities/paper-notebook-skillmimic-learning-basketball-interaction-skill.md；交叉 wiki/overview/paper-notebook-category-13-physics-based-animation.md、wiki/methods/ase.md、wiki/methods/hierarchical-reinforcement-learning.md、wiki/methods/imitation-learning.md、wiki/entities/paper-notebook-learning-to-ball.md

- **论文：** *SkillMimic: Learning Basketball Interaction Skills from Demonstrations*（HKUST × Unitree × PKU × Tsinghua × IDEA × Tencent × CMU；CVPR 2025 Highlight）
- **一手入口：** [arXiv:2408.15270](https://arxiv.org/abs/2408.15270) · 项目页 [ingrid789.github.io/SkillMimic](https://ingrid789.github.io/SkillMimic/) · 仓 [wyhuai/SkillMimic](https://github.com/wyhuai/SkillMimic)
- **归档：** [`sources/papers/skillmimic_arxiv_2408_15270.md`](sources/papers/skillmimic_arxiv_2408_15270.md)、[`sources/repos/skillmimic.md`](sources/repos/skillmimic.md)、[`sources/sites/ingrid789-skillmimic-github-io.md`](sources/sites/ingrid789-skillmimic-github-io.md)；更新 [`sources/papers/humanoid_pnb_skillmimic-learning-basketball-interaction-skill.md`](sources/papers/humanoid_pnb_skillmimic-learning-basketball-interaction-skill.md)
- **主升格：** [`wiki/entities/paper-notebook-skillmimic-learning-basketball-interaction-skill.md`](wiki/entities/paper-notebook-skillmimic-learning-basketball-interaction-skill.md) — 统一 HOI 模仿 + Contact Graph + HLC；结论 / 源码运行时序图
- **交叉：** [`wiki/overview/paper-notebook-category-13-physics-based-animation.md`](wiki/overview/paper-notebook-category-13-physics-based-animation.md)、[`wiki/methods/ase.md`](wiki/methods/ase.md)、[`wiki/methods/hierarchical-reinforcement-learning.md`](wiki/methods/hierarchical-reinforcement-learning.md)、[`wiki/methods/imitation-learning.md`](wiki/methods/imitation-learning.md)、[`wiki/entities/paper-notebook-learning-to-ball.md`](wiki/entities/paper-notebook-learning-to-ball.md)
- **开源核查（2026-07-28）：** 项目页链 GitHub；仓 **已开源** 训练/评测/预训练 LLC·HLC/BallPlay-M 子集/Blender（Apache-2.0）；完整原始 BallPlay-M + 数据处理 **TODO**

## [2026-07-28] ingest | sources/blogs/wechat_shenlan_sim2real_sysid_to_adaptation.md — 深蓝具身智能：Sim2Real 非训后一步；升格 wiki/queries/sim2real-closed-loop-engineering.md；交叉 wiki/concepts/sim2real.md、wiki/concepts/system-identification.md、wiki/concepts/domain-randomization.md、wiki/concepts/curriculum-learning.md、wiki/concepts/privileged-training.md、wiki/entities/paper-rma-rapid-motor-adaptation.md、wiki/queries/sim2real-checklist.md、wiki/queries/sim2real-gap-reduction.md、wiki/overview/topic-sim2real.md

- **公众号：** [最大误区：Sim-to-Real 不是训完之后的事情](https://mp.weixin.qq.com/s/6rbLz_6nQz9z6kma9K4BFQ)（深蓝具身智能，2026-07-28）
- **抓取：** Agent Reach v1.5.0 + wechat-article-for-ai（Camoufox；`--no-images`）
- **归档：** [`sources/blogs/wechat_shenlan_sim2real_sysid_to_adaptation.md`](sources/blogs/wechat_shenlan_sim2real_sysid_to_adaptation.md) · raw [`sources/raw/wechat_shenlan_sim2real_sysid_to_adaptation_2026-07-28.md`](sources/raw/wechat_shenlan_sim2real_sysid_to_adaptation_2026-07-28.md)
- **主升格：** [`wiki/queries/sim2real-closed-loop-engineering.md`](wiki/queries/sim2real-closed-loop-engineering.md) — 误差分流 + SysID→训练→前馈/RMA→分层安全闭环
- **交叉：** [`wiki/concepts/sim2real.md`](wiki/concepts/sim2real.md)、[`wiki/concepts/system-identification.md`](wiki/concepts/system-identification.md)、[`wiki/concepts/domain-randomization.md`](wiki/concepts/domain-randomization.md)、[`wiki/concepts/curriculum-learning.md`](wiki/concepts/curriculum-learning.md)、[`wiki/concepts/privileged-training.md`](wiki/concepts/privileged-training.md)、[`wiki/entities/paper-rma-rapid-motor-adaptation.md`](wiki/entities/paper-rma-rapid-motor-adaptation.md)、[`wiki/queries/sim2real-checklist.md`](wiki/queries/sim2real-checklist.md)、[`wiki/queries/sim2real-gap-reduction.md`](wiki/queries/sim2real-gap-reduction.md)、[`wiki/overview/topic-sim2real.md`](wiki/overview/topic-sim2real.md)
- **开源核查：** 课程宣传综述，无独立项目仓；文内锚点开源状态以各实体页为准

## [2026-07-28] ingest | sources/repos/humanoid-motion-intelligence.md — 接入具身智能研究室「人形机器人运动智能知识库」；升格 wiki/entities/humanoid-motion-intelligence.md

- **一手入口：** [RealXiaoze/humanoid-motion-intelligence](https://github.com/RealXiaoze/humanoid-motion-intelligence)（`main`；约 22★；分层许可 CC BY-NC-SA 4.0 + MIT 校验脚本）
- **归档：** [`sources/repos/humanoid-motion-intelligence.md`](sources/repos/humanoid-motion-intelligence.md)
- **主升格：** [`wiki/entities/humanoid-motion-intelligence.md`](wiki/entities/humanoid-motion-intelligence.md) — 六条技术路线 + ~145 论文解读 + ~166 开源主表 + 产业/求职；与微信策展同源
- **交叉：** [`wiki/overview/humanoid-rl-motion-control-body-system-stack.md`](wiki/overview/humanoid-rl-motion-control-body-system-stack.md)、[`wiki/overview/humanoid-motion-cerebellum-technology-map.md`](wiki/overview/humanoid-motion-cerebellum-technology-map.md)、[`wiki/queries/open-source-motion-control-projects.md`](wiki/queries/open-source-motion-control-projects.md)、[`sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md`](sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md)、[`sources/repos/awesome-humanoid-robot-learning.md`](sources/repos/awesome-humanoid-robot-learning.md)
- **机构注册：** `schema/institutions.json` → `embodied-ai-lab`（具身智能研究室）
- **开源核查（2026-07-28）：** 知识库 **已公开**；**无可运行训练入口**（策展仓，复现走各论文官方仓）

## [2026-07-28] structural | roadmap/motion-control.md — 主路线补齐 L5.4 动作重定向（此前只在 L5.3 模仿学习里以一条 bullet 出现，主干缺独立章节）

- **主更新：** [`roadmap/motion-control.md`](roadmap/motion-control.md) — L5.3 与 L6 之间新增 **L5.4 动作重定向**：场景隐喻 / 上一层局限 + 英文缩写速查（L5.4）+ 前置知识 → 核心问题 → 推荐做什么 → 推荐读什么 → 学完输出什么 + 3 道自测题（目标函数与硬约束、运动学可行 ≠ 动力学可行、遥操作 vs 批量造数据的选型）与 mermaid 参考答案
- **同步一致性：** 摘要「L5：RL / IL / 动作重定向扩展层」；L−1「Learning-based 主线」链路补 motion retargeting；L5 总览缩写表补 Retarget；L5.3「用 MoCap 做 retargeting」bullet 指向 L5.4；可选纵深表衔接点由「L2 + L5.3 之后」改为「L5.4 展开」
- **交叉入口：** [`wiki/concepts/motion-retargeting.md`](wiki/concepts/motion-retargeting.md)、[`wiki/concepts/motion-retargeting-pipeline.md`](wiki/concepts/motion-retargeting-pipeline.md)、[`wiki/formalizations/motion-retargeting-objective.md`](wiki/formalizations/motion-retargeting-objective.md)、[`wiki/comparisons/gmr-vs-nmr-vs-reactor.md`](wiki/comparisons/gmr-vs-nmr-vs-reactor.md)、[`wiki/concepts/kinematic-vs-dynamic-feasibility.md`](wiki/concepts/kinematic-vs-dynamic-feasibility.md)、[`wiki/concepts/motion-data-quality.md`](wiki/concepts/motion-data-quality.md)、[`wiki/comparisons/humanoid-reference-motion-datasets.md`](wiki/comparisons/humanoid-reference-motion-datasets.md)、[`wiki/methods/motion-retargeting-gmr.md`](wiki/methods/motion-retargeting-gmr.md)、[`wiki/methods/deepmimic.md`](wiki/methods/deepmimic.md)、[`wiki/methods/amp-reward.md`](wiki/methods/amp-reward.md)、[`wiki/entities/amass.md`](wiki/entities/amass.md)、[`wiki/entities/lafan1-dataset.md`](wiki/entities/lafan1-dataset.md)
- **纵深回链：** [`roadmap/depth-motion-retargeting.md`](roadmap/depth-motion-retargeting.md) 的「和主路线的关系」改为挂在 L5.4 上（原表述为「L2 与 L5 之间的数据侧展开」）

## [2026-07-28] ingest | sources/papers/icrowdnav_arxiv_2606_26047.md、sources/repos/icrowdnav.md、sources/sites/broln7-socialbev-io.md — iCrowdNav（arXiv:2606.26047，RA-L 2026）视觉人群导航；升格 wiki/entities/paper-icrowdnav.md；交叉 wiki/concepts/sim2real.md、wiki/methods/dwa.md、wiki/comparisons/mobile-robot-navigation-planning-methods.md、wiki/overview/navigation-slam-autonomy-stack.md、wiki/entities/paper-sru-spatially-enhanced-recurrent-memory.md、wiki/entities/paper-navwam-goal-conditioned-visual-navigation-wam.md、wiki/entities/paper-notebook-learning-social-navigation-from-positive-and-neg.md、roadmap/depth-navigation.md；注册 peng-cheng-lab / reconova；代码仓待发布

- **论文：** *Learning Robot Visual Navigation in Crowds via Intention-Aware Scene Representations*（SUSTech × Reconova × 鹏城实验室；RA-L 2026）
- **一手入口：** [arXiv:2606.26047](https://arxiv.org/abs/2606.26047) · 项目页 [socialbev.io](https://broln7.github.io/socialbev.io/) · 仓 [BRoln7/icrowdnav](https://github.com/BRoln7/icrowdnav)
- **归档：** [`sources/papers/icrowdnav_arxiv_2606_26047.md`](sources/papers/icrowdnav_arxiv_2606_26047.md)、[`sources/repos/icrowdnav.md`](sources/repos/icrowdnav.md)、[`sources/sites/broln7-socialbev-io.md`](sources/sites/broln7-socialbev-io.md)
- **主升格：** [`wiki/entities/paper-icrowdnav.md`](wiki/entities/paper-icrowdnav.md) — 时空 BEV + I²Former；SocNav-Gym；真机零样本；结论 / 源码时序图（不适用）
- **交叉：** [`wiki/concepts/sim2real.md`](wiki/concepts/sim2real.md)、[`wiki/methods/dwa.md`](wiki/methods/dwa.md)、[`wiki/comparisons/mobile-robot-navigation-planning-methods.md`](wiki/comparisons/mobile-robot-navigation-planning-methods.md)、[`wiki/overview/navigation-slam-autonomy-stack.md`](wiki/overview/navigation-slam-autonomy-stack.md)、[`wiki/entities/paper-sru-spatially-enhanced-recurrent-memory.md`](wiki/entities/paper-sru-spatially-enhanced-recurrent-memory.md)、[`wiki/entities/paper-navwam-goal-conditioned-visual-navigation-wam.md`](wiki/entities/paper-navwam-goal-conditioned-visual-navigation-wam.md)、[`wiki/entities/paper-notebook-learning-social-navigation-from-positive-and-neg.md`](wiki/entities/paper-notebook-learning-social-navigation-from-positive-and-neg.md)、[`roadmap/depth-navigation.md`](roadmap/depth-navigation.md)
- **机构注册：** `schema/institutions.json` → `peng-cheng-lab`、`reconova`（`sustech` 已有）
- **开源核查（2026-07-28）：** 仓仅 README + 附录 + 演示 GIF；**TODO: Release codes of iCrowdNav** → **待发布**

## [2026-07-28] ingest | sources/papers/llada2_2_tech_report.md + sources/repos/llada2-x.md + HF/ModelScope — 接入 LLaDA2.2-flash（Inclusion AI / Ant Group）：Levenshtein Editing dLLM；升格 wiki/entities/llada2-2-flash.md

- **一手开源面（2026-07-28 核查）：**
  - GitHub [`inclusionAI/LLaDA2.X`](https://github.com/inclusionAI/LLaDA2.X) → [`sources/repos/llada2-x.md`](sources/repos/llada2-x.md)（README / Apache-2.0 / 多版 tech report；仓内无训练脚本）
  - HF [`inclusionAI/LLaDA2.2-flash`](https://huggingface.co/inclusionAI/LLaDA2.2-flash) → [`sources/sites/huggingface-inclusionai-llada2-2-flash.md`](sources/sites/huggingface-inclusionai-llada2-2-flash.md)（32× safetensors，约 **192 GiB**；custom `generate` + DELETE/SPLIT）
  - ModelScope [`inclusionAI/LLaDA2.2-flash`](https://modelscope.cn/models/inclusionAI/LLaDA2.2-flash) → [`sources/sites/modelscope-inclusionai-llada2-2-flash.md`](sources/sites/modelscope-inclusionai-llada2-2-flash.md)
  - 技术报告 PDF → [`sources/papers/llada2_2_tech_report.md`](sources/papers/llada2_2_tech_report.md)（11 页；无独立 arXiv；CPT 128K + Block Routing + L-EBPO）
- **主升格：** [`wiki/entities/llada2-2-flash.md`](wiki/entities/llada2-2-flash.md) — 100B MoE dLLM；Levenshtein Editing；agentic 评测与吞吐；部署时序图
- **交叉：** [`wiki/concepts/diffusion-model.md`](wiki/concepts/diffusion-model.md)、[`wiki/entities/kimi-k3.md`](wiki/entities/kimi-k3.md)、[`wiki/queries/real-robot-policy-autoresearch-harness.md`](wiki/queries/real-robot-policy-autoresearch-harness.md)
- **机构注册：** `schema/institutions.json` → `inclusion-ai`、`ant-group`、`westlake`
- **开源结论：** **开放权重 + 技术报告 + HF custom 推理已开源**（Apache-2.0）；训练代码/数据未随公开仓发布；SGLang 对 2.2 标 coming soon；系列推理/微调见 dInfer / dFactory

## [2026-07-28] ingest | sources/papers/adams_orlandea_primary_refs.md — 接入 ADAMS（Automatic Dynamic Analysis of Mechanical Systems）一手学术链并升格 wiki/entities/adams.md

- **一手学术：** [`sources/papers/adams_orlandea_primary_refs.md`](sources/papers/adams_orlandea_primary_refs.md)
  - Orlandea 1973 密歇根博士论文（DOI [10.7302/10731](https://doi.org/10.7302/10731)；Deep Blue 校园限制 PDF）
  - Maros & Orlandea 1971（DOI [10.1115/1.3427874](https://doi.org/10.1115/1.3427874)）
  - Orlandea–Chace–Calahan 1977 ASME Part 1 / Part 2（DOI [10.1115/1.3439312](https://doi.org/10.1115/1.3439312) · [10.1115/1.3439313](https://doi.org/10.1115/1.3439313)）
  - Orlandea 2016 *Multibody Systems History of ADAMS*（DOI [10.1115/1.4034296](https://doi.org/10.1115/1.4034296)）
- **站点 / 交叉叙述：** [`sources/sites/umich-deepblue-orlandea-adams-thesis.md`](sources/sites/umich-deepblue-orlandea-adams-thesis.md) · [`sources/sites/cadence-msc-adams.md`](sources/sites/cadence-msc-adams.md) · [`sources/blogs/janevic_orlandea_adams_memorial.md`](sources/blogs/janevic_orlandea_adams_memorial.md)
- **主升格：** [`wiki/entities/adams.md`](wiki/entities/adams.md) — 工业 MBD 谱系；与 MuJoCo/Drake/RL 仿真栈分工
- **交叉：** [`wiki/entities/mujoco.md`](wiki/entities/mujoco.md)、[`wiki/entities/drake.md`](wiki/entities/drake.md)、[`wiki/overview/sim-platforms-decade-technology-map.md`](wiki/overview/sim-platforms-decade-technology-map.md)、[`wiki/queries/simulator-selection-guide.md`](wiki/queries/simulator-selection-guide.md)
- **机构注册：** `schema/institutions.json` → `university-of-michigan`、`iowa-state`、`cadence`、`msc-software`、`mechanical-dynamics`
- **开源核查（2026-07-28）：** 方法论文已发表；当代 Adams **确认未开源**（商业 CAE）；源码运行时序图不适用

## [2026-07-28] ingest | sources/papers/yahmp_arxiv_2607_19903.md + sources/repos/yahmp.md — 复核 YAHMP（arXiv:2607.19903，2026-07-22）：对齐论文声明代码 hucebot/yahmp；刷新 wiki/entities/paper-yahmp.md

- **论文：** *What Matters in Humanoid General Motion Tracking? An Empirical Study*（Inria / Université de Lorraine / CNRS · HUCEBOT；Submitted 2026-07-22）
- **开源状态（2026-07-28 再核）：** **已开源** — 论文声明 <https://github.com/hucebot/yahmp>（Apache-2.0）；开发上游 <https://github.com/fabio-amadio/yahmp>（parent，tip 领先 1 commit：`expand_npz_motion_dataset`）
- **刷新 papers / repos：** [`sources/papers/yahmp_arxiv_2607_19903.md`](sources/papers/yahmp_arxiv_2607_19903.md)、[`sources/repos/yahmp.md`](sources/repos/yahmp.md) — 双仓 tip 关系、手部力 ≤20 N、真机更硬 PD 踝振荡
- **主刷新 wiki：** [`wiki/entities/paper-yahmp.md`](wiki/entities/paper-yahmp.md) — `code` 对齐 hucebot；补 Table II 绝对量级与真机踝振荡读点；双仓选型说明
- **说明：** 该文已于 2026-07-24 首次 ingest；本次为对齐用户给定项目链接与开源再核的刷新，不重复造页

## [2026-07-27] query | wiki/queries/robot-perception-stack-selection-loop.md + wiki/concepts/2d-to-3d-semantic-lifting-gap.md — 沉淀「机器人视觉感知栈选型闭环」知识链（V31 P1 首项）；从 object-detection-model-selection / perception-backbone-selection 双向回链消孤儿

- **背景（V31 P1）：** 近周密集 ingest 了一批目标检测 / 分割 / 2D→3D 语义建图资料（Ultralytics YOLO、RF-DETR、SAM/SAM2、FindAnything、OV-SAM3D、CMU MSCV Semantic 3D 等），但各页独立、缺一条贯通的感知栈选型视角。本次把它们沉淀为一条位于策略**输入端**的四层选型链，与[执行器驱动链选型闭环](wiki/queries/actuator-drive-chain-selection-loop.md)（输出端）互为镜像。
- **主新建（端到端 Query）：** [`wiki/queries/robot-perception-stack-selection-loop.md`](wiki/queries/robot-perception-stack-selection-loop.md) — 传感与标定 → 2D 检测/分割选型 → 2D→3D 提升与语义建图 → 下游策略消费 四层选型决策树（配 Mermaid），逐层给「选什么 / 精度 vs 时延算力 / 闭集 vs 开放词汇 / 2D 框 vs 3D 语义几何 / 感知频率 ≠ 控制带宽」的取舍与典型误判，附矛盾速查与失败模式速查表
- **主新建（物理根因概念）：** [`wiki/concepts/2d-to-3d-semantic-lifting-gap.md`](wiki/concepts/2d-to-3d-semantic-lifting-gap.md) — 「2D 检测/分割结果 ↔ 可供策略消费的 3D 语义几何」取舍概念页：四类信息损失与歧义（尺度不确定 / 遮挡 / 时序不一致 / 语义-几何分离）、提升成立条件表、深度融合 / 多视角一致性 / 语义-几何联合建图三条收窄路线；与 Query 页双向回链
- **消孤儿回链：** [`wiki/queries/object-detection-model-selection.md`](wiki/queries/object-detection-model-selection.md)、[`wiki/queries/perception-backbone-selection.md`](wiki/queries/perception-backbone-selection.md) 的 `related` 补入新 Query/概念页，双向闭合（`graph-stats.json` 0 orphans）
- **派生同步：** `make graph` + `make export` + `make catalog` 重生成；图谱 **1949 → 1951 节点 / 16856 → 16888 边**（V31 目标 ≥1924 / ≥16460 已满足）；`make lint` 0 errors（仅 LICENSE 等既有信息型预警）
- **执行清单：** [`docs/checklists/tech-stack-next-phase-checklist-v31.md`](docs/checklists/tech-stack-next-phase-checklist-v31.md) P1「机器人视觉感知栈选型闭环知识链 (+2)」勾选完成

## [2026-07-27] ingest | sources/repos/kimi-k3.md + HF/ModelScope/tech report — Kimi K3 开放权重与技术报告一手资料；刷新 wiki/entities/kimi-k3.md；交叉 muon / enpire / autoresearch harness

- **一手开源面（2026-07-27 核查）：**
  - GitHub [`MoonshotAI/Kimi-K3`](https://github.com/MoonshotAI/Kimi-K3) → [`sources/repos/kimi-k3.md`](sources/repos/kimi-k3.md)（README / **Kimi K3 License** / `k3_tech_report.pdf`；仓内无训练脚本）
  - HF [`moonshotai/Kimi-K3`](https://huggingface.co/moonshotai/Kimi-K3) → [`sources/sites/huggingface-moonshotai-kimi-k3.md`](sources/sites/huggingface-moonshotai-kimi-k3.md)（96× MXFP4 safetensors，约 **1.56 TB**；custom code）
  - ModelScope [`moonshotai/Kimi-K3`](https://www.modelscope.cn/models/moonshotai/Kimi-K3) → [`sources/sites/modelscope-moonshotai-kimi-k3.md`](sources/sites/modelscope-moonshotai-kimi-k3.md)
  - 技术报告 PDF → [`sources/papers/kimi_k3_tech_report.md`](sources/papers/kimi_k3_tech_report.md)（47 页；无 arXiv；激活 **104B**、69 KDA + 24 Gated MLA、MoonViT-V2）
- **既有资料同步：** [`sources/blogs/kimi_k3_tech_blog.md`](sources/blogs/kimi_k3_tech_blog.md)、[`sources/courses/kimi_k3_api_quickstart.md`](sources/courses/kimi_k3_api_quickstart.md) — 权重状态改为已开源；`reasoning_effort` 档位补 `low/high/max`
- **主刷新：** [`wiki/entities/kimi-k3.md`](wiki/entities/kimi-k3.md) — 开源状态表、部署时序图、规格与 License 风险
- **交叉：** [`wiki/methods/muon.md`](wiki/methods/muon.md)、[`wiki/methods/enpire.md`](wiki/methods/enpire.md)、[`wiki/queries/real-robot-policy-autoresearch-harness.md`](wiki/queries/real-robot-policy-autoresearch-harness.md)
- **开源结论：** **开放权重 + 技术报告已开源**；训练代码/数据未随公开仓发布；推理走 vLLM / SGLang / TokenSpeed 官方 recipe

## [2026-07-27] ingest | sources/sites/3dgenstudio-com.md + sources/repos/3dgenstudio.md — 升格 wiki/entities/3dgenstudio.md；交叉 text-to-cad / blender / articraft / img2threejs / freecad-mcp；开源核查：Community License 已开源（禁 SaaS 转售）

- **官网 / 仓：** [`sources/sites/3dgenstudio-com.md`](sources/sites/3dgenstudio-com.md)（<https://www.3dgenstudio.com/>，v2.1.0）· [`sources/repos/3dgenstudio.md`](sources/repos/3dgenstudio.md)（<https://github.com/visualbruno/3DGenStudio>）
- **主升格：** [`wiki/entities/3dgenstudio.md`](wiki/entities/3dgenstudio.md) — 本地优先 AI 网格生产工作台：Kanban/Graph + ComfyUI + Mesh Editor + MCP；导出 GLB/OBJ/FBX
- **交叉：** [`wiki/concepts/text-to-cad.md`](wiki/concepts/text-to-cad.md)、[`wiki/entities/blender.md`](wiki/entities/blender.md)、[`wiki/entities/articraft.md`](wiki/entities/articraft.md)、[`wiki/entities/img2threejs.md`](wiki/entities/img2threejs.md)、[`wiki/entities/freecad-mcp.md`](wiki/entities/freecad-mcp.md)；索引 [`sources/sites/text-to-cad-tools.md`](sources/sites/text-to-cad-tools.md)
- **开源核查（2026-07-27）：** **已开源** — Community License（可自用/改代码与商用生成物；禁止转售软件本体与付费 SaaS 托管）；依赖本地 ComfyUI + 可选第三方 API
- **机构注册：** `schema/institutions.json` → `3d-gen-studio`（三维生成工作室（3D Gen Studio））

## [2026-07-27] structural | roadmap/depth-motion-retargeting.md — 新增 Stage 4 工程工具链与轨迹编辑器（原 Stage 4/5 顺延为 5/6），接入 wiki/entities/robot-motion-keyframe-editors.md、wiki/entities/human-humanoid-tools.md、wiki/entities/soma-retargeter.md、wiki/entities/robot-retargeter.md、wiki/entities/mocap-retarget.md 等一手工具资料

- **主更新：** [`roadmap/depth-motion-retargeting.md`](roadmap/depth-motion-retargeting.md) — 新增 **Stage 4 工程工具链与轨迹编辑器**：工具形态谱系（脚本级 / 库级 / 工作台级 / 框架内置 / 编辑器级）+ 三条一手编辑链路 + 四元数顺序与 FPS 重采样踩坑；原「下游闭环 / 进阶方向」顺延为 Stage 5 / Stage 6（站点 `export_minimal.py` 只解析整数 Stage，小数号不会进阶段速览）；同步更新路线图 Mermaid、前置知识链与快速入口表
- **接入的一手工具页：** [`wiki/entities/robot-motion-keyframe-editors.md`](wiki/entities/robot-motion-keyframe-editors.md)（cyoahs 浏览器 URDF+CSV / Stanford-TML MJCF+Mink IK / Project Instinct Flask+NPZ）、[`wiki/entities/human-humanoid-tools.md`](wiki/entities/human-humanoid-tools.md)、[`wiki/entities/soma-retargeter.md`](wiki/entities/soma-retargeter.md)、[`wiki/entities/robot-retargeter.md`](wiki/entities/robot-retargeter.md)、[`wiki/entities/mocap-retarget.md`](wiki/entities/mocap-retarget.md)、[`wiki/entities/human2humanoid.md`](wiki/entities/human2humanoid.md)、[`wiki/entities/mimickit.md`](wiki/entities/mimickit.md)、[`wiki/entities/fairmotion.md`](wiki/entities/fairmotion.md)、[`wiki/entities/blender.md`](wiki/entities/blender.md)、[`wiki/entities/robot-viewer.md`](wiki/entities/robot-viewer.md)、[`wiki/entities/generative-motion-rig.md`](wiki/entities/generative-motion-rig.md)
- **一手仓库来源：** [`sources/repos/cyoahs-robot-motion-editor.md`](sources/repos/cyoahs-robot-motion-editor.md)、[`sources/repos/stanford-tml-robot-keyframe-kit.md`](sources/repos/stanford-tml-robot-keyframe-kit.md)、[`sources/repos/project-instinct-robot-motion-editor.md`](sources/repos/project-instinct-robot-motion-editor.md)
- **回链：** [`wiki/entities/robot-motion-keyframe-editors.md`](wiki/entities/robot-motion-keyframe-editors.md) 关联页面补上纵深路线 Stage 4，消除该页只被概念页单向引用的状态

## [2026-07-27] ingest | sources/papers/worldscore_arxiv_2504_00983.md — WorldScore 统一世界生成评测（ICCV 2025）；升格 wiki/entities/paper-worldscore.md；交叉 ewmbench / generative-world-models / video-as-simulation / topic-embodied-eval-benchmark / embodied-eval-benchmark-selection-loop；归档 sites/haoyi-duan-worldscore-github-io.md、sites/worldscore-leaderboard-hf.md、repos/worldscore.md；开源核查：MIT 代码 + HF Dataset + HF Leaderboard 已开源

- **论文 / 项目页 / 仓 / 榜：** [`sources/papers/worldscore_arxiv_2504_00983.md`](sources/papers/worldscore_arxiv_2504_00983.md)（arXiv:2504.00983，ICCV 2025）· [`sources/sites/haoyi-duan-worldscore-github-io.md`](sources/sites/haoyi-duan-worldscore-github-io.md)（<https://haoyi-duan.github.io/WorldScore/>）· [`sources/repos/worldscore.md`](sources/repos/worldscore.md)（<https://github.com/haoyi-duan/WorldScore>）· [`sources/sites/worldscore-leaderboard-hf.md`](sources/sites/worldscore-leaderboard-hf.md)（<https://huggingface.co/spaces/Howieeeee/WorldScore_Leaderboard>；HF Dataset Howieeeee/WorldScore）
- **主升格：** [`wiki/entities/paper-worldscore.md`](wiki/entities/paper-worldscore.md) — next-scene + 显式相机轨迹；3000 例；Ctrl/Quality/Dynamics → Static/Dynamic；统一评 3D/4D/I2V/T2V
- **交叉：** [`wiki/entities/ewmbench.md`](wiki/entities/ewmbench.md)、[`wiki/methods/generative-world-models.md`](wiki/methods/generative-world-models.md)、[`wiki/concepts/video-as-simulation.md`](wiki/concepts/video-as-simulation.md)、[`wiki/overview/topic-embodied-eval-benchmark.md`](wiki/overview/topic-embodied-eval-benchmark.md)、[`wiki/queries/embodied-eval-benchmark-selection-loop.md`](wiki/queries/embodied-eval-benchmark-selection-loop.md)
- **开源核查（2026-07-27）：** **已开源** — MIT 评测仓 + HF static/dynamic 数据集 + HF Leaderboard（可提交 `worldscore.json`）；评测依赖 DROID-SLAM/SAM/VFI 较重

## [2026-07-27] ingest | sources/papers/diffgi_arxiv_2607_13365.md — DiffGI 可微 TSDF geometry image（ECCV 2026）；升格 wiki/entities/paper-diffgi.md；交叉 clothtransformer / physforge / articraft；归档 sites/ejshim-diffgi-github-io.md、repos/diffgi.md；注册 clo-virtual-fashion；开源核查：Code (soon)/仓内仅 docs，待发布

- **论文 / 项目页 / 仓：** [`sources/papers/diffgi_arxiv_2607_13365.md`](sources/papers/diffgi_arxiv_2607_13365.md)（arXiv:2607.13365）· [`sources/sites/ejshim-diffgi-github-io.md`](sources/sites/ejshim-diffgi-github-io.md)（<https://ejshim.github.io/diffgi/>）· [`sources/repos/diffgi.md`](sources/repos/diffgi.md)（<https://github.com/EJShim/diffgi>；仅 docs）
- **主升格：** [`wiki/entities/paper-diffgi.md`](wiki/entities/paper-diffgi.md) — 连续 2D TSDF geometry image + Differentiable Marching Squares + \(32\times32\) 潜扩散薄壳生成
- **交叉：** [`wiki/entities/paper-clothtransformer-unified-latent-cloth-simulation.md`](wiki/entities/paper-clothtransformer-unified-latent-cloth-simulation.md)、[`wiki/entities/paper-physforge-physics-grounded-3d-assets.md`](wiki/entities/paper-physforge-physics-grounded-3d-assets.md)、[`wiki/entities/articraft.md`](wiki/entities/articraft.md)
- **机构注册：** `schema/institutions.json` → `clo-virtual-fashion`（科洛虚拟时尚（CLO Virtual Fashion））
- **开源核查（2026-07-27）：** **宣称将开源 / 待发布** — 项目页 **Code (soon)**；公开仓仅项目页静态资源，无可运行训练/推理入口

## [2026-07-27] ingest | sources/papers/chronos_arxiv_2606_30318.md — Chronos 全历史 SSM + IMLE + 二阶桥；升格 wiki/entities/paper-chronos.md；交叉 vla / manipulation / robotwin / eventvla / kemo / fm-vla / action-chunking / diffusion-policy / imitation-learning；归档 sites/chronos-manipulation-github-io.md、repos/chronos.md

- **论文 / 项目页 / 仓：** [`sources/papers/chronos_arxiv_2606_30318.md`](sources/papers/chronos_arxiv_2606_30318.md)（arXiv:2606.30318）· [`sources/sites/chronos-manipulation-github-io.md`](sources/sites/chronos-manipulation-github-io.md)（<https://chronos-manipulation.github.io/>）· [`sources/repos/chronos.md`](sources/repos/chronos.md)（<https://github.com/yulinzhouZYL/Chronos>；HF Chronos-RMBench）
- **主升格：** [`wiki/entities/paper-chronos.md`](wiki/entities/paper-chronos.md) — 非马尔可夫长程操作：全历史 SSM + IMLE 粗先验 + 二阶 Schrödinger 加速度桥
- **交叉：** [`wiki/methods/vla.md`](wiki/methods/vla.md)、[`wiki/tasks/manipulation.md`](wiki/tasks/manipulation.md)、[`wiki/entities/robotwin.md`](wiki/entities/robotwin.md)、[`wiki/entities/paper-eventvla-visual-evidence-memory.md`](wiki/entities/paper-eventvla-visual-evidence-memory.md)、[`wiki/entities/paper-kemo-event-driven-keyframe-memory-vla.md`](wiki/entities/paper-kemo-event-driven-keyframe-memory-vla.md)、[`wiki/entities/paper-fm-vla.md`](wiki/entities/paper-fm-vla.md)、[`wiki/methods/action-chunking.md`](wiki/methods/action-chunking.md)、[`wiki/methods/diffusion-policy.md`](wiki/methods/diffusion-policy.md)、[`wiki/methods/imitation-learning.md`](wiki/methods/imitation-learning.md)
- **开源核查（2026-07-27）：** **已开源（部分）** — RMBench 仿真 + 真机 UR3 管线与 HF ckpt（MIT）；ALOHA / RoboTwin 2.0 清理代码 Coming soon

## [2026-07-27] ingest | sources/sites/lumina-embodied-ai.md — Lumina 具身智能社区官网；升格 wiki/entities/lumina-embodied.md；交叉 waytoagi / robotwin / isaac-sim / vla / openlet / vla-open-source-repro-landscape；归档 sources/repos/lumina-eai.md 并回链 embodied-ai-guide

- **官网 / Org：** [`sources/sites/lumina-embodied-ai.md`](sources/sites/lumina-embodied-ai.md)（<https://lumina-embodied.ai/>）· [`sources/repos/lumina-eai.md`](sources/repos/lumina-eai.md)（<https://github.com/Lumina-EAI>）· 百科仓回链 [`sources/repos/embodied-ai-guide.md`](sources/repos/embodied-ai-guide.md)
- **主升格：** [`wiki/entities/lumina-embodied.md`](wiki/entities/lumina-embodied.md) — Talks / Events / Guide / Isaac Sim 一百讲 / Lumina-Call / EAI-100 社区导航锚点
- **交叉：** [`wiki/entities/waytoagi.md`](wiki/entities/waytoagi.md)、[`wiki/entities/robotwin.md`](wiki/entities/robotwin.md)、[`wiki/entities/isaac-sim.md`](wiki/entities/isaac-sim.md)、[`wiki/methods/vla.md`](wiki/methods/vla.md)、[`wiki/entities/openlet.md`](wiki/entities/openlet.md)、[`wiki/overview/vla-open-source-repro-landscape-2025.md`](wiki/overview/vla-open-source-repro-landscape-2025.md)
- **开源核查（2026-07-27）：** 官网公开；Org **已开源** Awesome-EmbodiedAI-Jobs；技术百科在 `tianxingchen/Embodied-AI-Guide`（非 Org 内）

## [2026-07-27] ingest | sources/repos/wolfiemouse.md + UKMARSBOT/Micromouse 集群 — 升格 wiki/concepts/micromouse.md、wiki/entities/ukmarsbot.md、wiki/entities/wolfiemouse.md；交叉 a-star / pid-control / kicad / mushr；归档 opatiny/lime7/emstef/ianmhoffman/ukmars/micromouseonline 与两则 YouTube；注册 ukmars

- **主升格：** [`wiki/concepts/micromouse.md`](wiki/concepts/micromouse.md) · [`wiki/entities/ukmarsbot.md`](wiki/entities/ukmarsbot.md) · [`wiki/entities/wolfiemouse.md`](wiki/entities/wolfiemouse.md)
- **Repos：** WolfieMouse / opatiny Algernon / lime7git / UKMARSBOT / emstef Webots / Ian Hoffman；站点 UKMARS · Micromouse Online；视频 Algernon debug · UKMARSBOT 从零
- **交叉：** [`wiki/methods/a-star.md`](wiki/methods/a-star.md)、[`wiki/methods/pid-control.md`](wiki/methods/pid-control.md)、[`wiki/entities/kicad.md`](wiki/entities/kicad.md)、[`wiki/entities/mushr.md`](wiki/entities/mushr.md)
- **机构注册：** `schema/institutions.json` → `ukmars`、`ieee`
- **开源核查（2026-07-27）：** UKMARSBOT/WolfieMouse/Algernon/Webots **已开源**；lime7 **公开缺根 LICENSE**；Ian Hoffman **设计文档为主 / 部分**；micromouseonline.com **非 IEEE 法人官网**

## [2026-07-27] ingest | sources/repos/oomwoo.md — OOMWOO 开源家用扫地机；升格 wiki/entities/oomwoo.md；交叉 navigation2 / slam-toolbox / navigation-slam-autonomy-stack / mushr；归档 sources/sites/oomwoo-com.md；注册 makerspet

- **主仓 / 项目页：** [`sources/repos/oomwoo.md`](sources/repos/oomwoo.md)（<https://github.com/makerspet/oomwoo>；~6.5k★；Apache-2.0）· [`sources/sites/oomwoo-com.md`](sources/sites/oomwoo-com.md)（<https://oomwoo.com/>）
- **主升格：** [`wiki/entities/oomwoo.md`](wiki/entities/oomwoo.md) — ROS 2 / Nav2 / slam_toolbox + CM4·CM5 / STM32 安全分层；仿真优先、本地优先
- **交叉：** [`wiki/entities/navigation2.md`](wiki/entities/navigation2.md)、[`wiki/entities/slam-toolbox.md`](wiki/entities/slam-toolbox.md)、[`wiki/overview/navigation-slam-autonomy-stack.md`](wiki/overview/navigation-slam-autonomy-stack.md)、[`wiki/entities/mushr.md`](wiki/entities/mushr.md)
- **机构注册：** `schema/institutions.json` → `makerspet`（创客宠物（Maker's Pet））
- **开源核查（2026-07-27）：** **部分开源 / 早期开发**（主仓 + oomwoo-one / oomwoo-install 已开；完整 BoM / CAD / I/O 板与固件进行中）

## [2026-07-27] ingest | sources/sites/molingo-github-io.md — MoLingo（CVPR 2026）语义对齐连续 latent T2M；升格 wiki/entities/paper-molingo.md；交叉 diffusion-motion-generation / hy-motion-1 / dart-control / awesome-text-to-motion / phc / paper-notebook-humanml3d / paper-phygile；归档 papers/molingo_arxiv_2512_13840.md、repos/molingo.md；注册 university-of-tubingen / imperial-college

- **项目页 / 论文 / 仓：** [`sources/sites/molingo-github-io.md`](sources/sites/molingo-github-io.md)（<https://hynann.github.io/molingo/MoLingo.html>）· [`sources/papers/molingo_arxiv_2512_13840.md`](sources/papers/molingo_arxiv_2512_13840.md)（arXiv:2512.13840）· [`sources/repos/molingo.md`](sources/repos/molingo.md)（<https://github.com/hynann/MoLingo>）
- **主升格：** [`wiki/entities/paper-molingo.md`](wiki/entities/paper-molingo.md) — SAE + 掩码自回归 rectified flow + T5 cross-attn；HumanML3D SOTA；G1+PHC 演示
- **交叉：** [`wiki/methods/diffusion-motion-generation.md`](wiki/methods/diffusion-motion-generation.md)、[`wiki/methods/hy-motion-1.md`](wiki/methods/hy-motion-1.md)、[`wiki/methods/dart-control.md`](wiki/methods/dart-control.md)、[`wiki/entities/awesome-text-to-motion-zilize.md`](wiki/entities/awesome-text-to-motion-zilize.md)、[`wiki/entities/phc.md`](wiki/entities/phc.md)、[`wiki/entities/paper-notebook-humanml3d.md`](wiki/entities/paper-notebook-humanml3d.md)、[`wiki/entities/paper-phygile.md`](wiki/entities/paper-phygile.md)
- **机构注册：** `schema/institutions.json` → `university-of-tubingen`、`imperial-college`
- **开源核查（2026-07-27）：** **已开源（部分）** — 训推/评测/权重已放（Apache-2.0）；README TODO：**G1 tracking pipeline** 未发布

## [2026-07-27] ingest | sources/sites/xyzcorp-deux.md — DEUX 半人形服务机器人 + Glove X/Brain X；升格 wiki/entities/xyz-deux.md；交叉 teleoperation / data-gloves / hardware-101 / sunday-act2 / handumi / imitation-learning；注册 xyz-corp

- **产品页：** [`sources/sites/xyzcorp-deux.md`](sources/sites/xyzcorp-deux.md)（<https://xyzcorp.imweb.me/DEUX>；ABOUT/TECH 交叉摘录）
- **主升格：** [`wiki/entities/xyz-deux.md`](wiki/entities/xyz-deux.md) — 三指手 + Glove X 1:1 零样本重定向；Brain X IL/RL；预购规格
- **交叉：** [`wiki/tasks/teleoperation.md`](wiki/tasks/teleoperation.md)、[`wiki/comparisons/data-gloves-vs-vision-teleop.md`](wiki/comparisons/data-gloves-vs-vision-teleop.md)、[`wiki/overview/humanoid-hardware-101-sensing-end-effectors.md`](wiki/overview/humanoid-hardware-101-sensing-end-effectors.md)、[`wiki/entities/sunday-robotics-act2.md`](wiki/entities/sunday-robotics-act2.md)、[`wiki/entities/handumi.md`](wiki/entities/handumi.md)、[`wiki/methods/imitation-learning.md`](wiki/methods/imitation-learning.md)
- **机构注册：** `schema/institutions.json` → `xyz-corp`（艾克斯怀吉（XYZ））
- **开源核查（2026-07-27）：** **未开源**（无 GitHub/HF/数据集；ROS 2 仅产品叙事）

## [2026-07-27] ingest | sources/repos/handumi-sw.md — 刷新 HandUMI：硬件迁 robonet-ai/handumi-hw、统一 CLI、QA/convert 管线；wiki/entities/handumi.md；交叉 teleoperation / bimanual-manipulation / lerobot；归档 handumi-hw / handumi-quest-app / sites/handumi-sw

- **软件 / 硬件 / Quest / 文档：** [`sources/repos/handumi-sw.md`](sources/repos/handumi-sw.md)（<https://github.com/robonet-ai/handumi-sw>）· [`sources/repos/handumi-hw.md`](sources/repos/handumi-hw.md)（<https://github.com/robonet-ai/handumi-hw>；旧 BrikHMP18 仓 301）· [`sources/repos/handumi-quest-app.md`](sources/repos/handumi-quest-app.md) · [`sources/sites/handumi-sw.md`](sources/sites/handumi-sw.md)（<https://robonet-ai.github.io/handumi-sw/>）
- **主升格 / 刷新：** [`wiki/entities/handumi.md`](wiki/entities/handumi.md) — 一次采集、多臂重定向；模块化 tip；Feetech 直测开合；`handumi`/`hu` CLI；LeRobot v3 + validate/convert
- **交叉：** [`wiki/tasks/teleoperation.md`](wiki/tasks/teleoperation.md)、[`wiki/tasks/bimanual-manipulation.md`](wiki/tasks/bimanual-manipulation.md)、[`wiki/entities/lerobot.md`](wiki/entities/lerobot.md)
- **开源核查（2026-07-27）：** handumi-sw / handumi-hw **已开源**（Apache-2.0）；Quest 应用公开仓 + APK（仓页未标 SPDX）

## [2026-07-27] ingest | sources/papers/taco_tactile_sensor_benchmark_arxiv_2605_21976.md — TacO 跨模态触觉传感器真机 IL 基准；升格 wiki/entities/paper-taco-tactile-sensor-benchmark.md；交叉更新 wiki/concepts/tactile-sensing.md、wiki/concepts/visuo-tactile-fusion.md、wiki/concepts/contact-rich-manipulation.md、wiki/overview/topic-tactile.md、wiki/methods/imitation-learning.md、wiki/methods/action-chunking.md、wiki/entities/paper-taco-tactile-wm-vla-posttrain.md、wiki/entities/paper-vtap-gripper.md；归档 sources/sites/tacobench-github-io.md、sources/repos/taco-bench.md

- **论文 / 项目页 / 仓：** [`sources/papers/taco_tactile_sensor_benchmark_arxiv_2605_21976.md`](sources/papers/taco_tactile_sensor_benchmark_arxiv_2605_21976.md)（arXiv:2605.21976）· [`sources/sites/tacobench-github-io.md`](sources/sites/tacobench-github-io.md)（<https://tacobench.github.io/>）· [`sources/repos/taco-bench.md`](sources/repos/taco-bench.md)（<https://github.com/TacObench/TacO>）· 硬件 STL（<https://github.com/TacObench/TacObench.github.io/tree/main/3D_part_files>）
- **主升格：** [`wiki/entities/paper-taco-tactile-sensor-benchmark.md`](wiki/entities/paper-taco-tactile-sensor-benchmark.md) — 六传感器 × 四模态 × 三真机任务统一 ACT；无通用最佳触觉传感器；名称消歧 vs TACO-WM
- **交叉：** [`wiki/concepts/tactile-sensing.md`](wiki/concepts/tactile-sensing.md)、[`wiki/concepts/visuo-tactile-fusion.md`](wiki/concepts/visuo-tactile-fusion.md)、[`wiki/concepts/contact-rich-manipulation.md`](wiki/concepts/contact-rich-manipulation.md)、[`wiki/overview/topic-tactile.md`](wiki/overview/topic-tactile.md)、[`wiki/methods/imitation-learning.md`](wiki/methods/imitation-learning.md)、[`wiki/methods/action-chunking.md`](wiki/methods/action-chunking.md)、[`wiki/entities/paper-taco-tactile-wm-vla-posttrain.md`](wiki/entities/paper-taco-tactile-wm-vla-posttrain.md)、[`wiki/entities/paper-vtap-gripper.md`](wiki/entities/paper-vtap-gripper.md)
- **开源核查：** **部分开源**（代码 + 硬件 STL + 可重复性测试已开；示范数据/checkpoint 截至入库日未见公开下载链）

## [2026-07-27] ingest | sources/papers/topreward_arxiv_2602_19313.md — TOPReward 零样本 VLM token 进度奖励；升格 wiki/entities/paper-topreward.md；交叉更新 wiki/concepts/progress-reward-modeling.md、wiki/entities/paper-progress-reward-modeling-survey.md、wiki/methods/awr.md、wiki/methods/imitation-learning.md、wiki/concepts/open-x-embodiment.md；归档 sources/sites/topreward-github-io.md、sources/repos/topreward.md

- **论文 / 项目页 / 仓：** [`sources/papers/topreward_arxiv_2602_19313.md`](sources/papers/topreward_arxiv_2602_19313.md)（arXiv:2602.19313）· [`sources/sites/topreward-github-io.md`](sources/sites/topreward-github-io.md)（<https://topreward.github.io/webpage/>）· [`sources/repos/topreward.md`](sources/repos/topreward.md)
- **主升格：** [`wiki/entities/paper-topreward.md`](wiki/entities/paper-topreward.md) — 视频 VLM 完成 token log-likelihood → 零样本稠密进度；OXE VOC 0.857 / ManiRewardBench ≈0.95；TOP-AWR
- **交叉：** [`wiki/concepts/progress-reward-modeling.md`](wiki/concepts/progress-reward-modeling.md)、[`wiki/entities/paper-progress-reward-modeling-survey.md`](wiki/entities/paper-progress-reward-modeling-survey.md)、[`wiki/methods/awr.md`](wiki/methods/awr.md)、[`wiki/methods/imitation-learning.md`](wiki/methods/imitation-learning.md)、[`wiki/concepts/open-x-embodiment.md`](wiki/concepts/open-x-embodiment.md)
- **开源核查：** **已开源**（MIT 推理评测代码 + HF ManiRewardBench 子集；无专用 reward 训练权重）

## [2026-07-27] ingest | sources/blogs/robodojo_open_longterm_eval_2026-07.md — RoboDojo 开放长期公益评测；wiki/entities/robodojo.md · wiki/entities/xpolicylab.md

- **公告 / 站点：** [`sources/blogs/robodojo_open_longterm_eval_2026-07.md`](sources/blogs/robodojo_open_longterm_eval_2026-07.md) · [`sources/sites/robodojo-benchmark.md`](sources/sites/robodojo-benchmark.md)（<https://robodojo-benchmark.com/>；Eval / Leaderboard / Protocol 2026-07-27 核查）
- **论文 / 仓：** [`sources/papers/robodojo_arxiv_2607_04434.md`](sources/papers/robodojo_arxiv_2607_04434.md)（arXiv:2607.04434）· [`sources/repos/robodojo.md`](sources/repos/robodojo.md) · [`sources/repos/xpolicylab.md`](sources/repos/xpolicylab.md)
- **主升格：** [`wiki/entities/robodojo.md`](wiki/entities/robodojo.md) — 42 sim 五维 + 18 real；公益 verified 上榜（XPolicyLab 开源训推+权重+评测视频）；[`wiki/entities/xpolicylab.md`](wiki/entities/xpolicylab.md) — 40+ 策略适配
- **交叉：** [`wiki/queries/embodied-eval-benchmark-selection-loop.md`](wiki/queries/embodied-eval-benchmark-selection-loop.md)、[`wiki/overview/topic-embodied-eval-benchmark.md`](wiki/overview/topic-embodied-eval-benchmark.md)、[`wiki/methods/vla.md`](wiki/methods/vla.md)、[`wiki/concepts/simulation-evaluation-infrastructure.md`](wiki/concepts/simulation-evaluation-infrastructure.md)、[`wiki/entities/xiaomi-robotics-1.md`](wiki/entities/xiaomi-robotics-1.md)、[`wiki/entities/robo-bench.md`](wiki/entities/robo-bench.md)、[`wiki/entities/vla-sota-leaderboard.md`](wiki/entities/vla-sota-leaderboard.md)
- **机构注册：** `schema/institutions.json` → `ai-mmlab-club`（人工智能多媒体实验室俱乐部（AI MMLab Club））
- **开源核查：** RoboDojo **已开源**（eval-only，MIT LICENSE）；XPolicyLab **已开源**（Apache-2.0）；verified 上榜强制开源训推与 checkpoint

## [2026-07-27] ingest | sources/repos/roboflow_sports.md — Roboflow Sports 体育 CV 工具与足球俯视雷达；wiki/entities/roboflow-sports.md

- **来源归档：** [`sources/repos/roboflow_sports.md`](sources/repos/roboflow_sports.md)（<https://github.com/roboflow/sports>；~5.2k★；MIT；2026-07-27 核查）
- **主升格：** [`wiki/entities/roboflow-sports.md`](wiki/entities/roboflow-sports.md) — 球场关键点单应、球跟踪、SigLIP 分队、RADAR 俯视；与机载场线定位对照
- **交叉：** [`wiki/methods/soccer-field-line-detection.md`](wiki/methods/soccer-field-line-detection.md)、[`wiki/queries/soccer-visual-field-localization-pipeline.md`](wiki/queries/soccer-visual-field-localization-pipeline.md)、[`wiki/tasks/humanoid-soccer.md`](wiki/tasks/humanoid-soccer.md)、[`wiki/entities/ultralytics.md`](wiki/entities/ultralytics.md)、[`wiki/entities/rf-detr.md`](wiki/entities/rf-detr.md)
- **机构注册：** `schema/institutions.json` → `roboflow`（罗博福流（Roboflow））
- **开源核查：** **已开源**（库 MIT + soccer 示例；权重经 `setup.sh`/Drive；demo 绑 Ultralytics AGPL）

## [2026-07-27] structural | media/site-demo.gif — 按去标题后的首页重录 README 演示 GIF

- **脚本：** [`scripts/record_readme_demo.cjs`](scripts/record_readme_demo.cjs)（70 frames / 3.12 MB；图谱 1933 节点）
- **关联：** 承接首页入口区标题/副标题删除（[`docs/index.html`](docs/index.html)）

## [2026-07-27] ingest | sources/papers/ace_brain_0_5_arxiv_2607_04426.md — ACE-Brain-0.5 统一具身基础模型；wiki/entities/paper-ace-brain-0-5.md

- **论文：** [`sources/papers/ace_brain_0_5_arxiv_2607_04426.md`](sources/papers/ace_brain_0_5_arxiv_2607_04426.md)（arXiv:2607.04426；ACE-Brain Team / 大晓 Ace Robotics）
- **项目页 / 仓：** [`sources/sites/ace-brain-0-5-github-io.md`](sources/sites/ace-brain-0-5-github-io.md) · [`sources/repos/ace-brain-0-5.md`](sources/repos/ace-brain-0-5.md) — **部分开源**（HF 8B 权重 + transformers 推理；GitHub 仅 README/资产）
- **主升格：** [`wiki/entities/paper-ace-brain-0-5.md`](wiki/entities/paper-ace-brain-0-5.md) — 五功能闭环 + SSR+；LIBERO **98.2%** / Bridge VLA **82.3%** / RBM VOC 强
- **交叉：** [`wiki/methods/vla.md`](wiki/methods/vla.md)、[`foundation-policy`](wiki/concepts/foundation-policy.md)、[`progress-reward-modeling`](wiki/concepts/progress-reward-modeling.md)、[`paper-rynnbrain-1-1`](wiki/entities/paper-rynnbrain-1-1.md)、[`qwen-vla`](wiki/entities/qwen-vla.md)、[`vision-language-navigation`](wiki/tasks/vision-language-navigation.md)
- **机构：** 复用 `ace-robotics`（大晓机器人）
- **开源核查日：** 2026-07-27

## [2026-07-27] structural | docs/index.html — 删除首页目标入口区标题与副标题（与 Hero 重复）

- **改动：** 移除「选择你的入口」与「按当前目标进入最短路径…」；section 改用 `aria-label="入口"`
- **同步：** [`docs/style.css`](docs/style.css) 清理 `#home-start .section-subtitle`；[`docs/checklists/frontend-optimization-v1.md`](docs/checklists/frontend-optimization-v1.md) Phase 4 勾选

## [2026-07-27] ingest | sources/sites/arxiv-org.md — arXiv 开放获取预印本平台宏观节点；wiki/entities/arxiv.md

- **来源归档：** [`sources/sites/arxiv-org.md`](sources/sites/arxiv-org.md)（<https://arxiv.org/>；About / API 2026-07-27 核查）
- **主升格：** [`wiki/entities/arxiv.md`](wiki/entities/arxiv.md) — 预印本分发 vs 同行评审录用；cs.RO / API / 标识符；与本库 `sources/papers/*_arxiv_*` 默认外链层对齐
- **交叉：** [`wiki/comparisons/robotics-research-venues.md`](wiki/comparisons/robotics-research-venues.md)、[`wiki/overview/robot-learning-overview.md`](wiki/overview/robot-learning-overview.md)、[`sources/sites/robotics-venues-primary-refs.md`](sources/sites/robotics-venues-primary-refs.md)
- **机构注册：** `schema/institutions.json` → `arxiv`（学术预印本档案（arXiv））
- **开源核查：** 平台为开放获取 + 公共 API；单篇论文代码仍以各项目页为准

## [2026-07-27] ingest | sources/papers/egohtr_arxiv_2607_13472.md — EgoHTR 加深评测数字与开源再核查；wiki/entities/paper-egohtr.md + VisualMimic/MeshMimic 回链

- **主实体加深：** [`wiki/entities/paper-egohtr.md`](wiki/entities/paper-egohtr.md) — 补全局 HPS（W-MPJPE 151.3 / WA 66.7 / RTE 0.09%）、踏石接触奖励消融、HMR Table 4
- **sources 刷新：** [`sources/papers/egohtr_arxiv_2607_13472.md`](sources/papers/egohtr_arxiv_2607_13472.md)、[`sources/sites/egohtr-github-io.md`](sources/sites/egohtr-github-io.md)
- **交叉回链：** [`wiki/entities/paper-notebook-visualmimic.md`](wiki/entities/paper-notebook-visualmimic.md)、[`wiki/entities/paper-notebook-meshmimic.md`](wiki/entities/paper-notebook-meshmimic.md)
- **开源核查日：** 2026-07-27 — Dataset/Code 仍 *coming soon*（首次入库 2026-07-21）

## [2026-07-27] ingest | sources/papers/progress_reward_modeling_survey_arxiv_2607_21655.md — 过程奖励综述 + Awesome 索引；wiki/concepts/progress-reward-modeling.md · wiki/entities/paper-progress-reward-modeling-survey.md

- **论文：** [`sources/papers/progress_reward_modeling_survey_arxiv_2607_21655.md`](sources/papers/progress_reward_modeling_survey_arxiv_2607_21655.md)（arXiv:2607.21655；Northwestern / CMU / UW–Madison / UCSB / UIUC）
- **索引仓：** [`sources/repos/awesome-progress-models.md`](sources/repos/awesome-progress-models.md) — **MIT 已开源**（策展画廊，非算法实现）
- **主升格：** [`wiki/concepts/progress-reward-modeling.md`](wiki/concepts/progress-reward-modeling.md) · [`wiki/entities/paper-progress-reward-modeling-survey.md`](wiki/entities/paper-progress-reward-modeling-survey.md) — 接口三维 × 四范式 × 保真/鲁棒/效用透镜
- **交叉：** [`wiki/methods/reinforcement-learning.md`](wiki/methods/reinforcement-learning.md)
- **机构：** 注册 `uw-madison`、`ucsb`
- **开源核查日：** 2026-07-27

## [2026-07-27] ingest | sources/papers/vitacworld_arxiv_2607_22530.md — ViTacWorld 视触觉世界模型；wiki/entities/paper-vitacworld.md + 交叉 visuo-tactile / VT-WAM

- **论文：** [`sources/papers/vitacworld_arxiv_2607_22530.md`](sources/papers/vitacworld_arxiv_2607_22530.md)（arXiv:2607.22530；上海科技大学 / InstAdapt）
- **项目页：** [`sources/sites/vitacworld-github-io.md`](sources/sites/vitacworld-github-io.md) — **宣称将开源**（Code *coming soon*）
- **主升格：** [`wiki/entities/paper-vitacworld.md`](wiki/entities/paper-vitacworld.md) — view-aware DiT；π₀.₅+触觉四任务平均 **42.5%→67.5%**（Round-2 **80%**）
- **交叉：** [`wiki/concepts/visuo-tactile-fusion.md`](wiki/concepts/visuo-tactile-fusion.md)、[`paper-vt-wam`](wiki/entities/paper-vt-wam-visuotactile-contact-rich.md)、[`generative-world-models`](wiki/methods/generative-world-models.md)、[`paper-ctrl-world`](wiki/entities/paper-ctrl-world.md)
- **机构：** 注册 `instadapt`
- **开源核查日：** 2026-07-27

## [2026-07-27] ingest | sources/papers/rofacto_arxiv_2607_22535.md — Rofacto URDF 渲染动作世界模型；wiki/entities/paper-rofacto.md + 交叉 DWM / Ctrl-World

- **论文：** [`sources/papers/rofacto_arxiv_2607_22535.md`](sources/papers/rofacto_arxiv_2607_22535.md)（arXiv:2607.22535；SNU / RLWRLD）
- **项目页 / 仓：** [`sources/sites/rofacto-github-io.md`](sources/sites/rofacto-github-io.md) · [`sources/repos/rofacto.md`](sources/repos/rofacto.md) — **宣称将开源**（Code 链到 GitHub，仓 **404**）
- **主升格：** [`wiki/entities/paper-rofacto.md`](wiki/entities/paper-rofacto.md) — 名义轨迹 + URDF mesh/深度；Wan 上 DROID PSNR **18.57→21.87**
- **交叉：** [`wiki/methods/generative-world-models.md`](wiki/methods/generative-world-models.md)、[`wiki/methods/dwm.md`](wiki/methods/dwm.md)、[`paper-ctrl-world`](wiki/entities/paper-ctrl-world.md)
- **机构：** 注册 `rlwrld`
- **开源核查日：** 2026-07-27

## [2026-07-27] ingest | sources/papers/fm_vla_arxiv_2607_18231.md — FM-VLA 力觉记忆 VLA；wiki/entities/paper-fm-vla.md + 交叉 VLA/manipulation/CRM/KEMO/EventVLA

- **论文：** [`sources/papers/fm_vla_arxiv_2607_18231.md`](sources/papers/fm_vla_arxiv_2607_18231.md)（arXiv:2607.18231；清华 / 微软研究院 / 复旦 / 中科大）
- **项目页 / 仓：** [`sources/sites/fm-vla-page.md`](sources/sites/fm-vla-page.md) · [`sources/repos/fm-vla.md`](sources/repos/fm-vla.md) — **宣称将开源**（README「Code will be released soon」，仅 demo 媒体）
- **主升格：** [`wiki/entities/paper-fm-vla.md`](wiki/entities/paper-fm-vla.md) — Force-VAE（$K=8$）+ 短状态窗注入 π₀.₅；G1 三项任务平均 **83.3%**、+**3.3 ms**
- **交叉：** [`wiki/methods/vla.md`](wiki/methods/vla.md)、[`wiki/tasks/manipulation.md`](wiki/tasks/manipulation.md)、[`wiki/concepts/contact-rich-manipulation.md`](wiki/concepts/contact-rich-manipulation.md)、[`paper-kemo`](wiki/entities/paper-kemo-event-driven-keyframe-memory-vla.md)、[`paper-eventvla`](wiki/entities/paper-eventvla-visual-evidence-memory.md)
- **开源核查日：** 2026-07-27

## [2026-07-27] ingest | sources/blogs/wechat_embodied_ai_lab_world_model_physics_fidelity.md — 世界模型「学到多少真实物理」：按预测输出阅读；12 篇新建 complete 实体 + 复用已有节点

- **来源归档：** [`sources/blogs/wechat_embodied_ai_lab_world_model_physics_fidelity.md`](sources/blogs/wechat_embodied_ai_lab_world_model_physics_fidelity.md)（Agent Reach v1.5.0 + wechat-article-for-ai；<https://mp.weixin.qq.com/s/OawDKruG8zEepiy-x1nKuA>）
- **原始抓取：** [`sources/raw/wechat_world_model_physics_fidelity_2026-07-27/`](sources/raw/wechat_world_model_physics_fidelity_2026-07-27/)
- **主升格：** [`wiki/overview/world-model-physics-fidelity-outputs.md`](wiki/overview/world-model-physics-fidelity-outputs.md) — 潜变量 / 视频 / 持续状态 / 动作–世界分解 / 几何·触觉·物理混合 / 评测诊断 + 四类测试优先序
- **新建论文实体（complete，非 stub，0 重复 arXiv）：** World Models、PlaNet、TD-MPC2、UniSim、IRASim、V-JEPA 2、WorldWeaver、DWM-Separating（≠ Dexterous DWM）、PhysCoRe、Imagined Rollouts、KineBench、Thinking in Video
- **复用已有 complete：** DreamerV3（加厚）、Masked Visual Actions、RynnWorld-4D、MECo-WAM、VT-WAM；文首对照 Ego-VCP / MotionWAM
- **命名消歧：** [`wiki/methods/dwm.md`](wiki/methods/dwm.md) ↔ [`paper-dwm-separating-world-effects`](wiki/entities/paper-dwm-separating-world-effects.md)
- **机构：** 注册 `adobe`（奥多比研究院）
- **分批明细：** 见下三条同日 log（经典四篇+DreamerV3 / 视频四篇 / 评测四篇）

## [2026-07-27] ingest | PhysCoRe / Imagined-Rollouts / KineBench / Thinking-in-Video — 物理保真度评测与混合物理四篇 complete 实体

- **论文归档：**
  - [`sources/papers/physcore_arxiv_2607_20653.md`](sources/papers/physcore_arxiv_2607_20653.md)（**未开源**，2026-07-27）
  - [`sources/papers/imagined_rollouts_kinematic_not_dynamic_arxiv_2607_05966.md`](sources/papers/imagined_rollouts_kinematic_not_dynamic_arxiv_2607_05966.md)（diagnostic **未开源**）
  - [`sources/papers/kinebench_arxiv_2607_19876.md`](sources/papers/kinebench_arxiv_2607_19876.md) + [`sources/repos/kinebench.md`](sources/repos/kinebench.md)（**MIT 已开源**）
  - [`sources/papers/thinking_in_video_arxiv_2607_17523.md`](sources/papers/thinking_in_video_arxiv_2607_17523.md) + [`sources/repos/thinking-in-video.md`](sources/repos/thinking-in-video.md)（代码+HF 数据 **已开源**）
- **wiki 实体（complete）：**
  - [`wiki/entities/paper-physcore.md`](wiki/entities/paper-physcore.md)
  - [`wiki/entities/paper-imagined-rollouts-kinematic-not-dynamic.md`](wiki/entities/paper-imagined-rollouts-kinematic-not-dynamic.md)
  - [`wiki/entities/paper-kinebench.md`](wiki/entities/paper-kinebench.md)
  - [`wiki/entities/paper-thinking-in-video.md`](wiki/entities/paper-thinking-in-video.md)
- **交叉：** overview [`wiki/overview/world-model-physics-fidelity-outputs.md`](wiki/overview/world-model-physics-fidelity-outputs.md)；轻量更新 Masked Visual Actions / RynnWorld-4D / MECo-WAM / VT-WAM（related + blog 参考来源）
- **开源核查日：** 2026-07-27

## [2026-07-27] ingest | sources/papers/{irasim,vjepa2,worldweaver,dwm_separating_world_effects} — 物理保真轴四篇（IRASim / V-JEPA 2 / WorldWeaver / DWM-Separating）

- **策展语境：** [`sources/blogs/wechat_embodied_ai_lab_world_model_physics_fidelity.md`](sources/blogs/wechat_embodied_ai_lab_world_model_physics_fidelity.md)
- **overview：** [`wiki/overview/world-model-physics-fidelity-outputs.md`](wiki/overview/world-model-physics-fidelity-outputs.md)（与经典 latent/视频节点交叉）
- **新建论文实体（complete）：**
  - [`wiki/entities/paper-irasim.md`](wiki/entities/paper-irasim.md) — arXiv:2406.14540；**已开源** Apache-2.0 + 数据/checkpoints
  - [`wiki/entities/paper-vjepa2.md`](wiki/entities/paper-vjepa2.md) — arXiv:2506.09985；**已开源** MIT + V-JEPA 2-AC；latent 规划
  - [`wiki/entities/paper-worldweaver.md`](wiki/entities/paper-worldweaver.md) — arXiv:2607.21594；**宣称将开源**（README coming soon）
  - [`wiki/entities/paper-dwm-separating-world-effects.md`](wiki/entities/paper-dwm-separating-world-effects.md) — arXiv:2607.18715；CEM **+13.1pp**；**未开源**；≠ Dexterous DWM
- **sources：** papers×4；repos：irasim / vjepa2 / worldweaver；sites：gen-irasim / meta-vjepa2-blog / worldweaver-vail-ucla
- **消歧：** [`wiki/methods/dwm.md`](wiki/methods/dwm.md) 顶部注明与 2607.18715 同名不同文
- **机构：** `schema/institutions.json` 注册 `adobe`

## [2026-07-27] ingest | sources/papers/{ha_schmidhuber_world_models,planet_latent_dynamics,tdmpc2,unisim} — 物理保真度博客经典四篇升格完整论文实体；加厚 DreamerV3

- **论文归档：**
  - [`sources/papers/ha_schmidhuber_world_models_arxiv_1803_10122.md`](sources/papers/ha_schmidhuber_world_models_arxiv_1803_10122.md)
  - [`sources/papers/planet_latent_dynamics_arxiv_1811_04551.md`](sources/papers/planet_latent_dynamics_arxiv_1811_04551.md)
  - [`sources/papers/tdmpc2_arxiv_2310_16828.md`](sources/papers/tdmpc2_arxiv_2310_16828.md)
  - [`sources/papers/unisim_arxiv_2310_06114.md`](sources/papers/unisim_arxiv_2310_06114.md)
  - 加厚 [`sources/papers/shenlan_wm_survey_13_dreamerv3.md`](sources/papers/shenlan_wm_survey_13_dreamerv3.md)（交叉物理保真度博客 + 开源仓）
- **repos / sites：**
  - [`sources/sites/worldmodels-github-io.md`](sources/sites/worldmodels-github-io.md) · [`sources/repos/world-models-experiments.md`](sources/repos/world-models-experiments.md)
  - [`sources/sites/planetrl-github-io.md`](sources/sites/planetrl-github-io.md) · [`sources/repos/google-research-planet.md`](sources/repos/google-research-planet.md)
  - [`sources/sites/tdmpc2-com.md`](sources/sites/tdmpc2-com.md) · [`sources/repos/tdmpc2.md`](sources/repos/tdmpc2.md)
  - [`sources/sites/universal-simulator-github-io.md`](sources/sites/universal-simulator-github-io.md)（UniSim **未开源**）
  - [`sources/repos/danijar-dreamerv3.md`](sources/repos/danijar-dreamerv3.md)
- **wiki 实体（complete，非 stub）：**
  - [`wiki/entities/paper-ha-schmidhuber-world-models.md`](wiki/entities/paper-ha-schmidhuber-world-models.md)
  - [`wiki/entities/paper-planet-latent-dynamics.md`](wiki/entities/paper-planet-latent-dynamics.md)
  - [`wiki/entities/paper-td-mpc2.md`](wiki/entities/paper-td-mpc2.md)
  - [`wiki/entities/paper-unisim.md`](wiki/entities/paper-unisim.md)
  - 加厚 [`wiki/entities/paper-shenlan-wm-13-dreamerv3.md`](wiki/entities/paper-shenlan-wm-13-dreamerv3.md)（缩写≥3、结论、源码时序、仍链 15 地图）
- **交叉：** [`wiki/overview/world-model-physics-fidelity-outputs.md`](wiki/overview/world-model-physics-fidelity-outputs.md)；博客 [`sources/blogs/wechat_embodied_ai_lab_world_model_physics_fidelity.md`](sources/blogs/wechat_embodied_ai_lab_world_model_physics_fidelity.md)
- **开源口径：** World Models 交互站+实验仓已开源；PlaNet archived Apache；TD-MPC2 MIT+权重；UniSim 仅项目页；DreamerV3 → danijar/dreamerv3 + Open Dreamer 后继

## [2026-07-26] structural | docs/checklists — V30 收尾（详情页感知栈徽标）并新建 v31「机器人视觉感知栈选型闭环」执行清单

- **V30 P3② 收尾：** 详情页「所属专题」徽标行本就数据驱动（`main.js renderMetaTopicBadges` → `topic-filters.js topicsForNode`），P3① 写入单一事实源后自动联动
  - node 逐页复核发现驱动链 ③层 [`wiki/entities/paper-neuralactuator-neural-actuation-modeling.md`](wiki/entities/paper-neuralactuator-neural-actuation-modeling.md) 漏命中（id 分词为 `neuralactuator`/`actuation`，无干净 `actuator` 片段），遂显式补入 [`docs/topic-filters.js`](docs/topic-filters.js) `actuator-drive-chain.ids`，使该真·驱动链页详情页徽标与图谱专题视图同步点亮
  - 补后 node 复核 simplefoc/kicad/altium/neuralactuator/bam/ethercat-protocol/field-oriented-control/implicit-explicit/query/torque-source-abstraction-gap/枢纽页 11 页均命中，非驱动链的 vla/topic-grasp/robo-bench 均不命中；`make lint` 0 errors；至此 **v30 全数完成**
- **新建执行清单：** [`docs/checklists/tech-stack-next-phase-checklist-v31.md`](docs/checklists/tech-stack-next-phase-checklist-v31.md) — 聚焦「机器人视觉感知栈选型闭环」（传感与标定 → 2D 检测/分割选型 → 2D→3D 语义建图 → 下游策略消费 四层选型链），承接近周 YOLO/RF-DETR/SAM/SAM2/FindAnything/语义建图密集 ingest
- **看板维护：** v30 移入 [`docs/checklists/archive/`](docs/checklists/archive/)；[`docs/checklists/README.md`](docs/checklists/README.md) 当前入口与历史列表同步更新

## [2026-07-26] structural | roadmap/depth-motion-retargeting.md — 动作重定向纵深补四足支线（动物/视频关键点 → SMR/TMR → legged_gym 跟踪）

- **改写：** [`roadmap/depth-motion-retargeting.md`](roadmap/depth-motion-retargeting.md)
  - 路线图 mermaid：Stage 2 分出「Stage 3 支线（四足重定向）」并汇回 Stage 4
  - Stage 0/2/3/4 各补四足视角：目标平台差异、动物数据源更脏、腿部 DoF 与缺基座轨迹、legged_gym/AMP 下游生态
  - Stage 3 新增「四足支线」小节：人形/四足差异对照表 + motion_imitation → STMR（SMR/TMR）→ PAN / ReActor 三级台阶 + 选型经验
  - Stage 5 方向 A 明确「一份参考 → 多机型」与支线的分工；快速入口表补支线行
- **交叉：** [`wiki/entities/stmr-quadruped-retargeting.md`](wiki/entities/stmr-quadruped-retargeting.md)、[`wiki/entities/motion-imitation-quadruped.md`](wiki/entities/motion-imitation-quadruped.md)、[`wiki/entities/go2-motion-imitation.md`](wiki/entities/go2-motion-imitation.md)、[`wiki/entities/pan-motion-retargeting.md`](wiki/entities/pan-motion-retargeting.md)、[`wiki/entities/quadruped-robot.md`](wiki/entities/quadruped-robot.md)、[`wiki/entities/legged-gym.md`](wiki/entities/legged-gym.md)、[`wiki/methods/amp-reward.md`](wiki/methods/amp-reward.md)
- **入口同步：** [`roadmap/README.md`](roadmap/README.md) 与 [`README.md`](README.md) 该路线一行描述改为「人体/动物动作 → 人形或四足参考轨迹」

## [2026-07-26] ingest | sources/sites/waytoagi-feishu-wiki.md — WaytoAGI 飞书知识库首页入库；升格 wiki/entities/waytoagi.md；交叉 unitree / VLA 复现地图 / OpenLET

- **确认工具：** Agent Reach v1.5.0 已安装（`~/.local/bin/agent-reach`）；抓取用 Camoufox + 飞书公开 `wiki/v2/tree/*`（Jina 仅得壳层）
- **来源归档：**
  - [`sources/sites/waytoagi-feishu-wiki.md`](sources/sites/waytoagi-feishu-wiki.md)
  - [`sources/raw/feishu_waytoagi_wiki_home_2026-07-26.md`](sources/raw/feishu_waytoagi_wiki_home_2026-07-26.md) — 一级 TOC + AI硬件子树
- **主升格：** [`wiki/entities/waytoagi.md`](wiki/entities/waytoagi.md) — 社区门户实体；机器人主线读「AI硬件」雷达
- **开源核查（步骤 2.5）：** 飞书库 **Public access**；无独立算法仓；宇树 PDF 为**社区转载**
- **交叉：**
  - [`wiki/entities/unitree.md`](wiki/entities/unitree.md)
  - [`wiki/overview/vla-open-source-repro-landscape-2025.md`](wiki/overview/vla-open-source-repro-landscape-2025.md)
  - [`wiki/entities/openlet.md`](wiki/entities/openlet.md)

## [2026-07-26] ingest | sources/repos/ultralytics.md — Ultralytics YOLO 官方仓入库；升格 wiki/entities/ultralytics.md；交叉 wiki/methods/object-detection.md、wiki/queries/object-detection-model-selection.md、wiki/entities/rf-detr.md、wiki/entities/paper-yolo-unified-realtime-detection.md、wiki/entities/booster-robocup-demo.md、wiki/methods/soccer-field-line-detection.md

## [2026-07-26] ingest | sources/papers/segment_anything_arxiv_2304_02643.md + sources/papers/sam2_arxiv_2408_00714.md — SAM/SAM2 论文与官方仓入库；升格 wiki/entities/paper-segment-anything.md、wiki/entities/paper-sam2.md；交叉 wiki/queries/go2-3d-semantic-mapping-sam-pipeline.md、wiki/entities/ovo-semantic-mapping.md、wiki/entities/dualmap.md、wiki/entities/ov-sam3d.md

## [2026-07-26] structural | wiki/queries/go2-3d-semantic-mapping-sam-pipeline.md — 面向全体读者改写：先交代问题背景与核心疑问，去掉「可直接回复对方」等对单人话术标题

- **改写：** [`wiki/queries/go2-3d-semantic-mapping-sam-pipeline.md`](wiki/queries/go2-3d-semantic-mapping-sam-pipeline.md) — 新增「问题背景与核心疑问」+ 结论先行；§3 去第二人称；删除私聊式短答节

## [2026-07-26] structural | wiki/entities/{findanything,cmu-mscv-semantic-3d-mapping}.md — 为项目页建 site/项目实体；开源仓地址与源码分析待官方发布后补

- **新建：**
  - [`wiki/entities/findanything.md`](wiki/entities/findanything.md) ← [`sources/sites/findanything.md`](sources/sites/findanything.md)（宣称并入 OKVIS2-X）
  - [`wiki/entities/cmu-mscv-semantic-3d-mapping.md`](wiki/entities/cmu-mscv-semantic-3d-mapping.md) ← [`sources/sites/cmu-mscv-semantic-3d-mapping.md`](sources/sites/cmu-mscv-semantic-3d-mapping.md)
- **跟进约定：** 官方放出独立仓后 → 新建 `sources/repos/`、更新实体「代码仓」行、补运行/README 分析
- **交叉：** [`wiki/queries/go2-3d-semantic-mapping-sam-pipeline.md`](wiki/queries/go2-3d-semantic-mapping-sam-pipeline.md)、[`wiki/entities/autonomy-stack-go2.md`](wiki/entities/autonomy-stack-go2.md)、[`wiki/entities/dualmap.md`](wiki/entities/dualmap.md)

## [2026-07-26] structural | wiki/entities/{autonomy-stack-go2,dualmap,ovo-semantic-mapping,ov-sam3d}.md — 为本 PR 已开源仓补齐独立实体节点；回链 wiki/queries/go2-3d-semantic-mapping-sam-pipeline.md

- **新建实体：**
  - [`wiki/entities/autonomy-stack-go2.md`](wiki/entities/autonomy-stack-go2.md) ← [`sources/repos/autonomy_stack_go2.md`](sources/repos/autonomy_stack_go2.md)
  - [`wiki/entities/dualmap.md`](wiki/entities/dualmap.md) ← [`sources/repos/dualmap.md`](sources/repos/dualmap.md)
  - [`wiki/entities/ovo-semantic-mapping.md`](wiki/entities/ovo-semantic-mapping.md) ← [`sources/repos/ovo-semantic-mapping.md`](sources/repos/ovo-semantic-mapping.md)
  - [`wiki/entities/ov-sam3d.md`](wiki/entities/ov-sam3d.md) ← [`sources/repos/ov-sam3d.md`](sources/repos/ov-sam3d.md)
- **开源边界：** FindAnything / CMU MSCV Semantic 3D Mapping 分别为「宣称并入 OKVIS2-X」「项目页文档」，**暂不建**开源仓实体
- **交叉：** [`wiki/queries/go2-3d-semantic-mapping-sam-pipeline.md`](wiki/queries/go2-3d-semantic-mapping-sam-pipeline.md)、[`wiki/entities/point-lio-unilidar.md`](wiki/entities/point-lio-unilidar.md)、[`wiki/overview/navigation-slam-autonomy-stack.md`](wiki/overview/navigation-slam-autonomy-stack.md)

## [2026-07-26] query | wiki/queries/go2-3d-semantic-mapping-sam-pipeline.md — GO2 3D 语义建图与 SAM 2D→3D：几何先于语义；对照 point_lio_unilidar / autonomy_stack_go2 / DualMap / OVO / OV-SAM3D / FindAnything；交叉 wiki/entities/point-lio-unilidar.md、wiki/overview/navigation-slam-autonomy-stack.md、wiki/entities/paper-notebook-real-time-polygonal-semantic-mapping-for-humanoi.md

- **Q：** GO2 有无 3D 语义建图资料？移动时点云差；SAM 如何从 2D 到 3D（CMU 相关）？
- **A：** 先 Point-LIO 几何（同步/时间戳/外参/回环），再检测器+SAM 投影融合；CMU 几何导航栈与 DETR+SAM 伪标注是两条线。
- **写回：** [`wiki/queries/go2-3d-semantic-mapping-sam-pipeline.md`](wiki/queries/go2-3d-semantic-mapping-sam-pipeline.md)（新建）
- **来源归档：**
  - [`sources/personal/go2_3d_semantic_mapping_sam_answer.md`](sources/personal/go2_3d_semantic_mapping_sam_answer.md)
  - [`sources/repos/autonomy_stack_go2.md`](sources/repos/autonomy_stack_go2.md) · [`dualmap.md`](sources/repos/dualmap.md) · [`ovo-semantic-mapping.md`](sources/repos/ovo-semantic-mapping.md) · [`ov-sam3d.md`](sources/repos/ov-sam3d.md)
  - [`sources/sites/cmu-mscv-semantic-3d-mapping.md`](sources/sites/cmu-mscv-semantic-3d-mapping.md) · [`findanything.md`](sources/sites/findanything.md)
- **开源核查（步骤 2.5）：** DualMap / OVO / OV-SAM3D / autonomy_stack_go2 / point_lio_unilidar **已开源**；FindAnything **宣称并入 OKVIS2-X**；MSCV Semantic 3D Mapping 为 **项目页文档**
- **交叉：**
  - [`wiki/entities/point-lio-unilidar.md`](wiki/entities/point-lio-unilidar.md)
  - [`wiki/overview/navigation-slam-autonomy-stack.md`](wiki/overview/navigation-slam-autonomy-stack.md)
  - [`wiki/entities/paper-notebook-real-time-polygonal-semantic-mapping-for-humanoi.md`](wiki/entities/paper-notebook-real-time-polygonal-semantic-mapping-for-humanoi.md)
  - [`wiki/queries/README.md`](wiki/queries/README.md)

## [2026-07-26] ingest | sources/repos/onnxruntime-v1.28.0.md — ONNX Runtime 1.28.0（CUDA 13 与轻量部署）；更新 wiki/entities/onnxruntime.md、onnx.md、comparisons/onnxruntime-vs-mnn-vs-tensorrt.md

- **来源归档：** [`sources/repos/onnxruntime-v1.28.0.md`](sources/repos/onnxruntime-v1.28.0.md) — GitHub Release v1.28.0（2026-07-25）
- **索引续更：** [`sources/repos/onnxruntime-official.md`](sources/repos/onnxruntime-official.md) · [`sources/repos/onnx-official.md`](sources/repos/onnx-official.md)
- **主升格 / 更新：** [`wiki/entities/onnxruntime.md`](wiki/entities/onnxruntime.md) — 版本锚点（CUDA 12/13 双线包、cuDNN/cuFFT 可选、取消 nvrtc、ONNX 1.22.0）
- **开源核查（步骤 2.5）：** **已开源（MIT）** — `microsoft/onnxruntime` 完整源码 + Release 预编译资产（含 `gpu_cuda12` / `gpu_cuda13`）
- **交叉：**
  - [`wiki/entities/onnx.md`](wiki/entities/onnx.md)
  - [`wiki/comparisons/onnxruntime-vs-mnn-vs-tensorrt.md`](wiki/comparisons/onnxruntime-vs-mnn-vs-tensorrt.md)

## [2026-07-26] ingest | sources/sites/yale-grablab-openhand.md + yale-openhand-model-f3.md — Yale OpenHand 欠驱动开源手族与 Model F3；升格 wiki/entities/yale-openhand.md

- **项目页：** [`sources/sites/yale-grablab-openhand.md`](sources/sites/yale-grablab-openhand.md) — Grab Lab OpenHand 总站（型号目录 / HDM / Couplings）
- **型号页：** [`sources/sites/yale-openhand-model-f3.md`](sources/sites/yale-openhand-model-f3.md) — Model F3（Forces-for-Free；腕相机形变估力）
- **仓库：** [`sources/repos/openhand-hardware.md`](sources/repos/openhand-hardware.md)（CAD，CC BY-NC 3.0）· [`sources/repos/openhand_node.md`](sources/repos/openhand_node.md)（O/T/T42 控制，MIT）
- **主升格：** [`wiki/entities/yale-openhand.md`](wiki/entities/yale-openhand.md)
- **开源核查（步骤 2.5）：** **已开源（CAD + 装配 + 控制/仿真）**；Model F3 视觉力估论文页内 **[paper under review]**，截至入库日未见公开代码/DOI
- **机构注册：** `schema/institutions.json` 新增 `yale`（耶鲁大学（Yale University）；aliases 含 `grablab`）
- **交叉：** topic-grasp、manipulation、contact-rich-manipulation、en02-op、allegro-hand、paper-deimel-compliant-underactuated-robotic-hand

## [2026-07-26] ingest | sources/papers/pot_vla_arxiv_2607_18016.md — 深读补全 POT-VLA（arXiv:2607.18016）；升格 wiki/entities/paper-pot-vla.md；交叉 wiki/methods/vla.md、wiki/tasks/loco-manipulation.md、wiki/entities/unitree-g1.md、wiki/entities/isaac-gr00t.md、wiki/entities/paper-hrl-stack-34-gr00t_n1.md、wiki/entities/paper-loco-manip-161-057-being-0.md；机构注册 deepcybo + zgci→zgca

- **来源归档：** [`sources/papers/pot_vla_arxiv_2607_18016.md`](sources/papers/pot_vla_arxiv_2607_18016.md) — arXiv:2607.18016（Closing the Loop in Humanoid VLA）
- **主升格：** [`wiki/entities/paper-pot-vla.md`](wiki/entities/paper-pot-vla.md) — POT / object-state divergence / GR00T-N1.7 匹配对照 **39/80→71/80**；结论与消融齐全
- **开源核查（步骤 2.5）：** **确认未开源**（截至 2026-07-26；无项目页/代码/权重）；源码运行时序图不适用
- **机构注册：** `schema/institutions.json` 新增 `deepcybo`（机智赛博（DeepCybo））；`zgci` 并入 `zgca` aliases
- **交叉：**
  - [`wiki/methods/vla.md`](wiki/methods/vla.md)
  - [`wiki/tasks/loco-manipulation.md`](wiki/tasks/loco-manipulation.md)
  - [`wiki/entities/unitree-g1.md`](wiki/entities/unitree-g1.md)
  - [`wiki/entities/isaac-gr00t.md`](wiki/entities/isaac-gr00t.md)
  - [`wiki/entities/paper-hrl-stack-34-gr00t_n1.md`](wiki/entities/paper-hrl-stack-34-gr00t_n1.md)
  - [`wiki/entities/paper-loco-manip-161-057-being-0.md`](wiki/entities/paper-loco-manip-161-057-being-0.md)

## [2026-07-26] ingest | sources/sites/research-neoteai-com.md + neoteai-com.md — NeoteAI 𝒩₀ 三件套（Foundation/VTLA/TWAM）与公司站；OpenNeoData 部分开源

- **公司站：** [`sources/sites/neoteai-com.md`](sources/sites/neoteai-com.md) — 上海新智具身智能；InTac 传感器 / 数据平台 / N 系列大模型；天使轮近亿；源自复旦 TEAI
- **研究站：** [`sources/sites/research-neoteai-com.md`](sources/sites/research-neoteai-com.md) — 2026-07-25 同日发布三项目页
- **论文归档：** [`sources/papers/n0_foundation.md`](sources/papers/n0_foundation.md) · [`n0_vtla.md`](sources/papers/n0_vtla.md) · [`n0_twam.md`](sources/papers/n0_twam.md)
- **仓库：** [`sources/repos/n0-foundation.md`](sources/repos/n0-foundation.md) · [`n0-vtla.md`](sources/repos/n0-vtla.md) · [`n0-twam.md`](sources/repos/n0-twam.md) · [`neoteai-release.md`](sources/repos/neoteai-release.md)（GitCode 传感器 SDK）
- **Wiki 实体：**
  - [`wiki/entities/neoteai.md`](wiki/entities/neoteai.md)
  - [`wiki/entities/paper-n0-foundation.md`](wiki/entities/paper-n0-foundation.md)
  - [`wiki/entities/paper-n0-vtla.md`](wiki/entities/paper-n0-vtla.md)
  - [`wiki/entities/paper-n0-twam.md`](wiki/entities/paper-n0-twam.md)
- **开源核查（步骤 2.5）：** **部分开源** — OpenNeoData（5k h，HF+ModelScope，门禁，CC-BY-NC-SA-4.0）已放；三 GitHub 仓截至入库日仅 README/diagrams；NeoForce / VTLA / TWAM 代码与权重 Roadmap **By July 31, 2026**；传感器 SDK 在 GitCode 已开放
- **机构注册：** `schema/institutions.json` 新增 `neoteai`；`fudan` 增补 `fudan-teai` / `teai` aliases
- **交叉：** visuo-tactile-fusion（范式 §6）、contact-rich-manipulation、topic-tactile、world-action-models、VT-WAM、manipulation


## [2026-07-26] ingest | sources/papers/ggps_panolog_arxiv_2607_08769.md — PanoLOG/G²PS 全景户外 3DGS；新建 wiki/entities/paper-panolog-ggps.md；交叉 PanoWorld/Glob3R/GS-Playground/导航栈；训练代码已开源，HF 部分数据

- **来源归档：**
  - [`sources/papers/ggps_panolog_arxiv_2607_08769.md`](sources/papers/ggps_panolog_arxiv_2607_08769.md) — arXiv:2607.08769
  - [`sources/sites/insta360-research-team-ggps-website.md`](sources/sites/insta360-research-team-ggps-website.md) — 项目页核查
  - [`sources/repos/ggps.md`](sources/repos/ggps.md) — 官方训练仓
- **主升格：** [`wiki/entities/paper-panolog-ggps.md`](wiki/entities/paper-panolog-ggps.md) — PanoLOG 两阶段 + G²PS 划分；Pano360；源码运行时序图
- **开源核查（步骤 2.5）：** **已开源（训练代码）** + **数据集部分**（HF：FTP/NSC/NSK）；预训练 `.ply` 与 UE 5.8 插件待发布；许可 CC BY-NC 4.0
- **机构注册：** `schema/institutions.json` 新增 `insta360`（影石研究（Insta360 Research））
- **交叉：**
  - [`wiki/entities/paper-panoworld-real-world-panoramic-generation.md`](wiki/entities/paper-panoworld-real-world-panoramic-generation.md)
  - [`wiki/entities/paper-glob3r.md`](wiki/entities/paper-glob3r.md)
  - [`wiki/entities/gs-playground.md`](wiki/entities/gs-playground.md)
  - [`wiki/entities/spark-3dgs-renderer.md`](wiki/entities/spark-3dgs-renderer.md)
  - [`wiki/entities/unreal-engine-5.md`](wiki/entities/unreal-engine-5.md)
  - [`wiki/overview/navigation-slam-autonomy-stack.md`](wiki/overview/navigation-slam-autonomy-stack.md)
  - [`wiki/overview/topic-state-estimation.md`](wiki/overview/topic-state-estimation.md)

## [2026-07-26] ingest | sources/papers/abot_world_0_arxiv_2607_19191.md — ABot-World-0（arXiv:2607.19191）高德交互式视频世界模型；升格 wiki/entities/paper-abot-world-0.md；交叉更新 wiki/methods/generative-world-models.md、wiki/concepts/video-as-simulation.md、wiki/overview/robot-world-models-training-loop-taxonomy.md、wiki/entities/paper-wan-video.md、wiki/entities/paper-abot-m05-mobile-manipulation-wam.md；归档 sources/repos/abot-world.md、sources/sites/abot-world.md

## [2026-07-26] ingest | sources/blogs/wechat_embodied_ai_lab_wam_motion_control_five_paths.md — 人形运动控制进入 WAM 的五种系统位置；补齐缺失论文实体并复用已有非 stub 节点

- **来源归档：** [`sources/blogs/wechat_embodied_ai_lab_wam_motion_control_five_paths.md`](sources/blogs/wechat_embodied_ai_lab_wam_motion_control_five_paths.md)（Agent Reach v1.5.0 + wechat-article-for-ai；<https://mp.weixin.qq.com/s/2pP9LWlsTmTAgTglFuLwSA>）
- **主升格：** [`wiki/overview/wam-motion-control-five-paths.md`](wiki/overview/wam-motion-control-five-paths.md) — ①在线规划 ②模型式 RL/适配 ③内部动力学估计 ④未来进策略 ⑤评估与动作表示
- **新建论文实体（原先缺失）：**
  - [`wiki/entities/paper-1xwm-redwood-world-model.md`](wiki/entities/paper-1xwm-redwood-world-model.md) ← [`sources/papers/1x_world_model_redwood.md`](sources/papers/1x_world_model_redwood.md) + site/repo
  - [`wiki/entities/paper-egowm-egocentric-world-model.md`](wiki/entities/paper-egowm-egocentric-world-model.md) ← arXiv:2601.15284
  - [`wiki/entities/paper-unit-unified-physical-language.md`](wiki/entities/paper-unit-unified-physical-language.md) ← arXiv:2604.19734
- **加厚复用（不新建重复节点）：** [`wiki/entities/paper-hrl-stack-33-ego_vision_world_model_for_humanoid.md`](wiki/entities/paper-hrl-stack-33-ego_vision_world_model_for_humanoid.md)（Ego-VCP，arXiv:2510.11682，MIT 开源）
- **复用已有 complete 节点：** RWM-U [`robotic-world-model-eth-rsl.md`](wiki/entities/robotic-world-model-eth-rsl.md)、LIFT [`lift-humanoid.md`](wiki/entities/lift-humanoid.md)、HAIC [`paper-haic.md`](wiki/entities/paper-haic.md)、MotionWAM、Being-M0.7、Being-H0.7（[`methods/being-h07.md`](wiki/methods/being-h07.md)）
- **开源核查：** Ego-VCP / UniT **已开源**；EgoWM **部分**；1XWM 完整评测栈未全开源（公开 1xgpt Challenge）
- **交叉：** WAM 概念页、训练闭环 taxonomy、1X Technologies 实体互链；`panniantong_agent_reach` 维护备注追加本篇

## [2026-07-25] structural | 图谱页新增「执行器驱动链」专题视图（v30 P3① 完成）—— topic-filters.js + graph.html chip + 新建枢纽页 topic-actuator-drive-chain.md

- **执行清单推进：** [`docs/checklists/tech-stack-next-phase-checklist-v30.md`](docs/checklists/tech-stack-next-phase-checklist-v30.md) P3① 打勾——「执行器驱动链」专题视图落地，收口 V30 驱动链选型闭环知识链的交互层。
- **单一事实源：** [`docs/topic-filters.js`](docs/topic-filters.js) 新增 `actuator-drive-chain` 专题（第 20 项），三处联动：`TOPIC_HUB_IDS`（枢纽页路径）、`TOPIC_FILTERS`（`segments = actuator / foc` 干净片段 + 14 条 `ids` 显式纳入 EDA/FOC/建模/总线四层驱动链页）、`TOPIC_META`（⚡ emoji + 中英标签 + 四层导读）。与 communication（总线）/ physics-fidelity（执行器建模）保持最小重叠。
- **图谱 chip：** [`docs/graph.html`](docs/graph.html) `#filter-topic-chips` 追加「⚡ 执行器驱动链 (Actuator Drive Chain)」按钮，紧随「具身评测基准」。
- **新建枢纽页：** [`wiki/overview/topic-actuator-drive-chain.md`](wiki/overview/topic-actuator-drive-chain.md) —— 四层驱动链（EDA 电路设计 → FOC 驱动固件 → 执行器建模/摩擦辨识 → 实时总线闭环）统一入口，含一句话定义、缩写速查、四层选型表、关键取舍、跨专题关系；从 [`queries/actuator-drive-chain-selection-loop.md`](wiki/queries/actuator-drive-chain-selection-loop.md) `related` + 「所属专题」行双向回链消孤儿。
- **验证：** Node harness 校验 simplefoc/kicad/bam 命中该专题、grasp 页不命中；`make lint` 0 errors / 0 孤儿；`make export graph` 后 `graph-stats.json`：节点 1891、边 16127、16 社区、`largest_community_ratio 0.195 ≤ 0.25`、`community_quality_warning: false`、0 orphans。
- **待补：** 专题视图截图归档（`.cursor-artifacts/screenshots/graph-topic-actuator-drive-chain.png`，`.gitignore` 屏蔽，PR 时补）；P3② 详情页「同专题相关页」徽标已随单一事实源自动联动，端到端验证留待下一轮。

## [2026-07-25] ingest | sources/repos/bavaria_direct_winding_calculator.md — Bavaria Direct 绕组方案计算器（Bewicklungsrechner XL）standalone 源码；新建 wiki/entities/bavaria-direct-winding-calculator.md 详解算法原理

- **来源归档：** [`sources/repos/bavaria_direct_winding_calculator.md`](sources/repos/bavaria_direct_winding_calculator.md) — 用户上传的 standalone 离线包（`winding_calc.shtml` + 2005 行 JS + 2 个 CSS），(C) 2010 Felix Niessen，GPLv3，上游为 POWERCROCO 2004 版
- **新实体：** [`wiki/entities/bavaria-direct-winding-calculator.md`](wiki/entities/bavaria-direct-winding-calculator.md) — 逐块拆源码：排布字符串 DSL（`A/a` 相与方向、`-` 空齿、`/` 子电机、`|` 槽分隔）、槽电势星形法相带分配（`Winkel = 180·2p/Q` + 60° 相带）、三步规范化、槽电流负荷（集中绕组两线圈边符号翻转）、空间 DFT 出全极数 k_w 谱、t=0.25/0.55 双时刻平衡性判据、槽口/斜槽 `sin(x)/x` 因子、短距轮转、齿槽 LCM、Canvas 绘图与展开接线图
- **开源核查（步骤 2.5）：** **已开源**（GPLv3 源码随页面明文分发，无 GitHub 仓库）；**站点已下线** — `bavaria-direct.co.za` DNS 不解析，SimpleFOC 社区串确认工具不可访问，离线包经 RCGroups 分发
- **版本核对：** 与 Wayback 2014-05-28 抓取的线上副本逐行 diff，仅 3 处外观/文案差异（`Step by step`→`Winding animation`、定子齿填充色），算法与 2010 原版一致
- **数值核验：** 把相带分配与 DFT 端口到 Python 复算，12/10、12/14 双层 0.93301，9/8 0.94521，12/10 单层 0.96593，24/22 0.94947，36/4（q=3）0.95980，24/4 短距 1/2/3 槽 0.93301/0.83652/0.68301，均与教科书 k_w 及 k_d·k_p 一致；槽口/斜槽因子逐位吻合
- **机构注册：** `schema/institutions.json` 新增 `bavaria-direct-winding-calculator`
- **风险记录：** `eval(form.Nuten.value)` + URL 参数未校验拼进 `setTimeout` 字符串 → 勿挂公网；隐式全局循环变量、`if(i=0)` 赋值笔误、手输分布式排布勿以 `|` 结尾
- **交叉：** [`wiki/overview/motor-design-workflow.md`](wiki/overview/motor-design-workflow.md)（步骤 2「拓扑与槽极」补前置筛选）、[`wiki/comparisons/open-source-torque-motor-em-design.md`](wiki/comparisons/open-source-torque-motor-em-design.md)（「教材与工具」表补「槽极组合还没定」一行）、[`wiki/entities/femm-foc-simulation.md`](wiki/entities/femm-foc-simulation.md)、[`roadmap/depth-torque-motor-design.md`](roadmap/depth-torque-motor-design.md)

## [2026-07-25] ingest | sources/blogs/cadenkraft_coreless_axial_flux_motor_part1.md + sources/blogs/cadenkraft_ironless_cycloidal_planetary_actuator.md — Caden Kraft 无铁芯轴向磁通 Part 1 升格实体；Ironless QDD 博文加厚 + pygeartrain；交叉电磁完整度对比

- **博文（新）：** [`sources/blogs/cadenkraft_coreless_axial_flux_motor_part1.md`](sources/blogs/cadenkraft_coreless_axial_flux_motor_part1.md) · [`sources/sites/cadenkraft_coreless_axial_flux_motor_part1.md`](sources/sites/cadenkraft_coreless_axial_flux_motor_part1.md)
- **新实体：** [`wiki/entities/cadenkraft-ironless-axial-flux-motor.md`](wiki/entities/cadenkraft-ironless-axial-flux-motor.md) — Halbach 无铁芯轴向；手算匝数；**未开源 CAD**
- **博文（加厚已有 Ironless）：** [`sources/blogs/cadenkraft_ironless_cycloidal_planetary_actuator.md`](sources/blogs/cadenkraft_ironless_cycloidal_planetary_actuator.md)
- **新工具仓：** [`sources/repos/pygeartrain.md`](sources/repos/pygeartrain.md) → [`wiki/entities/pygeartrain.md`](wiki/entities/pygeartrain.md)
- **加厚：** [`wiki/entities/ironless-qdd-actuator.md`](wiki/entities/ironless-qdd-actuator.md) — ~7:1、Kv≈79、728 g、MKS XDrive、FEMM 四象限百分比
- **交叉：** [`wiki/comparisons/open-source-torque-motor-em-design.md`](wiki/comparisons/open-source-torque-motor-em-design.md)、[`wiki/entities/pcb-motor.md`](wiki/entities/pcb-motor.md)、[`wiki/overview/motor-design-workflow.md`](wiki/overview/motor-design-workflow.md)
- **开源核查：** Axial Part 1 **未开源**；Ironless-QDD + pygeartrain **已开源**（MIT）

## [2026-07-25] ingest | sources/papers/halbach_permanent_multipole_magnets_1980.md + mallinson_one_sided_fluxes_1973.md + zhu_howe_halbach_pm_machines_review_2001.md — Halbach 一手三篇；升格 wiki/concepts/halbach-array.md 与三篇论文实体

- **一手文献：**
  1. Mallinson 1973 平面单侧磁通 → [`sources/papers/mallinson_one_sided_fluxes_1973.md`](sources/papers/mallinson_one_sided_fluxes_1973.md) → [`wiki/entities/paper-mallinson-one-sided-fluxes.md`](wiki/entities/paper-mallinson-one-sided-fluxes.md)
  2. Halbach 1980 圆柱/多极 REC 配方（**绿色 OA**）→ [`sources/papers/halbach_permanent_multipole_magnets_1980.md`](sources/papers/halbach_permanent_multipole_magnets_1980.md) → [`wiki/entities/paper-halbach-permanent-multipole-magnets.md`](wiki/entities/paper-halbach-permanent-multipole-magnets.md)
  3. Zhu & Howe 2001 电机综述 → [`sources/papers/zhu_howe_halbach_pm_machines_review_2001.md`](sources/papers/zhu_howe_halbach_pm_machines_review_2001.md) → [`wiki/entities/paper-zhu-howe-halbach-pm-machines-review.md`](wiki/entities/paper-zhu-howe-halbach-pm-machines-review.md)
- **概念主页：** [`wiki/concepts/halbach-array.md`](wiki/concepts/halbach-array.md)
- **机构注册：** `lbnl`、`sheffield`、`ampex`
- **交叉：** [`wiki/entities/ironless-qdd-actuator.md`](wiki/entities/ironless-qdd-actuator.md)、[`wiki/comparisons/open-source-torque-motor-em-design.md`](wiki/comparisons/open-source-torque-motor-em-design.md)、[`wiki/overview/motor-design-workflow.md`](wiki/overview/motor-design-workflow.md)、[`wiki/entities/pcb-motor.md`](wiki/entities/pcb-motor.md)

## [2026-07-25] ingest | sources/sites/femm_info.md — FEMM 官方门户/文档/示例；升格 wiki/entities/femm.md

- **站点：** [`sources/sites/femm_info.md`](sources/sites/femm_info.md) — start / documentation / examples（并核查 download、FAQ）
- **新实体：** [`wiki/entities/femm.md`](wiki/entities/femm.md) — 2D/轴对称开源 FEA；磁/静电/热流/电流场；Lua/Octave/pyFEMM；AFPL + 源码 zip
- **开源核查：** **已开源**（稳定 21Apr2019 二进制 + `femm42src_21apr2019.zip`；无原生 Linux/无 3D；分析结果可商用，再分发程序另议）
- **机构：** 注册 `schema/institutions.json` → `femm`
- **交叉：** [`femm-foc-simulation`](wiki/entities/femm-foc-simulation.md)、[`pyleecan`](wiki/entities/pyleecan.md)、[`ironless-qdd-actuator`](wiki/entities/ironless-qdd-actuator.md)、[`motor-em-simulation-software`](wiki/comparisons/motor-em-simulation-software.md)、[`open-source-torque-motor-em-design`](wiki/comparisons/open-source-torque-motor-em-design.md)、[`motor-design-workflow`](wiki/overview/motor-design-workflow.md)、[`roadmap/depth-torque-motor-design`](roadmap/depth-torque-motor-design.md)

## [2026-07-25] ingest | sources/personal/mit_mini_cheetah_learning_stack_curator.md — MIT Mini Cheetah 学习栈策展；升格 wiki/entities/mit-mini-cheetah.md；归档 Cheetah-Software / ROS / CHAMP / 控制论文集合；交叉 Katz 执行器、SRBD-MPC、四足平台、开源 QDD 对比、力矩电机路线

- **策展源：** [`sources/personal/mit_mini_cheetah_learning_stack_curator.md`](sources/personal/mit_mini_cheetah_learning_stack_curator.md)
- **新实体：** [`wiki/entities/mit-mini-cheetah.md`](wiki/entities/mit-mini-cheetah.md) — 开源边界表、执行器→驱动→软件→Convex MPC→RL→ROS/CHAMP 学习优先序、流程总览
- **论文集合：** [`sources/papers/mit_mini_cheetah_control_papers.md`](sources/papers/mit_mini_cheetah_control_papers.md)（Super Mini / ICRA 2019 / Cheetah 3 / Convex MPC / Rapid Locomotion / Falling Cat / Landing）
- **repos：** [`cheetah-software`](sources/repos/cheetah-software.md)（与本 PR 原 `mit_biomimetics_cheetah_software` 合并指向）、[`derek_th_wang_quadruped_ctrl`](sources/repos/derek_th_wang_quadruped_ctrl.md)、[`chvmp_champ`](sources/repos/chvmp_champ.md)、[`gleboss1_mini_cheetah_ros`](sources/repos/gleboss1_mini_cheetah_ros.md)
- **开源核查（步骤 2.5）：** **部分开源** — 软件与驱动已开；整机 CAD / 绕线 / 电磁设计未公开
- **交叉：** [`wiki/entities/paper-low-cost-modular-actuator-katz.md`](wiki/entities/paper-low-cost-modular-actuator-katz.md)、[`wiki/concepts/srbd-convex-mpc-wbc.md`](wiki/concepts/srbd-convex-mpc-wbc.md)、[`wiki/concepts/mpc-wbc-integration.md`](wiki/concepts/mpc-wbc-integration.md)、[`wiki/entities/quadruped-robot.md`](wiki/entities/quadruped-robot.md)、[`wiki/comparisons/open-source-qdd-actuator-projects.md`](wiki/comparisons/open-source-qdd-actuator-projects.md)、[`roadmap/depth-torque-motor-design.md`](roadmap/depth-torque-motor-design.md)
- **合并说明：** 与同日 `robot_daycare_mini_cheetah_2019` ingest 的平台页做内容合并（见下条）

## [2026-07-25] ingest | sources/blogs/robot_daycare_mini_cheetah_2019.md — Robot Daycare / bgkatz / Mini Cheetah；博文清单 12 篇论文各升格独立实体；平台与人物节点

- **站点/人物：** [`sources/sites/robot-daycare.md`](sources/sites/robot-daycare.md) → [`wiki/entities/benjamin-katz.md`](wiki/entities/benjamin-katz.md)（MIT → Boston Dynamics Atlas → Physical Intelligence）
- **GitHub：** [`sources/repos/bgkatz.md`](sources/repos/bgkatz.md)；控制栈 [`sources/repos/cheetah-software.md`](sources/repos/cheetah-software.md)
- **博文：** [`sources/blogs/robot_daycare_mini_cheetah_2019.md`](sources/blogs/robot_daycare_mini_cheetah_2019.md)
- **平台：** [`wiki/entities/mit-mini-cheetah.md`](wiki/entities/mit-mini-cheetah.md)
- **12 篇独立论文实体（博文清单）：**
  1. [`paper-mini-cheetah-platform`](wiki/entities/paper-mini-cheetah-platform.md)
  2. [`paper-wbic-mpc-mini-cheetah`](wiki/entities/paper-wbic-mpc-mini-cheetah.md)
  3. [`paper-vision-aided-dynamic-exploration-mini-cheetah`](wiki/entities/paper-vision-aided-dynamic-exploration-mini-cheetah.md)
  4. [`paper-hs-ddp-legged`](wiki/entities/paper-hs-ddp-legged.md)
  5. [`paper-mhpc`](wiki/entities/paper-mhpc.md)
  6. [`paper-bledt-rpc-thesis`](wiki/entities/paper-bledt-rpc-thesis.md)
  7. [`paper-extracting-legged-locomotion-heuristics-rpc`](wiki/entities/paper-extracting-legged-locomotion-heuristics-rpc.md)
  8. [`paper-variational-underactuated-balancing-quadruped`](wiki/entities/paper-variational-underactuated-balancing-quadruped.md)
  9. [`paper-robust-autonomous-navigation-mini-cheetah-vision`](wiki/entities/paper-robust-autonomous-navigation-mini-cheetah-vision.md)
  10. [`paper-concurrent-policy-estimator-locomotion`](wiki/entities/paper-concurrent-policy-estimator-locomotion.md)
  11. [`paper-learning-to-jump-from-pixels`](wiki/entities/paper-learning-to-jump-from-pixels.md)
  12. [`paper-rapid-locomotion-rl`](wiki/entities/paper-rapid-locomotion-rl.md)（博文名 Agile Locomotion via Model-free Learning）
- **开源核查：** Cheetah-Software / rapid-locomotion-rl / HS-DDP-MATLAB 已归档；Jump 项目页无代码；机构注册 `notre-dame`、`boston-dynamics`
- **交叉：** [`mpc-wbc-integration`](wiki/concepts/mpc-wbc-integration.md)、[`quadruped-robot`](wiki/entities/quadruped-robot.md)、[`privileged_training`](sources/papers/privileged_training.md)（修正 Concurrent 文 arXiv 为 2202.05481）


## [2026-07-25] fix(wiki): ironless-qdd-actuator BOM 美元符被 KaTeX 成对 `$...$` 吞掉

- **页面：** [`wiki/entities/ironless-qdd-actuator.md`](wiki/entities/ironless-qdd-actuator.md)
- **现象：** 「为什么重要」中 `$40…$70` 被 `docs/main.js` 行内公式正则当成数学，详情页显示成无货币符号的斜体数字
- **修复：** 改为 `40 USD` / `70 USD` / `约 384 USD`，避开成对 `$`

## [2026-07-25] ingest | sources/papers/low_cost_modular_actuator_katz_mit_2018.md — Katz MIT S.M. 2018 低成本模块化 QDD 执行器（Mini Cheetah 蓝本）；升格 wiki/entities/paper-low-cost-modular-actuator-katz.md；交叉 wiki/comparisons/open-source-qdd-actuator-projects.md、roadmap/depth-torque-motor-design.md、wiki/entities/paper-open-torque-controlled-modular-robot-solo.md、wiki/entities/paper-notebook-proprioceptive-actuator-design-in-the-mit-cheeta.md

- **论文源：** [`sources/papers/low_cost_modular_actuator_katz_mit_2018.md`](sources/papers/low_cost_modular_actuator_katz_mit_2018.md)（DSpace `b85069e2-…` / handle `1721.1/118671`；PDF MD5 `46b87d00bb5d7d665c0e2b676055d995`；**不入库 PDF**，遵 MIT thesis 版权）
- **新实体：** [`wiki/entities/paper-low-cost-modular-actuator-katz.md`](wiki/entities/paper-low-cost-modular-actuator-katz.md) — COTS U8 级电机 + **6:1** 行星 + 集成 FOC/CAN；9 kg 十二关节四足与后空翻；含源码运行时序图
- **开源核查（步骤 2.5）：** **部分开源** — 附录 A：[`3phase_integrated`](sources/repos/bgkatz_3phase_integrated.md)、[`motorcontrol`](sources/repos/bgkatz_motorcontrol.md)、[`SPIne`](sources/repos/bgkatz_spine.md)、`bgkatz/actuator`；机械 CAD 未列；叙事页 [`robot_daycare_mini_cheetah`](sources/sites/robot_daycare_mini_cheetah.md)
- **交叉：** [`wiki/comparisons/open-source-qdd-actuator-projects.md`](wiki/comparisons/open-source-qdd-actuator-projects.md)、[`roadmap/depth-torque-motor-design.md`](roadmap/depth-torque-motor-design.md)、[`wiki/entities/paper-open-torque-controlled-modular-robot-solo.md`](wiki/entities/paper-open-torque-controlled-modular-robot-solo.md)、[`wiki/entities/paper-notebook-proprioceptive-actuator-design-in-the-mit-cheeta.md`](wiki/entities/paper-notebook-proprioceptive-actuator-design-in-the-mit-cheeta.md)、[`wiki/queries/actuator-drive-chain-selection-loop.md`](wiki/queries/actuator-drive-chain-selection-loop.md)

## [2026-07-25] ingest | sources/papers/open_torque_controlled_modular_robot_solo_arxiv_1910_00093.md — Grimminger et al. Solo/ODRI 开源力矩控制模块化腿足架构；升格 wiki/entities/paper-open-torque-controlled-modular-robot-solo.md；交叉 wiki/entities/odri-solo-and-bolt.md、wiki/comparisons/open-source-qdd-actuator-projects.md；sites/repos 回链；注册 schema/institutions.json laas

- **论文源：** [`sources/papers/open_torque_controlled_modular_robot_solo_arxiv_1910_00093.md`](sources/papers/open_torque_controlled_modular_robot_solo_arxiv_1910_00093.md)（arXiv:1910.00093 / RA-L 2020）
- **新实体：** [`wiki/entities/paper-open-torque-controlled-modular-robot-solo.md`](wiki/entities/paper-open-torque-controlled-modular-robot-solo.md) — 9:1 皮带 QDD、足底接触开关、阻抗标定、kino-dynamic + CoM 阻抗 QP；含源码运行时序图
- **交叉：** [`wiki/entities/odri-solo-and-bolt.md`](wiki/entities/odri-solo-and-bolt.md)、[`wiki/comparisons/open-source-qdd-actuator-projects.md`](wiki/comparisons/open-source-qdd-actuator-projects.md)
- **开源核查：** 项目页已列 BSD-3 全栈；硬件 [`open_robot_actuator_hardware`](sources/repos/open_robot_actuator_hardware.md)、门户 [`open_dynamic_robot_initiative`](sources/sites/open_dynamic_robot_initiative.md)
- **机构：** 注册 `laas`（系统分析与架构实验室（LAAS / CNRS））

## [2026-07-25] ingest | sources/personal/open_source_torque_motor_em_design_curator.md — 按电磁设计完整度策展开源力矩电机；升格 wiki/comparisons/open-source-torque-motor-em-design.md；实体 ironless-qdd-actuator（加厚）、pyleecan、axfluxmdo、pcb-motor、femm-foc-simulation、acmop；交叉 wiki/comparisons/open-source-qdd-actuator-projects.md、motor-em-simulation-software.md、wiki/overview/motor-design-workflow.md、roadmap/depth-torque-motor-design.md、wiki/queries/actuator-drive-chain-selection-loop.md

- **策展源：** [`sources/personal/open_source_torque_motor_em_design_curator.md`](sources/personal/open_source_torque_motor_em_design_curator.md)
- **对比主页：** [`wiki/comparisons/open-source-torque-motor-em-design.md`](wiki/comparisons/open-source-torque-motor-em-design.md) — 几何/绕组/磁钢/FEM/CAD/样机六维；结论：Ironless 最完整样机链路
- **加厚：** [`wiki/entities/ironless-qdd-actuator.md`](wiki/entities/ironless-qdd-actuator.md) — 10010 定子自绕 36N42P、Halbach、FEMM 四象限、保持力矩读法
- **新实体：** [`wiki/entities/pyleecan.md`](wiki/entities/pyleecan.md)、[`wiki/entities/axfluxmdo.md`](wiki/entities/axfluxmdo.md)、[`wiki/entities/pcb-motor.md`](wiki/entities/pcb-motor.md)、[`wiki/entities/femm-foc-simulation.md`](wiki/entities/femm-foc-simulation.md)、[`wiki/entities/acmop.md`](wiki/entities/acmop.md)
- **交叉：** [`wiki/comparisons/open-source-qdd-actuator-projects.md`](wiki/comparisons/open-source-qdd-actuator-projects.md)、[`wiki/comparisons/motor-em-simulation-software.md`](wiki/comparisons/motor-em-simulation-software.md)、[`wiki/overview/motor-design-workflow.md`](wiki/overview/motor-design-workflow.md)、[`roadmap/depth-torque-motor-design.md`](roadmap/depth-torque-motor-design.md)、[`wiki/queries/actuator-drive-chain-selection-loop.md`](wiki/queries/actuator-drive-chain-selection-loop.md)
- **sources：** repos（Ironless 加厚、FEMM-FOC、pcb-motor、axfluxmdo、pyleecan、ACMOP）、sites（axfluxmdo docs、pyleecan.org、Caden 项目页加厚）

## [2026-07-25] ingest | 补齐独立实体节点：wiki/entities/cycloidal-quasi-direct-drive-actuator.md、ironless-qdd-actuator.md、vesc.md；加厚 opentorque/tinymovr/moteus；交叉 wiki/comparisons/open-source-qdd-actuator-projects.md

- **新独立节点（有细节，非 stub）：**
  - [`wiki/entities/cycloidal-quasi-direct-drive-actuator.md`](wiki/entities/cycloidal-quasi-direct-drive-actuator.md) — Jeong 双摆线 10:1 + 36N42P + moteus-c1
  - [`wiki/entities/ironless-qdd-actuator.md`](wiki/entities/ironless-qdd-actuator.md) — Caden Kraft Halbach 无铁芯 + 摆线行星
  - [`wiki/entities/vesc.md`](wiki/entities/vesc.md) — vedderb/bldc + bldc-hardware
- **加厚既有节点：** [`opentorque-actuator`](wiki/entities/opentorque-actuator.md)、[`tinymovr`](wiki/entities/tinymovr.md)（标明 v3.1+ 源码私有）、[`moteus`](wiki/entities/moteus.md)（板型规格表）
- **未新建重复节点：** ODRI 仍用 [`odri-solo-and-bolt`](wiki/entities/odri-solo-and-bolt.md)；Doggo 仍用 [`stanford-doggo-and-pupper`](wiki/entities/stanford-doggo-and-pupper.md)；BHL/SimpleFOC/Internal Cycloidal/Urs 论文沿用既有页

## [2026-07-25] ingest | sources/personal/open_source_qdd_actuator_learning_curator.md — 开源 QDD/力矩关节两类项目与学习阶梯；升格 wiki/comparisons/open-source-qdd-actuator-projects.md；实体 moteus/tinymovr/opentorque/internal-cycloidal/paper-3d-printed-open-source-actuators-legged；深化 odri/berkeley-humanoid-lite/stanford-doggo；交叉 roadmap/depth-torque-motor-design.md

- **策展源：** [`sources/personal/open_source_qdd_actuator_learning_curator.md`](sources/personal/open_source_qdd_actuator_learning_curator.md)
- **对比主页：** [`wiki/comparisons/open-source-qdd-actuator-projects.md`](wiki/comparisons/open-source-qdd-actuator-projects.md)
- **新实体：** [`wiki/entities/moteus.md`](wiki/entities/moteus.md)、[`wiki/entities/tinymovr.md`](wiki/entities/tinymovr.md)、[`wiki/entities/opentorque-actuator.md`](wiki/entities/opentorque-actuator.md)、[`wiki/entities/internal-cycloidal-actuator.md`](wiki/entities/internal-cycloidal-actuator.md)、[`wiki/entities/paper-3d-printed-open-source-actuators-legged.md`](wiki/entities/paper-3d-printed-open-source-actuators-legged.md)
- **深化：** [`wiki/entities/odri-solo-and-bolt.md`](wiki/entities/odri-solo-and-bolt.md)、[`wiki/entities/berkeley-humanoid-lite.md`](wiki/entities/berkeley-humanoid-lite.md)、[`wiki/entities/stanford-doggo-and-pupper.md`](wiki/entities/stanford-doggo-and-pupper.md)、[`wiki/entities/simplefoc.md`](wiki/entities/simplefoc.md)
- **路线/选型交叉：** [`roadmap/depth-torque-motor-design.md`](roadmap/depth-torque-motor-design.md)、[`wiki/entities/open-source-humanoid-hardware.md`](wiki/entities/open-source-humanoid-hardware.md)、[`wiki/queries/actuator-drive-chain-selection-loop.md`](wiki/queries/actuator-drive-chain-selection-loop.md)
- **sources：** repos（ODRI/BHL/ICA/OpenTorque/Doggo/moteus/Tinymovr/VESC/Jeong/Ironless）、sites（ODRI/BHL/Aaed/Caden）、paper `2202.12395`

## [2026-07-25] ingest | sources/repos/open-dreamer.md — Dreamer 4 开源 JAX 复现；升格 wiki/entities/open-dreamer.md；交叉 wiki/overview/world-models-route-03-virtual-sandbox.md、wiki/entities/paper-shenlan-wm-13-dreamerv3.md、wiki/concepts/latent-imagination.md、wiki/methods/generative-world-models.md、wiki/methods/model-based-rl.md；sites: sources/sites/open-dreamer.md；推理仓: sources/repos/reactor-team-open-dreamer.md

## [2026-07-25] structural | 新增「人形整机硬件设计」纵深路线（第 21 条）：机械 / 电气 / 通信三条设计链

- **新增路线页：** [`roadmap/depth-humanoid-hardware-design.md`](roadmap/depth-humanoid-hardware-design.md) — Stage 0 指标与质量/功率/延迟三大预算 → Stage 1 构型与机械布局 → Stage 2 结构详设与验证（强度/疲劳/模态/公差链）→ Stage 3 电气架构与配电（母线/DC-DC/线束/E-Stop→STO）→ Stage 4 EMC 接地与 PDU 落板 → Stage 5 通信分域与时序预算（拓扑/同步/延迟）→ Stage 6 整机集成 bring-up 与验收交付（含可仿真数字副本）。
- **新增知识页（3）：**
  - [`wiki/concepts/humanoid-mechanical-layout-design.md`](wiki/concepts/humanoid-mechanical-layout-design.md) — 自由度分配、执行器布置、质量/惯量分布、刚度强度公差四层决策
  - [`wiki/concepts/robot-power-distribution-architecture.md`](wiki/concepts/robot-power-distribution-architecture.md) — 能量链分域、上电时序、分级保护、线束、安全回路与 EMC 接地
  - [`wiki/concepts/robot-onboard-communication-architecture.md`](wiki/concepts/robot-onboard-communication-architecture.md) — 三层分域、带宽与周期计算、三级时钟同步、端到端延迟预算与降级策略
- **定位分工：** [力矩电机设计纵深](roadmap/depth-torque-motor-design.md) 交付「一个可信的关节模组」，本路线交付「一台可信的整机」；Stage 3 功率预算与 Stage 5 延迟预算是关节模组交付物的整机侧约束。
- **排序：** 按起点里程碑历史顺序插在传统控制（ZMP 1972）之后、安全控制（CLF 1983）之前（起点：WABOT-1 全尺寸人形整机，1973）；纵深路线总数 20 → 21。
- **交叉更新：** [`roadmap/README.md`](roadmap/README.md)、[`roadmap/motion-control.md`](roadmap/motion-control.md)（导航 / 可选纵深表 / 末尾列表）、[`README.md`](README.md)、[`index.md`](index.md)、[`docs/index.html`](docs/index.html)（首页按钮「整机硬件」+ 折叠文案）、[`docs/main.js`](docs/main.js)、[`scripts/utils/community_labels.py`](scripts/utils/community_labels.py)，以及既有 20 条纵深路线的「其它纵深路径」清单。
- **wiki 侧挂接：** [`wiki/overview/humanoid-hardware-101-technology-map.md`](wiki/overview/humanoid-hardware-101-technology-map.md)、[`wiki/overview/humanoid-actuator-102-technology-map.md`](wiki/overview/humanoid-actuator-102-technology-map.md)、[`wiki/overview/humanoid-hardware-101-chassis-materials.md`](wiki/overview/humanoid-hardware-101-chassis-materials.md)、[`wiki/overview/humanoid-hardware-101-power-compute-electronics.md`](wiki/overview/humanoid-hardware-101-power-compute-electronics.md)、[`wiki/overview/topic-communication.md`](wiki/overview/topic-communication.md)、[`wiki/overview/topic-systems-engineering.md`](wiki/overview/topic-systems-engineering.md)。
- **门禁：** `make lint` 0 errors；`make ci-preflight` 同步 catalog/exports/搜索索引/统计。

## [2026-07-25] ingest | sources/blogs/wechat_robot_lecture_bioinspired_multimodal_2026-07-25.md — Agent Reach 抓取「机器人大讲堂」微信导读，补全 wiki/entities/paper-bioinspired-multimodal-robotics.md（三阶段历史、六权衡、指标数值例、切换分类、三模块架构）；raw: sources/raw/wechat_robot_lecture_bioinspired_multimodal_2026-07-25/

## [2026-07-25] ingest | sources/papers/bioinspired_multimodal_robotics_scirobotics_2026.md — Science Robotics 2026 Review「Bioinspired multimodal robotics」；升格 wiki/entities/paper-bioinspired-multimodal-robotics.md；交叉 wiki/tasks/locomotion.md、wiki/tasks/hybrid-locomotion.md、wiki/entities/paper-aerial-aquatic-remora-hitchhiking-robot.md、wiki/entities/paper-miniature-deep-sea-morphable-robot.md、wiki/entities/paper-octopus-inspired-esoam-soft-arm.md、wiki/entities/paper-learning-to-adapt-bio-inspired-quadruped-gait.md；注册机构 dut

## [2026-07-24] structural | V30 P2 事实库扩展：canonical-facts 240 → 250（驱动链选型矛盾检测规则）

- **改动：** [`schema/canonical-facts.json`](schema/canonical-facts.json) 新增 10 条「执行器驱动链选型闭环」矛盾检测规则，配合 V30 P1 交付的驱动链知识链（`queries/actuator-drive-chain-selection-loop.md` / `concepts/torque-source-abstraction-gap.md`）沉淀取舍事实。
- **新增规则（10）：** 理想力矩源假设 vs 摩擦/齿隙实际、数据手册峰值力矩 vs 持续力矩热约束、FOC 电流环带宽 vs 编码器分辨率制约、总线周期快 ≠ 闭环带宽高、执行器网络拟合好 vs 分布外温升漂移、高减速比力矩大 vs 反驱透明度损失、开源 EDA 够用 vs 高速多层板信号完整性、自研驱动板省钱 vs 可靠性/调试成本、仿真理想执行器 vs 真机 sim2real gap、驱动固件开环标定 vs 负载在环辨识。
- **校验：** 三段式（terms / pos_claims / neg_claims）与既有 240 条一致；逐条脚本校验对现存驱动链页（`actuator-drive-chain-selection-loop` / `torque-source-abstraction-gap` / `simplefoc` / `humanoid-actuator-102-gear-reflected-inertia`）均有 pos 命中；neg_claims 仅匹配朴素误判措辞，规避页面自身 `≠` / 误判速查行造成的误伤。
- **门禁：** `make lint` 0 errors、潜在矛盾 0 个（图谱节点/边不受本次纯 schema 改动影响，仍为 P1 交付基线）。
- **清单：** [`docs/checklists/tech-stack-next-phase-checklist-v30.md`](docs/checklists/tech-stack-next-phase-checklist-v30.md) P2 打勾。

## [2026-07-24] structural | unitreerobotics 去 stub / 去重：主线仓深化为 complete，合并重复节点

- **问题：** 批量「一仓一 stub」产生大量模板页，且与已有深页（如 `unitree-ros` 覆盖 `unitree_ros_to_real`）重复。
- **删除的重复/周边 stub（26）：** `unitree-ros-to-real`、`unitree-ros2-to-real`、`unitree-sdk2-python`、`unitree-dds-wrapper`、`unitree-actuator-sdk`、`teleimager`、`televuer`、`kinect-teleoperate`、Z1 三分页、灵巧手四分页、`unilidar-sdk`、以及 Publications/logging-mp/slam demos 等元仓 stub。
- **合并详情页：**
  - [`wiki/entities/unitree-sdk2.md`](wiki/entities/unitree-sdk2.md)（含 Python / 周边 SDK）
  - [`wiki/entities/z1-sdk.md`](wiki/entities/z1-sdk.md)（Z1 四仓）
  - [`wiki/entities/unitree-dexterous-hand-services.md`](wiki/entities/unitree-dexterous-hand-services.md)
  - [`wiki/entities/unilidar-sdk2.md`](wiki/entities/unilidar-sdk2.md)（L1+L2）
  - [`wiki/entities/xr-teleoperate.md`](wiki/entities/xr-teleoperate.md)（吸收 teleimager/televuer/kinect sources）
- **深化为 complete 的主线仓：** sdk2、ros2、mujoco、rl-gym、rl-lab、lerobot、sim-isaaclab、xr-teleoperate、unifolm-vla、unifolm-wma、guide、legged-sdk、model、qmini、point-lio 等。
- **枢纽更新：** [`wiki/entities/unitree.md`](wiki/entities/unitree.md) 改为「有详情节点 + 仅 sources 归档」策略表。
- **交叉：** [`wiki/entities/unitree-ros.md`](wiki/entities/unitree-ros.md)、[`wiki/entities/unitree-unistore.md`](wiki/entities/unitree-unistore.md)、各 `sources/repos/*` 映射字段。

## [2026-07-24] ingest | sources/repos/unitree.md — unitreerobotics 组织活跃仓全量独立节点（45）；跳过 7 个过时/元仓

- **资料：** [github.com/unitreerobotics](https://github.com/unitreerobotics)（组织 API，截至 2026-07-24 约 52 公开仓）
- **策略：** 为每个活跃仓库建立独立 `sources/repos/<name>.md` + `wiki/entities/<slug>.md`；组织枢纽 [`wiki/entities/unitree.md`](wiki/entities/unitree.md) 增加全量节点清单与跳过说明。
- **跳过（过时/元仓）：** `.github`、`unitreerobotics.github.io`、`Acknowledgement`、`laikago_ros`、`unitree_pybullet`、`aliengo_sdk`、`unitree_cad`
- **保留但标注 deprecated：** [`wiki/entities/unitree-model.md`](wiki/entities/unitree-model.md)（GitHub deprecated → Hugging Face）
- **既有深页未覆盖写：** [`wiki/entities/unitree-ros.md`](wiki/entities/unitree-ros.md)、[`wiki/entities/unitree-rl-mjlab.md`](wiki/entities/unitree-rl-mjlab.md)（仅补交叉链接）
- **新独立节点（示例，完整清单见组织枢纽）：**
  - [`wiki/entities/unitree-sdk2.md`](wiki/entities/unitree-sdk2.md)
  - [`wiki/entities/unitree-sdk2-python.md`](wiki/entities/unitree-sdk2-python.md)
  - [`wiki/entities/unitree-ros2.md`](wiki/entities/unitree-ros2.md)
  - [`wiki/entities/unitree-ros-to-real.md`](wiki/entities/unitree-ros-to-real.md)
  - [`wiki/entities/unitree-rl-gym.md`](wiki/entities/unitree-rl-gym.md)
  - [`wiki/entities/unitree-rl-lab.md`](wiki/entities/unitree-rl-lab.md)
  - [`wiki/entities/unitree-mujoco.md`](wiki/entities/unitree-mujoco.md)
  - [`wiki/entities/xr-teleoperate.md`](wiki/entities/xr-teleoperate.md)
  - [`wiki/entities/unitree-lerobot.md`](wiki/entities/unitree-lerobot.md)
  - [`wiki/entities/unitree-sim-isaaclab.md`](wiki/entities/unitree-sim-isaaclab.md)
  - [`wiki/entities/unifolm-vla.md`](wiki/entities/unifolm-vla.md)
  - [`wiki/entities/unifolm-world-model-action.md`](wiki/entities/unifolm-world-model-action.md)
  - [`wiki/entities/point-lio-unilidar.md`](wiki/entities/point-lio-unilidar.md)
  - [`wiki/entities/qmini.md`](wiki/entities/qmini.md)
  - 以及 Z1 / 灵巧手 / LiDAR / 工具 / SLAM / UniStore 模板等其余活跃仓节点
- **交叉更新：**
  - [`wiki/entities/unitree.md`](wiki/entities/unitree.md)
  - [`sources/repos/unitree.md`](sources/repos/unitree.md)
  - [`wiki/entities/unitree-ros.md`](wiki/entities/unitree-ros.md)
  - [`wiki/entities/unitree-rl-mjlab.md`](wiki/entities/unitree-rl-mjlab.md)
  - [`wiki/tasks/teleoperation.md`](wiki/tasks/teleoperation.md)
  - [`wiki/entities/lerobot.md`](wiki/entities/lerobot.md)

## [2026-07-24] ingest | sources/sites/motor_dynamometer_primary_refs.md — 测功机一手资料合集入库；升格 wiki/concepts/motor-dynamometer.md

- **资料（一手）：**
  - [GB/T 43200-2023](https://openstd.samr.gov.cn/bzgk/std/newGbInfo?hcno=B2E40B3445ACE9E166E8E402E89853AF) 机器人一体化关节性能及试验方法
  - [IEC 60034-2-1:2024](https://webstore.iec.ch/en/publication/67756) 旋转电机损耗与效率试验方法
  - [Magtrol Manuals](https://www.magtrol.com/manuals/)（HD/ED、WB、M-TEST 7）
  - [Capo01/odrive_based_electric_motor_dynamometer](https://github.com/Capo01/odrive_based_electric_motor_dynamometer) 开源四象限对拖
  - [AIP 旋转关节/人形电机对拖方案](https://aipuo.com/products/1336.html)（对标 GB/T 43200；商业未开源）
  - 艾诺「三层测试」工程分层文（仪器网转载，补边界）
- **归档：**
  - [`sources/sites/motor_dynamometer_primary_refs.md`](sources/sites/motor_dynamometer_primary_refs.md)
  - [`sources/sites/gbt_43200_2023_robot_joint_performance.md`](sources/sites/gbt_43200_2023_robot_joint_performance.md)
  - [`sources/sites/iec_60034_2_1_motor_efficiency.md`](sources/sites/iec_60034_2_1_motor_efficiency.md)
  - [`sources/sites/magtrol_dynamometer_manuals.md`](sources/sites/magtrol_dynamometer_manuals.md)
  - [`sources/sites/aip_robot_joint_dynamometer.md`](sources/sites/aip_robot_joint_dynamometer.md)
  - [`sources/repos/odrive_based_electric_motor_dynamometer.md`](sources/repos/odrive_based_electric_motor_dynamometer.md)
- **开源核查：** ODrive 对拖仓 **已开源**；Magtrol / AIP / 国标正文 **确认未开源**（手册或元数据公开）
- **升格：** [`wiki/concepts/motor-dynamometer.md`](wiki/concepts/motor-dynamometer.md)
- **交叉更新：**
  - [`wiki/overview/motor-design-workflow.md`](wiki/overview/motor-design-workflow.md)
  - [`wiki/concepts/motor-torque-speed-curve.md`](wiki/concepts/motor-torque-speed-curve.md)
  - [`wiki/concepts/motor-torque-current-curve.md`](wiki/concepts/motor-torque-current-curve.md)
  - [`roadmap/depth-torque-motor-design.md`](roadmap/depth-torque-motor-design.md)（Stage 6）
  - [`sources/README.md`](sources/README.md)


## [2026-07-24] fix(docs) | README 在线演示「知识图谱」粗体链接 `**` 原样显示

- **现象：** `**[知识图谱](url)**`（粗体包住整段链接）在 GitHub Markdown 下星号可原样露出；同模式亦见于「如何贡献」的 `**[CONTRIBUTING.md](...)**`。
- **修复：** 改为链接内粗体 `[**知识图谱**](url)` / `[**CONTRIBUTING.md**](CONTRIBUTING.md)`。

## [2026-07-24] fix(ux) | 纵深路线「路线一览」Mermaid 节点 `**Stage**` 原样显示 — 改为 `<b>` 并渲染前规范化

- **现象：** 二十条 `roadmap/depth-*.md`「路线一览」flowchart 节点写 `**Stage N**`，与已有 `<br/>` / `<em>` 混用；站点 `htmlLabels` 不解析 Markdown 粗体，星号原样出现在图上。
- **修复：**
  - 全部纵深路线一览标签：`**Stage N**` → `<b>Stage N</b>`（`roadmap/depth-*.md`）
  - `docs/main.js`：`normalizeMermaidMarkdownEmphasis` 在 `mermaid.run` 前将残留 `**…**` 转为 `<b>`
  - 回归：`tests/test_content_sync.py`；清单：`docs/checklists/frontend-optimization-v1.md`
- **涉及路线（示例）：** [`roadmap/depth-bfm.md`](roadmap/depth-bfm.md)、[`roadmap/depth-vla.md`](roadmap/depth-vla.md)、[`roadmap/depth-wam.md`](roadmap/depth-wam.md)（及其余 17 条 depth 纵深）

## [2026-07-24] ingest | sources/papers/zonda_arxiv_2607_21025.md — ZONDA 零样本多楼层动态 ObjectNav；升格 wiki/entities/paper-zonda.md；交叉更新 wiki/tasks/vision-language-navigation.md、wiki/overview/vln-open-source-repro-paradigms.md、wiki/entities/habitat-sim.md、wiki/entities/paper-uni-lavira.md、wiki/concepts/sim2real.md

- **资料：** [ZONDA arXiv:2607.21025](https://arxiv.org/abs/2607.21025)（v1，2026-07-23）
- **归档：** [`sources/papers/zonda_arxiv_2607_21025.md`](sources/papers/zonda_arxiv_2607_21025.md)
- **开源核查：** **确认未开源**（abs/HTML/PDF 无项目页 / GitHub；公开检索无官方仓）
- **升格：** [`wiki/entities/paper-zonda.md`](wiki/entities/paper-zonda.md)
- **交叉更新：**
  - [`wiki/tasks/vision-language-navigation.md`](wiki/tasks/vision-language-navigation.md)
  - [`wiki/overview/vln-open-source-repro-paradigms.md`](wiki/overview/vln-open-source-repro-paradigms.md)
  - [`wiki/entities/habitat-sim.md`](wiki/entities/habitat-sim.md)
  - [`wiki/entities/paper-uni-lavira.md`](wiki/entities/paper-uni-lavira.md)
  - [`wiki/concepts/sim2real.md`](wiki/concepts/sim2real.md)
- **机构注册：** `sustech`、`gbu`、`direct-drive-tech` → [`schema/institutions.json`](schema/institutions.json)

## [2026-07-24] ingest | sources/papers/egoworld_arxiv_2506_17896.md + sources/sites/egoworld-github-io.md + sources/repos/egoworld.md + sources/blogs/stellarnex_egoworld_100w.md + sources/sites/stellarnex-robotics.md — EgoWorld（ICLR 2026 exo→ego）与 StellarNex EgoWorld-100W 同名消歧入库；升格 wiki/entities/paper-egoworld.md、wiki/entities/egoworld-100w.md；交叉更新 wiki/overview/ego-category-01-data-collection.md、wiki/tasks/manipulation.md、wiki/methods/egoscale.md

- **资料 A（方法）：** [EgoWorld ICLR 2026](https://arxiv.org/abs/2506.17896) · [项目页](https://redorangeyellowy.github.io/EgoWorld/) · [代码](https://github.com/redorangeyellowy/EgoWorld)
- **资料 B（数据）：** [EgoWorld-100W blog](https://stellarnexrobotics.com/blog) · [StellarNex 官网](https://stellarnexrobotics.com/)
- **归档：**
  - [`sources/papers/egoworld_arxiv_2506_17896.md`](sources/papers/egoworld_arxiv_2506_17896.md)
  - [`sources/sites/egoworld-github-io.md`](sources/sites/egoworld-github-io.md)
  - [`sources/repos/egoworld.md`](sources/repos/egoworld.md)
  - [`sources/blogs/stellarnex_egoworld_100w.md`](sources/blogs/stellarnex_egoworld_100w.md)
  - [`sources/sites/stellarnex-robotics.md`](sources/sites/stellarnex-robotics.md)
- **开源核查：** EgoWorld 论文仓 **已开源**（MIT + train/test）；EgoWorld-100W **申请制合作开放**（无公开全量下载）
- **升格：**
  - [`wiki/entities/paper-egoworld.md`](wiki/entities/paper-egoworld.md)
  - [`wiki/entities/egoworld-100w.md`](wiki/entities/egoworld-100w.md)
- **交叉更新：**
  - [`wiki/overview/ego-category-01-data-collection.md`](wiki/overview/ego-category-01-data-collection.md)
  - [`wiki/tasks/manipulation.md`](wiki/tasks/manipulation.md)
  - [`wiki/methods/egoscale.md`](wiki/methods/egoscale.md)
- **机构注册：** `lg-electronics`、`oxford`、`stellarnex-robotics` → [`schema/institutions.json`](schema/institutions.json)

## [2026-07-24] ingest | sources/papers/egoverse_arxiv_2604_07607.md + sources/sites/egoverse-ai.md + sources/repos/egoverse.md — EgoVerse 联盟 egocentric 人示教活数据集与跨实验室共训研究；升格 wiki/entities/paper-egoverse.md；交叉更新 EgoWAM / EgoScale / HumanNet / manipulation / ego hubs / index.md

- **资料：** [EgoVerse 项目页](https://egoverse.ai/) · [arXiv:2604.07607](https://arxiv.org/abs/2604.07607) · [GaTech-RL2/EgoVerse](https://github.com/GaTech-RL2/EgoVerse)
- **归档：**
  - [`sources/papers/egoverse_arxiv_2604_07607.md`](sources/papers/egoverse_arxiv_2604_07607.md)
  - [`sources/sites/egoverse-ai.md`](sources/sites/egoverse-ai.md)
  - [`sources/repos/egoverse.md`](sources/repos/egoverse.md)
- **开源核查：** 代码 **MIT 已开源**；数据经 Dataset browser + SQL/R2 **受控开放**（需云凭证）
- **升格：** [`wiki/entities/paper-egoverse.md`](wiki/entities/paper-egoverse.md)
- **交叉更新：**
  - [`wiki/entities/paper-egowam-egocentric-human-wam-co-training.md`](wiki/entities/paper-egowam-egocentric-human-wam-co-training.md)
  - [`wiki/methods/egoscale.md`](wiki/methods/egoscale.md)
  - [`wiki/methods/imitation-learning.md`](wiki/methods/imitation-learning.md)
  - [`wiki/entities/humannet.md`](wiki/entities/humannet.md)
  - [`wiki/comparisons/humannet-table1-human-video-corpora.md`](wiki/comparisons/humannet-table1-human-video-corpora.md)
  - [`wiki/concepts/world-action-models.md`](wiki/concepts/world-action-models.md)
  - [`wiki/tasks/manipulation.md`](wiki/tasks/manipulation.md)
  - [`wiki/overview/ego-category-01-data-collection.md`](wiki/overview/ego-category-01-data-collection.md)
  - [`wiki/overview/ego-category-02-human-to-robot.md`](wiki/overview/ego-category-02-human-to-robot.md)
  - [`index.md`](index.md)

## [2026-07-24] ingest | sources/blogs/wechat_urkl_faq_01.md + sources/sites/urkl-org.md — 众擎 URKL 官方 FAQ 与 urkl.org 独立导读；更新 wiki/entities/urkl.md、wiki/tasks/teleoperation.md、sources/sites/engineai-urkl.md

- **资料 A：** [「众」所周知丨众擎URKL常见疑问解答（01篇）](https://mp.weixin.qq.com/s/mZiWICUnOJZ5bBObVcvUcw)（URKL / 众擎官方公众号，2026-07-24）
- **抓取：** Agent Reach v1.5.0 + `wechat-article-for-ai`；raw → [`sources/raw/wechat_urkl_faq_01_2026-07-24/`](sources/raw/wechat_urkl_faq_01_2026-07-24/)
- **资料 B：** [urkl.org](https://urkl.org/) — Independent fan guide（事实复核标注至 2026-07-23）
- **归档：**
  - [`sources/blogs/wechat_urkl_faq_01.md`](sources/blogs/wechat_urkl_faq_01.md)
  - [`sources/sites/urkl-org.md`](sources/sites/urkl-org.md)
- **开源核查：** 官方 FAQ **宣称将开源本届赛事相关代码**（无仓库 URL）→ 宣称将开源 / 待发布；urkl.org 为第三方导读（无代码）
- **升格策略：** **不新建**实体页；补充已有 [`wiki/entities/urkl.md`](wiki/entities/urkl.md)
- **交叉更新：**
  - [`wiki/entities/urkl.md`](wiki/entities/urkl.md)
  - [`wiki/tasks/teleoperation.md`](wiki/tasks/teleoperation.md)
  - [`roadmap/depth-humanoid-boxing.md`](roadmap/depth-humanoid-boxing.md)
  - [`sources/sites/engineai-urkl.md`](sources/sites/engineai-urkl.md)

## [2026-07-24] ingest | sources/blogs/generalist_thousand_hands.md — GEN-1 千手跨末端泛化；升格 wiki/entities/generalist-gen1-thousand-hands.md；更新 generalist-ai-robotics / topic-cross-embodiment / foundation-policy / embodied-scaling-laws / manipulation / cross-embodiment-transfer-strategy
- **资料：** [Towards Machines with a Thousand Hands](https://generalistai.com/blog/towards-machines-with-a-thousand-hands)（Generalist AI，2026-07）
- **归档：** [`sources/blogs/generalist_thousand_hands.md`](sources/blogs/generalist_thousand_hands.md)
- **开源核查：** 确认未开源（无 GitHub / HF / 公开数据集）
- **升格：** [`wiki/entities/generalist-gen1-thousand-hands.md`](wiki/entities/generalist-gen1-thousand-hands.md)
- **交叉更新：**
  - [`wiki/entities/generalist-ai-robotics.md`](wiki/entities/generalist-ai-robotics.md)
  - [`wiki/overview/topic-cross-embodiment.md`](wiki/overview/topic-cross-embodiment.md)
  - [`wiki/concepts/foundation-policy.md`](wiki/concepts/foundation-policy.md)
  - [`wiki/concepts/embodied-scaling-laws.md`](wiki/concepts/embodied-scaling-laws.md)
  - [`wiki/tasks/manipulation.md`](wiki/tasks/manipulation.md)
  - [`wiki/queries/cross-embodiment-transfer-strategy.md`](wiki/queries/cross-embodiment-transfer-strategy.md)


## [2026-07-24] ingest | sources/blogs/wechat_shenlan_rss2026_eight_papers_2026-07-24.md — 深蓝具身智能 RSS 2026 Final List 八篇盘点（复用已有节点）

- **资料：** [顶会 RSS 2026 释放明确信号：8篇力作聚焦底层控制、轨迹优化、物理硬件](https://mp.weixin.qq.com/s/qjhBjBqTYHcfnndPFNVb-g)（深蓝具身智能，2026-07-24）
- **抓取：** Agent Reach v1.5.0 + `wechat-article-for-ai`；raw → [`sources/raw/wechat_shenlan_rss2026_eight_papers_2026-07-24/`](sources/raw/wechat_shenlan_rss2026_eight_papers_2026-07-24/)
- **策略：** Final List 8 篇均已有 `status: complete` 独立节点（姊妹源 [量子位 RSS 2026](sources/blogs/wechat_qbitai_rss2026_awards_2026-07-16.md)）；**0 新建 / 0 stub**；FlashSAC 保持唯一枢纽 [`wiki/methods/flashsac.md`](wiki/methods/flashsac.md)（不另造 `paper-flashsac`）
- **交叉更新：**
  - [`wiki/methods/flashsac.md`](wiki/methods/flashsac.md)
  - [`wiki/entities/paper-muninn-trajectory-diffusion-acceleration.md`](wiki/entities/paper-muninn-trajectory-diffusion-acceleration.md)
  - [`wiki/entities/paper-neuralactuator-neural-actuation-modeling.md`](wiki/entities/paper-neuralactuator-neural-actuation-modeling.md)
  - [`wiki/entities/paper-oat-ordered-action-tokenization.md`](wiki/entities/paper-oat-ordered-action-tokenization.md)
  - [`wiki/entities/paper-dapl-extrinsic-dexterity-clutter.md`](wiki/entities/paper-dapl-extrinsic-dexterity-clutter.md)
  - [`wiki/entities/paper-cunrto-gpu-robust-trajectory-optimization.md`](wiki/entities/paper-cunrto-gpu-robust-trajectory-optimization.md)
  - [`wiki/entities/paper-unified-fluid-robot-multiphysics-swimming.md`](wiki/entities/paper-unified-fluid-robot-multiphysics-swimming.md)
  - [`wiki/entities/paper-automated-facial-mechanisms-animatronic.md`](wiki/entities/paper-automated-facial-mechanisms-animatronic.md)
- **补齐：** 上表中 6 篇缺 `## 结论` 的 `paper-*` 页按 schema 补结论（NeuralActuator 已有）；姊妹博客互链

## [2026-07-24] structural | Top-100 论文枢纽补齐「结论」— hub-rankings paper 榜 paper-* 全覆盖

- **范围：** [`exports/hub-rankings.json`](exports/hub-rankings.json) `paper` 榜 Top-100 中全部 `wiki/entities/paper-*.md`（方法页 / 非 paper-* 实体不在「结论」政策内）
- **本轮新增：** 第 51–100 名中缺结论的 36 页；第 1–50 名已在同日先一轮补齐
- **工具：** [`scripts/insert_paper_conclusion.py`](scripts/insert_paper_conclusion.py)
- **格式：** 1 句总判 + 3–7 条可操作要点（对齐 [`schema/page-types.md`](schema/page-types.md)）
- **本轮补齐页面：**
  - [`wiki/entities/paper-explicit-stair-geometry-humanoid-locomotion.md`](wiki/entities/paper-explicit-stair-geometry-humanoid-locomotion.md)
  - [`wiki/entities/paper-hrl-stack-06-hdmi.md`](wiki/entities/paper-hrl-stack-06-hdmi.md)
  - [`wiki/entities/paper-rove-humanoid-vla-intervention.md`](wiki/entities/paper-rove-humanoid-vla-intervention.md)
  - [`wiki/entities/paper-splitadapter-load-aware-loco-manipulation.md`](wiki/entities/paper-splitadapter-load-aware-loco-manipulation.md)
  - [`wiki/entities/paper-omnicontact-humanoid-loco-manipulation.md`](wiki/entities/paper-omnicontact-humanoid-loco-manipulation.md)
  - [`wiki/entities/paper-pace-sim2real-legged-robots.md`](wiki/entities/paper-pace-sim2real-legged-robots.md)
  - [`wiki/entities/paper-ssr-humanoid-open-world-traversal.md`](wiki/entities/paper-ssr-humanoid-open-world-traversal.md)
  - [`wiki/entities/paper-vesta-generalist-embodied-reasoning.md`](wiki/entities/paper-vesta-generalist-embodied-reasoning.md)
  - [`wiki/entities/paper-wem-world-ego-modeling.md`](wiki/entities/paper-wem-world-ego-modeling.md)
  - [`wiki/entities/paper-adp.md`](wiki/entities/paper-adp.md)
  - [`wiki/entities/paper-cwi-composite-humanoid-whole-body-imitation.md`](wiki/entities/paper-cwi-composite-humanoid-whole-body-imitation.md)
  - [`wiki/entities/paper-hrl-stack-27-learning_whole_body_humanoid_locomot.md`](wiki/entities/paper-hrl-stack-27-learning_whole_body_humanoid_locomot.md)
  - [`wiki/entities/paper-omnitactune-tactile-residual-adaptation.md`](wiki/entities/paper-omnitactune-tactile-residual-adaptation.md)
  - [`wiki/entities/paper-steam-advantage-modeling.md`](wiki/entities/paper-steam-advantage-modeling.md)
  - [`wiki/entities/paper-chord-contact-wrench-dexterous-manipulation.md`](wiki/entities/paper-chord-contact-wrench-dexterous-manipulation.md)
  - [`wiki/entities/paper-contactmimic.md`](wiki/entities/paper-contactmimic.md)
  - [`wiki/entities/paper-coordex-dexterous-humanoid-loco-manipulation.md`](wiki/entities/paper-coordex-dexterous-humanoid-loco-manipulation.md)
  - [`wiki/entities/paper-dit4dit-video-action-model.md`](wiki/entities/paper-dit4dit-video-action-model.md)
  - [`wiki/entities/paper-egohtr.md`](wiki/entities/paper-egohtr.md)
  - [`wiki/entities/paper-egosteer.md`](wiki/entities/paper-egosteer.md)
  - [`wiki/entities/paper-harness-vla.md`](wiki/entities/paper-harness-vla.md)
  - [`wiki/entities/paper-humanoidmimicgen.md`](wiki/entities/paper-humanoidmimicgen.md)
  - [`wiki/entities/paper-loco-manip-07-wt-umi.md`](wiki/entities/paper-loco-manip-07-wt-umi.md)
  - [`wiki/entities/paper-mujica-wheel-legged-multi-skill.md`](wiki/entities/paper-mujica-wheel-legged-multi-skill.md)
  - [`wiki/entities/paper-navwam-goal-conditioned-visual-navigation-wam.md`](wiki/entities/paper-navwam-goal-conditioned-visual-navigation-wam.md)
  - [`wiki/entities/paper-notebook-ame-2-agile-and-generalized-legged-locomotion-vi.md`](wiki/entities/paper-notebook-ame-2-agile-and-generalized-legged-locomotion-vi.md)
  - [`wiki/entities/paper-notebook-kungfubot-2.md`](wiki/entities/paper-notebook-kungfubot-2.md)
  - [`wiki/entities/paper-now-you-see-that-humanoid-vision-locomotion.md`](wiki/entities/paper-now-you-see-that-humanoid-vision-locomotion.md)
  - [`wiki/entities/paper-phygile.md`](wiki/entities/paper-phygile.md)
  - [`wiki/entities/paper-slowrl-safe-lora-locomotion-sim2real.md`](wiki/entities/paper-slowrl-safe-lora-locomotion-sim2real.md)
  - [`wiki/entities/paper-abot-m05-mobile-manipulation-wam.md`](wiki/entities/paper-abot-m05-mobile-manipulation-wam.md)
  - [`wiki/entities/paper-ame-attention-based-map-encoding.md`](wiki/entities/paper-ame-attention-based-map-encoding.md)
  - [`wiki/entities/paper-amp-survey-16-clot.md`](wiki/entities/paper-amp-survey-16-clot.md)
  - [`wiki/entities/paper-bam-extended-friction-servo-actuators.md`](wiki/entities/paper-bam-extended-friction-servo-actuators.md)
  - [`wiki/entities/paper-being-m07-humanoid-latent-wam.md`](wiki/entities/paper-being-m07-humanoid-latent-wam.md)
  - [`wiki/entities/paper-deep-whole-body-parkour.md`](wiki/entities/paper-deep-whole-body-parkour.md)

## [2026-07-24] structural | Top-50 论文枢纽补齐「结论」— hub-rankings paper 榜 paper-* 全覆盖 + insert_paper_conclusion.py

- **范围：** [`exports/hub-rankings.json`](exports/hub-rankings.json) `paper` 榜 Top-100 中全部 `wiki/entities/paper-*.md`（方法页 / 非 paper-* 实体不在「结论」政策内）
- **工具：** [`scripts/insert_paper_conclusion.py`](scripts/insert_paper_conclusion.py)（评测节后、对比/局限前插入）
- **格式：** 1 句总判 + 3–7 条可操作要点（对齐 [`schema/page-types.md`](schema/page-types.md)）
- **补齐页面（含初批度排序遗漏、按官方 hub 榜补入的 5 页）：**
  - [`wiki/entities/paper-behavior-foundation-model-humanoid.md`](wiki/entities/paper-behavior-foundation-model-humanoid.md)
  - [`wiki/entities/paper-hrl-stack-03-omniretarget.md`](wiki/entities/paper-hrl-stack-03-omniretarget.md)
  - [`wiki/entities/paper-unified-walk-run-recovery-sdamp.md`](wiki/entities/paper-unified-walk-run-recovery-sdamp.md)
  - [`wiki/entities/paper-motionwam-humanoid-loco-manipulation-wam.md`](wiki/entities/paper-motionwam-humanoid-loco-manipulation-wam.md)
  - [`wiki/entities/paper-twist2.md`](wiki/entities/paper-twist2.md)
  - [`wiki/entities/paper-legs-embodied-gaussian-splatting-vla.md`](wiki/entities/paper-legs-embodied-gaussian-splatting-vla.md)
  - [`wiki/entities/paper-masked-visual-actions.md`](wiki/entities/paper-masked-visual-actions.md)
  - [`wiki/entities/paper-viral-humanoid-visual-sim2real.md`](wiki/entities/paper-viral-humanoid-visual-sim2real.md)
  - [`wiki/entities/paper-bifrost-umi.md`](wiki/entities/paper-bifrost-umi.md)
  - [`wiki/entities/paper-gigaworld-1-policy-evaluation.md`](wiki/entities/paper-gigaworld-1-policy-evaluation.md)
  - [`wiki/entities/paper-heracles-humanoid-diffusion.md`](wiki/entities/paper-heracles-humanoid-diffusion.md)
  - [`wiki/entities/paper-gmt.md`](wiki/entities/paper-gmt.md)
  - [`wiki/entities/paper-amp-survey-08-more.md`](wiki/entities/paper-amp-survey-08-more.md)
  - [`wiki/entities/paper-any2any-cross-embodiment-wbt.md`](wiki/entities/paper-any2any-cross-embodiment-wbt.md)
  - [`wiki/entities/paper-loco-manip-04-oasis.md`](wiki/entities/paper-loco-manip-04-oasis.md)
  - [`wiki/entities/paper-driftworld.md`](wiki/entities/paper-driftworld.md)
  - [`wiki/entities/paper-simfoundry-real2sim-scene-generation.md`](wiki/entities/paper-simfoundry-real2sim-scene-generation.md)
  - [`wiki/entities/paper-trex-tactile-reactive-dexterous-manipulation.md`](wiki/entities/paper-trex-tactile-reactive-dexterous-manipulation.md)
  - [`wiki/entities/paper-hrl-stack-34-gr00t_n1.md`](wiki/entities/paper-hrl-stack-34-gr00t_n1.md)
  - [`wiki/entities/paper-bfm-zero.md`](wiki/entities/paper-bfm-zero.md)
  - [`wiki/entities/paper-s-squared-vla.md`](wiki/entities/paper-s-squared-vla.md)
  - [`wiki/entities/paper-faststair-humanoid-stair-ascent.md`](wiki/entities/paper-faststair-humanoid-stair-ascent.md)
  - [`wiki/entities/paper-dreamsteer-vla-deployment-steering.md`](wiki/entities/paper-dreamsteer-vla-deployment-steering.md)
  - [`wiki/entities/paper-egowam-egocentric-human-wam-co-training.md`](wiki/entities/paper-egowam-egocentric-human-wam-co-training.md)
  - [`wiki/entities/paper-doorman-opening-sim2real-door.md`](wiki/entities/paper-doorman-opening-sim2real-door.md)
  - [`wiki/entities/paper-hrl-stack-22-perceptive_humanoid_parkour.md`](wiki/entities/paper-hrl-stack-22-perceptive_humanoid_parkour.md)
  - [`wiki/entities/paper-humanoid-gpt.md`](wiki/entities/paper-humanoid-gpt.md)
  - [`wiki/entities/paper-lehome-learning-to-fold.md`](wiki/entities/paper-lehome-learning-to-fold.md)
  - [`wiki/entities/paper-resmimic.md`](wiki/entities/paper-resmimic.md)
  - [`wiki/entities/paper-scenebot.md`](wiki/entities/paper-scenebot.md)
  - [`wiki/entities/paper-tacrefinenet-tactile-grasp-refinement.md`](wiki/entities/paper-tacrefinenet-tactile-grasp-refinement.md)
  - [`wiki/entities/paper-rpl-robust-humanoid-perceptive-locomotion.md`](wiki/entities/paper-rpl-robust-humanoid-perceptive-locomotion.md)
  - [`wiki/entities/paper-humanoidarena.md`](wiki/entities/paper-humanoidarena.md)
  - [`wiki/entities/paper-m4world.md`](wiki/entities/paper-m4world.md)
  - [`wiki/entities/paper-perceptive-bfm.md`](wiki/entities/paper-perceptive-bfm.md)
  - [`wiki/entities/paper-sprint-humanoid-athletic-sprints.md`](wiki/entities/paper-sprint-humanoid-athletic-sprints.md)
  - [`wiki/entities/paper-oscar.md`](wiki/entities/paper-oscar.md)
  - [`wiki/entities/paper-ctrl-world.md`](wiki/entities/paper-ctrl-world.md)
  - [`wiki/entities/paper-grail.md`](wiki/entities/paper-grail.md)
  - [`wiki/entities/paper-reactivebfm.md`](wiki/entities/paper-reactivebfm.md)
  - [`wiki/entities/paper-halomi-humanoid-loco-manipulation.md`](wiki/entities/paper-halomi-humanoid-loco-manipulation.md)
  - [`wiki/entities/paper-infinite-diffusion-terrain-diffusion.md`](wiki/entities/paper-infinite-diffusion-terrain-diffusion.md)
  - [`wiki/entities/paper-neuralactuator-neural-actuation-modeling.md`](wiki/entities/paper-neuralactuator-neural-actuation-modeling.md)
  - [`wiki/entities/paper-pilot-perceptive-loco-manipulation.md`](wiki/entities/paper-pilot-perceptive-loco-manipulation.md)
  - [`wiki/entities/paper-ladderman-humanoid-perceptive-ladder-climbing.md`](wiki/entities/paper-ladderman-humanoid-perceptive-ladder-climbing.md)
  - [`wiki/entities/paper-mpc-rl-humanoid-locomotion-manipulation.md`](wiki/entities/paper-mpc-rl-humanoid-locomotion-manipulation.md)
  - [`wiki/entities/paper-omg-omni-modal-humanoid-control.md`](wiki/entities/paper-omg-omni-modal-humanoid-control.md)
  - [`wiki/entities/paper-worldvln-aerial-vln-wam.md`](wiki/entities/paper-worldvln-aerial-vln-wam.md)
  - [`wiki/entities/paper-agentic-real2sim.md`](wiki/entities/paper-agentic-real2sim.md)
  - [`wiki/entities/paper-twist.md`](wiki/entities/paper-twist.md)
  - [`wiki/entities/paper-kungfuathlete-humanoid-martial-arts-tracking.md`](wiki/entities/paper-kungfuathlete-humanoid-martial-arts-tracking.md)
  - [`wiki/entities/paper-loco-manip-161-154-openhlm.md`](wiki/entities/paper-loco-manip-161-154-openhlm.md)
  - [`wiki/entities/paper-notebook-kungfubot-physics-based-humanoid-whole-body-cont.md`](wiki/entities/paper-notebook-kungfubot-physics-based-humanoid-whole-body-cont.md)
  - [`wiki/entities/paper-amp-survey-15-physhsi.md`](wiki/entities/paper-amp-survey-15-physhsi.md)

## [2026-07-24] ingest | sources/papers/aware_arxiv_2604_23761.md — AWARE（arXiv:2604.23761）轮足高动态反射式避障；wiki/entities/paper-aware-wheeled-legged-reflexive-evasion.md；交叉 wheel-legged / hybrid-locomotion / HRL / MUJICA / sim2real；确认未开源

- **开源状态：** **确认未开源** — 无项目页；arXiv abs/HTML/API 未列 GitHub；作者主页未列 AWARE 仓
- **新建 papers：** [`sources/papers/aware_arxiv_2604_23761.md`](sources/papers/aware_arxiv_2604_23761.md)
- **新建 wiki：** [`wiki/entities/paper-aware-wheeled-legged-reflexive-evasion.md`](wiki/entities/paper-aware-wheeled-legged-reflexive-evasion.md)（含结论；源码运行时序图不适用）
- **机构：** [`schema/institutions.json`](schema/institutions.json) 新增 `zhongguancun-academy`（北京中关村学院）、`deeprobotics`（云深处科技）；复用 `tju` / `nus`
- **交叉更新：** [`wheel-legged-quadruped`](wiki/concepts/wheel-legged-quadruped.md)、[`hybrid-locomotion`](wiki/tasks/hybrid-locomotion.md)、[`locomotion`](wiki/tasks/locomotion.md)、[`hierarchical-reinforcement-learning`](wiki/methods/hierarchical-reinforcement-learning.md)、[`curriculum-learning`](wiki/concepts/curriculum-learning.md)、[`domain-randomization`](wiki/concepts/domain-randomization.md)、[`sim2real`](wiki/concepts/sim2real.md)、[`paper-mujica-wheel-legged-multi-skill`](wiki/entities/paper-mujica-wheel-legged-multi-skill.md)、[`robot-lab`](wiki/entities/robot-lab.md)

## [2026-07-24] ingest | sources/papers/vtap_gripper_arxiv_2607_15448.md — VTAP Gripper（arXiv:2607.15448）视触觉主动掌三指夹爪；wiki/entities/paper-vtap-gripper.md；sources/sites/yuhao-zhou-vtap.md；交叉 wiki/concepts/visuo-tactile-fusion.md、tactile-sensing.md、contact-rich-manipulation.md、wiki/tasks/manipulation.md、wiki/entities/gel-slim.md、wiki/methods/in-hand-reorientation.md；确认未开源

- **开源状态：** **确认未开源** — 项目页 <https://yuhao-zhou.com/vtap/>（`yuhochau.github.io/vtap/` 301）无 VTAP 仓；页上 Code 为 ViTacFormer 模板残留；指尖传感上游 FlexiTac 另开源
- **新建 papers/sites：** [`sources/papers/vtap_gripper_arxiv_2607_15448.md`](sources/papers/vtap_gripper_arxiv_2607_15448.md)、[`sources/sites/yuhao-zhou-vtap.md`](sources/sites/yuhao-zhou-vtap.md)
- **新建 wiki：** [`wiki/entities/paper-vtap-gripper.md`](wiki/entities/paper-vtap-gripper.md)
- **交叉更新：** [`visuo-tactile-fusion`](wiki/concepts/visuo-tactile-fusion.md)、[`tactile-sensing`](wiki/concepts/tactile-sensing.md)、[`contact-rich-manipulation`](wiki/concepts/contact-rich-manipulation.md)、[`manipulation`](wiki/tasks/manipulation.md)、[`gel-slim`](wiki/entities/gel-slim.md)、[`in-hand-reorientation`](wiki/methods/in-hand-reorientation.md)、[`motion-retargeting`](wiki/concepts/motion-retargeting.md)

## [2026-07-24] ingest | sources/papers/kairos_arxiv_2606_16533.md — Kairos v3 regret-aware 叙事与开源仓更名；wiki/entities/paper-kairos-native-world-model-stack.md；sources/repos/kairos.md、sources/sites/kairos-acerobotics.md；交叉 wiki/methods/generative-world-models.md、wiki/concepts/world-action-models.md

- **开源状态：** **已开源** — 代码 [kairos-agi/kairos](https://github.com/kairos-agi/kairos)（旧 `kairos-sensenova` 301）；权重 [huggingface.co/kairos-agi](https://huggingface.co/kairos-agi) / Kairos3.1；平台 [kairos.acerobotics.com](https://kairos.acerobotics.com)
- **修订 papers：** [`sources/papers/kairos_arxiv_2606_16533.md`](sources/papers/kairos_arxiv_2606_16533.md)（v3 标题 *Regret-Aware Native World-Action Model Stack*；control-sufficient / \(\operatorname{Reg}_H\)）
- **新建 repos/sites：** [`sources/repos/kairos.md`](sources/repos/kairos.md)、[`sources/sites/kairos-acerobotics.md`](sources/sites/kairos-acerobotics.md)；历史索引 [`kairos_sensenova.md`](sources/repos/kairos_sensenova.md) 改为重定向说明
- **改写 wiki：** [`wiki/entities/paper-kairos-native-world-model-stack.md`](wiki/entities/paper-kairos-native-world-model-stack.md)（补「结论」「源码运行时序图」）
- **交叉更新：** [`generative-world-models`](wiki/methods/generative-world-models.md)、[`world-action-models`](wiki/concepts/world-action-models.md)；机构注册 `ace-robotics`
- **机构：** [`schema/institutions.json`](schema/institutions.json) 新增 `ace-robotics` → 大晓机器人（Ace Robotics）

## [2026-07-24] ingest | sources/papers/faro_arxiv_2607_18362.md — FARO（arXiv:2607.18362）嵌套可行性剪枝加速人形 loco-manipulation 接触搜索；wiki/entities/paper-faro-feasibility-aware-robot-motion-optimization.md；交叉 wiki/entities/paper-motiondisco-extreme-humanoid-loco-manipulation.md、wiki/tasks/loco-manipulation.md、wiki/methods/trajectory-optimization.md；确认未开源

- **开源状态：** **确认未开源** — 无项目页；arXiv/HTML 未列 GitHub；仅补充视频 <https://youtu.be/R6qCHoCormQ>
- **新建 papers：** [`sources/papers/faro_arxiv_2607_18362.md`](sources/papers/faro_arxiv_2607_18362.md)
- **新建 wiki：** [`wiki/entities/paper-faro-feasibility-aware-robot-motion-optimization.md`](wiki/entities/paper-faro-feasibility-aware-robot-motion-optimization.md)
- **交叉更新：** [`paper-motiondisco-extreme-humanoid-loco-manipulation`](wiki/entities/paper-motiondisco-extreme-humanoid-loco-manipulation.md)、[`loco-manipulation`](wiki/tasks/loco-manipulation.md)、[`trajectory-optimization`](wiki/methods/trajectory-optimization.md)、[`motiondisco_arxiv_2606_06139`](sources/papers/motiondisco_arxiv_2606_06139.md)

## [2026-07-24] structural | schema/page-types.md + ingest-workflow + AGENTS — 规定 paper-* 必须含「结论」；lint 信息型巡检；scaffold --paper；示例补齐 wiki/entities/paper-yahmp.md、paper-deed.md、paper-extreme-rgmt.md

- **规范：** [`schema/page-types.md`](schema/page-types.md)、[`schema/ingest-workflow.md`](schema/ingest-workflow.md)、[`AGENTS.md`](AGENTS.md)
- **工具：** `lint_wiki.py` 新增 `paper_missing_conclusions`（信息型，报告截断）；`scaffold_wiki_page.py --paper`
- **示例页：** [`wiki/entities/paper-yahmp.md`](wiki/entities/paper-yahmp.md)、[`wiki/entities/paper-deed.md`](wiki/entities/paper-deed.md)、[`wiki/entities/paper-extreme-rgmt.md`](wiki/entities/paper-extreme-rgmt.md)

## [2026-07-24] structural | wiki/entities/paper-yahmp.md — 补充「结论」小节：GMT 设计选择何为真影响跟踪 / 力矩 / 交互力

- **页面：** [`wiki/entities/paper-yahmp.md`](wiki/entities/paper-yahmp.md)（详情页 id：`entity-paper-yahmp`）
- **改动：** 在「实验与评测」之后新增 **结论** 七条，汇总消融与真机部署读法

## [2026-07-24] ingest | sources/sites/extreme-rgmt-github-io.md — Extreme-RGMT（arXiv:2607.20110）复核：项目页仍未开源；交叉 wiki/entities/paper-extreme-rgmt.md ↔ wiki/entities/paper-yahmp.md

- **开源状态：** **确认未开源** — 2026-07-24 复核项目页仍仅 arXiv PDF + BibTeX + 视频，无 GitHub / HF
- **更新：** [`sources/sites/extreme-rgmt-github-io.md`](sources/sites/extreme-rgmt-github-io.md)、[`wiki/entities/paper-extreme-rgmt.md`](wiki/entities/paper-extreme-rgmt.md)（回链 YAHMP）

## [2026-07-24] ingest | sources/papers/deed_arxiv_2607_20345.md — DEED（arXiv:2607.20345）G1-Edu + GR00T N1.6 零售补货后训练；wiki/entities/paper-deed.md；交叉 wiki/methods/vla.md、wiki/entities/paper-hrl-stack-34-gr00t_n1.md、wiki/entities/paper-steam-advantage-modeling.md；确认未开源

- **开源状态：** **确认未开源** — 全文无项目页 / GitHub
- **新建 papers：** [`sources/papers/deed_arxiv_2607_20345.md`](sources/papers/deed_arxiv_2607_20345.md)
- **新建 wiki：** [`wiki/entities/paper-deed.md`](wiki/entities/paper-deed.md)
- **机构：** 新增 `hive-robots`、`dtu`
- **交叉更新：** [`vla`](wiki/methods/vla.md)、[`paper-hrl-stack-34-gr00t_n1`](wiki/entities/paper-hrl-stack-34-gr00t_n1.md)、[`paper-steam-advantage-modeling`](wiki/entities/paper-steam-advantage-modeling.md)

## [2026-07-24] ingest | sources/papers/yahmp_arxiv_2607_19903.md + sources/repos/yahmp.md — YAHMP（arXiv:2607.19903）G1 GMT 消融与 ONNX 部署；wiki/entities/paper-yahmp.md；交叉 wiki/entities/paper-twist2.md、wiki/entities/mjlab.md、wiki/entities/paper-extreme-rgmt.md、wiki/overview/humanoid-rl-motion-control-body-system-stack.md、wiki/queries/humanoid-motion-tracking-method-selection.md

- **开源状态：** **已开源** — <https://github.com/fabio-amadio/yahmp>（Apache-2.0；含 ONNX 导出/部署）
- **新建 papers：** [`sources/papers/yahmp_arxiv_2607_19903.md`](sources/papers/yahmp_arxiv_2607_19903.md)
- **新建 repos：** [`sources/repos/yahmp.md`](sources/repos/yahmp.md)
- **新建 wiki：** [`wiki/entities/paper-yahmp.md`](wiki/entities/paper-yahmp.md)（含源码运行时序图）
- **交叉更新：** [`paper-twist2`](wiki/entities/paper-twist2.md)、[`mjlab`](wiki/entities/mjlab.md)、[`paper-extreme-rgmt`](wiki/entities/paper-extreme-rgmt.md)、[`humanoid-rl-motion-control-body-system-stack`](wiki/overview/humanoid-rl-motion-control-body-system-stack.md)、[`humanoid-motion-tracking-method-selection`](wiki/queries/humanoid-motion-tracking-method-selection.md)

## [2026-07-24] ingest | sources/blogs/wechat_shenlan_ai_ad_e2e_top10.md — 深蓝AI《端到端自动驾驶：十大前沿算法盘点》；十篇论文独立完整实体（非 stub）

- **工具：** Agent Reach v1.5.0 + wechat-article-for-ai（Camoufox）；短链直连成功 — <https://mp.weixin.qq.com/s/kb4aNFyCLWMKEVgjiX6F_g>
- **原始抓取：** [`sources/raw/wechat_shenlan_ai_ad_e2e_top10_2026-07-23/`](sources/raw/wechat_shenlan_ai_ad_e2e_top10_2026-07-23/)
- **新建 blogs：** [`sources/blogs/wechat_shenlan_ai_ad_e2e_top10.md`](sources/blogs/wechat_shenlan_ai_ad_e2e_top10.md)
- **新建 catalog：** [`sources/papers/e2e_ad_top10_catalog.md`](sources/papers/e2e_ad_top10_catalog.md) + 十篇 `sources/papers/e2e_ad_*.md`
- **父节点：** [`wiki/overview/e2e-autonomous-driving-top10-algorithms.md`](wiki/overview/e2e-autonomous-driving-top10-algorithms.md)
- **十篇完整实体（status: complete）：** [`paper-uniad`](wiki/entities/paper-uniad.md)、[`paper-vad-vectorized-scene`](wiki/entities/paper-vad-vectorized-scene.md)、[`paper-drivevlm`](wiki/entities/paper-drivevlm.md)、[`paper-emma-waymo-e2e`](wiki/entities/paper-emma-waymo-e2e.md)、[`paper-gaia1`](wiki/entities/paper-gaia1.md)、[`paper-sparsedrive`](wiki/entities/paper-sparsedrive.md)、[`paper-senna`](wiki/entities/paper-senna.md)、[`paper-momad`](wiki/entities/paper-momad.md)、[`paper-drivetransformer`](wiki/entities/paper-drivetransformer.md)、[`paper-diffusiondrive`](wiki/entities/paper-diffusiondrive.md)
- **开源核查：** UniAD / VAD / SparseDrive / Senna / MomAD / DriveTransformer / DiffusionDrive **已开源**；DriveVLM **项目页**；EMMA / GAIA-1 **未开源**；配套 `sources/repos/*` 与 `sources/sites/*`
- **机构：** 新增 `waymo` / `wayve` / `li-auto`；`sensenova` 增 alias `sensetime`
- **交叉更新：** [`autonomous-driving-core-algorithms-series`](wiki/overview/autonomous-driving-core-algorithms-series.md)、[`generative-world-models`](wiki/methods/generative-world-models.md)、[`paper-s-squared-vla`](wiki/entities/paper-s-squared-vla.md)、[`paper-ego-04-emma`](wiki/entities/paper-ego-04-emma.md)（EMMA 消歧）

## [2026-07-23] structural | V30 P1 驱动链层级专题交叉补强 — 五页双向回链「执行器驱动链选型闭环」Query

- 在 `wiki/entities/kicad.md`（①层 EDA 电路设计）、`wiki/entities/simplefoc.md`（②层 电机驱动固件 FOC）、`wiki/entities/paper-neuralactuator-neural-actuation-modeling.md`（③层 神经执行器网络）、`wiki/entities/bam-better-actuator-models.md`（③层 显式摩擦辨识）、`wiki/concepts/ethercat-protocol.md`（④层 实时总线闭环集成）五页的 `related` 与「关联页面」补入 `wiki/queries/actuator-drive-chain-selection-loop.md` 并标注本页所在驱动链层，与 Query 页 `related` 双向闭合
- 合并 origin/main（PR #1230 全量 lint 清零）后去重：main 已为这五页补入同一枢纽回链，合并保留其一，避免 `related`/「关联页面」重复条目
- 勾选 `docs/checklists/tech-stack-next-phase-checklist-v30.md` 之 P1「驱动链层级专题交叉补强」

## [2026-07-23] structural | 全量 wiki lint 修复所有信息型预警（46→0）— 修复 lint 关键词子串误配（token 前缀匹配，'eda' 不再误配 impedance/bipedal/pedagogy/bytedance，去除 13 条假阳性并补回归测试）；为 22 个执行器/EDA/FOC 真实相关页补「执行器驱动链选型闭环」枢纽双向回链；soccer-field-simulation 回链物理保真度专题；paper-wan-move/paper-wan-video 补「对比/评测」三段式；lidar-slam-lio-vio-selection 接入 lidar-odometry-fusion；新建 3 个落地页 wiki/comparisons/mobile-robot-navigation-planning-methods.md、wiki/queries/soccer-visual-field-localization-pipeline.md、wiki/queries/humanoid-voice-interaction-pipeline.md 覆盖 a-star/dwa/smooth-navigation/soccer 三段/voice 共 7 个高频 methods；同步 graph(1805节点/15099边,0孤儿)/home-stats/exports/catalog；ci-preflight 12/12、lint 0/0

## [2026-07-23] ingest | sources/papers/generative_motion_rig_siggraph_talks_2026.md — Disney Generative Motion Rig（SIGGRAPH Talks 2026）Blender generative keyframing；wiki/entities/generative-motion-rig.md；交叉 wiki/entities/blender.md、wiki/entities/robot-motion-keyframe-editors.md、wiki/concepts/character-animation-vs-robotics.md、wiki/methods/motion-retargeting-gmr.md（GMR 缩写消歧）、wiki/methods/diffusion-motion-generation.md；sources/sites/disney-generative-motion-rig.md（插件未开源）

## [2026-07-23] ingest | sources/papers/rigmo_arxiv_2601_06378.md — RigMo（arXiv:2601.06378）无标注 mesh 联合学 Gaussian bones+SE(3)；wiki/entities/rigmo.md；交叉 wiki/methods/diffusion-motion-generation.md、wiki/entities/ardy.md、wiki/entities/blender.md、wiki/concepts/character-animation-vs-robotics.md；sources/sites/rigmo-page.md、sources/repos/rigmo.md（VAE 已开源 / Motion-DiT 未发布）

## [2026-07-23] ingest | sources/papers/tacrefinenet_arxiv_2509_25746.md — TacRefineNet（arXiv:2509.25746）纯触觉目标条件抓取精修：Siamese 多指策略 + 外在灵巧 regrasp；wiki/entities/paper-tacrefinenet-tactile-grasp-refinement.md；交叉 wiki/concepts/tactile-sensing.md、wiki/overview/topic-tactile.md、wiki/overview/topic-grasp.md、wiki/methods/visual-servoing.md、wiki/methods/in-hand-reorientation.md、wiki/queries/grasp-policy-selection.md、wiki/concepts/sim2real.md、wiki/entities/paper-omnitactune-tactile-residual-adaptation.md；sources/sites/tacrefinenet-google-sites.md、sources/repos/tacrefinenet.md（空仓待发布）

## [2026-07-23] fix(ux): 详情页 Markdown 还原 A\* 等反斜杠转义

- `docs/main.js`：`renderInlineMarkdown` / `renderLinkLabel` 在强调语法**之前**将 CommonMark `\*` 转为 HTML 实体（避免 `**A\***` 被吃成 `A\`）；策展表与 A* 方法页正文均显示为 `A*`
- `wiki/entities/humanoid-system-curriculum.md`：第 4 章映射表改为直接写 `A*`
- 全库复查：约 24 个 wiki 页正文仍含合法 `A\*` / `RRT\*` / `π\*` 等转义；标题/图谱/搜索索引无泄漏；抽查 a-star、dwa、FAR、footstep、STEAM、CWI、策展等详情页 DOM 均无可见 `\`；补充 `test_wiki_prose_backslash_star_does_not_leak_after_render` 回归

## [2026-07-23] fix(export): 站点标题还原 Markdown 转义（A\* → A*）

- `extract_title`（export_minimal / search_indexing / generate_link_graph）：去掉 H1 中的 `\*` 等转义，避免详情页/图谱把反斜杠当正文显示
- `wiki/methods/a-star.md`：H1 改为不转义的 `A* 全局路径规划`（正文内链仍可按需转义）

## [2026-07-23] structural | wiki 人形系统课独立节点加厚 — 方法/概念/实体/overview 由 stub 扩为含原理·路线·工程实践·调试指标的完整页

- 针对 PR #1225 新建的 A\*/DWA/融合定位/动态障碍/足球感知定位/语音/TARE/FAR/RealSense/G1 软件栈/历史与现状/大模型赋能/自主探索等页：补公式与流程、主要技术路线表、工程 SOP/调参、失败模式与误区，正文体量对齐仓库完整方法页（约 2–3k 非空白字符）
- `make ci-preflight` 通过

## [2026-07-23] ingest | sources/courses/shenlan_humanoid_system_theory_practice.md — 深蓝学院《人形机器人系统—理论与实践》八章大纲入库；策展 hub + 缺失知识点独立节点全覆盖

- **课程源：** [`sources/courses/shenlan_humanoid_system_theory_practice.md`](sources/courses/shenlan_humanoid_system_theory_practice.md)（具身智能研究室大纲图 / 深蓝学院 course 802）
- **开源配套：** [`sources/repos/tare_planner.md`](sources/repos/tare_planner.md)、[`sources/repos/far_planner.md`](sources/repos/far_planner.md)、[`sources/sites/cmu-exploration.md`](sources/sites/cmu-exploration.md)（TARE/FAR **已开源**）
- **策展 hub：** [`wiki/entities/humanoid-system-curriculum.md`](wiki/entities/humanoid-system-curriculum.md) — 八章每节 ↔ 独立 wiki 节点映射表
- **新建 overview：** `humanoid-robot-history`、`humanoid-algorithm-research-status`、`large-model-empowered-humanoids`
- **新建 entity：** `unitree-g1-software-stack`、`tare-planner`、`far-planner`、`intel-realsense`
- **新建 method：** `a-star`、`dwa`、`lidar-odometry-fusion`、`soccer-field-line-detection`、`visual-line-matching-localization`、`visual-line-ekf-fusion`、`humanoid-voice-interaction`
- **新建 concept：** `dynamic-obstacle-filtering`、`soccer-field-simulation`、`perception-coordinate-postprocessing`
- **新建 task：** `autonomous-exploration`
- **复用既有：** `unitree-g1`、`lip-zmp`、`ppo`、`sim2real`、`navigation-slam-autonomy-stack`、`object-detection`、`humanoid-soccer`、`vision-language-navigation`、`paper-vln-10-navid` 等
- **交叉更新：** `unitree-g1`、`python-robotics`、`humanoid-soccer`、`navigation-slam-autonomy-stack`、`sources/notes/know-how.md`、`sources/README.md`

## [2026-07-23] structural | roadmap/depth-teleoperation.md — 新增遥操作（人形全身遥操作 + 手指遥操作）纵深路线，二十条纵深入口全站同步

- roadmap/depth-teleoperation.md：新建「遥操作」纵深路线（起点：Goertz 主从机械手遥操作，1954），Stage 0 全景与定位（三种输出形态：示范数据 / 部署期接管 / 竞技操控）→ Stage 1 输入接口与硬件通道（XR / leader–follower / 上肢外骨骼 / 数据手套 / 纯视觉 / UMI 手持）→ Stage 2 映射与延迟（笛卡尔解析 IK · 实时重定向 · 双边遥操作稳定）→ **Stage 3 人形全身遥操作**（稀疏三点补全 → 全身跟踪策略 GMT → WBC / BFM 流形约束；HEFT / PILOT / TWIST2 / CLOT / TeleGate / CWI）→ **Stage 4 手指与灵巧手遥操作**（手姿估计 → 手部重定向 UHAS / 接触保持 TopoRetarget·SPIDER → 触觉手套 / 灵巧手硬件谱系）→ Stage 5 下游闭环与进阶（teleop → IL · 无机器人采集 · VLA 干预 · 数据集规模 · 极端场景）
- 按起点里程碑历史序排在**首位**（1954，早于 FOC 1971）：README.md、index.md、roadmap/README.md、roadmap/motion-control.md、docs/index.html、docs/main.js 纵深路线数同步为二十条；docs/index.html 首页按钮以 data-route-extra + hidden 折叠（默认仍只展示里程碑最新 4 条）
- 其余 19 条 depth-* 路线页「其它纵深路径」各补一条指向遥操作纵深的反向链接，维持纵深互链网络
- 顺带订正 roadmap/motion-control.md 纵深表此前遗留的「十八条」计数（实为 19，Real2Sim 未同步）→ 现更新为二十条，与全站一致
- 派生文件（catalog.md / exports / docs/exports / search-index / sitemap 与统计徽章）经 make ci-preflight 同步

## [2026-07-23] structural | scripts/generate_link_graph.py + docs/main.js + export_minimal — 纵深路线更新纳入首页最新知识节点；路线页元信息显示更新时间标签

- scripts/generate_link_graph.py：log.md 路径解析扩展至 roadmap/…，latest_wiki_nodes / wiki-activity 收录纵深与主路线；git first-add 扫描同步覆盖 roadmap/
- scripts/export_minimal.py：roadmap 页无 frontmatter updated 时用 git 最近提交日回填；roadmap_pages 导出带 updated/path
- docs/main.js：最新知识节点/更新时间线链接走 roadmap.html；路线元信息用 detail-meta-date 徽章展示更新时间
- schema/ingest-workflow.md、AGENTS.md：同步最新知识节点收录规则
- tests：覆盖 roadmap 进入 latest_wiki_nodes 与 updated 回填

## [2026-07-23] structural | roadmap/depth-real2sim.md — 新增 Real2Sim（真实世界 → 可仿真资产/场景/孪生）纵深路线，十九条纵深入口全站同步

- roadmap/depth-real2sim.md：新建「Real2Sim」纵深路线（起点：3D Gaussian Splatting 规模化重建，2023），Stage 0 定位与可仿真判据（Real2Sim vs Sim2Real · 三种产物粒度）→ Stage 1 几何与外观重建（接触动力学 vs 光真实感）→ Stage 2 物性与关节化补全（碰撞 / 质量摩擦 / 关节 / 接触引导遮挡补全 / sim-ready 资产生成）→ Stage 3 场景 / episode 孪生与数字表亲（VLM-agent 编排）→ Stage 4 Real2Sim2Real 闭环（回训 / real-to-sim 评测 / 真机回放）→ Stage 5 前沿口径与选型（像素世界模型 vs 物理孪生）
- roadmap/depth-sim2real.md：Stage 5 方向 B 与「其它纵深路径」补 Real2Sim 纵深双向链接（本路线的反向补集）
- roadmap/README.md、index.md、roadmap/motion-control.md、README.md、docs/index.html、docs/main.js 与其余十七条 depth-* 路线页：纵深路线数 18 → 19，按里程碑历史序在 VLA（2023）后、WAM（2026）前插入 Real2Sim 并补双向链接；顺带补回 README.md 此前遗漏的人形群控展演路线，使根 README 路线表与全站一致（19 条）
- 派生文件（catalog.md / exports / docs/exports / search-index / sitemap 等）经 make ci-preflight 同步

## [2026-07-23] ingest | sources/repos/img2threejs.md + sources/sites/img2threejs-showcase.md — img2threejs（单图→质量门控程序化 Three.js Agent Skill）；升格 wiki/entities/img2threejs.md；交叉 wiki/entities/articraft.md、cad-skills.md、gsap-skills.md、wiki/concepts/text-to-cad.md；已开源 MIT

- **开源状态：** **已开源** — [`hoainho/img2threejs`](https://github.com/hoainho/img2threejs)（MIT，`SKILL.md` + `forge/` stdlib）+ 画廊 [`img2threejs-showcase`](https://hoainho.github.io/img2threejs-showcase/)
- **新建 repos：** [`sources/repos/img2threejs.md`](sources/repos/img2threejs.md)
- **新建 sites：** [`sources/sites/img2threejs-showcase.md`](sources/sites/img2threejs-showcase.md)
- **新建 wiki：** [`wiki/entities/img2threejs.md`](wiki/entities/img2threejs.md)
- **交叉更新：** [`wiki/entities/articraft.md`](wiki/entities/articraft.md)、[`wiki/entities/cad-skills.md`](wiki/entities/cad-skills.md)、[`wiki/entities/gsap-skills.md`](wiki/entities/gsap-skills.md)、[`wiki/concepts/text-to-cad.md`](wiki/concepts/text-to-cad.md)
- **定位提示：** reconstruction-by-code / WebGL prop，非 URDF·MJCF 仿真就绪关节资产

## [2026-07-23] ingest | sources/repos/exercises-dataset.md — Exercises Dataset（1,324 健身动作目录+多语说明+GIF）；升格 wiki/entities/exercises-dataset.md；交叉 wiki/comparisons/humanoid-reference-motion-datasets.md、wiki/entities/amass.md、wiki/concepts/motion-data-quality.md；分层开源（MIT 文本 + Gym visual 媒体）

- **开源状态：** **已开源（分层）** — 代码/JSON/说明文本 MIT；`images/` `videos/` © Gym visual（LICENSE 媒体例外 + NOTICE.md）
- **新建 repos：** [`sources/repos/exercises-dataset.md`](sources/repos/exercises-dataset.md)
- **新建 wiki：** [`wiki/entities/exercises-dataset.md`](wiki/entities/exercises-dataset.md)
- **交叉更新：** [`wiki/comparisons/humanoid-reference-motion-datasets.md`](wiki/comparisons/humanoid-reference-motion-datasets.md)、[`wiki/entities/amass.md`](wiki/entities/amass.md)、[`wiki/concepts/motion-data-quality.md`](wiki/concepts/motion-data-quality.md)
- **定位提示：** 非 MoCap / 非关节轨迹；勿当作 WBT 参考源

## [2026-07-23] ingest | sources/papers/egosteer_arxiv_2607_09701.md + sources/sites/egosteer-github-io.md + sources/repos/egosteer.md / egosmith.md / egosteer-robot-stack.md — EgoSteer（arXiv:2607.09701）EgoSmith+Robot Stack+WM-VLA 全栈；wiki/entities/paper-egosteer.md；交叉 wiki/methods/egoscale.md、wiki/entities/paper-egowam-egocentric-human-wam-co-training.md、wiki/methods/vla.md、wiki/methods/dagger.md、wiki/tasks/manipulation.md、wiki/overview/vla-open-source-repro-landscape-2025.md；部分开源（代码+权重已发，全量处理后数据待齐）

- **开源状态：** **部分开源** — [`egosteer/egosteer`](https://github.com/egosteer/egosteer) / [`egosmith`](https://github.com/egosteer/egosmith) / [`robot-stack`](https://github.com/egosteer/robot-stack)（Apache-2.0）+ HF `EgoSteer-3B-Base` / `EgoSteer-3B-RealMan`；处理后 9.6K h / 187 h 全量数据集截至 2026-07-23 仍未在 HF 公开
- **新建 papers：** [`sources/papers/egosteer_arxiv_2607_09701.md`](sources/papers/egosteer_arxiv_2607_09701.md)
- **新建 sites：** [`sources/sites/egosteer-github-io.md`](sources/sites/egosteer-github-io.md)
- **新建 repos：** [`sources/repos/egosteer.md`](sources/repos/egosteer.md)、[`sources/repos/egosmith.md`](sources/repos/egosmith.md)、[`sources/repos/egosteer-robot-stack.md`](sources/repos/egosteer-robot-stack.md)
- **新建 wiki：** [`wiki/entities/paper-egosteer.md`](wiki/entities/paper-egosteer.md)
- **机构：** 沿用既有 `pku` / `upenn`；新增 `psibot`
- **交叉更新：** [`wiki/methods/egoscale.md`](wiki/methods/egoscale.md)、[`wiki/entities/paper-egowam-egocentric-human-wam-co-training.md`](wiki/entities/paper-egowam-egocentric-human-wam-co-training.md)、[`wiki/methods/vla.md`](wiki/methods/vla.md)、[`wiki/methods/dagger.md`](wiki/methods/dagger.md)、[`wiki/tasks/manipulation.md`](wiki/tasks/manipulation.md)、[`wiki/overview/vla-open-source-repro-landscape-2025.md`](wiki/overview/vla-open-source-repro-landscape-2025.md)

## [2026-07-23] ingest | sources/papers/s_squared_vla_arxiv_2607_13926.md — S²-VLA（arXiv:2607.13926）语义∥空间双流驾驶 VLA；升格 wiki/entities/paper-s-squared-vla.md；交叉 wiki/methods/vla.md、wiki/entities/paper-x-foresight.md、wiki/overview/autonomous-driving-core-algorithms-series.md、wiki/entities/qwen-robot-nav.md、wiki/tasks/vision-language-navigation.md、wiki/overview/topic-vla.md；注册机构 whut；截至 2026-07-23 未开源

- **开源状态：** **确认未开源** — arXiv 页无项目页 / GitHub / HF / 权重（截至 2026-07-23）
- **新建 papers：** [`sources/papers/s_squared_vla_arxiv_2607_13926.md`](sources/papers/s_squared_vla_arxiv_2607_13926.md)
- **新建 wiki：** [`wiki/entities/paper-s-squared-vla.md`](wiki/entities/paper-s-squared-vla.md)
- **机构：** 新增 `whut`（武汉理工大学）
- **交叉更新：** [`wiki/methods/vla.md`](wiki/methods/vla.md)、[`wiki/entities/paper-x-foresight.md`](wiki/entities/paper-x-foresight.md)、[`wiki/overview/autonomous-driving-core-algorithms-series.md`](wiki/overview/autonomous-driving-core-algorithms-series.md)、[`wiki/entities/qwen-robot-nav.md`](wiki/entities/qwen-robot-nav.md)、[`wiki/tasks/vision-language-navigation.md`](wiki/tasks/vision-language-navigation.md)、[`wiki/overview/topic-vla.md`](wiki/overview/topic-vla.md)

## [2026-07-23] ingest | sources/papers/extreme_rgmt_arxiv_2607_20110.md + sources/sites/extreme-rgmt-github-io.md — Extreme-RGMT（arXiv:2607.20110）两阶段 PACE+STAR 高动态 generalist 跟踪；wiki/entities/paper-extreme-rgmt.md；交叉 wiki/entities/paper-hrl-stack-14-robust_and_generalized_humanoid_moti.md、wiki/methods/any2track.md、wiki/methods/sonic-motion-tracking.md、wiki/methods/beyondmimic.md、wiki/entities/paper-hrl-stack-16-omnixtreme.md、wiki/overview/humanoid-rl-motion-control-body-system-stack.md；截至 2026-07-23 项目页未开源

- **开源状态：** **确认未开源** — 项目页 <https://zeonsunlightyu.github.io/Extreme-RGMT.github.io/> 仅 arXiv PDF + BibTeX + 演示视频，无 GitHub / HF（截至 2026-07-23）
- **新建 papers：** [`sources/papers/extreme_rgmt_arxiv_2607_20110.md`](sources/papers/extreme_rgmt_arxiv_2607_20110.md)
- **新建 sites：** [`sources/sites/extreme-rgmt-github-io.md`](sources/sites/extreme-rgmt-github-io.md)
- **新建 wiki：** [`wiki/entities/paper-extreme-rgmt.md`](wiki/entities/paper-extreme-rgmt.md)
- **机构：** 沿用既有 `bit` / `openloong` / `sdu`（论文署名为 BIT + 人形机器人（上海）有限公司 + SDU）
- **交叉更新：** [`wiki/entities/paper-hrl-stack-14-robust_and_generalized_humanoid_moti.md`](wiki/entities/paper-hrl-stack-14-robust_and_generalized_humanoid_moti.md)、[`wiki/methods/any2track.md`](wiki/methods/any2track.md)、[`wiki/methods/sonic-motion-tracking.md`](wiki/methods/sonic-motion-tracking.md)、[`wiki/methods/beyondmimic.md`](wiki/methods/beyondmimic.md)、[`wiki/entities/paper-hrl-stack-16-omnixtreme.md`](wiki/entities/paper-hrl-stack-16-omnixtreme.md)、[`wiki/overview/humanoid-rl-motion-control-body-system-stack.md`](wiki/overview/humanoid-rl-motion-control-body-system-stack.md)

## [2026-07-23] fix | docs/index.html — 恢复首页「更多路线」缺失的人形群控展演按钮（18 条纵深与 main.js 对齐）

- docs/index.html：合并冲突后首页「更多路线」漏掉 [人形群控展演](roadmap/depth-humanoid-swarm-performance.md) 按钮，展开文案仍写「17 条」；按历史序插回动作重定向与 Sim2Real 之间，文案与注释同步为 18 条（与 `docs/main.js` 折叠切换文案一致）

## [2026-07-23] ingest | sources/sites/sota-evomind-tech.md + sources/repos/evo-sota-io.md — VLA SOTA Leaderboard（EvoMind/MINT-SJTU）；升格 wiki/entities/vla-sota-leaderboard.md；交叉 wiki/methods/vla.md、wiki/overview/vla-open-source-repro-landscape-2025.md、wiki/queries/embodied-eval-benchmark-selection-loop.md、wiki/tasks/manipulation.md、wiki/overview/topic-vla.md；榜站 MIT 开源

- **开源状态：** **已开源（榜站）** — [`MINT-SJTU/Evo-SOTA.io`](https://github.com/MINT-SJTU/Evo-SOTA.io)（MIT）+ 线上 <https://sota.evomind-tech.com/>（截至 2026-07-23）
- **新建 sites：** [`sources/sites/sota-evomind-tech.md`](sources/sites/sota-evomind-tech.md)
- **新建 repos：** [`sources/repos/evo-sota-io.md`](sources/repos/evo-sota-io.md)
- **新建 wiki：** [`wiki/entities/vla-sota-leaderboard.md`](wiki/entities/vla-sota-leaderboard.md)
- **机构：** 沿用既有 `sjtu`；新增 `evomind`
- **交叉更新：** [`wiki/methods/vla.md`](wiki/methods/vla.md)、[`wiki/overview/vla-open-source-repro-landscape-2025.md`](wiki/overview/vla-open-source-repro-landscape-2025.md)、[`wiki/queries/embodied-eval-benchmark-selection-loop.md`](wiki/queries/embodied-eval-benchmark-selection-loop.md)、[`wiki/tasks/manipulation.md`](wiki/tasks/manipulation.md)、[`wiki/overview/topic-vla.md`](wiki/overview/topic-vla.md)

## [2026-07-23] ingest | sources/papers/fabrivla_arxiv_2607_08575.md + sources/repos/fabrivla.md — FabriVLA（arXiv:2607.08575）轻量 VLA；升格 wiki/entities/paper-fabrivla.md；交叉 wiki/entities/paper-evo1-lightweight-vla.md、wiki/entities/vla-sota-leaderboard.md、wiki/methods/vla.md、wiki/overview/vla-open-source-repro-landscape-2025.md、wiki/tasks/manipulation.md；已开源（Apache-2.0）

- **开源状态：** **已开源** — [`Youi-FabriX/FabriVLA`](https://github.com/Youi-FabriX/FabriVLA)（Apache-2.0）+ HF [`Youi-FabriX/FabriVLA`](https://huggingface.co/Youi-FabriX/FabriVLA) `checkpoint_step_93000.pt`（截至 2026-07-23）；无独立项目页
- **新建 papers：** [`sources/papers/fabrivla_arxiv_2607_08575.md`](sources/papers/fabrivla_arxiv_2607_08575.md)
- **新建 repos：** [`sources/repos/fabrivla.md`](sources/repos/fabrivla.md)
- **新建 wiki：** [`wiki/entities/paper-fabrivla.md`](wiki/entities/paper-fabrivla.md)
- **机构：** 新增 `umac` / `youibot` / `mese`；榜站侧另见 `evomind`
- **交叉更新：** [`wiki/entities/paper-evo1-lightweight-vla.md`](wiki/entities/paper-evo1-lightweight-vla.md)、[`wiki/entities/vla-sota-leaderboard.md`](wiki/entities/vla-sota-leaderboard.md)、[`wiki/methods/vla.md`](wiki/methods/vla.md)、[`wiki/overview/vla-open-source-repro-landscape-2025.md`](wiki/overview/vla-open-source-repro-landscape-2025.md)、[`wiki/tasks/manipulation.md`](wiki/tasks/manipulation.md)、[`wiki/queries/embodied-eval-benchmark-selection-loop.md`](wiki/queries/embodied-eval-benchmark-selection-loop.md)

## [2026-07-23] structural | roadmap/depth-humanoid-swarm-performance.md — 新增人形群控展演（HSP）纵深路线（群舞同步 → 编队走位 → 群体特技），十八条纵深入口全站同步

- roadmap/depth-humanoid-swarm-performance.md：新建「人形群控展演」纵深路线（简写 HSP，Humanoid Swarm Performance；起点：央视春晚 540 台 Alpha 1S 群舞，2016），Stage 0 展演任务谱系与预编排 vs 在线协调两大范式 → Stage 1 单机动作基座（重定向 / 跟踪 RL / 音乐驱动生成）→ Stage 2 群舞同步（时间轴编排 / 时钟同步 / 失步容错）→ Stage 3 编队走位（队形变换 / 互避碰 / 场地定位）→ Stage 4 群体特技（高动态技能 × 多机协同 × 安全边界）→ Stage 5 进阶方向（在线互动 / 学习式协调 / 异构群 / 作业群控）
- wiki/concepts/humanoid-multi-robot-coordination.md：关联页面补纵深路线入口（「预录时间轴 vs 在线战术」分界即其 Stage 0 入口）
- roadmap/README.md、README.md、index.md、roadmap/motion-control.md、docs/index.html、docs/main.js 与其余十七条 depth-* 路线页：纵深路线数 17 → 18，按里程碑历史序插入（动作重定向 1998 后、Sim2Real 2017 前）并补双向链接

## [2026-07-23] ingest | sources/papers/wan_video_arxiv_2503_20314.md + sources/sites/wan-video.md + sources/repos/wan2.1.md — Wan（arXiv:2503.20314）开源视频基础模型；升格 wiki/entities/paper-wan-video.md；交叉 wiki/entities/paper-wan-move.md、wiki/entities/paper-masked-visual-actions.md、wiki/entities/paper-ctrl-world.md、wiki/methods/generative-world-models.md、wiki/concepts/video-as-simulation.md、wiki/overview/robot-world-models-training-loop-taxonomy.md；已开源（Wan2.1/2.2 Apache-2.0）。注：masked-visual-actions.github.io 已于 2026-07-22 ingest，本次补对照实体页

- **开源状态：** **已开源** — [`Wan-Video/Wan2.1`](https://github.com/Wan-Video/Wan2.1) / [`Wan2.2`](https://github.com/Wan-Video/Wan2.2)（Apache-2.0）+ HF/ModelScope Wan-AI（截至 2026-07-23）
- **新建 papers：** [`sources/papers/wan_video_arxiv_2503_20314.md`](sources/papers/wan_video_arxiv_2503_20314.md)
- **新建 sites：** [`sources/sites/wan-video.md`](sources/sites/wan-video.md)
- **新建 repos：** [`sources/repos/wan2.1.md`](sources/repos/wan2.1.md)
- **新建 wiki：** [`wiki/entities/paper-wan-video.md`](wiki/entities/paper-wan-video.md)
- **机构：** 沿用既有 `alibaba`
- **交叉更新：** [`wiki/methods/generative-world-models.md`](wiki/methods/generative-world-models.md)、[`wiki/concepts/video-as-simulation.md`](wiki/concepts/video-as-simulation.md)、[`wiki/overview/robot-world-models-training-loop-taxonomy.md`](wiki/overview/robot-world-models-training-loop-taxonomy.md)、[`wiki/entities/paper-masked-visual-actions.md`](wiki/entities/paper-masked-visual-actions.md)

## [2026-07-23] ingest | sources/papers/wan_move_arxiv_2512_08765.md + sources/sites/wan-move-github-io.md + sources/repos/wan-move.md — Wan-Move（arXiv:2512.08765，NeurIPS 2025）latent 轨迹运动控制；升格 wiki/entities/paper-wan-move.md；交叉 wiki/entities/paper-wan-video.md、wiki/entities/paper-masked-visual-actions.md、wiki/methods/generative-world-models.md、wiki/concepts/video-as-simulation.md、wiki/overview/world-models-route-03-virtual-sandbox.md；已开源（Apache-2.0）

- **开源状态：** **已开源** — [`ali-vilab/Wan-Move`](https://github.com/ali-vilab/Wan-Move)（Apache-2.0）+ HF `Ruihang/Wan-Move-14B-480P` + MoveBench（截至 2026-07-23）
- **新建 papers：** [`sources/papers/wan_move_arxiv_2512_08765.md`](sources/papers/wan_move_arxiv_2512_08765.md)
- **新建 sites：** [`sources/sites/wan-move-github-io.md`](sources/sites/wan-move-github-io.md)
- **新建 repos：** [`sources/repos/wan-move.md`](sources/repos/wan-move.md)
- **新建 wiki：** [`wiki/entities/paper-wan-move.md`](wiki/entities/paper-wan-move.md)
- **机构：** 沿用既有 `alibaba` / `tsinghua` / `hku` / `cuhk`
- **交叉更新：** [`wiki/entities/paper-wan-video.md`](wiki/entities/paper-wan-video.md)、[`wiki/entities/paper-masked-visual-actions.md`](wiki/entities/paper-masked-visual-actions.md)、[`wiki/methods/generative-world-models.md`](wiki/methods/generative-world-models.md)、[`wiki/concepts/video-as-simulation.md`](wiki/concepts/video-as-simulation.md)

## [2026-07-23] ingest | sources/papers/ctrl_world_arxiv_2510_10125.md + sources/sites/ctrl-world-github-io.md + sources/repos/ctrl-world.md — Ctrl-World（arXiv:2510.10125，ICLR 2026）可控多视角 WM；升格 wiki/entities/paper-ctrl-world.md；交叉 wiki/methods/generative-world-models.md、wiki/concepts/video-as-simulation.md、wiki/overview/world-models-route-03-virtual-sandbox.md、wiki/overview/robot-world-models-training-loop-taxonomy.md、wiki/entities/paper-masked-visual-actions.md、wiki/tasks/manipulation.md、wiki/queries/embodied-eval-benchmark-selection-loop.md；已开源（MIT）

- **开源状态：** **已开源** — [`Robert-gyj/Ctrl-World`](https://github.com/Robert-gyj/Ctrl-World)（MIT）+ HF `yjguo/Ctrl-World`（截至 2026-07-23）
- **新建 papers：** [`sources/papers/ctrl_world_arxiv_2510_10125.md`](sources/papers/ctrl_world_arxiv_2510_10125.md)
- **新建 sites：** [`sources/sites/ctrl-world-github-io.md`](sources/sites/ctrl-world-github-io.md)
- **新建 repos：** [`sources/repos/ctrl-world.md`](sources/repos/ctrl-world.md)
- **新建 wiki：** [`wiki/entities/paper-ctrl-world.md`](wiki/entities/paper-ctrl-world.md)
- **机构：** 沿用既有 `stanford` / `tsinghua`
- **交叉更新：** [`wiki/methods/generative-world-models.md`](wiki/methods/generative-world-models.md)、[`wiki/concepts/video-as-simulation.md`](wiki/concepts/video-as-simulation.md)、[`wiki/overview/world-models-route-03-virtual-sandbox.md`](wiki/overview/world-models-route-03-virtual-sandbox.md)、[`wiki/overview/robot-world-models-training-loop-taxonomy.md`](wiki/overview/robot-world-models-training-loop-taxonomy.md)、[`wiki/entities/paper-masked-visual-actions.md`](wiki/entities/paper-masked-visual-actions.md)、[`wiki/tasks/manipulation.md`](wiki/tasks/manipulation.md)、[`wiki/queries/embodied-eval-benchmark-selection-loop.md`](wiki/queries/embodied-eval-benchmark-selection-loop.md)

## [2026-07-23] ingest | sources/papers/robointer_1_5_arxiv_2607_18709.md + sources/sites/lihaohn-robointer-github-io.md + sources/repos/robointer.md — RoboInter1.5（arXiv:2607.18709）中间表示套件+World；升格 wiki/entities/paper-robointer-1-5.md；交叉 wiki/methods/vla.md、wiki/methods/generative-world-models.md、wiki/concepts/video-as-simulation.md、wiki/tasks/manipulation.md、wiki/overview/vla-open-source-repro-landscape-2025.md、wiki/entities/paper-internvla-a15-unified-vla.md、wiki/entities/paper-masked-visual-actions.md；部分开源（Data/VLM 已发，VLA/World 待齐）

- **开源状态：** **部分开源** — [`InternRobotics/RoboInter`](https://github.com/InternRobotics/RoboInter)（MIT）+ HF Data/VQA/VLM；`RoboInterVLA` 与 **RoboInter-World** 代码/权重待齐（截至 2026-07-23）
- **新建 papers：** [`sources/papers/robointer_1_5_arxiv_2607_18709.md`](sources/papers/robointer_1_5_arxiv_2607_18709.md)
- **新建 sites：** [`sources/sites/lihaohn-robointer-github-io.md`](sources/sites/lihaohn-robointer-github-io.md)
- **新建 repos：** [`sources/repos/robointer.md`](sources/repos/robointer.md)
- **新建 wiki：** [`wiki/entities/paper-robointer-1-5.md`](wiki/entities/paper-robointer-1-5.md)
- **机构：** 沿用既有 `ustc` / `buaa` / `shanghai-ai-lab`
- **交叉更新：** [`wiki/methods/vla.md`](wiki/methods/vla.md)、[`wiki/methods/generative-world-models.md`](wiki/methods/generative-world-models.md)、[`wiki/concepts/video-as-simulation.md`](wiki/concepts/video-as-simulation.md)、[`wiki/tasks/manipulation.md`](wiki/tasks/manipulation.md)、[`wiki/overview/vla-open-source-repro-landscape-2025.md`](wiki/overview/vla-open-source-repro-landscape-2025.md)、[`wiki/entities/paper-internvla-a15-unified-vla.md`](wiki/entities/paper-internvla-a15-unified-vla.md)

## [2026-07-23] ingest | sources/papers/agentic_real2sim_arxiv_2607_19190.md + sources/sites/agentic-real2sim-github-io.md — Agentic Real2Sim（arXiv:2607.19190）VLM agent 编排 DROID→MuJoCo episode twin；升格 wiki/entities/paper-agentic-real2sim.md；交叉 wiki/concepts/sim2real.md、wiki/methods/crisp-real2sim.md、wiki/entities/paper-simfoundry-real2sim-scene-generation.md、wiki/entities/articraft.md、wiki/entities/paper-bfm-zero.md、wiki/tasks/manipulation.md；代码 coming soon；机构新增 columbia/jhu/fau/style3d

- **开源状态：** **宣称将开源 / 待发布** — 项目页 **Code (coming soon)**；无可运行官方仓（截至 2026-07-23）
- **新建 papers：** [`sources/papers/agentic_real2sim_arxiv_2607_19190.md`](sources/papers/agentic_real2sim_arxiv_2607_19190.md)
- **新建 sites：** [`sources/sites/agentic-real2sim-github-io.md`](sources/sites/agentic-real2sim-github-io.md)
- **新建 wiki：** [`wiki/entities/paper-agentic-real2sim.md`](wiki/entities/paper-agentic-real2sim.md)
- **机构：** `schema/institutions.json` 新增 `columbia` / `jhu` / `fau` / `style3d`；沿用既有 `ubc` / `ucla` / `nus`
- **交叉更新：** [`wiki/concepts/sim2real.md`](wiki/concepts/sim2real.md)、[`wiki/methods/crisp-real2sim.md`](wiki/methods/crisp-real2sim.md)、[`wiki/entities/paper-simfoundry-real2sim-scene-generation.md`](wiki/entities/paper-simfoundry-real2sim-scene-generation.md)、[`wiki/entities/articraft.md`](wiki/entities/articraft.md)、[`wiki/entities/paper-bfm-zero.md`](wiki/entities/paper-bfm-zero.md)、[`wiki/tasks/manipulation.md`](wiki/tasks/manipulation.md)

## [2026-07-22] lint | scripts/lint_wiki.py — V30 P0 执行器驱动链页交叉链路巡检 V1（INFO 级，不阻塞 CI）

- **新增检查：** `_check_actuator_drive_chain_crosslink`——对 frontmatter `tags` 含 `actuator` / `eda` / `foc`（子串匹配派生标签）的 `wiki/entities/` `wiki/comparisons/` `wiki/concepts/` 页，检查正文是否回链到「执行器驱动链选型闭环」专题枢纽（`actuator-drive-chain-selection-loop` / `topic-actuator-drive-chain`），缺失记为 INFO 级 `actuator_drive_chain_crosslink` 提示，枢纽页自身豁免
- **登记点：** 新增 `ACTUATOR_DRIVE_CHAIN_HUBS` / `ACTUATOR_DRIVE_CHAIN_TAG_KEYWORDS` 常量，登记进 `INFO_ONLY_KEYS`、`_empty_results`、`lint()` 检查序列与 `format_report` 报告段
- **测试：** 新建 [`tests/test_lint_wiki_actuator_drive_chain_crosslink.py`](tests/test_lint_wiki_actuator_drive_chain_crosslink.py) 10 条用例（列表式/内联式 tag、有/无回链、双枢纽同现、comparison/concept 覆盖、枢纽豁免、INFO 不计失败），全过
- **基线快照：** `python3 scripts/lint_wiki.py --report` 更新 [`exports/lint-report.md`](exports/lint-report.md)，新 INFO 段列出 35 页待补驱动链枢纽回链；`make lint` 0 errors，ruff/mypy 通过
- **执行清单：** [`docs/checklists/tech-stack-next-phase-checklist-v30.md`](docs/checklists/tech-stack-next-phase-checklist-v30.md) P0「驱动链页交叉链路巡检 V1」勾选完成

## [2026-07-22] lint | 清零全量知识库 lint 信息型预警（27→0）：为 25 篇 paper-* 实体补齐「方法/评测/对比」三段式 ## 区块（源自各页正文与 sources/ 归档，对比表沿用 htd-refine 房型、定性无杜撰数字）；为 paper-wolf-vla.md 补「重定向就绪度」速查维度；新建 wiki/concepts/lora.md + sources/notes/lora.md（LoRA 低秩适配概念页），并把 mimic-video/fada/any2any/m4world/wam-ttt/rldx-1 六处 **LoRA** 引用改为回链概念页消除孤儿；同步 catalog/graph-stats/home-stats/README/docs/lint-report 派生索引

## [2026-07-22] ingest | sources/papers/da_nav_arxiv_2607_11638.md — DA-Nav（arXiv:2607.11638）方向感知城市尺度 VLN；升格 wiki/entities/paper-da-nav.md；交叉 wiki/tasks/vision-language-navigation.md、wiki/concepts/sim2real.md、wiki/methods/vla.md、wiki/overview/vln-open-source-repro-paradigms.md、wiki/entities/paper-notebook-navila-legged-robot-vision-language-action-model.md；未开源

- **开源状态：** **未开源** — arXiv abs/HTML/PDF 无项目页或 GitHub；公开检索无官方同名可运行仓（截至 2026-07-22）
- **新建 papers：** [`sources/papers/da_nav_arxiv_2607_11638.md`](sources/papers/da_nav_arxiv_2607_11638.md)
- **新建 wiki：** [`wiki/entities/paper-da-nav.md`](wiki/entities/paper-da-nav.md)
- **机构：** 沿用既有 `shanghaitech` / `ict` / `casia` / `ucas` / `leju`
- **交叉更新：** [`wiki/tasks/vision-language-navigation.md`](wiki/tasks/vision-language-navigation.md)、[`wiki/concepts/sim2real.md`](wiki/concepts/sim2real.md)、[`wiki/methods/vla.md`](wiki/methods/vla.md)、[`wiki/overview/vln-open-source-repro-paradigms.md`](wiki/overview/vln-open-source-repro-paradigms.md)、[`wiki/entities/paper-notebook-navila-legged-robot-vision-language-action-model.md`](wiki/entities/paper-notebook-navila-legged-robot-vision-language-action-model.md)、[`wiki/methods/behavior-cloning.md`](wiki/methods/behavior-cloning.md)

## [2026-07-22] ingest | sources/papers/lehome_learning_to_fold_arxiv_2606_27163.md + sources/sites/ilialarchenko-lehome2026.md + sources/repos/lehome_solution.md + sources/sites/huggingface-lehome-{sim,real}.md — Learning to Fold（arXiv:2606.27163）LeHome 2026 全链路开源；升格 wiki/entities/paper-lehome-learning-to-fold.md；交叉 vla/awr/dagger/sim2real/lerobot/manipulation/vla-open-source-repro-landscape/steam/sunday/nvidia-so101；机构新增 ilia-larchenko；仿真1st/真机2nd权重已发

- **开源状态：** **已开源（完整工程）** — [`IliaLarchenko/lehome_solution`](https://github.com/IliaLarchenko/lehome_solution)（Apache-2.0）+ HF [`lehome_sim`](https://huggingface.co/IliaLarchenko/lehome_sim)（仿真 1st）/ [`lehome_real`](https://huggingface.co/IliaLarchenko/lehome_real)（真机 2nd）；博客含成功/失败推理视频（截至 2026-07-22）
- **新建 papers：** [`sources/papers/lehome_learning_to_fold_arxiv_2606_27163.md`](sources/papers/lehome_learning_to_fold_arxiv_2606_27163.md)
- **新建 sites：** [`sources/sites/ilialarchenko-lehome2026.md`](sources/sites/ilialarchenko-lehome2026.md)、[`sources/sites/huggingface-lehome-sim.md`](sources/sites/huggingface-lehome-sim.md)、[`sources/sites/huggingface-lehome-real.md`](sources/sites/huggingface-lehome-real.md)
- **新建 repos：** [`sources/repos/lehome_solution.md`](sources/repos/lehome_solution.md)
- **新建 wiki：** [`wiki/entities/paper-lehome-learning-to-fold.md`](wiki/entities/paper-lehome-learning-to-fold.md)
- **机构：** `schema/institutions.json` 新增 `ilia-larchenko`；沿用既有 `huggingface` / `physical-intelligence` / `nvidia`
- **交叉更新：** [`wiki/methods/vla.md`](wiki/methods/vla.md)、[`wiki/methods/awr.md`](wiki/methods/awr.md)、[`wiki/methods/dagger.md`](wiki/methods/dagger.md)、[`wiki/concepts/sim2real.md`](wiki/concepts/sim2real.md)、[`wiki/entities/lerobot.md`](wiki/entities/lerobot.md)、[`wiki/tasks/manipulation.md`](wiki/tasks/manipulation.md)、[`wiki/overview/vla-open-source-repro-landscape-2025.md`](wiki/overview/vla-open-source-repro-landscape-2025.md)、[`wiki/entities/paper-steam-advantage-modeling.md`](wiki/entities/paper-steam-advantage-modeling.md)、[`wiki/entities/sunday-robotics-act2.md`](wiki/entities/sunday-robotics-act2.md)、[`wiki/entities/nvidia-so101-sim2real-lab-workflow.md`](wiki/entities/nvidia-so101-sim2real-lab-workflow.md)

## [2026-07-22] ingest | sources/papers/xrobotoolkit_arxiv_2508_00097.md + sources/sites/xr-robotics-github-io.md + sources/repos/xrobotoolkit.md — XRoboToolkit（arXiv:2508.00097）OpenXR 跨平台 XR 遥操作套件；升格 wiki/entities/paper-xrobotoolkit.md；交叉 wiki/tasks/teleoperation.md、wiki/entities/paper-twist2.md、wiki/entities/isaac-teleop.md、wiki/entities/paper-loco-manip-161-131-open-television.md；SII 2026 Best Paper，全栈已开源

- **开源状态：** **已开源** — 项目页 GitHub → [`XR-Robotics`](https://github.com/XR-Robotics)（PC-Service / Unity Client / Teleop-Sample-Python 等）；π₀ 折毯示范集未单独发布（截至 2026-07-22）
- **新建 papers：** [`sources/papers/xrobotoolkit_arxiv_2508_00097.md`](sources/papers/xrobotoolkit_arxiv_2508_00097.md)
- **新建 sites：** [`sources/sites/xr-robotics-github-io.md`](sources/sites/xr-robotics-github-io.md)
- **新建 repos：** [`sources/repos/xrobotoolkit.md`](sources/repos/xrobotoolkit.md)
- **新建 wiki：** [`wiki/entities/paper-xrobotoolkit.md`](wiki/entities/paper-xrobotoolkit.md)
- **机构：** `schema/institutions.json` 新增 `george-mason`；沿用既有 `bytedance` / `georgia-tech`
- **交叉更新：** [`wiki/tasks/teleoperation.md`](wiki/tasks/teleoperation.md)、[`wiki/entities/paper-twist2.md`](wiki/entities/paper-twist2.md)、[`wiki/entities/isaac-teleop.md`](wiki/entities/isaac-teleop.md)、[`wiki/entities/paper-loco-manip-161-131-open-television.md`](wiki/entities/paper-loco-manip-161-131-open-television.md)、[`sources/papers/teleoperation.md`](sources/papers/teleoperation.md)

## [2026-07-22] ingest | sources/blogs/wechat_embodied_ai_lab_loco_manip_contact_survey.md — 复抓公众号 Loco-Manip 接触专题并深化文中论文独立节点（去索引级 stub）

- **工具：** Agent Reach v1.5.0 + wechat-article-for-ai（已预装）
- **原始链接：** <https://mp.weixin.qq.com/s/UjShbwl8p1h9ukymfiRNaw>
- **深化 wiki 论文实体（含新建）：** [`wiki/entities/paper-human-as-humanoid.md`](wiki/entities/paper-human-as-humanoid.md)、[`wiki/entities/paper-humanoidumi.md`](wiki/entities/paper-humanoidumi.md)、[`wiki/entities/paper-vlk-synthetic-loco-manipulation.md`](wiki/entities/paper-vlk-synthetic-loco-manipulation.md)、[`wiki/entities/paper-imagine2real-zero-shot-hoi.md`](wiki/entities/paper-imagine2real-zero-shot-hoi.md)、[`wiki/entities/paper-humanoid-dart.md`](wiki/entities/paper-humanoid-dart.md)、[`wiki/entities/paper-wolf-vla.md`](wiki/entities/paper-wolf-vla.md)、[`wiki/entities/paper-gentlehumanoid.md`](wiki/entities/paper-gentlehumanoid.md)、[`wiki/entities/paper-haic.md`](wiki/entities/paper-haic.md)，以及文中复用的 [`wiki/entities/paper-loco-manip-161-076-sugar.md`](wiki/entities/paper-loco-manip-161-076-sugar.md)、[`wiki/entities/paper-loco-manip-07-wt-umi.md`](wiki/entities/paper-loco-manip-07-wt-umi.md)、[`wiki/entities/paper-motion-cerebellum-ceer.md`](wiki/entities/paper-motion-cerebellum-ceer.md)、[`wiki/entities/paper-loco-manip-161-074-pro-hoi.md`](wiki/entities/paper-loco-manip-161-074-pro-hoi.md)、[`wiki/entities/paper-loco-manip-03-genhoi.md`](wiki/entities/paper-loco-manip-03-genhoi.md)、[`wiki/entities/paper-loco-manip-161-109-falcon.md`](wiki/entities/paper-loco-manip-161-109-falcon.md)、[`wiki/entities/paper-loco-manip-161-039-hmc.md`](wiki/entities/paper-loco-manip-161-039-hmc.md)、[`wiki/entities/paper-loco-manip-161-116-wococo.md`](wiki/entities/paper-loco-manip-161-116-wococo.md)、[`wiki/entities/paper-motion-cerebellum-hoist.md`](wiki/entities/paper-motion-cerebellum-hoist.md)、[`wiki/entities/paper-loco-manip-161-154-openhlm.md`](wiki/entities/paper-loco-manip-161-154-openhlm.md)、[`wiki/entities/paper-hrl-stack-30-wholebodyvla.md`](wiki/entities/paper-hrl-stack-30-wholebodyvla.md)、[`wiki/entities/paper-hrl-stack-36-chip.md`](wiki/entities/paper-hrl-stack-36-chip.md)、[`wiki/entities/paper-hrl-stack-42-thor.md`](wiki/entities/paper-hrl-stack-42-thor.md)、[`wiki/entities/paper-hrl-stack-05-humanx.md`](wiki/entities/paper-hrl-stack-05-humanx.md)、[`wiki/entities/paper-hrl-stack-06-hdmi.md`](wiki/entities/paper-hrl-stack-06-hdmi.md)、[`wiki/entities/paper-cwi-composite-humanoid-whole-body-imitation.md`](wiki/entities/paper-cwi-composite-humanoid-whole-body-imitation.md)、[`wiki/entities/paper-grail.md`](wiki/entities/paper-grail.md)、[`wiki/entities/paper-motionwam-humanoid-loco-manipulation-wam.md`](wiki/entities/paper-motionwam-humanoid-loco-manipulation-wam.md) 等由索引级升格为详细页
- **分类 hub：** [`wiki/overview/loco-manip-contact-technology-map.md`](wiki/overview/loco-manip-contact-technology-map.md)、[`wiki/overview/loco-manip-contact-category-01-contact-data.md`](wiki/overview/loco-manip-contact-category-01-contact-data.md)、[`wiki/overview/loco-manip-contact-category-02-contact-representation.md`](wiki/overview/loco-manip-contact-category-02-contact-representation.md)、[`wiki/overview/loco-manip-contact-category-03-generative-data.md`](wiki/overview/loco-manip-contact-category-03-generative-data.md)、[`wiki/overview/loco-manip-contact-category-04-post-contact-stability.md`](wiki/overview/loco-manip-contact-category-04-post-contact-stability.md)、[`wiki/overview/loco-manip-contact-category-05-vla-world-models.md`](wiki/overview/loco-manip-contact-category-05-vla-world-models.md)

## [2026-07-22] structural | wiki/entities/gr00t-wholebodycontrol.md、wiki/entities/holomotion.md — 独立节点补「源码运行时序图」（GEAR-SONIC / HoloSMPL→Retarget→Train→Docker）；六项目序列图齐备（MotionWAM 仍为不适用）

## [2026-07-22] ingest | sources/sites/{perceptive-bfm,openhlm-project,humanoidarena,dit4dit,holomotion-docs,gr00t-wholebodycontrol-docs} + repos/{mondo_robotics_pmt,openhlm,humanoidarena,gr00t_wholebodycontrol,horizon_robotics_holomotion} — 六项目页/仓开源再核：PerceptiveBFM(PMT 已开源)·OpenHLM(全栈开源升格)·HumanoidArena(代码/数据已发)·GR00T-WBC·HoloMotion 刷新·MotionWAM 仍无独立仓(dit4dit 为前序)；wiki/entities/paper-perceptive-bfm.md wiki/entities/paper-loco-manip-161-154-openhlm.md wiki/entities/paper-humanoidarena.md wiki/entities/gr00t-wholebodycontrol.md wiki/entities/holomotion.md wiki/entities/paper-motionwam-humanoid-loco-manipulation-wam.md wiki/tasks/loco-manipulation.md

## [2026-07-22] ingest | sources/papers/m4world_arxiv_2607_14005.md — 接入 M⁴World（arXiv:2607.14005）多视角多模态驾驶世界模型；升格 wiki/entities/paper-m4world.md；交叉 wiki/methods/generative-world-models.md、wiki/concepts/video-as-simulation.md、wiki/entities/paper-x-world.md、wiki/overview/robot-world-models-training-loop-taxonomy.md、wiki/overview/autonomous-driving-core-algorithms-series.md；机构新增 meituan；未开源

- **开源状态：** **未开源** — arXiv abs/HTML/PDF 均无项目页或 GitHub；训练数据为自采驾驶日志（截至 2026-07-22）
- **新建 papers：** [`sources/papers/m4world_arxiv_2607_14005.md`](sources/papers/m4world_arxiv_2607_14005.md)
- **新建 wiki：** [`wiki/entities/paper-m4world.md`](wiki/entities/paper-m4world.md)
- **机构：** `schema/institutions.json` 新增 `meituan`（美团）；沿用既有 `casia` / `bit`
- **交叉更新：** [`wiki/methods/generative-world-models.md`](wiki/methods/generative-world-models.md)、[`wiki/concepts/video-as-simulation.md`](wiki/concepts/video-as-simulation.md)、[`wiki/entities/paper-x-world.md`](wiki/entities/paper-x-world.md)、[`wiki/overview/robot-world-models-training-loop-taxonomy.md`](wiki/overview/robot-world-models-training-loop-taxonomy.md)、[`wiki/overview/autonomous-driving-core-algorithms-series.md`](wiki/overview/autonomous-driving-core-algorithms-series.md)

## [2026-07-22] ingest | sources/papers/masked_visual_actions_arxiv_2607_19343.md、sources/sites/masked-visual-actions-github-io.md、sources/repos/masked-visual-actions.md — 接入 Masked Visual Actions（arXiv:2607.19343）；升格 wiki/entities/paper-masked-visual-actions.md；交叉 generative-world-models、video-as-simulation、world-models-route-01/03、robot-world-models-training-loop-taxonomy、paper-driftworld、paper-oscar、embodied-eval-benchmark-selection-loop、manipulation；机构新增 umd

- **开源状态：** **部分开源** — 项目页 Code → [`HadiZayer/masked-visual-actions`](https://github.com/HadiZayer/masked-visual-actions)（Apache-2.0）+ HF 双专家 LoRA；推理 / DiffSynth LoRA 训练可运行；DROID URDF 渲染工具 README *coming soon*；项目页 `paper.pdf` 404（以 arXiv:2607.19343 为准）（截至 2026-07-22）
- **新建 papers：** [`sources/papers/masked_visual_actions_arxiv_2607_19343.md`](sources/papers/masked_visual_actions_arxiv_2607_19343.md)
- **新建 sites：** [`sources/sites/masked-visual-actions-github-io.md`](sources/sites/masked-visual-actions-github-io.md)
- **新建 repos：** [`sources/repos/masked-visual-actions.md`](sources/repos/masked-visual-actions.md)
- **新建 wiki：** [`wiki/entities/paper-masked-visual-actions.md`](wiki/entities/paper-masked-visual-actions.md)
- **机构：** `schema/institutions.json` 新增 `umd`（马里兰大学学院公园分校）；沿用既有 `stanford` / `harvard`
- **交叉更新：** [`wiki/methods/generative-world-models.md`](wiki/methods/generative-world-models.md)、[`wiki/concepts/video-as-simulation.md`](wiki/concepts/video-as-simulation.md)、[`wiki/overview/world-models-route-03-virtual-sandbox.md`](wiki/overview/world-models-route-03-virtual-sandbox.md)、[`wiki/overview/world-models-route-01-cascade.md`](wiki/overview/world-models-route-01-cascade.md)、[`wiki/overview/robot-world-models-training-loop-taxonomy.md`](wiki/overview/robot-world-models-training-loop-taxonomy.md)、[`wiki/entities/paper-driftworld.md`](wiki/entities/paper-driftworld.md)、[`wiki/entities/paper-oscar.md`](wiki/entities/paper-oscar.md)、[`wiki/queries/embodied-eval-benchmark-selection-loop.md`](wiki/queries/embodied-eval-benchmark-selection-loop.md)、[`wiki/tasks/manipulation.md`](wiki/tasks/manipulation.md)

## [2026-07-22] ingest | sources/papers/uni_lavira_arxiv_2605_27582.md、sources/sites/xetroubadour-uni-lavira-github-io.md、sources/repos/uni-lavira-code.md — 接入 Uni-LaViRA（arXiv:2605.27582）统一具身导航三层翻译；升格 wiki/entities/paper-uni-lavira.md；交叉 VLN、四范式复现、VLA、WorldVLN、unitree-g1；机构新增 casia/rochester/bmw-nanjing

- **开源状态：** **已开源** — 项目页 Code → [`NJU-R-L-Group-Embodied-Lab/uni-lavira-code`](https://github.com/NJU-R-L-Group-Embodied-Lab/uni-lavira-code)；Habitat + AirSim 评测与四真机入口可运行；License **CC BY-NC-SA 4.0**；依赖 MLLM API（截至 2026-07-22）
- **新建 papers：** [`sources/papers/uni_lavira_arxiv_2605_27582.md`](sources/papers/uni_lavira_arxiv_2605_27582.md)
- **新建 sites：** [`sources/sites/xetroubadour-uni-lavira-github-io.md`](sources/sites/xetroubadour-uni-lavira-github-io.md)
- **新建 repos：** [`sources/repos/uni-lavira-code.md`](sources/repos/uni-lavira-code.md)
- **新建 wiki：** [`wiki/entities/paper-uni-lavira.md`](wiki/entities/paper-uni-lavira.md)
- **机构：** `schema/institutions.json` 新增 `casia`（中科院自动化所）、`rochester`（罗切斯特大学）、`bmw-nanjing`（宝马南京信息技术）；沿用既有 `nju` / `buaa`
- **交叉更新：** [`wiki/tasks/vision-language-navigation.md`](wiki/tasks/vision-language-navigation.md)、[`wiki/overview/vln-open-source-repro-paradigms.md`](wiki/overview/vln-open-source-repro-paradigms.md)、[`wiki/methods/vla.md`](wiki/methods/vla.md)、[`wiki/entities/paper-worldvln-aerial-vln-wam.md`](wiki/entities/paper-worldvln-aerial-vln-wam.md)、[`wiki/entities/unitree-g1.md`](wiki/entities/unitree-g1.md)

## [2026-07-22] ingest | sources/papers/harness_vla_arxiv_2607_08448.md、sources/sites/harnessvla-github-io.md、sources/repos/rpent.md — 接入 Harness VLA（arXiv:2607.08448）；升格 wiki/entities/paper-harness-vla.md；交叉 wiki/methods/vla.md、wiki/concepts/behavior-tree-vla-orchestration.md、wiki/methods/aspire.md、wiki/overview/vla-open-source-repro-landscape-2025.md、wiki/entities/paper-dreamsteer-vla-deployment-steering.md、wiki/entities/lingbot-vla.md、wiki/entities/rldx-1.md、wiki/tasks/manipulation.md；机构新增 casia/striding-ai/infinigence；代码已开源 RLinf/RPent

- **开源状态：** **已开源** — 项目页 Code → [`RLinf/RPent`](https://github.com/RLinf/RPent)；`rpent` CLI + LIBERO-Pro 默认栈；RoboCasa/真机条目以 README Feature Matrix 为准（截至 2026-07-22）
- **新建 papers：** [`sources/papers/harness_vla_arxiv_2607_08448.md`](sources/papers/harness_vla_arxiv_2607_08448.md)
- **新建 sites：** [`sources/sites/harnessvla-github-io.md`](sources/sites/harnessvla-github-io.md)
- **新建 repos：** [`sources/repos/rpent.md`](sources/repos/rpent.md)
- **新建 wiki：** [`wiki/entities/paper-harness-vla.md`](wiki/entities/paper-harness-vla.md)
- **机构：** `schema/institutions.json` 新增 `casia`（中科院自动化所）、`striding-ai`（跨步智能）、`infinigence`（无问芯穹）；沿用 `tsinghua` / `purdue` / `hkust` / `zgca`
- **交叉更新：** [`wiki/methods/vla.md`](wiki/methods/vla.md)、[`wiki/concepts/behavior-tree-vla-orchestration.md`](wiki/concepts/behavior-tree-vla-orchestration.md)、[`wiki/methods/aspire.md`](wiki/methods/aspire.md)、[`wiki/overview/vla-open-source-repro-landscape-2025.md`](wiki/overview/vla-open-source-repro-landscape-2025.md)、[`wiki/entities/paper-dreamsteer-vla-deployment-steering.md`](wiki/entities/paper-dreamsteer-vla-deployment-steering.md)、[`wiki/entities/lingbot-vla.md`](wiki/entities/lingbot-vla.md)、[`wiki/entities/rldx-1.md`](wiki/entities/rldx-1.md)、[`wiki/tasks/manipulation.md`](wiki/tasks/manipulation.md)、[`sources/repos/rlinf.md`](sources/repos/rlinf.md)


## [2026-07-22] ingest | sources/papers/adp_arxiv_2607_03454.md、sources/sites/seokju-lee-adp-github-io.md — 接入 ADP（arXiv:2607.03454）对抗动力学先验；升格 wiki/entities/paper-adp.md；交叉 amp-reward、locomotion、balance-recovery、humanoid-locomotion、centroidal-dynamics、unitree-g1、amp-add-smp、paper-unified-walk-run-recovery-sdamp；机构新增 samsung/hanyang/kimm；代码 coming soon

- **开源状态：** **宣称将开源 / 待发布** — 项目页 Code *(coming soon)*；[`seokju-lee/adp`](https://github.com/seokju-lee/adp) 仅为项目站（`index.html` + `static/`），无可运行训练/推理入口（截至 2026-07-22）
- **新建 papers：** [`sources/papers/adp_arxiv_2607_03454.md`](sources/papers/adp_arxiv_2607_03454.md)
- **新建 sites：** [`sources/sites/seokju-lee-adp-github-io.md`](sources/sites/seokju-lee-adp-github-io.md)
- **新建 wiki：** [`wiki/entities/paper-adp.md`](wiki/entities/paper-adp.md)
- **机构：** `schema/institutions.json` 新增 `samsung`（三星电子）、`hanyang`（汉阳大学）、`kimm`（韩国机械材料研究院）；沿用既有 `kaist`
- **交叉更新：** [`wiki/methods/amp-reward.md`](wiki/methods/amp-reward.md)、[`wiki/tasks/locomotion.md`](wiki/tasks/locomotion.md)、[`wiki/tasks/balance-recovery.md`](wiki/tasks/balance-recovery.md)、[`wiki/tasks/humanoid-locomotion.md`](wiki/tasks/humanoid-locomotion.md)、[`wiki/concepts/centroidal-dynamics.md`](wiki/concepts/centroidal-dynamics.md)、[`wiki/entities/unitree-g1.md`](wiki/entities/unitree-g1.md)、[`wiki/comparisons/amp-add-smp-motion-prior-variants.md`](wiki/comparisons/amp-add-smp-motion-prior-variants.md)、[`wiki/entities/paper-unified-walk-run-recovery-sdamp.md`](wiki/entities/paper-unified-walk-run-recovery-sdamp.md)

## [2026-07-22] ingest | sources/papers/driftworld_arxiv_2607_15065.md、sources/sites/susie-lu-driftworld-github-io.md、sources/repos/driftworld.md — 接入 DriftWorld（arXiv:2607.15065）1-step drifting 动作条件世界模型；升格 wiki/entities/paper-driftworld.md

- **开源状态：** **部分开源** — 项目页 Paper/Code 双链；[`Susie-Lu/driftworld`](https://github.com/Susie-Lu/driftworld) + HF checkpoint；Push-T 训练/可视化/指标/GPC-RANK/策略评估可运行；其它数据集 README *Will be added soon*；License 未声明（截至 2026-07-22）
- **新建 papers：** [`sources/papers/driftworld_arxiv_2607_15065.md`](sources/papers/driftworld_arxiv_2607_15065.md)
- **新建 sites：** [`sources/sites/susie-lu-driftworld-github-io.md`](sources/sites/susie-lu-driftworld-github-io.md)
- **新建 repos：** [`sources/repos/driftworld.md`](sources/repos/driftworld.md)
- **新建 wiki：** [`wiki/entities/paper-driftworld.md`](wiki/entities/paper-driftworld.md)
- **机构：** `schema/institutions.json` 新增 `harvard`（哈佛大学）
- **交叉更新：** [`wiki/methods/generative-world-models.md`](wiki/methods/generative-world-models.md)、[`wiki/overview/world-models-route-03-virtual-sandbox.md`](wiki/overview/world-models-route-03-virtual-sandbox.md)、[`wiki/overview/robot-world-models-training-loop-taxonomy.md`](wiki/overview/robot-world-models-training-loop-taxonomy.md)、[`wiki/concepts/video-as-simulation.md`](wiki/concepts/video-as-simulation.md)、[`wiki/entities/paper-oscar.md`](wiki/entities/paper-oscar.md)、[`wiki/entities/paper-gigaworld-1-policy-evaluation.md`](wiki/entities/paper-gigaworld-1-policy-evaluation.md)、[`wiki/entities/paper-shenlan-wm-15-worldgym.md`](wiki/entities/paper-shenlan-wm-15-worldgym.md)、[`wiki/queries/embodied-eval-benchmark-selection-loop.md`](wiki/queries/embodied-eval-benchmark-selection-loop.md)

## [2026-07-22] ingest | sources/papers/pot_vla_arxiv_2607_18016.md、patch_policy_arxiv_2607_18236.md、world_translation_arxiv_2607_18154.md、mevion_arxiv_2607_17970.md + sites/repos — 接入每日机器人情报四项高价值资料

- **新建 wiki：** [`wiki/entities/paper-pot-vla.md`](wiki/entities/paper-pot-vla.md)、[`wiki/entities/paper-patch-policy.md`](wiki/entities/paper-patch-policy.md)、[`wiki/entities/paper-world-translation.md`](wiki/entities/paper-world-translation.md)、[`wiki/entities/paper-mevion.md`](wiki/entities/paper-mevion.md)
- **开源核查：** MEVION **已开源、研究原型**；Patch Policy **代码待发布**；POT-VLA 与 World Translation **未发现官方代码入口**（截至 2026-07-22）。
- **交叉更新：** [`wiki/methods/vla.md`](wiki/methods/vla.md)、[`wiki/entities/unitree-g1.md`](wiki/entities/unitree-g1.md)、[`wiki/concepts/sim2real.md`](wiki/concepts/sim2real.md)、[`wiki/tasks/bimanual-manipulation.md`](wiki/tasks/bimanual-manipulation.md)

## [2026-07-21] concept | wiki/concepts/torque-source-abstraction-gap.md — V30 P1 执行器驱动链选型闭环知识链第二页（力矩源抽象 Gap）

- 新建概念页：「理想力矩源」抽象 ↔ 真实执行器 取舍——明示 RL/MPC 策略把执行器当理想力矩源的抽象在摩擦/齿隙/带宽/热约束下何时破，讲成「策略力矩指令能否被真实驱动链忠实执行」的物理根因；配抽象成立条件表、四层 gap 归因 Mermaid、收窄力矩执行 gap 三条工程路线（摩擦辨识补偿 / 执行器网络 / 力矩传感闭环）与常见误判/误区速查、英文缩写表。
- 沉淀页面：[`wiki/concepts/torque-source-abstraction-gap.md`](wiki/concepts/torque-source-abstraction-gap.md)
- 双向回链（消孤儿）：[`wiki/queries/actuator-drive-chain-selection-loop.md`](wiki/queries/actuator-drive-chain-selection-loop.md)（Query 页 `related` + 关联页面补入本页）与 [`wiki/concepts/implicit-explicit-actuator-modeling.md`](wiki/concepts/implicit-explicit-actuator-modeling.md)（`related` + 关联页面互链）。
- 门禁：`make lint` 0 errors（仅 1 条既有 INFO 陈旧声明，与本次无关）；`make export graph` 后 `graph-stats.json` 0 orphans、`largest_community_ratio` 0.219（`community_quality_warning: false`）；节点 1748→1749、边 14078→14092。
- 对应 v30 清单 P1「执行器驱动链选型闭环知识链 (+2)」两项子任务全部打勾（`[x]`），该组小节由 `[~]` 转 `[x]`。

## [2026-07-21] ingest | sources/blogs/wechat_shenlan_ai_ad_2d_detection.md、sources/blogs/wechat_shenlan_ai_ad_3d_detection.md — 短链补抓深蓝AI 专辑第 1–2 篇（2D/3D 检测）；替换 pending；专辑总览升至 5/5

- **工具：** Agent Reach v1.5.0 + wechat-article-for-ai（Camoufox）；**短链直连成功** — <https://mp.weixin.qq.com/s/7Mm5OwVKgoyT4Zpr45E34A>（2D）、<https://mp.weixin.qq.com/s/1d7P4HDXmmZUZiVNx1HfXw>（3D）
- **新建 blogs：** [`sources/blogs/wechat_shenlan_ai_ad_2d_detection.md`](sources/blogs/wechat_shenlan_ai_ad_2d_detection.md)、[`sources/blogs/wechat_shenlan_ai_ad_3d_detection.md`](sources/blogs/wechat_shenlan_ai_ad_3d_detection.md)
- **删除占位：** `sources/blogs/wechat_shenlan_ai_ad_2d_detection_pending.md`、`sources/blogs/wechat_shenlan_ai_ad_3d_detection_pending.md`
- **新建 raw：** [`sources/raw/wechat_shenlan_ai_ad_2d_detection_2026-06-14.md`](sources/raw/wechat_shenlan_ai_ad_2d_detection_2026-06-14.md)、[`sources/raw/wechat_shenlan_ai_ad_3d_detection_2026-06-22.md`](sources/raw/wechat_shenlan_ai_ad_3d_detection_2026-06-22.md)
- **更新：** [`sources/raw/wechat_shenlan_ai_ad_core_algorithms_album_2026.json`](sources/raw/wechat_shenlan_ai_ad_core_algorithms_album_2026.json)（1–2 篇 status=scraped）、[`wiki/overview/autonomous-driving-core-algorithms-series.md`](wiki/overview/autonomous-driving-core-algorithms-series.md)（5/5 + 检测速查表）、[`wiki/methods/object-detection.md`](wiki/methods/object-detection.md)

## [2026-07-21] ingest | sources/sites/x-humanoid.md、sources/sites/x-humanoid-opensource-cloud.md、sources/repos/open-x-humanoid.md — 接入 X-Humanoid 官网/天工造物社区/Open-X-Humanoid；升格 wiki/entities/x-humanoid.md；交叉 tienkung-humanoid-open-source、open-source-humanoid-hardware、openloong、pelican-unified-1、paper-loco-manip-161-038-hex、paper-heracles-humanoid-diffusion、humanoid-robot、robot-open-source-wechat-issue01-curator

- **开源状态：** **已开源（多入口）** — 官网开源页 + 天工造物文档中心 + [`Open-X-Humanoid`](https://github.com/Open-X-Humanoid)（约 23 公开仓）；权重多在 HF `X-Humanoid`
- **新建 sites：** [`sources/sites/x-humanoid.md`](sources/sites/x-humanoid.md)、[`sources/sites/x-humanoid-opensource-cloud.md`](sources/sites/x-humanoid-opensource-cloud.md)
- **新建 repos：** [`sources/repos/open-x-humanoid.md`](sources/repos/open-x-humanoid.md)
- **新建 wiki：** [`wiki/entities/x-humanoid.md`](wiki/entities/x-humanoid.md)
- **交叉更新：** [`wiki/entities/tienkung-humanoid-open-source.md`](wiki/entities/tienkung-humanoid-open-source.md)、[`wiki/entities/open-source-humanoid-hardware.md`](wiki/entities/open-source-humanoid-hardware.md)、[`wiki/entities/openloong.md`](wiki/entities/openloong.md)、[`wiki/methods/pelican-unified-1.md`](wiki/methods/pelican-unified-1.md)、[`wiki/entities/paper-loco-manip-161-038-hex.md`](wiki/entities/paper-loco-manip-161-038-hex.md)、[`wiki/entities/paper-heracles-humanoid-diffusion.md`](wiki/entities/paper-heracles-humanoid-diffusion.md)、[`wiki/entities/humanoid-robot.md`](wiki/entities/humanoid-robot.md)、[`wiki/overview/robot-open-source-wechat-issue01-curator.md`](wiki/overview/robot-open-source-wechat-issue01-curator.md)

## [2026-07-21] ingest | sources/blogs/wechat_shenlan_ai_ad_* + album — 深蓝AI《自动驾驶核心算法盘点》专辑（5 篇）；升格 wiki/overview/autonomous-driving-core-algorithms-series.md；正文入库 3/5（规划控制 / SLAM高精地图 / 跟踪预测）；第 1–2 篇 2D/3D 检测 CAPTCHA 占位

- **工具：** [Agent Reach](https://github.com/Panniantong/Agent-Reach) v1.5.0 + [wechat-article-for-ai](https://github.com/bzd6661/wechat-article-for-ai)（Camoufox）；专辑页可直取列表；文章直连遇「环境异常」滑块，**搜狗微信中转**成功抓取第 3–5 篇；第 1–2 篇搜狗未收录且直连 CAPTCHA，仅占位
- **专辑：** <https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzY4NjA5NTgyMQ==&action=getalbum&album_id=4596755873481310212>（深蓝AI，5 篇）
- **新建 blogs：** [`sources/blogs/wechat_shenlan_ai_ad_planning_control.md`](sources/blogs/wechat_shenlan_ai_ad_planning_control.md)、[`sources/blogs/wechat_shenlan_ai_ad_slam_hdmap.md`](sources/blogs/wechat_shenlan_ai_ad_slam_hdmap.md)、[`sources/blogs/wechat_shenlan_ai_ad_tracking_prediction.md`](sources/blogs/wechat_shenlan_ai_ad_tracking_prediction.md)、[`sources/blogs/wechat_shenlan_ai_ad_2d_detection_pending.md`](sources/blogs/wechat_shenlan_ai_ad_2d_detection_pending.md)、[`sources/blogs/wechat_shenlan_ai_ad_3d_detection_pending.md`](sources/blogs/wechat_shenlan_ai_ad_3d_detection_pending.md)
- **新建 raw：** [`sources/raw/wechat_shenlan_ai_ad_planning_control_2026-07-09.md`](sources/raw/wechat_shenlan_ai_ad_planning_control_2026-07-09.md)、[`sources/raw/wechat_shenlan_ai_ad_slam_hdmap_2026-07-13.md`](sources/raw/wechat_shenlan_ai_ad_slam_hdmap_2026-07-13.md)、[`sources/raw/wechat_shenlan_ai_ad_tracking_prediction_2026-07-19.md`](sources/raw/wechat_shenlan_ai_ad_tracking_prediction_2026-07-19.md)、[`sources/raw/wechat_shenlan_ai_ad_core_algorithms_album_2026.json`](sources/raw/wechat_shenlan_ai_ad_core_algorithms_album_2026.json)
- **新建 wiki：** [`wiki/overview/autonomous-driving-core-algorithms-series.md`](wiki/overview/autonomous-driving-core-algorithms-series.md)
- **交叉更新：** [`wiki/overview/navigation-slam-autonomy-stack.md`](wiki/overview/navigation-slam-autonomy-stack.md)、[`wiki/methods/object-detection.md`](wiki/methods/object-detection.md)、[`wiki/methods/lqr-ilqr.md`](wiki/methods/lqr-ilqr.md)、[`wiki/methods/model-predictive-control.md`](wiki/methods/model-predictive-control.md)、[`wiki/entities/python-robotics.md`](wiki/entities/python-robotics.md)
- **待跟进：** 第 1–2 篇正文补抓后替换 pending blogs 并扩充专辑总览检测子索引

## [2026-07-21] ingest | sources/papers/gmt_arxiv_2506_14770.md、sources/sites/gmt-humanoid-github-io.md、sources/repos/humanoid-general-motion-tracking.md — 接入 GMT（arXiv:2506.14770）项目页与部分开源代码；新建 wiki/entities/paper-gmt.md；校正 161 策展误写扩散；交叉 whole-body-tracking-pipeline、humanoid-motion-tracking-method-selection、paper-phygile、paper-resmimic、egm-efficient-general-mimic、paper-loco-manip-161-009-gmt、paper-notebook-general-motion-tracking-for-humanoid-whole-body

- **开源状态：** **部分开源** — 项目页 → [`zixuan417/humanoid-general-motion-tracking`](https://github.com/zixuan417/humanoid-general-motion-tracking)；MuJoCo sim2sim + pretrained + 示例 motion；训练/数据处理与重定向 **待发布**（截至 2026-07-21）
- **新建 papers：** [`sources/papers/gmt_arxiv_2506_14770.md`](sources/papers/gmt_arxiv_2506_14770.md)
- **新建 sites：** [`sources/sites/gmt-humanoid-github-io.md`](sources/sites/gmt-humanoid-github-io.md)
- **新建 repos：** [`sources/repos/humanoid-general-motion-tracking.md`](sources/repos/humanoid-general-motion-tracking.md)
- **新建 wiki：** [`wiki/entities/paper-gmt.md`](wiki/entities/paper-gmt.md)
- **校正：** [`wiki/entities/paper-loco-manip-161-009-gmt.md`](wiki/entities/paper-loco-manip-161-009-gmt.md)、[`sources/papers/loco_manip_161_survey_009_gmt.md`](sources/papers/loco_manip_161_survey_009_gmt.md) — 纠正公众号「扩散/流匹配」误述
- **交叉更新：** [`wiki/concepts/whole-body-tracking-pipeline.md`](wiki/concepts/whole-body-tracking-pipeline.md)、[`wiki/queries/humanoid-motion-tracking-method-selection.md`](wiki/queries/humanoid-motion-tracking-method-selection.md)、[`wiki/entities/paper-phygile.md`](wiki/entities/paper-phygile.md)、[`wiki/entities/paper-resmimic.md`](wiki/entities/paper-resmimic.md)、[`wiki/methods/egm-efficient-general-mimic.md`](wiki/methods/egm-efficient-general-mimic.md)、[`wiki/entities/paper-notebook-general-motion-tracking-for-humanoid-whole-body.md`](wiki/entities/paper-notebook-general-motion-tracking-for-humanoid-whole-body.md)、[`wiki/entities/paper-humanoidarena.md`](wiki/entities/paper-humanoidarena.md)、[`wiki/overview/loco-manip-161-category-01-motion-base-wbt.md`](wiki/overview/loco-manip-161-category-01-motion-base-wbt.md)、[`wiki/overview/paper-notebook-category-04-loco-manipulation-and-wbc.md`](wiki/overview/paper-notebook-category-04-loco-manipulation-and-wbc.md)

## [2026-07-21] ingest | sources/papers/glob3r_arxiv_2607_09225.md — Glob3R（HKUST×通义×NJU×Fudan）全局 SfM + 3D 基础模型；升格 wiki/entities/paper-glob3r.md；交叉 wiki/methods/lingbot-map.md、wiki/concepts/state-estimation.md、wiki/overview/topic-state-estimation.md、wiki/overview/navigation-slam-autonomy-stack.md；官方仓 aigc3d/Glob3R 占位（Inference TODO）

- **开源状态：** **部分开源（占位仓）** — 项目页 Code → [`aigc3d/Glob3R`](https://github.com/aigc3d/Glob3R)；仓内仅 README，Inference/Evaluation 仍 TODO（截至 2026-07-21）
- **新建 papers：** [`sources/papers/glob3r_arxiv_2607_09225.md`](sources/papers/glob3r_arxiv_2607_09225.md)
- **新建 sites：** [`sources/sites/junyuandeng-glob3r-github-io.md`](sources/sites/junyuandeng-glob3r-github-io.md)
- **新建 repos：** [`sources/repos/glob3r.md`](sources/repos/glob3r.md)
- **新建 wiki：** [`wiki/entities/paper-glob3r.md`](wiki/entities/paper-glob3r.md)
- **机构：** `schema/institutions.json` 为 `alibaba` 追加 alias `tongyi` / `tongyi-lab`
- **交叉更新：** [`wiki/methods/lingbot-map.md`](wiki/methods/lingbot-map.md)、[`wiki/concepts/state-estimation.md`](wiki/concepts/state-estimation.md)、[`wiki/overview/topic-state-estimation.md`](wiki/overview/topic-state-estimation.md)、[`wiki/overview/navigation-slam-autonomy-stack.md`](wiki/overview/navigation-slam-autonomy-stack.md)

## [2026-07-21] ingest | sources/papers/egohtr_arxiv_2607_13472.md — EgoHTR（ETH×Stanford×Berkeley×TUM）rough-terrain 人–场景 4D 数据集；升格 wiki/entities/paper-egohtr.md；交叉 wiki/comparisons/humanoid-reference-motion-datasets.md、wiki/entities/amass.md、wiki/tasks/locomotion.md、wiki/concepts/terrain-adaptation.md、wiki/concepts/motion-retargeting.md、wiki/entities/paper-hrl-stack-03-omniretarget.md、wiki/entities/paper-rpl-robust-humanoid-perceptive-locomotion.md；项目页 Dataset/Code coming soon

- **开源状态：** **宣称将开源 / 待发布** — 项目页 Dataset / Code 均 *coming soon*；GitHub org 仅站点仓（截至 2026-07-21）
- **新建 papers：** [`sources/papers/egohtr_arxiv_2607_13472.md`](sources/papers/egohtr_arxiv_2607_13472.md)
- **新建 sites：** [`sources/sites/egohtr-github-io.md`](sources/sites/egohtr-github-io.md)
- **新建 wiki：** [`wiki/entities/paper-egohtr.md`](wiki/entities/paper-egohtr.md)
- **机构：** `schema/institutions.json` 新增 `tum`（慕尼黑工业大学）
- **交叉更新：** [`wiki/comparisons/humanoid-reference-motion-datasets.md`](wiki/comparisons/humanoid-reference-motion-datasets.md)、[`wiki/entities/amass.md`](wiki/entities/amass.md)、[`wiki/tasks/locomotion.md`](wiki/tasks/locomotion.md)、[`wiki/concepts/terrain-adaptation.md`](wiki/concepts/terrain-adaptation.md)、[`wiki/concepts/motion-retargeting.md`](wiki/concepts/motion-retargeting.md)、[`wiki/entities/paper-hrl-stack-03-omniretarget.md`](wiki/entities/paper-hrl-stack-03-omniretarget.md)、[`wiki/entities/paper-rpl-robust-humanoid-perceptive-locomotion.md`](wiki/entities/paper-rpl-robust-humanoid-perceptive-locomotion.md)、[`wiki/methods/motion-retargeting-gmr.md`](wiki/methods/motion-retargeting-gmr.md)

## [2026-07-21] ingest | sources/papers/ergocub_shared_embodied_intelligence_nmi_s42256_026_01272_2.md + sites/repos — ergoCub Shared Embodied Intelligence（Nat Mach Intell 2026）；wiki/entities/paper-ergocub-shared-embodied-intelligence.md；交叉 WBC / locomotion / sim2real / humanoid-robot / icub3 / open-source-hardware；注册 iit/inail/generative-bionics/university-of-manchester

- **开源状态：** **已开源** — 论文复现仓（BSD-3-Clause）+ [adam](https://github.com/ami-iit/adam) + [shared-controllers](https://github.com/gbionics/shared-controllers)；非整机 CAD/BOM（截至 2026-07-21）
- **新建 papers：** [`sources/papers/ergocub_shared_embodied_intelligence_nmi_s42256_026_01272_2.md`](sources/papers/ergocub_shared_embodied_intelligence_nmi_s42256_026_01272_2.md)
- **新建 sites：** [`sources/sites/ergocub-eu.md`](sources/sites/ergocub-eu.md)
- **新建 repos：** [`sources/repos/paper-sartore-2025-ergocub-nmi.md`](sources/repos/paper-sartore-2025-ergocub-nmi.md)、[`sources/repos/ami-iit-adam.md`](sources/repos/ami-iit-adam.md)、[`sources/repos/gbionics-shared-controllers.md`](sources/repos/gbionics-shared-controllers.md)
- **新建 wiki：** [`wiki/entities/paper-ergocub-shared-embodied-intelligence.md`](wiki/entities/paper-ergocub-shared-embodied-intelligence.md)
- **机构：** `schema/institutions.json` 新增 `iit` / `inail` / `generative-bionics` / `university-of-manchester`
- **交叉更新：** [`wiki/concepts/whole-body-control.md`](wiki/concepts/whole-body-control.md)、[`wiki/tasks/locomotion.md`](wiki/tasks/locomotion.md)、[`wiki/concepts/sim2real.md`](wiki/concepts/sim2real.md)、[`wiki/entities/humanoid-robot.md`](wiki/entities/humanoid-robot.md)、[`wiki/entities/paper-notebook-icub3-avatar-system-enabling-remote-fully-immers.md`](wiki/entities/paper-notebook-icub3-avatar-system-enabling-remote-fully-immers.md)、[`wiki/entities/open-source-humanoid-hardware.md`](wiki/entities/open-source-humanoid-hardware.md)

## [2026-07-21] ingest | sources/papers/rynnbrain_1_1_arxiv_2607_17977.md + sites/repos — RynnBrain 1.1 具身基础模型；wiki/entities/paper-rynnbrain-1-1.md；交叉 vla / embodied-scaling-laws / foundation-policy / qwen-vla / rynnworld-4d

- **开源状态：** **部分开源** — 2B/9B/122B-A10B 权重（HF/ModelScope）+ 推理/`cookbooks`（Apache-2.0）；**未见** RynnBrain-VLA 训练码与 VLA 权重（截至 2026-07-21）
- **新建 papers：** [`sources/papers/rynnbrain_1_1_arxiv_2607_17977.md`](sources/papers/rynnbrain_1_1_arxiv_2607_17977.md)
- **新建 sites：** [`sources/sites/rynnbrain-alibaba-damo.md`](sources/sites/rynnbrain-alibaba-damo.md)
- **新建 repos：** [`sources/repos/rynnbrain.md`](sources/repos/rynnbrain.md)
- **新建 wiki：** [`wiki/entities/paper-rynnbrain-1-1.md`](wiki/entities/paper-rynnbrain-1-1.md)
- **机构：** `schema/institutions.json` 为 `alibaba` 追加 alias `damo` / `damo-academy`
- **交叉更新：** [`wiki/methods/vla.md`](wiki/methods/vla.md)、[`wiki/concepts/embodied-scaling-laws.md`](wiki/concepts/embodied-scaling-laws.md)、[`wiki/concepts/foundation-policy.md`](wiki/concepts/foundation-policy.md)、[`wiki/entities/qwen-vla.md`](wiki/entities/qwen-vla.md)、[`wiki/entities/paper-rynnworld-4d-rgb-depth-flow.md`](wiki/entities/paper-rynnworld-4d-rgb-depth-flow.md)

## [2026-07-21] ingest | sources/papers/turingvit+x_world+x_cache+x_foresight+x_mind + sites — 小鹏 TuringViT / X-World 系列五篇；wiki 实体页；交叉更新 WM/VLA/ViT

- **开源状态：** 五篇项目页截至 2026-07-21 **均未列 GitHub/权重**（仅 arXiv / Tech Report）
- **新建 papers：** [`sources/papers/turingvit_arxiv_2606_24253.md`](sources/papers/turingvit_arxiv_2606_24253.md)、[`sources/papers/x_world_arxiv_2603_19979.md`](sources/papers/x_world_arxiv_2603_19979.md)、[`sources/papers/x_cache_arxiv_2604_20289.md`](sources/papers/x_cache_arxiv_2604_20289.md)、[`sources/papers/x_foresight_arxiv_2605_24892.md`](sources/papers/x_foresight_arxiv_2605_24892.md)、[`sources/papers/x_mind_arxiv_2606_28758.md`](sources/papers/x_mind_arxiv_2606_28758.md)
- **新建 sites：** [`sources/sites/turingvit-github-io.md`](sources/sites/turingvit-github-io.md)、[`sources/sites/x-world-1-github-io.md`](sources/sites/x-world-1-github-io.md)、[`sources/sites/x-cache-1-github-io.md`](sources/sites/x-cache-1-github-io.md)、[`sources/sites/x-foresight-1-github-io.md`](sources/sites/x-foresight-1-github-io.md)、[`sources/sites/xp-x-mind-github-io.md`](sources/sites/xp-x-mind-github-io.md)
- **新建 wiki：** [`wiki/entities/paper-turingvit.md`](wiki/entities/paper-turingvit.md)、[`wiki/entities/paper-x-world.md`](wiki/entities/paper-x-world.md)、[`wiki/entities/paper-x-cache.md`](wiki/entities/paper-x-cache.md)、[`wiki/entities/paper-x-foresight.md`](wiki/entities/paper-x-foresight.md)、[`wiki/entities/paper-x-mind.md`](wiki/entities/paper-x-mind.md)
- **交叉更新：** [`wiki/methods/generative-world-models.md`](wiki/methods/generative-world-models.md)、[`wiki/concepts/world-action-models.md`](wiki/concepts/world-action-models.md)、[`wiki/concepts/video-as-simulation.md`](wiki/concepts/video-as-simulation.md)、[`wiki/concepts/vision-transformer.md`](wiki/concepts/vision-transformer.md)、[`wiki/concepts/vision-backbones.md`](wiki/concepts/vision-backbones.md)、[`wiki/methods/vla.md`](wiki/methods/vla.md)

## [2026-07-21] ingest | sources/repos/rebot-devarm.md + sources/sites/rebot-devarm-seeed-wiki.md — Seeed reBot-DevArm 全栈开源桌面六轴臂；wiki/entities/rebot-devarm.md；交叉 lerobot / manipulation / teleoperation / stackforce / parol6；注册 seeed 机构

- **开源状态：** 已开源（全栈）— 硬件 CERN-OHL-W-2.0 + 软件 Apache-2.0；OSHWA CN000024；STEP/BOM/组装 + Motorbridge / ROS2 / LeRobot / Pinocchio；RS Isaac Sim 演示仓 `Seeed-Projects/reBot-Isaacsim`
- **新建：** [`sources/repos/rebot-devarm.md`](sources/repos/rebot-devarm.md)、[`sources/sites/rebot-devarm-seeed-wiki.md`](sources/sites/rebot-devarm-seeed-wiki.md)、[`wiki/entities/rebot-devarm.md`](wiki/entities/rebot-devarm.md)
- **机构：** `schema/institutions.json` 新增 `seeed`（矽递科技 / Seeed Studio）
- **交叉更新：** [`wiki/entities/lerobot.md`](wiki/entities/lerobot.md)、[`wiki/tasks/manipulation.md`](wiki/tasks/manipulation.md)、[`wiki/tasks/teleoperation.md`](wiki/tasks/teleoperation.md)、[`wiki/entities/stackforce.md`](wiki/entities/stackforce.md)、[`wiki/entities/parol6-source-robotics.md`](wiki/entities/parol6-source-robotics.md)


## [2026-07-21] ingest | sources/sites/systems_engineering_* + dds_omg_rtos_edge_ota_safety — 系统工程主题盘点：复用 CAN/EtherCAT/ROS2/UDP；新建 15 概念 + 1 专题总览

- **盘点结论（已有独立节点，复用不重建）：**
  - CAN / CAN FD / EtherCAT / ROS 2 / LCM / UDP 组播 / 通信专题 / 实时中间件 Query / 控制环延迟
- **一手资料：**
  - [`sources/sites/systems_engineering_os_network_primary_refs.md`](sources/sites/systems_engineering_os_network_primary_refs.md)
  - [`sources/sites/systems_engineering_data_distributed_primary_refs.md`](sources/sites/systems_engineering_data_distributed_primary_refs.md)
  - [`sources/sites/systems_engineering_deploy_obs_security_primary_refs.md`](sources/sites/systems_engineering_deploy_obs_security_primary_refs.md)
  - [`sources/sites/dds_omg_rtos_edge_ota_safety_primary_refs.md`](sources/sites/dds_omg_rtos_edge_ota_safety_primary_refs.md)
- **专题枢纽：** [`wiki/overview/topic-systems-engineering.md`](wiki/overview/topic-systems-engineering.md)
- **新建概念页：**
  - `operating-system-basics` / `network-protocol-stack` / `database-fundamentals` / `cache-consistency-pitfalls`
  - `message-queue-reliability` / `distributed-systems-basics` / `container-orchestration-cicd`
  - `observability-logs-metrics-tracing` / `software-security-basics` / `rtos-realtime-scheduling`
  - `dds-communication` / `edge-cloud-robotics` / `control-inference-frequency-decoupling`
  - `model-versioning-ota` / `robot-safety-state-machine`
- **交叉更新：** `ros2-basics`、`topic-communication`、`real-time-control-middleware-guide`、`wbc-fsm`

## [2026-07-21] ingest | sources/papers/handroid_arxiv_2607_16187.md — 复核 Handroid（arXiv:2607.16187）：CAD/BOM 已开源、Code 仍占位；wiki/entities/handroid.md 刷新核查日

## [2026-07-21] ingest | sources/papers/raven_rl_adaptive_visibility_graph_arxiv_2607_15701.md — UCLA RoMeLa RAVEN；wiki/entities/paper-raven-rl-adaptive-visibility-graph-mpc.md；交叉 ARTEMIS / Booster Gym / mpc-vs-rl / humanoid-locomotion

## [2026-07-21] ingest | sources/papers/xiaomi_robotics_1_arxiv_2607_15330.md — 接入 XR-1 arXiv:2607.15330；刷新项目页/占位仓开源边界；wiki/entities/xiaomi-robotics-1.md

## [2026-07-21] structural | Isaac Gym / Isaac Sim / Isaac Lab 独立节点补齐运行时序图与类图

- 新建独立实体：[`wiki/entities/isaac-sim.md`](wiki/entities/isaac-sim.md)（此前仅散见于总览与对比页）
- 为三页补齐 **`## 核心类图`** + **`## 源码运行时序图`**（mermaid `classDiagram` / `sequenceDiagram`）：
  - [`wiki/entities/isaac-gym.md`](wiki/entities/isaac-gym.md)
  - [`wiki/entities/isaac-sim.md`](wiki/entities/isaac-sim.md)
  - [`wiki/entities/isaac-lab.md`](wiki/entities/isaac-lab.md)
- 总览枢纽改写为三代产品：[`wiki/entities/isaac-gym-isaac-lab.md`](wiki/entities/isaac-gym-isaac-lab.md)
- Source 拆分：[`sources/repos/isaac_sim.md`](sources/repos/isaac_sim.md)、[`isaac_lab.md`](sources/repos/isaac_lab.md)、[`isaac_gym.md`](sources/repos/isaac_gym.md)；联合索引更新 [`isaac_gym_isaac_lab.md`](sources/repos/isaac_gym_isaac_lab.md)
- 交叉：`nvidia-omniverse`、`nvidia-physical-ai-learning`、`curobo`、`mjlab`、`mujoco-vs-isaac-sim`、`robot-training-stack-layers-technology-map`、`index.md`

## [2026-07-21] ingest | sources/repos/ssik.md — 接入 UW PRL 解析 IK 库 ssik；wiki/entities/ssik.md；交叉更新 manipulation / teleoperation / moveit2 / curobo

## [2026-07-20] query | wiki/queries/actuator-drive-chain-selection-loop.md — V30 P1 执行器驱动链选型闭环知识链

- 新建端到端 Query 页：EDA 电路设计 → 电机驱动固件 FOC → 执行器建模与摩擦辨识 → 实时总线闭环集成 四层驱动链选型决策链，配四层决策树 Mermaid、选型矛盾/失败模式速查、英文缩写表。
- 沉淀页面：[`wiki/queries/actuator-drive-chain-selection-loop.md`](wiki/queries/actuator-drive-chain-selection-loop.md)
- 交叉回链（消孤儿）：[`wiki/queries/ethercat-master-optimization.md`](wiki/queries/ethercat-master-optimization.md) 与 [`wiki/concepts/implicit-explicit-actuator-modeling.md`](wiki/concepts/implicit-explicit-actuator-modeling.md) 各补入本页链接。
- 门禁：`make lint` 0 errors；`make export graph` 后 `graph-stats.json` 0 orphans、`largest_community_ratio` 0.22（`community_quality_warning: false`）；节点 1687→1716、边 13524→13723（均已越过 V30 目标 ≥1698 / ≥13580）。
- 对应 v30 清单 P1 首项已打勾（`[x]`），该组小节转 `[~]`。

## [2026-07-20] ingest | sources/blogs/flexion_niantic_nvidia_sim2real_rgb_2026-07-20.md — Flexion×Niantic×NVIDIA RGB 导航 Sim2Real 管线；wiki/entities/flexion-niantic-nvidia-rgb-sim2real-pipeline.md；交叉 sim2real / flexion-reflect-v1 / gs-playground

## [2026-07-20] ingest | sources/blogs/wechat_shenlan_scirobotics_china_top3_2026-07-02.md — 深蓝AI 近五年 Science Robotics 中国顶尖高校盘点；9 篇独立论文实体（浙大/北航/清华）

- 原始资料：[`sources/blogs/wechat_shenlan_scirobotics_china_top3_2026-07-02.md`](sources/blogs/wechat_shenlan_scirobotics_china_top3_2026-07-02.md)（<https://mp.weixin.qq.com/s/hbz9VPNH84CUtqORychPeA>）；raw [`sources/raw/wechat_shenlan_scirobotics_china_top3_2026-07-02/`](sources/raw/wechat_shenlan_scirobotics_china_top3_2026-07-02/)
- 抓取：Agent Reach v1.5.0 + wechat-article-for-ai（Camoufox）
- 沉淀页面：
  - [`wiki/entities/paper-swarm-micro-flying-robots-in-the-wild.md`](wiki/entities/paper-swarm-micro-flying-robots-in-the-wild.md)
  - [`wiki/entities/paper-bistable-soft-jumper-magnetic.md`](wiki/entities/paper-bistable-soft-jumper-magnetic.md)
  - [`wiki/entities/paper-microsaccade-inspired-event-camera.md`](wiki/entities/paper-microsaccade-inspired-event-camera.md)
  - [`wiki/entities/paper-octopus-inspired-esoam-soft-arm.md`](wiki/entities/paper-octopus-inspired-esoam-soft-arm.md)
  - [`wiki/entities/paper-aerial-aquatic-remora-hitchhiking-robot.md`](wiki/entities/paper-aerial-aquatic-remora-hitchhiking-robot.md)
  - [`wiki/entities/paper-miniature-deep-sea-morphable-robot.md`](wiki/entities/paper-miniature-deep-sea-morphable-robot.md)
  - [`wiki/entities/paper-subcentimeter-pipeline-inspection-robot.md`](wiki/entities/paper-subcentimeter-pipeline-inspection-robot.md)
  - [`wiki/entities/paper-tianjicx-neuromorphic-chip-robots.md`](wiki/entities/paper-tianjicx-neuromorphic-chip-robots.md)
  - [`wiki/entities/paper-neurogpr-brain-inspired-place-recognition.md`](wiki/entities/paper-neurogpr-brain-inspired-place-recognition.md)
- 开源要点：AMI-EV→Zenodo 全套；蜂群规划关联 EGO-Planner；北航/清华多篇硬件论文截至入库日无训练栈公开仓

## [2026-07-20] ingest | sources/blogs/wechat_qbitai_rss2026_awards_2026-07-16.md — 量子位 RSS 2026 三项最佳/Final List/ToT；新建 7 实体 + 交叉 3 已有大奖页

- 原始资料：[`sources/blogs/wechat_qbitai_rss2026_awards_2026-07-16.md`](sources/blogs/wechat_qbitai_rss2026_awards_2026-07-16.md)（<https://mp.weixin.qq.com/s/M3gYuB1gB2c3XL1GMk-p-A>）；raw [`sources/raw/wechat_qbitai_rss2026_awards_2026-07-16/`](sources/raw/wechat_qbitai_rss2026_awards_2026-07-16/)
- 抓取：Agent Reach v1.5.0 + wechat-article-for-ai（Camoufox）
- **复用已有完整节点（不重复造页）：** [`wiki/methods/flashsac.md`](wiki/methods/flashsac.md)（最佳论文）、[`wiki/entities/paper-muninn-trajectory-diffusion-acceleration.md`](wiki/entities/paper-muninn-trajectory-diffusion-acceleration.md)（最佳学生）、[`wiki/entities/paper-neuralactuator-neural-actuation-modeling.md`](wiki/entities/paper-neuralactuator-neural-actuation-modeling.md)（最佳系统）
- **新建：**
  - [`wiki/entities/paper-automated-facial-mechanisms-animatronic.md`](wiki/entities/paper-automated-facial-mechanisms-animatronic.md)
  - [`wiki/entities/paper-oat-ordered-action-tokenization.md`](wiki/entities/paper-oat-ordered-action-tokenization.md)（官方仓 Chaoqi-LIU/oat）
  - [`wiki/entities/paper-dapl-extrinsic-dexterity-clutter.md`](wiki/entities/paper-dapl-extrinsic-dexterity-clutter.md)
  - [`wiki/entities/paper-cunrto-gpu-robust-trajectory-optimization.md`](wiki/entities/paper-cunrto-gpu-robust-trajectory-optimization.md)
  - [`wiki/entities/paper-unified-fluid-robot-multiphysics-swimming.md`](wiki/entities/paper-unified-fluid-robot-multiphysics-swimming.md)（Aquarium.jl）
  - [`wiki/entities/euroc-mav-datasets.md`](wiki/entities/euroc-mav-datasets.md)（IJRR ToT）
  - [`wiki/entities/paper-deimel-compliant-underactuated-robotic-hand.md`](wiki/entities/paper-deimel-compliant-underactuated-robotic-hand.md)（RSS ToT）

## [2026-07-20] ingest | sources/papers/miniature_deep_sea_morphable_scirobotics_2025.md — 北航文力/丁希仑组深海软体可变形机器人；wiki/entities/paper-miniature-deep-sea-morphable-robot.md；交叉 locomotion / paper-octopus-inspired-esoam-soft-arm / paper-aerial-aquatic-remora-hitchhiking-robot

- 原始资料：[`sources/papers/miniature_deep_sea_morphable_scirobotics_2025.md`](sources/papers/miniature_deep_sea_morphable_scirobotics_2025.md)
- 沉淀页面：[`wiki/entities/paper-miniature-deep-sea-morphable-robot.md`](wiki/entities/paper-miniature-deep-sea-morphable-robot.md)
- 交叉更新：[`wiki/tasks/locomotion.md`](wiki/tasks/locomotion.md)
- 开源状态：截至 2026-07-20 **未开源**（无官方代码仓库）

## [2026-07-20] ingest | sources/papers/aerial_aquatic_remora_scirobotics_2022.md — 北航文力组仿印鱼两栖搭便车多旋翼；wiki/entities/paper-aerial-aquatic-remora-hitchhiking-robot.md；交叉 locomotion / crazyswarm2 / paper-octopus-inspired-esoam-soft-arm

- 原始资料：[`sources/papers/aerial_aquatic_remora_scirobotics_2022.md`](sources/papers/aerial_aquatic_remora_scirobotics_2022.md)
- 沉淀页面：[`wiki/entities/paper-aerial-aquatic-remora-hitchhiking-robot.md`](wiki/entities/paper-aerial-aquatic-remora-hitchhiking-robot.md)
- 交叉更新：[`wiki/tasks/locomotion.md`](wiki/tasks/locomotion.md)、[`wiki/entities/crazyswarm2.md`](wiki/entities/crazyswarm2.md)
- 开源状态：截至 2026-07-20 **未开源**（无官方代码仓库）

## [2026-07-20] ingest | sources/papers/octopus_inspired_esoam_scirobotics_2023.md — 北航文力组仿章鱼传感软臂 E-SOAM；wiki/entities/paper-octopus-inspired-esoam-soft-arm.md；交叉 manipulation / teleoperation / bimanual-manipulation

- 原始资料：[`sources/papers/octopus_inspired_esoam_scirobotics_2023.md`](sources/papers/octopus_inspired_esoam_scirobotics_2023.md)
- 沉淀页面：[`wiki/entities/paper-octopus-inspired-esoam-soft-arm.md`](wiki/entities/paper-octopus-inspired-esoam-soft-arm.md)
- 交叉更新：[`wiki/tasks/manipulation.md`](wiki/tasks/manipulation.md)、[`wiki/tasks/teleoperation.md`](wiki/tasks/teleoperation.md)、[`wiki/tasks/bimanual-manipulation.md`](wiki/tasks/bimanual-manipulation.md)
- 开源状态：截至 2026-07-20 **未开源**（无官方代码仓库）

## [2026-07-20] ingest | sources/papers/neurogpr_scirobotics_2023.md — NeuroGPR 脑启发多模态 ANN+SNN 场所识别；wiki/entities/paper-neurogpr-brain-inspired-place-recognition.md；交叉 paper-tianjicx / quadruped-robot / locomotion

- 原始资料：[`sources/papers/neurogpr_scirobotics_2023.md`](sources/papers/neurogpr_scirobotics_2023.md)
- 沉淀页面：[`wiki/entities/paper-neurogpr-brain-inspired-place-recognition.md`](wiki/entities/paper-neurogpr-brain-inspired-place-recognition.md)
- 机构：清华大学类脑计算研究中心；通讯作者施路平
- DOI：10.1126/scirobotics.abm6996（Science Robotics 2023）
- 关键贡献：NeuroGPR；视觉 + LiDAR 多模态；ANN 语义分支 + SNN 时序分支；对光照/视角/遮挡鲁棒；四足机器人实机部署
- 交叉更新：[`wiki/entities/paper-tianjicx-neuromorphic-chip-robots.md`](wiki/entities/paper-tianjicx-neuromorphic-chip-robots.md)（mutual link）、[`wiki/entities/quadruped-robot.md`](wiki/entities/quadruped-robot.md)、[`wiki/tasks/locomotion.md`](wiki/tasks/locomotion.md)
- 开源状态：确认未开源（截至 2026-07-20 无 GitHub）

## [2026-07-20] ingest | sources/papers/tianjicx_neuromorphic_scirobotics_2022.md — TianjicX 时空弹性神经形态芯片多任务机器人；wiki/entities/paper-tianjicx-neuromorphic-chip-robots.md；交叉 paper-neurogpr / locomotion

- 原始资料：[`sources/papers/tianjicx_neuromorphic_scirobotics_2022.md`](sources/papers/tianjicx_neuromorphic_scirobotics_2022.md)
- 沉淀页面：[`wiki/entities/paper-tianjicx-neuromorphic-chip-robots.md`](wiki/entities/paper-tianjicx-neuromorphic-chip-robots.md)
- 机构：清华大学类脑计算研究中心；通讯作者施路平
- DOI：10.1126/scirobotics.abk2948（Science Robotics 2022）
- 关键贡献：TianjicX 神经形态芯片；时空弹性 ANN+SNN 调度；轮式机器人三任务并发（跟踪/避障/语音识别）；低功耗边缘计算
- 交叉更新：[`wiki/entities/paper-neurogpr-brain-inspired-place-recognition.md`](wiki/entities/paper-neurogpr-brain-inspired-place-recognition.md)、[`wiki/tasks/locomotion.md`](wiki/tasks/locomotion.md)
- 开源状态：确认未开源（截至 2026-07-20 无 GitHub）

## [2026-07-20] ingest | sources/papers/subcentimeter_pipeline_inspection_scirobotics_2022.md — 亚厘米级 DEA 蠕动管道检测机器人；wiki/entities/paper-subcentimeter-pipeline-inspection-robot.md；交叉 locomotion / teleoperation

- 原始资料：[`sources/papers/subcentimeter_pipeline_inspection_scirobotics_2022.md`](sources/papers/subcentimeter_pipeline_inspection_scirobotics_2022.md)
- 沉淀页面：[`wiki/entities/paper-subcentimeter-pipeline-inspection-robot.md`](wiki/entities/paper-subcentimeter-pipeline-inspection-robot.md)
- 机构：清华大学机械工程系；通讯作者赵慧婵
- DOI：10.1126/scirobotics.abm8597（Science Robotics 2022）
- 关键贡献：2.2 g / 47 mm 蠕虫式软体机器人；DEA 人工肌肉三段蠕动；< 1 cm 管道穿行；直/L/S/螺旋管 + 玻璃/金属/碳纤维壁 + 空气/油液介质；> 1 体长/秒；微摄像头遥控内窥
- 交叉更新：[`wiki/tasks/locomotion.md`](wiki/tasks/locomotion.md)、[`wiki/tasks/teleoperation.md`](wiki/tasks/teleoperation.md)
- 开源状态：确认未开源（截至 2026-07-20 无 GitHub 或 CAD）

## [2026-07-20] structural | schema/ingest-workflow.md + page-types.md + AGENTS.md — 有源码论文 ingest 必加源码运行时序图；ClothTransformer 页标注不适用

## [2026-07-20] ingest | sources/papers/clothtransformer_arxiv_2605_27852.md — ClothTransformer 统一 latent Transformer 布料仿真；wiki/entities/paper-clothtransformer-unified-latent-cloth-simulation.md；交叉 manipulation / deform360

## [2026-07-20] ingest | sources/repos/unitree.md — 深度补全 unitreerobotics 组织开源地图；升级 wiki/entities/unitree.md 软件生态；交叉 unitree-g1 / unitree-rl-mjlab / unitree-ros / teleoperation / lerobot

- 原始资料：深化 [`sources/repos/unitree.md`](sources/repos/unitree.md)（组织级；约 49 公开仓；SDK2 / ROS / 三条 RL / XR+IL / UnifoLM 分类导航）
- 沉淀页面：升级 [`wiki/entities/unitree.md`](wiki/entities/unitree.md)（官方开源软件生态 + Mermaid 研发栈图 + 选型提示）
- 交叉更新：[`wiki/entities/unitree-g1.md`](wiki/entities/unitree-g1.md)、[`wiki/entities/unitree-rl-mjlab.md`](wiki/entities/unitree-rl-mjlab.md)、[`wiki/entities/unitree-ros.md`](wiki/entities/unitree-ros.md)、[`wiki/tasks/teleoperation.md`](wiki/tasks/teleoperation.md)、[`wiki/entities/lerobot.md`](wiki/entities/lerobot.md)、[`sources/README.md`](sources/README.md)
- 开源状态：组织仓已开源；`unitree_model` GitHub deprecated → HF；UnifoLM VLA/WMA 训练/推理/权重已发布
- 未新建子实体页（留待后续单独 ingest）：`unitree_rl_gym`、`unitree_rl_lab`、`unitree_sdk2`、`xr_teleoperate`、`unifolm-vla`、`unifolm-world-model-action`

## [2026-07-20] structural | wiki/methods/sonic-motion-tracking.md — 读源码导航：模块边界/文件树/算法↔代码映射/FSQ数据流；交叉 gr00t-wholebodycontrol

## [2026-07-20] ingest | sources/repos/mediapipe.md + sources/repos/gnm.md — Google MediaPipe 与 GNM Head 官方仓库；wiki/entities/mediapipe.md、wiki/entities/gnm-head.md；交叉 dexterous-data-collection-guide / midas-hand / sam-3d-body

## [2026-07-20] ingest | sources/papers/roller_skating_amp_arxiv_2607_10815.md + vision_dribbling + semantic_audio_wbc — 三篇 arXiv:2607 论文入库；wiki/entities/paper-roller-skating-amp-humanoid-passive-wheels.md、paper-vision-dribbling-humanoid-soccer-privileged-representation.md、paper-semantic-audio-wbc-humanoid.md；交叉 humanoid-soccer

## [2026-07-20] structural | Top-50 论文枢纽有代码缺图补齐 — 19 页源码运行时序图 + RPL/PILOT 开源状态

- 范围：`exports/hub-rankings.json` paper 榜 Top-50 中「有官方代码信号且缺 mermaid 时序图」的节点
- 新增时序图（19）：ASE、BFM-Zero、OASIS、VIRAL、SPIDER（`facebookresearch/spider`）、DoorMan、GentleHumanoid、MoRE、TWIST、GVHMR、ResMimic、T-Rex、PHC、GR00T N1、KungFuAthleteBot、MPC-RL、KungfuBot/PBHC、Humanoid-GPT、WEM
- 开源状态不适用（2）：RPL、PILOT（WholebodyVLA 仅为索引仓，无独立可运行官方实现）
- 链接校正：SPIDER 官方仓改为 facebookresearch/spider；GentleHumanoid 训练/部署双仓写入 frontmatter `code`

## [2026-07-20] structural | 论文链接 Top-10 枢纽再核官方源码 — SONIC 补 GR00T-WholeBodyControl 源码运行时序图；BFM / MotionWAM / LEGS / SD-AMP / PHP 标注未开源

- 触发：首页 `top_paper_hubs` 前十论文节点源码地址再确认；纠正此前「SONIC 官方源码未发布」误判
- `wiki/methods/sonic-motion-tracking.md`：写入官方代码 / HF 权重 / BONES-SEED；新增「源码运行时序图」（`gear_sonic` 训练 → `eval_agent_trl` → `gear_sonic_deploy`）
- `sources/repos/sonic-humanoid-motion-tracking.md`：元数据补代码 / 权重 / 数据 / 文档站与工程入口表
- `wiki/entities/paper-behavior-foundation-model-humanoid.md`、`paper-motionwam-…`、`paper-legs-…`、`paper-unified-walk-run-recovery-sdamp.md`、`paper-hrl-stack-22-perceptive_humanoid_parkour.md`：各加「开源状态（2026-07-20）」——均无官方可运行仓（SD-AMP 明确 AMP_mjlab 仅为工程对照）
- 已有时序图且源码仍有效：BeyondMimic（`whole_body_tracking`）、OmniRetarget（`holosoma`）、DeepMimic（`xbpeng/DeepMimic`）、TWIST2（`amazon-far/TWIST2`）HTTP 200 复核

## [2026-07-20] ingest | sources/papers/handroid_arxiv_2607_16187.md — Handroid 双形态桌面机器人；wiki/entities/handroid.md；交叉 manipulation / loco-manipulation / teleoperation

## [2026-07-20] ingest | sources/papers/midas_hand_arxiv_2607_14487.md — MIDAS Hand UCLA 开源直驱触觉灵巧手；wiki/entities/midas-hand.md；交叉 dexterous-data-collection-guide / ruka-v2-hand

## [2026-07-20] structural | wiki 链接 Top-10 论文节点新增「源码运行时序图」模块 — beyondmimic（whole_body_tracking）/ paper-hrl-stack-03-omniretarget（holosoma）/ paper-twist2（amazon-far/TWIST2）/ deepmimic（xbpeng/DeepMimic）四页各加 mermaid sequenceDiagram；其余 5 个 Top-10 论文节点（BFM / MotionWAM / LEGS / SD-AMP / PHP）官方源码未发布，不适用；**SONIC 已开源（NVlabs/GR00T-WholeBodyControl），时序图见同日后续 structural 条目**

## [2026-07-20] ingest | sources/papers/actuator_constrained_rl_high_speed_quadruped_arxiv_2312_17507.md — 执行器约束 RL 高速四足 MOR；wiki/entities/paper-actuator-constrained-rl-high-speed-quadruped-locomotion.md；交叉 sim2real / locomotion / APT-RL / hub-sim2real

## [2026-07-20] ingest | sources/sites/chingmu.md — 青瞳视觉光学动捕全栈与 MotionDecode 数据计划；wiki/entities/chingmu.md；交叉更新 notable-commercial-robot-platforms

## [2026-07-20] ingest | sources/sites/engineai-urkl.md — EngineAI URKL 人形格斗联赛；wiki/entities/urkl.md；交叉 rek / teleoperation / depth-humanoid-boxing

## [2026-07-19] ingest | sources/sites/roboscience.md — RoboScience VLOA/Visics/RoboMirage；wiki/entities/roboscience-vloa.md；交叉更新 generative-world-models / vla / manipulation

## [2026-07-19] ingest | sources/papers/humi_arxiv_2602_06643.md — HuMI 无机器人人形全身操作；wiki/entities/paper-notebook-humanoid-manipulation-interface.md；交叉更新 wiki/entities/paper-halomi-humanoid-loco-manipulation.md

## [2026-07-19] ingest | sources/papers/apt_rl_science_robotics_2026.md — APT-RL Science Robotics 2026 封面；wiki/entities/paper-apt-rl-agile-perceptive-quadruped-locomotion.md

## [2026-07-19] ingest | sources/blogs/kimi_k3_tech_blog.md + sources/courses/kimi_k3_api_quickstart.md — Kimi K3 技术博客与 API 接入；wiki/entities/kimi-k3.md；交叉更新 muon / enpire

## [2026-07-19] ingest | sources/repos/handumi-sw.md — HandUMI 无机器人双臂示教软件；wiki/entities/handumi.md；交叉更新 teleoperation / bimanual-manipulation / lerobot

## [2026-07-19] structural | roadmap/depth-sim2real.md — 纵深路线扩容至十七条：新增 Sim2Real 纵深

- roadmap/depth-sim2real.md：新建「Sim2Real（域差画像 → 执行器对齐 → 鲁棒训练 → 真机部署）」纵深路线（起点：域随机化 DR，2017），Stage 0 域差六类画像与方法三分（随机化 / 对齐 / 适应）→ Stage 1 资产与执行器对齐（SysID / ActuatorNet / BAM / PACE / SAGE / 并联闭链）→ Stage 2 训练期鲁棒化（DR / 课程 / 特权蒸馏 / RMA / 电机包络约束）→ Stage 3 感知与视觉迁移（视觉 DR / RGB 蒸馏 / VFM 替代 / 合成深度预训练）→ Stage 4 部署工程闭环（sim2sim 回归 / 处理器在环 / ONNX / 渐进 SOP）→ Stage 5 真机安全微调 / Real2Sim / 评测基础设施 / 跨具身汇合；链接 wiki/concepts/sim2real.md、sim2real-approaches 对比页、sim2real-checklist 等枢纽页
- roadmap/README.md、README.md、index.md、roadmap/motion-control.md、docs/index.html、docs/main.js 与其余十六条 depth-* 路线页：纵深路线数 16 → 17，按里程碑历史序插入（Sim2Real 2017 列动作重定向后、人形拳击前）并补双向链接；首页「更多路线」按钮扩为十七个
- catalog.md 与 graph/home 统计、README badge 由 make ci-preflight 同步再生成

## [2026-07-18] structural | docs/topic-filters.js + docs/graph.html + wiki/overview/topic-embodied-eval-benchmark.md — V29 P3① 图谱页「具身评测基准」专题视图落地

- docs/topic-filters.js：单一事实源新增第 19 个专题 `embodied-eval-benchmark`（🧪）——`TOPIC_HUB_IDS` 挂枢纽页、`TOPIC_FILTERS` 用干净片段 `bench`/`eval`/`benchmark` 并集 + 7 页 `ids` 显式纳入（robo-bench/esi-bench/ewmbench/gigaworld-1/simulation-evaluation-infrastructure/选型闭环 Query/sim-vs-real-eval-gap）、`TOPIC_META` 补 emoji+简称+导读；与 sim2real/physics-fidelity 保持最小重叠
- docs/graph.html：`#filter-topic-chips` 增加对应 chip（第 19 个专题）
- wiki/overview/topic-embodied-eval-benchmark.md：补建专题汇总枢纽页（英文缩写速查 + 四层评测选型表 + 关键取舍），从 queries/embodied-eval-benchmark-selection-loop.md、concepts/sim-vs-real-eval-gap.md 双向回链
- node 端复核 7 个目标评测页 + 枢纽命中、vla.md 未命中；export+graph 重生 1682 节点/13396 边、0 orphans、largest_community_ratio 0.17 且 community_quality_warning=false；lint_wiki 0 errors 0 信息型预警

## [2026-07-18] structural | roadmap/depth-humanoid-soccer.md + depth-humanoid-boxing.md — 纵深路线扩容至十六条：新增人形足球与人形拳击两条竞技纵深

- roadmap/depth-humanoid-soccer.md：新建「人形足球（全向行走 → 感知踢球 → 多机战术）」纵深路线（起点：首届 RoboCup，1997），Stage 0 RoboCup 全景与感知–决策–运动耦合 → Stage 1 参数化全向行走与跌倒恢复 → Stage 2 机载感知与自定位 → Stage 3 追球–对齐–射门课程 RL（PAiD / RoboNaldo / 教师–学生蒸馏）→ Stage 4 多机战术群控 → Stage 5 整队系统 / 端到端 / 体育谱系；链接 wiki/tasks/humanoid-soccer.md、humanoid-soccer-skill-learning-method-selection、humanoid-multi-robot-coordination 等枢纽页
- roadmap/depth-humanoid-boxing.md：新建「人形拳击（动作跟踪 → 潜空间技能 → 对抗自博弈）」纵深路线（起点：MuJoCo 人形对抗自博弈，2017），Stage 0 对抗全身任务与自主 / 遥操作两条路线（RoboStriker vs REK）→ Stage 1 拳击 MoCap 全身跟踪基座 → Stage 2 技能蒸馏潜空间动作流形 → Stage 3 两玩家零和博弈与 LS-NFSP → Stage 4 高冲击真机安全（SafeFall）→ Stage 5 类人度评测 / 体育谱系 / 赛事产业；链接 paper-notebook-robostriker、rek、smplolympics、topic-wbt 等枢纽页
- roadmap/README.md、README.md、index.md、roadmap/motion-control.md、docs/index.html、docs/main.js 与其余十四条 depth-* 路线页：纵深路线数 14 → 16，按里程碑历史序插入（人形足球 1997 列移动操作后、人形拳击 2017 列动作重定向后）并补双向链接；首页「更多路线」按钮扩为十六个
- catalog.md、exports/graph-stats.json、exports/home-stats.json 及 docs/exports/ 同步再生成；README 知识图谱 badge 更新至 1681节点/13385边（含合并 main 的 KiCad ingest）；make lint、make ci-preflight 12/12、pytest 320 passed 全部通过

## [2026-07-18] structural | scripts/utils/community_labels.py — 补 roadmap/depth-perceptive-locomotion.md 社区名 override，修复 pytest 社区命名测试

- scripts/utils/community_labels.py

## [2026-07-18] ingest | sources/sites/kicad-org.md + sources/courses/kicad_docs_10_zh.md + sources/repos/kicad.md — 接入 KiCad 开源 EDA；wiki/entities/kicad.md；交叉更新 depth-torque-motor-design Stage 4、humanoid-hardware-101-power-compute-electronics、simplefoc

- sources/sites/kicad-org.md
- sources/courses/kicad_docs_10_zh.md
- sources/repos/kicad.md
- wiki/entities/kicad.md
- wiki/overview/humanoid-hardware-101-power-compute-electronics.md
- wiki/entities/simplefoc.md
- roadmap/depth-torque-motor-design.md
- schema/institutions.json

## [2026-07-18] ingest | sources/sites/altium-designer-primary-refs.md — Altium Designer 官方文档一手资料；升格 wiki/entities/altium-designer.md

- schema/institutions.json（注册 Altium 机构 alias）
- sources/sites/altium-designer-primary-refs.md
- wiki/entities/altium-designer.md
- roadmap/depth-torque-motor-design.md（Stage 4 推荐读什么补充 Altium 入口）

## [2026-07-18] structural | roadmap/depth-torque-motor-design.md — 力矩电机设计纵深新增 Stage 4 电机驱动 PCB 设计，Stage 0–5 扩为 Stage 0–6

- roadmap/depth-torque-motor-design.md：在 Stage 3（驱动硬件与电流环）与原 Stage 4（FOC 力矩闭环标定）之间插入「Stage 4 电机驱动 PCB 设计：把电流环装进自己的板子」——功率级选型 / 电流采样链路落板 / 布局散热 / 分步 bring-up，原 Stage 4/5 顺延为 Stage 5/6；摘要、Mermaid 路线图、快速入口表同步
- wiki/overview/motor-design-workflow.md、wiki/overview/humanoid-actuator-102-technology-map.md、wiki/concepts/field-oriented-control.md：入口文案 Stage 0–5 → Stage 0–6

## [2026-07-18] ingest | sources/blogs/wechat_shenlan_robot_control_eight_paradigms.md — 深蓝八大机器人控制体系；taxonomy + 8 体系 overview + 19 代表算法 method 页

- sources/blogs/wechat_shenlan_robot_control_eight_paradigms.md
- sources/raw/wechat_shenlan_robot_control_eight_paradigms_2026-07-18.md
- wiki/comparisons/robot-control-eight-paradigms-taxonomy.md
- wiki/overview/robot-control-paradigm-classical-linear-feedback.md
- wiki/overview/robot-control-paradigm-model-based-nonlinear-dynamics.md
- wiki/overview/robot-control-paradigm-robust-control.md
- wiki/overview/robot-control-paradigm-adaptive-control.md
- wiki/overview/robot-control-paradigm-hybrid-position-force.md
- wiki/overview/robot-control-paradigm-receding-horizon-ilc.md
- wiki/overview/robot-control-paradigm-ml-driven-control.md
- wiki/overview/robot-control-paradigm-rl-intelligent-control.md
- wiki/methods/pole-placement-control.md
- wiki/methods/computed-torque-control.md
- wiki/methods/inverse-dynamics-control.md
- wiki/methods/feedback-linearization-control.md
- wiki/methods/sliding-mode-control.md
- wiki/methods/h-infinity-control.md
- wiki/methods/mu-synthesis-control.md
- wiki/methods/mrac.md
- wiki/methods/adaptive-computed-torque-control.md
- wiki/methods/recursive-least-squares-control.md
- wiki/methods/admittance-control.md
- wiki/methods/direct-force-feedback-control.md
- wiki/methods/iterative-learning-control.md
- wiki/methods/neural-network-compensation-control.md
- wiki/methods/gaussian-process-control.md
- wiki/methods/fuzzy-logic-control.md
- wiki/methods/unsupervised-clustering-fault-compensation.md
- wiki/methods/value-based-reinforcement-learning.md
- wiki/methods/hierarchical-reinforcement-learning.md

## [2026-07-18] ingest | sources/papers/scaling_bfm_arxiv_2607_15163.md — ScaleBFM 三轴 scaling 配方与 Humanoid Transformer；升格 wiki/entities/paper-scaling-bfm-humanoid.md

- wiki/entities/paper-scaling-bfm-humanoid.md
- wiki/entities/paper-behavior-foundation-model-humanoid.md
- wiki/concepts/behavior-foundation-model.md
- wiki/methods/sonic-motion-tracking.md
- wiki/entities/paper-reactivebfm.md
- sources/papers/scaling_bfm_arxiv_2607_15163.md
- sources/sites/scalebfm-github-io.md
- sources/repos/scalebfm.md

## [2026-07-17] structural | schema/canonical-facts.json 230 → 240 条 — V29 P2 补 10 条具身评测选型矛盾检测规则

- schema/canonical-facts.json（新增 10 条：仿真可复现 vs 真机代表性、任务成功率 vs 过程/中间指标、世界模型视频质量 ≠ 策略收益、MLLM 认知评分 ≠ 可执行动作、单任务过拟合 vs 跨任务泛化、离线回放 vs 在线闭环、成功率均值掩盖长尾、基准饱和 ≠ 场景就绪、评测集泄漏致虚高、静态基准不覆盖分布漂移）
- 校验：每条 `pos_claims` 锚定 `queries/embodied-eval-benchmark-selection-loop.md` / `concepts/sim-vs-real-eval-gap.md` 现存正文，pos 命中 ≥1 页；`neg_claims` 经全量 wiki 页复核 0 命中（0 误报）
- `lint_wiki.py --report` 0 errors、潜在矛盾 0 个（信息型预警仍 22 条）；`ci-preflight` 12/12 通过
- docs/checklists/tech-stack-next-phase-checklist-v29.md（P2 打勾 + DoD 事实库 240 条打勾）

## [2026-07-17] ingest | sources/blogs/sunday_act2_preview.md — Sunday ACT-2 预览：Solve 叠衣 99.1%、泛化鸿沟缩放与单示范 SFT；升格 wiki/entities/sunday-robotics-act2.md、wiki/concepts/robotics-solve-standard.md

- wiki/entities/sunday-robotics-act2.md
- wiki/concepts/robotics-solve-standard.md
- wiki/tasks/manipulation.md
- wiki/overview/humanoid-hardware-101-sensing-end-effectors.md
- wiki/entities/tidybot2.md
- sources/blogs/sunday_act2_preview.md
- sources/sites/sunday-robotics.md
- schema/institutions.json（sunday-robotics）

## [2026-07-17] ingest | sources/blogs/mimicrobotics_m1_u1_full_stack.md — mimic hand M1 与 wearable U1 全栈灵巧平台发布

- wiki/entities/mimic-hand-m1.md
- wiki/entities/mimic-wearable-u1.md
- wiki/methods/mimic-video.md
- wiki/tasks/teleoperation.md
- wiki/tasks/manipulation.md
- wiki/queries/dexterous-data-collection-guide.md
- wiki/concepts/motion-retargeting.md
- sources/blogs/mimicrobotics_m1_u1_full_stack.md
- sources/sites/mimicrobotics.md
- sources/sites/mimicrobotics-smooth-operator.md
- sources/repos/mimicrobotics_mimic_retargeter_lab.md
- schema/institutions.json（mimic-robotics）

## [2026-07-17] ingest | sources/sites/lejurobot.md、openlet-openatom.md、lingbot-vla.md、rhino-auto.md — 乐聚/OpenLET、LingBot-VLA 1.0、辉羲智能入库

- wiki/entities/leju-robotics.md
- wiki/entities/openlet.md
- wiki/entities/lingbot-vla.md
- wiki/entities/rhino-auto.md
- wiki/methods/vla.md
- wiki/entities/lingbot-vla-v2.md
- sources/sites/lejurobot.md
- sources/sites/openlet-openatom.md
- sources/sites/rhino-auto.md
- sources/sites/lingbot-vla-technology-robbant.md
- sources/repos/lingbot-vla.md
- sources/repos/openlet-let-base-dataset.md
- sources/papers/lingbot_vla_arxiv_2601_18692.md
- sources/papers/loco_manip_161_survey_152_lingbot-vla.md（修正 1.0 误链至 2.0）
- schema/institutions.json（openatom、rhino-auto）

## [2026-07-17] ingest | sources/papers/fmpose3d_arxiv_2602_05755.md — FMPose3D Flow Matching 单目 3D 姿态；wiki/entities/paper-fmpose3d-monocular-3d-pose-flow-matching.md；交叉更新 motion-retargeting-pipeline

- wiki/entities/paper-fmpose3d-monocular-3d-pose-flow-matching.md
- wiki/concepts/motion-retargeting-pipeline.md
- sources/papers/fmpose3d_arxiv_2602_05755.md
- sources/sites/fmpose3d-xiu-cs-github-io.md
- sources/repos/fmpose3d.md

## [2026-07-16] lint | V29 P0 评测基准页交叉链路巡检 V1 — scripts/lint_wiki.py 新增 `_check_eval_benchmark_crosslink`（INFO 级，不阻塞 CI）

- scripts/lint_wiki.py（新增 `EVAL_BENCHMARK_HUBS` / `EVAL_BENCHMARK_TAG_KEYWORDS` / `_check_eval_benchmark_crosslink`；接入 `INFO_ONLY_KEYS`、`_empty_results`、runner 与报告段）
- tests/test_lint_wiki_eval_benchmark_crosslink.py（9 条用例：entities/comparisons/concepts 三目录、列表式/内联式 tag、有/无回链、双枢纽、枢纽豁免、INFO 不计失败）
- exports/lint-report.md（新增基线段：benchmark/evaluation 页缺回链专题枢纽 20 页）

## [2026-07-16] ingest | sources/papers/touchworld_arxiv_2607_07287.md — TouchWorld 预测–反应式触觉基础模型；wiki/entities/paper-touchworld-tactile-foundation-dexterous-manipulation.md；交叉更新 visuo-tactile-fusion / contact-rich-manipulation / T-Rex

- wiki/entities/paper-touchworld-tactile-foundation-dexterous-manipulation.md
- wiki/concepts/visuo-tactile-fusion.md
- wiki/concepts/contact-rich-manipulation.md
- wiki/entities/paper-trex-tactile-reactive-dexterous-manipulation.md
- sources/papers/touchworld_arxiv_2607_07287.md
- sources/sites/touchworld-phanes-lab.md

## [2026-07-16] ingest | sources/papers/simple_arxiv_2606_08278.md — SIMPLE 人形 loco-manip 仿真 testbed 深读；升格 wiki/entities/paper-loco-manip-161-075-simple.md

- wiki/entities/paper-loco-manip-161-075-simple.md

## [2026-07-16] ingest | sources/repos/flashsac.md — 补全 FlashSAC 官方 GitHub 仓库链接与训练框架要点

- wiki/methods/flashsac.md
- wiki/comparisons/ppo-vs-sac.md
- sources/repos/flashsac.md
- sources/papers/flashsac_arxiv_2604_04539.md
- sources/sites/flashsac-project.md

## [2026-07-16] ingest | sources/papers/muninn_arxiv_2605_09999.md + sources/repos/muninn.md — Muninn 轨迹扩散免训练缓存加速（RSS 2026）；wiki/entities/paper-muninn-trajectory-diffusion-acceleration.md

- wiki/entities/paper-muninn-trajectory-diffusion-acceleration.md
- wiki/methods/diffusion-policy.md
- wiki/methods/diffusion-motion-generation.md
- sources/papers/muninn_arxiv_2605_09999.md
- sources/repos/muninn.md

## [2026-07-16] ingest | sources/papers/neuralactuator_arxiv_2607_11734.md — 新增 NeuralActuator 论文实体；交叉更新 actuator-network / BAM

- wiki/entities/paper-neuralactuator-neural-actuation-modeling.md
- wiki/methods/actuator-network.md
- wiki/entities/paper-bam-extended-friction-servo-actuators.md
- sources/papers/neuralactuator_arxiv_2607_11734.md

## [2026-07-16] ingest | sources/papers/flashsac_arxiv_2604_04539.md + sources/sites/flashsac-project.md — FlashSAC 高维 off-policy RL；wiki/methods/flashsac.md 与 SAC/PPO 对比、locomotion 交叉更新

- wiki/entities/paper-notebook-learning-sim-to-real-humanoid-locomotion-in-15-m.md
- wiki/methods/flashsac.md
- wiki/methods/sac.md
- wiki/methods/ppo.md
- wiki/methods/policy-optimization.md
- wiki/comparisons/ppo-vs-sac.md
- wiki/tasks/locomotion.md
- wiki/entities/unilab.md
- sources/papers/flashsac_arxiv_2604_04539.md
- sources/sites/flashsac-project.md
- sources/papers/unilab_arxiv_2605_30313.md

## [2026-07-16] structural | wiki — 重复节点合并第二批（12 组 A 类）+ C 类互链补全 + 深读页模板措辞修正

延续同日第一批：BeyondMimic / DeepMimic / GentleHumanoid / HAIC / ASE / ADD / SMP / MotionBricks / AMS（survey 槽位实体并入 methods 深读页）、ReActor / SPIDER（论文笔记 stub 并入 methods 页）、PHC（BFM 槽位并入代码库实体页）；旧 URL 全部登记 schema/page-aliases.json 重定向。核实并保留：Muon（论文页 vs 优化器方法页，合理分层）、PAiD ≠ RoboNaldo（并发不同工作）、OpenTrack / Retargeting Matters / Make Tracking Easy（methods 页为多工作技术页，不合并）。补全 project-instinct ↔ Embrace Collisions、gr00t-wholebodycontrol ↔ GR00T N1 互链；修正 10 处「深读页」模板误用（不同论文改「同主题深读」，通用方法页改「方法背景」）。

- wiki/methods/beyondmimic.md
- wiki/methods/deepmimic.md
- wiki/methods/gentlehumanoid-motion-tracking.md
- wiki/methods/haic.md
- wiki/methods/ase.md
- wiki/methods/add.md
- wiki/methods/smp.md
- wiki/methods/motionbricks.md
- wiki/methods/ams.md
- wiki/methods/reactor-physics-aware-motion-retargeting.md
- wiki/methods/spider-physics-informed-dexterous-retargeting.md
- wiki/entities/phc.md
- wiki/entities/project-instinct.md
- wiki/entities/gr00t-wholebodycontrol.md

## [2026-07-16] structural | wiki — 同论文重复节点合并（10 组）+ 站点 alias 重定向 + arxiv 唯一性 lint 护栏

一篇论文只保留一个 canonical 节点：合并 SONIC（entity+method 双页 → 方法页）、VIRAL / DoorMan / Heracles（42 栈槽位 → 深读实体）、SD-AMP / Kimodo（AMP 专题槽位 → 深读实体）、HOMIE / TextOp / Open-TeleVision（161 双槽位 → 单页双坐标）、Berkeley Humanoid Lite（论文笔记 stub → 硬件实体）。删除页旧 URL 经 schema/page-aliases.json + docs/main.js 重定向到合并后页面；lint 新增 frontmatter arxiv 全站唯一性检查（阻塞 CI）。AGILE（161 #056 = NVIDIA WBC-AGILE ≠ 智元 AGILE）确认为同名不同工作，不合并。

- wiki/methods/sonic-motion-tracking.md
- wiki/entities/paper-viral-humanoid-visual-sim2real.md
- wiki/entities/paper-doorman-opening-sim2real-door.md
- wiki/entities/paper-heracles-humanoid-diffusion.md
- wiki/entities/paper-unified-walk-run-recovery-sdamp.md
- wiki/entities/kimodo.md
- wiki/entities/paper-loco-manip-161-040-homie.md
- wiki/entities/paper-loco-manip-161-022-textop.md
- wiki/entities/paper-loco-manip-161-131-open-television.md
- wiki/entities/berkeley-humanoid-lite.md

## [2026-07-16] ingest | sources/papers/wam_ttt_arxiv_2607_06988.md — WAM-TTT 人视频测试时 steering；wiki/entities/paper-wam-ttt-human-video-test-time-steering.md；交叉 world-action-models / manipulation / EgoWAM / RoboTTT

## [2026-07-16] ingest | sources/papers/lumo2_arxiv_2607_11270.md + sources/papers/philia_arxiv_2607_11377.md — Astribot Lumo-2 latent WAM 与 Philia 多机器人助手；项目页全量演示视频归档

- wiki/entities/lumo-2.md
- wiki/entities/philia.md
- wiki/concepts/world-action-models.md
- wiki/methods/vla.md
- wiki/tasks/manipulation.md
- wiki/entities/hermes-agent.md
- sources/papers/lumo2_arxiv_2607_11270.md
- sources/papers/philia_arxiv_2607_11377.md
- sources/sites/astribot-lumo2-project-page.md
- sources/sites/astribot-philia-project-page.md
- schema/institutions.json
## [2026-07-16] ingest | sources/papers/robottt_nvidia_gear.md — RoboTTT：GR00T N1.7 内嵌 TTT fast weights，8K visuomotor 上下文与部署后在线学习

- wiki/entities/paper-robottt-test-time-training-vla-context.md
- wiki/methods/vla.md
- wiki/entities/paper-hrl-stack-34-gr00t_n1.md
- wiki/tasks/manipulation.md
- sources/papers/robottt_nvidia_gear.md
- sources/sites/nvidia-research-robottt.md
## [2026-07-16] ingest | sources/sites/xiaomi-robotics-1.md — Xiaomi-Robotics-1 100k h UMI 预训练具身基座 VLA；wiki/entities/xiaomi-robotics-1.md、wiki/methods/vla.md、wiki/entities/xiaomi-robotics-0.md、wiki/entities/xiaomi-robotics-u0.md

- wiki/entities/xiaomi-robotics-1.md
- wiki/methods/vla.md
- wiki/entities/xiaomi-robotics-0.md
- wiki/entities/xiaomi-robotics-u0.md
- sources/sites/xiaomi-robotics-1.md

## [2026-07-16] structural | wiki — 合并 REGRIND 方法页与论文实体页为单页 wiki/methods/regrind-retargeting-guided-rl.md

- wiki/methods/regrind-retargeting-guided-rl.md
- wiki/tasks/manipulation.md
- wiki/methods/toporetarget-interaction-preserving-dexterous-retargeting.md

## [2026-07-16] ingest | sources/papers/regrind_arxiv_2607_11874.md — REGRIND MoCap+交互保留重定向+残差 RL 灵巧工具操作；wiki/methods/regrind-retargeting-guided-rl.md、wiki/entities/paper-regrind-dexterous-manipulation.md

- wiki/methods/regrind-retargeting-guided-rl.md
- wiki/entities/paper-regrind-dexterous-manipulation.md
- wiki/methods/toporetarget-interaction-preserving-dexterous-retargeting.md
- wiki/methods/spider-physics-informed-dexterous-retargeting.md
- wiki/concepts/motion-retargeting-pipeline.md
- wiki/tasks/manipulation.md
- sources/papers/regrind_arxiv_2607_11874.md
- sources/sites/regrind-project-yunhaifeng.md
- sources/repos/regrind.md

## [2026-07-16] ingest | sources/blogs/limx_cosa_05_release_2026-07-15.md — LimX COSA 0.5 / FluxVLA Engine 实体页与 VLA、loco-manipulation 交叉更新

- wiki/entities/limx-cosa.md
- wiki/entities/fluxvla-engine.md
- wiki/methods/vla.md
- wiki/tasks/loco-manipulation.md
- sources/blogs/limx_cosa_05_release_2026-07-15.md

## [2026-07-16] ingest | sources/sites/legsvla-github-io.md — 刷新 LEGS 项目页摘录（1,110 trials、9/9、LEGS-AUG、~15× 成本）；wiki/entities/paper-legs-embodied-gaussian-splatting-vla.md

- wiki/entities/paper-legs-embodied-gaussian-splatting-vla.md
- sources/sites/legsvla-github-io.md
## [2026-07-16] ingest | sources/repos/dexmal_opendw.md — 接入 Dexmal OpenDW/DW05 开源世界模型与 DW05-Base 权重；新建 wiki/entities/dexmal-dw05.md；互链 WAM、DM0.5、τ0-WM、RoboTwin、manipulation

- wiki/entities/dexmal-dw05.md
- wiki/concepts/world-action-models.md
- wiki/entities/dexmal-dm05.md
- wiki/tasks/manipulation.md
- wiki/entities/robotwin.md
- sources/repos/dexmal_opendw.md
## [2026-07-16] ingest | sources/papers/co_calib_observation_quality_fisheye_arxiv_2607_05777.md — CO-Calib 多鱼眼标定 failure-oriented 分析；wiki/entities/paper-co-calib-multi-fisheye-calibration.md

- wiki/entities/paper-co-calib-multi-fisheye-calibration.md
- wiki/formalizations/3d-coordinate-transforms-vision-robotics.md
- wiki/entities/vins-fusion.md
- sources/papers/co_calib_observation_quality_fisheye_arxiv_2607_05777.md
- sources/repos/co_calib.md

## [2026-07-16] ingest | sources/repos/graphify-labs_graphify.md — Graphify 多模态知识图 Agent Skill；wiki/entities/graphify.md；交叉更新 llm-wiki-karpathy / superpowers / agent-reach / caveman

- wiki/entities/graphify.md
- wiki/references/llm-wiki-karpathy.md
- wiki/entities/superpowers-obra.md
- wiki/entities/agent-reach.md
- wiki/entities/caveman.md
- sources/repos/graphify-labs_graphify.md

## [2026-07-16] ingest | sources/papers/gaitspan_arxiv_2607_12114.md — GaitSpan 行走种子技能生长；wiki/entities/paper-gaitspan-humanoid-locomotion-walking-running.md、wiki/tasks/humanoid-locomotion.md

## [2026-07-15] structural | V29 P1「评测基准家族层专题交叉补强」——robo-bench / ewmbench / gigaworld-1 三页与「具身大模型评测基准选型闭环」Query 页双向回链，标注各自评测层

- wiki/entities/robo-bench.md（① 具身大脑/MLLM 认知评测层，related + 关联页面双向回链）
- wiki/entities/ewmbench.md（② 世界模型预测保真度评测层，related + 关联页面双向回链）
- wiki/entities/paper-gigaworld-1-policy-evaluation.md（② 世界模型作策略评估器层，related + 关联页面双向回链）
- 结果：`ci-preflight` 12/12 通过，graph 边数 12775→12778（+3），0 孤儿，`community_quality_warning: false`

## [2026-07-15] ingest | sources/papers/sru_spatially_enhanced_recurrent_memory_ijrr_2025.md、sources/repos/sru_odin.md — SRU 无地图循环导航（IJRR 2025）与 SRU-Odin Go2+Odin1 部署；wiki/entities/paper-sru-spatially-enhanced-recurrent-memory.md、wiki/entities/sru-odin.md；交叉更新 sim2real、vision-language-navigation

- wiki/entities/paper-sru-spatially-enhanced-recurrent-memory.md
- wiki/entities/sru-odin.md
- wiki/concepts/sim2real.md
- wiki/tasks/vision-language-navigation.md
- sources/papers/sru_spatially_enhanced_recurrent_memory_ijrr_2025.md
- sources/repos/sru_odin.md
## [2026-07-15] ingest | sources/papers/eventvla_arxiv_2606_20092.md — EventVLA 稀疏视觉证据记忆 VLA + RoboTwin-MeM；wiki/entities/paper-eventvla-visual-evidence-memory.md；交叉更新 vla / robotwin / KEMO

- wiki/entities/paper-eventvla-visual-evidence-memory.md
- wiki/methods/vla.md
- wiki/entities/robotwin.md
- wiki/entities/paper-kemo-event-driven-keyframe-memory-vla.md
- sources/papers/eventvla_arxiv_2606_20092.md

## [2026-07-15] ingest | sources/papers/panoworld_arxiv_2607_09661.md — PanoWorld 真实世界全景可控生成 WM + World360；wiki/entities/paper-panoworld-real-world-panoramic-generation.md；交叉更新 generative-world-models

- wiki/entities/paper-panoworld-real-world-panoramic-generation.md
- wiki/methods/generative-world-models.md
- sources/papers/panoworld_arxiv_2607_09661.md

## [2026-07-15] ingest | sources/papers/genception_arxiv_2607_09024.md — GenCeption ECCV 2026 视频生成统一视觉感知；wiki/entities/genception.md；交叉更新 generative-vision-pretraining / vision-banana

## [2026-07-15] ingest | sources/repos/xiaomi-robotics-u0.md、sources/papers/xiaomi_robotics_u0_arxiv_2607_11643.md — Xiaomi-Robotics-U0 38B 统一具身合成 WM；wiki/entities/xiaomi-robotics-u0.md；交叉更新 generative-world-models / xiaomi-robotics-0

- wiki/entities/xiaomi-robotics-u0.md
- wiki/entities/xiaomi-robotics-0.md
- wiki/methods/generative-world-models.md
- sources/repos/xiaomi-robotics-u0.md
- sources/papers/xiaomi_robotics_u0_arxiv_2607_11643.md

## [2026-07-15] ingest | sources/papers/being_m07.md — Being-M0.7 人形潜空间 WAM；wiki/entities/paper-being-m07-humanoid-latent-wam.md；交叉更新 world-action-models / loco-manipulation / being-h07 / teleoperation

## [2026-07-15] ingest | sources/courses/learn_robotics_qqfly_guide.md — 接入 qqfly 开源机器人学学习指南并新建 wiki/entities/learn-robotics-qqfly-guide.md

- wiki/entities/learn-robotics-qqfly-guide.md
- wiki/entities/modern-robotics-book.md
- roadmap/motion-control.md
- sources/courses/learn_robotics_qqfly_guide.md

## [2026-07-15] ingest | sources/papers/pace_sim2real_arxiv_2509_06342.md、sources/repos/pace-sim2real.md、sources/sites/pace-filipbjelonic-com.md — PACE 足式系统化 sim2real（arXiv:2509.06342）

- wiki/entities/paper-pace-sim2real-legged-robots.md
- wiki/concepts/sim2real.md
- wiki/concepts/system-identification.md
- wiki/comparisons/sim2real-approaches.md
- wiki/methods/actuator-network.md
- wiki/entities/anymal.md
- sources/papers/pace_sim2real_arxiv_2509_06342.md
- sources/repos/pace-sim2real.md
- sources/sites/pace-filipbjelonic-com.md

## [2026-07-14] structural | wiki/queries/embodied-eval-benchmark-selection-loop.md、wiki/concepts/sim-vs-real-eval-gap.md — 沉淀「具身大模型评测基准选型闭环知识链」（V29 P1①）

- wiki/queries/embodied-eval-benchmark-selection-loop.md（端到端 Query：MLLM 认知评测 → 世界模型预测保真度 → 策略成功率 → sim↔real gap 校准 四层评测选型决策树 + Mermaid，含矛盾/失败模式速查）
- wiki/concepts/sim-vs-real-eval-gap.md（仿真可复现性 ↔ 真实代表性取舍概念页，双向回链 Query）
- wiki/concepts/simulation-evaluation-infrastructure.md（关联页面补入评测选型闭环回链，消孤儿）
- 图谱见本次自动重算的 graph-stats.json（已并入 main 新增页）；largest_community_ratio 健康；`make lint` 0 errors

## [2026-07-14] ingest | sources/sites/unity-com.md — 接入 Unity Engine 官网与文档并升格 wiki/entities/unity-engine.md

## [2026-07-14] ingest | sources/sites/unreal-engine-5-com.md、sources/sites/unreal-engine-5-8-docs.md、sources/repos/epicgames-github-org.md、sources/repos/unrealengine-github.md — 接入 UE5 官网/5.8 文档/Epic GitHub 与引擎私有仓说明，新建 wiki/entities/unreal-engine-5.md 并交叉更新 MetaHuman/AirSim/CARLA/SPEAR/MATRiX

- wiki/entities/unreal-engine-5.md
- sources/sites/unreal-engine-5-com.md
- sources/sites/unreal-engine-5-8-docs.md
- sources/repos/epicgames-github-org.md
- sources/repos/unrealengine-github.md

## [2026-07-14] ingest | sources/raw/feishu_humanoid_motion_control_know_how_full_2026-07-14.md — 飞书 Know-How 全文（2260 行）再 ingest；补技术框架路线展望与方法论要点

- wiki/overview/humanoid-motion-control-framework-outlook.md
- sources/raw/feishu_humanoid_motion_control_know_how_full_2026-07-14.md
- sources/papers/humanoid_motion_control_know_how.md
- wiki/overview/humanoid-motion-control-know-how-technology-map.md
- wiki/concepts/modeling-and-solving-for-control.md
- wiki/concepts/humanoid-vs-other-robots.md
- wiki/concepts/humanoid-rubber-man-analogy.md
- wiki/concepts/kinematic-vs-dynamic-feasibility.md
- wiki/methods/slip-vmc.md
- wiki/methods/dreamwaq.md
- wiki/methods/teacher-student-multi-skill-bfm.md
- wiki/queries/humanoid-motion-control-know-how.md

## [2026-07-14] ingest | sources/papers/humanoid_motion_control_know_how.md、sources/raw/feishu_humanoid_motion_control_know_how_2026-07-14.md — RoboParty 飞书 Know-How 全主题独立节点与技术地图

- wiki/overview/humanoid-motion-control-know-how-technology-map.md
- wiki/overview/humanoid-motion-control-trends.md
- wiki/overview/humanoid-model-based-control-stack.md
- wiki/overview/humanoid-rl-motion-control-methods.md
- wiki/methods/slip-vmc.md
- wiki/methods/pie-perceptive-locomotion.md
- wiki/methods/dreamwaq.md
- wiki/methods/attention-foot-placement.md
- wiki/methods/teacher-student-dagger-training.md
- wiki/methods/teacher-student-multi-skill-bfm.md
- wiki/methods/centroidal-nmpc-wbc-stack.md
- wiki/concepts/humanoid-vs-other-robots.md
- wiki/concepts/kinematic-vs-dynamic-feasibility.md
- wiki/concepts/humanoid-rubber-man-analogy.md
- wiki/concepts/modeling-and-solving-for-control.md
- wiki/concepts/srbd-convex-mpc-wbc.md
- wiki/queries/humanoid-motion-control-know-how.md
- sources/papers/pie_arxiv_2408_13740.md
- sources/papers/dreamwaq_arxiv_2301_10602.md
- 工具：Agent Reach v1.5.0（`~/.local/bin/agent-reach`）+ Jina Reader 抓取飞书公开页

## [2026-07-14] ingest | sources/papers/muon_optimizer_primary_refs.md、sources/blogs/muon_keller_jordan_2024.md — Muon 优化器博客提出、LLM 规模化论文与理论文献入库

- wiki/methods/muon.md
- wiki/entities/paper-muon-scalable-llm-training.md
- wiki/comparisons/deep-learning-optimizers.md
- sources/papers/muon_optimizer_primary_refs.md
- sources/blogs/muon_keller_jordan_2024.md
- sources/repos/kellerjordan-muon.md

## [2026-07-14] structural | roadmap/depth-torque-motor-design.md — 新增力矩控制电机设计全流程纵深路线（Stage 0–5），十四条纵深入口全站同步

- roadmap/depth-torque-motor-design.md：新建「力矩控制电机设计」纵深路线（起点：磁场定向控制 FOC，1971），Stage 0 电机学地基与 TN/TI 读图 → Stage 1 关节指标与执行器架构选型 → Stage 2 电磁热设计 → Stage 3 驱动硬件与电流环 → Stage 4 FOC 力矩闭环标定补偿 → Stage 5 台架验收与关节模组交付；是 [电机设计流程](wiki/overview/motor-design-workflow.md) 工序页的学习顺序展开版
- wiki/overview/motor-design-workflow.md、wiki/overview/humanoid-actuator-102-technology-map.md、wiki/concepts/field-oriented-control.md：关联页面补纵深路线入口
- roadmap/README.md、README.md、index.md、roadmap/motion-control.md、docs/index.html、docs/main.js 与其余十三条 depth-* 路线页：纵深路线数 13 → 14，按里程碑排序（FOC 1971 列首位）并补双向链接

## [2026-07-14] structural | roadmap/depth-perceptive-locomotion.md、wiki/entities/paper-faststair-humanoid-stair-ascent.md、wiki/tasks/stair-obstacle-perceptive-locomotion.md — FastStair 补链感知越障纵深 Stage 3

- roadmap/depth-perceptive-locomotion.md：Stage 3 补 [FastStair](wiki/entities/paper-faststair-humanoid-stair-ascent.md)、[Explicit Stair Geometry](wiki/entities/paper-explicit-stair-geometry-humanoid-locomotion.md)、[Capture Point / DCM](wiki/concepts/capture-point-dcm.md)；快速入口与关联知识页同步
- wiki/entities/paper-faststair-humanoid-stair-ascent.md：related 与正文链回感知越障纵深路线
- wiki/tasks/stair-obstacle-perceptive-locomotion.md：关联页面补纵深路线入口

## [2026-07-14] fix | scripts/utils/community_labels.py — 补 roadmap/depth-navigation.md 社区名 override，修复 CI 社区命名测试

- 六条纵深路线补链 +15 边后社区结构变化，community-14 枢纽变为 roadmap/depth-navigation.md，其 H1 不符合「中文（English） 社区」命名模式，`tests/test_community_naming.py` 失败
- 按 schema/naming.md § 图谱社区命名的维护方式，在 `COMMUNITY_NAME_OVERRIDES` 补「导航纵深路线（Navigation Deep-Dive Roadmap）」；`make ci-preflight` 12/12 通过、WARNING 消除，`pytest` 全量通过

## [2026-07-14] structural | roadmap/depth-*.md 增量刷新 — 依据 07-10 全量刷新后新入库知识节点补链六条纵深路线

- roadmap/depth-bfm.md：Stage 3 预训练三线补 [TeCH](wiki/entities/paper-tech-humanoid-control.md)（TLDR 对比时间距离无监督预训练，G1 上对标 SONIC 且 GPU 时长降近两个数量级）
- roadmap/depth-vla.md：Stage 2 主线补 [InternVLA-A1.5](wiki/entities/paper-internvla-a15-unified-vla.md)（统一理解 + 潜式前瞻 MoT），Stage 4 部署补 [Evo-1](wiki/entities/paper-evo1-lightweight-vla.md)（0.77B 轻量边缘侧样本），Stage 5 方向 A 补 [STEAM](wiki/entities/paper-steam-advantage-modeling.md)（自监督时序 advantage 离线提纯 π₀）
- roadmap/depth-perceptive-locomotion.md：Stage 1 补 [AME](wiki/entities/paper-ame-attention-based-map-encoding.md) 与 AME-2 深读笔记（注意力高程图编码），Stage 2 补 [InfiniteDiffusion / Terrain Diffusion](wiki/entities/paper-infinite-diffusion-terrain-diffusion.md)（学习式程序化地形）
- roadmap/depth-wam.md：Stage 1 世界模型基座补 [PhysisForcing](wiki/entities/paper-physisforcing.md)（训练期分层物理对齐）与 [PhysMani](wiki/entities/paper-physmani-dynamic-manipulation-world-model.md)（3D Gaussian 速度场动态操作世界模型）
- roadmap/depth-contact-manipulation.md：Stage 2 补 [DexVerse](wiki/entities/paper-dexverse.md)（100 任务多具身灵巧 benchmark），Stage 3 补 [ContactMimic](wiki/entities/paper-contactmimic.md)、[UHAS](wiki/methods/uhas-unified-hand-action-space.md)、[FastGrasp](wiki/entities/paper-fastgrasp-mobile-dexterous-grasping.md)
- roadmap/depth-motion-retargeting.md：Stage 2 数据源补 [PEAR](wiki/entities/paper-pear-pixel-aligned-expressive-hmr.md) 与 [ViDiHand](wiki/entities/paper-vidihand.md)（表达级单图 SMPL-X 实时恢复 / egocentric 双手 4D 估计）

## [2026-07-14] ingest | sources/sites/roboparty_com.md — RoboParty 官网与文档站一手资料；公司实体页与 Roboto Origin 参数/仓库名同步

- wiki/entities/roboparty.md
- wiki/entities/roboto-origin.md
- wiki/entities/party-os.md
- wiki/overview/roboparty-lab-party-os-technology-map.md
- wiki/entities/open-source-humanoid-hardware.md
- sources/sites/roboparty_com.md
- sources/sites/roboparty_com_roboto_origin_doc.md
- sources/sites/lab_roboparty_com.md
- sources/repos/roboto_origin.md

## [2026-07-14] ingest | sources/sites/roboparty_lab_tech_humanoid_control.md — TeCH 无监督人形全身控制实体页与 UFO/Party OS 交叉更新

- wiki/entities/paper-tech-humanoid-control.md
- wiki/entities/roboparty-ufo.md
- wiki/overview/roboparty-lab-party-os-technology-map.md
- wiki/entities/party-os.md
- wiki/entities/paper-bfm-zero.md
- sources/sites/roboparty_lab_tech_humanoid_control.md

## [2026-07-14] ingest | sources/blogs/wechat_roboparty_lab_party_os_3_tools.md — RoboParty Lab / Party OS 技术地图与 MimicLite、UFO、hhtools 子实体页

- wiki/overview/roboparty-lab-party-os-technology-map.md
- wiki/entities/party-os.md
- wiki/entities/mimiclite.md
- wiki/entities/roboparty-ufo.md
- wiki/entities/human-humanoid-tools.md
- wiki/entities/roboto-origin.md
- wiki/concepts/motion-retargeting.md
- sources/blogs/wechat_roboparty_lab_party_os_3_tools.md

## [2026-07-14] ingest | sources/papers/generalized_value_functions_gvf_primary_refs.md、sources/papers/bayesian_analysis_rl_primary_refs.md — GVF 与贝叶斯信念分析一手资料；wiki/concepts/generalized-value-functions.md、wiki/concepts/bayesian-belief-analysis.md

- wiki/concepts/generalized-value-functions.md
- wiki/concepts/bayesian-belief-analysis.md
- wiki/entities/richard-sutton.md
- wiki/formalizations/pomdp.md
- wiki/methods/model-based-rl.md
- sources/papers/generalized_value_functions_gvf_primary_refs.md
- sources/papers/bayesian_analysis_rl_primary_refs.md

## [2026-07-14] ingest | sources/sites/incompleteideas-net-rich-sutton.md — Richard Sutton / incompleteideas.net 一手资料；wiki/entities/richard-sutton.md、wiki/entities/sutton-barto-rl-book.md、wiki/concepts/bitter-lesson.md

- wiki/entities/richard-sutton.md
- wiki/entities/sutton-barto-rl-book.md
- wiki/concepts/bitter-lesson.md
- wiki/methods/reinforcement-learning.md
- wiki/methods/model-based-rl.md
- wiki/concepts/embodied-scaling-laws.md
- sources/sites/incompleteideas-net-rich-sutton.md
- sources/blogs/sutton_bitter_lesson.md
- sources/blogs/sutton_one_step_trap.md

## [2026-07-14] ingest | sources/papers/physmani_arxiv_2607_01938.md — PhysMani ECCV 2026 动态操作 3D Gaussian 世界模型；wiki/entities/paper-physmani-dynamic-manipulation-world-model.md

## [2026-07-14] structural | wiki/tasks/loco-manipulation.md + roadmap/depth-loco-manipulation.md — 任务页新增「术语辨析：Loco-Manipulation vs Mobile Manipulation」小节（轮式底座/腿式全身耦合分界、本库中文「移动操作」统一指 Loco-Manipulation 的约定、mobile-manipulation tag 专标轮式工作）；纵深路线更名 depth-mobile-manipulation.md → depth-loco-manipulation.md，README/index/主路线/12 条纵深页共 28 处链接与主页路线按钮统一为「Loco-Manipulation（移动操作）」

## [2026-07-14] ingest | sources/papers/infinite_diffusion_terrain_diffusion_siggraph_2026.md、sources/repos/terrain-diffusion.md — InfiniteDiffusion/Terrain Diffusion（SIGGRAPH 2026）；wiki/entities/paper-infinite-diffusion-terrain-diffusion.md；交叉 procedural-terrain-generation、generative-world-models

- wiki/entities/paper-infinite-diffusion-terrain-diffusion.md
- wiki/concepts/procedural-terrain-generation.md
- wiki/methods/generative-world-models.md
- sources/papers/infinite_diffusion_terrain_diffusion_siggraph_2026.md
- sources/repos/terrain-diffusion.md

## [2026-07-14] ingest | sources/papers/dexverse_arxiv_2607_08751.md — DexVerse 多任务多具身灵巧 benchmark；wiki/entities/paper-dexverse.md

- wiki/entities/paper-dexverse.md
- wiki/tasks/manipulation.md
- wiki/entities/isaac-lab.md
- wiki/queries/dexterous-manipulation-data-pipeline.md
- sources/papers/dexverse_arxiv_2607_08751.md
## [2026-07-14] ingest | sources/repos/embodiedgen.md、sources/datasets/embodiedgen-data.md — 接入 EmbodiedGen 官方仓库/文档/数据集；wiki/entities/paper-embodiedgen-v2-sim-ready-world-engine.md 增补工程实践节

- wiki/entities/paper-embodiedgen-v2-sim-ready-world-engine.md
- sources/repos/embodiedgen.md
- sources/datasets/embodiedgen-data.md

## [2026-07-13] structural | 执行清单换版 v28 → v29 —— V28（具身大模型分类学选型闭环）P0–P3 全数交付、DoD 逐项达标后新建 V29（具身大模型评测基准选型闭环）

- V28 收尾：P0 lint 巡检 / P1 选型链 query+concept / P2 事实库 +10 矛盾规则 / P3 图谱专题视图+详情徽标 五档全绿；DoD 逐项复核达标——`make lint` 0 errors（另含 4 条信息型预警）、node_count=1597（≥1595）/ edge_count=12168（≥10970）/ orphan_nodes=0、事实库 230 条、`largest_community_ratio=0.199`/`community_quality_warning=false`、log 记录齐全，清单 0 未勾项
- 换版动作（对齐 [`docs/checklists/README.md`](docs/checklists/README.md) 维护规则）：`git mv` V28 至 [`archive/`](docs/checklists/archive/tech-stack-next-phase-checklist-v28.md)；新建 [`tech-stack-next-phase-checklist-v29.md`](docs/checklists/tech-stack-next-phase-checklist-v29.md)；README「当前入口」更新为 v29、「历史执行清单」补 v28
- V29 主题：承接 V28「选哪一类具身大模型」，回答「怎么评测/证明它」——把近周密集 ingest 的评测基准（RoboBench / EWMBench / ESI-Bench / GigaWorld-1 policy evaluation / MimickingBench / ManiSkill-HAB / Barkour 等）沉淀为「具身大脑/MLLM 认知评测 → 世界模型预测保真度评测 → 策略任务成功率评测 → sim↔real 评测 gap 校准」贯通选型链；起点基线 1597 节点 / 12168 边 / 230 事实 / 18 专题，目标 ≥1610 / ≥12230 / ≥240 / 第 19 专题「具身评测基准」
- P0 lint 巡检 / P1 评测选型链 query+concept / P2 事实库 +10 评测矛盾规则 / P3 图谱专题视图+详情徽标，均为 `[ ]` 待后续每日推进

## [2026-07-13] structural | scripts/lint_wiki.py — 修复全量 lint 信息型预警：SOTA 匹配加词边界（消除 Minnesota 误报，补回归单测）；wbc 归入 COVERED_ELSEWHERE（已由 concepts/whole-body-control.md 覆盖）；复核并 bump paper-gigaworld-1/paper-rynnworld updated=2026-07-13；lint 全绿 0 问题 0 信息型

## [2026-07-13] ingest | sources/papers/physisforcing_arxiv_2606_28128.md — PhysisForcing 训练期分层物理对齐世界模拟器；wiki/entities/paper-physisforcing.md；交叉 generative-world-models / cosmos-3 / manipulation

## [2026-07-13] ingest | sources/papers/vidihand_arxiv_2606_30308.md — ViDiHand 视频扩散双手 4D 重建；wiki/entities/paper-vidihand.md

## [2026-07-13] ingest | sources/repos/more.md — 接入 TeleHuman/MoRE 官方仓库；wiki/entities/paper-amp-survey-08-more.md 增补代码复现节

## [2026-07-13] ingest | sources/papers/ame_arxiv_2506_09588.md、sources/papers/humanoid_pnb_ame-2-agile-and-generalized-legged-locomotion-vi.md — AME/AME-2 ETH RSL 注意力地图编码感知 loco；wiki/entities/paper-ame-attention-based-map-encoding.md、wiki/entities/paper-notebook-ame-2-agile-and-generalized-legged-locomotion-vi.md；交叉 stair-obstacle-perceptive-locomotion、terrain-adaptation

## [2026-07-13] ingest | sources/courses/jason_peng_synthetic_motion_humanoid_youtube.md — Jason Peng NUS 研讨会合成运动数据与通用人形控制（2looxieN53o）；wiki/overview/jason-peng-flexible-motion-skill-learning.md、wiki/entities/xue-bin-peng.md、wiki/entities/paper-notebook-parc-physics-based-augmentation-with-reinforceme.md

## [2026-07-13] ingest | sources/repos/greensock-gsap-skills.md — GSAP 官方 Agent Skills；新增 wiki/entities/gsap-skills.md 并交叉 mattpocock-skills / 前端清单

- wiki/entities/gsap-skills.md
- wiki/entities/mattpocock-skills.md
- sources/repos/greensock-gsap-skills.md

## [2026-07-13] ingest | sources/blogs/wechat_embodied_ai_lab_isaac_gr00t_n17_g1_e2e.md — 具身智能研究室 GR00T 1.7 G1 端到端中文策展；增补 wiki/entities/isaac-gr00t.md G1 仿真教程节

- wiki/entities/isaac-gr00t.md
- sources/blogs/wechat_embodied_ai_lab_isaac_gr00t_n17_g1_e2e.md

## [2026-07-13] ingest | sources/repos/isaac_gr00t.md — 接入 Isaac-GR00T 仓库与 NVIDIA 端到端博客，新增 wiki/entities/isaac-gr00t.md 并交叉 GR00T N1 / WBC / LeRobot

- wiki/entities/isaac-gr00t.md
- wiki/entities/paper-hrl-stack-34-gr00t_n1.md
- wiki/entities/gr00t-wholebodycontrol.md
- sources/repos/isaac_gr00t.md
- sources/blogs/nvidia_develop_humanoid_robot_policies_isaac_gr00t.md

## [2026-07-12] structural | checklist-v28 P3② 详情页「同专题相关页」提示 —— 具身大模型徽标端到端验证与截图归档

- 详情页「所属专题」徽标行（`docs/main.js renderMetaTopicBadges`）本就以 `docs/topic-filters.js` 为单一事实源、`topicsForNode` 数据驱动：P3① 把 `embodied-foundation-model` 写入单一事实源后，详情页徽标已自动联动，无需二次实现——命中即渲染「🧠 具身大模型」徽标并跳 `graph.html?topic=embodied-foundation-model`，空态降级隐藏整行。
- 端到端验证：`wiki/methods/vla.md` 详情页（`detail.html?id=wiki-methods-vla`）「所属专题」行同渲「👀 视觉-语言-动作 (VLA)」+「🧠 具身大模型 (Embodied Foundation Model)」双徽标；node 逐页校验 `topicsForNode` 对 methods/vla、tasks/vision-language-navigation、concepts/world-action-models、queries/embodied-fm-taxonomy-loop、methods/generative-world-models 五页均稳定命中 `embodied-foundation-model` 专题。截图归档 `.cursor-artifacts/screenshots/detail-topic-embodied-foundation-model.png`。
- `make lint` 0 errors（另含 1 条信息型预警，不阻塞 CI）；勾选 v28 P3「详情页『同专题相关页』提示」第②项，P3 两个子项全部完成。

## [2026-07-12] ingest | sources/papers/robo_bench_arxiv_2510_17801.md — RoboBench MLLM 具身大脑五维评测；wiki/entities/robo-bench.md 及 VLA/ESI-Bench 交叉引用

- wiki/entities/robo-bench.md
- wiki/methods/vla.md
- wiki/entities/esi-bench.md

## [2026-07-12] ingest | sources/papers/fastgrasp_arxiv_2604_12879.md — FastGrasp 移动全身快速灵巧抓取；wiki/entities/paper-fastgrasp-mobile-dexterous-grasping.md、wiki/tasks/manipulation.md、wiki/tasks/loco-manipulation.md、wiki/overview/topic-grasp.md

- wiki/entities/paper-fastgrasp-mobile-dexterous-grasping.md
- wiki/tasks/manipulation.md
- wiki/tasks/loco-manipulation.md
- wiki/overview/topic-grasp.md

## [2026-07-12] ingest | sources/papers/contactmimic_arxiv_2607_08742.md — ContactMimic 人形 HOI contact 条件跟踪；wiki/entities/paper-contactmimic.md 及选型/流水线交叉引用

## [2026-07-12] ingest | sources/sites/lerobot-huggingface-org.md — LeRobot HF Hub 组织页；wiki/entities/lerobot.md

- wiki/entities/lerobot.md

## [2026-07-12] ingest | sources/courses/karpathy_zero_to_hero_youtube.md、sources/repos/nn-zero-to-hero.md — Karpathy Zero to Hero 播放列表（10 集）与配套仓；wiki/entities/andrej-karpathy.md、wiki/concepts/backpropagation.md、wiki/concepts/transformer.md、wiki/entities/llms-from-scratch-raschka.md、roadmap/depth-vla.md

- wiki/entities/andrej-karpathy.md
- wiki/concepts/backpropagation.md
- wiki/concepts/transformer.md
- wiki/entities/llms-from-scratch-raschka.md
- roadmap/depth-vla.md

## [2026-07-12] ingest | sources/courses/karpathy_intro_llms_youtube.md、sources/courses/karpathy_deep_dive_llms_youtube.md — Karpathy LLM 科普双视频（Intro 1h + Deep Dive 3.5h）；wiki/entities/andrej-karpathy.md、wiki/concepts/deep-learning-foundations.md、wiki/entities/llms-from-scratch-raschka.md、roadmap/depth-vla.md

- wiki/entities/andrej-karpathy.md
- wiki/concepts/deep-learning-foundations.md
- wiki/entities/llms-from-scratch-raschka.md
- roadmap/depth-vla.md

## [2026-07-11] ingest | sources/repos/rasbt_llms_from_scratch.md、sources/courses/rasbt_llms_from_scratch_youtube.md — Raschka LLMs-from-scratch 书/仓/YouTube 三角入库

- wiki/entities/llms-from-scratch-raschka.md
- wiki/concepts/transformer.md
- wiki/concepts/deep-learning-foundations.md
- wiki/entities/andrej-karpathy.md
- roadmap/depth-vla.md

## [2026-07-11] structural(topic) | docs/topic-filters.js + docs/graph.html — V28 P3① 图谱页「🧠 具身大模型」专题视图落地

- docs/topic-filters.js：新增 `embodied-foundation-model` 专题（TOPIC_HUB_IDS / TOPIC_FILTERS / TOPIC_META 三处），干净片段 `vlm`/`vln`/`vlx`，`vla` 仅从 segments 剔除（不入 excludeSegments，避免误伤含 `vla` 词元的 `vlm-vln-vla-vlx-world-model-taxonomy` 等 ids 页），`ids` 精选纳入五层闭环家族页（embodied-fm-taxonomy-loop / behavior-foundation-model / unified-multimodal-tokens / methods-vla / vision-language-navigation / world-action-models / generative-world-models / foundation-policy / behavior-tree-vla-orchestration / 3d-spatial-vqa）
- docs/graph.html：`#filter-topic-chips` 增加对应 🧠 chip（接 contact-force-control 之后）
- 枢纽页 wiki/overview/topic-embodied-foundation-model.md 早前已建并交叉回链；专题视图筛出 30 节点，graph-stats 0 orphans；截图归档 .cursor-artifacts/screenshots/graph-topic-embodied-foundation-model.png
- make lint 0 errors；docs/checklists/tech-stack-next-phase-checklist-v28.md P3① 打勾

## [2026-07-11] ingest | sources/papers/steam_arxiv_2606_29834.md — STEAM 自监督 ensemble advantage + CFGRL；wiki/entities/paper-steam-advantage-modeling.md

- wiki/entities/paper-steam-advantage-modeling.md
- wiki/methods/vla.md
- wiki/overview/vla-open-source-repro-landscape-2025.md
- wiki/entities/paper-rove-humanoid-vla-intervention.md

## [2026-07-11] ingest | sources/papers/internvla_a15_arxiv_2607_04988.md — InternVLA-A1.5 统一理解/潜式前瞻/动作 VLA；wiki/entities/paper-internvla-a15-unified-vla.md

- wiki/entities/paper-internvla-a15-unified-vla.md
- wiki/methods/vla.md
- wiki/concepts/world-action-models.md

## [2026-07-11] ingest | sources/papers/pear_arxiv_2601_22693.md — PEAR 像素对齐表意 HMR（SIGGRAPH 2026）；wiki/entities/paper-pear-pixel-aligned-expressive-hmr.md

- wiki/entities/paper-pear-pixel-aligned-expressive-hmr.md
- wiki/entities/sam-3d-body.md
- wiki/entities/gvhmr.md
- wiki/concepts/motion-retargeting-pipeline.md

## [2026-07-11] fix(ux): docs/style.css + docs/main.js — 修复移动端「更多路线」折叠态 WAM 通栏：末行居中规则改挂 `.is-expanded`，避免 hidden 节点被 `:nth-child` 计入

## [2026-07-11] structural | roadmap/depth-wam.md — 纵深路线扩容至十三条：新增 WAM（世界–动作模型）纵深（边界与族谱 → 世界模型基座 → Cascaded → Joint → 部署职责三分 → 数据评测进阶，起点 World Action Models 综述形式化 2026）；主页「更多路线」按钮扩为十三个并按历史序插入，默认展示最新四条（感知越障 / 动作生成 / VLA / WAM）；README/index/主路线与全部纵深页互链；链接 wiki/concepts/world-action-models.md、动作后果技术地图、DiT4DiT/MotionWAM/DSWAM 等枢纽页

## [2026-07-11] ingest | sources/blogs/wechat_embodied_ai_lab_robot_world_models_action_consequence_2026.md — 世界模型动作后果专题：12 篇独立论文节点 + 四线技术地图；wiki/overview/robot-world-models-action-consequence-technology-map.md wiki/overview/wm-action-consequence-category-01-wam-action-prediction.md … 04 wiki/entities/paper-dswam-dual-system-wam.md … paper-gigaworld-1-policy-evaluation.md

## [2026-07-11] ingest | sources/repos/awesome_bfm_papers.md、sources/papers/bfm_survey_arxiv_2506_20487.md — 刷新 awesome-bfm-papers（42 篇 + Any2Any）与 BFM 综述；新增 wiki/entities/paper-bfm-survey-tpami-2025.md；交叉更新 behavior-foundation-model、bfm-category-04-adaptation、bfm-41-papers-technology-map

- wiki/entities/paper-bfm-survey-tpami-2025.md
- wiki/concepts/behavior-foundation-model.md
- wiki/overview/bfm-category-04-adaptation.md
- wiki/overview/bfm-41-papers-technology-map.md

## [2026-07-11] ingest | sources/papers/ardy_siggraph_2026.md — ARDY 交互式自回归扩散人体运动生成（SIGGRAPH 2026）；wiki/entities/ardy.md，交叉更新 kimodo、diffusion-motion-generation、sonic-motion-tracking、motionbricks、depth-motion-generation

## [2026-07-11] ingest | sources/papers/uhas_arxiv_2607_03570.md — UHAS 统一手部动作空间：sources + wiki/methods/uhas-unified-hand-action-space.md，交叉更新 in-hand-reorientation 与 cross-embodiment-transfer-strategy

## [2026-07-11] ingest | arXiv 摘要与项目页细节补强 — BeyondMimic（arXiv:2508.08241 v4）与 SONIC（arXiv:2511.07820 v3）四个知识节点补论文信息（作者/机构/版本）、BeyondMimic 两阶段机制（LAFAN1 单一设置跟踪 → 统一潜空间扩散 + classifier guidance 零样本下游）与 SONIC 三轴 scaling / 统一 token 接口细节，并修复实体页策展占位与截断文本；wiki/methods/beyondmimic.md、wiki/methods/sonic-motion-tracking.md、wiki/entities/paper-beyondmimic.md、wiki/entities/paper-sonic.md

## [2026-07-10] structural | schema/canonical-facts.json — 清单 v28 P2 事实库扩展：220 → 230 条，新增 10 条具身大模型选型矛盾检测规则（端到端 VLA 泛化 vs 分层 VLN 可解释可调、显式世界模型前瞻 vs 反应式省算力、大模型参数量 vs 控制带宽/推理时延、统一 VLX 通用性 vs 专精分立精度、多模态跨度 vs 注意力开销、世界模型推演步长 vs 累积误差、VLM 语义理解 ≠ 可执行动作接口、VLN 仅底盘移动无力控/操作分支、共享 Transformer 底座不免真机数据、模型规模不替代真机动作数据）；逐条经脚本校验对现存 wiki 页有 pos 命中且 0 误报，`make lint` 潜在矛盾 0 个、0 errors；同步勾选 checklist v28 P2 与 DoD 事实库项

## [2026-07-10] structural | roadmap/depth-motion-retargeting.md + depth-motion-generation.md — 纵深路线扩容至十二条：新增动作重定向纵深（问题定义与管线定位 → IK/优化工具箱 → 数据源与质量 → GMR→NMR→物理感知方法谱系 → WBT/遥操作下游闭环，起点 Gleicher 动作重定向 1998）与动作生成纵深（扩散/流匹配基础 → 表示与数据集 → MDM→可控生成→HY-Motion/GENMO/Kimodo 规模化三线 → PhysDiff/PhyGile/Gen2Humanoid 物理化落地 → 控制环内生成器，起点 MDM 扩散动作生成 2022）两条姊妹路线；主页「更多路线」按钮扩为十二个并按历史序插入，README/index/主路线与全部纵深页互链；链接 wiki/concepts/motion-retargeting.md、wiki/comparisons/gmr-vs-nmr-vs-reactor.md、wiki/methods/diffusion-motion-generation.md、wiki/comparisons/hy-motion-vs-genmo-vs-kimodo.md、wiki/overview/topic-motion-retargeting.md 等枢纽页

## [2026-07-10] ingest | sources/blogs/perceptron_egocentric_api.md — Perceptron Egocentric 子任务自动标注 API；wiki/entities/perceptron-egocentric.md、wiki/methods/auto-labeling-pipelines.md、wiki/entities/gemini-robotics.md

## [2026-07-10] structural | roadmap/depth-vla.md + depth-bfm.md + depth-mobile-manipulation.md + depth-navigation.md — 纵深路线扩容至十条：将原「VLA 与 BFM」合并路线拆分为 VLA 纵深（分类学 → IL 策略基座 → RT→π0 主线 → 数据与 Scaling → 部署整合，起点 RT-2 2023）与 BFM 纵深（训练基座 → DeepMimic→HOVER 跟踪谱系 → 预训练三线 → 适应与双栈整合，起点 DeepMimic 2018）两条姊妹路线，另新增移动操作纵深（全身耦合 → 上下身解耦基座 → 数据入口 → 技能学习 → 统一模型，起点移动操作臂协调 1994）与导航纵深（SLAM/状态估计 → Nav2 → 学习型导航 → VLN → 导航 VLA，起点概率 SLAM 1986）；主页「更多路线」按钮扩为十个并按历史序重排，README/index/主路线与全部纵深页互链；链接 wiki/methods/vla.md、wiki/concepts/behavior-foundation-model.md、wiki/tasks/loco-manipulation.md、wiki/tasks/vision-language-navigation.md、wiki/overview/navigation-slam-autonomy-stack.md 等枢纽页

## [2026-07-10] ingest | sources/papers/3d_ic_icml_2026.md — 3D-IC OVMM 联合导航操作规划；wiki/entities/paper-3d-ic-joint-navigation-manipulation-planning.md、wiki/tasks/loco-manipulation.md、wiki/tasks/vision-language-navigation.md、wiki/tasks/manipulation.md

## [2026-07-10] ingest | Paper Notebooks 深读升格批次 — 113 个 paper-notebook 实体页升格为深读索引并补详情

- 索引重建：[`schema/paper-notebook-index.json`](schema/paper-notebook-index.json) 137 → 289 篇（`_data/papers.json` 286 + progress.json done 补 RMA / Unitree H1 Whitepaper / MotionVAE 3 篇）；重生成 `paper-notebook-wiki-full-map.yml`（548 篇）与 `paper-notebook-categories.json`
- 升格补详情：111 个 `status: planned` 实体页升格 + 新建 2 页（[EgoNav](wiki/entities/paper-notebook-egonav.md)、[GAIT](wiki/entities/paper-notebook-gait.md)）——每页从深读笔记编译 `一句话定义`、论文专属英文缩写速查、为什么重要（领域意义）、解决什么问题、核心机制（核心贡献 + 方法拆解）、核心信息（机构/作者/发表/项目页/源码/笔记阅读日期）
- 溯源锚点：对应 `sources/papers/humanoid_pnb_*` 改写为深读笔记锚点（真实笔记 URL + 一句话总结）；69 个旧标题 slug 源文件更名为目录规范名
- 分类树：14 个 `paper-notebook-category-*` 父节点与 [总索引](wiki/overview/humanoid-paper-notebooks-index.md) 重生成，待深读 → 深读笔记链接翻转；`make paper-notebook-links` 向 45 个既有深读实体页注入笔记链接
- 数据纠错：footstep-planner 占位页误挂 arXiv 2510.12215（实为 Learning Social Navigation，已升格独立实体）→ 删除占位；PROGRESS.md `arxiv.org/pdf/` 链接解析修复（[bootstrap_paper_notebook_knowledge.py](scripts/bootstrap_paper_notebook_knowledge.py) 兼容 abs|pdf，消 EgoPoser 待读/深读双行）；删除与 [HTD 方法页](wiki/methods/humanoid-transformer-touch-dreaming.md) 重复的 visual-tactile 占位
- 映射钉扎：[`schema/paper-notebook-wiki-overrides.yml`](schema/paper-notebook-wiki-overrides.yml) 新增 EgoDex / Visual-Tactile(HTD) / LapSurgie / Humanoids-in-Hospitals 4 条，避免论文节点被引用其 arXiv 的相邻页面吸附错绑
- 门禁：`make ci-preflight` 全过（lint 0 问题、搜索回归通过、导出检查 12/12）；pytest 303 passed

## [2026-07-10] structural | README.md / index.md / roadmap/README.md / roadmap/motion-control.md — 纵深路线列表统一按方向起点里程碑历史排序（与首页按钮一致），README 与 roadmap 总览向读者说明排序方法（ZMP 1972 → CLF 1983 → 阻抗控制 1985 → 行为克隆 1988 → Q-learning 1989 → 感知越障 2020s）

## [2026-07-10] structural | docs/index.html — 主页「更多路线」六按钮改按方向起点里程碑历史排序（传统控制 ZMP 1972 → 安全控制 CLF 1983 → 接触操作 阻抗控制 1985 → 模仿学习 行为克隆 1988 → 强化学习 Q-learning 1989 → 感知越障 2020s）；同步 frontend-optimization-v1 checklist

## [2026-07-10] structural | docs/style.css — 修复首页「更多路线」六按钮在 861–940px 宽度下 5+1 孤行的排版问题（auto-fit 最小列宽 132px → 118px）；同步 frontend-optimization-v1 checklist

## [2026-07-10] structural | roadmap/depth-*.md 全量刷新 — 依据当前知识图谱把六条纵深路线的推荐阅读升级为已有 wiki 页：RL 路线接 ppo/sac/reward-design/actuator-network/sim2real-checklist 与感知越障、动作跟踪出口；IL 路线接 diffusion-policy/action-chunking/GMR-NMR 对比/AMP 综述地图/VLA 出口；安全控制补 safety-filter/safe-rl/cmdp；接触操作补 force-control/tactile 系列；传统控制补 MPC/WBC 调参与 Pinocchio 上手 query

## [2026-07-10] structural | roadmap/depth-perceptive-locomotion.md — 新增感知越障（Perceptive Locomotion）纵深路线（盲走基线 → 地形表征 → Teacher-Student → 楼梯/跑酷 → 导航栈整合）；主页「更多路线」新增感知越障按钮，README/index/主路线与五条兄弟纵深页互链；链接 wiki/tasks/stair-obstacle-perceptive-locomotion.md 等枢纽页

## [2026-07-10] structural | roadmap/depth-classical-control.md — 新增传统模型控制（LIP/ZMP → MPC → WBC）纵深路线；主页「传统控制」按钮改指该路线，README/index/主路线与四条兄弟纵深页交叉互链

## [2026-07-10] ingest | sources/papers/realm_last_3_meter_vln_arxiv_2607_03792.md — REALM Last-3-Meter VLN 实例接地；wiki/entities/paper-realm-last-3-meter-vln-grounding.md、wiki/tasks/vision-language-navigation.md、wiki/entities/paper-vln-03-reverie.md、wiki/entities/paper-vln-09-etpnav.md

## [2026-07-10] ingest | sources/papers/evo1_arxiv_2511_04555.md — Evo-1 轻量 VLA（CVPR 2026）；wiki/entities/paper-evo1-lightweight-vla.md；交叉更新 vla.md、lerobot.md

## [2026-07-10] ingest | sources/repos/freecad-mcp.md — 接入 FreeCAD MCP 桥接栈；新增 wiki/entities/freecad-mcp.md，交叉更新 freecad 与 text-to-cad

## [2026-07-09] structural | 清单 v28 P1 具身大模型家族层专题交叉补强 — vla / vision-language-navigation / world-action-models / generative-world-models 四页与 `queries/embodied-fm-taxonomy-loop.md` 双向回链并标注所在层（③执行 / ②导航 / ⑤推演联合建模 / ⑤推演级联预演），消孤儿；lint 0 errors、embodied_fm_crosslink 0

## [2026-07-09] ingest | sources/papers/face_anything_arxiv_2604_19702.md — Face Anything 4D 人脸重建与跟踪；wiki/entities/paper-face-anything-4d-face-reconstruction.md；交叉更新 gvhmr / humanoid-training-data-pipeline / visual-representation-for-policy

## [2026-07-09] ingest | sources/repos/dimensionalos_dimos.md — DimOS agent-native 物理空间 OS 入库；wiki/entities/dimensionalos-dimos.md；交叉更新 ros2-basics / ros2-vs-lcm / unitree / unitree-g1 / lerobot / navigation-slam-autonomy-stack

## [2026-07-09] ingest | sources/papers/egowam.md — 新增 EgoWAM 论文实体并交叉更新 WAM/IL/Manipulation；wiki/entities/paper-egowam-egocentric-human-wam-co-training.md

## [2026-07-09] ingest | sources/blogs/dexmal_dm05.md — 接入 Dexmal DM0.5 开放世界 VLA 博客；新建 wiki/entities/dexmal-dm05.md；互链 vla.md

## [2026-07-09] ingest | sources/repos/pbhc.md + kungfubot 项目页 — PBHC 仓库与 KungfuBot/KungfuBot2 实体页升格；wiki/entities/paper-notebook-kungfubot-physics-based-humanoid-whole-body-cont.md、wiki/entities/paper-notebook-kungfubot-2.md

## [2026-07-09] ingest | sources/courses/isaac_lab_implicit_explicit_actuators.md — Isaac Lab/mjlab Implicit vs Explicit 执行器；wiki/concepts/implicit-explicit-actuator-modeling.md；交叉更新 sim2real / armature / actuator-network / legged-humanoid-rl-pd-gain-setting / isaac-lab

## [2026-07-09] ingest | sources/repos/hightorque_robotics.md — 高擎机电 GitHub 组织入库；wiki/entities/hightorque-robotics.md；交叉更新 paper-host-humanoid-standingup / open-source-humanoid-hardware / humanoid-robot

## [2026-07-09] ingest | sources/papers/humanoid_surgeon_nature_2026.md — UCSD Humanoid Surgeon Nature 2026 活体腹腔镜可行性；wiki/entities/paper-humanoid-surgeon-in-vivo-laparoscopy.md；交叉更新 teleoperation / paper-notebook-lapsurgie / paper-notebook-humanoids-in-hospitals

## [2026-07-09] fix(ux) | docs/style.css — TOC 浮窗封顶仅桌面端 min(76vh, 680px)，移动端抽屉恢复原样

## [2026-07-08] ingest | sources/papers/discrete_terrain_minimal_proximity_sensing_arxiv_2606_31912.md — ETH RSL 四足足底 ToF 最小感知离散地形；wiki/entities/paper-discrete-terrain-minimal-proximity-sensing.md；交叉更新 quadruped-robot / terrain-adaptation / stair-obstacle-perceptive-locomotion

## [2026-07-08] ingest | sources/sites/unitree-unistore.md — UniStore 宇树应用平台；wiki/entities/unitree-unistore.md；交叉更新 unitree / unitree-g1

## [2026-07-08] ingest | sources/sites/stackforce-workbench.md、stackforce-cad2urdf.md — StackForce 工作台与 CAD2URDF；wiki/entities/stackforce.md；交叉 step2urdf / urdf-studio / isaac-lab / gpufree

## [2026-07-08] ingest | sources/papers/navwam_arxiv_2606_13494.md — NavWAM 目标条件视觉导航 WAM；wiki/entities/paper-navwam-goal-conditioned-visual-navigation-wam.md；交叉更新 world-action-models / cosmos-policy / vision-language-navigation

## [2026-07-08] ingest | sources/papers/lingbot_vla_v2_tech_report.md — LingBot-VLA 2.0 技术报告入库；wiki/entities/lingbot-vla-v2.md、wiki/methods/vla.md 交叉更新；修正 161 综述 LingBot-VLA 误链至 lingbot-map

## [2026-07-08] ingest | sources/repos/plfm_radar.md — AERIS-10 开源 PLFM 相控阵雷达；wiki/entities/aeris-10-plfm-radar.md；交叉更新 navigation-slam-autonomy-stack / multirotor-simulation-planning-control-stack / topic-state-estimation / field-robotics-troubleshooting

## [2026-07-08] ingest | sources/blogs/wechat_shenlan_tro_manip_5_papers_survey.md — T-RO 2026 操作学习 5 篇精选；wiki/overview/tro-manip-5-papers-technology-map.md、wiki/overview/tro-manip-category-01-data-scaling.md … category-04、wiki/entities/paper-tro-manip-01-diversity-scaling.md … 05、wiki/tasks/manipulation.md

## [2026-07-08] ingest | sources/repos/graph_robots_graph_as_policy.md — 补录 GaP 官方 graph-as-policy 仓库；wiki/entities/paper-gap-graph-as-policy.md 关联 sources/repos 以启用更新记录 ⭐️

## [2026-07-08] ingest | sources/papers/kemo_arxiv_2606_23589.md — 事件驱动关键帧记忆 VLA；wiki/entities/paper-kemo-event-driven-keyframe-memory-vla.md、wiki/tasks/manipulation.md、wiki/methods/vla.md、wiki/methods/pi07-policy.md

## [2026-07-08] ingest | sources/papers/gap_arxiv_2607_05369.md — GaP Graph-as-Policy 变体自动化；wiki/entities/paper-gap-graph-as-policy.md、wiki/concepts/variational-automation.md；交叉更新 manipulation / aspire / vla / nvidia-gear-lab

## [2026-07-08] ingest | sources/papers/omnitactune_arxiv_2607_03723.md — OmniTacTune 策略无关触觉残差真机 RL；wiki/entities/paper-omnitactune-tactile-residual-adaptation.md；交叉更新 visuo-tactile-fusion / tactile-feedback-in-rl / manipulation / contact-rich-manipulation

## [2026-07-08] ingest | sources/papers/athena_wbc_arxiv_2607_04837.md — Athena-WBC 能力对齐专家蒸馏与人形 WBC 训练集长尾；wiki/entities/paper-athena-wbc-humanoid-longtail.md；交叉更新 SONIC / 选型 query / DeepInsight

## [2026-07-07] ingest | sources/papers/quietwalk_arxiv_2604_23702.md — QuietWalk PINN-GRF 低噪人形行走；wiki/entities/paper-quietwalk-humanoid-locomotion.md；交叉更新 humanoid-locomotion / locomotion-reward-design-guide / unitree-g1

## [2026-07-07] ingest | sources/repos/cangjie-skill.md + sources/repos/nuwa-skill.md + sources/repos/darwin-skill.md — 仓颉/女娲/达尔文 skill 生态；wiki/entities/cangjie-skill.md、wiki/entities/nuwa-skill.md、wiki/entities/darwin-skill.md；交叉更新 karpathy-autoresearch / ponytail / superpowers / mattpocock / llm-wiki-karpathy

## [2026-07-06] lint | scripts/lint_wiki.py — V28 P0 具身大模型家族概念页交叉链路巡检 V1（信息型，不阻塞 CI）

- 新增 `_check_embodied_fm_crosslink`：对 `tags` 含 `vlm` / `vln` / `vla` / `vlx` / `world-model`（子串匹配派生标签）的 `wiki/concepts/*`、`wiki/comparisons/*` 页，检查正文是否回链专题枢纽 `embodied-fm-taxonomy-loop` / `topic-embodied-foundation-model`，缺失给 INFO 级 `embodied_fm_crosslink` 提示，枢纽页自身豁免
- 注册 `embodied_fm_crosslink` 至 `INFO_ONLY_KEYS` / `_empty_results` / 报告分节；`make lint` 0 errors，基线快照 `exports/lint-report.md` 首批命中 10 页
- 新增 `tests/test_lint_wiki_embodied_fm_crosslink.py` 8 用例（列表式/内联式 tag、comparisons 页、有/无回链、双枢纽、枢纽豁免、INFO 不计失败），lint_wiki 测试 72 passed
- 勾选 [`tech-stack-next-phase-checklist-v28.md`](docs/checklists/tech-stack-next-phase-checklist-v28.md) P0 项

## [2026-07-06] ingest | sources/repos/cyclo_intelligence.md — ROBOTIS Cyclo Intelligence 行为树×VLA；wiki/entities/cyclo-intelligence.md、wiki/concepts/behavior-tree-vla-orchestration.md；交叉更新 vla / lerobot / topic-vla / vla-deployment-guide / vla-with-low-level-controller

## [2026-07-06] ingest | sources/papers/humanoidarena_arxiv_2606_17833.md — HumanoidArena egocentric 分层全身 benchmark；wiki/entities/paper-humanoidarena.md；交叉更新 wiki/tasks/loco-manipulation.md、wiki/tasks/teleoperation.md、wiki/entities/paper-twist2.md、wiki/methods/sonic-motion-tracking.md

## [2026-07-06] ingest | sources/papers/abot_m05_arxiv_2607_00678.md — ABot-M0.5 移动操作 WAM；wiki/entities/paper-abot-m05-mobile-manipulation-wam.md；交叉更新 wiki/concepts/world-action-models.md、wiki/tasks/loco-manipulation.md、wiki/overview/loco-manip-contact-category-05-vla-world-models.md

## [2026-07-06] structural | scripts/dedupe_loco_manip_161_entities.py — 全量合并 Loco-Manip 161 与其它 survey 重叠的重复实体（74 stub 删除，69 catalog 槽位指向 canonical）

- 新增工具：`scripts/dedupe_loco_manip_161_entities.py`（按 `同题深读`、项目 URL 与姊妹篇映射批量删除 stub、重定向 catalog / category hub / survey source）
- 删除 74 个重复实体（含 `paper-motion-cerebellum-humanoidmimicgen`）；保留 hrl-stack / bfm / amp / loco-manip-8 / methods 等 canonical 页
- 未合并：`paper-loco-manip-161-157-refine-dp`（与 monocular HMR 非同题，保留独立槽位）
- 更新：`sources/papers/humanoid_loco_manip_161_catalog.md`、`scripts/bootstrap_loco_manip_161_entities.py`（`CANONICAL_ENTITY_BY_NUM` 扩至 69 槽位）

## [2026-07-06] structural | wiki/entities/paper-hrl-stack-06-hdmi.md — 合并 HDMI 重复实体页（原 paper-loco-manip-161-110-hdmi 与 paper-hrl-stack-06-hdmi）；更新 Loco-Manip 161 catalog、category-05 hub、接触数据 hub 与 bootstrap_loco_manip_161_entities CANONICAL_ENTITY_BY_NUM

- 删除：`wiki/entities/paper-loco-manip-161-110-hdmi.md`
- 保留 canonical：`wiki/entities/paper-hrl-stack-06-hdmi.md`（双 survey 坐标：42 篇栈 #06 + Loco-Manip 161 #110）
- 相关：`wiki/overview/loco-manip-161-category-05-mocap-human-video.md`、`wiki/overview/loco-manip-contact-category-01-contact-data.md`、`sources/papers/humanoid_loco_manip_161_catalog.md`、`sources/papers/loco_manip_161_survey_110_hdmi.md`

## [2026-07-06] ingest | sources/papers/last_hd_arxiv_2606_23685.md — LaST-HD 潜式物理推理 VLA + OOL Glove；wiki/entities/paper-last-hd-latent-physical-reasoning.md；交叉更新 vla / imitation-learning / topic-cross-embodiment

## [2026-07-05] structural | 执行清单换版 v27 → v28 —— V27（接触力旋量闭环链）P0–P3 全数交付、DoD 逐项达标后新建 V28（具身大模型分类学选型闭环）

- V27 收尾：P3 详情页专题徽标端到端验证通过、DoD lint 0 errors / 节点边数 / 事实库 220 / 社区均衡 / log 记录 五项全绿，清单 0 未勾项
- 换版动作（对齐 [`docs/checklists/README.md`](docs/checklists/README.md) 维护规则）：`git mv` V27 至 [`archive/`](archive/tech-stack-next-phase-checklist-v27.md)；新建 [`tech-stack-next-phase-checklist-v28.md`](docs/checklists/tech-stack-next-phase-checklist-v28.md)；README「当前入口」更新为 v28、「历史执行清单」补 v27
- V28 主题：把近周密集 ingest 的 VLM/VLN/VLA/VLX/World-Model 五大具身模型家族沉淀为「感知理解→空间导航→动作执行→一体化扩展→世界模型推演」贯通选型链；起点基线 1581 节点 / 10909 边 / 220 事实 / 17 专题，目标 ≥1595 / ≥10970 / ≥230 / 第 18 专题「具身大模型」
- P0 lint 巡检 / P1 选型链 query+concept / P2 事实库 +10 矛盾规则 / P3 图谱专题视图+详情徽标，均为 `[ ]` 待后续每日推进

## [2026-07-05] structural | 详情页「所属专题」徽标端到端验证（V27 P3 收尾）——`docs/main.js renderMetaTopicBadges` 对「接触力控」专题命中页自动渲染徽标

- 验证目标：V27 P3「详情页"同专题相关页"提示」项——`renderMetaTopicBadges` → `TF.topicsForNode` 已是单一事实源数据驱动，无需改代码，本次做端到端行为验证并归档证据
- 结果：`detail.html?id=wiki-concepts-contact-force-loop-bandwidth` 的「所属专题」行渲染出 `🤝 接触力控 (Contact Force Control)` 徽标，`href=graph.html?topic=contact-force-control`；同页并列渲染 `✋ 触觉` / `🛡️ 安全微调` / `⚙️ 物理保真度` 徽标，多专题命中互不干扰；空态由 `renderDetailMetaItemRow` 隐藏（对齐 2026-07-05 hidden 行 grid gap 修复）
- 截图归档：`.cursor-artifacts/screenshots/detail-topic-contact-force-control.png`（Playwright/Chromium headless，等待 `data-detailMetaReady=true` 后截图）
- 清单勾稽：[`docs/checklists/tech-stack-next-phase-checklist-v27.md`](docs/checklists/tech-stack-next-phase-checklist-v27.md) P3「详情页同专题相关页提示」项完成，至此 V27 P0–P3 全数交付、DoD 逐项达标

## [2026-07-05] ingest | sources/blogs/wechat_human_five_vit_intro.md — human five ViT入门；wiki/concepts/vision-transformer.md；交叉更新 cnn-vs-vit、vision-backbones、topic-vision-backbone

## [2026-07-05] ingest | sources/blogs/wechat_shenlan_five_embodied_model_taxonomy.md — 深蓝五大具身模型分类；wiki/comparisons/vlm-vln-vla-vlx-world-model-taxonomy.md；交叉更新 vla、vln、topic-vla

## [2026-07-05] ingest | sources/patents/boston_dynamics_legged_robot_patents.md + sources/patents/tesla_robot_knee_joint_wo2024073135.md + sources/papers/autonomous_spot_arxiv_2010_09259.md + sources/papers/spot_rl_distributional_sim2real_arxiv_2504_17857.md — Spot/BD 专利栈、Tesla 膝部专利、NeBula 自主探索与 Spot RL Sim2Real；wiki/entities/paper-autonomous-spot-nebula-exploration.md、wiki/entities/paper-spot-rl-distributional-sim2real.md、wiki/entities/patent-boston-dynamics-legged-control-stack.md、wiki/entities/patent-tesla-robot-knee-joint-assembly.md；交叉更新 boston-dynamics、quadruped-robot

## [2026-07-04] structural | docs/topic-filters.js + docs/graph.html + wiki/overview/topic-contact-force-control.md（V27 P3）——新增「🤝 接触力控」图谱专题视图（第 17 项）

- 单一事实源 `docs/topic-filters.js` 新增 `contact-force-control` 专题：干净片段并集 `impedance/admittance/wrench/force/compliance/forcecontrol`（刻意剔除过宽的 `contact`、易误命中的 `hybrid`，与 `physics-fidelity`/`tactile`/`grasp` 保持最小重叠），并用 `ids` 显式纳入四层闭环感知/操作页（`contact-wrench-closed-loop`/`contact-rich-manipulation-guide`/`contact-force-loop-bandwidth`/`contact-estimation`/`visuo-tactile-fusion`/`contact-rich-manipulation`）
- `docs/graph.html` `#filter-topic-chips` 增加对应 chip；新建汇总枢纽页 [`wiki/overview/topic-contact-force-control.md`](wiki/overview/topic-contact-force-control.md)，从 query（`contact-wrench-closed-loop`）/concept（`contact-force-loop-bandwidth`）双向回链
- 校验：`make lint` 0 errors（仅 1 条既有陈旧页预警，与本次无关）；重跑 `generate_link_graph` → 1575 节点 / 10853 边 / 0 orphans、枢纽页 15 条边、`largest_community_ratio` 0.196、`community_quality_warning` false；专题视图筛出 18 个节点，Puppeteer 截图归档 `.cursor-artifacts/screenshots/graph-topic-contact-force-control.png`
- 清单勾稽：[`docs/checklists/tech-stack-next-phase-checklist-v27.md`](docs/checklists/tech-stack-next-phase-checklist-v27.md) P3「图谱页专题视图」项完成，DoD 节点/边数、社区均衡三项一并达标

## [2026-07-04] structural | scripts/generate_link_graph.py — 更新记录「新增/维护」改按 log.md ingest/structural 首次出现判定（修复 git 日期偏早与 lint glob 误展开）

## [2026-07-04] ingest | sources/blogs/wechat_human_five_diffusion_model_intro.md — human five Diffusion Model入门；wiki/concepts/diffusion-model.md；交叉更新 generative-foundations、diffusion-policy、diffusion-motion-generation

## [2026-07-04] ingest | sources/papers/heft_arxiv_2607_02332.md — HEFT 重载全尺寸人形 VR 遥操作（PMG+WPC）；wiki/entities/paper-heft.md、wiki/entities/axellwppr-motion-tracking.md；交叉更新 wiki/tasks/teleoperation.md、wiki/entities/paper-twist2.md

## [2026-07-03] structural | schema/canonical-facts.json 210 → 220（V27 P2）——新增 10 条接触力控矛盾检测规则

- 覆盖：力控带宽↑ 与控制刚度/稳定裕度冲突、阻抗 vs 导纳因果对偶在接触刚度未知时失稳、刚性高带宽与柔顺安全取舍、纯视觉时延致接触前过冲、触觉采样率不足致打滑漏检、混合力位方向选择错误致约束冲突、力旋量估计依赖雅可比/惯量标定、接触离散化致力旋量高估、过度柔顺牺牲定位精度、域随机化不替代真机力标定
- 校验：逐条 pos 命中现存 wiki 页（`contact-force-loop-bandwidth` / `impedance-control` / `visuo-tactile-fusion` / `hybrid-force-position-control` / `contact-estimation` / `contact-wrench-closed-loop`），`make lint` 潜在矛盾 0 个、0 errors；`make ci-preflight` 12/12 通过
- 清单勾稽：[`docs/checklists/tech-stack-next-phase-checklist-v27.md`](docs/checklists/tech-stack-next-phase-checklist-v27.md) P2 事实库扩展项完成

## [2026-07-03] ingest | sources/blogs/wechat_human_five_jason_peng_flexible_motion_skills.md — human five Jason Peng 更灵活的运动技能学习；wiki/overview/jason-peng-flexible-motion-skill-learning.md；交叉更新 xue-bin-peng、deepmimic、amp-reward、humanoid-rl-motion-control-body-system-stack、PARC

- 工具：已安装 [Panniantong/Agent-Reach](https://github.com/Panniantong/Agent-Reach) v1.5.0（`pip install git+https://github.com/Panniantong/Agent-Reach.git` + [wechat-article-for-ai](https://github.com/bzd6661/wechat-article-for-ai) 至 `~/.agent-reach/tools/`（Camoufox；`playwright==1.49.1`））
- 原始链接：<https://mp.weixin.qq.com/s/b-5UIRB1mkEDcIJlAT2jwg>
- 沉淀页面：[`wiki/overview/jason-peng-flexible-motion-skill-learning.md`](wiki/overview/jason-peng-flexible-motion-skill-learning.md)
- 交叉更新：[`wiki/entities/xue-bin-peng.md`](wiki/entities/xue-bin-peng.md)、[`wiki/methods/deepmimic.md`](wiki/methods/deepmimic.md)、[`wiki/methods/amp-reward.md`](wiki/methods/amp-reward.md)、[`wiki/overview/humanoid-rl-motion-control-body-system-stack.md`](wiki/overview/humanoid-rl-motion-control-body-system-stack.md)、[`wiki/entities/paper-notebook-parc-physics-based-augmentation-with-reinforceme.md`](wiki/entities/paper-notebook-parc-physics-based-augmentation-with-reinforceme.md)、[`sources/README.md`](sources/README.md)、[`sources/repos/panniantong_agent_reach.md`](sources/repos/panniantong_agent_reach.md)

## [2026-07-03] ingest | sources/repos/ponytail.md — 接入 Ponytail 编码代理必要性阶梯技能并交叉更新 caveman/superpowers/mattpocock/llm-wiki/hermes 实体页；wiki/entities/ponytail.md

## [2026-07-03] ingest | sources/papers/simfoundry_arxiv_2606_28276.md — SimFoundry Real2Sim 场景生成与策略评测/训练闭环；wiki/entities/paper-simfoundry-real2sim-scene-generation.md；交叉更新 sim2real、manipulation、simulation-evaluation-infrastructure、nvidia-gear-lab

## [2026-07-03] ingest | sources/papers/mint_rss_2026.md — MINT RSS 2026 频域意图分词与单样本迁移；wiki/entities/paper-mint-vla.md；交叉更新 wiki/methods/vla.md、wiki/formalizations/vla-tokenization.md

## [2026-07-03] ingest | sources/repos/parallel_ankle_joint.md — G1/天工并联踝 IK·FK·雅可比参考实现；wiki/concepts/humanoid-parallel-joint-kinematics.md

## [2026-07-03] ingest | sources/sites/telegate-project.md — TeleGate 门控专家全身遥操作；wiki/entities/paper-telegate.md；交叉更新 wiki/tasks/teleoperation.md

## [2026-07-03] ingest | sources/repos/freecad.md — FreeCAD 开源参数化 CAD 入库；wiki/entities/freecad.md；交叉更新 blender、step2urdf、urdf-robot-description

## [2026-07-03] structural | wiki/entities/paper-human-as-humanoid.md 等 6 篇 — 补全 Loco-Manip 接触专题缺失论文独立节点并挂接五组 category hub

- 新建：`paper-human-as-humanoid`、`paper-humanoidumi`、`paper-vlk-synthetic-loco-manipulation`、`paper-imagine2real-zero-shot-hoi`、`paper-humanoid-dart`、`paper-wolf-vla`
- 交叉更新：`loco-manip-contact-category-01/03/05`、`loco-manip-contact-technology-map`、`sources/blogs/wechat_embodied_ai_lab_loco_manip_contact_survey.md`

## [2026-07-03] ingest | sources/blogs/wechat_embodied_ai_lab_loco_manip_contact_survey.md — 具身智能研究室 Loco-Manip 接触五段链路专题；父节点 loco-manip-contact-technology-map + 五组 loco-manip-contact-category-* 子节点；复用约 30 篇既有 paper 实体，6 篇仅外链不新建节点

- 工具：已安装 [Panniantong/Agent-Reach](https://github.com/Panniantong/Agent-Reach) v1.5.0（`pip install git+...` + 手动安装 [wechat-article-for-ai](https://github.com/bzd6661/wechat-article-for-ai) 至 `~/.agent-reach/tools/`（Camoufox；`playwright==1.49.1` 规避 viewport 协议错误））
- 原始链接：<https://mp.weixin.qq.com/s/UjShbwl8p1h9ukymfiRNaw>
- 沉淀页面：[`wiki/overview/loco-manip-contact-technology-map.md`](wiki/overview/loco-manip-contact-technology-map.md)（**父**）、[`loco-manip-contact-category-01-contact-data.md`](wiki/overview/loco-manip-contact-category-01-contact-data.md) … [`loco-manip-contact-category-05-vla-world-models.md`](wiki/overview/loco-manip-contact-category-05-vla-world-models.md)（**子**）
- 交叉更新：[`wiki/tasks/loco-manipulation.md`](wiki/tasks/loco-manipulation.md)、[`wiki/overview/humanoid-loco-manip-161-papers-technology-map.md`](wiki/overview/humanoid-loco-manip-161-papers-technology-map.md)、[`sources/README.md`](sources/README.md)、[`sources/repos/panniantong_agent_reach.md`](sources/repos/panniantong_agent_reach.md)

## [2026-07-02] lint | scripts/lint_wiki.py 新增 `_check_contact_control_crosslink`（V27 P0）——接触/力控/操作概念页交叉链路巡检 V1，INFO 级不阻塞 CI，回链「接触力旋量闭环」枢纽（contact-wrench-closed-loop / topic-contact-force-control）；新增 tests/test_lint_wiki_contact_control_crosslink.py（7 例）；刷新 exports/lint-report.md 基线（10 页 backlog，0 errors）

## [2026-07-02] ingest | sources/papers/flying_knots_arxiv_2602_21302.md — Flying Knots Task-Level ILC 可变形绳操作；wiki/entities/paper-flying-knots.md、wiki/entities/flying-knots-public.md；交叉更新 manipulation、contact-rich-manipulation

## [2026-07-02] ingest | sources/repos/robot_retargeter.md — robot_retargeter SMPL-X/多机型 mink 重定向；wiki/entities/robot-retargeter.md；交叉更新 motion-retargeting、motion-retargeting-pipeline、soma-retargeter、amass

## [2026-07-02] ingest | sources/papers/vsgraphs_arxiv_2503_01783.md — vS-Graphs 视觉 SLAM+3D 场景图；沉淀 wiki/entities/paper-vs-graphs-visual-slam-scene-graph.md，互链 orb-slam3、navigation-slam-autonomy-stack

## [2026-07-02] ingest | sources/sites/runpod.md 等 — 国外 GPU 云六平台入库；wiki/comparisons/international-gpu-cloud-platforms.md；交叉更新 china-gpu-cloud-platforms、simulator-selection-guide、isaac-lab

## [2026-07-02] ingest | sources/sites/matpool.md、featurize.md、gpushare.md、ai-galaxy.md — 扩展国内 GPU 云平台实体；wiki/entities/matpool.md、wiki/entities/featurize.md、wiki/entities/gpushare.md、wiki/entities/ai-galaxy.md；wiki/comparisons/china-gpu-cloud-platforms.md 统一选型对比；移除 autodl-vs-gpufree

## [2026-07-02] ingest | sources/sites/autodl.md、sources/sites/gpufree.md — AutoDL 与算力自由 GPU 云入库；wiki/entities/autodl.md、wiki/entities/gpufree.md；交叉更新 wiki/entities/isaac-lab.md、wiki/queries/simulator-selection-guide.md

## [2026-07-02] ingest | sources/papers/kung_fu_athlete_bot.md — KungFuAthleteBot 高动态武术数据集与 tracking+recovery；wiki/entities/paper-kungfuathlete-humanoid-martial-arts-tracking.md、wiki/comparisons/humanoid-reference-motion-datasets.md、wiki/tasks/balance-recovery.md

## [2026-07-02] ingest | sources/papers/humanoidmimicgen_arxiv_2605_27724.md — HumanoidMimicGen 全身规划 loco-manip 合成示范；wiki/entities/paper-humanoidmimicgen.md、wiki/tasks/loco-manipulation.md

## [2026-07-02] ingest | sources/papers/gr00t_n1_arxiv_2503_14734.md — 基于 arXiv:2503.14734 与 NVIDIA 白皮书深化 GR00T N1 canonical 实体页

- 论文源：[arXiv:2503.14734](https://arxiv.org/abs/2503.14734)、[GR00T_1_Whitepaper.pdf](https://d1qx31qr3h6wln.cloudfront.net/publications/GR00T_1_Whitepaper.pdf)
- wiki：[paper-hrl-stack-34-gr00t_n1.md](wiki/entities/paper-hrl-stack-34-gr00t_n1.md) — 补充双系统架构、数据金字塔、GR00T-N1-2B 规格与仿真/真机量化评测

## [2026-07-02] structural | wiki/entities/paper-hrl-stack-34-gr00t_n1.md — 合并 GR00T N1 重复实体页（原 paper-loco-manip-161-148-gr00t-n1 与 paper-hrl-stack-34-gr00t_n1）；更新 Loco-Manip 161 catalog、category-09 hub、paper-grail 交叉引用与 bootstrap_loco_manip_161_entities CANONICAL_ENTITY_BY_NUM

- 删除：`wiki/entities/paper-loco-manip-161-148-gr00t-n1.md`
- 保留 canonical：`wiki/entities/paper-hrl-stack-34-gr00t_n1.md`（双 survey 坐标：42 篇栈 #34 + Loco-Manip 161 #148）
- 相关：`wiki/overview/loco-manip-161-category-09-vla-world-models.md`、`sources/papers/humanoid_loco_manip_161_catalog.md`、`sources/papers/loco_manip_161_survey_148_gr00t-n1.md`

## [2026-07-02] ingest | sources/sites/jim-fan.md — Jim Fan 个人主页/NVIDIA 档案/Google Scholar 入库；wiki/entities/jim-fan.md；交叉更新 wiki/entities/nvidia-gear-lab.md、wiki/entities/tairan-he.md、wiki/entities/zhengyi-luo.md

## [2026-07-01] structural | V27 P1 接触/力控层专题交叉补强 — contact-estimation / force-control-basics / hybrid-force-position-control / impedance-control / visuo-tactile-fusion 五页与接触力旋量闭环链新页（contact-wrench-closed-loop / contact-force-loop-bandwidth）形成双向回链，明示「感知①→力旋量②→控制③→操作④」四层定位；重生派生统计（node 1534 / edge 10412，largest_community_ratio 0.162，community_quality_warning=false）

## [2026-07-01] structural | scripts/bootstrap_paper_notebook_knowledge.py — 同步 Humanoid_Robot_Learning_Paper_Notebooks 最新 progress.json / PROGRESS.md：full-map 549 篇、索引 513 篇；新建 4 个 `wiki/entities/paper-notebook-*` 与 11 个 sources；修复 61 处深读 URL；更新 14 类分类父节点与 `humanoid-paper-notebooks-index.md`；修复 `sync_paper_notebook_links.py` 替换 URL 时吞掉 `>` 的 lint 问题

- 数据源：[progress.json](https://github.com/ImChong/Humanoid_Robot_Learning_Paper_Notebooks/blob/main/progress.json)（284 完成 / 87 待读）+ [papers/PROGRESS.md](https://github.com/ImChong/Humanoid_Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
- 工具：`make paper-notebook-bootstrap`、`make paper-notebook-links`、`make paper-notebook-summaries`
- 新建：`wiki/entities/paper-notebook-learning-contact-representation-for-leg-odometry.md`、`wiki/entities/paper-notebook-learning-multi-modal-whole-body-control-for-real.md`、`wiki/entities/paper-notebook-physics-based-motion-tracking-of-contact-rich-in.md`、`wiki/entities/paper-notebook-simulator-adaptation-via-proprioceptive-distribu.md`
- 相关：`wiki/overview/paper-notebook-category-*.md`、`wiki/overview/humanoid-paper-notebooks-index.md`、`schema/paper-notebook-wiki-full-map.yml`

## [2026-07-01] ingest | sources/papers/hrdexdb_arxiv_2604_14944.md — HRDexDB 配对灵巧抓取数据集；wiki/entities/hrdexdb-dataset.md、wiki/tasks/manipulation.md、wiki/queries/dexterous-data-collection-guide.md、wiki/overview/topic-grasp.md

## [2026-07-01] ingest | sources/papers/omnicontact_arxiv_2606_26201.md — OmniContact Contact Flow meta-skill 链式 loco-manipulation；wiki/entities/paper-omnicontact-humanoid-loco-manipulation.md、wiki/entities/omnicontact-sim2sim.md

## [2026-07-01] ingest | sources/papers/argus_dynamic_symmetry_scirobotics_2026.md、sources/repos/argus_general_robotics_lab.md — Argus 动态对称与全向球形腿式机器人（Science Robotics 2026）；wiki/entities/paper-argus-dynamic-symmetry.md、wiki/tasks/locomotion.md、wiki/tasks/loco-manipulation.md、wiki/tasks/balance-recovery.md

## [2026-07-01] ingest | sources/repos/tsil.md — TSIL 官方仓库接入；更新 wiki/entities/paper-tsil-temporal-self-imitation-learning.md

## [2026-07-01] ingest | sources/papers/tsil_arxiv_2606_19752.md — Temporal Self-Imitation Learning 论文入库与 wiki/entities/paper-tsil-temporal-self-imitation-learning.md

## [2026-07-01] ingest | sources/papers/ultra_fusion_arxiv_2606_21223.md — Ultra-Fusion 补全作者与代码/M3DGR 仓库归档；wiki/entities/paper-ultra-fusion-multi-sensor-slam.md

## [2026-07-01] ingest | sources/papers/trex_arxiv_2606_17055.md — T-Rex 触觉反应式灵巧操作；wiki/entities/paper-trex-tactile-reactive-dexterous-manipulation.md、wiki/methods/vla.md、wiki/methods/egoscale.md、wiki/tasks/manipulation.md、wiki/concepts/contact-rich-manipulation.md、wiki/concepts/visuo-tactile-fusion.md、wiki/queries/manipulation-vla-architecture-selection.md

## [2026-07-01] ingest | sources/papers/robustness_robotic_manipulation_dong_2026.md — 操作鲁棒性系统综述；wiki/entities/paper-robustness-robotic-manipulation-survey.md、wiki/tasks/manipulation.md

## [2026-07-01] ingest | sources/papers/aspire_nvidia_gear_2026.md — 接入 NVIDIA ASPIRE 持续学习技能库系统；wiki/methods/aspire.md、wiki/entities/nvidia-gear-lab.md、wiki/methods/enpire.md、wiki/tasks/manipulation.md、wiki/queries/real-robot-policy-autoresearch-harness.md

## [2026-07-01] ingest | sources/papers/gpc_arxiv_2606_29148.md — GPC 生成式预训练物理控制器；wiki/entities/paper-gpc-generative-pretrained-controllers.md

## [2026-07-01] ingest | sources/sites/tnkr-open-duck-mini-v2.md — 接入 Tnkr Open Duck Mini v2 项目文档；交叉更新 wiki/entities/open-duck-mini.md、wiki/entities/tnkr.md、wiki/entities/open-duck-mini-runtime.md

## [2026-07-01] ingest | sources/papers/reactivebfm_arxiv_2606_30362.md — ReactiveBFM 闭环规划–控制；wiki/entities/paper-reactivebfm.md

## [2026-07-01] ingest | sources/papers/fada_arxiv_2606_28476.md — FADA 少样本动力学对齐；wiki/entities/paper-fada-humanoid.md

## [2026-06-30] query | checklist-v27 P1 接触力旋量闭环知识链（+2）—— 新建端到端 Query 与带宽概念页

- 新建 [`wiki/queries/contact-wrench-closed-loop.md`](wiki/queries/contact-wrench-closed-loop.md)：把分散的「① 接触感知/估计 → ② 力旋量表示 → ③ 阻抗/导纳/混合力位控制 → ④ 接触丰富操作策略」四层串成端到端决策链，含 TL;DR 四层定位表、Mermaid 四层决策树、按层归因的失败模式速查；内链回 `contact-estimation` / `force-control-basics` / `hybrid-force-position-control` / `impedance-control` / `visuo-tactile-fusion` 与 CHORD / SceneBot / HapMorph 来源。
- 新建 [`wiki/concepts/contact-force-loop-bandwidth.md`](wiki/concepts/contact-force-loop-bandwidth.md)：力控闭环带宽 ↔ 接触稳定性，明示感知时延、控制刚度、接触离散化（ZOH）三者「短板约束」共同钳住可达带宽，量化震荡/穿透两条边界，并解释「环境越硬阻抗刚度越低」的来源与阻抗/导纳选型耦合。
- 两页正文双向内链互为入链，`make graph` 后 0 orphans；知识图谱 1517→**1519** 节点、10258→**10271** 边；`update_badge.py` 同步 README 徽标。
- `make lint` **0 errors**（仅 3 条既有信息型预警，不阻塞 CI）；勾选 v27 P1「接触力旋量闭环知识链 (+2)」。

## [2026-06-30] ingest | sources/repos/mujoco.md, sources/repos/mujoco_wasm.md — 官方 MuJoCo WASM 绑定与 zalo 社区 demo；wiki/entities/mujoco-wasm.md；交叉 mujoco / robot-viewer

## [2026-06-30] ingest | sources/papers/opencap_monocular_arxiv_2603_24733.md — OpenCap Monocular 单手机生物力学运动学/动力学；wiki/entities/paper-opencap-monocular.md

## [2026-06-30] ingest | sources/repos/en02-op.md — Westwood EN02-OP 开源三指末端；wiki/entities/en02-op.md；交叉 topic-grasp / manipulation

## [2026-06-30] ingest | sources/sites/grail-locomanipulation-huggingface.md — 入库 GRAIL Hugging Face 数据集；wiki/entities/grail-locomanipulation-dataset.md

## [2026-06-30] ingest | sources/papers/grail_arxiv_2606_05160.md — 合并重复 GRAIL 实体页并深读入库；wiki/entities/paper-grail.md

## [2026-06-30] structural | 删除冗余详情节点 wiki-queries-sim2real-deployment-checklist

- 删除：`wiki/queries/sim2real-deployment-checklist.md`（内容已并入 `sim2real-checklist.md`「快速部署检查」节，重定向桩不再保留）
- 交叉更新：`wiki/queries/sim2real-checklist.md`、`wiki/queries/README.md`

## [2026-06-30] structural | 移除冗余 Sim2Real 详情节点 — tech-map/modules/rl/sim2real.md、references/papers/sim2real.md

- 删除：`tech-map/modules/rl/sim2real.md`（tech-node-rl-sim2real 空桩）、`references/papers/sim2real.md`（reference-papers-sim2real 与 wiki/concepts/sim2real 重复）
- 交叉更新：`wiki/concepts/sim2real.md`、`wiki/concepts/system-identification.md`、`wiki/methods/crisp-real2sim.md`、`references/papers/README.md`；内链改指向 wiki 概念页与 comparisons/sim2real-approaches


- **P0 合并**：`wiki/queries/sim2real-deployment-checklist.md` 内容并入 `wiki/queries/sim2real-checklist.md`「快速部署检查」节；原页保留重定向桩
- **P0 瘦身**：`wiki/queries/sim2real-gap-reduction.md` 删除重复 Pipeline checklist；`wiki/concepts/sim2real.md` 主要方法改为链向 `comparisons/sim2real-approaches.md`
- **P1 搜索**：`scripts/search_wiki_core.py` — paper-notebook stub/planned 降权；sim2real 部署/gap/调试意图提权
- **P2 图谱**：`docs/topic-filters.js` — sim2real 专题 segments 移除宽泛 `domain`
- 交叉更新：`topic-sim2real.md`、`robot-policy-debug-playbook.md`、`domain-randomization-guide.md` 等；`schema/search-regression-cases.json`

## [2026-06-30] ingest | sources/blogs/flexion_reflect_v1_0.md — Flexion Reflect v1.0 长程人形自主平台；wiki/entities/flexion-reflect-v1.md；交叉 loco-manipulation / VLA / WBC

## [2026-06-30] ingest | sources/papers/cwi_arxiv_2606_27676.md — CWI 复合全身模仿 loco-manipulation；wiki/entities/paper-cwi-composite-humanoid-whole-body-imitation.md；交叉 loco-manipulation / teleoperation

## [2026-06-29] structural | checklist-v26 P3 详情页「物理保真度」专题徽标端到端验证 —— 复用单一事实源补归档截图

- 详情页「所属专题」徽标行（[`docs/main.js`](docs/main.js) `renderMetaTopicBadges`）本就以 [`docs/topic-filters.js`](docs/topic-filters.js) 为单一事实源、`topicsForNode` 数据驱动：V26 P3 把 `physics-fidelity` 写入单一事实源后，动力学/仿真/新建页命中即自动渲染「⚙️ 物理保真度」徽标并跳 `graph.html?topic=physics-fidelity`，空态降级隐藏整行，无需二次实现。
- node 逐页校验 `contact-dynamics` / `physics-fidelity-sim2real-gap` / `simulation-physics-fidelity` / `articulated-body-algorithms` / `joint-friction-models` / `topic-physics-fidelity` 等候选页全部稳定命中 `physics-fidelity`（全库 86 节点）。
- Puppeteer 截图归档 [`detail-topic-physics-fidelity.png`](.cursor-artifacts/screenshots/detail-topic-physics-fidelity.png)：`contact-dynamics` 详情页「所属专题」行实测渲染「✋ 触觉 + ⚙️ 物理保真度」双徽标。
- `make lint` 0 errors（仅 2 条信息型预警，不阻塞 CI）；勾选 v26 P3「详情页『同专题相关页』提示」与 DoD「make lint 0 errors」「log.md 记录」三项，清单全数完成。
- v26 全数完成后按维护规则新建 [`tech-stack-next-phase-checklist-v27.md`](docs/checklists/tech-stack-next-phase-checklist-v27.md)（聚焦「接触力旋量闭环」知识链：感知/估计 → 力旋量 → 阻抗/导纳/混合力位控制 → 接触丰富操作策略），把 v26 移入 `archive/` 并刷新 `docs/checklists/README.md` 当前入口/历史链接。

## [2026-06-29] ingest | sources/papers/humanoid_pnb_vmp.md — VMP β-VAE motion prior + 条件 PPO 全身跟踪；wiki/entities/paper-notebook-vmp.md；交叉 whole-body-tracking-pipeline / character-animation-vs-robotics / humanoid-motion-tracking-method-selection

## [2026-06-29] structural | wiki/entities/paper-sonic.md — 合并 Loco-Manip 161 重复 SONIC stub（#019/#103）至 canonical 实体 + 方法页

- 删除：[`paper-loco-manip-161-019-sonic.md`](wiki/entities/paper-loco-manip-161-019-sonic.md)、[`paper-loco-manip-161-103-sonic.md`](wiki/entities/paper-loco-manip-161-103-sonic.md)
- 保留 canonical：[`paper-sonic.md`](wiki/entities/paper-sonic.md) + [`sonic-motion-tracking.md`](wiki/methods/sonic-motion-tracking.md)
- 交叉更新：Loco-Manip 161 category 01/04 表、[`humanoid_loco_manip_161_catalog.md`](sources/papers/humanoid_loco_manip_161_catalog.md)、对应 source 映射；[`bootstrap_loco_manip_161_entities.py`](scripts/bootstrap_loco_manip_161_entities.py) 增加 `CANONICAL_ENTITY_BY_NUM` 防再生

## [2026-06-29] ingest | sources/papers/chord_nvidia_video_to_data_2026.md — CHORD 接触力旋量引导灵巧操作；wiki/entities/paper-chord-contact-wrench-dexterous-manipulation.md；交叉 contact-rich-manipulation / manipulation / SPIDER / dexterous-data-pipeline / Isaac Lab

## [2026-06-29] ingest | sources/papers/scenebot_arxiv_2606_27581.md — 接入 SceneBot contact-prompted 全身场景交互跟踪；沉淀 wiki/entities/paper-scenebot.md；交叉更新 SONIC、运动跟踪选型、loco-manipulation、OmniRetarget

## [2026-06-29] ingest | sources/papers/hapmorph_arxiv_2509_05433.md — HapMorph AFPA 可穿戴气动解耦尺寸+刚度；wiki/entities/paper-hapmorph-pneumatic-haptic-render.md + teleoperation/topic-tactile 交叉

## [2026-06-28] feat(ui): V26 P3 — 图谱页"物理保真度"专题视图（专题扩至 16 项）

- 改动：[`docs/topic-filters.js`](docs/topic-filters.js) 新增 `physics-fidelity` 专题（`TOPIC_HUB_IDS` / `TOPIC_FILTERS` / `TOPIC_META`），复用 path 片段并集机制（`dynamics/contact/friction/articulated/body/differentiable/simulation/urdf/floating/centroidal/fidelity`）并按需 `ids` 显式纳入新建 query/concept；[`docs/graph.html`](docs/graph.html) `#filter-topic-chips` 增加 `data-topic="physics-fidelity"`（⚙️ 物理保真度）chip
- 新页：[`wiki/overview/topic-physics-fidelity.md`](wiki/overview/topic-physics-fidelity.md) 专题汇总枢纽，并从 [`simulation-physics-fidelity`](wiki/queries/simulation-physics-fidelity.md) / [`physics-fidelity-sim2real-gap`](wiki/concepts/physics-fidelity-sim2real-gap.md) 回链消除孤儿
- 校验：`make lint` 0 errors；`graph-stats.json` 0 孤儿；专题命中 **85** 节点；派生站点文件同步至 1512 节点/10143 边并刷新 badge；Puppeteer 截图归档 `.cursor-artifacts/screenshots/graph-topic-physics-fidelity.png`（页头实测 `85 / 1512 节点`）
- 清单：[`docs/checklists/tech-stack-next-phase-checklist-v26.md`](docs/checklists/tech-stack-next-phase-checklist-v26.md) P3 图谱专题视图项打勾

## [2026-06-28] ingest | sources/sites/rek-com.md — REK VR 人形格斗联赛；wiki/entities/rek.md + 交叉 unitree-g1 / teleoperation / robostriker

## [2026-06-28] ingest | sources/repos/gymnasium.md — Gymnasium RL 环境 API 标准；wiki/entities/gymnasium.md + 交叉 mujoco / dm-control / reinforcement-learning / gym-pybullet-drones / sim-platforms-decade

## [2026-06-28] ingest | sources/papers/flap_arxiv_2606_17630.md — FLAP 无先验地图 FOV 主动感知 3D UAV 规划；wiki/entities/paper-flap-fov-active-perception-3d-navigation.md + 交叉 multirotor-simulation-planning-control-stack / ego-planner-swarm

## [2026-06-27] feat(facts): V26 P2 — 事实库扩展 12 条物理保真度矛盾检测规则（198 → 210）

- 改动：[`schema/canonical-facts.json`](schema/canonical-facts.json) 由 198 → **210** 条，新增 12 条围绕「仿真物理保真度链路」的矛盾检测规则：接触保真度↑ 与可微性/吞吐冲突、几何/URDF 惯量误差被上层逐级放大、硬接触穿透致冲击力偏大、库仑摩擦低估静摩擦致打滑、理想力矩源致执行器力矩 gap、可微仿真梯度受接触不连续制约、硬 LCP 接触不可微、积分步长过大致能量漂移/发散、软接触引入穿透与虚假阻尼、域随机化覆盖残差非替代保真度、保真度+SysID 互补、几何/URDF 层最便宜应优先做
- 校验：逐条经脚本核对对现存 wiki 页（[`simulation-physics-fidelity`](wiki/queries/simulation-physics-fidelity.md) / [`physics-fidelity-sim2real-gap`](wiki/concepts/physics-fidelity-sim2real-gap.md) / `contact-dynamics` / `joint-friction-models` / `differentiable-simulation` / `urdf-robot-description`）有 pos 命中且 0 误报；`make lint` 潜在矛盾 **0** 个、0 errors；`ci-preflight` 12/12 通过
- 清单：[`docs/checklists/tech-stack-next-phase-checklist-v26.md`](docs/checklists/tech-stack-next-phase-checklist-v26.md) P2 与 DoD 事实库项打勾

## [2026-06-27] ingest | sources/papers/second_order_optimizers.md — 6 类二阶/拟牛顿优化器；wiki/methods/newtons-method.md、gauss-newton.md、levenberg-marquardt.md、bfgs.md、l-bfgs.md、truncated-newton.md + wiki/comparisons/second-order-optimizers.md + 交叉 quasi-newton-bfgs / line-search / convex-functions

## [2026-06-27] ingest | sources/papers/deep_learning_optimizers.md — 9 类深度学习优化器一手资料；wiki/methods/sgd.md、sgd-momentum.md、nesterov-momentum.md、adagrad.md、rmsprop.md、adadelta.md、adam.md、adamw.md、lion.md + wiki/comparisons/deep-learning-optimizers.md + 交叉 deep-learning-foundations / backpropagation

## [2026-06-27] ingest | sources/blogs/thehumanoid_kinetiq_ascend.md — Humanoid KinetIQ Ascend 真机 CFM-VLA PPO；wiki/entities/kinetiq-ascend.md + 交叉 VLA/BC/manipulation

- 原始资料：[thehumanoid_kinetiq_ascend.md](sources/blogs/thehumanoid_kinetiq_ascend.md)（<https://thehumanoid.ai/technology/kinetiq-ascend/>）
- 沉淀页面：[wiki/entities/kinetiq-ascend.md](wiki/entities/kinetiq-ascend.md)
- 交叉更新：[wiki/methods/vla.md](wiki/methods/vla.md)、[wiki/methods/behavior-cloning.md](wiki/methods/behavior-cloning.md)、[wiki/tasks/manipulation.md](wiki/tasks/manipulation.md)、[schema/institutions.json](schema/institutions.json)

## [2026-06-26] feat(lint): V26 P0 — 动力学/仿真概念页交叉链路巡检 `physics_concept_crosslink`（INFO 级）

- 改动：[`scripts/lint_wiki.py`](scripts/lint_wiki.py) 新增 `_check_physics_concept_crosslink`——对 `tags` 含 `dynamics`/`simulation`/`physics` 的 `wiki/concepts/*` 与 `wiki/formalizations/*` 概念页，检查正文是否回链「仿真物理保真度」专题枢纽（[`simulation-physics-fidelity`](wiki/queries/simulation-physics-fidelity.md) / [`physics-fidelity-sim2real-gap`](wiki/concepts/physics-fidelity-sim2real-gap.md)），缺失给 INFO 级提示不阻塞 CI；枢纽页自身豁免；同时支持列表式与内联式 `tags`
- 测试：新增 [`tests/test_lint_wiki_physics_crosslink.py`](tests/test_lint_wiki_physics_crosslink.py) 6 用例（有/无回链、列表式/内联式 tag、枢纽豁免、INFO 不计失败）全绿；`ruff` / `mypy` 通过
- 基线快照：[`exports/lint-report.md`](exports/lint-report.md) 现 **15** 页待回链；P1 已回链的 5 页（contact-dynamics / joint-friction-models / articulated-body-algorithms / differentiable-simulation / urdf-robot-description）正确豁免；`make lint` 0 errors
- 清单：[`docs/checklists/tech-stack-next-phase-checklist-v26.md`](docs/checklists/tech-stack-next-phase-checklist-v26.md) P0 打勾

## [2026-06-26] structural(wiki): Loco-Manip 161 与 paper-notebook stub 去重合并 — 33 对并入 paper-loco-manip-161-* / genie-sim-3

- 工具：`make paper-notebook-dedupe`（[dedupe_paper_notebook_nodes.py](scripts/dedupe_paper_notebook_nodes.py)）
- 合并：33 对 `paper-notebook-*` stub → 对应 [`wiki/entities/paper-loco-manip-161-{NNN}-*.md`](wiki/entities/)（含 [`genie-sim-3.md`](wiki/entities/genie-sim-3.md) ← Genie Sim 3.0 stub）；删除 34 条冗余 `sources/papers/humanoid_pnb_*` stub source
- 复跑判据：loco-manip 相关 dedupe 对 **0** 残留
- 交叉更新：[`schema/paper-notebook-wiki-full-map.yml`](schema/paper-notebook-wiki-full-map.yml)、若干 category 页与引用 stub 的 wiki 页

## [2026-06-26] ingest | sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md — 人形 Loco-Manip 161 篇十方向全景；父节点 + 十组 category 子节点 + 161 篇 paper-loco-manip-161-* 独立实体

- 原始资料：[wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md](sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md)、[wechat_humanoid_loco_manip_161_2026-06-26.md](sources/raw/wechat_humanoid_loco_manip_161_2026-06-26.md)、[humanoid_loco_manip_161_catalog.md](sources/papers/humanoid_loco_manip_161_catalog.md)、`sources/papers/loco_manip_161_survey_{001..161}_*.md`
- 工具：Agent Reach v1.5.0 + wechat-article-for-ai（Camoufox）；[bootstrap_loco_manip_161_entities.py](scripts/bootstrap_loco_manip_161_entities.py)；<https://mp.weixin.qq.com/s/pACh9EhsISiyPGdiiR0C3A>
- 沉淀页面：[`wiki/overview/humanoid-loco-manip-161-papers-technology-map.md`](wiki/overview/humanoid-loco-manip-161-papers-technology-map.md)（**父**）、[`loco-manip-161-category-01-motion-base-wbt.md`](wiki/overview/loco-manip-161-category-01-motion-base-wbt.md) … [`loco-manip-161-category-10-ego-video.md`](wiki/overview/loco-manip-161-category-10-ego-video.md)（**子**）、**161** 篇 [`wiki/entities/paper-loco-manip-161-{NNN}-*.md`](wiki/entities/)（**独立节点**；与姊妹篇重叠者在 `related` 交叉链深读页）
- 交叉更新：[`wiki/tasks/loco-manipulation.md`](wiki/tasks/loco-manipulation.md)、[`wiki/overview/humanoid-motion-cerebellum-technology-map.md`](wiki/overview/humanoid-motion-cerebellum-technology-map.md)、[`sources/README.md`](sources/README.md)、[`sources/repos/panniantong_agent_reach.md`](sources/repos/panniantong_agent_reach.md)

## [2026-06-26] ingest | sources/blogs/wechat_embodied_ai_lab_agibot_june_2026_release.md — 智元 2026-06 发布七段落地链路；父节点 agibot-june-2026-release-technology-map + 六组 agibot-release-category-* 子节点 + 七项目实体

- 原始资料：[wechat_embodied_ai_lab_agibot_june_2026_release.md](sources/blogs/wechat_embodied_ai_lab_agibot_june_2026_release.md)、[wechat_agibot_june_2026_release_2026-06-26.md](sources/raw/wechat_agibot_june_2026_release_2026-06-26.md)
- 工具：已安装 [Panniantong/Agent-Reach](https://github.com/Panniantong/Agent-Reach) v1.5.0（`pip install` + 手动安装 [wechat-article-for-ai](https://github.com/bzd6661/wechat-article-for-ai) 至 `~/.agent-reach/tools/`（Camoufox））
- 沉淀页面：[`wiki/overview/agibot-june-2026-release-technology-map.md`](wiki/overview/agibot-june-2026-release-technology-map.md)（**父**）、[`agibot-release-category-01-data-entry.md`](wiki/overview/agibot-release-category-01-data-entry.md) … [`agibot-release-category-06-application-delivery.md`](wiki/overview/agibot-release-category-06-application-delivery.md)（**子**）、[`agibot-world-2026.md`](wiki/entities/agibot-world-2026.md)、[`genie-sim-3.md`](wiki/entities/genie-sim-3.md)、[`go-2.md`](wiki/entities/go-2.md)、[`agibot-bfm-2.md`](wiki/entities/agibot-bfm-2.md)、[`agibot-agile.md`](wiki/entities/agibot-agile.md)、[`genie-studio-agent.md`](wiki/entities/genie-studio-agent.md)；复用 [`ge-sim-2.md`](wiki/entities/ge-sim-2.md)
- 交叉更新：[`bfm-41-papers-technology-map.md`](wiki/overview/bfm-41-papers-technology-map.md)、[`robot-world-models-training-loop-taxonomy.md`](wiki/overview/robot-world-models-training-loop-taxonomy.md)、[`sources/README.md`](sources/README.md)、[`sources/repos/panniantong_agent_reach.md`](sources/repos/panniantong_agent_reach.md)

## [2026-06-25] structural | checklist-v26 P1 — 仿真物理保真度知识链 +2 页落地并完成四层交叉回链

- 新增 Query：[simulation-physics-fidelity.md](wiki/queries/simulation-physics-fidelity.md)（几何/URDF → 刚体动力学（ABA/RNEA）→ 接触/摩擦 → 执行器四层保真度取舍决策树，配 Mermaid，覆盖每层对 sim2real gap 的贡献/建模成本/典型失败模式）
- 新增 Concept：[physics-fidelity-sim2real-gap.md](wiki/concepts/physics-fidelity-sim2real-gap.md)（物理保真度 ↔ sim2real gap 因果分层，明示各层简化如何转化为可观测 gap，及与域随机化/系统辨识的互补关系）
- 交叉回链：[contact-dynamics](wiki/concepts/contact-dynamics.md)、[joint-friction-models](wiki/concepts/joint-friction-models.md)、[urdf-robot-description](wiki/concepts/urdf-robot-description.md)、[differentiable-simulation](wiki/concepts/differentiable-simulation.md)、[articulated-body-algorithms](wiki/formalizations/articulated-body-algorithms.md) 五页与新页双向回链，消除孤儿页
- 图谱：节点 1336→1338、边 9109→9139；社区重分区后新增 `humanoid-soccer` 社区，补 `COMMUNITY_NAME_OVERRIDES` 命名 override（`community_quality_warning=false`，`largest_community_ratio=0.183`）
- 门禁：`make lint` 0 问题；`tests/test_community_naming`、`test_generate_link_graph_communities` 等单测通过

## [2026-06-25] ingest | sources/papers/lhbs_learning_human_like_badminton_skills_arxiv_2602_08370.md — LHBS Imitation-to-Interaction 四阶段羽毛球技能；升格 wiki/entities/paper-notebook-learning-human-like-badminton-skills-for-humanoi.md

## [2026-06-25] ingest | sources/repos/tensorrt-official.md + openvino-official.md + ncnn-official.md — 补全 TensorRT 实体并扩展 OpenVINO/ncnn 与机载推理选型对比

- 原始资料：[tensorrt-official.md](sources/repos/tensorrt-official.md)、[openvino-official.md](sources/repos/openvino-official.md)、[ncnn-official.md](sources/repos/ncnn-official.md)
- 升格实体：[tensorrt.md](wiki/entities/tensorrt.md)、[openvino.md](wiki/entities/openvino.md)、[ncnn.md](wiki/entities/ncnn.md)
- 更新对比：[onnxruntime-vs-mnn-vs-tensorrt.md](wiki/comparisons/onnxruntime-vs-mnn-vs-tensorrt.md)（延伸 LiteRT/ExecuTorch/LibTorch 一览）
- 交叉更新：[onnx.md](wiki/entities/onnx.md)、[onnxruntime.md](wiki/entities/onnxruntime.md)、[mnn.md](wiki/entities/mnn.md)

## [2026-06-25] ingest | sources/repos/onnx-official.md + onnxruntime-official.md + mnn-official.md — 接入 ONNX/MNN 一手资料并升格实体与 runtime 选型对比

- 原始资料：[onnx-official.md](sources/repos/onnx-official.md)、[onnxruntime-official.md](sources/repos/onnxruntime-official.md)、[mnn-official.md](sources/repos/mnn-official.md)
- 升格实体：[onnx.md](wiki/entities/onnx.md)、[onnxruntime.md](wiki/entities/onnxruntime.md)、[mnn.md](wiki/entities/mnn.md)
- 选型对比：[onnxruntime-vs-mnn-vs-tensorrt.md](wiki/comparisons/onnxruntime-vs-mnn-vs-tensorrt.md)
- 交叉更新：[pytorch.md](wiki/entities/pytorch.md)、[tensorflow.md](wiki/entities/tensorflow.md)、[sim2real.md](wiki/concepts/sim2real.md)、[whole-body-tracking-pipeline.md](wiki/concepts/whole-body-tracking-pipeline.md)、[robot-policy-debug-playbook.md](wiki/queries/robot-policy-debug-playbook.md)

## [2026-06-25] ingest | sources/repos/tensorflow-official.md — 接入 TensorFlow 官网与 GitHub 并升格 wiki/entities/tensorflow.md；交叉更新 deep-learning-foundations、pytorch、htwk-gym

- 原始资料：[tensorflow-official.md](sources/repos/tensorflow-official.md)
- 升格实体：[tensorflow.md](wiki/entities/tensorflow.md)
- 交叉更新：[deep-learning-foundations.md](wiki/concepts/deep-learning-foundations.md)、[pytorch.md](wiki/entities/pytorch.md)、[htwk-gym.md](wiki/methods/htwk-gym.md)

## [2026-06-25] ingest | sources/sites/weights-and-biases.md + sources/repos/tensorboard.md — W&B / TensorBoard 实体与选型对比；交叉 amp-mjlab、mjlab、robot-policy-debug-playbook

- 原始资料：[weights-and-biases.md](sources/sites/weights-and-biases.md)、[tensorboard.md](sources/repos/tensorboard.md)
- 升格实体：[weights-and-biases.md](wiki/entities/weights-and-biases.md)、[tensorboard.md](wiki/entities/tensorboard.md)
- 选型对比：[wandb-vs-tensorboard.md](wiki/comparisons/wandb-vs-tensorboard.md)
- 交叉更新：[amp-mjlab.md](wiki/entities/amp-mjlab.md)、[mjlab.md](wiki/entities/mjlab.md)、[robot-policy-debug-playbook.md](wiki/queries/robot-policy-debug-playbook.md)

## [2026-06-25] fix(wiki): 修复 DAgger / BC 页 LaTeX 中 `\theta` 被制表符破坏导致 KaTeX 渲染失败

- 根因：`wiki/methods/dagger.md`、`wiki/methods/behavior-cloning.md` 中 `\theta` 的 `\t` 被存成字面制表符，KaTeX 将 `_heta` 当作无效下标
- 修复：还原为 `\pi_\theta`、`\min_\theta` 等正确 LaTeX

## [2026-06-25] structural(wiki): 重复节点审计修复 — extreme-parkour 合并、BFM 误标 arXiv 清理、TWIST2 更正为 2511.02832

- dedupe：扩展 [scripts/dedupe_paper_notebook_nodes.py](scripts/dedupe_paper_notebook_nodes.py) `find_arxiv_merge_pairs` 扫描全实体；`paper-notebook-extreme-parkour-with-legged-robots` → [extreme-parkour.md](wiki/entities/extreme-parkour.md)；同步 [schema/paper-notebook-wiki-full-map.yml](schema/paper-notebook-wiki-full-map.yml)
- BFM 元数据：移除 5 篇单篇实体误标综述 arXiv `2506.20487`（[paper-bfm-04](wiki/entities/paper-bfm-04-fast-imitation-bfm.md)、[05](wiki/entities/paper-bfm-05-learning-one-representation.md)、[12](wiki/entities/paper-bfm-12-clone.md)、[17](wiki/entities/paper-bfm-17-maskedmimic.md)、[19](wiki/entities/paper-bfm-19-calm.md)）
- TWIST2：全库更正为 arXiv:2511.02832（保留 [paper-twist.md](wiki/entities/paper-twist.md) 的 2505.02833）；[paper-twist2.md](wiki/entities/paper-twist2.md)、sources、BFM 生成器 id 10

## [2026-06-25] structural(wiki): 批量深化 119 篇 survey 策展实体页 — HRL 42 / BFM 41 / VLN 10 / 深蓝 WM 15 / Ego 9 / Loco-Manip 8 / 运动小脑 15 + 脚本

- 工具：[scripts/deepen_survey_stub_pages.py](scripts/deepen_survey_stub_pages.py) — 从 raw 微信抓取与 catalog 元数据编译 `一句话定义` / `核心机制` / `常见误区`；已有深读页（SONIC、BeyondMimic、GMR 等）保留 survey 坐标并链至方法/实体深读
- 栈覆盖：42 篇 RL 身体系统栈、`paper-bfm-*`（41）、`paper-vln-*`（10）、`paper-shenlan-wm-*`（15）、`paper-ego-*`（9）、`paper-loco-manip-*`（8）、`paper-motion-cerebellum-*`（15）、`paper-sonic` / `paper-twist` / `paper-beyondmimic` 等别名节点
- 刻意保留浅页：`paper-notebook-visualmimic`（Humanoid Paper Notebooks 外链索引）
- 总览：[humanoid-rl-motion-control-body-system-stack.md](wiki/overview/humanoid-rl-motion-control-body-system-stack.md) 局限段改为「编译实体页」

## [2026-06-25] ingest | sources/sites/tairan-he.md — 复核 tairanhe.com：OpenAI MTS、博士答辩与 CVPR 2026 VIRAL/DoorMan 等更新

- 原始资料：[tairan-he.md](sources/sites/tairan-he.md)
- 更新实体：[tairan-he.md](wiki/entities/tairan-he.md)

## [2026-06-25] ingest | sources/sites/yanjieze.md — Yanjie Ze 个人主页归档并升格 wiki/entities/yanjie-ze.md；交叉 TWIST/TWIST2/GMR/VisualMimic/ResMimic 等

- 原始资料：[yanjieze.md](sources/sites/yanjieze.md)
- 升格实体：[yanjie-ze.md](wiki/entities/yanjie-ze.md)
- 交叉更新：[paper-twist.md](wiki/entities/paper-twist.md)、[paper-twist2.md](wiki/entities/paper-twist2.md)、[motion-retargeting-gmr.md](wiki/methods/motion-retargeting-gmr.md)、[paper-notebook-visualmimic.md](wiki/entities/paper-notebook-visualmimic.md)、[paper-resmimic.md](wiki/entities/paper-resmimic.md)
## [2026-06-25] fix(actions): COMMUNITY_NAME_OVERRIDES 补全身运动跟踪流水线 — 修复 community-12 命名不符合 中文（English） 社区 导致 pytest 失败

- `scripts/generate_link_graph.py`：`wiki/concepts/whole-body-tracking-pipeline.md` → `全身运动跟踪流水线（Whole-Body Tracking Pipeline, WBT）`
- 验证：`make ci-preflight`、`make test`（含 `test_community_naming`）通过

## [2026-06-25] structural | AMP 专题 19 篇占位页批量深化收口 — 总览局限段更新为「深读实体页」

- 变更：[humanoid-amp-motion-prior-survey.md](wiki/overview/humanoid-amp-motion-prior-survey.md) 不再将 19 篇标为「策展索引级」；#01–#19 均已 MoRE/CLOT 级深读（#08/#16 先行，#01–#07/#09–#15/#17–#19 本日批次完成）

## [2026-06-25] ingest | AMP 专题 #01–#12 batch deepen — 12 实体页 MoRE/CLOT 级深读 + 策展 source 深读指针 + 5 篇 arXiv/MDPI 归档

- 升格实体（#01–#07、#09–#12）：[paper-amp-survey-01-amp](wiki/entities/paper-amp-survey-01-amp.md)、[paper-amp-survey-02-physics_based_motion_imitation_with](wiki/entities/paper-amp-survey-02-physics_based_motion_imitation_with.md)、[paper-amp-survey-03-smp](wiki/entities/paper-amp-survey-03-smp.md)、[paper-amp-survey-04-kimodo](wiki/entities/paper-amp-survey-04-kimodo.md)、[paper-amp-survey-05-motionbricks](wiki/entities/paper-amp-survey-05-motionbricks.md)、[paper-amp-survey-06-natural_humanoid_robot_locomotion_wi](wiki/entities/paper-amp-survey-06-natural_humanoid_robot_locomotion_wi.md)、[paper-amp-survey-07-adversarial_locomotion_and_motion_im](wiki/entities/paper-amp-survey-07-adversarial_locomotion_and_motion_im.md)、[paper-hiking-in-the-wild](wiki/entities/paper-hiking-in-the-wild.md)、[paper-amp-survey-10-unified_walking_running_and_recovery](wiki/entities/paper-amp-survey-10-unified_walking_running_and_recovery.md)、[paper-adaptive-humanoid-control](wiki/entities/paper-adaptive-humanoid-control.md)、[paper-amp-survey-12-haml](wiki/entities/paper-amp-survey-12-haml.md)
- 未改写（已深）：#08 [paper-amp-survey-08-more](wiki/entities/paper-amp-survey-08-more.md)、#16 CLOT、#13–#19
- 原始资料深读归档：[gmp_generative_motion_prior_arxiv_2503_09015.md](sources/papers/gmp_generative_motion_prior_arxiv_2503_09015.md)、[almi_adversarial_locomotion_motion_imitation_arxiv_2504_14305.md](sources/papers/almi_adversarial_locomotion_motion_imitation_arxiv_2504_14305.md)、[hiking_in_the_wild_arxiv_2601_07718.md](sources/papers/hiking_in_the_wild_arxiv_2601_07718.md)、[adaptive_humanoid_control_ahc_arxiv_2511_06371.md](sources/papers/adaptive_humanoid_control_ahc_arxiv_2511_06371.md)、[haml_humanoid_adversarial_multi_skill_learning_mdpi_2026.md](sources/papers/haml_humanoid_adversarial_multi_skill_learning_mdpi_2026.md)
- 策展索引补强：`humanoid_amp_survey_01`–`12`（#08 既有深读指针保留）
- #10 技术深读主入口：[paper-unified-walk-run-recovery-sdamp](wiki/entities/paper-unified-walk-run-recovery-sdamp.md)

## [2026-06-25] ingest | AMP 专题 #13–#19 深读（Goalkeeper/HUSKY/PhysHSI/TeamHOI/Deep Parkour/Embrace Collisions）— 6 篇 arXiv source + 6 实体页；策展 source 交叉链接

- arXiv 深读：[humanoid_goalkeeper_arxiv_2510_18002.md](sources/papers/humanoid_goalkeeper_arxiv_2510_18002.md)、[husky_humanoid_skateboarding_arxiv_2602_03205.md](sources/papers/husky_humanoid_skateboarding_arxiv_2602_03205.md)、[physhsi_arxiv_2510_11072.md](sources/papers/physhsi_arxiv_2510_11072.md)、[teamhoi_arxiv_2603_07988.md](sources/papers/teamhoi_arxiv_2603_07988.md)、[deep_whole_body_parkour_arxiv_2601_07701.md](sources/papers/deep_whole_body_parkour_arxiv_2601_07701.md)、[embrace_collisions_arxiv_2502_01465.md](sources/papers/embrace_collisions_arxiv_2502_01465.md)
- 升格实体：[paper-amp-survey-13-humanoid_goalkeeper](wiki/entities/paper-amp-survey-13-humanoid_goalkeeper.md)、[paper-amp-survey-14-husky](wiki/entities/paper-amp-survey-14-husky.md)、[paper-amp-survey-15-physhsi](wiki/entities/paper-amp-survey-15-physhsi.md)、[paper-amp-survey-17-teamhoi](wiki/entities/paper-amp-survey-17-teamhoi.md)、[paper-deep-whole-body-parkour](wiki/entities/paper-deep-whole-body-parkour.md)、[paper-amp-survey-19-embrace_collisions](wiki/entities/paper-amp-survey-19-embrace_collisions.md)
- 策展索引补强：`humanoid_amp_survey_13`–`19`（含 #18）、[humanoid_rl_stack_23_deep_whole_body_parkour](sources/papers/humanoid_rl_stack_23_deep_whole_body_parkour.md)

## [2026-06-25] ingest | sources/papers/gmp_generative_motion_prior_arxiv_2503_09015.md 等 — AMP 专题 #06 GMP、#07 ALMI、#09 Hiking、#10 SD-AMP 索引、#11 AHC、#12 HAML 深读实体与 arXiv/MDPI 归档；交叉 humanoid-amp-motion-prior-survey

- 原始资料：[gmp_generative_motion_prior_arxiv_2503_09015.md](sources/papers/gmp_generative_motion_prior_arxiv_2503_09015.md)、[almi_adversarial_locomotion_motion_imitation_arxiv_2504_14305.md](sources/papers/almi_adversarial_locomotion_motion_imitation_arxiv_2504_14305.md)、[hiking_in_the_wild_arxiv_2601_07718.md](sources/papers/hiking_in_the_wild_arxiv_2601_07718.md)、[adaptive_humanoid_control_ahc_arxiv_2511_06371.md](sources/papers/adaptive_humanoid_control_ahc_arxiv_2511_06371.md)、[haml_humanoid_adversarial_multi_skill_learning_mdpi_2026.md](sources/papers/haml_humanoid_adversarial_multi_skill_learning_mdpi_2026.md)
- 升格实体：[wiki/entities/paper-amp-survey-06-natural_humanoid_robot_locomotion_wi.md](wiki/entities/paper-amp-survey-06-natural_humanoid_robot_locomotion_wi.md)、[wiki/entities/paper-amp-survey-07-adversarial_locomotion_and_motion_im.md](wiki/entities/paper-amp-survey-07-adversarial_locomotion_and_motion_im.md)、[wiki/entities/paper-hiking-in-the-wild.md](wiki/entities/paper-hiking-in-the-wild.md)、[wiki/entities/paper-amp-survey-10-unified_walking_running_and_recovery.md](wiki/entities/paper-amp-survey-10-unified_walking_running_and_recovery.md)、[wiki/entities/paper-adaptive-humanoid-control.md](wiki/entities/paper-adaptive-humanoid-control.md)、[wiki/entities/paper-amp-survey-12-haml.md](wiki/entities/paper-amp-survey-12-haml.md)
- 策展索引补强：humanoid_amp_survey_06/07/09/10/11/12
- SD-AMP 深读页（既有）：[wiki/entities/paper-unified-walk-run-recovery-sdamp.md](wiki/entities/paper-unified-walk-run-recovery-sdamp.md)

## [2026-06-25] ingest | sources/papers/more_mixture_residual_experts_arxiv_2506_08840.md — 补完成 MoRE（AMP #08）深读；wiki/entities/paper-amp-survey-08-more.md；交叉 amp-reward、terrain-adaptation、explicit-stair-geometry、locomotion

- 原始资料：[more_mixture_residual_experts_arxiv_2506_08840.md](sources/papers/more_mixture_residual_experts_arxiv_2506_08840.md) — arXiv:2506.08840 + 项目页
- 策展索引补强：[humanoid_amp_survey_08_more_mixture_of_residual_experts_for_humanoid_li.md](sources/papers/humanoid_amp_survey_08_more_mixture_of_residual_experts_for_humanoid_li.md)
- 升格实体：[wiki/entities/paper-amp-survey-08-more.md](wiki/entities/paper-amp-survey-08-more.md) — 两阶段管线、MoE 残差、多判别器 AMP、Mermaid 流程图
- 交叉更新：[paper-notebook-category-05-locomotion](wiki/overview/paper-notebook-category-05-locomotion.md)、[amp-reward](wiki/methods/amp-reward.md)、[locomotion](wiki/tasks/locomotion.md)、[terrain-adaptation](wiki/concepts/terrain-adaptation.md)、[paper-explicit-stair-geometry-humanoid-locomotion](wiki/entities/paper-explicit-stair-geometry-humanoid-locomotion.md)

## [2026-06-25] ingest | sources/papers/oasis_humanoid_loco_manip_2606_08548.md — 复核 OASIS arXiv:2606.08548 入库；交叉补强 wiki/entities/paper-loco-manip-04-oasis.md、wiki/queries/humanoid-training-data-pipeline.md、wiki/concepts/sim2real.md

## [2026-06-25] ingest | sources/sites/kyberlabs-ai.md — Kyber Labs 背驱动灵巧手操作平台；wiki/entities/kyber-labs.md；交叉 notable-commercial-robot-platforms、wuji-robotics、allegro-hand

## [2026-06-25] ingest | sources/sites/wuji_robotics.md — 补正舞肌科技官网 wuji.tech（中/英）与智能数据手套叙事；修正 schema/institutions.json「无界机器人」误标；wiki/entities/wuji-robotics.md

## [2026-06-25] ingest | sources/papers/coordex_arxiv_2606_23680.md — CoorDex body/hand 潜先验协调残差 dexterous loco-manipulation；wiki/entities/paper-coordex-dexterous-humanoid-loco-manipulation.md；交叉 loco-manipulation、wuji-robotics

## [2026-06-25] ingest | sources/sites/hiw-500-dataset.md — HIW-500 野外 G1 遥操作数据集；升格 wiki/entities/hiw-500-dataset.md，互链 teleoperation / humanoid-training-data-pipeline / topic-data-pipeline

## [2026-06-24] checklist-v25 | DoD 收口 & 初始化 V26

- V25 全部条目收口：P0（数据集页元数据巡检 `dataset_metadata_check` + scaffold `--dataset` 旗标）、P1（`humanoid-training-data-pipeline` query + `motion-data-quality` concept + 数据层四段衔接交叉补强）、P2（事实库 186 → 198 条，新增 12 条数据层矛盾检测规则）、P3（图谱第 15 项「训练数据管线」专题视图 `data-pipeline` + 详情页徽标联动）逐条 `[x]`；DoD 末项「`log.md` 记录 V25 关键改动」由本条目收口。
- V25 交付基线：`make lint` 0 errors（仅 1 条信息型预警），图谱 **1322 节点 / 8809 边**，事实库 **198 条**，`community_quality_warning=false`、最大社区占比 **0.165**，图谱专题视图 15 项。
- 新建 [`docs/checklists/tech-stack-next-phase-checklist-v26.md`](docs/checklists/tech-stack-next-phase-checklist-v26.md)：专题选定为「仿真物理保真度链路」，承接 V25 收尾密集 ingest 的 `differentiable-simulation` / `articulated-body-algorithms` / `contact-dynamics` / `joint-friction-models` / `friction-compensation` / `urdf-robot-description` / `procedural-terrain-generation` 等仿真物理底座概念页，规划「几何/URDF → 刚体动力学算法 → 接触/摩擦模型 → 执行器模型」端到端保真度知识链（P1 query+concept）、物理保真度矛盾检测规则扩展（P2 事实库 198→≥208）、动力学/仿真概念页交叉链路巡检（P0）与图谱第 16 项「物理保真度」专题视图（P3）。
- 同步将 README badge / 维护看板、`AGENTS.md`、`docs/README.md`、`docs/checklists/README.md` 的「当前清单」指针从 V25 切到 V26；V25 移入 `archive/` 并修正其内部相对链接（上一版清单同级、方法论参考 `../../../wiki/...`），进入历史归档区。

## [2026-06-24] structural | scripts/generate_link_graph.py — 兜底社区标签改为「其他（Other） 社区」

## [2026-06-24] structural | docs/topic-filters.js、docs/graph.html、wiki/overview/topic-*.md — 专题标签统一为「中文 (English)」格式（与类型图例一致）


## [2026-06-24] ingest | sources/papers/ultra_fusion_arxiv_2606_21223.md — Ultra-Fusion 多传感器 SLAM；沉淀 wiki/entities/paper-ultra-fusion-multi-sensor-slam.md，互链 sensor-fusion、lidar-slam-lio-vio-selection、topic-state-estimation

## [2026-06-24] structural | schema/institutions.json、scripts/sync_institution_tags.py — 实体页机构标签批量补全（表格/sources/覆盖表）

- 扩展机构注册表：中科大、BIGAI、HIT、TeleAI、NYU、Motphys 等 **60+** 条目（46→109）
- 新增 `scripts/sync_institution_tags.py`：从 `|机构|` 表、sources 机构行、GitHub org 与覆盖表写入 frontmatter tags
- 批量更新 **~200** 个 `wiki/entities/` 页（含 HRL 栈、BFM、Ego、深澜 WM、公司/数据集实体）；可派生机构节点 **310→529**（实体非占位 **467/484**）
- 测试：`tests/test_sync_institution_tags.py`

## [2026-06-24] ingest | sources/sites/wokwi-com.md — Wokwi 在线嵌入式仿真平台；升格 wiki/entities/wokwi.md 并交叉 motor-drive-firmware-bus-protocols / simplefoc

- 原始资料：[wokwi-com.md](sources/sites/wokwi-com.md)
- 升格实体：[wiki/entities/wokwi.md](wiki/entities/wokwi.md)
- 交叉更新：[motor-drive-firmware-bus-protocols](wiki/overview/motor-drive-firmware-bus-protocols.md)、[simplefoc](wiki/entities/simplefoc.md)
- 机构注册表：`schema/institutions.json` 追加 Wokwi

## [2026-06-23] structural | schema/institutions.json、scripts/bump_institution_tags.py — 批量补全 wiki 节点所属机构 tags；工具实体 lint 门禁

- 扩展机构注册表（Hugging Face、AI2、Amazon、INRIA、Blender Foundation、X-Humanoid、SDU、RoboParty、FreeMoCap 等）与 Unitree 产品 alias
- 新增 `scripts/bump_institution_tags.py`：从摘要区/H1/显式覆盖表推断机构并写入 frontmatter tags
- 批量更新 ~150 页 wiki（含 Isaac Lab、LeRobot、legged_gym、OpenVLA 等工具实体）；可派生机构节点 89→309
- lint：`tool_missing_institution` 检查工具实体须有所属机构
- 代表页：[wiki/entities/isaac-lab.md](wiki/entities/isaac-lab.md)、[wiki/entities/lerobot.md](wiki/entities/lerobot.md)、[wiki/entities/legged-gym.md](wiki/entities/legged-gym.md)

## [2026-06-23] ingest | sources/sites/nvidia-research-gear-lab.md — NVIDIA GEAR Lab 主页；升格 wiki/entities/nvidia-gear-lab.md 并交叉 EgoScale/ENPIRE/SONIC/GR00T-WBC

## [2026-06-23] ingest | sources/papers/vesta_arxiv_2606_20905.md — Vesta 通才具身 planner；升格 wiki/entities/paper-vesta-generalist-embodied-reasoning.md 并交叉更新 vla / VLN / SayCan

## [2026-06-23] ingest | sources/papers/stubborn_arxiv_2606_12814.md — Stubborn 统一 RL 人形跟踪与跌倒恢复；深读 arXiv:2606.12814 并升格 wiki/entities/paper-motion-cerebellum-stubborn.md

- 原始资料：[stubborn_arxiv_2606_12814.md](sources/papers/stubborn_arxiv_2606_12814.md) — arXiv:2606.12814 + [项目页](https://aislab-sustech.github.io/Stubborn/)
- 策展增补：[motion_cerebellum_survey_34_stubborn.md](sources/papers/motion_cerebellum_survey_34_stubborn.md) — 运动小脑 34/64 索引同步
- 升格实体：[wiki/entities/paper-motion-cerebellum-stubborn.md](wiki/entities/paper-motion-cerebellum-stubborn.md) — yaw-aligned 表征、Bernoulli PT、AdpS 采样、LAFAN1/G1 实验与 Mermaid 流程图

## [2026-06-23] ingest | sources/courses/quadruped_control_simulation_rl_curriculum.md — 具身智能研究室《四足机器人：从动力学建模到强化学习》八章课程大纲；新增 quadruped-control-curriculum 策展页 + 11 个 concept/method/formalization/entity 节点

- 原始资料：[quadruped_control_simulation_rl_curriculum.md](sources/courses/quadruped_control_simulation_rl_curriculum.md) — 四足控制与仿真 RL 课程大纲整理
- 策展入口：[wiki/entities/quadruped-control-curriculum.md](wiki/entities/quadruped-control-curriculum.md)
- 新建 entity：[matrix-simulation-platform](wiki/entities/matrix-simulation-platform.md)、[roamerx-navigation](wiki/entities/roamerx-navigation.md)
- 新建 concept：[differentiable-simulation](wiki/concepts/differentiable-simulation.md)、[urdf-robot-description](wiki/concepts/urdf-robot-description.md)、[joint-friction-models](wiki/concepts/joint-friction-models.md)、[friction-compensation](wiki/concepts/friction-compensation.md)、[procedural-terrain-generation](wiki/concepts/procedural-terrain-generation.md)、[hierarchical-quadruped-navigation-stack](wiki/concepts/hierarchical-quadruped-navigation-stack.md)
- 新建 formalization：[articulated-body-algorithms](wiki/formalizations/articulated-body-algorithms.md)
- 新建 method：[pid-control](wiki/methods/pid-control.md)
- 交叉更新：[quadruped-robot](wiki/entities/quadruped-robot.md)、[system-identification](wiki/concepts/system-identification.md)、[sim2real](wiki/concepts/sim2real.md)、[domain-randomization](wiki/concepts/domain-randomization.md)、[floating-base-dynamics](wiki/concepts/floating-base-dynamics.md)、[simulator-selection-guide](wiki/queries/simulator-selection-guide.md)

## [2026-06-23] structural | wiki/entities/paper-amp-survey-05-motionbricks.md、wiki/methods/motionbricks.md — 消歧 MotionBricks 实体索引页与方法页，加强双向互链

- 实体页 `paper-amp-survey-05-motionbricks`：H1 改为「AMP 专题 #05」、顶部引导至方法页、「与其他页面的关系」补方法页与 Kimodo 对照
- 方法页 `motionbricks`：补 AMP 专题 #05/19 策展语境与回链实体索引页
- 验证：`make ci-preflight`

## [2026-06-23] ingest | sources/courses/numerical_optimization_foundations_robotics.md — 具身智能研究室《数值优化基础》六章课程大纲；新增 numerical-optimization-curriculum 策展页 + 18 个 formalization/method/concept/query 节点

- 原始资料：[numerical_optimization_foundations_robotics.md](sources/courses/numerical_optimization_foundations_robotics.md) — 数值优化基础（机器人应用）课程大纲整理
- 策展入口：[wiki/entities/numerical-optimization-curriculum.md](wiki/entities/numerical-optimization-curriculum.md)
- 新建 formalization：[convex-functions](wiki/formalizations/convex-functions.md)、[kkt-conditions](wiki/formalizations/kkt-conditions.md)、[quadratic-programming](wiki/formalizations/quadratic-programming.md)、[symmetric-cone-programming](wiki/formalizations/symmetric-cone-programming.md)、[adjoint-sensitivity-analysis](wiki/formalizations/adjoint-sensitivity-analysis.md)
- 新建 method：[line-search-steepest-descent](wiki/methods/line-search-steepest-descent.md)、[quasi-newton-bfgs](wiki/methods/quasi-newton-bfgs.md)、[conjugate-gradient-method](wiki/methods/conjugate-gradient-method.md)、[penalty-barrier-augmented-lagrangian](wiki/methods/penalty-barrier-augmented-lagrangian.md)、[nonlinear-model-predictive-control](wiki/methods/nonlinear-model-predictive-control.md)、[time-optimal-path-parameterization](wiki/methods/time-optimal-path-parameterization.md)、[smooth-navigation-path-generation](wiki/methods/smooth-navigation-path-generation.md)、[convex-relaxation-robotics](wiki/methods/convex-relaxation-robotics.md)
- 新建 concept：[constrained-optimization](wiki/concepts/constrained-optimization.md)、[control-allocation](wiki/concepts/control-allocation.md)、[collision-distance-optimization](wiki/concepts/collision-distance-optimization.md)
- 新建 query：[optimization-software-selection](wiki/queries/optimization-software-selection.md)
- 交叉更新：[optimal-control](wiki/concepts/optimal-control.md)、[trajectory-optimization](wiki/methods/trajectory-optimization.md)、[model-predictive-control](wiki/methods/model-predictive-control.md)、[linear-algebra-curriculum](wiki/entities/linear-algebra-curriculum.md)、[roadmap/motion-control.md](roadmap/motion-control.md)

## [2026-06-22] structural | checklist-v25 P3 新增「训练数据管线」图谱专题视图（`data-pipeline`）；docs/topic-filters.js、docs/graph.html、wiki/overview/topic-data-pipeline.md

- 执行清单：[docs/checklists/tech-stack-next-phase-checklist-v25.md](docs/checklists/tech-stack-next-phase-checklist-v25.md) P3「图谱页"训练数据管线"专题视图」收口（V24 14 项 → 15 项专题）
- 单一事实源：[docs/topic-filters.js](docs/topic-filters.js) 的 `TOPIC_HUB_IDS`/`TOPIC_FILTERS`/`TOPIC_META` 新增 `data-pipeline`（emoji 📦、label「训练数据」）；segments=`dataset/datasets/amass/lafan1/lafan/omomo/phuma/everyday/retargeting`，ids 显式纳入 [humanoid-training-data-pipeline](wiki/queries/humanoid-training-data-pipeline.md) query + [motion-data-quality](wiki/concepts/motion-data-quality.md)/[motion-retargeting](wiki/concepts/motion-retargeting.md) concept + [humanoid-reference-motion-datasets](wiki/comparisons/humanoid-reference-motion-datasets.md) 对比
- 新建 hub 页：[wiki/overview/topic-data-pipeline.md](wiki/overview/topic-data-pipeline.md)（专题汇总锚点），并由 [topic-motion-retargeting](wiki/overview/topic-motion-retargeting.md)/[topic-wbt](wiki/overview/topic-wbt.md)「与其他专题的关系」回链消除孤儿
- 交互层：[docs/graph.html](docs/graph.html) `#filter-topic-chips` 新增 `data-topic="data-pipeline"`（📦 训练数据）chip
- 校验：node 载入 topic-filters.js 对 `exports/link-graph.json` 命中 42 节点（数据集 + 重定向 + 质量 + hub）；`make export graph` 重生成派生文件（1288 节点 / 8450 边）；`python3 scripts/update_badge.py` 同步 README badge；`make lint` 0 errors（仅 2 条信息型预警）
- 待补：截图（apt 镜像 404 无法装 Chromium，Puppeteer `graph-topic-data-pipeline.png` 由后续带 Chrome 环境补归档）
- 验证：`make lint`

## [2026-06-22] ingest | sources/papers/htd_refine_arxiv_2605_26879.md — 复核 arXiv:2605.26879（HTD-Refine）已入库；修正 wiki/entities/paper-htd-refine-monocular-hmr.md 英文缩写速查

- 原始资料：[htd_refine_arxiv_2605_26879.md](sources/papers/htd_refine_arxiv_2605_26879.md)（<https://arxiv.org/abs/2605.26879>）；项目页代码仍为 Coming Soon（2026-06-22 复核）
- 沉淀页面：[wiki/entities/paper-htd-refine-monocular-hmr.md](wiki/entities/paper-htd-refine-monocular-hmr.md)（修正 HMR/PVA/MPJVE 等核心缩写表）
- 首次入库：2026-06-04（source + wiki + motion-retargeting / whole-body-tracking / GMR / GVHMR 交叉引用均已就绪）

## [2026-06-22] ingest | sources/papers/x_ionet_arxiv_2511_08277.md — X-IONet 跨平台单 IMU 惯性里程计；wiki/entities/paper-x-ionet-cross-platform-inertial-odometry.md；交叉 state-estimation、ekf、topic-state-estimation

- 原始资料：[x_ionet_arxiv_2511_08277.md](sources/papers/x_ionet_arxiv_2511_08277.md)（<https://arxiv.org/abs/2511.08277>，IEEE RA-L Vol. 11 No. 7, July 2026）
- 沉淀页面：[wiki/entities/paper-x-ionet-cross-platform-inertial-odometry.md](wiki/entities/paper-x-ionet-cross-platform-inertial-odometry.md)
- 交叉更新：[wiki/concepts/state-estimation.md](wiki/concepts/state-estimation.md)、[wiki/formalizations/ekf.md](wiki/formalizations/ekf.md)、[wiki/overview/topic-state-estimation.md](wiki/overview/topic-state-estimation.md)

## [2026-06-22] ingest | sources/papers/rf_detr_arxiv_2511_09554.md — RF-DETR 实时 DETR；wiki/entities/rf-detr.md；交叉 object-detection、object-detection-model-selection

- 原始资料：[rf_detr_arxiv_2511_09554.md](sources/papers/rf_detr_arxiv_2511_09554.md)、[rf_detr.md](sources/repos/rf_detr.md)、[rfdetr-docs.md](sources/sites/rfdetr-docs.md)（<https://arxiv.org/abs/2511.09554>、<https://github.com/roboflow/rf-detr>、<https://rfdetr.roboflow.com/latest/>）
- 沉淀页面：[wiki/entities/rf-detr.md](wiki/entities/rf-detr.md)
- 交叉更新：[wiki/methods/object-detection.md](wiki/methods/object-detection.md)、[wiki/queries/object-detection-model-selection.md](wiki/queries/object-detection-model-selection.md)

## [2026-06-22] ingest | sources/blogs/wechat_shenlan_sim_platforms_top8_decade.md — 十年 TOP 8 仿真平台盘点；wiki/overview/sim-platforms-decade-technology-map.md；8 平台各建实体节点

- 工具：已安装 [Panniantong/Agent-Reach](https://github.com/Panniantong/Agent-Reach) v1.5.0（`pip install` + 手动安装 [wechat-article-for-ai](https://github.com/bzd6661/wechat-article-for-ai) 至 `~/.agent-reach/tools/`（Camoufox））
- 原始资料：`sources/blogs/wechat_shenlan_sim_platforms_top8_decade.md`、`sources/raw/wechat_sim_platforms_top8_2026-06-22.md`（<https://mp.weixin.qq.com/s/iaw_lWAR--AwppyMeIK4lw>）
- 沉淀页面：[`wiki/overview/sim-platforms-decade-technology-map.md`](wiki/overview/sim-platforms-decade-technology-map.md)
- 新建实体：`wiki/entities/ai2-thor.md`、`matterport3d-simulator.md`、`habitat-sim.md`、`igibson.md`、`maniskill2.md`、`behavior-1k.md`、`carla.md`、`robogen.md`
- 交叉更新：[`mujoco.md`](wiki/entities/mujoco.md)、[`isaac-gym.md`](wiki/entities/isaac-gym.md)、[`pybullet.md`](wiki/entities/pybullet.md)、[`genesis-sim.md`](wiki/entities/genesis-sim.md)、[`sapien.md`](wiki/entities/sapien.md)、[`simulator-selection-guide.md`](wiki/queries/simulator-selection-guide.md)、[`sources/README.md`](sources/README.md)、[`sources/repos/panniantong_agent_reach.md`](sources/repos/panniantong_agent_reach.md)

## [2026-06-22] ingest | sources/papers/halomi_arxiv_2606_18772.md — HALOMI 主动感知无机器人示范→人形 loco-manipulation；wiki/entities/paper-halomi-humanoid-loco-manipulation.md；交叉 loco-manipulation、teleoperation

## [2026-06-21] structural | checklist-v25 P2 事实库扩展 +12 条数据层矛盾检测规则（186 → 198 条）；schema/canonical-facts.json

- 执行清单：[docs/checklists/tech-stack-next-phase-checklist-v25.md](docs/checklists/tech-stack-next-phase-checklist-v25.md) P2「事实库扩展」收口（≥ 196 条目标达成，实际 198 条）
- 新增规则（数据层矛盾检测）：纯光学 MoCap 缺力/接触不可直执行、人体视频 3D/接触信息弱、形态差距大重定向不可省略、几何重定向≠物理可执行、PHUMA 物理过滤已重定向免工程、接触一致性为物理可行性前置、规模不能替代物理可行性、真机执行数据天然物理可行但任务窄、四质量轴串联门体检顺序、Humanoid Everyday 非重定向源、已重定向数据集免重定向直接训练、物理不可行参考致 RL 学错力矩
- 校验：逐条经脚本核验对 `motion-data-quality` / `humanoid-training-data-pipeline` / `motion-retargeting` / `humanoid-reference-motion-datasets` 等现存页有 pos 命中、neg 0 命中；`make lint` 0 errors、潜在矛盾 0 条
- 验证：`make lint`

## [2026-06-21] structural | 图谱社区 — 弱归属节点归入「其他社区」；scripts/generate_link_graph.py、docs/graph.html

- 规则：与同社区邻居边占比 &lt; 50% 的非枢纽节点不再强行贴标签，统一落入 `community-other`（其他社区）；图谱图例/筛选始终展示该桶
- 验证：`make ci-preflight`

## [2026-06-21] ingest | sources/repos/spear-sim.md — SPEAR UE 光真实感具身仿真库；wiki/entities/spear-sim.md；交叉 simulator-selection-guide、metahuman、airsim

- 原始资料：[spear-sim.md](sources/repos/spear-sim.md)（<https://github.com/spear-sim/spear>）
- 说明：UE 反射 API、begin_frame/end_frame 事务模型、56 FPS GT 渲染、MuJoCo co-sim、MetaHumans 多视角示例
- 沉淀页面：[wiki/entities/spear-sim.md](wiki/entities/spear-sim.md)
- 交叉更新：[wiki/queries/simulator-selection-guide.md](wiki/queries/simulator-selection-guide.md)、[wiki/entities/metahuman.md](wiki/entities/metahuman.md)、[wiki/entities/airsim.md](wiki/entities/airsim.md)
- 验证：`make ci-preflight`

## [2026-06-21] ingest | sources/papers/gvhmr_arxiv_2409_06662.md — GVHMR Gravity-View 单目 world-grounded HMR；深化 wiki/entities/gvhmr.md

- 原始资料：[gvhmr_arxiv_2409_06662.md](sources/papers/gvhmr_arxiv_2409_06662.md)、[gvhmr-zju3dv-github-io.md](sources/sites/gvhmr-zju3dv-github-io.md)、[gvhmr.md](sources/repos/gvhmr.md)（<https://zju3dv.github.io/gvhmr/>、<https://github.com/zju3dv/GVHMR>）
- 说明：Gravity-View 坐标逐帧 HMR、预处理→Transformer→世界轨迹管线、AMASS/BEDLAM/H36M/3DPW 训练、SimpleVO 工程更新
- 沉淀页面：[wiki/entities/gvhmr.md](wiki/entities/gvhmr.md)
- 验证：`make ci-preflight`
## [2026-06-20] ingest | sources/repos/karpathy-autoresearch.md — Karpathy 单 GPU 自主 LLM 训练实验环；wiki/entities/karpathy-autoresearch.md；交叉 ai-auto-research、andrej-karpathy

## [2026-06-20] ingest | sources/papers/ai_auto_research_survey_2605_18661.md — AI Auto-Research 综述与 Awesome 列表；wiki/concepts/ai-auto-research.md；交叉 llm-wiki-karpathy、agent-reach、hermes-agent、world-action-models

## [2026-06-20] ingest | sources/papers/oscar_arxiv_2606_04463.md — OSCAR 跨具身骨架条件世界模型；wiki/entities/paper-oscar.md；交叉 generative-world-models、roboarena、robot-world-models-training-loop-taxonomy、world-models-route-03-virtual-sandbox

## [2026-06-20] ingest | sources/blogs/wechat_shenlan_vln_10_papers_survey.md — VLN 10 篇技术地图与论文节点

- 工具：已安装 [Panniantong/Agent-Reach](https://github.com/Panniantong/Agent-Reach) v1.5.0（`pip install -e` + `agent-reach install --channels=wechat`）；微信正文经 `~/.agent-reach/tools/wechat-article-for-ai`（Camoufox）
- 原始资料：`sources/blogs/wechat_shenlan_vln_10_papers_survey.md`、`sources/raw/wechat_vln_10_papers_2026-06-20.md`、`sources/papers/vln_10_papers_catalog.md`、`sources/papers/vln_survey_*.md`（10 篇）
- 沉淀页面：[`wiki/overview/vln-10-papers-technology-map.md`](wiki/overview/vln-10-papers-technology-map.md)（父）、[`vln-category-01-datasets-platforms.md`](wiki/overview/vln-category-01-datasets-platforms.md)、[`vln-category-02-algorithm-frameworks.md`](wiki/overview/vln-category-02-algorithm-frameworks.md)（子）、`wiki/entities/paper-vln-01-r2r.md` … `paper-vln-10-navid.md`
- 去重：**NaVid**（RSS 2024，arXiv:2402.15852）≠ **Uni-NaVid**（RSS 2025 导航 VLA 复现栈）
- 交叉更新：[`wiki/tasks/vision-language-navigation.md`](wiki/tasks/vision-language-navigation.md)、[`wiki/overview/vln-open-source-repro-paradigms.md`](wiki/overview/vln-open-source-repro-paradigms.md)、[`sources/README.md`](sources/README.md)、[`sources/repos/panniantong_agent_reach.md`](sources/repos/panniantong_agent_reach.md)
- 验证：`make ci-preflight`

## [2026-06-19] structural | checklist-v25 P1 训练数据管线知识链（+2 页）；新建 wiki/queries/humanoid-training-data-pipeline.md、wiki/concepts/motion-data-quality.md

- 执行清单：[docs/checklists/tech-stack-next-phase-checklist-v25.md](docs/checklists/tech-stack-next-phase-checklist-v25.md) P1 第一项「训练数据管线知识链 (+2)」收口
- 新建页面：
  - [wiki/queries/humanoid-training-data-pipeline.md](wiki/queries/humanoid-training-data-pipeline.md) — 端到端 Query：原始 MoCap / 人体视频 → 重定向 → RL/IL 训练输入三层决策树（含 Mermaid、端到端 pipeline、5 条误区、缩写速查）
  - [wiki/concepts/motion-data-quality.md](wiki/concepts/motion-data-quality.md) — 动作数据质量四轴（物理可行性/接触一致性/形态差距/规模多样性）串联门模型 + 与重定向必要性因果链 + 五集对照
- 交叉补强：[wiki/concepts/motion-retargeting.md](wiki/concepts/motion-retargeting.md)、[wiki/comparisons/humanoid-reference-motion-datasets.md](wiki/comparisons/humanoid-reference-motion-datasets.md) 补入对两新页的入链（无孤儿）
- 验证：`make export graph` → 1257 节点 / 8186 边 / 0 孤儿；`make ci-preflight` 12/12 通过；`lint_wiki` 0 error

## [2026-06-19] ingest | sources/sites/metahuman-epic-docs.md — Epic MetaHuman 官方文档索引；补全 wiki/entities/metahuman.md 文档节

- 原始资料：[metahuman-epic-docs.md](sources/sites/metahuman-epic-docs.md)（<https://dev.epicgames.com/documentation/metahuman/metahuman-documentation>）；侧栏 15 篇子文档摘要
- 说明：Animator 三路径（实时/深度/单目）、UE Cine/Optimized/UEFN 管线、Facial Description Standard、Devkit/OpenRigLogic
- 沉淀页面：[wiki/entities/metahuman.md](wiki/entities/metahuman.md)

## [2026-06-19] ingest | sources/sites/metahuman-com.md — Epic MetaHuman 数字人平台；wiki/entities/metahuman.md；交叉 motion-retargeting、mixamo、blender

- 原始资料：[metahuman-com.md](sources/sites/metahuman-com.md)（<https://www.metahuman.com/>）；复核 5.8 发布说明
- 说明：Creator + Animator 高保真数字人；5.8 全身 Mesh to MetaHuman、单相机无标记全身 Animator、Crowds、OpenRigLogic（MIT）
- 沉淀页面：[wiki/entities/metahuman.md](wiki/entities/metahuman.md)
- 交叉更新：[wiki/concepts/motion-retargeting.md](wiki/concepts/motion-retargeting.md)、[wiki/entities/mixamo.md](wiki/entities/mixamo.md)、[wiki/entities/blender.md](wiki/entities/blender.md)

## [2026-06-19] ingest | sources/repos/crisp_real2sim_repo.md — 校正 CRISP 官方实现为 Z1hanW/CRISP-Real2Sim 并补全 scripts 1–8 + MotionTracking 工程管线

- 原始资料：[crisp_real2sim_repo.md](sources/repos/crisp_real2sim_repo.md)（<https://github.com/Z1hanW/CRISP-Real2Sim>）；复核 [crisp_real2sim_iclr2026.md](sources/papers/crisp_real2sim_iclr2026.md)、[crisp-real2sim-project-github-io.md](sources/sites/crisp-real2sim-project-github-io.md)
- 说明：先前误链 `crisp-real2sim/CRISP-Real2Sim`（实为 GitHub Pages 站点仓）；主代码为作者仓，含 `run_crisp_video.sh`、可选 contact hallucination / NKSR、Google Drive 视频数据集
- 沉淀页面：[wiki/methods/crisp-real2sim.md](wiki/methods/crisp-real2sim.md)（新增「工程实现」节与 Mermaid）

## [2026-06-19] ingest | sources/papers/mujica_arxiv_2605_13058.md — MUJICA 轮足多技能统一控制（wiki/entities/paper-mujica-wheel-legged-multi-skill.md 及轮足/混合运动/sim2real 交叉引用）

- 原始资料：[mujica_arxiv_2605_13058.md](sources/papers/mujica_arxiv_2605_13058.md)（<https://arxiv.org/abs/2605.13058>）；[项目页](https://hyzenthlayer.github.io/mujica/)
- 说明：Go2-W **纯本体** 单策略联合全向移动、高台攀爬、摔倒恢复；**P3O + DC 电机硬约束** + 两阶段技能选择器；真机 **1 m 高台** 与连续多技能链
- 沉淀页面：[wiki/entities/paper-mujica-wheel-legged-multi-skill.md](wiki/entities/paper-mujica-wheel-legged-multi-skill.md)
- 交叉更新：[wiki/concepts/wheel-legged-quadruped.md](wiki/concepts/wheel-legged-quadruped.md)、[wiki/tasks/hybrid-locomotion.md](wiki/tasks/hybrid-locomotion.md)、[wiki/tasks/locomotion.md](wiki/tasks/locomotion.md)、[wiki/concepts/sim2real.md](wiki/concepts/sim2real.md)

## [2026-06-19] ingest | sources/papers/swap_parkour_arxiv_2606_19928.md、sources/sites/swap-parkour-github-io.md — SWAP 对称等变世界模型四足跑酷；wiki/entities/paper-swap-parkour.md；交叉 locomotion、stair-obstacle、extreme-parkour

- 原始资料：[swap_parkour_arxiv_2606_19928.md](sources/papers/swap_parkour_arxiv_2606_19928.md)（<https://arxiv.org/abs/2606.19928>）；[swap-parkour-github-io.md](sources/sites/swap-parkour-github-io.md)（<https://swap-parkour.github.io/>）
- 说明：SE-RSSM 对称等变潜变量世界模型 + 等变 Actor / 不变 Critic 端到端四足跑酷；Apollo 实机 **2.13 m 远跳 / 1.63 m 攀台**；镜像 OOD 与户外零样本泛化
- 沉淀页面：[wiki/entities/paper-swap-parkour.md](wiki/entities/paper-swap-parkour.md)
- 交叉更新：[wiki/tasks/locomotion.md](wiki/tasks/locomotion.md)、[wiki/tasks/stair-obstacle-perceptive-locomotion.md](wiki/tasks/stair-obstacle-perceptive-locomotion.md)、[wiki/entities/extreme-parkour.md](wiki/entities/extreme-parkour.md)

## [2026-06-19] ingest | sources/papers/phygile_arxiv_2603_19305.md、sources/sites/phygile-page.md — PhyGile 文本驱动机器人原生扩散与 GMT 闭环；wiki/entities/paper-phygile.md；交叉 diffusion-motion-generation、humanoid-motion-tracking-method-selection、paper-notebook-gmt

- 原始资料：[phygile_arxiv_2603_19305.md](sources/papers/phygile_arxiv_2603_19305.md)（<https://arxiv.org/abs/2603.19305>）；[phygile-page.md](sources/sites/phygile-page.md)（<https://baojch.github.io/phygile-page/>）
- 说明：physics-prefix 引导的 **262D robot-native** 扩散生成 + 课程式 **MoE GMT** 跟踪器闭环；真机 breakdance、侧手翻、高踢、旋跳等高动态全身动作
- 沉淀页面：[wiki/entities/paper-phygile.md](wiki/entities/paper-phygile.md)
- 交叉更新：[wiki/methods/diffusion-motion-generation.md](wiki/methods/diffusion-motion-generation.md)、[wiki/queries/humanoid-motion-tracking-method-selection.md](wiki/queries/humanoid-motion-tracking-method-selection.md)、[wiki/entities/paper-notebook-gmt.md](wiki/entities/paper-notebook-gmt.md)

## [2026-06-19] ingest | sources/papers/humanoid_gpt_arxiv_2606_03985.md、sources/repos/humanoid_gpt_galaxy_general_robotics.md — Humanoid-GPT 复核：仓库已发布推理/部署与 ONNX checkpoint；wiki/entities/paper-humanoid-gpt.md 补工程节

- 原始资料：[humanoid_gpt_arxiv_2606_03985.md](sources/papers/humanoid_gpt_arxiv_2606_03985.md)（<https://arxiv.org/abs/2606.03985>）；[humanoid_gpt_galaxy_general_robotics.md](sources/repos/humanoid_gpt_galaxy_general_robotics.md)（<https://github.com/GalaxyGeneralRobotics/Humanoid-GPT>）
- 说明：2026-06-04 首入库后复核官方仓库 README——已发布推理/评测/真机部署、`pns_wo_priv216.onnx` checkpoint 与 `projects/{hme,gqs,tracking_transformer}`；训练代码与 2B 数据仍 TODO；补 RoPE、视频估计动作与 G1_VERSION 工程细节
- 沉淀页面：[wiki/entities/paper-humanoid-gpt.md](wiki/entities/paper-humanoid-gpt.md)

## [2026-06-19] ingest | sources/sites/gfr-project.md — 补录 GfR 项目页（RSS 2026）与 PDF 镜像；交叉更新 wiki/methods/mtrg-reference-goal-driven-rl.md、wiki/comparisons/hil-vs-mtrg-vs-zest-parkour-imitation.md、sources/papers/mtrg_reference_goal_driven_rl_arxiv_2602_20375.md

- 原始资料：[`sources/sites/gfr-project.md`](sources/sites/gfr-project.md)（<https://jiashunwang.github.io/GfR/>；PDF：<https://jiashunwang.github.io/GfR/static/mat/gfr_paper.pdf>）
- 说明：arXiv:2602.20375 已于 2026-06-12 以 MTRG 入库；本次补 **GfR** 官方项目名、**RSS 2026** 定稿、长程状态机组合、MuJoCo sim-to-sim 与 elevation map 扩展
- 交叉更新：[`wiki/methods/mtrg-reference-goal-driven-rl.md`](wiki/methods/mtrg-reference-goal-driven-rl.md)、[`wiki/comparisons/hil-vs-mtrg-vs-zest-parkour-imitation.md`](wiki/comparisons/hil-vs-mtrg-vs-zest-parkour-imitation.md)、[`wiki/tasks/humanoid-locomotion.md`](wiki/tasks/humanoid-locomotion.md)

## [2026-06-18] structural | scripts/scaffold_wiki_page.py — V25 P0「数据集选型脚手架强化」：新增 `--dataset` 旗标生成数据集实体骨架（五维度速查块 + `dataset` tag）

- 改动：`scripts/scaffold_wiki_page.py` 新增 `--dataset`（仅 `entity` 类型，否则 rc=2），输出「## 数据集速查」表格覆盖「规模 / 模态 / 许可证 / 适配形态 / 重定向就绪度」并在 frontmatter 写入 `dataset` tag；速查块关键词全覆盖 `lint_wiki._check_dataset_entity_metadata` 四维度，新建数据集页元数据巡检 0 缺失。
- 测试：[`tests/test_scaffold_wiki_page.py`](tests/test_scaffold_wiki_page.py) 新增 3 用例（速查块/tag/位置、lint 巡检 0 缺失、非 entity 拒绝）；`ruff check/format` 与 `python3 scripts/lint_wiki.py` 通过。
- 清单：[`docs/checklists/tech-stack-next-phase-checklist-v25.md`](docs/checklists/tech-stack-next-phase-checklist-v25.md) P0「数据集选型脚手架强化」打勾。

## [2026-06-18] ingest | sources/papers/ume_exo_arxiv_2606_14218.md、sources/sites/ume-exo-project.md — UME 外骨骼力矩反馈遥操作；wiki/entities/paper-ume-exo.md；交叉 teleoperation、bimanual-manipulation、loco-manipulation、motion-retargeting、action-chunking

## [2026-06-18] ingest | sources/blogs/allenai_molmo_motion.md — MolmoMotion 语言条件 3D 点轨迹预测；wiki/entities/molmo-motion.md；交叉 generative-world-models、manipulation、video-as-simulation

## [2026-06-18] ingest | sources/papers/greenvla_arxiv_2602_00919.md — Green-VLA 五阶段 VLA；wiki/entities/paper-greenvla-staged-vla-humanoid.md、wiki/methods/vla.md、wiki/tasks/manipulation.md

## [2026-06-18] ingest | sources/papers/enpire_nvidia_gear_2026.md、sources/sites/nvidia-research-enpire.md — ENPIRE 真机 coding-agent 策略自改进；wiki/methods/enpire.md；交叉 manipulation、simulation-evaluation-infrastructure

## [2026-06-18] ingest | sources/papers/kairos_arxiv_2606_16533.md — Kairos 原生世界模型栈；wiki/entities/paper-kairos-native-world-model-stack.md；交叉 generative-world-models、world-action-models、homeworld

## [2026-06-18] ingest | sources/repos/xpad.md — 接入 Linux Xbox USB 手柄内核驱动 xpad 并新建 wiki/entities/xpad.md，交叉更新 teleoperation 与 open-duck-mini-runtime

- 原始资料：[`sources/repos/xpad.md`](sources/repos/xpad.md)（<https://github.com/paroj/xpad>）
- 沉淀页面：[`wiki/entities/xpad.md`](wiki/entities/xpad.md)
- 交叉更新：[`wiki/tasks/teleoperation.md`](wiki/tasks/teleoperation.md)、[`wiki/entities/open-duck-mini-runtime.md`](wiki/entities/open-duck-mini-runtime.md)
## [2026-06-18] ingest | sources/sites/botworld.md — 接入 BotWorld 机器人资产平台；wiki/entities/botworld.md；交叉 urdf-studio、botlab-motioncanvas、step2urdf、motrix

- 原始资料：[`sources/sites/botworld.md`](sources/sites/botworld.md)（<https://botworld.enkeebot.com/>；前端 bundle 策展）
- 沉淀页面：[`wiki/entities/botworld.md`](wiki/entities/botworld.md)
- 交叉更新：[`wiki/entities/urdf-studio.md`](wiki/entities/urdf-studio.md)、[`wiki/entities/botlab-motioncanvas.md`](wiki/entities/botlab-motioncanvas.md)、[`wiki/entities/step2urdf.md`](wiki/entities/step2urdf.md)、[`wiki/entities/motrix.md`](wiki/entities/motrix.md)

## [2026-06-18] ingest | sources/sites/motrixsim-web-viewer.md — 归档 MotrixSim Web Viewer 并更新 wiki/entities/motrix.md

- 原始资料：[`sources/sites/motrixsim-web-viewer.md`](sources/sites/motrixsim-web-viewer.md)（<https://motrix.motphys.com/>、ReadTheDocs WebViewer 指南）
- 交叉更新：[`sources/repos/motphys-motrix.md`](sources/repos/motphys-motrix.md)
- 沉淀页面：[`wiki/entities/motrix.md`](wiki/entities/motrix.md) — 新增 MotrixSim Web Viewer 小节（WASM、Online/Customize、拖文件夹加载、快捷键）

## [2026-06-18] ingest | sources/repos/step2urdf.md、sources/sites/step2urdf-top.md — STEP→URDF 浏览器工具 step2urdf；wiki/entities/step2urdf.md；交叉 urdf-studio、cad-skills、references/repos/utilities.md

## [2026-06-18] ingest | sources/blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md — 运动小脑 64 篇长文：父节点 wiki/overview/humanoid-motion-cerebellum-technology-map.md + 九组 motion-cerebellum-category-* hub；复用 paper-hrl-stack-* 等既有节点，新建 15 篇 paper-motion-cerebellum-*

- 工具：Agent Reach v1.5.0 + wechat-article-for-ai（Camoufox）；短链 <https://mp.weixin.qq.com/s/Kx9myecE1Z0eGqOapoqQnA>
- 原始资料：[`sources/blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md`](sources/blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md)、[`sources/raw/wechat_motion_cerebellum_64_survey_2026-06-18.md`](sources/raw/wechat_motion_cerebellum_64_survey_2026-06-18.md)、[`sources/papers/motion_cerebellum_64_catalog.md`](sources/papers/motion_cerebellum_64_catalog.md)
- 沉淀页面：[`wiki/overview/humanoid-motion-cerebellum-technology-map.md`](wiki/overview/humanoid-motion-cerebellum-technology-map.md)、九组 [`wiki/overview/motion-cerebellum-category-*.md`](wiki/overview/motion-cerebellum-category-01-locomotion-base.md)
- 新建索引（15）：`wiki/entities/paper-motion-cerebellum-*`（GuideWalk、T-GMP、MARCH、TAGA、TRAM、Stubborn、ConstrainedMimic、SafeWBC、MuGen、CEER、HANDOFF、主动空间大脑、HOIST、HumanoidMimicGen、GRAIL）
- 交叉更新：[`wiki/overview/humanoid-rl-motion-control-body-system-stack.md`](wiki/overview/humanoid-rl-motion-control-body-system-stack.md)、[`sources/README.md`](sources/README.md)、[`sources/repos/panniantong_agent_reach.md`](sources/repos/panniantong_agent_reach.md)

## [2026-06-18] ingest | sources/papers/deepinsight_arxiv_2606_17574.md — DeepInsight Physical AI 全栈统一评测基础设施；wiki/entities/deepinsight.md；交叉 simulation-evaluation-infrastructure、robot-training-stack-layers-technology-map

## [2026-06-18] ingest | sources/papers/rove_arxiv_2606_17011.md — ROVE 人形 VLA 干预后训练；wiki/entities/paper-rove-humanoid-vla-intervention.md；交叉 teleoperation、vla、online-vs-offline-rl

## [2026-06-18] ingest | sources/blogs/wechat_shenlan_homogeneous_coordinates_transform.md — 《具身智能基础》专栏 05 齐次坐标与齐次变换；新建 wiki/formalizations/homogeneous-coordinates-transform.md；更新专栏父节点

- 工具：已安装 [Panniantong/Agent-Reach](https://github.com/Panniantong/Agent-Reach) v1.5.0 + [wechat-article-for-ai](https://github.com/bzd6661/wechat-article-for-ai)（Camoufox）；专辑 <https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzkwMDcyNDUzMQ==&action=getalbum&album_id=4525948187102363653> 共 5 篇，本篇短链 <https://mp.weixin.qq.com/s/3vwaizPOgJKCwQ9e5LuKGA>
- 原始资料：[`sources/blogs/wechat_shenlan_homogeneous_coordinates_transform.md`](sources/blogs/wechat_shenlan_homogeneous_coordinates_transform.md)、[`sources/raw/wechat_shenlan_homogeneous_coords_2026-06-18.md`](sources/raw/wechat_shenlan_homogeneous_coords_2026-06-18.md)
- 沉淀页面：[`wiki/formalizations/homogeneous-coordinates-transform.md`](wiki/formalizations/homogeneous-coordinates-transform.md)
- 交叉更新：[`wiki/overview/shenlan-embodied-ai-fundamentals-series.md`](wiki/overview/shenlan-embodied-ai-fundamentals-series.md)、[`wiki/formalizations/lie-group-rigid-body-motions.md`](wiki/formalizations/lie-group-rigid-body-motions.md)、[`wiki/formalizations/3d-coordinate-transforms-vision-robotics.md`](wiki/formalizations/3d-coordinate-transforms-vision-robotics.md)、[`wiki/formalizations/se3-representation.md`](wiki/formalizations/se3-representation.md)、[`sources/README.md`](sources/README.md)、[`sources/repos/panniantong_agent_reach.md`](sources/repos/panniantong_agent_reach.md)

## [2026-06-18] ingest | sources/repos/wtfos.md、sources/sites/fpv-wtf.md — wtfOS DJI 数字 FPV 固件框架入库；新建 wiki/entities/wtfos.md；交叉 multirotor-simulation-planning-control-stack、betaflight

- 原始资料：[`sources/repos/wtfos.md`](sources/repos/wtfos.md)（<https://github.com/fpv-wtf/wtfos>）、[`sources/sites/fpv-wtf.md`](sources/sites/fpv-wtf.md)（<https://fpv.wtf/>）
- 沉淀页面：[`wiki/entities/wtfos.md`](wiki/entities/wtfos.md)
- 交叉更新：[`wiki/overview/multirotor-simulation-planning-control-stack.md`](wiki/overview/multirotor-simulation-planning-control-stack.md)、[`wiki/entities/betaflight.md`](wiki/entities/betaflight.md)
## [2026-06-17] lint | scripts/lint_wiki.py — V25 P0 数据集页元数据巡检 V1（`dataset_metadata_check`）

- 变更：[`scripts/lint_wiki.py`](scripts/lint_wiki.py) 新增 `_check_dataset_entity_metadata`，针对 frontmatter `tags` 含 `dataset` 的 `wiki/entities/*.md`（兼容列表式与内联式 tags），按关键词命中近似检查正文是否覆盖「规模 / 模态 / 许可证 / 重定向就绪度」四类标准化速查维度，缺失维度作为 INFO 级 result key `dataset_missing_metadata` 写入报告，加入 `INFO_ONLY_KEYS`（不计入失败总数、不阻塞 CI）。
- 测试：新增 [`tests/test_lint_wiki_dataset_metadata.py`](tests/test_lint_wiki_dataset_metadata.py) 4 用例（完整页通过、内联 tags 命中并记缺失维度、非 dataset 页跳过、INFO 不计失败总数）。
- 验证：`make lint` 0 errors（信息型预警 21→22）；全库巡检命中 17 页缺失维度，基线快照写入 [`exports/lint-report.md`](exports/lint-report.md)；`ruff check` 通过；lint_wiki 相关 51 用例全绿。

## [2026-06-17] ingest | sources/repos/betaflight.md、sources/sites/betaflight-com.md — Betaflight FPV 飞控固件入库；新建 wiki/entities/betaflight.md；交叉 multirotor-simulation-planning-control-stack、px4-autopilot、gym-pybullet-drones

- 原始资料：[`sources/repos/betaflight.md`](sources/repos/betaflight.md)（<https://github.com/betaflight/betaflight>）、[`sources/sites/betaflight-com.md`](sources/sites/betaflight-com.md)（<https://betaflight.com/>）
- 沉淀页面：[`wiki/entities/betaflight.md`](wiki/entities/betaflight.md)
- 交叉更新：[`wiki/overview/multirotor-simulation-planning-control-stack.md`](wiki/overview/multirotor-simulation-planning-control-stack.md)、[`wiki/entities/px4-autopilot.md`](wiki/entities/px4-autopilot.md)、[`wiki/entities/gym-pybullet-drones.md`](wiki/entities/gym-pybullet-drones.md)

## [2026-06-17] ingest | sources/repos/plotjuggler.md — PlotJuggler 时序可视化工具入库；新建 wiki/entities/plotjuggler.md；交叉 robot-policy-debug-playbook、ros2-basics、px4-autopilot

## [2026-06-17] ingest | sources/papers/toporetarget_arxiv_2606_16272.md — 接入 TopoRetarget 交互保留灵巧重定向；wiki/methods/toporetarget-interaction-preserving-dexterous-retargeting.md、wiki/concepts/motion-retargeting.md、wiki/tasks/manipulation.md、wiki/entities/wuji-robotics.md

## [2026-06-17] ingest | sources/repos/nvlabs-soma-x.md、sources/sites/soma-x-docs.md、sources/papers/soma_arxiv_2603_16858.md — NVlabs/SOMA-X 统一参数化人体模型入库；新建 wiki/entities/soma-x.md；交叉 soma-retargeter、motion-retargeting、genmo、kimodo

## [2026-06-17] ingest | sources/repos/sbto.md — 深化 Atarilab/sbto 仓库 ingest；新建 wiki/entities/sbto.md

## [2026-06-17] ingest | sources/papers/dynaretarget_arxiv_2602_06827.md — DynaRetarget/SBTO 全文 ingest；wiki/methods/dynaretarget-sbto-motion-retargeting.md、wiki/entities/paper-notebook-dynaretarget-dynamically-feasible-retargeting-us.md、sources/repos/sbto.md、sources/sites/dynaretarget-github-io.md

## [2026-06-17] structural | wiki/overview/topic-*.md + docs/topic-filters.js + docs/graph.html — 14 项图谱专题各增汇总节点与导读 UI

- 新建 14 个专题汇总页：[`wiki/overview/topic-motion-retargeting.md`](wiki/overview/topic-motion-retargeting.md) … [`topic-vision-backbone.md`](wiki/overview/topic-vision-backbone.md)（一句话定义、缩写速查、覆盖范围、专题互链、参考来源）
- [`docs/topic-filters.js`](docs/topic-filters.js)：`TOPIC_HUB_IDS` / `TOPIC_META.wikiPath+description`；各专题 `ids` 显式纳入汇总节点，保证专题视图下始终可见
- [`docs/graph.html`](docs/graph.html)：选中专题时画布左上角显示「专题汇总」导读条 + 汇总节点高亮（`.node-topic-hub`）；链接跳转详情页
- 验证：`make ci-preflight` 通过（1209 nodes / 7615 edges）

## [2026-06-17] ingest | sources/repos/bullet3.md, sources/sites/pybullet-org.md — 官方 Bullet3 仓与 pybullet.org 入库；升格 wiki/entities/pybullet.md

- 原始资料：[`sources/repos/bullet3.md`](sources/repos/bullet3.md)（<https://github.com/bulletphysics/bullet3>）、[`sources/sites/pybullet-org.md`](sources/sites/pybullet-org.md)（<https://pybullet.org/wordpress/>）
- 沉淀页面：[`wiki/entities/pybullet.md`](wiki/entities/pybullet.md)
- 交叉更新：[`sources/README.md`](sources/README.md)
## [2026-06-17] ingest | sources/papers/motiondisco_arxiv_2606_06139.md — MotionDisco LLM 引导运动发现

- 原始资料：[`sources/papers/motiondisco_arxiv_2606_06139.md`](sources/papers/motiondisco_arxiv_2606_06139.md)（<https://arxiv.org/abs/2606.06139>、<https://atarilab.github.io/motiondisco.io/>）
- 沉淀页面：[`wiki/entities/paper-motiondisco-extreme-humanoid-loco-manipulation.md`](wiki/entities/paper-motiondisco-extreme-humanoid-loco-manipulation.md)
- 交叉更新：[`wiki/tasks/loco-manipulation.md`](wiki/tasks/loco-manipulation.md)（路线 §19 + 关联页）

## [2026-06-16] checklist-v25 | V24 收口 & 初始化 V25

- V24 全部条目此前已收口（P0–P3 + DoD 逐条 `[x]`，`make lint` 0 errors，图谱 1193 节点 / 7421 边、`community_quality_warning=false`、最大社区占比 0.177、事实库 186 条）。
- 新建 [`docs/checklists/tech-stack-next-phase-checklist-v25.md`](docs/checklists/tech-stack-next-phase-checklist-v25.md)：专题选定为「人形训练数据管线」，承接 V24 收尾密集 ingest 的 AMASS/LaFAN1/OMOMO/PHUMA/Humanoid Everyday 五套数据集与 motion-retargeting 概念页，规划「原始动作捕捉/视频 → 重定向 → RL/IL 训练输入」端到端知识链（P1 query+concept）、数据层矛盾检测规则扩展（P2 事实库 186→≥196）、数据集页元数据巡检（P0）与图谱第 15 项「训练数据管线」专题视图（P3）。
- 同步将 README badge / 维护看板、`AGENTS.md`、`docs/README.md`、`docs/checklists/README.md` 的「当前清单」指针从 V24 切到 V25；V24 移入 `archive/` 并修正其内部相对链接，进入历史归档区。

## [2026-06-16] ingest | sources/repos/omomo_release.md, sources/repos/phuma.md, sources/sites/humanoideveryday.md — AMASS/LaFAN1/OMOMO/PHUMA/Humanoid Everyday 五集入库

- 原始资料：[`sources/sites/amass-dataset.md`](sources/sites/amass-dataset.md)、[`sources/repos/ubisoft-laforge-animation-dataset.md`](sources/repos/ubisoft-laforge-animation-dataset.md)、[`sources/repos/omomo_release.md`](sources/repos/omomo_release.md)、[`sources/repos/phuma.md`](sources/repos/phuma.md)、[`sources/sites/humanoideveryday.md`](sources/sites/humanoideveryday.md)
- 沉淀页面：[`wiki/entities/omomo-dataset.md`](wiki/entities/omomo-dataset.md)、[`wiki/entities/humanoid-everyday-dataset.md`](wiki/entities/humanoid-everyday-dataset.md)、[`wiki/comparisons/humanoid-reference-motion-datasets.md`](wiki/comparisons/humanoid-reference-motion-datasets.md)；升格 [`wiki/entities/dataset-bfm-phuma.md`](wiki/entities/dataset-bfm-phuma.md)
- 交叉更新：[`wiki/entities/amass.md`](wiki/entities/amass.md)、[`wiki/entities/lafan1-dataset.md`](wiki/entities/lafan1-dataset.md)、[`wiki/concepts/motion-retargeting.md`](wiki/concepts/motion-retargeting.md)、[`wiki/entities/omniretarget-dataset.md`](wiki/entities/omniretarget-dataset.md)、[`wiki/entities/paper-notebook-humanoid-everyday-a-comprehensive-robotic-datase.md`](wiki/entities/paper-notebook-humanoid-everyday-a-comprehensive-robotic-datase.md)

## [2026-06-16] ingest | sources/blogs/qwen_robot_suite.md — 入库 Qwen-Robot Suite 总览与 Nav/Manip/World 子博客；新建 wiki/entities/qwen-robot-{suite,nav,manip,world}.md；交叉 qwen-vla、vla、vln、generative-world-models

- 原始资料：[`sources/blogs/qwen_robot_suite.md`](sources/blogs/qwen_robot_suite.md)（<https://qwen.ai/blog?id=qwen-robotsuite>）、[`qwen_robot_nav.md`](sources/blogs/qwen_robot_nav.md)、[`qwen_robot_manip.md`](sources/blogs/qwen_robot_manip.md)、[`qwen_robot_world.md`](sources/blogs/qwen_robot_world.md)
- 沉淀页面：[`wiki/entities/qwen-robot-suite.md`](wiki/entities/qwen-robot-suite.md)、[`wiki/entities/qwen-robot-nav.md`](wiki/entities/qwen-robot-nav.md)、[`wiki/entities/qwen-robot-manip.md`](wiki/entities/qwen-robot-manip.md)、[`wiki/entities/qwen-robot-world.md`](wiki/entities/qwen-robot-world.md)
- 交叉更新：[`wiki/entities/qwen-vla.md`](wiki/entities/qwen-vla.md)、[`wiki/methods/vla.md`](wiki/methods/vla.md)、[`wiki/tasks/vision-language-navigation.md`](wiki/tasks/vision-language-navigation.md)、[`wiki/methods/generative-world-models.md`](wiki/methods/generative-world-models.md)
## [2026-06-16] structural | wiki/entities/humanoid-robot.md — 补全 HIL（Hardware-in-the-Loop）及正文全部缩写速查

- 页面：[`wiki/entities/humanoid-robot.md`](wiki/entities/humanoid-robot.md) — 流程图「HIL 与台架安全测试」指硬件在环，非 Hybrid Imitation Learning
- 词典：[`schema/abbrev-glossary.json`](schema/abbrev-glossary.json) 新增 HIL / PRS / HAL / FastDDS

## [2026-06-16] ingest | sources/blogs/current_robotics_curr0_loco_dexterous_manipulation.md — Current Robotics Curr-0 人形 loco-dexterous 全栈博客入库

- 原始资料：[`sources/blogs/current_robotics_curr0_loco_dexterous_manipulation.md`](sources/blogs/current_robotics_curr0_loco_dexterous_manipulation.md)（<https://current-robotics.com/blog/curr-0>）
- 沉淀页面：[`wiki/entities/current-robotics-curr0.md`](wiki/entities/current-robotics-curr0.md)
- 交叉更新：[`wiki/tasks/loco-manipulation.md`](wiki/tasks/loco-manipulation.md)（路线 §18 + 关联页）、[`wiki/entities/wuji-robotics.md`](wiki/entities/wuji-robotics.md)、[`sources/README.md`](sources/README.md)

## [2026-06-16] ingest | sources/blogs/wechat_shenlan_rl_embodied_minimal_closed_loop.md — Agent Reach 抓取深蓝《具身智能基础》专栏 04（RL 最小闭环）并并入运动控制路线

- 工具：已安装 [Panniantong/Agent-Reach](https://github.com/Panniantong/Agent-Reach) v1.5.0 + [wechat-article-for-ai](https://github.com/bzd6661/wechat-article-for-ai)（Camoufox）；Jina Reader 对 `mp.weixin.qq.com` 返回 CAPTCHA，未采用
- 原始资料：[`sources/blogs/wechat_shenlan_rl_embodied_minimal_closed_loop.md`](sources/blogs/wechat_shenlan_rl_embodied_minimal_closed_loop.md)（<https://mp.weixin.qq.com/s/hHkQqLfIOTn0CoAZNuLWJA>）；落盘 [`sources/raw/wechat_shenlan_rl_minimal_closed_loop_2026-06-16.md`](sources/raw/wechat_shenlan_rl_minimal_closed_loop_2026-06-16.md)
- 沉淀页面：[`wiki/entities/pybullet.md`](wiki/entities/pybullet.md)、[`wiki/concepts/embodied-rl-minimal-closed-loop.md`](wiki/concepts/embodied-rl-minimal-closed-loop.md)（**公众号文本身不设独立 wiki 节点**）
- 交叉更新：[`roadmap/motion-control.md`](roadmap/motion-control.md)（L5.0 桥梁 + L5.1 推荐路径 + L7.5 趋势表）、[`wiki/methods/reinforcement-learning.md`](wiki/methods/reinforcement-learning.md)、[`wiki/formalizations/mdp.md`](wiki/formalizations/mdp.md)、[`wiki/formalizations/pomdp.md`](wiki/formalizations/pomdp.md)、[`wiki/overview/shenlan-embodied-ai-fundamentals-series.md`](wiki/overview/shenlan-embodied-ai-fundamentals-series.md)、[`sources/README.md`](sources/README.md)

## [2026-06-15] structural | docs/detail.html + docs/main.js + docs/topic-filters.js — V24 P3 详情页"所属专题"轻量徽标，专题命中规则抽共享模块并收口 V24

- 新增 `docs/topic-filters.js` 作为专题命中规则单一事实源（`TOPIC_FILTERS` / `TOPIC_META` / `matches` / `topicsForNode`）；`docs/graph.html` 移除内联 `TOPIC_FILTERS` 与 `nodeMatchesTopic` 实现改为消费共享模块，并新增 `?topic=<key>` URL 参数自动激活对应专题视图
- `docs/main.js` 新增 `renderDetailTopicBadges`：复用 `link-graph.json` 现成社区数据计算当前页命中的专题，渲染"所属专题"徽标 → `graph.html?topic=<key>`（无命中静默隐藏）；`docs/detail.html` 增 `#detailTopicBadges` 容器，`docs/style.css` 增 `.detail-topic-badge` 胶囊样式
- `docs/checklists/tech-stack-next-phase-checklist-v24.md`：P3 可选项打勾，验收标准（make lint / 节点边数 / community 均衡 / log 记录）逐条复核打勾，V24 全部条目收口
- 验证：`make lint` 退出码 0（「✅ 所有检查通过！」）；节点 1183 / 边 7292、`community_quality_warning=false`、最大社区占比 0.179；Puppeteer 端到端截图归档 `.cursor-artifacts/screenshots/detail-topic-badge.png`、`graph-topic-from-url.png`（详情页徽标→图谱专题视图链路打通）

## [2026-06-15] ingest | sources/repos/gen2humanoid.md — 入库 Gen2Humanoid 文本→HY-Motion→GMR 人形管线；新建 wiki/entities/gen2humanoid.md；交叉 hy-motion-1、motion-retargeting-gmr、motion-retargeting-pipeline

## [2026-06-15] ingest | MoveIt/MoveIt 2 一手资料 — sources/sites/moveit-*.md + sources/repos/moveit-*.md、ros-planning-srdfdom.md；新建 wiki/entities/moveit2.md；交叉 manipulation、curobo、ros2-official-documentation

- 原始资料：[`sources/sites/moveit-official-portal.md`](sources/sites/moveit-official-portal.md)、[`sources/sites/moveit2-picknik-documentation.md`](sources/sites/moveit2-picknik-documentation.md)、[`sources/sites/moveit1-noetic-tutorials.md`](sources/sites/moveit1-noetic-tutorials.md)、[`sources/repos/moveit-moveit2.md`](sources/repos/moveit-moveit2.md)、[`sources/repos/moveit-moveit1.md`](sources/repos/moveit-moveit1.md)、[`sources/repos/ros-planning-srdfdom.md`](sources/repos/ros-planning-srdfdom.md)
- 沉淀页面：[`wiki/entities/moveit2.md`](wiki/entities/moveit2.md)；交叉更新 [`wiki/tasks/manipulation.md`](wiki/tasks/manipulation.md)、[`wiki/entities/curobo.md`](wiki/entities/curobo.md)、[`sources/sites/ros2-official-documentation.md`](sources/sites/ros2-official-documentation.md)

## [2026-06-15] ingest | sources/repos/earthtojake-text-to-cad.md — 入库 CAD Skills（earthtojake/text-to-cad）Agent Skills 库；新建 wiki/entities/cad-skills.md；交叉 text-to-cad、urdf-studio、articraft、mattpocock-skills

- 原始资料：[`sources/repos/earthtojake-text-to-cad.md`](sources/repos/earthtojake-text-to-cad.md)
- 沉淀页面：[`wiki/entities/cad-skills.md`](wiki/entities/cad-skills.md)；交叉更新 [`wiki/concepts/text-to-cad.md`](wiki/concepts/text-to-cad.md)、[`sources/sites/text-to-cad-tools.md`](sources/sites/text-to-cad-tools.md)、[`wiki/entities/urdf-studio.md`](wiki/entities/urdf-studio.md)、[`wiki/entities/articraft.md`](wiki/entities/articraft.md)、[`wiki/entities/mattpocock-skills.md`](wiki/entities/mattpocock-skills.md)

## [2026-06-15] ingest | sources/papers/rumelhart_backprop_learning_representations_nature_1986.md — Rumelhart et al. 1986 Nature 反向传播一手归档；新建 wiki/concepts/backpropagation.md；交叉 deep-learning-foundations / transformer / udl_book

## [2026-06-14] structural | docs/graph.html — V24 P3 图谱新增「视觉感知骨干」专题视图（vision-backbone）

- `TOPIC_FILTERS` 新增 `vision-backbone` 项：path 片段 `backbone/backbones/cnn/vit/resnet/yolo/detection` 并集命中；因核心页同处 community-3（与动作重定向共享）不宜按社区命中，`nodeMatchesTopic` 扩展支持 `ids` 显式纳入 `visual-representation-for-policy` / `generative-vision-pretraining` 两页
- `#filter-topic-chips` 新增 `data-topic="vision-backbone"`（👁️ 视觉骨干）chip，专题视图精准命中 9 个相关节点（cnn-vs-vit / vision-backbones / visual-representation-for-policy / perception-backbone-selection / object-detection / object-detection-model-selection / generative-vision-pretraining + ResNet/YOLO 实体）
- `docs/checklists/tech-stack-next-phase-checklist-v24.md` P3 首项打勾
- 验证：`make lint` 退出码 0（仅 1 条信息型预警）；Puppeteer 截图归档 `.cursor-artifacts/screenshots/graph-topic-vision-backbone.png`

## [2026-06-14] ingest | sources/personal/amp_mjlab_policy_training_essence.md + perceptive_locomotion_representation_essence.md — 两条 ChatGPT 对话核心知识点：wiki/concepts/neural-feedback-controller.md、terrain-latent-representation.md；增补 privileged-training、amp-mjlab

## [2026-06-14] ingest | sources/papers/agi_to_asi_arxiv_2606_12683.md — DeepMind From AGI to ASI 技术报告；wiki/entities/paper-from-agi-to-asi.md；交叉 embodied-scaling-laws / data-flywheel / robot-learning-three-eras-narrative

## [2026-06-14] ingest | sources/papers/oasis_humanoid_loco_manip_2606_08548.md — OASIS arXiv:2606.08548 全文精读入库，升格 wiki/entities/paper-loco-manip-04-oasis.md

## [2026-06-14] ingest | sources/papers/mighty_arxiv_2511_10822.md + sources/repos/mighty.md — MIT ACL MIGHTY Hermite 样条 UAV 轨迹规划（RA-L 2026）；wiki/entities/paper-mighty-hermite-spline-trajectory-planning.md；交叉 multirotor-simulation-planning-control-stack、ego-planner-swarm

## [2026-06-14] ingest | sources/blogs/wechat_embodied_ai_lab_loco_manip_8_papers_survey.md — 具身智能研究室 Loco-Manip 8 篇数据入口周报；父节点 loco-manip-8-papers-technology-map + 四组 loco-manip-category-* 子节点 + 8 篇论文实体

- 工具：已安装 [Panniantong/Agent-Reach](https://github.com/Panniantong/Agent-Reach) v1.4.0（`pip install -e` + `agent-reach install --channels=wechat`）；微信正文经 `~/.agent-reach/tools/wechat-article-for-ai`（Camoufox）
- 原始资料：`sources/blogs/wechat_embodied_ai_lab_loco_manip_8_papers_survey.md`、`sources/raw/wechat_loco_manip_8_papers_2026-06-14.md`、`sources/papers/loco_manip_8_papers_catalog.md`、`sources/papers/loco_manip_survey_*.md`（8 篇）
- 沉淀页面：[`wiki/overview/loco-manip-8-papers-technology-map.md`](wiki/overview/loco-manip-8-papers-technology-map.md)（父）、[`loco-manip-category-01-egocentric-data.md`](wiki/overview/loco-manip-category-01-egocentric-data.md) … [`loco-manip-category-04-contact-teleop.md`](wiki/overview/loco-manip-category-04-contact-teleop.md)（子）、`wiki/entities/paper-loco-manip-01-ego-pi.md` … `paper-loco-manip-08-x-op.md`
- 去重：GenHOI（arXiv:2606.12995）≠ 既有 SimGenHOI 节点
- 交叉更新：[`wiki/tasks/loco-manipulation.md`](wiki/tasks/loco-manipulation.md)、[`sources/README.md`](sources/README.md)、[`sources/repos/panniantong_agent_reach.md`](sources/repos/panniantong_agent_reach.md)
- 验证：`make ci-preflight`

## [2026-06-14] structural | wiki/methods/ppo.md + wiki/concepts/transformer.md + sources/papers/attention_is_all_you_need.md — 新建 PPO/Transformer 概念方法页，消除 lint 高频术语缺页误报

- 新建：[`wiki/methods/ppo.md`](wiki/methods/ppo.md)（clip 代理目标、GAE、机器人 RL 落地要点）、[`wiki/concepts/transformer.md`](wiki/concepts/transformer.md)（自注意力 / MHA / 机器人 VLA·ACT 角色）
- 入库：[`sources/papers/attention_is_all_you_need.md`](sources/papers/attention_is_all_you_need.md)（Vaswani et al. 2017 一手摘要）
- 交叉更新：[`wiki/methods/policy-optimization.md`](wiki/methods/policy-optimization.md)、[`wiki/concepts/deep-learning-foundations.md`](wiki/concepts/deep-learning-foundations.md)、[`wiki/concepts/humanoid-policy-network-architecture.md`](wiki/concepts/humanoid-policy-network-architecture.md)、[`wiki/comparisons/ppo-vs-sac.md`](wiki/comparisons/ppo-vs-sac.md)
- 验证：`make ci-preflight`

## [2026-06-13] ingest | sources/papers/ruka_v2_arxiv_2603_26660.md + sources/repos/ruka-v2.md + sources/sites/ruka-hand-v2-github-io.md — NYU 全开源腱驱动灵巧手 RUKA-v2；升格 wiki/entities/ruka-v2-hand.md，交叉 orca-hand / dexterous-data-collection-guide / manipulation

- 新建实体：[`wiki/entities/ruka-v2-hand.md`](wiki/entities/ruka-v2-hand.md)（16 指 DoF + 2-DoF 腕、AnyTeleop + OpenTeach + BAKU 验证链）
- 交叉更新：[`wiki/entities/orca-hand.md`](wiki/entities/orca-hand.md)、[`wiki/queries/dexterous-data-collection-guide.md`](wiki/queries/dexterous-data-collection-guide.md)、[`wiki/tasks/manipulation.md`](wiki/tasks/manipulation.md)、[`wiki/entities/paper-notebook-ruka-rethinking-the-design-of-humanoid-hands-wit.md`](wiki/entities/paper-notebook-ruka-rethinking-the-design-of-humanoid-hands-wit.md)
- 后续修正：[`wiki/entities/ruka-v2-hand.md`](wiki/entities/ruka-v2-hand.md) 成本字段 `$1,500` 改 `$1{,}500` 避免 KaTeX 误解析
- 验证：`make ci-preflight`

## [2026-06-13] structural | schema/canonical-facts.json — V24 P2 事实库由 172 → 186 条，补齐视觉骨干/机器人表征矛盾检测规则

- 新增 14 条矛盾检测规则：ResNet 残差缓解退化、深层网络退化非过拟合、ViT 数据量门槛、ViT 归纳偏置弱、CNN 归纳偏置强、YOLO 单阶段实时、两阶段精度高延迟大、YOLO 误差结构、注意力二次复杂度、冻结预训练表征样本效率高、端到端视觉策略样本效率低、R3M 人类视频预训练表征、VC-1 具身视觉骨干、视觉域差距优先于换骨干
- 逐条经脚本校验：每条 `terms`+`pos_claims` 对现存 wiki 页（cnn-vs-vit-backbones / vision-backbones / object-detection / visual-representation-for-policy / perception-backbone-selection 等）均有命中，`neg_claims` 仅刻画错误论断、不误伤正文
- `docs/checklists/tech-stack-next-phase-checklist-v24.md` P2 与 DoD 事实库条目打勾
- 验证：`python3 scripts/lint_wiki.py` 退出码 0，矛盾检测 0 项；JSON 合法

## [2026-06-13] structural | tech-map/modules/system/ros2.md + sources/sites/ros2-official-documentation.md — 填充 tech-node-system-ros2 空详情页；归档 ROS 2 Humble 一手文档，交叉 ros2-basics / ros2-vs-lcm / sim2real 部署链

## [2026-06-12] structural | wiki/comparisons/hil-vs-mtrg-vs-zest-parkour-imitation.md、wiki/queries/table-tennis-hierarchical-skill-learning-guide.md + 5 页陈旧措辞 + 2 页 paper 实体 — 消除 10 条 lint 信息型预警

- 为高频引用 methods 补落地：[`hil-vs-mtrg-vs-zest-parkour-imitation.md`](wiki/comparisons/hil-vs-mtrg-vs-zest-parkour-imitation.md)（覆盖 HIL/MTRG）、[`table-tennis-hierarchical-skill-learning-guide.md`](wiki/queries/table-tennis-hierarchical-skill-learning-guide.md)
- `paper-humanoid-soccer-swarm-intelligence` 补 `venue` 键；`paper-notebook-a-hierarchical-model-based-system-for-high-perfo` 补「方法栈」段
- 软化 5 页绝对化 SOTA 措辞：`generative-vision-pretraining`、`paper-resnet`、`paper-wem`、`paper-worldvln`、`paper-yolo`
- 验证：`make ci-preflight`


- 全库自动统计正文以 `**加粗**`/`` `反引号` `` 高频出现（≥6 个不同页面引用）但缺独立 `concepts/methods/formalizations` 页的术语，输出"建议新建页"候选（INFO 级，不阻塞 CI），作为后续 ingest/query 选题入口
- 与既有 `_check_missing_concepts`（人工 watch 列表映射已知 slug）互补；单 token 词形过滤路径/文件名、大小写归并、候选上限 15、停用词剔除 frontmatter 键
- 实测候选 8 条（PPO/MuJoCo/Transformer 等），新增 INFO 区块至健康报告；`docs/checklists/tech-stack-next-phase-checklist-v24.md` P0 该项打勾
- 验证：`tests/test_lint_wiki_missing_concept_pages.py` 6 例通过；`ruff format/check`、`mypy scripts/lint_wiki.py` 全绿；`python3 scripts/lint_wiki.py` 退出码 0

## [2026-06-12] ingest | sources/repos/manim-community.md + manim-3b1b.md + sites/manim-community.md — Manim/ManimCE/ManimGL 程序化数学动画；wiki/entities/manim.md，交叉 character-animation-vs-robotics、blender

## [2026-06-12] fix(wiki): 合并 MuJoCo Playground 等重复节点并修复 paper-notebook 标题 `[ ]` 残留

- 合并 4 对计划子节点 → 已有实体：`paper-notebook-mujoco-playground-*` → [`wiki/entities/mujoco-playground.md`](wiki/entities/mujoco-playground.md)；另 Genesis / ORB-SLAM3 / VINS-Fusion 同理
- 根因修复：`short_label` / `clean_display_title` 剥离 markdown 链接与残留方括号；`collect_wiki_index` 索引全部 `wiki/entities/*`；`dedupe_paper_notebook_nodes.py` 增加按标题合并 + 全量标题修复
- 相关：`wiki/overview/paper-notebook-category-09-state-estimation.md`、`wiki/overview/paper-notebook-category-11-simulation-benchmark.md`、`schema/paper-notebook-wiki-full-map.yml`
- 验证：`tests/test_paper_notebook_title_cleanup.py`、`make ci-preflight`

## [2026-06-12] ingest | sources/papers/visualmimic_arxiv_2509_20322.md — VisualMimic 视觉分层 sim2real loco-manipulation；升格 wiki/entities/paper-notebook-visualmimic.md，更新 wiki/tasks/loco-manipulation.md

## [2026-06-12] fix(wiki): Paper Notebooks 分类索引去重 — 已完成深读笔记不再与 PROGRESS.md 待深读别名并列；`merge_paper_catalog` + `dedupe_category_entries`；重生成 `wiki/overview/paper-notebook-category-*.md`

- 相关：`wiki/overview/humanoid-paper-notebooks-index.md`、`scripts/bootstrap_paper_notebook_knowledge.py`
- 清理 4 个已无入链的 `paper-notebook-*` 计划占位实体（PPO/PULSE/Expressive WBC/Generating Diverse → 已有深读或概念页承接）
- 验证：`tests/test_bootstrap_paper_notebook_dedupe.py`、`make ci-preflight`

## [2026-06-12] structural | 站点大 JSON（search-index/index-v1/site-data-v1/link-graph×2 份）与 sitemap 移出 git，改为 pages.yml 部署时生成；export.yml 仅提交小型派生文件；tests.yml pytest 前生成快照；修复 archive v3–v9 共 7 处 Karpathy Wiki 断链

## [2026-06-12] ingest | sources/papers/humanoid_soccer_swarm_intelligence_sensors_2025.md + robocup_spl + artemis — 人形机器人群控一手资料；wiki/concepts/humanoid-multi-robot-coordination.md wiki/entities/paper-humanoid-soccer-swarm-intelligence.md wiki/tasks/humanoid-soccer.md

## [2026-06-12] structural | docs/checklists 清理首批 — v1–v23 归档至 archive/、移除 PR 验证截图产物、.obsidian/workspace.json 停止跟踪、删除 .codex、AGENTS.md 清单指针 v23→v24；详见 docs/change-log.md

## [2026-06-12] ingest | sources/papers/smplolympics_arxiv_2407_00187.md + table_tennis_strategy_skill_arxiv_2407_16210.md — SMPLOlympics 体育 benchmark 与 PhysicsPingPong 乒乓球分层控制；wiki/entities/smplolympics.md wiki/methods/table-tennis-strategy-skill-learning.md

## [2026-06-12] ingest | sources/papers/hil_hybrid_imitation_learning_arxiv_2505_12619.md + mtrg_reference_goal_driven_rl_arxiv_2602_20375.md + zest.md — HIL/MTRG 新入库，ZEST 交叉引用；wiki/methods/hil-hybrid-imitation-learning.md wiki/methods/mtrg-reference-goal-driven-rl.md

## [2026-06-12] fix(graph): 合并 Paper Notebooks 分类页与对应 task/concept 页的重复图谱社区

- 实现：`COMMUNITY_HUB_ALIASES` + `_merge_partition_by_hub_equivalence`（如 `论文深读·灵巧操作` 与 `操作（Manipulation）` 合并为同一社区）
- 测试：`test_merge_partition_by_hub_equivalence_merges_alias_hubs`、`test_exported_communities_have_no_duplicate_canonical_hubs`
- 验证：`make ci-preflight` 通过

## [2026-06-12] ingest | sources/papers/bifrost_umi_arxiv_2605_03452.md — 增补 arXiv Related Works 与 TWIST2 采集范式对照；交叉 wiki/entities/paper-bifrost-umi.md

## [2026-06-12] ingest | sources/papers/clot_arxiv_2602_15060.md + sources/sites/clot-project.md + sources/repos/clot.md — CLOT arXiv/项目页/代码 ingest；深化 wiki/entities/paper-amp-survey-16-clot.md（闭环全局、Observation Pre-shift）

## [2026-06-12] ingest | sources/sites/twist2-project.md + sources/repos/twist2.md — TWIST2 项目页/仓库一手 ingest；深化 wiki/entities/paper-twist2.md（Mermaid 管线、ICRA 2026、分层 visuomotor）

## [2026-06-11] structural | scripts/lint_wiki.py — V24 P0「陈旧声明（stale claim）巡检 V1」：新增 `_check_stale_claims` 信息型检查 + 6 例单测 + lint 报告基线快照

- 实现：正文（去 frontmatter / 代码块 / 误区区块）命中「SOTA / state-of-the-art / 当前最强 / 最新」等绝对化措辞，且本页 frontmatter `updated` 早于库内共享 ≥1 个 tag 的更晚页面时，输出 💡 INFO 级提示；列入 `INFO_ONLY_KEYS`，不计入 lint 失败总数、不阻塞 CI
- 新增：`STALE_CLAIM_PATTERNS`、`_frontmatter_block`、`_frontmatter_tags` 辅助函数；`format_report` 新增「陈旧声明」小节
- 基线快照：`exports/lint-report.md` 当前 5 条（generative-vision-pretraining / paper-resnet / paper-wem / paper-worldvln / paper-yolo）
- 测试：`tests/test_lint_wiki_stale_claims.py` 6 例（命中/最新页不报/无共享 tag/无绝对化措辞/代码块忽略/info-only），`pytest -k lint` 41 passed；`ruff check`、`ruff format` 通过
- 清单：勾选 [`tech-stack-next-phase-checklist-v24.md`](docs/checklists/tech-stack-next-phase-checklist-v24.md) P0 首项

## [2026-06-11] structural | scripts/dedupe_paper_notebook_nodes.py — 全量去重合并 4 对 `paper-notebook-*` 计划子节点与已有深读实体（按 frontmatter arXiv）；`make paper-notebook-dedupe` 复跑零残留

- 合并：`paper-notebook-behavior-foundation-model-for-humanoid-robots` → [`paper-behavior-foundation-model-humanoid.md`](wiki/entities/paper-behavior-foundation-model-humanoid.md)（2509.13780）
- 合并：`paper-notebook-reinforcement-learning-for-versatile-dynamic-and` → [`paper-cassie-biped-versatile-locomotion-rl.md`](wiki/entities/paper-cassie-biped-versatile-locomotion-rl.md)（2401.16889）
- 合并：`paper-notebook-real-world-humanoid-locomotion-with-rl` → [`paper-digit-humanoid-locomotion-rl.md`](wiki/entities/paper-digit-humanoid-locomotion-rl.md)（2303.03381）
- 合并：`paper-notebook-pilot` → [`paper-pilot-perceptive-loco-manipulation.md`](wiki/entities/paper-pilot-perceptive-loco-manipulation.md)（2601.17440）
- 工具：新增 [`scripts/dedupe_paper_notebook_nodes.py`](scripts/dedupe_paper_notebook_nodes.py)、`make paper-notebook-dedupe`；补强 [`scripts/sync_paper_notebook_links.py`](scripts/sync_paper_notebook_links.py) 多候选 arXiv 评分与 bootstrap 防重建
- 相关：`schema/paper-notebook-wiki-full-map.yml`、分类父节点表项、`make ci-preflight` 通过

## [2026-06-11] structural | scripts/bootstrap_paper_notebook_knowledge.py — 同步 papers/PROGRESS.md 全量 563 条：合并 progress.json 后 665 篇入图谱，新建约 380 个 `wiki/entities/paper-notebook-*` 计划子节点；14 类分类父节点扩表

- 数据源：[papers/PROGRESS.md](https://github.com/ImChong/Humanoid_Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md) + [progress.json](https://github.com/ImChong/Humanoid_Robot_Learning_Paper_Notebooks/blob/main/progress.json)
- 工具：`make paper-notebook-bootstrap`；`schema/paper-notebook-wiki-full-map.yml` 扩至 665 篇
- 相关：`wiki/overview/paper-notebook-category-*.md`、`wiki/overview/humanoid-paper-notebooks-index.md`

## [2026-06-11] structural | scripts/bootstrap_paper_notebook_knowledge.py — 同步 Paper Notebooks progress.json 待深读 115 篇：新建 87 个 `wiki/entities/paper-notebook-*` 计划子节点 + sources；更新分类父节点与 full-map（252 篇）

- 数据源：[Humanoid_Robot_Learning_Paper_Notebooks/progress.json](https://github.com/ImChong/Humanoid_Robot_Learning_Paper_Notebooks/blob/main/progress.json) 中 `status=pending` 且尚无完整深读笔记的条目
- 新建：`sources/papers/humanoid_pnb_*.md`（87）、`wiki/entities/paper-notebook-*.md`（87，`status: planned`）
- 交叉更新：`wiki/overview/paper-notebook-category-04-loco-manipulation-and-wbc.md`（33→147 篇）、`wiki/overview/paper-notebook-category-02-motion-retargeting.md`、`wiki/overview/humanoid-paper-notebooks-index.md`、`schema/paper-notebook-wiki-full-map.yml`
- 工具：`make paper-notebook-bootstrap`；`make ci-preflight` 通过

## [2026-06-11] ingest | sources/papers/now_you_see_that_arxiv_2602_06382.md — Now You See That（RSS 2026）8 步深度增广 + 多 critic/discriminator 特权 RL + vision-aware DAgger；wiki/entities/paper-now-you-see-that-humanoid-vision-locomotion.md；交叉 humanoid-locomotion / stair-obstacle-perceptive-locomotion

## [2026-06-11] structural | scripts: preflight 提速（Louvain 图谱、lint 去重、stale 规则、bump-wiki-from-sources）

## [2026-06-11] ingest | sources/papers/rma_arxiv_2107_04034.md — RMA（RSS 2021）论文/项目页/rl_locomotion 代码入库；沉淀 wiki/entities/paper-rma-rapid-motor-adaptation.md；交叉更新 privileged-training、sim2real、locomotion

## [2026-06-11] structural | wiki/entities/paper-perceptive-bfm.md、docs/main.js — 正文 PMT 公式改 `$...$` 启用 KaTeX 蓝框；修复 Mermaid `<br/>` 渲染语法错误

## [2026-06-11] ingest | sources/papers/perceptive_bfm_corl_2026.md — Perceptive BFM（CoRL 2026）TCRS+PMT 地形感知 BFM；wiki/entities/paper-perceptive-bfm.md 及 behavior-foundation-model / privileged-training / footstep-planning / sonic / stair-obstacle 交叉更新

## [2026-06-11] structural | wiki/queries/humanoid-soccer-skill-learning-method-selection.md、wiki/entities/paper-omg-omni-modal-humanoid-control.md、wiki/methods/paid-framework.md、wiki/tasks/humanoid-soccer.md — lint 清零 + 人形足球技能学习选型 Query

- 新建 Query：[`wiki/queries/humanoid-soccer-skill-learning-method-selection.md`](wiki/queries/humanoid-soccer-skill-learning-method-selection.md)（PAiD vs RoboNaldo 选型指南；落地高频引用的 paid-framework 交叉链）
- 补强实体：[`wiki/entities/paper-omg-omni-modal-humanoid-control.md`](wiki/entities/paper-omg-omni-modal-humanoid-control.md)（frontmatter 补 code 来源键、正文补「评测与开放进度」段）
- 交叉更新：[`wiki/methods/paid-framework.md`](wiki/methods/paid-framework.md)、[`wiki/tasks/humanoid-soccer.md`](wiki/tasks/humanoid-soccer.md)
- 51 页 stale 复核：2026-06-10 去重合并对 source catalog 的链接改写 bump `updated`→2026-06-11（经 review 内容仍准确，非内容重写）
- 门禁：`make ci-preflight` 通过、lint-report 归零

## [2026-06-11] tooling | scripts/scaffold_wiki_page.py、tests/test_scaffold_wiki_page.py — V24 P0「query → wiki 回填脚手架」：新增页面骨架生成脚本与测试

- 新增 [`scripts/scaffold_wiki_page.py`](scripts/scaffold_wiki_page.py)：`type + 标题`（可选 `--slug`）按全库 frontmatter 规范生成骨架——含 `## 英文缩写速查` 落在规范位置（定义之后、为什么重要之前）、`related`/`sources` 占位、三段式正文；query 类型额外含 `**Query 产物**` / `## 参考来源` / `## 关联页面`
- 复用 `lint_wiki.has_section` / `wiki_abbrev_section.is_abbrev_glossary_well_placed` 做生成后结构自检；`--dry-run` 只打印不落盘、`--force` 控制覆盖
- 新增 [`tests/test_scaffold_wiki_page.py`](tests/test_scaffold_wiki_page.py) 7 例：结构自检、缩写区块位序、query 标记、frontmatter 键、slug 推断、dry-run 不落盘、写入与防覆盖
- 验证：新增测试全绿（全量 149 passed）、ruff/format/mypy 通过、`lint_wiki` 基线 51 项不变
- 勾选 checklist v24 P0「query → wiki 回填脚手架」

## [2026-06-10] structural | wiki/concepts/vision-backbones.md、wiki/methods/object-detection.md — V24 P1「视觉感知专题交叉补强」：明示「骨干特征 → 检测/分割头 → 策略输入」衔接链并与新页双向回链

- 在 [`vision-backbones.md`](wiki/concepts/vision-backbones.md) 新增「骨干特征 → 检测/分割头 → 策略输入」小节，补回链 [`cnn-vs-vit-backbones.md`](wiki/comparisons/cnn-vs-vit-backbones.md)、[`perception-backbone-selection.md`](wiki/queries/perception-backbone-selection.md)（frontmatter `related` + 关联页面）
- 在 [`object-detection.md`](wiki/methods/object-detection.md) 新增「在感知链中的位置」小节，把检测头明示为衔接链中间环节；补回链 [`cnn-vs-vit-backbones.md`](wiki/comparisons/cnn-vs-vit-backbones.md)、[`visual-representation-for-policy.md`](wiki/concepts/visual-representation-for-policy.md)、[`perception-backbone-selection.md`](wiki/queries/perception-backbone-selection.md)，消除单向链接孤儿
- 与 P1 三新页（对比 / 概念 / Query）形成双向回链；`make ci-preflight` 通过、派生索引与站点导出同步重生
- 清单：[`tech-stack-next-phase-checklist-v24.md`](docs/checklists/tech-stack-next-phase-checklist-v24.md) P1「视觉感知专题交叉补强」勾选完成

## [2026-06-10] structural | wiki/entities/paper-bfm-zero.md、paper-opentrack.md、paper-ams.md、paper-hiking-in-the-wild.md、paper-adaptive-humanoid-control.md、paper-deep-whole-body-parkour.md — 全量去重合并 6 对重复实体页（HRL 栈 / BFM / AMP 双索引）

- 合并为单一实体：[`paper-bfm-zero.md`](wiki/entities/paper-bfm-zero.md)、[`paper-opentrack.md`](wiki/entities/paper-opentrack.md)、[`paper-ams.md`](wiki/entities/paper-ams.md)、[`paper-hiking-in-the-wild.md`](wiki/entities/paper-hiking-in-the-wild.md)、[`paper-adaptive-humanoid-control.md`](wiki/entities/paper-adaptive-humanoid-control.md)、[`paper-deep-whole-body-parkour.md`](wiki/entities/paper-deep-whole-body-parkour.md)
- 删除 12 个重复页（`paper-hrl-stack-*` / `paper-bfm-*` / `paper-amp-survey-*` 各 2 对）
- 交叉更新：[`humanoid-rl-motion-control-body-system-stack.md`](wiki/overview/humanoid-rl-motion-control-body-system-stack.md)、[`humanoid-amp-motion-prior-survey.md`](wiki/overview/humanoid-amp-motion-prior-survey.md)、BFM 技术地图/分类页、[`stair-obstacle-perceptive-locomotion.md`](wiki/tasks/stair-obstacle-perceptive-locomotion.md)、catalog 源表

## [2026-06-10] ingest | sources/papers/robonaldo_arxiv_2606_11092.md — RoboNaldo 三阶段射门课程 RL；wiki/entities/paper-robonaldo-humanoid-soccer-shooting.md、wiki/tasks/humanoid-soccer.md、wiki/methods/paid-framework.md

## [2026-06-10] structural | wiki/entities/paper-beyondmimic.md、paper-sentinel.md、paper-sonic.md、paper-twist.md — 合并 HRL 栈与 BFM 双索引重复实体页（4 对）

- 合并为单一实体：[`paper-beyondmimic.md`](wiki/entities/paper-beyondmimic.md)、[`paper-sentinel.md`](wiki/entities/paper-sentinel.md)、[`paper-sonic.md`](wiki/entities/paper-sonic.md)、[`paper-twist.md`](wiki/entities/paper-twist.md)
- 删除 8 个重复页（`paper-hrl-stack-*` / `paper-bfm-*` 各 4 对）
- 交叉更新：[`humanoid-rl-motion-control-body-system-stack.md`](wiki/overview/humanoid-rl-motion-control-body-system-stack.md)、BFM 技术地图/分类页、[`paper-resmimic.md`](wiki/entities/paper-resmimic.md)、[`schema/paper-notebook-wiki-map.yml`](schema/paper-notebook-wiki-map.yml)

## [2026-06-10] structural | wiki/entities/paper-twist2.md — 合并 TWIST2 重复实体页（paper-hrl-stack-10-twist2 与 paper-bfm-10-twist2 为同一篇 arXiv:2505.02833）

- 合并为单一实体：[`wiki/entities/paper-twist2.md`](wiki/entities/paper-twist2.md)（保留 42 篇 RL 栈与 BFM 41 篇双索引语境）
- 删除重复页：[`paper-hrl-stack-10-twist2.md`](wiki/entities/paper-hrl-stack-10-twist2.md)、[`paper-bfm-10-twist2.md`](wiki/entities/paper-bfm-10-twist2.md)
- 交叉更新：[`humanoid-rl-motion-control-body-system-stack.md`](wiki/overview/humanoid-rl-motion-control-body-system-stack.md)、[`bfm-41-papers-technology-map.md`](wiki/overview/bfm-41-papers-technology-map.md)、[`bfm-category-02-goal-conditioned-learning.md`](wiki/overview/bfm-category-02-goal-conditioned-learning.md)、[`limmt-gqs-motion-curation.md`](wiki/methods/limmt-gqs-motion-curation.md)

## [2026-06-10] ingest | sources/sites/ttl_uart_logic_level_primary_refs.md、rs232_tia_eia_primary_refs.md、rs485_tia_eia_primary_refs.md — TTL/RS-232/RS-485 一手资料入库；wiki/concepts/ttl-serial-logic-level.md、rs-232-serial-interface.md、rs-485-serial-bus.md 及 uart-serial-communication 交叉链接

## [2026-06-10] ingest | sources/papers/mpc_rl_arxiv_2606_05687.md — MPC-RL 与 π MPC 入库；wiki/entities/paper-mpc-rl-humanoid-locomotion-manipulation.md、wiki/methods/pi-mpc.md 及 mpc-vs-rl / loco-manipulation 交叉更新

- 原始资料：[`mpc_rl_arxiv_2606_05687.md`](sources/papers/mpc_rl_arxiv_2606_05687.md)（<https://arxiv.org/abs/2606.05687>）；[`pi_mpc_arxiv_2601_14414.md`](sources/papers/pi_mpc_arxiv_2601_14414.md)（<https://arxiv.org/abs/2601.14414>）；[`junhengl_mpc_rl.md`](sources/repos/junhengl_mpc_rl.md)（<https://github.com/junhengl/mpc-rl>）
- 新建实体：[`wiki/entities/paper-mpc-rl-humanoid-locomotion-manipulation.md`](wiki/entities/paper-mpc-rl-humanoid-locomotion-manipulation.md)（CD-MPC landmark reward、πⁿ MPC 批训练、部署纯 RL、Mermaid 管线）
- 新建方法页：[`wiki/methods/pi-mpc.md`](wiki/methods/pi-mpc.md)（parallel-in-horizon ADMM、velocity-form、construction-free）
- 交叉更新：[`wiki/comparisons/mpc-vs-rl.md`](wiki/comparisons/mpc-vs-rl.md)、[`wiki/tasks/loco-manipulation.md`](wiki/tasks/loco-manipulation.md)、[`wiki/concepts/centroidal-dynamics.md`](wiki/concepts/centroidal-dynamics.md)、[`wiki/methods/model-predictive-control.md`](wiki/methods/model-predictive-control.md)、[`wiki/queries/mpc-solver-selection.md`](wiki/queries/mpc-solver-selection.md)、[`sources/README.md`](sources/README.md)

## [2026-06-10] ingest | sources/sites/omg-tsinghua-mars-lab-github-io.md — OMG omni-modal G1 运动生成（清华 MARS Lab）入库并建实体页

- 原始资料：[`omg-tsinghua-mars-lab-github-io.md`](sources/sites/omg-tsinghua-mars-lab-github-io.md)（<https://tsinghua-mars-lab.github.io/OMG/>）；[`omg-tsinghua-mars-lab.md`](sources/repos/omg-tsinghua-mars-lab.md)（<https://github.com/tsinghua-mars-lab/OMG>）
- 新建实体：[`wiki/entities/paper-omg-omni-modal-humanoid-control.md`](wiki/entities/paper-omg-omni-modal-humanoid-control.md)（generator–tracker 分层、OMG-DiT 多模态条件、OMG-Data 规模、HoloMotion tracker 部署与 Mermaid 管线）
- 交叉更新：[`wiki/methods/diffusion-motion-generation.md`](wiki/methods/diffusion-motion-generation.md)、[`wiki/concepts/whole-body-tracking-pipeline.md`](wiki/concepts/whole-body-tracking-pipeline.md)、[`wiki/entities/holomotion.md`](wiki/entities/holomotion.md)、[`sources/README.md`](sources/README.md)

## [2026-06-10] structural | wiki/formalizations/field-oriented-control-derivation.md — FOC Clarke/Park 与 dq 转矩方程逐步推导；交叉链接概念页与设计流程

## [2026-06-10] ingest | sources/sites/ansys_motor_cad_electric_machine_design.md — 电机设计流程入库；wiki/overview/motor-design-workflow.md 并与 FOC/TN/仿真选型交叉链接

## [2026-06-10] ingest | sources/personal/motor_curves_and_em_simulation_faq.md — TN/TI 曲线与电机电磁仿真软件入库；wiki/concepts/motor-torque-speed-curve.md、wiki/concepts/motor-torque-current-curve.md、wiki/comparisons/motor-em-simulation-software.md

## [2026-06-10] ingest | sources/papers/rhythm_arxiv_2603_02856.md — Rhythm 双 G1 交互全身控制（IAMR+IGRL+MAGIC）入库并建实体页

- 原始资料：[`rhythm_arxiv_2603_02856.md`](sources/papers/rhythm_arxiv_2603_02856.md)（<https://arxiv.org/abs/2603.02856>）；[`hoshi-no-ai-rhythm-github-io.md`](sources/sites/hoshi-no-ai-rhythm-github-io.md)（<https://hoshi-no-ai.github.io/Rhythm/>）
- 新建实体：[`wiki/entities/paper-rhythm-dual-humanoid-interaction.md`](wiki/entities/paper-rhythm-dual-humanoid-interaction.md)（IAMR 解耦重定向 + IGRL 图奖励 MAPPO + 真机部署 + MAGIC 数据集 + Mermaid 管线）
- 交叉更新：[`wiki/concepts/whole-body-tracking-pipeline.md`](wiki/concepts/whole-body-tracking-pipeline.md)、[`wiki/concepts/motion-retargeting-pipeline.md`](wiki/concepts/motion-retargeting-pipeline.md)、[`wiki/methods/marl.md`](wiki/methods/marl.md)、[`wiki/entities/paper-assistmimic.md`](wiki/entities/paper-assistmimic.md)、[`sources/README.md`](sources/README.md)

## [2026-06-10] structural | wiki/concepts/humanoid-policy-network-architecture.md — 新增「架构代际对比表」

- 在 [`wiki/concepts/humanoid-policy-network-architecture.md`](wiki/concepts/humanoid-policy-network-architecture.md) 的架构演化总览后新增六行对比表：浅层 MLP / AMP / MoE / Transformer-Diffusion / VLA-WAM / 低层小网，按「骨干规模、输入输出、代表工作、强项、主要局限」横向对比，并强调真机低层高频策略与上层新架构分层共存

## [2026-06-10] ingest | sources/papers/dit4dit_arxiv_2603_10448.md — DiT4DiT 双 DiT 联合 VAM 入库；wiki/entities/paper-dit4dit-video-action-model.md 并与 MotionWAM 双向链接

## [2026-06-10] ingest | sources/papers/motionwam_arxiv_2606_09215.md — MotionWAM 实时人形 loco-manipulation WAM 入库；wiki/entities/paper-motionwam-humanoid-loco-manipulation-wam.md、wiki/concepts/world-action-models.md、wiki/tasks/loco-manipulation.md 等交叉更新

## [2026-06-09] query | wiki/queries/perception-backbone-selection.md — V24 P1 视觉表征知识链收官（机器人感知骨干/表征选型 Query）

- 新建 Query：[`wiki/queries/perception-backbone-selection.md`](wiki/queries/perception-backbone-selection.md)（「分类骨干 / 检测头 / 通用预训练表征」三类选型决策树、推荐组合 pipeline、关键工程经验与典型失败模式；含 Mermaid 决策树与缩写速查）
- 交叉补强（消孤儿）：在 [`wiki/comparisons/cnn-vs-vit-backbones.md`](wiki/comparisons/cnn-vs-vit-backbones.md)、[`wiki/concepts/visual-representation-for-policy.md`](wiki/concepts/visual-representation-for-policy.md)、[`wiki/queries/object-detection-model-selection.md`](wiki/queries/object-detection-model-selection.md) 增加双向回链
- 清单推进：[`docs/checklists/tech-stack-next-phase-checklist-v24.md`](docs/checklists/tech-stack-next-phase-checklist-v24.md) P1「视觉表征知识链 (+3)」三页全部 `[x]`；同步更新 [`wiki/queries/README.md`](wiki/queries/README.md) 索引
- 门禁：`make lint` 全绿、`ci-preflight` 派生文件已重生（图谱 804 节点 / 5644 边，无孤儿节点）

## [2026-06-09] ingest | sources/papers/vision_banana_arxiv_2604_20329.md — Vision Banana（DeepMind）生成式视觉预训练入库并建实体/概念页

- 原始资料：[`vision_banana_arxiv_2604_20329.md`](sources/papers/vision_banana_arxiv_2604_20329.md)（<https://arxiv.org/abs/2604.20329>）；[`vision-banana-project.md`](sources/sites/vision-banana-project.md)（<https://vision-banana.github.io/>、<https://deepmind.google/research/publications/240658/>）
- 新建实体：[`wiki/entities/vision-banana.md`](wiki/entities/vision-banana.md)（NBP instruction-tuning、RGB 任务统一接口、2D/3D benchmark 表与 Mermaid 管线）
- 新建概念：[`wiki/concepts/generative-vision-pretraining.md`](wiki/concepts/generative-vision-pretraining.md)（生成预训练 ≈ LLM 预训练、三条技术谱系）
- 交叉更新：[`wiki/concepts/vision-backbones.md`](wiki/concepts/vision-backbones.md)、[`wiki/concepts/visual-representation-for-policy.md`](wiki/concepts/visual-representation-for-policy.md)、[`wiki/formalizations/3d-coordinate-transforms-vision-robotics.md`](wiki/formalizations/3d-coordinate-transforms-vision-robotics.md)、[`sources/README.md`](sources/README.md)

## [2026-06-09] ingest | sources/papers/mamma_arxiv_2506_13040.md — MAMMA（CVPR 2026 Oral）markerless 多视角双人 SMPL-X 采集入库并建实体页

- 原始资料：[`mamma_arxiv_2506_13040.md`](sources/papers/mamma_arxiv_2506_13040.md)（<https://arxiv.org/abs/2506.13040>）；[`mamma-tue-mpg-de.md`](sources/sites/mamma-tue-mpg-de.md)（<https://mamma.is.tue.mpg.de/>）；[`mamma.md`](sources/repos/mamma.md)（<https://github.com/cuevhv/mamma>）
- 新建实体：[`wiki/entities/paper-mamma-markerless-motion-capture.md`](wiki/entities/paper-mamma-markerless-motion-capture.md)（MammaNet 稠密 landmark + 跨视角匹配 + SMPL-X 优化 + Mermaid 管线）
- 交叉更新：[`wiki/concepts/motion-retargeting-pipeline.md`](wiki/concepts/motion-retargeting-pipeline.md)、[`wiki/entities/freemocap.md`](wiki/entities/freemocap.md)、[`wiki/overview/paper-notebook-category-14-human-motion.md`](wiki/overview/paper-notebook-category-14-human-motion.md)、[`sources/README.md`](sources/README.md)

## [2026-06-09] ingest | sources/papers/dimos_arxiv_2305_12411.md — DIMOS 室内人–场景运动合成（ICCV 2023）入库并建实体页

- 原始资料：[`dimos_arxiv_2305_12411.md`](sources/papers/dimos_arxiv_2305_12411.md)（<https://arxiv.org/abs/2305.12411>）；[`dimos-zkf1997-github-io.md`](sources/sites/dimos-zkf1997-github-io.md)（<https://zkf1997.github.io/DIMOS/>）；[`dimos.md`](sources/repos/dimos.md)（<https://github.com/zkf1997/DIMOS>）
- 新建实体：[`wiki/entities/paper-dimos-human-scene-motion-synthesis.md`](wiki/entities/paper-dimos-human-scene-motion-synthesis.md)（RL + CVAE 潜空间 + 场景感知 locomotion/interaction + Mermaid 管线）
- 交叉更新：[`wiki/concepts/character-animation-vs-robotics.md`](wiki/concepts/character-animation-vs-robotics.md)、[`wiki/methods/diffusion-motion-generation.md`](wiki/methods/diffusion-motion-generation.md)、[`wiki/methods/crisp-real2sim.md`](wiki/methods/crisp-real2sim.md)、[`wiki/entities/paper-amp-survey-15-physhsi.md`](wiki/entities/paper-amp-survey-15-physhsi.md)、[`wiki/overview/paper-notebook-category-14-human-motion.md`](wiki/overview/paper-notebook-category-14-human-motion.md)

## [2026-06-09] ingest | sources/papers/dart_control_arxiv_2410_05260.md — DART/DartControl（ICLR 2025）论文+仓库+项目页入库并建方法页

- 原始资料：[`dart_control_arxiv_2410_05260.md`](sources/papers/dart_control_arxiv_2410_05260.md)（<https://arxiv.org/abs/2410.05260>）；[`zkf1997_dart.md`](sources/repos/zkf1997_dart.md)（<https://github.com/zkf1997/DART>）；[`dart-control-project.md`](sources/sites/dart-control-project.md)（<https://zkf1997.github.io/DART/>）
- 新建方法页：[`wiki/methods/dart-control.md`](wiki/methods/dart-control.md)（自回归运动原语潜扩散 + 在线文本/空间控制 + Mermaid 管线）
- 交叉更新：[`wiki/methods/diffusion-motion-generation.md`](wiki/methods/diffusion-motion-generation.md)、[`wiki/methods/hy-motion-1.md`](wiki/methods/hy-motion-1.md)、[`wiki/methods/genmo.md`](wiki/methods/genmo.md)、[`wiki/entities/awesome-text-to-motion-zilize.md`](wiki/entities/awesome-text-to-motion-zilize.md)、[`wiki/entities/phc.md`](wiki/entities/phc.md)、[`wiki/entities/amass.md`](wiki/entities/amass.md)、[`wiki/comparisons/wbc-vs-rl.md`](wiki/comparisons/wbc-vs-rl.md)、[`sources/README.md`](sources/README.md)

## [2026-06-09] ingest | sources/papers/coins_arxiv_2207_12824.md — COINS 论文 + 项目页 + 仓库入库并建实体页

- 原始资料：[`coins_arxiv_2207_12824.md`](sources/papers/coins_arxiv_2207_12824.md)（<https://arxiv.org/abs/2207.12824>）；[`coins-zkf1997-github-io.md`](sources/sites/coins-zkf1997-github-io.md)（<https://zkf1997.github.io/COINS/index.html>）；[`coins.md`](sources/repos/coins.md)（<https://github.com/zkf1997/COINS>）
- 新建实体：[`wiki/entities/paper-coins-compositional-human-scene-interaction.md`](wiki/entities/paper-coins-compositional-human-scene-interaction.md)（PelvisVAE/BodyVAE 三阶段 + 组合交互 + PROX-S + Mermaid 管线）
- 交叉更新：[`wiki/methods/crisp-real2sim.md`](wiki/methods/crisp-real2sim.md)、[`sources/README.md`](sources/README.md)

## [2026-06-09] ingest | sources/repos/robot_lab.md — 复核 fan-ziqi/robot_lab 并刷新实体页（24+ 环境、新机型、rl_sar 部署链、Mermaid）

- 原始资料：[`sources/repos/robot_lab.md`](sources/repos/robot_lab.md)（<https://github.com/fan-ziqi/robot_lab>）
- 刷新实体：[`wiki/entities/robot-lab.md`](wiki/entities/robot-lab.md)（版本矩阵、24 环境机型表、BeyondMimic/AMP 实验任务、Sim2Real→rl_sar Mermaid）
- 交叉更新：[`wiki/concepts/wheel-legged-quadruped.md`](wiki/concepts/wheel-legged-quadruped.md)、[`wiki/entities/openloong.md`](wiki/entities/openloong.md)

## [2026-06-09] ingest | sources/papers/humanoid_gym_arxiv_2404_05695.md — Humanoid-Gym 论文 + 官方/社区仓库入库并建实体页

- 原始资料：[`humanoid_gym_arxiv_2404_05695.md`](sources/papers/humanoid_gym_arxiv_2404_05695.md)（<https://arxiv.org/abs/2404.05695>）；[`humanoid-gym.md`](sources/repos/humanoid-gym.md)（<https://github.com/roboterax/humanoid-gym>）；[`humanoid-gym-modified.md`](sources/repos/humanoid-gym-modified.md)（<https://github.com/roboman-ly/humanoid-gym-modified>）
- 新建实体：[`wiki/entities/humanoid-gym.md`](wiki/entities/humanoid-gym.md)（步态相位奖励 + 非对称 AC + MuJoCo sim2sim + Mermaid 管线；含 Pandaman/Gazebo fork 小节）
- 交叉更新：[`wiki/entities/legged-gym.md`](wiki/entities/legged-gym.md)、[`references/repos/rl-frameworks.md`](references/repos/rl-frameworks.md)、[`wiki/overview/paper-notebook-category-03-high-impact-selection.md`](wiki/overview/paper-notebook-category-03-high-impact-selection.md)、[`sources/README.md`](sources/README.md)

## [2026-06-09] ingest | sources/papers/resmimic_arxiv_2510_05070.md — ResMimic GMT→残差全身 loco-manipulation 入库并建实体页

- 原始资料：[`resmimic_arxiv_2510_05070.md`](sources/papers/resmimic_arxiv_2510_05070.md)（<https://arxiv.org/abs/2510.05070>）；[`resmimic-github-io.md`](sources/sites/resmimic-github-io.md)（<https://resmimic.github.io/>）；[`resmimic.md`](sources/repos/resmimic.md)（<https://github.com/amazon-far/ResMimic>）
- 新建实体：[`wiki/entities/paper-resmimic.md`](wiki/entities/paper-resmimic.md)（两阶段残差 + 点云/接触奖励 + 虚拟力课程 + Mermaid 管线）
- 交叉更新：[`wiki/tasks/loco-manipulation.md`](wiki/tasks/loco-manipulation.md)、[`wiki/concepts/whole-body-tracking-pipeline.md`](wiki/concepts/whole-body-tracking-pipeline.md)、[`sources/README.md`](sources/README.md)

## [2026-06-09] ingest | sources/papers/ladderman_arxiv_2606_05873.md — LadderMan 人形感知梯子攀爬（项目页 + arXiv）消化并建实体页

- 原始资料：[`sources/papers/ladderman_arxiv_2606_05873.md`](sources/papers/ladderman_arxiv_2606_05873.md)（<https://arxiv.org/abs/2606.05873>）；[`sources/sites/ladderman-robot-github-io.md`](sources/sites/ladderman-robot-github-io.md)（<https://ladderman-robot.github.io/>）
- 新建实体：[`wiki/entities/paper-ladderman-humanoid-perceptive-ladder-climbing.md`](wiki/entities/paper-ladderman-humanoid-perceptive-ladder-climbing.md)（两阶段 hybrid tracking + DAgger+RL、VFM/RFM sim-to-real、梯上双智能体操作 + Mermaid 管线）
- 交叉更新：[`wiki/tasks/stair-obstacle-perceptive-locomotion.md`](wiki/tasks/stair-obstacle-perceptive-locomotion.md)、[`wiki/tasks/loco-manipulation.md`](wiki/tasks/loco-manipulation.md)、[`wiki/entities/unitree-g1.md`](wiki/entities/unitree-g1.md)、[`wiki/entities/paper-rpl-robust-humanoid-perceptive-locomotion.md`](wiki/entities/paper-rpl-robust-humanoid-perceptive-locomotion.md)、[`wiki/methods/dagger.md`](wiki/methods/dagger.md)、[`wiki/concepts/privileged-training.md`](wiki/concepts/privileged-training.md)

## [2026-06-09] ingest | sources/papers/limmt_arxiv_2606_06953.md — LIMMT（ICML 2026，GQS 数据策展 3% AMASS 胜全量）入库

- 原始资料：[`limmt_arxiv_2606_06953.md`](sources/papers/limmt_arxiv_2606_06953.md)（<https://arxiv.org/abs/2606.06953>）；[`limmt-giraffeguan-github-io.md`](sources/sites/limmt-giraffeguan-github-io.md)（<https://giraffeguan.github.io/limmt/>）
- 新建方法页：[`wiki/methods/limmt-gqs-motion-curation.md`](wiki/methods/limmt-gqs-motion-curation.md)（GQS 三阶段 + Mermaid + Any2Track/TWIST2/PHUMA/G1 实验归纳）
- 交叉更新：[`wiki/methods/egm-efficient-general-mimic.md`](wiki/methods/egm-efficient-general-mimic.md)、[`wiki/queries/humanoid-motion-tracking-method-selection.md`](wiki/queries/humanoid-motion-tracking-method-selection.md)、[`wiki/concepts/whole-body-tracking-pipeline.md`](wiki/concepts/whole-body-tracking-pipeline.md)、[`sources/README.md`](sources/README.md)

## [2026-06-09] ingest | sources/repos/python_robotics.md — 接入 PythonRobotics 代码库/教材/arXiv 论文并新建实体页与导航栈交叉引用

- 原始资料：[`sources/repos/python_robotics.md`](sources/repos/python_robotics.md)、[`sources/papers/python_robotics_arxiv_1808_10703.md`](sources/papers/python_robotics_arxiv_1808_10703.md)、[`sources/courses/python_robotics_textbook.md`](sources/courses/python_robotics_textbook.md)
- 新建实体：[`wiki/entities/python-robotics.md`](wiki/entities/python-robotics.md)
- 交叉更新：[`wiki/overview/navigation-slam-autonomy-stack.md`](wiki/overview/navigation-slam-autonomy-stack.md)、[`wiki/entities/navigation2.md`](wiki/entities/navigation2.md)、[`wiki/entities/modern-robotics-book.md`](wiki/entities/modern-robotics-book.md)、[`wiki/formalizations/kalman-filter.md`](wiki/formalizations/kalman-filter.md)

## [2026-06-08] structural | wiki/concepts/visual-representation-for-policy.md — V24 P1 视觉表征知识链第二项：视觉表征作为策略输入

- 新建概念页：[`wiki/concepts/visual-representation-for-policy.md`](wiki/concepts/visual-representation-for-policy.md)（端到端联合训练 vs 冻结预训练骨干 vs 机器人专用预训练表征（R3M / VC-1 / DINOv2）三条路径与取舍 + Mermaid 决策图）
- 交叉回链：[`wiki/concepts/vision-backbones.md`](wiki/concepts/vision-backbones.md) 新增 related/关联出边，消除孤儿页
- 进展：V24 P1「视觉表征知识链 (+3)」第二项交付（首项 cnn-vs-vit-backbones 已于 2026-06-07 完成），余 `wiki/queries/perception-backbone-selection.md`
- lint：`python3 scripts/lint_wiki.py` 全绿；同步重建全站索引与图谱统计

## [2026-06-08] ingest | sources/repos/* — 补充 fairmotion / AMP-RSL-RL 两个重定向相关成熟仓库实体并互链

- 原始资料：`sources/repos/amp_rsl_rl.md`、`sources/repos/fairmotion.md`
- 新建实体：[`wiki/entities/amp-rsl-rl.md`](wiki/entities/amp-rsl-rl.md)、[`wiki/entities/fairmotion.md`](wiki/entities/fairmotion.md)
- 交叉更新：[`wiki/concepts/motion-retargeting.md`](wiki/concepts/motion-retargeting.md)、[`references/repos/retarget-tools.md`](references/repos/retarget-tools.md)、[`sources/README.md`](sources/README.md)
- 说明：fairmotion 经核实为已归档(2023)的通用动捕工具、本身不做机器人重定向，按「上游数据基础设施」收录（与 FreeMoCap/MotionCode 同列）；AMP-RSL-RL 为 IIT 的 rsl_rl+AMP 人形模仿实现。

## [2026-06-08] ingest | sources/repos/* — 补全人形/四足重定向成熟开源仓库实体（14 页）并互链 motion-retargeting 主线

- 原始资料：`sources/repos/mocap_retarget.md`、`soma_retargeter.md`、`gvhmr.md`、`videomimic.md`、`phc.md`、`human2humanoid.md`、`motion_imitation_peng.md`、`amp_for_hardware.md`、`metalhead.md`、`leggedgym_ex.md`、`stmr_quadruped_retargeting.md`、`go2_motion_imitation.md`、`pan_motion_retargeting.md`、`walk_the_dog.md`
- 新建实体：[`wiki/entities/mocap-retarget.md`](wiki/entities/mocap-retarget.md)、[`soma-retargeter.md`](wiki/entities/soma-retargeter.md)、[`gvhmr.md`](wiki/entities/gvhmr.md)、[`videomimic.md`](wiki/entities/videomimic.md)、[`phc.md`](wiki/entities/phc.md)、[`human2humanoid.md`](wiki/entities/human2humanoid.md)、[`motion-imitation-quadruped.md`](wiki/entities/motion-imitation-quadruped.md)、[`amp-for-hardware.md`](wiki/entities/amp-for-hardware.md)、[`metalhead.md`](wiki/entities/metalhead.md)、[`leggedgym-ex.md`](wiki/entities/leggedgym-ex.md)、[`stmr-quadruped-retargeting.md`](wiki/entities/stmr-quadruped-retargeting.md)、[`go2-motion-imitation.md`](wiki/entities/go2-motion-imitation.md)、[`pan-motion-retargeting.md`](wiki/entities/pan-motion-retargeting.md)、[`walk-the-dog.md`](wiki/entities/walk-the-dog.md)
- 交叉更新：[`wiki/concepts/motion-retargeting.md`](wiki/concepts/motion-retargeting.md)、[`references/repos/retarget-tools.md`](references/repos/retarget-tools.md)、[`sources/README.md`](sources/README.md)

## [2026-06-08] ingest | sources/papers/omniretarget_arxiv_2509_26633.md — OmniRetarget（ICRA 2026）全文消化：holosoma 代码 + HF 数据集 + 项目页

- 原始资料：[`omniretarget_arxiv_2509_26633.md`](sources/papers/omniretarget_arxiv_2509_26633.md)（<https://arxiv.org/abs/2509.26633>、PDF <https://omniretarget.github.io/static/images/paper.pdf>）；[`omniretarget-github-io.md`](sources/sites/omniretarget-github-io.md)；[`holosoma.md`](sources/repos/holosoma.md)（<https://github.com/amazon-far/holosoma>）；[`omniretarget-dataset-huggingface.md`](sources/sites/omniretarget-dataset-huggingface.md)（<https://huggingface.co/datasets/omniretarget/OmniRetarget_Dataset>）
- 深化实体：[`wiki/entities/paper-hrl-stack-03-omniretarget.md`](wiki/entities/paper-hrl-stack-03-omniretarget.md)（interaction mesh + Sequential SOCP + 5 reward 下游 RL + Mermaid 管线）
- 新建实体：[`wiki/entities/holosoma.md`](wiki/entities/holosoma.md)、[`wiki/entities/omniretarget-dataset.md`](wiki/entities/omniretarget-dataset.md)
- 交叉更新：[`wiki/concepts/motion-retargeting.md`](wiki/concepts/motion-retargeting.md)、[`sources/README.md`](sources/README.md)

## [2026-06-08] ingest | sources/blogs/wechat_embodied_ai_lab_robot_training_stack_layers_2026.md — Agent Reach 抓取训练栈分层长文并建六层技术地图

- 工具：已安装 [Panniantong/Agent-Reach](https://github.com/Panniantong/Agent-Reach) v1.4.0（修复 hatchling `guides` force-include 重复后 `pip install -e` + `agent-reach install --channels=wechat`）；微信正文经 `~/.agent-reach/tools/wechat-article-for-ai`（Camoufox）
- 原始资料：[`wechat_embodied_ai_lab_robot_training_stack_layers_2026.md`](sources/blogs/wechat_embodied_ai_lab_robot_training_stack_layers_2026.md)（<https://mp.weixin.qq.com/s/Z9pgVa48wQKLYVRD3psnhw>）；[`mujoco_playground.md`](sources/repos/mujoco_playground.md)；落盘 [`sources/raw/wechat_embodied_ai_lab_robot_training_stack_layers_2026-06-08.md`](sources/raw/wechat_embodied_ai_lab_robot_training_stack_layers_2026-06-08.md)
- 沉淀页面：[`wiki/overview/robot-training-stack-layers-technology-map.md`](wiki/overview/robot-training-stack-layers-technology-map.md)（六层训练–评估栈 + Mermaid）；新建 [`wiki/entities/mujoco-playground.md`](wiki/entities/mujoco-playground.md)
- 交叉更新：[`isaac-lab.md`](wiki/entities/isaac-lab.md)、[`mujoco.md`](wiki/entities/mujoco.md)、[`mjlab.md`](wiki/entities/mjlab.md)、[`unilab.md`](wiki/entities/unilab.md)、[`newton-physics.md`](wiki/entities/newton-physics.md)、[`genesis-world-10.md`](wiki/entities/genesis-world-10.md)、[`simulator-selection-guide.md`](wiki/queries/simulator-selection-guide.md)、[`simulation-evaluation-infrastructure.md`](wiki/concepts/simulation-evaluation-infrastructure.md)、[`humanoid-rl-motion-control-body-system-stack.md`](wiki/overview/humanoid-rl-motion-control-body-system-stack.md)、[`mujoco-vs-isaac-lab.md`](wiki/comparisons/mujoco-vs-isaac-lab.md)、[`agent-reach.md`](wiki/entities/agent-reach.md)、[`sources/README.md`](sources/README.md)

## [2026-06-07] structural | wiki/comparisons/cnn-vs-vit-backbones.md — V24 P1 视觉表征知识链首页：CNN vs ViT 视觉骨干对比

- 新增页面：[wiki/comparisons/cnn-vs-vit-backbones.md](wiki/comparisons/cnn-vs-vit-backbones.md)（归纳偏置、数据量需求、分辨率/吞吐、多尺度特征、边缘部署、下游迁移六维对比；机器人感知取舍决策图与误区）
- 交叉回链：[wiki/concepts/vision-backbones.md](wiki/concepts/vision-backbones.md)、[wiki/methods/object-detection.md](wiki/methods/object-detection.md) 关联页面区块增补对比页入链，消除孤儿节点
- 派生同步：`make graph` + `make badge` + `make export`（知识图谱 772→773 节点 / 5332→5338 边，README badge 同步）；`make lint` 0 阻塞问题，搜索回归 37/37，导出质量 12/12
- 清单推进：[docs/checklists/tech-stack-next-phase-checklist-v24.md](docs/checklists/tech-stack-next-phase-checklist-v24.md) P1「视觉表征知识链 (+3)」首项打勾

## [2026-06-07] ingest | sources/papers/esi_bench_arxiv_2605_18746.md — 补强 ESI-Bench 动作空间/基准对比并交叉 VLN

- 原始资料：[sources/papers/esi_bench_arxiv_2605_18746.md](sources/papers/esi_bench_arxiv_2605_18746.md)（<https://arxiv.org/abs/2605.18746>）、[sources/sites/esi-bench-project.md](sources/sites/esi-bench-project.md)（<https://esi-bench.github.io/>）、[sources/repos/esi_bench.md](sources/repos/esi_bench.md)（<https://github.com/ESI-Bench/ESI-Bench>）
- 消化实体：[wiki/entities/esi-bench.md](wiki/entities/esi-bench.md)（任务形式化、高层动作空间、与 VSI-Bench/EmbodiedBench 定位表）
- 交叉补强：[wiki/concepts/3d-spatial-vqa.md](wiki/concepts/3d-spatial-vqa.md)、[wiki/tasks/vision-language-navigation.md](wiki/tasks/vision-language-navigation.md)

## [2026-06-07] ingest | sources/papers/eth-g1-diffusion.md — Learning Whole-Body Humanoid Locomotion（arXiv:2604.17335）扩散生成 + RL 全身跟踪真机 G1

- 原始资料：[sources/papers/eth-g1-diffusion.md](sources/papers/eth-g1-diffusion.md)、[sources/sites/wholebody-locomotion.md](sources/sites/wholebody-locomotion.md)
- 消化实体：[wiki/entities/paper-hrl-stack-27-learning_whole_body_humanoid_locomot.md](wiki/entities/paper-hrl-stack-27-learning_whole_body_humanoid_locomot.md)（由 42 篇栈索引级升格为全文消化）
- 交叉补强：[wiki/methods/diffusion-motion-generation.md](wiki/methods/diffusion-motion-generation.md)、[wiki/tasks/humanoid-locomotion.md](wiki/tasks/humanoid-locomotion.md)、[wiki/entities/unitree-g1.md](wiki/entities/unitree-g1.md)

## [2026-06-07] ingest | sources/sites/blender-org.md、sources/repos/blender.md — Blender 开源 DCC 官网与官方仓库；wiki/entities/blender.md

- 原始资料：[sources/sites/blender-org.md](sources/sites/blender-org.md)、[sources/repos/blender.md](sources/repos/blender.md)
- 新增实体：[wiki/entities/blender.md](wiki/entities/blender.md)
- 交叉补强：[wiki/entities/nvidia-omniverse.md](wiki/entities/nvidia-omniverse.md)、[wiki/concepts/character-animation-vs-robotics.md](wiki/concepts/character-animation-vs-robotics.md)、[wiki/entities/robot-motion-keyframe-editors.md](wiki/entities/robot-motion-keyframe-editors.md)、[wiki/entities/sam3dbody-cpp.md](wiki/entities/sam3dbody-cpp.md)

## [2026-06-07] ingest | sources/papers/rpl_arxiv_2602_03002.md — RPL（arXiv:2602.03002）Amazon FAR 人形多向深度感知行走与载荷 loco-manipulation

- 原始资料：[sources/papers/rpl_arxiv_2602_03002.md](sources/papers/rpl_arxiv_2602_03002.md)、[sources/sites/rpl-humanoid-github-io.md](sources/sites/rpl-humanoid-github-io.md)
- 新增实体：[wiki/entities/paper-rpl-robust-humanoid-perceptive-locomotion.md](wiki/entities/paper-rpl-robust-humanoid-perceptive-locomotion.md)
- 交叉补强：[wiki/tasks/stair-obstacle-perceptive-locomotion.md](wiki/tasks/stair-obstacle-perceptive-locomotion.md)、[wiki/tasks/loco-manipulation.md](wiki/tasks/loco-manipulation.md)

## [2026-06-07] structural | schema + wiki — Paper Notebooks 64 篇 stub 实体从深读笔记同步一句话总结

- 工具：[scripts/sync_paper_notebook_summaries.py](scripts/sync_paper_notebook_summaries.py)；`make paper-notebook-summaries`
- 从 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html) 各篇 `## 🎯 一句话总结` 同步至 `wiki/entities/paper-notebook-*` 的 frontmatter `summary` 与 `## 一句话定义`，并更新对应 `sources/papers/humanoid_pnb_*.md` 一句话说明
- RL Sim2Sim Demo 映射 7 页（[schema/rl-sim2sim-demo-wiki-map.yml](schema/rl-sim2sim-demo-wiki-map.yml)）经审计已具备有效 summary，无需改动

## [2026-06-07] structural | schema + wiki — Paper Notebooks 全量分类父节点与 64 篇未映射论文 sources/实体入库

- 工具：[scripts/bootstrap_paper_notebook_knowledge.py](scripts/bootstrap_paper_notebook_knowledge.py)；`make paper-notebook-bootstrap`；完整映射 [schema/paper-notebook-wiki-full-map.yml](schema/paper-notebook-wiki-full-map.yml)（137/137）
- 父节点：[wiki/overview/humanoid-paper-notebooks-index.md](wiki/overview/humanoid-paper-notebooks-index.md) + 14 类 `wiki/overview/paper-notebook-category-*.md`（03 类含 5 个子分类段落）
- 新增 64 组 `sources/papers/humanoid_pnb_*.md` + `wiki/entities/paper-notebook-*.md` 索引实体；既有 73 篇保留原深度 wiki 并挂入分类子节点

## [2026-06-07] structural | schema + wiki — 同步 RL Sim2Sim Demo 在线演示链接至对应 wiki 节点

- 工具：[scripts/sync_rl_sim2sim_demo_links.py](scripts/sync_rl_sim2sim_demo_links.py)；`make rl-sim2sim-demo-links`；映射 [schema/rl-sim2sim-demo-wiki-map.yml](schema/rl-sim2sim-demo-wiki-map.yml) + [schema/rl-sim2sim-demo-index.json](schema/rl-sim2sim-demo-index.json)
- 来源归档：[sources/sites/rl-sim2sim-demo-website.md](sources/sites/rl-sim2sim-demo-website.md)
- 挂接节点：[wiki/concepts/sim2real.md](wiki/concepts/sim2real.md)、[wiki/entities/amp-mjlab.md](wiki/entities/amp-mjlab.md)、[wiki/entities/paper-hrl-stack-22-perceptive_humanoid_parkour.md](wiki/entities/paper-hrl-stack-22-perceptive_humanoid_parkour.md)、[wiki/methods/gentlehumanoid-motion-tracking.md](wiki/methods/gentlehumanoid-motion-tracking.md) 等 7 页

## [2026-06-07] structural | schema + wiki — 同步 Humanoid Paper Notebooks 深读笔记链接至对应 wiki 节点

- 工具：[scripts/sync_paper_notebook_links.py](scripts/sync_paper_notebook_links.py)；`make paper-notebook-links`；映射 [schema/paper-notebook-wiki-overrides.yml](schema/paper-notebook-wiki-overrides.yml) + [schema/paper-notebook-index.json](schema/paper-notebook-index.json)
- 覆盖 73/138 篇已有对应节点的论文笔记；修复 5 处旧版 `02_High_Impact` / `09_Sim-to-Real` URL；示例节点 [wiki/entities/paper-sonic.md](wiki/entities/paper-sonic.md)、[wiki/methods/awr.md](wiki/methods/awr.md)、[wiki/tasks/teleoperation.md](wiki/tasks/teleoperation.md)

## [2026-06-06] structural | docs — V23 P3 图谱页「专题视图」扩充（WBT / 跨具身 / 真机安全微调）

- 清单推进：[tech-stack-next-phase-checklist-v23.md](docs/checklists/tech-stack-next-phase-checklist-v23.md) P3 末项打勾，V23 清单全部完成
- 前端改动：[docs/graph.html](docs/graph.html) `TOPIC_FILTERS` 在 V22 10 项基础上新增 `wbt`（segments 11 项，命中 22 节点）、`cross-embodiment`（segments 3 项，命中 3 节点）、`safe-fine-tuning`（community-13 + segments 9 项，命中 18 节点）；`#filter-topic-chips` 同步新增 🕺 WBT / 🔀 跨具身 / 🛡️ 安全微调 三枚 `data-topic` chip，复用既有 `nodeMatchesTopic` 双路并集逻辑
- 工具修复：[scripts/screenshot_graph_topic.cjs](scripts/screenshot_graph_topic.cjs) 由点击 V22 已移除的 `#topic-view` 下拉改为展开 `#filter-topic-section` 后点击 `[data-topic]` chip
- 验证：`make lint` 全绿（仅 1 条无关信息型预警）、内联 JS `new Function` 语法校验通过；三专题视图截图归档至 `.cursor-artifacts/screenshots/graph-topic-{wbt,cross-embodiment,safe-fine-tuning}.png`
- V23 验收：节点 690（≥445）/ 边 4993（≥3320）/ 事实库 172（≥170）/ `largest_community_ratio` 0.104（≤0.25）/ `community_quality_warning` false，全部达标

## [2026-06-06] ingest | sources/sites/karpathy-ai.md、sources/blogs/karpathy_llm_wiki_gist.md — Andrej Karpathy 个人站与 LLM Wiki Gist；wiki/entities/andrej-karpathy.md、wiki/references/llm-wiki-karpathy.md、wiki/overview/robot-learning-overview.md

## [2026-06-06] structural | wiki — 全库 587 页英文缩写速查区块重排至「一句话定义」后、「为什么重要」前；新增 reorder 脚本与 lint 位置校验

- 工具：`scripts/wiki_abbrev_section.py`、`scripts/reorder_abbrev_glossary.py`；`lint_wiki.py` 新增位置错误阻塞项；`gen_abbrev_glossary.py` 插入锚点对齐 schema
- 代表页：[wiki/concepts/sim2real.md](wiki/concepts/sim2real.md)、[wiki/entities/lerobot.md](wiki/entities/lerobot.md)、[wiki/entities/paper-bfm-zero.md](wiki/entities/paper-bfm-zero.md)、[wiki/queries/sim2real-checklist.md](wiki/queries/sim2real-checklist.md)

## [2026-06-06] ingest | sources/papers/cosmos3_arxiv_2606_02800.md、sources/sites/cosmos3-project.md、sources/repos/nvidia_cosmos.md — Cosmos 3 全模态 Physical AI 世界模型；wiki/entities/cosmos-3.md、wiki/methods/generative-world-models.md、wiki/concepts/world-action-models.md、wiki/methods/mimic-video.md、wiki/entities/nvidia-so101-sim2real-lab-workflow.md

## [2026-06-06] ingest | sources/papers/vision_backbone_detection_classics.md — 入库 ResNet (1512.03385) 与 YOLO v1 (1506.02640) 及视觉骨干/目标检测 wiki

- 原始资料：[resnet_arxiv_1512_03385.md](sources/papers/resnet_arxiv_1512_03385.md)（<https://arxiv.org/abs/1512.03385>）、[yolo_arxiv_1506_02640.md](sources/papers/yolo_arxiv_1506_02640.md)（<https://arxiv.org/abs/1506.02640>）、[vision_backbone_detection_classics.md](sources/papers/vision_backbone_detection_classics.md)
- 沉淀页面：[wiki/entities/paper-resnet-deep-residual-learning.md](wiki/entities/paper-resnet-deep-residual-learning.md)、[wiki/entities/paper-yolo-unified-realtime-detection.md](wiki/entities/paper-yolo-unified-realtime-detection.md)、[wiki/concepts/vision-backbones.md](wiki/concepts/vision-backbones.md)、[wiki/methods/object-detection.md](wiki/methods/object-detection.md)
- 交叉更新：[wiki/concepts/deep-learning-foundations.md](wiki/concepts/deep-learning-foundations.md)、[wiki/tasks/manipulation.md](wiki/tasks/manipulation.md)

## [2026-06-06] lint | health-check — 全库健康度提升至满分：isaac-gym / isaac-lab 补 summary frontmatter（health_score 2→3）；paper-learning-to-adapt-bio-inspired-quadruped-gait 补 venue 元数据；684/684 节点 health_score=3，lint 0 issues

## [2026-06-06] ingest | sources/papers/learning_to_adapt_nature_2025.md、sources/repos/ihcr_learning_to_adapt.md — Learning to Adapt（Nature MI 2025）四足 bio-inspired 多步态 DRL；wiki/entities/paper-learning-to-adapt-bio-inspired-quadruped-gait.md、wiki/concepts/gait-generation.md、wiki/tasks/locomotion.md、wiki/entities/quadruped-robot.md、wiki/entities/paper-walk-these-ways-quadruped-mob.md

## [2026-06-05] structural | docs — V23 P3 详情页「最近相关 ingest」时间线

- 清单推进：[tech-stack-next-phase-checklist-v23.md](docs/checklists/tech-stack-next-phase-checklist-v23.md) P3 首项打勾
- 前端改动：[docs/detail.html](docs/detail.html) 在 `detail-related` 与 `detail-recommended` 之间新增 `#detail-recent-ingest-section`（默认 `hidden`，空态整段含标题不渲染）；[docs/main.js](docs/main.js) 新增 `renderDetailRecentIngestTimeline`，并发取 `link-graph.json`（1-hop 邻居）与 `graph-stats.json`（`latest_wiki_nodes`）求交，窗口锚定最新一条 ingest 回溯 30 天，按 `recency` 倒序、最多 6 项；[docs/style.css](docs/style.css) 新增 `.detail-recent-ingest-*` 时间线样式
- 验证：`node --check` + `eslint docs/main.js` 全绿、`make lint` 全部检查通过；BFM 详情页截图命中 5 项

## [2026-06-05] structural | wiki — 连接度前 50 hub 页补齐英文缩写速查（第 11–50 名，40 页）

- 依据 `exports/link-graph.json` 总度排序：第 1–10 名已在 main，本次为第 11–50 名共 40 页新增 `## 英文缩写速查`；批量工具 `scripts/batch_insert_abbrev_glossary.py`
- 页面：`wiki/concepts/behavior-foundation-model.md`、`wiki/entities/humanoid-robot.md`、`wiki/methods/generative-world-models.md`、`wiki/entities/unitree-g1.md`、`wiki/tasks/loco-manipulation.md`、`wiki/concepts/motion-retargeting.md`、`wiki/entities/isaac-gym-isaac-lab.md`、`wiki/entities/mujoco.md`、`wiki/overview/robot-world-models-training-loop-taxonomy.md`、`wiki/concepts/foundation-policy.md`、`wiki/methods/sonic-motion-tracking.md`、`wiki/overview/bfm-category-02-goal-conditioned-learning.md`、`wiki/overview/world-models-15-open-source-technology-map.md`、`wiki/queries/humanoid-motion-tracking-method-selection.md`、`wiki/methods/amp-reward.md`、`wiki/overview/navigation-slam-autonomy-stack.md`、`wiki/concepts/world-action-models.md`、`wiki/concepts/contact-rich-manipulation.md`、`wiki/concepts/whole-body-tracking-pipeline.md`、`wiki/methods/motion-retargeting-gmr.md`、`wiki/methods/model-predictive-control.md`、`wiki/concepts/motion-retargeting-pipeline.md`、`wiki/entities/legged-gym.md`、`wiki/tasks/teleoperation.md`、`wiki/overview/ego-9-papers-technology-map.md`、`wiki/queries/legged-humanoid-rl-pd-gain-setting.md`、`wiki/methods/model-based-rl.md`、`wiki/tasks/stair-obstacle-perceptive-locomotion.md`、`wiki/concepts/domain-randomization.md`、`wiki/overview/robot-learning-overview.md`、`wiki/methods/diffusion-policy.md`、`wiki/entities/paper-behavior-foundation-model-humanoid.md`、`wiki/entities/open-source-humanoid-hardware.md`、`wiki/methods/policy-optimization.md`、`wiki/concepts/reward-design.md`、`wiki/concepts/privileged-training.md`、`wiki/methods/beyondmimic.md`、`wiki/overview/bfm-category-05-hierarchical-control.md`、`wiki/entities/quadruped-robot.md`、`wiki/concepts/video-as-simulation.md`

## [2026-06-05] structural | wiki — 连接数前十 hub 页补齐英文缩写速查表

- 依据 `exports/graph-stats.json` 的 `top_hubs`（总度前十），在下列页面一句话定义/观点之后新增 `## 英文缩写速查` 三列表：`wiki/concepts/sim2real.md`、`wiki/tasks/locomotion.md`、`wiki/overview/humanoid-rl-motion-control-body-system-stack.md`、`wiki/methods/reinforcement-learning.md`、`wiki/methods/vla.md`、`wiki/overview/bfm-41-papers-technology-map.md`、`wiki/methods/imitation-learning.md`、`wiki/tasks/manipulation.md`、`wiki/concepts/whole-body-control.md`、`wiki/overview/humanoid-amp-motion-prior-survey.md`

## [2026-06-05] ingest | sources/papers/pilot_arxiv_2601_17440.md — PILOT 感知统一 loco-manipulation LLC（arXiv:2601.17440）；wiki/entities/paper-pilot-perceptive-loco-manipulation.md、wiki/tasks/loco-manipulation.md、wiki/tasks/stair-obstacle-perceptive-locomotion.md、wiki/entities/unitree-g1.md

- 原始资料：[pilot_arxiv_2601_17440.md](sources/papers/pilot_arxiv_2601_17440.md)（<https://arxiv.org/abs/2601.17440>）
- 沉淀页面：[wiki/entities/paper-pilot-perceptive-loco-manipulation.md](wiki/entities/paper-pilot-perceptive-loco-manipulation.md)
- 交叉更新：[wiki/tasks/loco-manipulation.md](wiki/tasks/loco-manipulation.md)、[wiki/tasks/stair-obstacle-perceptive-locomotion.md](wiki/tasks/stair-obstacle-perceptive-locomotion.md)、[wiki/concepts/whole-body-control.md](wiki/concepts/whole-body-control.md)、[wiki/tasks/teleoperation.md](wiki/tasks/teleoperation.md)、[wiki/entities/unitree-g1.md](wiki/entities/unitree-g1.md)

## [2026-06-05] structural | schema + wiki — 英文缩写速查表工作流与 SSR 页试点

- 工作流：[schema/page-types.md](schema/page-types.md)、[schema/ingest-workflow.md](schema/ingest-workflow.md)、[schema/linking.md](schema/linking.md)、[AGENTS.md](AGENTS.md)；`lint_wiki.py` 新增 `missing_abbrev_glossary` 信息型检查
- 试点页面：[wiki/entities/paper-ssr-humanoid-open-world-traversal.md](wiki/entities/paper-ssr-humanoid-open-world-traversal.md)（`## 英文缩写速查` 三列表）

## [2026-06-05] ingest | sources/papers/ssr_arxiv_2605_30770.md、sources/sites/ssr-humanoid-github-io.md — SSR 开放世界人形穿越（想象落脚点 + 潜空间对称 + 分地形 AMP）入库

- 原始资料：[ssr_arxiv_2605_30770.md](sources/papers/ssr_arxiv_2605_30770.md)（<https://arxiv.org/abs/2605.30770>、<https://arxiv.org/html/2605.30770v1>）；[ssr-humanoid-github-io.md](sources/sites/ssr-humanoid-github-io.md)（<https://ssr-humanoid.github.io/>）
- 沉淀页面：[wiki/entities/paper-ssr-humanoid-open-world-traversal.md](wiki/entities/paper-ssr-humanoid-open-world-traversal.md)（含单阶段 PPO + 三项机制 Mermaid 管线）
- 交叉更新：[wiki/tasks/stair-obstacle-perceptive-locomotion.md](wiki/tasks/stair-obstacle-perceptive-locomotion.md)、[wiki/tasks/humanoid-locomotion.md](wiki/tasks/humanoid-locomotion.md)、[sources/README.md](sources/README.md)

## [2026-06-05] ingest | sources/papers/sprint_arxiv_2605_28549.md、sources/sites/sprint-anonymous-project-page.md — SPRINT 人形竞技冲刺（频谱先验 + 6 m/s G1 真机）入库

- 原始资料：[sprint_arxiv_2605_28549.md](sources/papers/sprint_arxiv_2605_28549.md)（<https://arxiv.org/abs/2605.28549>、<https://arxiv.org/html/2605.28549v1>）；[sprint-anonymous-project-page.md](sources/sites/sprint-anonymous-project-page.md)（<https://anonymous.4open.science/w/SPRINT-138A/>）
- 沉淀页面：[wiki/entities/paper-sprint-humanoid-athletic-sprints.md](wiki/entities/paper-sprint-humanoid-athletic-sprints.md)（含三阶段 Mermaid 管线）
- 交叉更新：[wiki/tasks/humanoid-locomotion.md](wiki/tasks/humanoid-locomotion.md)、[wiki/queries/humanoid-motion-tracking-method-selection.md](wiki/queries/humanoid-motion-tracking-method-selection.md)、[sources/README.md](sources/README.md)

## [2026-06-05] ingest | sources/papers/homeworld_arxiv_2606_06390.md — HomeWorld（Kairos）全屋 sim-ready 场景生成入库

- 原始资料：[homeworld_arxiv_2606_06390.md](sources/papers/homeworld_arxiv_2606_06390.md)（<https://arxiv.org/abs/2606.06390>）；[kairos-homeworld-github-io.md](sources/sites/kairos-homeworld-github-io.md)（<https://kairos-homeworld.github.io/>）；[homeworld.md](sources/repos/homeworld.md)（<https://github.com/Kairos-HomeWorld/HomeWorld>，Coming Soon）
- 沉淀页面：[wiki/entities/paper-homeworld-whole-home-scene-generation.md](wiki/entities/paper-homeworld-whole-home-scene-generation.md)（含 Mermaid 四阶段流水线）
- 交叉更新：[wiki/concepts/video-as-simulation.md](wiki/concepts/video-as-simulation.md)、[wiki/tasks/manipulation.md](wiki/tasks/manipulation.md)、[wiki/tasks/vision-language-navigation.md](wiki/tasks/vision-language-navigation.md)、[sources/README.md](sources/README.md)

## [2026-06-05] ingest | sources/papers/host_humanoid_standingup_arxiv_2502_08378.md — HoST（RSS 2025）人形多姿态起身 RL 入库

- 原始资料：[host_humanoid_standingup_arxiv_2502_08378.md](sources/papers/host_humanoid_standingup_arxiv_2502_08378.md)（<https://arxiv.org/abs/2502.08378>）；[host-humanoid-standingup-project.md](sources/sites/host-humanoid-standingup-project.md)（<https://taohuang13.github.io/humanoid-standingup.github.io/>）；[host_internrobotics.md](sources/repos/host_internrobotics.md)（<https://github.com/InternRobotics/HoST>）
- 沉淀页面：[wiki/entities/paper-host-humanoid-standingup.md](wiki/entities/paper-host-humanoid-standingup.md)
- 交叉更新：[wiki/tasks/balance-recovery.md](wiki/tasks/balance-recovery.md)、[wiki/tasks/locomotion.md](wiki/tasks/locomotion.md)、[wiki/entities/unitree-g1.md](wiki/entities/unitree-g1.md)、[wiki/entities/paper-unified-walk-run-recovery-sdamp.md](wiki/entities/paper-unified-walk-run-recovery-sdamp.md)、[sources/README.md](sources/README.md)

## [2026-06-05] structural | wiki/methods/model-predictive-control.md — MPC 页补充滚动时域 Mermaid 流程图

- 页面：[wiki/methods/model-predictive-control.md](wiki/methods/model-predictive-control.md)（「有限时域优化」小节）

## [2026-06-05] ingest | sources/papers/explicit_stair_geometry_arxiv_2605_09944.md — 显式楼梯几何条件化人形爬梯（arXiv:2605.09944）入库

- 原始资料：[explicit_stair_geometry_arxiv_2605_09944.md](sources/papers/explicit_stair_geometry_arxiv_2605_09944.md)（<https://arxiv.org/abs/2605.09944>）
- 沉淀页面：[wiki/entities/paper-explicit-stair-geometry-humanoid-locomotion.md](wiki/entities/paper-explicit-stair-geometry-humanoid-locomotion.md)（含 Mermaid 训练—部署管线）
- 交叉更新：[wiki/tasks/locomotion.md](wiki/tasks/locomotion.md)、[wiki/tasks/stair-obstacle-perceptive-locomotion.md](wiki/tasks/stair-obstacle-perceptive-locomotion.md)、[wiki/concepts/terrain-adaptation.md](wiki/concepts/terrain-adaptation.md)、[wiki/entities/unitree-g1.md](wiki/entities/unitree-g1.md)、[wiki/entities/paper-faststair-humanoid-stair-ascent.md](wiki/entities/paper-faststair-humanoid-stair-ascent.md)、[sources/README.md](sources/README.md)

## [2026-06-05] ingest | sources/papers/faststair_arxiv_2601_10365.md — FastStair 挂接楼梯/障碍中心节点并刷新交叉引用

- 原始资料（已存在，本次补摘录 §5 感知定位）：[`sources/papers/faststair_arxiv_2601_10365.md`](sources/papers/faststair_arxiv_2601_10365.md)、[`sources/sites/npcliu-faststair-github-io.md`](sources/sites/npcliu-faststair-github-io.md)
- 新建中心节点：[`wiki/tasks/stair-obstacle-perceptive-locomotion.md`](wiki/tasks/stair-obstacle-perceptive-locomotion.md) — **带/不带感知 · 上下楼梯 · 越障** 维护挂接点
- 交叉更新：[`wiki/entities/paper-faststair-humanoid-stair-ascent.md`](wiki/entities/paper-faststair-humanoid-stair-ascent.md)、[`wiki/tasks/locomotion.md`](wiki/tasks/locomotion.md)、[`wiki/tasks/humanoid-locomotion.md`](wiki/tasks/humanoid-locomotion.md)、[`wiki/concepts/terrain-adaptation.md`](wiki/concepts/terrain-adaptation.md)、[`wiki/concepts/footstep-planning.md`](wiki/concepts/footstep-planning.md)、[`wiki/concepts/capture-point-dcm.md`](wiki/concepts/capture-point-dcm.md)、[`wiki/entities/paper-e-sds-environment-aware-humanoid-locomotion-rl.md`](wiki/entities/paper-e-sds-environment-aware-humanoid-locomotion-rl.md)、[`wiki/entities/dreamwaq-plus.md`](wiki/entities/dreamwaq-plus.md)、[`wiki/entities/extreme-parkour.md`](wiki/entities/extreme-parkour.md)、[`wiki/entities/paper-hrl-stack-22-perceptive_humanoid_parkour.md`](wiki/entities/paper-hrl-stack-22-perceptive_humanoid_parkour.md)、[`wiki/entities/paper-deep-whole-body-parkour.md`](wiki/entities/paper-deep-whole-body-parkour.md)、[`wiki/entities/paper-hiking-in-the-wild.md`](wiki/entities/paper-hiking-in-the-wild.md)、[`wiki/entities/paper-walk-these-ways-quadruped-mob.md`](wiki/entities/paper-walk-these-ways-quadruped-mob.md)、[`wiki/entities/jackhan-mujoco-walke3-simulation.md`](wiki/entities/jackhan-mujoco-walke3-simulation.md)

## [2026-06-04] structural | schema/canonical-facts.json — V23 P2 事实库扩展 156→172，补 WBT 跨具身与真机安全微调矛盾检测规则

- 推进 [tech-stack-next-phase-checklist-v23.md](docs/checklists/tech-stack-next-phase-checklist-v23.md) P2「事实库扩展」一项，达成 ≥170 条目标（实际 172）
- 新增 16 条事实：SONIC 规模化预训练 / Any2Any 跨具身迁移 / BFM 无参考 / SD-AMP 双判别器门控 / Heracles 扩散中间件 / WBT pipeline 端到端 / WBT 跨具身解耦 / SONIC-vs-Any2Any 训练范式 / SD-AMP-vs-Heracles 抽象层 / BeyondMimic 失败率采样 / SLowRL 安全 LoRA / 真机 RL 安全约束 / Sim2Real-vs-Real2Sim / 安全 LoRA 投影 / 跨具身策略迁移三路径 / CRISP Real2Sim
- 修正：收紧 `SD-AMP 状态门控双判别器` 的 neg 正则，避免误命中 [Heracles 页](wiki/entities/paper-heracles-humanoid-diffusion.md)「SD-AMP…单策略…判别器」对照描述
- 验证：`make lint` 潜在矛盾 0；`make ci-preflight` 导出质量 12/12 通过

## [2026-06-04] ingest | sources/blogs/wechat_shenlan_3d_coordinate_transforms.md、wechat_shenlan_riemannian_manifold_tangent_space.md — Agent Reach 抓取《具身智能基础》专栏 02/03 并建几何三篇父节点

- 工具：已安装 [Panniantong/Agent-Reach](https://github.com/Panniantong/Agent-Reach) v1.4.0（修复 hatchling `force-include` 重复后 `pip install -e` + `agent-reach install --channels=wechat`）；微信正文经 `~/.agent-reach/tools/wechat-article-for-ai`（Camoufox）
- 原始资料：[`wechat_shenlan_3d_coordinate_transforms.md`](sources/blogs/wechat_shenlan_3d_coordinate_transforms.md)（<https://mp.weixin.qq.com/s/P5Jm7bMhaTHsytHStFbbLg>）；[`wechat_shenlan_riemannian_manifold_tangent_space.md`](sources/blogs/wechat_shenlan_riemannian_manifold_tangent_space.md)（<https://mp.weixin.qq.com/s/uFTKN5FDvlHQxOSspvxVZw>）；落盘 [`sources/raw/wechat_shenlan_3d_coord_transforms_2026-06-04.md`](sources/raw/wechat_shenlan_3d_coord_transforms_2026-06-04.md)、[`sources/raw/wechat_shenlan_riemannian_manifold_2026-06-04.md`](sources/raw/wechat_shenlan_riemannian_manifold_2026-06-04.md)；专栏 01 已存在 [`wechat_shenlan_lie_group_lie_algebra_quaternion.md`](sources/blogs/wechat_shenlan_lie_group_lie_algebra_quaternion.md)（<https://mp.weixin.qq.com/s/JviRH2LW-fkCHA5gY7Qflw>）
- 沉淀页面：**父节点** [`wiki/overview/shenlan-embodied-ai-fundamentals-series.md`](wiki/overview/shenlan-embodied-ai-fundamentals-series.md)；子节点 [`wiki/formalizations/3d-coordinate-transforms-vision-robotics.md`](wiki/formalizations/3d-coordinate-transforms-vision-robotics.md)、[`wiki/formalizations/riemannian-manifold-tangent-space.md`](wiki/formalizations/riemannian-manifold-tangent-space.md)
- 交叉更新：[`wiki/formalizations/lie-group-rigid-body-motions.md`](wiki/formalizations/lie-group-rigid-body-motions.md)、[`wiki/overview/vla-open-source-repro-landscape-2025.md`](wiki/overview/vla-open-source-repro-landscape-2025.md)、[`sources/README.md`](sources/README.md)、[`sources/repos/panniantong_agent_reach.md`](sources/repos/panniantong_agent_reach.md)

## [2026-06-04] structural | wiki/concepts/motion-retargeting-pipeline.md — 流水线页三处公式改 `$...$` 以启用 KaTeX 蓝框；`docs/main.js` 保留 Mermaid `htmlLabels` 的 `<br/>` 换行

- 页面：[wiki/concepts/motion-retargeting-pipeline.md](wiki/concepts/motion-retargeting-pipeline.md)
- 前端：`docs/main.js` 中 `escapeMermaidForInnerHtml` 不再转义 `<br/>`，修复流程图节点多行标签被拼成一行的问题

## [2026-06-04] ingest | sources/papers/splitadapter_arxiv_2606_03297.md — SplitAdapter 负载感知人形搬箱因子化适配入库；wiki/entities/paper-splitadapter-load-aware-loco-manipulation.md

- 原始资料：[splitadapter_arxiv_2606_03297.md](sources/papers/splitadapter_arxiv_2606_03297.md)（<https://arxiv.org/abs/2606.03297>）；[splitadapter-github-io.md](sources/sites/splitadapter-github-io.md)（<https://splitadapter.github.io/>）
- 沉淀页面：[wiki/entities/paper-splitadapter-load-aware-loco-manipulation.md](wiki/entities/paper-splitadapter-load-aware-loco-manipulation.md)（含 Mermaid 因子化适配管线）
- 交叉更新：[wiki/tasks/loco-manipulation.md](wiki/tasks/loco-manipulation.md)、[wiki/concepts/sim2real.md](wiki/concepts/sim2real.md)、[wiki/entities/paper-amp-survey-15-physhsi.md](wiki/entities/paper-amp-survey-15-physhsi.md)、[sources/README.md](sources/README.md)

## [2026-06-04] ingest | sources/papers/htd_refine_arxiv_2605_26879.md — HTD-Refine（CVPR 2026）单目 HMR 高阶动力学后处理入库；wiki/entities/paper-htd-refine-monocular-hmr.md

- 原始资料：[htd_refine_arxiv_2605_26879.md](sources/papers/htd_refine_arxiv_2605_26879.md)（<https://arxiv.org/abs/2605.26879>）；[htd-refine-zju3dv-github-io.md](sources/sites/htd-refine-zju3dv-github-io.md)（<https://zju3dv.github.io/htd-refine/>）
- 沉淀页面：[wiki/entities/paper-htd-refine-monocular-hmr.md](wiki/entities/paper-htd-refine-monocular-hmr.md)
- 交叉更新：[wiki/concepts/motion-retargeting-pipeline.md](wiki/concepts/motion-retargeting-pipeline.md)、[wiki/concepts/whole-body-tracking-pipeline.md](wiki/concepts/whole-body-tracking-pipeline.md)、[wiki/methods/motion-retargeting-gmr.md](wiki/methods/motion-retargeting-gmr.md)

## [2026-06-04] ingest | sources/blogs/wechat_embodied_ai_lab_legs_vla_3dgs_loco_manip.md — Agent Reach 抓取 LEGS/3DGS 人形 VLA loco-manip 专题并沉淀论文实体

- 工具：已安装 [Panniantong/Agent-Reach](https://github.com/Panniantong/Agent-Reach) v1.4.0（editable 安装，修复 hatchling `force-include` 重复文件后 `pip install -e` + `wechat-article-for-ai`/Camoufox）
- 原始资料：[`sources/blogs/wechat_embodied_ai_lab_legs_vla_3dgs_loco_manip.md`](sources/blogs/wechat_embodied_ai_lab_legs_vla_3dgs_loco_manip.md)（<https://mp.weixin.qq.com/s/B1sYOPKg6TQwnNGs-_8NDw>）；落盘 [`sources/raw/wechat_legs_vla_3dgs_2026-06-04.md`](sources/raw/wechat_legs_vla_3dgs_2026-06-04.md)；论文 [`sources/papers/legs_arxiv_2606_01458.md`](sources/papers/legs_arxiv_2606_01458.md)；项目页 [`sources/sites/legsvla-github-io.md`](sources/sites/legsvla-github-io.md)
- 沉淀页面：[`wiki/entities/paper-legs-embodied-gaussian-splatting-vla.md`](wiki/entities/paper-legs-embodied-gaussian-splatting-vla.md)（含 Mermaid 管线总览）
- 交叉更新：[`wiki/tasks/loco-manipulation.md`](wiki/tasks/loco-manipulation.md)、[`wiki/methods/vla.md`](wiki/methods/vla.md)、[`wiki/methods/sonic-motion-tracking.md`](wiki/methods/sonic-motion-tracking.md)、[`wiki/entities/gs-playground.md`](wiki/entities/gs-playground.md)、[`sources/README.md`](sources/README.md)、[`sources/repos/panniantong_agent_reach.md`](sources/repos/panniantong_agent_reach.md)

## [2026-06-04] ingest | sources/sites/pupper-v3-documentation-readthedocs.md、sources/repos/pupperv3_monorepo.md、sources/courses/stanford_cs123_robotics_ai.md — Pupper v3 官方文档与 monorepo/CS123 入库；更新 wiki/entities/stanford-doggo-and-pupper.md

- 原始资料：[pupper-v3-documentation-readthedocs.md](sources/sites/pupper-v3-documentation-readthedocs.md)（<https://pupper-v3-documentation.readthedocs.io/en/latest/index.html>）；[pupperv3_monorepo.md](sources/repos/pupperv3_monorepo.md)（<https://github.com/Nate711/pupperv3-monorepo>）；[stanford_cs123_robotics_ai.md](sources/courses/stanford_cs123_robotics_ai.md)（<https://cs123-stanford.readthedocs.io/en/latest/>）
- 沉淀页面：[wiki/entities/stanford-doggo-and-pupper.md](wiki/entities/stanford-doggo-and-pupper.md)（补充 v3 规格、安全、RL/VLM、ROS 2 流程图；区分 v2/easy_quadruped lineage）
- 交叉更新：[wiki/entities/easy-quadruped.md](wiki/entities/easy-quadruped.md)、[wiki/entities/quadruped-robot.md](wiki/entities/quadruped-robot.md)、[sources/README.md](sources/README.md)

## [2026-06-04] ingest | sources/papers/humanoid_gpt_arxiv_2606_03985.md — Humanoid-GPT（CVPR 2026，2B 帧零样本 motion tracking）入库

- 原始资料：[humanoid_gpt_arxiv_2606_03985.md](sources/papers/humanoid_gpt_arxiv_2606_03985.md)（<https://arxiv.org/abs/2606.03985>）；[humanoid-gpt-qizekun-github-io.md](sources/sites/humanoid-gpt-qizekun-github-io.md)（<https://qizekun.github.io/Humanoid-GPT/>）；[humanoid_gpt_galaxy_general_robotics.md](sources/repos/humanoid_gpt_galaxy_general_robotics.md)（<https://github.com/GalaxyGeneralRobotics/Humanoid-GPT>）
- 沉淀页面：[wiki/entities/paper-humanoid-gpt.md](wiki/entities/paper-humanoid-gpt.md)
- 交叉更新：[wiki/methods/sonic-motion-tracking.md](wiki/methods/sonic-motion-tracking.md)、[wiki/queries/humanoid-motion-tracking-method-selection.md](wiki/queries/humanoid-motion-tracking-method-selection.md)

## [2026-06-04] ingest | sources/papers/extreme_parkour_arxiv_2309_14341.md、sources/repos/extreme-parkour.md、sources/sites/extreme-parkour-github-io.md — Extreme Parkour（ICRA 2024）入库；wiki/entities/extreme-parkour.md

- 原始资料：[extreme_parkour_arxiv_2309_14341.md](sources/papers/extreme_parkour_arxiv_2309_14341.md)（<https://arxiv.org/abs/2309.14341>）；[extreme-parkour.md](sources/repos/extreme-parkour.md)（<https://github.com/chengxuxin/extreme-parkour>）；[extreme-parkour-github-io.md](sources/sites/extreme-parkour-github-io.md)（<https://extreme-parkour.github.io/>）
- 沉淀页面：[wiki/entities/extreme-parkour.md](wiki/entities/extreme-parkour.md)
- 交叉更新：[wiki/tasks/locomotion.md](wiki/tasks/locomotion.md)、[wiki/concepts/privileged-training.md](wiki/concepts/privileged-training.md)、[roadmap/motion-control.md](roadmap/motion-control.md)、[sources/README.md](sources/README.md)

## [2026-06-04] structural | wiki/comparisons/sim2real-vs-real2sim-fine-tuning.md — 新建「Sim2Real 残差适配 vs Real2Sim 真机回放 vs 真机直接 RL 微调」对比页（V23 P2 安全微调知识链 3/3，专题收官）

- 新建 wiki：[sim2real-vs-real2sim-fine-tuning.md](wiki/comparisons/sim2real-vs-real2sim-fine-tuning.md)——把真机适配「最后一公里」拆成三策略：残差适配（冻结 $W_0$ + 低秩残差 + Recovery/Safety Filter 吸收残差，SLowRL）/ Real2Sim 真机回放（用真机数据反修仿真后回仿真重训，CRISP）/ 真机直接 RL 微调；给出 11 维核心对照表 + 数据流 Mermaid + 成本/安全/数据效率三维深读 + 三场景选型 + 5 类误判 + 决策矩阵，明确三者本质是「gap 在真机侧 / 仿真侧 / 真机侧端到端消化」的连续谱。
- 交叉更新：[safe-real-world-rl-fine-tuning.md](wiki/concepts/safe-real-world-rl-fine-tuning.md)、[sim2real-approaches.md](wiki/comparisons/sim2real-approaches.md) 补双向入链（消除孤儿页）。
- 进度：V23 P2「安全微调知识链 (+3)」3/3 完成，父项 `[~]`→`[x]`。
- 验证：`make lint` 仅余 2 条与本页无关的预存陈旧页警告（`generative-world-models` / `π0-policy`），`eval_search_quality` 37/37 通过；`make ci-preflight` 重生成派生文件，图谱 666 节点 / 4681 边 / 孤儿 0、`largest_community_ratio` 0.179、`community_quality_warning: false`。

## [2026-06-03] ingest | sources/papers/assistmimic_arxiv_2603_11346.md — AssistMimic（CVPR 2026 双人 assistive MARL tracking）入库

- 原始资料：[assistmimic_arxiv_2603_11346.md](sources/papers/assistmimic_arxiv_2603_11346.md)（<https://arxiv.org/abs/2603.11346>）；[yutoshibata07-assistmimic-github-io.md](sources/sites/yutoshibata07-assistmimic-github-io.md)（<https://yutoshibata07.github.io/AssistMimic/>）
- 沉淀页面：[wiki/entities/paper-assistmimic.md](wiki/entities/paper-assistmimic.md)
- 交叉更新：[wiki/methods/marl.md](wiki/methods/marl.md)、[wiki/entities/paper-bfm-22-phc.md](wiki/entities/paper-bfm-22-phc.md)、[wiki/concepts/whole-body-tracking-pipeline.md](wiki/concepts/whole-body-tracking-pipeline.md)

## [2026-06-03] ingest | sources/papers/dwm_arxiv_2512_17907.md、sources/repos/snuvclab_dwm.md — 补充 DWM 官方代码工程栈（CogVideoX-5B LoRA / WAN / 三元组数据目录）并更新 wiki/methods/dwm.md

- 原始资料：[`sources/papers/dwm_arxiv_2512_17907.md`](sources/papers/dwm_arxiv_2512_17907.md)（[arXiv:2512.17907](https://arxiv.org/abs/2512.17907)）；[`sources/sites/snuvclab-dwm-github-io.md`](sources/sites/snuvclab-dwm-github-io.md)；[`sources/repos/snuvclab_dwm.md`](sources/repos/snuvclab_dwm.md)（[snuvclab/dwm](https://github.com/snuvclab/dwm)，2026-04-03 代码发布）
- 沉淀页面：[`wiki/methods/dwm.md`](wiki/methods/dwm.md) — 增补「工程实现」节（CogVideoX-5B LoRA、VideoX-Fun 初始化、数据三元组目录、WAN 变体）
- 验证：`make ci-preflight` 通过

## [2026-06-03] ingest | sources/blogs/wechat_shenlan_world_models_15_open_source_2026.md — Agent Reach 抓取深蓝世界模型 15 项目并建三线图谱

- 工具：已安装 [Panniantong/Agent-Reach](https://github.com/Panniantong/Agent-Reach) v1.4.0（editable 安装 + `wechat-article-for-ai`/Camoufox）
- 原始资料：[`sources/blogs/wechat_shenlan_world_models_15_open_source_2026.md`](sources/blogs/wechat_shenlan_world_models_15_open_source_2026.md)（<https://mp.weixin.qq.com/s/KZT8sI4n7GvHWyM20wN3gg>）；落盘 [`sources/raw/wechat_world_models_15_2026-06-03.md`](sources/raw/wechat_world_models_15_2026-06-03.md)；参考资料 [`sources/papers/shenlan_world_models_15_reference_catalog.md`](sources/papers/shenlan_world_models_15_reference_catalog.md)
- 沉淀页面：[`wiki/overview/world-models-15-open-source-technology-map.md`](wiki/overview/world-models-15-open-source-technology-map.md)（**父节点**）；子节点 [`wiki/overview/world-models-route-01-cascade.md`](wiki/overview/world-models-route-01-cascade.md)、[`wiki/overview/world-models-route-02-joint.md`](wiki/overview/world-models-route-02-joint.md)、[`wiki/overview/world-models-route-03-virtual-sandbox.md`](wiki/overview/world-models-route-03-virtual-sandbox.md)；论文实体 `paper-shenlan-wm-01`…`03`、`05`…`15`（04→[`mimic-video`](wiki/methods/mimic-video.md)）
- 交叉更新：[`wiki/overview/robot-world-models-training-loop-taxonomy.md`](wiki/overview/robot-world-models-training-loop-taxonomy.md)、[`sources/README.md`](sources/README.md)、[`sources/repos/panniantong_agent_reach.md`](sources/repos/panniantong_agent_reach.md)
- 验证：`make ci-preflight` 通过

## [2026-06-03] structural | wiki/formalizations/safe-lora-update-projection.md — 新建「安全 LoRA 投影更新形式化」（V23 P2 安全微调知识链 2/3）

- 新建 wiki：[safe-lora-update-projection.md](wiki/formalizations/safe-lora-update-projection.md)（「冻结 $W_0$ + 低秩残差 $\frac{\alpha}{r}BA$ + 两层安全投影」统一形式：参数侧秩约束作隐式正则、动作侧 $\Pi_{\mathcal{S}}$ 分硬切换 Recovery 与连续 QP 安全壳两谱系，写成低秩子空间 CMDP；SLowRL 实例化表 + 全参 CMDP / 纯 QP 安全壳 / 生成式改写退化对照 + 评测口径）。
- 交叉更新：[safe-real-world-rl-fine-tuning.md](wiki/concepts/safe-real-world-rl-fine-tuning.md)、[SLowRL 实体](wiki/entities/paper-slowrl-safe-lora-locomotion-sim2real.md) 补双向入链（消除孤儿页）。
- 检索回归修复：[`scripts/search_wiki_core.py`](scripts/search_wiki_core.py) `_canonical_topic_boost` 由 1.4→1.7——safe-RL 专题扩张后「CBF 安全集 barrier」query 的 CBF 定义页被 clf-vs-cbf / safe-RL 系列页挤出 top5（main 既有回归），提权后 CBF 定义页回到 top5。
- 进度：V23 P2「安全微调知识链 (+3)」2/3（仍 `[~]`，余 sim2real-vs-real2sim-fine-tuning 对比页）。
- 验证：`make lint` 全绿（`eval_search_quality` 37/37）；`make ci-preflight` 重生成派生文件。

## [2026-06-02] ingest | sources/papers/shape_your_body_arxiv_2606_00702.md、sources/sites/shape-your-body-nico-bohlinger.md — Shape Your Body（VGDS 多具身价值梯度共设计）入库

- 原始资料：[shape_your_body_arxiv_2606_00702.md](sources/papers/shape_your_body_arxiv_2606_00702.md)（[PDF](https://www.ias.informatik.tu-darmstadt.de/uploads/Team/NicoBohlinger/shape_your_body.pdf)、[arXiv HTML](https://arxiv.org/html/2606.00702v1)）；[shape-your-body-nico-bohlinger.md](sources/sites/shape-your-body-nico-bohlinger.md)（[项目页](https://nico-bohlinger.github.io/shape-your-body/)）
- 沉淀页面：[wiki/entities/paper-shape-your-body-value-gradient-design.md](wiki/entities/paper-shape-your-body-value-gradient-design.md)
- 交叉更新：[cross-embodiment-transfer-strategy.md](wiki/queries/cross-embodiment-transfer-strategy.md)、[reinforcement-learning.md](wiki/methods/reinforcement-learning.md)、[foundation-policy.md](wiki/concepts/foundation-policy.md)
- 验证：`make ci-preflight` 通过

## [2026-06-02] structural | wiki/entities/paper-hrl-stack-22-perceptive_humanoid_parkour.md — 补全 motion matching / 蒸馏损失公式的 KaTeX 蓝框显示

- 修正页面：[`wiki/entities/paper-hrl-stack-22-perceptive_humanoid_parkour.md`](wiki/entities/paper-hrl-stack-22-perceptive_humanoid_parkour.md)（$\hat{x}_t$、$\arg\min_i \|\hat{x}_t - x_i\|^2$、$L = \lambda_{\mathrm{PPO}} L_{\mathrm{PPO}} + \lambda_D L_D$ 改用 `$...$` 包裹，detail 页可触发 `math-inline` 与 KaTeX）
- 验证：`make ci-preflight` 通过

## [2026-06-02] structural | wiki/concepts/safe-real-world-rl-fine-tuning.md — 新建「真机安全 RL 微调」概念页（V23 P2 安全微调知识链 1/3）

- 新建 wiki：[safe-real-world-rl-fine-tuning.md](wiki/concepts/safe-real-world-rl-fine-tuning.md)（残差视角 + 三路径详解：SLowRL 低秩残差 + Recovery/Safety Filter、Heracles 生成式兜底中间件、CBF/CLF 安全壳；5 维对比表 + 5 类常见误区）。
- 交叉更新：[sim2real.md](wiki/concepts/sim2real.md)（安全微调段落引导 + frontmatter related + 关联页面）、[safety-filter.md](wiki/concepts/safety-filter.md)、[SLowRL 实体](wiki/entities/paper-slowrl-safe-lora-locomotion-sim2real.md)、[Heracles 实体](wiki/entities/paper-heracles-humanoid-diffusion.md) 补入站边。
- 进度：V23 P2「安全微调知识链 (+3)」标记 1/3（`[~]` 进行中）。
- 验证：`make lint` 全绿（1 条 INFO 不阻塞）；`make ci-preflight` 重生成派生文件。

## [2026-06-02] ingest | sources/papers/mobilegym_arxiv_2605_26114.md、sources/repos/purewhiter_mobilegym.md、sources/sites/mobilegym-dev.md — MobileGym 入库

- 沉淀页面：[`wiki/entities/mobilegym.md`](wiki/entities/mobilegym.md)

## [2026-06-02] structural | schema/naming.md、scripts/generate_link_graph.py — 统一图谱社区名为「中文（English） 社区」并补全 override

- 规范：[`schema/naming.md`](schema/naming.md) 新增「图谱社区命名」；[`scripts/generate_link_graph.py`](scripts/generate_link_graph.py) 增加 `COMMUNITY_HUB_NAME_RE` 校验与 WARNING
- 修正社区名：规模化运动跟踪（SONIC）、人形硬件技术地图（Humanoid Hardware 101）、机器人学习（Robot Learning）、行为基础模型技术地图（BFM）
- 测试：[`tests/test_community_naming.py`](tests/test_community_naming.py)
- 验证：`make ci-preflight` 通过
## [2026-06-02] structural | wiki/overview humanoid-hardware-101-* 与 humanoid-actuator-102-* — 首页最新知识节点补登子 hub

- 父节点：[`wiki/overview/humanoid-hardware-101-technology-map.md`](wiki/overview/humanoid-hardware-101-technology-map.md)、[`wiki/overview/humanoid-actuator-102-technology-map.md`](wiki/overview/humanoid-actuator-102-technology-map.md)
- Hardware 101 子 hub：[`wiki/overview/humanoid-hardware-101-chassis-materials.md`](wiki/overview/humanoid-hardware-101-chassis-materials.md)、[`wiki/overview/humanoid-hardware-101-actuation-sensing-chain.md`](wiki/overview/humanoid-hardware-101-actuation-sensing-chain.md)、[`wiki/overview/humanoid-hardware-101-linear-transmission-bearings.md`](wiki/overview/humanoid-hardware-101-linear-transmission-bearings.md)、[`wiki/overview/humanoid-hardware-101-integrated-actuators.md`](wiki/overview/humanoid-hardware-101-integrated-actuators.md)、[`wiki/overview/humanoid-hardware-101-power-compute-electronics.md`](wiki/overview/humanoid-hardware-101-power-compute-electronics.md)、[`wiki/overview/humanoid-hardware-101-sensing-end-effectors.md`](wiki/overview/humanoid-hardware-101-sensing-end-effectors.md)、[`wiki/overview/humanoid-hardware-101-supply-chain-economics.md`](wiki/overview/humanoid-hardware-101-supply-chain-economics.md)
- Actuator 102 子 hub：[`wiki/overview/humanoid-actuator-102-load-and-mass-spiral.md`](wiki/overview/humanoid-actuator-102-load-and-mass-spiral.md)、[`wiki/overview/humanoid-actuator-102-split-architecture.md`](wiki/overview/humanoid-actuator-102-split-architecture.md)、[`wiki/overview/humanoid-actuator-102-gear-reflected-inertia.md`](wiki/overview/humanoid-actuator-102-gear-reflected-inertia.md)、[`wiki/overview/humanoid-actuator-102-thermal-and-control.md`](wiki/overview/humanoid-actuator-102-thermal-and-control.md)、[`wiki/overview/humanoid-actuator-102-compliance-sensing.md`](wiki/overview/humanoid-actuator-102-compliance-sensing.md)、[`wiki/overview/humanoid-actuator-102-industrial-actuator-trap.md`](wiki/overview/humanoid-actuator-102-industrial-actuator-trap.md)、[`wiki/overview/humanoid-actuator-102-decision-species.md`](wiki/overview/humanoid-actuator-102-decision-species.md)、[`wiki/overview/humanoid-actuator-102-future-artificial-muscle.md`](wiki/overview/humanoid-actuator-102-future-artificial-muscle.md)
- 验证：`make graph` 后 `exports/home-stats.json` 的 `latest_wiki_nodes` 含上述 17 个 hub。

## [2026-06-02] ingest | sources/blogs/wechat_human_five_humanoid_actuator_102.md、sources/papers/humanoid_actuator_102_reference_catalog.md — Agent Reach 抓取 Humanoid 执行器 102 并建八章图谱

- 工具：[Agent Reach](https://github.com/Panniantong/Agent-Reach) v1.4.0 + `wechat-article-for-ai`（Camoufox）
- 原始资料：[`sources/blogs/wechat_human_five_humanoid_actuator_102.md`](sources/blogs/wechat_human_five_humanoid_actuator_102.md)（<https://mp.weixin.qq.com/s/zinp6ulTorzfqmCR_HaI5A>）；落盘 [`sources/raw/wechat_humanoid_actuator_102_2026-06-02.md`](sources/raw/wechat_humanoid_actuator_102_2026-06-02.md)；参考资料 [`sources/papers/humanoid_actuator_102_reference_catalog.md`](sources/papers/humanoid_actuator_102_reference_catalog.md)
- 沉淀页面：[`wiki/overview/humanoid-actuator-102-technology-map.md`](wiki/overview/humanoid-actuator-102-technology-map.md)（父节点）；子节点 [`wiki/overview/humanoid-actuator-102-load-and-mass-spiral.md`](wiki/overview/humanoid-actuator-102-load-and-mass-spiral.md)、[`wiki/overview/humanoid-actuator-102-split-architecture.md`](wiki/overview/humanoid-actuator-102-split-architecture.md)、[`wiki/overview/humanoid-actuator-102-gear-reflected-inertia.md`](wiki/overview/humanoid-actuator-102-gear-reflected-inertia.md)、[`wiki/overview/humanoid-actuator-102-thermal-and-control.md`](wiki/overview/humanoid-actuator-102-thermal-and-control.md)、[`wiki/overview/humanoid-actuator-102-compliance-sensing.md`](wiki/overview/humanoid-actuator-102-compliance-sensing.md)、[`wiki/overview/humanoid-actuator-102-industrial-actuator-trap.md`](wiki/overview/humanoid-actuator-102-industrial-actuator-trap.md)、[`wiki/overview/humanoid-actuator-102-decision-species.md`](wiki/overview/humanoid-actuator-102-decision-species.md)、[`wiki/overview/humanoid-actuator-102-future-artificial-muscle.md`](wiki/overview/humanoid-actuator-102-future-artificial-muscle.md)
- 交叉更新：[`wiki/overview/humanoid-hardware-101-technology-map.md`](wiki/overview/humanoid-hardware-101-technology-map.md)、[`wiki/overview/humanoid-hardware-101-integrated-actuators.md`](wiki/overview/humanoid-hardware-101-integrated-actuators.md)、[`sources/README.md`](sources/README.md)

## [2026-06-02] ingest | sources/repos/nvidia_isaac_teleop.md — Isaac Teleop 入库；新建 wiki/entities/isaac-teleop.md；交叉 wiki/entities/isaac-lab.md、wiki/tasks/teleoperation.md

- 原始资料：[nvidia_isaac_teleop.md](sources/repos/nvidia_isaac_teleop.md)（[GitHub](https://github.com/NVIDIA/IsaacTeleop)、[官方文档](https://nvidia.github.io/IsaacTeleop/main/index.html)、[Isaac Lab 功能页](https://isaac-sim.github.io/IsaacLab/main/source/features/isaac_teleop.html)）
- 沉淀页面：[wiki/entities/isaac-teleop.md](wiki/entities/isaac-teleop.md)；交叉 [isaac-lab.md](wiki/entities/isaac-lab.md)、[teleoperation.md](wiki/tasks/teleoperation.md)
- 验证：`make ci-preflight` 通过

## [2026-06-01] structural | wiki/concepts/motion-retargeting.md、sim2real.md — 跨具身专题交叉补强（V23 P1 收官）

- 变更：在 [motion-retargeting.md](wiki/concepts/motion-retargeting.md) 新增「三段流水线衔接：重定向产物 → WBT 训练数据 → 跨具身策略蒸馏」小节（映射/训练/迁移三段表格 + 衔接段落），关联页面补 [WBT pipeline](wiki/concepts/whole-body-tracking-pipeline.md) / [跨具身迁移选型](wiki/queries/cross-embodiment-transfer-strategy.md) / [SONIC 四方对比](wiki/comparisons/sonic-vs-beyondmimic-vs-sdamp-vs-heracles.md) 三条出边。
- 变更：在 [sim2real.md](wiki/concepts/sim2real.md) 新增「在『映射 → 训练 → 迁移』三段流水线中的位置」小节（点明 Sim2Real 横切训练与迁移两段），frontmatter `related` 与关联页面同步补 motion-retargeting / WBT pipeline / 跨具身迁移 / SONIC 对比。
- 进度：V23 P1「跨具身专题交叉补强」标记为 `[x]`，至此 P1 全部完成。
- 验证：`make lint` 全绿；`make ci-preflight` 重生成派生文件。

## [2026-06-01] ingest | sources/papers/kalman_filter_ekf_primary_refs.md、sources/papers/lqr_ilqr_primary_refs.md — KF/EKF/LQR/iLQR 一手资料入库并新建 KF 形式化页

- 原始资料：[kalman_filter_ekf_primary_refs.md](sources/papers/kalman_filter_ekf_primary_refs.md)（Kalman 1960/61、Gelb 1974、Simon 2006 等）、[lqr_ilqr_primary_refs.md](sources/papers/lqr_ilqr_primary_refs.md)（Bryson & Ho 1975、Li & Todorov 2004、Tassa 2012/14 等）；课程：[welch_bishop_kalman_filter.md](sources/courses/welch_bishop_kalman_filter.md)、[mit_underactuated_kalman_lqr.md](sources/courses/mit_underactuated_kalman_lqr.md)
- 沉淀页面：[wiki/formalizations/kalman-filter.md](wiki/formalizations/kalman-filter.md)（新建）；交叉更新 [ekf.md](wiki/formalizations/ekf.md)、[lqr.md](wiki/formalizations/lqr.md)、[lqr-ilqr.md](wiki/methods/lqr-ilqr.md)、[state-estimation.md](wiki/concepts/state-estimation.md)
- 验证：`make ci-preflight` 通过

## [2026-06-01] ingest | sources/blogs/wechat_embodied_ai_lab_ego_9_papers_survey.md — Agent Reach 抓取 Ego 9 篇专题并建四类子系统图谱与 9 论文节点

- 工具：已安装 [Panniantong/Agent-Reach](https://github.com/Panniantong/Agent-Reach) v1.4.0（`pip install git+https://github.com/Panniantong/Agent-Reach.git` + `agent-reach install --channels=wechat`）；微信正文经 `wechat-article-for-ai`（Camoufox）
- 原始资料：[`sources/blogs/wechat_embodied_ai_lab_ego_9_papers_survey.md`](sources/blogs/wechat_embodied_ai_lab_ego_9_papers_survey.md)（<https://mp.weixin.qq.com/s/4JQ1xa-cJ7J1ep_e4txNnA>）；落盘 [`sources/raw/wechat_ego_9_papers_2026-06-01.md`](sources/raw/wechat_ego_9_papers_2026-06-01.md)
- 沉淀页面：[`wiki/overview/ego-9-papers-technology-map.md`](wiki/overview/ego-9-papers-technology-map.md)（父节点 + Mermaid）；子节点 [`ego-category-01-data-collection`](wiki/overview/ego-category-01-data-collection.md)、[`ego-category-02-human-to-robot`](wiki/overview/ego-category-02-human-to-robot.md)、[`ego-category-03-world-models`](wiki/overview/ego-category-03-world-models.md)、[`ego-category-04-ego-exo-fusion`](wiki/overview/ego-category-04-ego-exo-fusion.md)；论文实体 `paper-ego-01`…`05`、`08`、`09`（06→[`paper-hrl-stack-33`](wiki/entities/paper-hrl-stack-33-ego_vision_world_model_for_humanoid.md)、07→[`paper-wem`](wiki/entities/paper-wem-world-ego-modeling.md)）
- 交叉更新：[`humanoid-rl-motion-control-body-system-stack.md`](wiki/overview/humanoid-rl-motion-control-body-system-stack.md)、[`robot-world-models-training-loop-taxonomy.md`](wiki/overview/robot-world-models-training-loop-taxonomy.md)、[`sources/repos/panniantong_agent_reach.md`](sources/repos/panniantong_agent_reach.md)、[`sources/README.md`](sources/README.md)

## [2026-06-01] fix(ux) | docs/style.css — 撤销路线页首屏 360px 左缩进，与面包屑/正文左缘对齐

- 变更：去掉 #470 误加的 `margin-inline-start: 360px`；保留首屏列表满宽与 `#roadmapSummary` 使用 `<div>`。
- 验证：`make ci-preflight` 通过。

## [2026-06-01] fix(wiki) | wiki/overview/humanoid-hardware-101-technology-map.md — 修复七类子系统 Mermaid（子图直连改节点边、去 click、htmlLabels 换行）

- 根因：子图 `G1 --> G4` 直连在 `securityLevel: strict` 下易解析失败；`click` 指令被站点 Mermaid 配置禁用；标签内 `/` 与 `·` 增加歧义。
- 变更：改为 `CH/M/LS --> ACT --> ROB` 节点级边；标签用 `<br/>` 分行；分类入口保留下方表格链接。
- 关联页面：`wiki/overview/humanoid-hardware-101-technology-map.md`
- 验证：`make ci-preflight` 通过。

## [2026-06-01] ingest | sources/blogs/wechat_human_five_humanoid_hardware_101.md — Agent Reach 抓取 human five《Humanoid Hardware 入门 101》并建七类子系统图谱

- 工具：已安装 [Panniantong/Agent-Reach](https://github.com/Panniantong/Agent-Reach) v1.4.0（`pip install git+https://github.com/Panniantong/Agent-Reach.git` + `agent-reach install --channels=wechat`）；微信正文经 `~/.agent-reach/tools/wechat-article-for-ai`（Camoufox）
- 原始资料：[`sources/blogs/wechat_human_five_humanoid_hardware_101.md`](sources/blogs/wechat_human_five_humanoid_hardware_101.md)（<https://mp.weixin.qq.com/s/10hYwFzC1EuCypFVzC6QGQ>）；落盘 [`sources/raw/wechat_humanoid_hardware_101_2026-06-01.md`](sources/raw/wechat_humanoid_hardware_101_2026-06-01.md)
- 沉淀页面：[`wiki/overview/humanoid-hardware-101-technology-map.md`](wiki/overview/humanoid-hardware-101-technology-map.md)（父节点 + Mermaid）；子节点 [`humanoid-hardware-101-chassis-materials`](wiki/overview/humanoid-hardware-101-chassis-materials.md)、[`actuation-sensing-chain`](wiki/overview/humanoid-hardware-101-actuation-sensing-chain.md)、[`linear-transmission-bearings`](wiki/overview/humanoid-hardware-101-linear-transmission-bearings.md)、[`integrated-actuators`](wiki/overview/humanoid-hardware-101-integrated-actuators.md)、[`power-compute-electronics`](wiki/overview/humanoid-hardware-101-power-compute-electronics.md)、[`sensing-end-effectors`](wiki/overview/humanoid-hardware-101-sensing-end-effectors.md)、[`supply-chain-economics`](wiki/overview/humanoid-hardware-101-supply-chain-economics.md)
- 交叉更新：[`wiki/queries/humanoid-hardware-selection.md`](wiki/queries/humanoid-hardware-selection.md)、[`wiki/entities/open-source-humanoid-hardware.md`](wiki/entities/open-source-humanoid-hardware.md)、[`sources/repos/panniantong_agent_reach.md`](sources/repos/panniantong_agent_reach.md)、[`sources/README.md`](sources/README.md)

## [2026-06-01] fix(ux) | docs/main.js — 修复运动控制路线页 L1–L7 章节被 L0 自测块吞没

- 根因：`<details class="selftest-answers">` 内的 ` ```mermaid ` 会提前 `flushHtmlBlock()`，导致 `<details>` 未闭合，后续 L1–L7 的 h2 落入错误 DOM，`wrapRoadmapCollapsibleMajorHeadings` 只显示到 L0。
- 变更：HTML 块解析中保留围栏行直至 `</details>`；`flushHtmlBlock` 时将块内 mermaid 转为 `.mermaid`。
- 验证：`roadmap.html?id=roadmap-motion-control` 顶层折叠章节含 L0–L7；`make ci-preflight` 通过。

## [2026-05-31] query | wiki/queries/cross-embodiment-transfer-strategy.md — 跨具身策略迁移选型指南（V23 P1 WBT 知识链收官）

- 新建 wiki：[cross-embodiment-transfer-strategy.md](wiki/queries/cross-embodiment-transfer-strategy.md)（单具身重训 + 重定向 / Any2Any 高效后训练 / 多具身联合训练三路径：9 维「算力 × 数据 × 泛化」对照表 + Mermaid 决策树 + 7 类典型故障模式 + 4 条推荐组合 pipeline；定位为 [WBT pipeline](wiki/concepts/whole-body-tracking-pipeline.md) 阶段 5 选型横切面）。
- 交叉更新：[whole-body-tracking-pipeline.md](wiki/concepts/whole-body-tracking-pipeline.md) 阶段 5 与关联页面、[humanoid-motion-tracking-method-selection.md](wiki/queries/humanoid-motion-tracking-method-selection.md) §6 加入站链接。
- 进度：V23 P1「WBT 知识链 (+3)」三页（pipeline / 四方法对比 / 跨具身 Query）全部完成，标记为 `[x]`。
- 验证：`make lint` 全绿（孤儿页消解）、`eval_search_quality` 37/37 通过。

## [2026-05-31] ingest | sources/papers/php_parkour_arxiv_2602_15827.md、sources/sites/php-parkour-github-io.md、sources/papers/omniretarget_arxiv_2509_26633.md — PHP/RSS2026 与 OmniRetarget 深读；wiki/entities/paper-hrl-stack-22-perceptive_humanoid_parkour.md、wiki/entities/paper-hrl-stack-03-omniretarget.md

## [2026-05-31] fix(ux) | docs/main.js — 路线页自测参考答案等 HTML 块内公式补蓝色边框包裹

- 根因：`<details class="selftest-answers">` 等原样 HTML 块绕过 `renderMathBlocks`，KaTeX 能渲染但缺少 `math-inline` / `math-block` 与 detail 一致的蓝框样式。
- 变更：新增 `applyMathBlocksInHtmlFragment`，在 `flushHtmlBlock` 中对 HTML 片段文本节点补公式包裹。
- 验证：`roadmap.html?id=roadmap-motion-control` 展开参考答案后行内公式带蓝框；`make ci-preflight` 通过。

## [2026-05-31] structural | roadmap/motion-control.md — 各 L 层补充英文缩写速查表（缩写 / 全称 / 简要说明）

- 变更：[roadmap/motion-control.md](roadmap/motion-control.md) 在 L−1～L7 及 L4.1–L4.4、L5.1–L5.3、L7.1–L7.5 增加统一格式「英文缩写速查」表；L−1 原「必备术语速查」改为三列英文全称版。
- 关联页面：`roadmap/motion-control.md`
- 验证：`make ci-preflight` 通过。

## [2026-05-31] structural | references/repos/simulation.md、tech-map/modules/system/simulation.md — 区分「仿真平台索引」与「技术栈模块」详情页，消除同名 Simulation 混淆

- 问题：`detail.html?id=reference-repos-simulation` 与 `tech-node-system-simulation` 标题同为 Simulation、正文量差异大，易被误判为重复页。
- 变更：reference 页改名为「仿真平台与工具链」并链回 tech-map；tech-map 页改名为「仿真（系统集成层）」、补充模块定位与 wiki 互链；更新 `references/repos/README.md` 入口文案。
- 验证：`make ci-preflight` 通过。

## [2026-05-31] ingest | sources/papers/tau0_wm_tech_report.md、sources/sites/tau0-wm-agibot-finch.md、sources/repos/sii_research_tau_0_wm.md — τ₀-WM 统一视频–动作世界模型入库

- 原始资料：[tau0_wm_tech_report.md](sources/papers/tau0_wm_tech_report.md)（<https://finch-static.agibot.com/VAM/blog/tau_0_wm.pdf>）、[tau0-wm-agibot-finch.md](sources/sites/tau0-wm-agibot-finch.md)（<https://finch.agibot.com/research/tau0-wm>）、[sii_research_tau_0_wm.md](sources/repos/sii_research_tau_0_wm.md)（<https://github.com/sii-research/tau-0-wm>、<https://huggingface.co/sii-research/tau-0-wm>）
- 新建 wiki：[tau0-world-model.md](wiki/entities/tau0-world-model.md)（5B VAM、异构 ~27.3k h 掩码预训练、动作条件仿真 + 测试时 propose–evaluate–revise）
- 交叉更新：[world-action-models.md](wiki/concepts/world-action-models.md)、[generative-world-models.md](wiki/methods/generative-world-models.md)、[mimic-video.md](wiki/methods/mimic-video.md)、[ge-sim-2.md](wiki/entities/ge-sim-2.md)、[robot-world-models-training-loop-taxonomy.md](wiki/overview/robot-world-models-training-loop-taxonomy.md)、[manipulation.md](wiki/tasks/manipulation.md)

## [2026-05-31] fix(ux) | docs/main.js — 路线页 Mermaid 在章节折叠后补渲染

- 根因：L4 方法链等 Mermaid 在 `wrapRoadmapCollapsibleMajorHeadings` 之前渲染，部分环境下折叠 DOM 重组后流程图空白或单行。
- 变更：路线正文在折叠包装完成后再 `renderDetailMermaid`；`bindRoadmapSectionMermaidRerender` 在展开章节时对未出 SVG 的图补跑；保留 `htmlLabels: true` 以支持 `<br/>` / `<b>` 多行标签。
- 验证：`roadmap.html?id=roadmap-motion-control` L4 方法链四节点多行渲染；`make ci-preflight` 通过。

## [2026-05-31] fix(ux) | docs/main.js — 修复详情页链接标签内 `*斜体*` / `**粗体**` 未渲染

- 根因：`renderInlineMarkdown` 在链接 token 化时对 label 仅 `escapeHtml`，强调语法在还原后不会再次处理。
- 变更：新增 `renderLinkLabel`，在 `<a>` 内应用与正文一致的 inline 样式；影响含 `*…*` 书名的外链（如 linear-algebra-curriculum）。
- 验证：`detail.html?id=entity-linear-algebra-curriculum` 中 Axler 链接呈现 `<em>`。

## [2026-05-31] structural | roadmap/motion-control.md、docs/main.js — 修复 L4 方法链 Mermaid 换行与加粗渲染

- 根因：`flowchart.htmlLabels: false` 时节点内 `<br/>` / `<em>` 被当作纯文本，四段 L4 标签挤成单行。
- 变更：`docs/main.js` 启用 `htmlLabels: true`；`roadmap/motion-control.md` L4.0 流程图标题改用 `<b>`，去掉易干扰解析的弯引号。
- 验证：本地 `roadmap.html?id=roadmap-motion-control` 中 L4 图 `foreignObject` 多行标签正常；`make ci-preflight` 通过。

## [2026-05-31] ingest | sources/courses/gatech_interactive_linear_algebra.md、sources/courses/axler_linear_algebra_done_right_4e.md、sources/courses/linear_algebra_teaching_materials_curated.md — 线性代数优秀教学材料入库；L0 策展页与运动控制路线互链

- 原始资料：[gatech_interactive_linear_algebra.md](sources/courses/gatech_interactive_linear_algebra.md)（<https://textbooks.math.gatech.edu/ila/>）、[axler_linear_algebra_done_right_4e.md](sources/courses/axler_linear_algebra_done_right_4e.md)（<https://linear.axler.net/LADR4e.pdf>）、[linear_algebra_teaching_materials_curated.md](sources/courses/linear_algebra_teaching_materials_curated.md)（3Blue1Brown、Strang 18.06 等策展）
- 新建 wiki：[linear-algebra-curriculum.md](wiki/entities/linear-algebra-curriculum.md)（机器人 L0 章节地图 + 2–4 周学习路径）
- 交叉更新：[roadmap/motion-control.md](roadmap/motion-control.md) L0 推荐读什么/入口、[modern-robotics-book.md](wiki/entities/modern-robotics-book.md)、[tech-map/modules/math/linear-algebra.md](tech-map/modules/math/linear-algebra.md)

## [2026-05-31] ingest | sources/papers/unilab_arxiv_2605_30313.md、sources/repos/unilab.md、sources/sites/unilabsim-project.md — UniLab 异构 CPU 仿真 / GPU 学习训练系统入库

- 原始资料：[unilab_arxiv_2605_30313.md](sources/papers/unilab_arxiv_2605_30313.md)（<https://arxiv.org/abs/2605.30313>）、[unilab.md](sources/repos/unilab.md)（<https://github.com/unilabsim/UniLab>）、[unilabsim-project.md](sources/sites/unilabsim-project.md)（<https://unilabsim.github.io>）
- 新建 wiki：[unilab.md](wiki/entities/unilab.md)（统一 runtime、MuJoCoUni/MotrixSim 双后端、3–10× 端到端墙钟、跨平台训练）
- 交叉更新：[isaac-gym-isaac-lab.md](wiki/entities/isaac-gym-isaac-lab.md)、[motrix.md](wiki/entities/motrix.md)、[simulator-selection-guide.md](wiki/queries/simulator-selection-guide.md)、[mujoco-vs-isaac-lab.md](wiki/comparisons/mujoco-vs-isaac-lab.md)

## [2026-05-31] structural | wiki/comparisons/ctde-vs-decentralized-marl.md、wiki/queries/humanoid-motion-tracking-method-selection.md、wiki/methods/marl.md — 为两个高频引用 methods 补 queries/comparisons 落地，消除 lint 信息型预警

- 背景：`make lint` 报两条信息型预警——[egm-efficient-general-mimic.md](wiki/methods/egm-efficient-general-mimic.md)、[marl.md](wiki/methods/marl.md) 被多页引用却无 queries/comparisons 落地。
- 新建 wiki：[ctde-vs-decentralized-marl.md](wiki/comparisons/ctde-vs-decentralized-marl.md)（CTDE 集中式训练分布式执行 vs 完全去中心化选型对比）。
- 交叉更新：[humanoid-motion-tracking-method-selection.md](wiki/queries/humanoid-motion-tracking-method-selection.md) 通用 tracker 段补 [EGM](wiki/methods/egm-efficient-general-mimic.md)；[marl.md](wiki/methods/marl.md) 关联页面回链新对比页。
- 验证：`make lint` 全绿，两条信息型预警归零；搜索回归 37/37。

## [2026-05-30] checklist-v23 | wiki/comparisons/sonic-vs-beyondmimic-vs-sdamp-vs-heracles.md — V23 P1「WBT 知识链」第二页落地

- 变更：新建 [wiki/comparisons/sonic-vs-beyondmimic-vs-sdamp-vs-heracles.md](wiki/comparisons/sonic-vs-beyondmimic-vs-sdamp-vs-heracles.md)，把四条主流 WBT 「策略学习」路线（**SONIC 规模化预训练 / BeyondMimic 精准物理 + 失败采样 / SD-AMP 状态门控双判别器 / Heracles 状态条件扩散中间件**）放进同一张 13 维度对照表 + 数据流 Mermaid + 四类适用场景 + 6 类常见误判 + 决策矩阵；显式声明四者按「OOD 修补位置」（数据池 / 训练物理 / 训练判别器 / 部署参考层）构成连续谱而非互斥选择，工程系统常**串联组合**。
- 链接：frontmatter `related` 拉入 [WBT pipeline](wiki/concepts/whole-body-tracking-pipeline.md)、[motion-retargeting-pipeline.md](wiki/concepts/motion-retargeting-pipeline.md)、[whole-body-control.md](wiki/concepts/whole-body-control.md)、[behavior-foundation-model.md](wiki/concepts/behavior-foundation-model.md)、[SONIC](wiki/methods/sonic-motion-tracking.md)、[BeyondMimic](wiki/methods/beyondmimic.md)、[SD-AMP](wiki/entities/paper-unified-walk-run-recovery-sdamp.md)、[Heracles](wiki/entities/paper-heracles-humanoid-diffusion.md)、[Any2Any](wiki/entities/paper-any2any-cross-embodiment-wbt.md)、[AMP](wiki/methods/amp-reward.md)、[DeepMimic](wiki/methods/deepmimic.md)、[扩散运动生成](wiki/methods/diffusion-motion-generation.md)、[motion tracking 选型 query](wiki/queries/humanoid-motion-tracking-method-selection.md)、[balance recovery](wiki/tasks/balance-recovery.md)；`sources` 链入 9 条原始资料（SONIC / BeyondMimic / SD-AMP / Heracles 的 arXiv 摘要 + HRL stack 策展条目 + 项目页 + 代码仓 + sites）。
- 验证：`python3 scripts/lint_wiki.py` — 仅余 2 条与本页无关的预存 INFO（`egm-efficient-general-mimic` / `marl` 高频引用缺 queries/comparisons 落地），新页面在 type / summary / sources / 关联出边等所有阻塞检查项上均 0 错误。
- 关联清单：[`docs/checklists/tech-stack-next-phase-checklist-v23.md`](docs/checklists/tech-stack-next-phase-checklist-v23.md) P1「WBT 知识链 (+3)」中的第二条 `sonic-vs-beyondmimic-vs-sdamp-vs-heracles.md` 打勾；剩 1 页 `queries/cross-embodiment-transfer-strategy.md` 待落地。

## [2026-05-30] ingest | sources/papers/sam_3d_body_arxiv_2602_15989.md、sources/repos/sam-3d-body.md、sources/repos/sam3dbody-cpp.md — SAM 3D Body（MHR 全身 HMR）与 SAM3DBody-cpp 工程运行时入库

- 原始资料：[sam_3d_body_arxiv_2602_15989.md](sources/papers/sam_3d_body_arxiv_2602_15989.md)（<https://arxiv.org/abs/2602.15989>）、[sam-3d-body.md](sources/repos/sam-3d-body.md)（<https://github.com/facebookresearch/sam-3d-body>）、[sam3dbody-cpp.md](sources/repos/sam3dbody-cpp.md)（<https://github.com/AmmarkoV/SAM3DBody-cpp>）
- 新建 wiki：[sam-3d-body.md](wiki/entities/sam-3d-body.md)（可提示单图 MHR、官方 checkpoint/数据集）、[sam3dbody-cpp.md](wiki/entities/sam3dbody-cpp.md)（ONNX+ggml、BVH/CSV、离线五遍精修）
- 交叉更新：[motion-retargeting-pipeline.md](wiki/concepts/motion-retargeting-pipeline.md)、[wilor.md](wiki/methods/wilor.md)、[genmo.md](wiki/methods/genmo.md)

## [2026-05-30] ingest | sources/papers/ge_sim_2_arxiv_2605_27491.md、sources/repos/ge_sim_v2.md、sources/sites/ge-sim-v2-project.md — GE-Sim 2.0 闭环视频世界模拟器入库；wiki/entities/ge-sim-2.md

- 原始资料：[ge_sim_2_arxiv_2605_27491.md](sources/papers/ge_sim_2_arxiv_2605_27491.md)（<https://arxiv.org/abs/2605.27491>）、[ge_sim_v2.md](sources/repos/ge_sim_v2.md)（<https://github.com/AgibotTech/GE-Sim-V2>）、[ge-sim-v2-project.md](sources/sites/ge-sim-v2-project.md)（<https://ge-sim-v2.github.io/>）
- 新建 wiki：[ge-sim-2.md](wiki/entities/ge-sim-2.md)（视觉+本体双专家、World Judge、加速 rollout；WorldArena 2B 榜首）
- 交叉更新：[generative-world-models.md](wiki/methods/generative-world-models.md)、[video-as-simulation.md](wiki/concepts/video-as-simulation.md)、[robot-world-models-training-loop-taxonomy.md](wiki/overview/robot-world-models-training-loop-taxonomy.md)、[ewmbench.md](wiki/entities/ewmbench.md)

## [2026-05-30] structural | wiki/concepts/sim2real.md — 对齐 Sim2Real 工程流程 Mermaid：训练前准备、训练期 DR/RMA、SAGE/中间件/real-to-sim 与可选 Real2Sim 上游

- 变更：更新 [sim2real.md](wiki/concepts/sim2real.md)「Sim2Real 工程流程总览」Mermaid，与 §7 SOP、[sim2real-checklist](wiki/queries/sim2real-checklist.md)、[sim2real-gap-reduction](wiki/queries/sim2real-gap-reduction.md) 时序一致。
- 关联页面：`wiki/concepts/sim2real.md`

## [2026-05-30] ingest | sources/repos/qwen-vla.md、sources/papers/qwenvla_arxiv_2605_30280.md — Qwen-VLA 统一 VLA 通才入库；wiki/entities/qwen-vla.md

- 原始资料：[qwen-vla.md](sources/repos/qwen-vla.md)（<https://github.com/QwenLM/Qwen-VLA>）、[qwenvla_arxiv_2605_30280.md](sources/papers/qwenvla_arxiv_2605_30280.md)（<https://arxiv.org/abs/2605.30280>）
- 新建 wiki：[qwen-vla.md](wiki/entities/qwen-vla.md)（Qwen3.5-4B + 1.15B DiT flow；操作–导航–轨迹统一；embodiment prompt；渐进训练 SFT/RL）
- 交叉更新：[vla.md](wiki/methods/vla.md)、[star-vla.md](wiki/methods/star-vla.md)、[xiaomi-robotics-0.md](wiki/entities/xiaomi-robotics-0.md)、[vla-open-source-repro-landscape-2025.md](wiki/overview/vla-open-source-repro-landscape-2025.md)

## [2026-05-30] ingest | sources/papers/dreamwaq_plus_arxiv_2409_19709.md、sources/sites/dreamwaqpp-github-io.md — DreamWaQ++（arXiv:2409.19709 / T-RO 2026）入库；wiki/entities/dreamwaq-plus.md

- 原始资料：[dreamwaq_plus_arxiv_2409_19709.md](sources/papers/dreamwaq_plus_arxiv_2409_19709.md)（<https://arxiv.org/abs/2409.19709>）、[dreamwaqpp-github-io.md](sources/sites/dreamwaqpp-github-io.md)（<https://dreamwaqpp.github.io/>）
- 新建 wiki：[dreamwaq-plus.md](wiki/entities/dreamwaq-plus.md)（多模态点云+本体、分层外感知记忆、非对称 AC+PPO、楼梯/陡坡/OOD）
- 交叉更新：[privileged-training.md](wiki/concepts/privileged-training.md)、[terrain-adaptation.md](wiki/concepts/terrain-adaptation.md)、[locomotion.md](wiki/tasks/locomotion.md)、[privileged_training.md](sources/papers/privileged_training.md)、[motion-control.md](roadmap/motion-control.md)

## [2026-05-30] ingest | sources/papers/schedulestream_arxiv_2511_04758.md、sources/repos/nvlabs-schedulestream.md — ScheduleStream/TAMPAS 入库；wiki/entities/schedulestream.md

## [2026-05-30] ingest | sources/papers/gamma_world_arxiv_2605_28816.md — Gamma-World 多智能体世界模型入库；wiki/entities/paper-gamma-world-multi-agent.md

## [2026-05-30] ingest | sources/papers/physx_omni_arxiv_2605_21572.md — PhysX-Omni/PhysXVerse/PhysX-Bench 入库；wiki/entities/physx-omni.md

## [2026-05-29] checklist-v23 | wiki/concepts/whole-body-tracking-pipeline.md — V23 P1「WBT 知识链」首页落地

- 变更：新建 [wiki/concepts/whole-body-tracking-pipeline.md](wiki/concepts/whole-body-tracking-pipeline.md)，把 Whole-Body Tracking 端到端流水线统一为 **6 阶段**（参考采集 → 重定向 → 训练数据 → 策略学习 → 跨具身迁移 → 真机部署），并把 **SONIC / BeyondMimic / SD-AMP / Heracles / Any2Any / GMT(RGMT)** 6 条主流落地路径以 6 列对照表展开；包含 mermaid 端到端流程图、与 [人形 RL 身体系统栈](wiki/overview/humanoid-rl-motion-control-body-system-stack.md) 8 层框架的映射、6 类常见失败模式、评测视角。
- 链接：frontmatter `related` 拉入 [motion-retargeting-pipeline.md](wiki/concepts/motion-retargeting-pipeline.md)、[whole-body-control.md](wiki/concepts/whole-body-control.md)、[sim2real.md](wiki/concepts/sim2real.md)、[behavior-foundation-model.md](wiki/concepts/behavior-foundation-model.md) 与 SONIC / BeyondMimic / SD-AMP / Heracles / Any2Any / RGMT 全部 6 条路径的对应 method / entity 页；`sources` 链入对应 8 条原始资料。
- 关联清单：[`docs/checklists/tech-stack-next-phase-checklist-v23.md`](docs/checklists/tech-stack-next-phase-checklist-v23.md) P1「WBT 知识链 (+3)」中的首条页面打勾，剩余 2 页（`comparisons/sonic-vs-beyondmimic-vs-sdamp-vs-heracles.md`、`queries/cross-embodiment-transfer-strategy.md`）继续后推。

## [2026-05-29] lint | wiki/entities/paper-*.md、scripts/fix_paper_entity_lint.py — 清零 paper 实体信息型 lint 预警（261→0）

- 变更：批量补齐 **131** 个 `wiki/entities/paper-*.md` 的 frontmatter 来源键（`arxiv` / `venue` / `code`，从正文 URL、sources 文件名与索引表提取）；缺「方法 / 评测 / 对比」三段式的页面在 [参考来源] 前追加 **方法栈 / 实验与评测 / 与其他工作对比** 策展级摘要块；新增维护脚本 [fix_paper_entity_lint.py](scripts/fix_paper_entity_lint.py) 供后续复跑。
- 验证：`make lint` 信息型预警 **261→0**（`paper_missing_source_meta` / `paper_missing_three_sections` 均为 0）；`make ci-preflight` 通过。
- 关联 PR：与 [2026-05-29] stale-wiki-review 同期分支 `cursor/stale-wiki-review-d40c`（#420）。

## [2026-05-29] lint | stale-wiki-review — 14 个陈旧 wiki 页与较新 sources 交叉同步

- 逐页 review lint 陈旧报告：concepts（character-animation、terrain-adaptation、whole-body-coordination、capture-point-dcm、footstep-planning、embodied-data-cleaning、state-estimation）、methods（exoactor、gae）、comparisons（ppo-vs-sac）、formalizations（probability-flow）、entities（lafan1-dataset、pinocchio）、overview（robot-world-models-training-loop-taxonomy）；补 source 映射与正文切片，`updated: 2026-05-29`。
- 验证：lint 阻塞 **14→0**；`make ci-preflight` 通过。

## [2026-05-28] checklist-v23 | scripts/generate_link_graph.py、docs/main.js、docs/style.css、tests/test_generate_link_graph_latest_nodes.py — V23 P0「图谱 latest_wiki_nodes 时间窗口可配置」收口

- 变更：`scripts/generate_link_graph.py` 把 `latest_wiki_nodes_from_log` 从「锁定最新日历日」改为「最近 30 天回看 + 取前 N 项」；新增形参 `max_items` / `window_days`、模块级常量 `LATEST_NODES_DEFAULT=10` / `LATEST_NODES_CAP=30` / `LATEST_NODES_WINDOW_DAYS=30` / `LATEST_NODES_ENV_VAR="GRAPH_LATEST_NODES_MAX"`；新增 `resolve_latest_nodes_max()` 解析 CLI flag `--latest-nodes-max N`（优先级最高）→ 环境变量 `GRAPH_LATEST_NODES_MAX` → 默认 10，并 clamp 至 [1, 30]；`main()` 接入 argparse 后将 N 透传给 `_compute_graph_stats`。
- 前端：`docs/main.js renderLatestWikiNode` 在跨日返回时按 `recency` 分组渲染「维护日志时间线」（日期 + 项数小标 + 卡片网格），单日时维持原 cards 渲染；只通过 `#homeLatestWikiModule` 挂载点生效（即仅首页 `docs/index.html`），详情/图谱/路线图页不受影响。`docs/style.css` 新增 `.home-latest-wiki-timeline*` 三条轻量样式。
- 测试：`tests/test_generate_link_graph_latest_nodes.py` 新增 10 用例覆盖 `max_items` 单日截断 / `window_days` 多日合并 / 30 天外日期被排除 / `max_items=0` 返空 / CLI vs env 优先级 / 非法 env 回退默认值 / clamp 上限 30 + 下限 1；`PYTHONPATH=scripts python3 -m unittest discover tests -v` 87 通过（含 V22/V23 既有用例），唯一 ERROR `test_lint_wiki_stale_pages` 为 pytest 缺失的预存问题，与本次改动无关。
- 清单：[`docs/checklists/tech-stack-next-phase-checklist-v23.md`](docs/checklists/tech-stack-next-phase-checklist-v23.md) P0「图谱 latest_wiki_nodes 时间窗口可配置」打勾。

## [2026-05-28] ingest | sources/papers/bam_extended_friction_servos_arxiv_2410_08650.md、sources/repos/rhoban_bam.md — BAM 舵机扩展摩擦（arXiv:2410.08650 / ICRA 2025）与 Rhoban/bam 开源管线入库

- 原始资料：[bam_extended_friction_servos_arxiv_2410_08650.md](sources/papers/bam_extended_friction_servos_arxiv_2410_08650.md)（<https://arxiv.org/abs/2410.08650v1>、PDF/HTML）、[rhoban_bam.md](sources/repos/rhoban_bam.md)（<https://github.com/Rhoban/bam>）
- 新建 wiki：[paper-bam-extended-friction-servo-actuators.md](wiki/entities/paper-bam-extended-friction-servo-actuators.md)、[bam-better-actuator-models.md](wiki/entities/bam-better-actuator-models.md)
- 交叉更新：[sim2real.md](wiki/concepts/sim2real.md)、[system-identification.md](wiki/concepts/system-identification.md)、[actuator-network.md](wiki/methods/actuator-network.md)、[sim2real-gap-reduction.md](wiki/queries/sim2real-gap-reduction.md)

## [2026-05-28] ingest | sources/repos/open_duck_*.md — 接入 Open Duck 四仓（Mini Hub / Playground / Reference Motion / Runtime）并沉淀 wiki 实体与 sim2real、locomotion 交叉引用

- 原始资料：[open_duck_mini.md](sources/repos/open_duck_mini.md)、[open_duck_playground.md](sources/repos/open_duck_playground.md)、[open_duck_reference_motion_generator.md](sources/repos/open_duck_reference_motion_generator.md)、[open_duck_mini_runtime.md](sources/repos/open_duck_mini_runtime.md)
- 新建 wiki：[open-duck-mini.md](wiki/entities/open-duck-mini.md)、[open-duck-playground.md](wiki/entities/open-duck-playground.md)、[open-duck-reference-motion-generator.md](wiki/entities/open-duck-reference-motion-generator.md)、[open-duck-mini-runtime.md](wiki/entities/open-duck-mini-runtime.md)
- 交叉更新：[sim2real.md](wiki/concepts/sim2real.md)、[locomotion.md](wiki/tasks/locomotion.md)、[open-source-humanoid-hardware.md](wiki/entities/open-source-humanoid-hardware.md)、[disney-olaf-character-robot.md](wiki/methods/disney-olaf-character-robot.md)

## [2026-05-28] ingest | sources/repos/tnkr.md、sources/blogs/tnkr_launch_youtube_nlv.md — 接入 Tnkr 协作平台（tnkr.ai）与官方 launch 视频；沉淀 wiki/entities/tnkr.md；交叉更新 lerobot、urdf-studio、humanoid-robot

## [2026-05-28] ingest | sources/blogs/genesis_ai_simulation_world_10_blog.md — Genesis AI 博客：仿真评测引擎、Genesis World 1.0（Nyx/Quadrants）与 real-to-sim 相关性叙事

- 原始资料：[genesis_ai_simulation_world_10_blog.md](sources/blogs/genesis_ai_simulation_world_10_blog.md)（<https://www.genesis.ai/blog/the-role-of-simulation-in-scalable-robotics-genesis-world-10-and-the-path-forward>）
- 新建 wiki：[genesis-world-10.md](wiki/entities/genesis-world-10.md)、[simulation-evaluation-infrastructure.md](wiki/concepts/simulation-evaluation-infrastructure.md)
- 交叉更新：[genesis-sim.md](wiki/entities/genesis-sim.md)、[gene-26-5-genesis-ai.md](wiki/entities/gene-26-5-genesis-ai.md)、[genesis_gene_ecosystem.md](sources/papers/genesis_gene_ecosystem.md)、[sim2real.md](wiki/concepts/sim2real.md)

## [2026-05-28] ingest | sources/repos/aholo-viewer.md、sources/blogs/worldlabs_spark_2_0_streaming_3dgs.md — 接入 Aholo Viewer 与 Spark 2.0；沉淀 wiki/entities/spark-3dgs-renderer.md、wiki/entities/aholo-viewer.md、wiki/comparisons/spark-vs-aholo-web-3dgs-renderers.md；交叉更新 world-labs、gs-playground、generative-world-models

## [2026-05-28] fix(lint) | scripts/lint_wiki.py — 陈旧页面检测改用 git commit time，根治 cloud Agent 容器 fresh-clone 伪阳性

- 问题：`_check_sources_health` 用 `Path.stat().st_mtime` 比较 source 与 wiki 的修改时间；cloud Agent 容器 clone 时 `sources/papers/` 的 mtime 被刷成 checkout 时间，wiki 文件保留更早 mtime，导致 18+ 个 wiki 页被误报为「陈旧」（实际两边 git 提交日同日）。
- 修复：新增 `_build_git_mtime_map()`，一次 `git log --format=%ct --name-only` 全量扫描得到 `{Path: 最近提交 unix ts}`；`_check_sources_health` 改为优先使用 git mtime，未入库的本地新文件 fallback 到 fs mtime；非 git 环境下整体退化到旧行为不报硬错。
- 测试：`tests/test_lint_wiki_stale_pages.py` 4 用例覆盖 fresh-clone 场景不误报 / 真陈旧仍报 / 未提交本地编辑回退 fs mtime / 非 git 环境回退 fs mtime；全量 pytest 111 通过，ruff + mypy 均绿。
- 效果：本次 cloud 环境 `make lint` 陈旧条目从 19 个全伪阳性，变为 30 个均为 git 历史中真实「wiki 早于 source ≥1 天」的需 review 列表。

## [2026-05-27] query | wiki/queries/humanoid-motion-tracking-method-selection.md — 对照 smp.md 扩写后的源文件 review：纠正 mermaid 将 SMP 标为「判别器先验」的事实错误（SMP 为评分匹配，非判别器）；section 2 增补 SMP 选型轴（冻结扩散 + 可丢 MoCap vs AMP 同采样量 wall-clock ~1.8×）；frontmatter 加 smp.md 源、bump updated

## [2026-05-27] checklist-v23 | scripts/lint_wiki.py、tests/test_lint_wiki_paper_metadata.py — V23 P0「Entity-Paper 类页元数据 Lint」收口

- 变更：`scripts/lint_wiki.py` 新增 `_check_paper_entity_metadata`，针对 `wiki/entities/paper-*.md` 做两项信息型检查——frontmatter 是否含 `arxiv` / `venue` / `code` 任一来源键、正文是否覆盖「方法 / 评测 / 对比」三段式；两个新 result key `paper_missing_source_meta` 与 `paper_missing_three_sections` 加入 `INFO_ONLY_KEYS`，缺失不阻塞 CI 但写入 lint 报告作为 ingest 流水线基线。
- 基线快照：当前 131 个 paper-* 实体全部缺 arxiv/venue/code 显式键（依赖 `sources:` 间接来源），130/131 缺三段式之一（多数缺「评测」段），将作为后续 ingest 模板对齐的目标。
- 测试：`tests/test_lint_wiki_paper_metadata.py` 新增 6 用例（命中 arxiv/venue/code、缺失定位、三段式部分缺失、非 paper- 实体豁免、信息型计数）；`python -m pytest tests/ --ignore=tests/test_graph_layout.py --no-cov` 106 通过；`ruff check` 与 `mypy scripts/lint_wiki.py` 均绿。
- 清单：[`docs/checklists/tech-stack-next-phase-checklist-v23.md`](docs/checklists/tech-stack-next-phase-checklist-v23.md) P0「Entity-Paper 类页元数据 Lint」打勾。

## [2026-05-27] ingest | sources/papers/smp.md — 扩写 arXiv:2512.03028 完整摘录（摘要、SDS/ESM/GSI、任务与 AMP 对比、G1 部署、wiki 映射）；同步 sources/README 论文索引

## [2026-05-27] ingest | sources/repos/smp_suz_tsinghua.md — 接入清华 SUZ-tsinghua/smp（G1 上 SMP + mjlab 复现）；沉淀 wiki/entities/smp-g1-mjlab.md；交叉更新 wiki/methods/smp.md、wiki/entities/mimickit.md、wiki/entities/mjlab.md、wiki/comparisons/amp-add-smp-motion-prior-variants.md

## [2026-05-27] structural | wiki/entities/* — 为导航·SLAM 栈 21 仓补齐实体节点（slam-toolbox/cartographer/FAST-LIO 等 15 页）

## [2026-05-27] ingest | sources/repos/navigation_slam_autonomy_stack_catalog.md — 入库 Nav2/SLAM/Autoware/Isaac ROS/LeRobot/OpenVLA 等 21 仓；沉淀 wiki/overview/navigation-slam-autonomy-stack.md、wiki/comparisons/lidar-slam-lio-vio-selection.md、wiki/entities/navigation2.md、wiki/entities/autoware.md、wiki/entities/openvla.md、wiki/entities/isaac-ros-visual-slam.md、wiki/entities/isaac-ros-nvblox.md；互链 openloong/lerobot/vla/ros2

## [2026-05-27] ingest | sources/repos/multirotor_uav_stack_catalog.md 及 10 仓 — PX4/XTDrone/EGO-Planner/PyBullet Gym/AirSim/quad-swarm-rl/Crazyswarm2/Crazyflie/Flightmare/MAVSDK 入库；沉淀 wiki/overview/multirotor-simulation-planning-control-stack.md、wiki/entities/px4-autopilot.md、wiki/entities/airsim.md

## [2026-05-27] structural | 首页最新节点 — 人形 RL/AMP 61 篇 + BFM 41 篇论文实体与五类分类 hub

- 人形 RL 身体系统栈（42 篇）：`wiki/entities/paper-hrl-stack-*.md`
- 人形 AMP 运动先验（19 篇）：`wiki/entities/paper-amp-survey-*.md`
- BFM 论文实体（41 篇）：`wiki/entities/paper-bfm-*.md`、`wiki/entities/paper-behavior-foundation-model-humanoid.md`
- BFM 五类分类 hub（5）：`wiki/overview/bfm-category-01-forward-backward-representation.md`、`wiki/overview/bfm-category-02-goal-conditioned-learning.md`、`wiki/overview/bfm-category-03-intrinsic-reward-pretraining.md`、`wiki/overview/bfm-category-04-adaptation.md`、`wiki/overview/bfm-category-05-hierarchical-control.md`
- 总览：[`wiki/overview/humanoid-rl-motion-control-body-system-stack.md`](wiki/overview/humanoid-rl-motion-control-body-system-stack.md)、[`wiki/overview/humanoid-amp-motion-prior-survey.md`](wiki/overview/humanoid-amp-motion-prior-survey.md)、[`wiki/overview/bfm-41-papers-technology-map.md`](wiki/overview/bfm-41-papers-technology-map.md)、[`wiki/concepts/behavior-foundation-model.md`](wiki/concepts/behavior-foundation-model.md)

## [2026-05-27] structural | wiki/overview/bfm-category-01-* … bfm-category-05-* — BFM 五类问题各建图谱分类 hub 节点并交叉链接 41 篇论文实体

- 原始资料：[wechat_embodied_ai_lab_bfm_41_papers_survey.md](sources/blogs/wechat_embodied_ai_lab_bfm_41_papers_survey.md)（<https://mp.weixin.qq.com/s/Ei32la_vo0UW9Y_QCAqB2g>）
- 新增分类页（5）：见 `wiki/overview/bfm-category-01-forward-backward-representation.md` … `bfm-category-05-hierarchical-control.md`
- 交叉更新：41 个 `wiki/entities/paper-bfm-*` 回链对应分类 hub；[bfm-41-papers-technology-map.md](wiki/overview/bfm-41-papers-technology-map.md)、[behavior-foundation-model.md](wiki/concepts/behavior-foundation-model.md)
- 生成脚本：[scripts/generate_bfm_category_overviews.py](scripts/generate_bfm_category_overviews.py)

## [2026-05-27] structural | scripts/generate_bfm_awesome_wiki_entities.py — awesome-bfm 41 篇论文升格 wiki 实体详情页；图谱 + 搜索 + SW 缓存版本同步

- 新增实体（40 篇论文）：`wiki/entities/paper-bfm-zero.md` … `wiki/entities/paper-bfm-41-unihsi.md`（#13 复用 `wiki/entities/paper-behavior-foundation-model-humanoid.md`）
- 新增实体（9 个数据集）：`wiki/entities/dataset-bfm-humanoid-x.md` 等（AMASS 复用 `wiki/entities/amass.md`）
- 交叉更新：`wiki/overview/bfm-41-papers-technology-map.md`（Wiki 实体索引表）、`scripts/sync_sw_cache_version.py`（`sync_all_stats` 链内按 `exports/graph-stats.json` bump `docs/sw.js` CACHE_NAME）

## [2026-05-27] structural | docs/checklists/github-actions-ci-gate.md — 补 CI 门禁看板并开 PR 触发全量 GitHub Actions

- 变更：`docs/checklists/github-actions-ci-gate.md`、`docs/checklists/README.md`、`docs/checklists/cloud-agent-pr-workflow.md`、`schema/README.md`（交叉链接触发 Search & Export Quality Check）
- 目的：在 PR #387 未跑 Actions 即合并后，用 chore PR 重新拉起 `Tests` / `Wiki Lint` / `Search & Export Quality Check`；合并前以 Checks 全绿为准

## [2026-05-27] ingest | sources/papers/bfm_awesome_41_catalog.md、sources/papers/bfm_awesome_*.md（41+10）— awesome-bfm-papers 论文与数据集分别入库；消化更新 wiki/overview/bfm-41-papers-technology-map.md

- 原始资料：[`sources/papers/bfm_awesome_41_catalog.md`](sources/papers/bfm_awesome_41_catalog.md) 及 51 个 `bfm_awesome_<slug>.md`（41 篇论文 + 10 数据集；#13 交叉指向既有 [`bfm_humanoid_arxiv_2509_13780.md`](sources/papers/bfm_humanoid_arxiv_2509_13780.md)）；生成脚本 [`scripts/generate_bfm_awesome_sources.py`](scripts/generate_bfm_awesome_sources.py)；索引 [`sources/README.md`](sources/README.md)
- 沉淀/交叉更新：[`wiki/overview/bfm-41-papers-technology-map.md`](wiki/overview/bfm-41-papers-technology-map.md)（原始资料索引节、01 组 Source 列）、[`sources/blogs/wechat_embodied_ai_lab_bfm_41_papers_survey.md`](sources/blogs/wechat_embodied_ai_lab_bfm_41_papers_survey.md)

## [2026-05-26] checklist-v23 | scripts/search_wiki_core.py、tests/test_search_wiki_core.py — V23 P0「缩写/别名归一化检索 V2」收口

- 变更：`scripts/search_wiki_core.py` 的 `WIKI_ABBREVIATIONS` 在 V22 16 条基础上补齐 9 条 V22 期间高频缩写（**WBT** / **BFM** / **DAgger** / **RSI** / **RFC** / **RMA** / **EMA** / **LoRA** / **DoF**），共 25 条；映射均双向化（`_build_alias_indexes` 自动构造 forward + reverse）。
- 测试：`tests/test_search_wiki_core.py` 新增两组 subTest——`test_v22_abbreviations_expand_to_full`（9 条缩写 → 全称展开）与 `test_v22_full_phrases_expand_to_abbreviation`（9 条全称 → 缩写大写化反向命中），`python -m unittest tests.test_search_wiki_core -v` 26/26 通过。
- 门禁：`ruff check`、`ruff format --check`、`PYTHONPATH=scripts mypy scripts/search_wiki_core.py` 均绿。
- 清单：[`docs/checklists/tech-stack-next-phase-checklist-v23.md`](docs/checklists/tech-stack-next-phase-checklist-v23.md) P0「缩写/别名归一化检索 V2」打勾。

## [2026-05-26] ingest | sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md、sources/blogs/wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md — Agent Reach 重抓两篇微信公众号长文；42+19 篇论文分别入库并升格 wiki 实体节点

- 工具：已安装 [Panniantong/Agent-Reach](https://github.com/Panniantong/Agent-Reach) v1.4.0（`pip install` + `agent-reach install --channels=wechat`）；微信正文经 `~/.agent-reach/tools/wechat-article-for-ai`（Camoufox）
- 原始资料：[`sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md`](sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md)（<https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA>）、[`sources/blogs/wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md`](sources/blogs/wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md)（<https://mp.weixin.qq.com/s/YZsm3855iP3TNTTt1aou7w>）；抓取落盘 [`sources/raw/wechat_humanoid_rl_42_survey_2026-05-26.md`](sources/raw/wechat_humanoid_rl_42_survey_2026-05-26.md)、[`sources/raw/wechat_humanoid_amp_19_survey_2026-05-26.md`](sources/raw/wechat_humanoid_amp_19_survey_2026-05-26.md)
- 论文 source：[`sources/papers/humanoid_rl_stack_42_catalog.md`](sources/papers/humanoid_rl_stack_42_catalog.md) + 42× `humanoid_rl_stack_*`；[`sources/papers/humanoid_amp_survey_19_catalog.md`](sources/papers/humanoid_amp_survey_19_catalog.md) + 19× `humanoid_amp_survey_*`；生成脚本 [`scripts/generate_humanoid_stack_survey.py`](scripts/generate_humanoid_stack_survey.py)
- 沉淀实体（61）：`wiki/entities/paper-hrl-stack-01-*.md` … `paper-hrl-stack-42-*.md`；`wiki/entities/paper-amp-survey-01-*.md` … `paper-amp-survey-19-*.md`
- 交叉更新：[`wiki/overview/humanoid-rl-motion-control-body-system-stack.md`](wiki/overview/humanoid-rl-motion-control-body-system-stack.md)、[`wiki/overview/humanoid-amp-motion-prior-survey.md`](wiki/overview/humanoid-amp-motion-prior-survey.md)、[`sources/README.md`](sources/README.md)

## [2026-05-26] ingest | sources/repos/simplefoc_arduino_foc.md、sources/sites/simplefoc_documentation.md — 接入 SimpleFOC 生态；沉淀 wiki/entities/simplefoc.md、wiki/concepts/field-oriented-control.md；交叉更新 wiki/overview/motor-drive-firmware-bus-protocols.md

## [2026-05-26] ingest | sources/blogs/wechat_embodied_ai_lab_bfm_41_papers_survey.md — Agent Reach 抓取具身智能研究室 BFM 41 篇专题长文并消化入库

- 工具：已安装 [Panniantong/Agent-Reach](https://github.com/Panniantong/Agent-Reach) v1.4.0（`pip install` + `agent-reach install --channels=wechat`）；微信正文经 `~/.agent-reach/tools/wechat-article-for-ai`（Camoufox），Jina Reader 对 `mp.weixin.qq.com` 返回 CAPTCHA
- 原始资料：[`sources/blogs/wechat_embodied_ai_lab_bfm_41_papers_survey.md`](sources/blogs/wechat_embodied_ai_lab_bfm_41_papers_survey.md)（<https://mp.weixin.qq.com/s/Ei32la_vo0UW9Y_QCAqB2g>）；配套 [`sources/repos/awesome_bfm_papers.md`](sources/repos/awesome_bfm_papers.md)、[`sources/papers/bfm_survey_arxiv_2506_20487.md`](sources/papers/bfm_survey_arxiv_2506_20487.md)；索引 [`sources/README.md`](sources/README.md)
- 沉淀页面：[`wiki/overview/bfm-41-papers-technology-map.md`](wiki/overview/bfm-41-papers-technology-map.md)（五类问题 × 41 篇地图 + Mermaid + 智元/众擎策展观察）
- 交叉更新：[`wiki/concepts/behavior-foundation-model.md`](wiki/concepts/behavior-foundation-model.md)、[`wiki/overview/humanoid-rl-motion-control-body-system-stack.md`](wiki/overview/humanoid-rl-motion-control-body-system-stack.md)、[`wiki/entities/paper-behavior-foundation-model-humanoid.md`](wiki/entities/paper-behavior-foundation-model-humanoid.md)、[`sources/repos/panniantong_agent_reach.md`](sources/repos/panniantong_agent_reach.md)

## [2026-05-26] ingest | sources/repos/awesome_bfm_papers.md、sources/papers/bfm_survey_arxiv_2506_20487.md — 接入 awesome-bfm-papers 与 BFM 综述；沉淀 wiki/concepts/behavior-foundation-model.md；交叉更新 foundation-policy、whole-body-control、paper-behavior-foundation-model-humanoid、humanoid-rl-motion-control-body-system-stack

- 原始资料：<https://github.com/friedrichyuan/awesome-bfm-papers>、[`sources/papers/bfm_survey_arxiv_2506_20487.md`](sources/papers/bfm_survey_arxiv_2506_20487.md)（arXiv:2506.20487）；索引 [`sources/README.md`](sources/README.md)
- 沉淀页面：[`wiki/concepts/behavior-foundation-model.md`](wiki/concepts/behavior-foundation-model.md)（BFM 定义、预训练三线 + 适应两线 taxonomy、Mermaid 流程图）
- 交叉更新：[`wiki/concepts/foundation-policy.md`](wiki/concepts/foundation-policy.md)、[`wiki/concepts/whole-body-control.md`](wiki/concepts/whole-body-control.md)、[`wiki/entities/paper-behavior-foundation-model-humanoid.md`](wiki/entities/paper-behavior-foundation-model-humanoid.md)、[`wiki/overview/humanoid-rl-motion-control-body-system-stack.md`](wiki/overview/humanoid-rl-motion-control-body-system-stack.md)

## [2026-05-25] checklist-v22 | DoD 收口 & 初始化 V23

- V22 DoD 最后一项「`log.md` 记录 V22 关键改动」收口：本条目即为兑现物，把 V22 P0–P3 与 DoD 数值快照沉淀到日志，与 [`docs/checklists/tech-stack-next-phase-checklist-v22.md`](docs/checklists/tech-stack-next-phase-checklist-v22.md) 同步勾选并标注 2026-05-25 验证日期。
- V22 完整交付：
    - **P0 自动化**：① `scripts/search_wiki_core.py` 缩写归一化检索（16 条 WBC/VLA/IL/RL/MPC/PPO/SAC/HQP/CBF/CLF/BC/IK/FK/LIP/ZMP/TSID，双向展开 + "已展开为…"提示）；② `scripts/generate_link_graph.py` 社区粒度二级拆分（Girvan-Newman 一级 + Louvain `resolution=1.15` 二级），最大社区占比由 V21 46.1% → V22 15.9%（-30.2 pp），17 社区均衡分布；③ `scripts/lint_wiki.py` 新增 `methods_without_practitioner_query` 方法-Query 闭环 Lint（INFO 级，不阻塞 CI，作为 P1/P2 推进基线）。
    - **P1 动作重定向与角色化人形**：新增 5 页 `wiki/concepts/motion-retargeting-pipeline.md`、`wiki/formalizations/motion-retargeting-objective.md`、`wiki/comparisons/gmr-vs-nmr-vs-reactor.md`、`wiki/concepts/character-animation-vs-robotics.md`，覆盖「映射几何 → 目标函数 → 谱系对比 → 角色 vs 工业边界」四视角，双向回链 GMR / NMR / ReActor / SONIC / ExoActor / WBC / Sim2Real / Disney Olaf / Roboto Origin。
    - **P2 抓取与操作感知**：新增 3 页 `wiki/methods/grasp-pose-estimation.md`、`wiki/queries/grasp-policy-selection.md`、`wiki/comparisons/anygrasp-vs-graspnet.md`，覆盖 GraspNet 三代谱系（GPD → GraspNet-1Billion → Contact-GraspNet/AnyGrasp）与「检测式 + IL/VLA」选型；同步在 `wiki/concepts/contact-rich-manipulation.md` 与 `wiki/concepts/visuo-tactile-fusion.md` 中补「抓取 → 插装 → 精细操作」三段式级联，把 P1 触觉链路与 P2 抓取链路打通。
    - **P3 交互层**：① `docs/detail.html` + `docs/main.js` + `docs/style.css` 新增「关联页面社区分布」横向条形小图，按 link-graph 17 社区聚类显示当前节点邻域偏向；② `docs/graph.html` 新增「专题视图」切换器（10 项专题：动作重定向 / 抓取 / 触觉 / 通信协议 / WBC / Locomotion / VLA / IL+RL / Sim2Real / 状态估计），社区 id + path 片段双路并集判定，切换时自动 `fitToVisibleNodes()`。
    - **事实库**：`schema/canonical-facts.json` 由 140 → **156** 条（+16），重点补动作重定向 5 条 / 抓取与感知 10 条 / 近期 ingest 2 条（BifrostUMI / OpenLoong）。
- DoD 数值快照（验证日 2026-05-25 `exports/graph-stats.json` `generated_at: 2026-05-25`）：
    | 维度 | V22 目标 | V22 实测 | 达成情况 |
    |------|----------|----------|----------|
    | `make lint` | 0 errors | 0 errors（419/419 wiki/entity 页 ingest 来源覆盖率 100%） | ✅ 远超 |
    | 图谱节点 | ≥ 312 | 429 | ✅ +117 / +37.5% |
    | 图谱边 | ≥ 2050 | 3200 | ✅ +1150 / +56.1% |
    | 事实库 | ≥ 155 | 156 | ✅ 达标 |
    | `community_quality_warning` | false | false（最大社区 VLA 15.9% / 17 社区） | ✅ 远超 ≤ 40% 阈值 |
- 新建 [`docs/checklists/tech-stack-next-phase-checklist-v23.md`](docs/checklists/tech-stack-next-phase-checklist-v23.md)：专题选定「全身运动跟踪（WBT）与跨具身迁移」，配合「真机安全微调与 Sim2Real 深化」；P1 直接消化 V22 期间已 ingest 的 SONIC / SD-AMP / Heracles / Any2Any / SLowRL / BifrostUMI / BFM 等 WBT 谱系论文，P2 围绕 SLowRL 安全 LoRA / Heracles 扩散兜底等真机安全微调路径展开；P3 详情页新增「最近 ingest 时间线」与图谱专题视图扩充 3 项（WBT / 跨具身 / 真机安全）。V23 目标：节点 ≥ 445、边 ≥ 3320、事实库 ≥ 170、`largest_community_ratio ≤ 0.25`。
- 同步将 `README.md` 维护看板 + Sources Coverage badge、`AGENTS.md` § `docs/checklists/`、`docs/README.md` 常用入口、`docs/checklists/README.md` 当前入口与历史归档的「当前清单」指针从 V22 切到 V23；V22 进入历史归档区（`docs/checklists/README.md` 历史列表追加 v22 条目）。
- 本轮无代码改动，仅清单/日志状态回填与索引/指针同步；下一日按"每日推进一项"节奏从 V23 P0「缩写/别名归一化检索 V2」起步。

## [2026-05-25] ingest | sources/papers/slowrl_arxiv_2603_17092.md + any2any_arxiv_2605_23733.md — 接入 SLowRL（Go2 安全 LoRA 真机微调）与 Any2Any（跨具身 WBT 迁移）；沉淀 wiki/entities/paper-slowrl-safe-lora-locomotion-sim2real.md、wiki/entities/paper-any2any-cross-embodiment-wbt.md；交叉更新 sim2real、locomotion、humanoid-motion-tracking-method-selection、sonic-motion-tracking

- 原始资料：`sources/papers/slowrl_arxiv_2603_17092.md`（<https://arxiv.org/abs/2603.17092>）、`sources/papers/any2any_arxiv_2605_23733.md`（<https://arxiv.org/abs/2605.23733>）；索引 `sources/README.md`
- 沉淀页面：`wiki/entities/paper-slowrl-safe-lora-locomotion-sim2real.md`、`wiki/entities/paper-any2any-cross-embodiment-wbt.md`
- 交叉更新：`wiki/concepts/sim2real.md`、`wiki/tasks/locomotion.md`、`wiki/queries/humanoid-motion-tracking-method-selection.md`、`wiki/methods/sonic-motion-tracking.md`

## [2026-05-25] ingest | sources/papers/unified_walk_run_recovery_sdamp_arxiv_2605_18611.md + heracles_humanoid_diffusion_arxiv_2603_27756.md — 沉淀 wiki/entities/paper-unified-walk-run-recovery-sdamp.md、wiki/entities/paper-heracles-humanoid-diffusion.md；交叉更新 amp-reward、locomotion、balance-recovery、diffusion-motion-generation、humanoid-motion-tracking-method-selection、amp-mjlab、unitree-g1

## [2026-05-25] ingest | sources/repos/ppf-contact-solver.md — 接入 ZOZO GPU 接触求解器并沉淀 wiki/entities/ppf-contact-solver.md、wiki/entities/paper-ppf-cubic-barrier-contact-solver.md

## [2026-05-24] structural | docs/checklists/tech-stack-next-phase-checklist-v22.md — V22 DoD「community_quality_warning: false」回填打勾

- 触发：[`docs/checklists/tech-stack-next-phase-checklist-v22.md`](docs/checklists/tech-stack-next-phase-checklist-v22.md) DoD 余 2 项中数值最直接可验项；按"每日推进一项"节奏继续顺次回填
- 验证：`exports/graph-stats.json`（`generated_at: 2026-05-24`）实测 `community_count = 17`、`largest_community_ratio = 0.248`（最大社区 = "VLA（Vision-Language-Action） 社区" 105 / 423 = 24.8%，远低于 V22 ≤ 40% 阈值）、`community_quality_warning = false`、`singleton_communities = []`；最大社区占比相对 V21 基线 46.1% 累计下降 21.3 pp，结构稳定且 17 个社区中 ≥ 10 项节点的有 12 个
- 归因：V22 P0「社区粒度二级拆分」（Girvan-Newman 一级 + Louvain `resolution=1.15` 二级对占比 > 40% 且节点数 ≥ 30 的巨型社区做二级拆分）持续生效，叠加 P1 / P2 / P3 累积新增页面（motion-retargeting × 5 / 抓取链 × 3 / 接触-操作交叉 / VLA-WAM / BifrostUMI / OpenLoong / WorldVLN / easy_quadruped 等）形成的多向回链让原 Locomotion 巨型社区进一步均匀化
- 状态联动：V22 checklist DoD「community_quality_warning: false」由 `[ ]` 变 `[x]`；checklist 文件就地追加 2026-05-24 验证日期与数值快照
- 后续：DoD 余 1 项（`log.md` 记录 V22 关键改动）按节奏继续回填，本日新增日志本身即对该项的部分兑现；DoD 全部清零后基于 llm-wiki 与最新 graph-stats / 事实库 / 站点状态新建 V23 清单
- 本轮无代码改动，仅清单与日志状态回填

## [2026-05-24] ingest | sources/papers/worldvln_arxiv_2605_15964.md — 接入 WorldVLN 空中 VLN 自回归 WAM；沉淀 wiki/entities/paper-worldvln-aerial-vln-wam.md；交叉更新 vision-language-navigation、world-action-models

- 原始资料：`sources/papers/worldvln_arxiv_2605_15964.md`（<https://arxiv.org/abs/2605.15964>）、`sources/sites/worldvln-embodiedcity.md`、 `sources/repos/worldvln_embodiedcity.md`；索引 `sources/README.md`
- 沉淀页面：`wiki/entities/paper-worldvln-aerial-vln-wam.md`
- 交叉更新：`wiki/tasks/vision-language-navigation.md`、`wiki/concepts/world-action-models.md`

## [2026-05-24] ingest | sources/repos/easy_quadruped.md — 接入 Xzgz718/easy_quadruped（StanfordQuadruped 二次开发）并沉淀 wiki 实体

- 原始资料：`sources/repos/easy_quadruped.md`（上游 MIT [StanfordQuadruped](https://github.com/stanfordroboticsclub/StanfordQuadruped)，公开快照含 `src/` 步态控制、`pupper/` IK/标定、`sim/` MuJoCo 浮动机身闭环）
- 沉淀页面：`wiki/entities/easy-quadruped.md`
- 交叉更新：`wiki/entities/stanford-doggo-and-pupper.md`、`wiki/entities/quadruped-robot.md`、`wiki/concepts/gait-generation.md`、`references/repos/simulation.md`、`sources/README.md`

## [2026-05-24] structural | docs/checklists/tech-stack-next-phase-checklist-v22.md — V22 DoD「图谱节点 ≥ 312 / 边 ≥ 2050」回填打勾

- 触发：[`docs/checklists/tech-stack-next-phase-checklist-v22.md`](docs/checklists/tech-stack-next-phase-checklist-v22.md) DoD 余 3 项中数值最直接可验项；按"每日推进一项"节奏，今日选定图谱规模口径
- 验证：`exports/graph-stats.json`（`generated_at: 2026-05-23`）实测 `node_count = 421`（V22 目标 312，超 +109 / +34.9%）、`edge_count = 3122`（V22 目标 2050，超 +1072 / +52.3%）、`community_count = 17`、`largest_community_ratio = 0.254`、`orphan_nodes = []`，两项数值远超 V22 目标且与 V22 P1（动作重定向 5 页 + 多向回链）/ P2（抓取链 3 页 + 接触-操作交叉 + AnyGrasp/GraspNet 互链）/ P3（详情页社区分布 + 图谱专题视图）历史推升轨迹一致
- 状态联动：V22 checklist DoD「图谱节点 ≥ 312 边 ≥ 2050」由 `[ ]` 变 `[x]`；checklist 文件就地追加 2026-05-24 验证日期与数值快照
- 后续：DoD 余 2 项（`community_quality_warning: false`、`log.md` 记录 V22 关键改动）按节奏继续回填，全部完成后基于 llm-wiki 与最新 graph-stats / 事实库 / 站点状态新建 V23 清单
- 本轮无代码改动，仅清单与日志状态回填

## [2026-05-24] lint | docs/checklists/tech-stack-next-phase-checklist-v22.md — V22 DoD「`make lint`: 0 errors」回填打勾

- 触发：[`docs/checklists/tech-stack-next-phase-checklist-v22.md`](docs/checklists/tech-stack-next-phase-checklist-v22.md) DoD 余 4 项中最确定可验项；按 2026-05-23 后续计划「每日推进一项」执行
- 验证：`make lint` 实跑 = `python3 scripts/eval_search_quality.py`（通过率 37/37，≥ 80% 阈值）→ `python3 scripts/lint_wiki.py`（0 矛盾 / 0 空壳页 / 0 高频缺页 / 0 缺 type / 0 log.md 活跃度警告 / 0 缺摘要 / 0 Query 格式残缺 / 0 Formalization 缺公式 / 0 公式变量缺解释 / 0 README 版本不一致 / 0 图谱孤儿节点 / 0 Methods 缺 Formalization / Concept / 主要路线 / 0 Entities 缺 Methods/Tasks 出边 / 0 高频 methods 缺 queries/comparisons）；419/419 wiki/entity 页 ingest 来源覆盖率 100%；终行 "✅ 所有检查通过！"
- 状态联动：V22 checklist DoD「`make lint`: 0 errors」由 `[ ]` 变 `[x]`；checklist 文件就地追加验证日期与项目级 0 警告快照
- 后续：DoD 余 3 项（图谱节点 ≥ 312 边 ≥ 2050、`community_quality_warning: false`、log.md 记录 V22 关键改动）按节奏继续回填，全部完成后基于 llm-wiki 与最新 graph-stats / 事实库 / 站点状态新建 V23 清单
- 本轮无代码改动，仅清单与日志状态回填

## [2026-05-23] structural | schema/canonical-facts.json — V22 DoD 事实库扩展：140 → 156 条，补全动作重定向 / 抓取 / 近期 ingest 矛盾检测规则

- 触发：[`docs/checklists/tech-stack-next-phase-checklist-v22.md`](docs/checklists/tech-stack-next-phase-checklist-v22.md) DoD「事实库扩展至 155 条以上（重点补 motion-retargeting / grasp-pose 矛盾检测规则）」尚未打勾；P1 / P2 主线已沉淀大量新页（motion-retargeting-pipeline / motion-retargeting-objective / gmr-vs-nmr-vs-reactor / character-animation-vs-robotics / grasp-pose-estimation / grasp-policy-selection / anygrasp-vs-graspnet）以及 OpenLoong / BifrostUMI 实体页，需要让 `lint_wiki._check_contradictions` 覆盖到位
- 新增条目（17 条 → 总计 156 条）：
    - **动作重定向 5 条**：`GMR 运动学优化定位`（IK/QP/运动学层 vs 强化学习/仿真闭环）、`ReActor 双层联合优化`（参数化参考 + 单一策略联合更新 vs 纯运动学/离线/开环）、`Motion Retargeting Pipeline 端到端阶段`（8 阶段流水线 vs 单次映射/单阶段）、`Motion Retargeting 目标函数加权组合`（姿态 + 接触 + 平衡 + 限位 + 平滑 vs 单一姿态项）、`Character Humanoid 目标双重性`（表演可信度 × 物理可控性的三方博弈 vs 与工业人形等价）
    - **抓取与感知 10 条**：`6-DoF vs 7-DoF 抓取`（7-DoF = 6-DoF + 夹爪开度）、`GraspNet 三代谱系演进`（采样评估 → 稠密回归 → 时序关联）、`Contact-GraspNet 接触点参数化`（每点回归基线方向 + 接近向量 + 抓取宽度）、`AnyGrasp 跨帧时序关联`（many-to-many + COG 稳定度 + bin clearing vs 单帧独立）、`AnyGrasp SDK License 分发`（二进制 + License vs 完全开源）、`MPPH 抓取吞吐指标`（Mean Picks Per Hour，吞吐口径 vs 精度/AP 等价）、`抓取候选需显式碰撞检查`（网络分数 ≠ 物理可执行）、`抓取选型 检测式优先`（先检测式 grasp pose 起步，再 IL/VLA 替换可学环节）、`GraspNet-1Billion 评测基准`（百万级真实标注、公开 benchmark）
    - **近期 ingest 2 条**：`BifrostUMI 无机器人示范`（robot-free 全身示范 + 扩散 47-D 高层 + SKR vs 依赖真机遥操作/无扩散）、`OpenLoong 全栈开源`（青龙公版机硬件 + 软件 + 社区门户 vs 闭源/仅软件）
- 验证：`python3 scripts/lint_wiki.py` 退出码 0、0 contradictions、0 ⚠️ / 0 💡，"✅ 所有检查通过！"；419/419 wiki/entity 页 ingest 来源覆盖率 100%。`exports/graph-stats.json` 维持 421 nodes / 3122 edges / `community_quality_warning: false`，本次仅触动 schema，未派生重排
- 状态联动：V22 checklist DoD「事实库 155 条」由 `[ ]` 变 `[x]`；P1「动作重定向知识链 (+3)」父项由 `[~]` 变 `[x]`（3/3 子项已早期落地，仅此次回填父项状态）
- 后续：DoD 余 4 项（`make lint` 0 errors / 图谱节点 ≥ 312 边 ≥ 2050 / `community_quality_warning: false` / log.md 记录 V22 关键改动）均已在历史记录中达成或自然满足，下日按"每日推进一项"继续顺次打勾或在 V22 完全收尾时新建 V23

## [2026-05-23] ingest | sources/papers/bifrost_umi_arxiv_2605_03452.md — 接入 BifrostUMI 无机器人人形全身示范与 SKR 管线并沉淀 wiki/entities/paper-bifrost-umi.md

- 原始资料：`sources/papers/bifrost_umi_arxiv_2605_03452.md`（<https://arxiv.org/abs/2605.03452>）、`sources/sites/bifrost-umi-project.md`（<https://baai-aether.github.io/BifrostUMI/>）；索引 `sources/README.md`
- 沉淀页面：`wiki/entities/paper-bifrost-umi.md`（Robot-Free 采集、扩散 47-D 关键点高层、SKR、mink IK + WBC、G1 实验与 Mermaid 管线）
- 交叉更新：`wiki/tasks/teleoperation.md`、`wiki/tasks/loco-manipulation.md`、`wiki/concepts/motion-retargeting.md`、`wiki/methods/diffusion-policy.md`、`wiki/entities/unitree-g1.md`、`sources/papers/teleoperation.md`
- 派生再生成：`make ci-preflight`

## [2026-05-23] ingest | sources/repos/openloong.md — 接入 OpenLoong 青龙全栈开源（硬件 AtomGit、Framework、Dyn-Control、社区门户）并沉淀 wiki/entities/openloong.md

- 原始资料：`sources/repos/openloong.md`、`sources/repos/openloong_hardware.md`（<https://atomgit.com/openloong/OpenLoongHardware/tree/main>）、`sources/sites/openloong_community.md`（<https://www.openloong.org.cn/cn/projects/openloong>）；索引 `sources/README.md`
- 沉淀页面：`wiki/entities/openloong.md`（四层架构、硬件 TA 子系统、Framework 子仓矩阵、MPC+WBC 与并行栈链接）
- 交叉更新：`wiki/entities/open-source-humanoid-hardware.md`、`wiki/entities/humanoid-robot.md`
- 派生再生成：`make ci-preflight`

## [2026-05-22] ingest | sources/papers/esi_bench_arxiv_2605_18746.md — 接入 ESI-Bench 具身空间智能基准并沉淀 wiki/entities/esi-bench.md

- 原始资料：`sources/papers/esi_bench_arxiv_2605_18746.md`（<https://arxiv.org/abs/2605.18746>）、`sources/sites/esi-bench-project.md`（<https://esi-bench.github.io/>）、`sources/repos/esi_bench.md`（<https://github.com/ESI-Bench/ESI-Bench>）；索引 `sources/README.md`
- 沉淀页面：`wiki/entities/esi-bench.md`（感知–行动环、10/29/3081 任务 taxonomy、MLLM 主动/被动/oracle 发现）
- 交叉更新：`wiki/concepts/3d-spatial-vqa.md`
- 派生再生成：`make ci-preflight`

## [2026-05-22] ingest | sources/papers/wem_arxiv_2605_19957.md — 接入 WEM/World-Ego Modeling 与 HTEWorld；沉淀 wiki/entities/paper-wem-world-ego-modeling.md 并交叉更新 generative-world-models、robot-world-models-taxonomy、loco-manipulation、ewmbench

## [2026-05-22] ingest | sources/blogs/wechat_shenlan_vln_repro_four_paradigms_2026.md — Agent Reach 抓取深蓝具身智能 VLN 四范式新手复现长文并消化入库

- 原始资料：`sources/blogs/wechat_shenlan_vln_repro_four_paradigms_2026.md`（<https://mp.weixin.qq.com/s/AzCDukzwrfIyms_65kh1mg>）；索引 `sources/README.md`
- 沉淀页面：`wiki/overview/vln-open-source-repro-paradigms.md`（VLFM / NavGPT / NoMaD / Uni-NaVid + Mermaid 演进 + 复现门槛表）
- 交叉更新：`wiki/tasks/vision-language-navigation.md`、`wiki/overview/vla-open-source-repro-landscape-2025.md`、`sources/blogs/wechat_shenlan_vla_github_repro_survey_2025.md`
- 派生再生成：`make ci-preflight`

## [2026-05-22] ingest | sources/blogs/wechat_shenlan_vla_github_repro_survey_2025.md — Agent Reach 抓取深蓝具身智能 VLA GitHub 复现推荐长文并消化入库

- 原始资料：`sources/blogs/wechat_shenlan_vla_github_repro_survey_2025.md`（<https://mp.weixin.qq.com/s/k_i-1NEBP-lEzth19HOHkQ>）；索引 `sources/README.md`
- 沉淀页面：`wiki/overview/vla-open-source-repro-landscape-2025.md`（11 项开源栈 + Mermaid 景观 + 复现目标表）
- 交叉更新：`wiki/methods/vla.md`、`wiki/methods/star-vla.md`、`wiki/queries/manipulation-vla-architecture-selection.md`、`sources/blogs/wechat_shenlan_lie_group_lie_algebra_quaternion.md`
- 派生再生成：`make ci-preflight`

## [2026-05-22] ingest | sources/blogs/wechat_shenlan_lie_group_lie_algebra_quaternion.md — 安装 Agent Reach 抓取深蓝具身智能李群/李代数/四元数专栏文并消化入库

- 工具：已安装 [Panniantong/Agent-Reach](https://github.com/Panniantong/Agent-Reach) v1.4.0（`pip install` + `agent-reach install --channels=wechat`）；微信正文经 `~/.agent-reach/tools/wechat-article-for-ai`（Camoufox），Jina Reader 对 `mp.weixin.qq.com` 返回 CAPTCHA
- 原始资料：`sources/blogs/wechat_shenlan_lie_group_lie_algebra_quaternion.md`（<https://mp.weixin.qq.com/s/JviRH2LW-fkCHA5gY7Qflw>）；索引 `sources/README.md`
- 沉淀页面：`wiki/formalizations/lie-group-rigid-body-motions.md`
- 交叉更新：`wiki/formalizations/se3-representation.md`、`wiki/entities/modern-robotics-book.md`、`sources/repos/panniantong_agent_reach.md`
- 派生再生成：`make ci-preflight`

## [2026-05-22] ingest | sources/sites/robotics-venues-primary-refs.md — 汇总 ICRA、IROS、CoRL、RSS、T-RO、IJRR、Science Robotics 官方介绍与投稿入口；沉淀 wiki/comparisons/robotics-research-venues.md

- 原始资料：`sources/sites/robotics-venues-primary-refs.md`；索引 `sources/README.md`
- 沉淀页面：`wiki/comparisons/robotics-research-venues.md`
- 交叉更新：`wiki/overview/robot-learning-overview.md`、`references/papers/README.md`
- 派生再生成：`make ci-preflight`（与 main 上 PR #353 一致；合并时与深蓝专栏 ingest 同日块并列保留）

## [2026-05-21] ingest | sources/sites/kimodo-project.md、sources/papers/kimodo_arxiv_2603_15546.md — 深化 Kimodo 官方项目页与 arXiv:2603.15546 论文摘录；扩充 sources/repos/kimodo.md、wiki/entities/kimodo.md（两阶段去噪、变体选型、Mermaid 管线、GEM/SONIC/ProtoMotions 互链）；交叉更新 wiki/methods/diffusion-motion-generation.md

## [2026-05-21] feat(ux) | docs/detail.html、docs/main.js、docs/style.css — V22 P3 详情页「关联项按社区分布」小条形图（基于 link-graph 社区，替换早些时候的按 type 分桶版本）

- 触发：PR #347 review，社区维度（link-graph 的 Girvan-Newman + Louvain 二级拆分）比类型维度更有信息量——type 字段与 frontmatter 直接重复，而社区分桶能体现「当前节点的 1-hop 邻域聚集在哪几个主题」，与 V22 P0 的社区粒度二级拆分（17 个社区 / largest_community_ratio ≤ 0.40）形成闭环
- 改动形态：
  - [`docs/detail.html`](docs/detail.html)：`#detailRelatedTypeDist` 容器更名为 `#detailRelatedCommunityDist`，标题文案改「按社区分布」
  - [`docs/main.js`](docs/main.js)：移除按 type 派生中文标签的 `deriveDetailCategoryLabel()` 与 `renderRelatedTypeDistribution()`；新增 `ensureDetailCommunityIndex()`（懒加载 `exports/link-graph.json`，建立 `pathToCommunity` Map 与 `communityLabel` 字典，失败兜底为空 Map）与 `renderRelatedCommunityDistribution()`（按 detail page 的 `path` 查表 → 拿社区 ID → 计数；不在图谱内的 roadmap / reference / tech_map 统一桶为「未分类」并永远排在末尾，避免遮挡有效社区）；社区标签显式 `replace(/\s*社区\s*$/, '')` 去掉末尾「社区」二字以节省横向空间，悬停 `title` 仍保留完整原始标签
  - [`docs/style.css`](docs/style.css)：`.related-type-*` 系列样式整体改名为 `.related-community-*`，桌面端标签列宽 92px → 160px，540px 窄屏 78px → 110px，以容纳更长的社区中文标签（如 "Whole-Body Control (WBC，全身控制)"）
  - [`docs/checklists/tech-stack-next-phase-checklist-v22.md`](docs/checklists/tech-stack-next-phase-checklist-v22.md)：P3 首项标题与实现说明同步换为「关联社区分布」版本，附 type→community 切换理由
- 验证：`make lint-js` 通过；本地 http.server + Puppeteer 视口截图 `wiki-concepts-whole-body-control` 桌面 / 移动双端（共 12 项 · 8 个社区，含 Whole-Body Control 4 / Motion Retargeting 3 / Imitation Learning 1 / Locomotion 1 / Sim2Real 1 / Unitree G1 1 / Reward Design 1 / 未分类 1）与 `wiki-concepts-armature-modeling`（共 5 项 · 3 个社区，WBC 3 / Motion Retargeting 1 / Sim2Real 1）均正确落稳
- 截图：`.cursor-artifacts/screenshots/detail-related-community-dist-wbc.png`、`detail-related-community-dist-wbc-mobile.png`、`detail-related-community-dist.png`

## [2026-05-21] feat(ux) | docs/detail.html、docs/main.js、docs/style.css — V22 P3 详情页「关联类型分布」小条形图

- 触发：[`docs/checklists/tech-stack-next-phase-checklist-v22.md`](docs/checklists/tech-stack-next-phase-checklist-v22.md) P3「详情页关联类型分布小条形图」唯一子项；P0–P2 已全部落地，进入交互层关系视角增强阶段
- 改动形态：
  - [`docs/detail.html`](docs/detail.html)：在 `#detail-related` 标题下新增 `#detailRelatedTypeDist` 容器（含标题 / Meta / 横向条形栅格），默认 hidden，由 JS 在有关联项时显式打开
  - [`docs/main.js`](docs/main.js)：新增 `deriveDetailCategoryLabel()`（按 `path` 优先 → `type` 兜底 → `id` 前缀兜底，输出中文标签：概念 / 方法 / 形式化 / 对比 / Query / 任务 / 实体 / 总览 / 深挖 / 路线图 / 技术地图）；新增 `renderRelatedTypeDistribution()`（统计、按计数倒序+标签字典序、最大计数为 100% 基准、其余按比例并保底 6% 可见宽度），在 `renderDetailPage` 的正常态与「未匹配 detail page」空态均调用一次以避免幽灵骨架
  - [`docs/style.css`](docs/style.css)：新增 `.related-type-dist*` 系列样式（卡片化容器 / 三列网格 `92px 1fr 56px`：标签—轨道—计数 / `var(--accent)` 填充 / 540px 窄屏自适应缩列至 `78px 1fr 46px`）
- 验证：`make lint-js` 通过（仅一条 pre-existing `resetMermaidLightboxView` 未使用警告，与本次改动无关）；本地 `python3 -m http.server 8765` + `puppeteer-core` 视口截图（`/opt/pw-browsers/chromium-1194/chrome-linux/chrome`）打开 `detail.html?id=wiki-concepts-armature-modeling` 锚点 `detail-related`，条形图正确显示「方法 / 概念」两类共 5 项；截图落 `.cursor-artifacts/screenshots/detail-related-type-dist.png`
- 状态联动：V22 checklist 「详情页关联类型分布小条形图」由 `[ ]` 变 `[x]`；P3 剩余「图谱页专题视图切换器」与 DoD 收尾继续推进

## [2026-05-21] ingest | sources/papers/deeprl_locomotion_action_space_sca2017.md — Peng & van de Panne SCA 2017 四动作空间 DeepRL 对照；沉淀 wiki/entities/paper-deeprl-locomotion-action-space-sca2017.md；交叉更新 rl_pd 索引、legged-humanoid-rl-pd-gain-setting、xue-bin-peng、locomotion

## [2026-05-21] ingest | sources/papers/gencad_arxiv_2409_16294.md、sources/papers/gencad3d_arxiv_2509_15246.md、sources/sites/gencad-github-io.md、sources/sites/gencad3d-github-io.md、sources/repos/ferdous-alam-gencad.md、sources/repos/yunomi-git-gencad-3d.md — 入库 GenCAD / GenCAD-3D 论文、项目页与代码仓；沉淀 wiki/entities/gencad.md、wiki/entities/gencad-3d.md；交叉更新 wiki/concepts/text-to-cad.md、sources/sites/text-to-cad-tools.md

## [2026-05-21] structural | wiki/concepts/contact-rich-manipulation.md、wiki/concepts/visuo-tactile-fusion.md — V22 P2 接触/操作交叉补强：补「抓取 → 插装 → 精细操作」级联引用，打通 P1 触觉链路与 P2 抓取链路

- 触发：[`docs/checklists/tech-stack-next-phase-checklist-v22.md`](docs/checklists/tech-stack-next-phase-checklist-v22.md) P2「接触/操作交叉补强」唯一子项；V22 P2 抓取知识链 (+3) 已落地，需把上游检测式 grasp 与本页中段执行层、下游触觉精细操作连成一条流水线视角
- 改动形态：
  - [`wiki/concepts/contact-rich-manipulation.md`](wiki/concepts/contact-rich-manipulation.md)：新增「抓取 → 插装 → 精细操作（级联视角）」小节，三段式表格显式串联 P2 上游（[Grasp Pose Estimation](wiki/methods/grasp-pose-estimation.md)、[AnyGrasp](wiki/entities/anygrasp.md)、[ContactNet](wiki/methods/contact-net.md)、[抓取策略选型 Query](wiki/queries/grasp-policy-selection.md)、[AnyGrasp vs GraspNet](wiki/comparisons/anygrasp-vs-graspnet.md)）→ 本页中段 → P1 下游（[Impedance Control](wiki/concepts/impedance-control.md)、[Tactile Impedance Control](wiki/methods/tactile-impedance-control.md)、[TSID](wiki/concepts/tsid.md)/[WBC](wiki/concepts/whole-body-control.md)），并补「① 准但 ② 没接管会撞死」的工程含义说明；frontmatter `related` 与「关联页面」尾部互链至 P2 抓取链
  - [`wiki/concepts/visuo-tactile-fusion.md`](wiki/concepts/visuo-tactile-fusion.md)：新增同名小节，附 Mermaid 流水线图与三段式表格，强调「检测式 grasp 不带接触可信度，门控/注意力必须在触觉给出几何漂移信号时立即让出权重」这一常被忽略的衔接点；frontmatter `related` 与「关联页面」加入 P2 抓取链与 [Tactile Impedance Control](wiki/methods/tactile-impedance-control.md)、[Hybrid Force-Position Control](wiki/concepts/hybrid-force-position-control.md)
  - 两页 `updated` 字段刷新至 2026-05-21
- 验证：`python3 scripts/eval_search_quality.py` 37/37 通过；`python3 scripts/check_export_quality.py` 12/12 通过；`make ci-preflight` 同步派生产物（page catalog / exports / search-index / link-graph / docs/index.html / sitemap）。`exports/graph-stats.json`：节点 410、边 3004（远超 V22 目标 312/2050）、largest_community_ratio 0.207、`community_quality_warning: false`。`lint_wiki.py` 9 项 `stale_pages` 均为同日早些 ingest 引入的历史 baseline，与本次改动无关
- 状态联动：V22 checklist 「接触/操作交叉补强」由 `[ ]` 变 `[x]`；P2 全部子项落地完毕

## [2026-05-21] ingest | sources/repos/sensenova-skills.md — OpenSenseNova/SenseNova-Skills 入库并沉淀 wiki/entities/sensenova-skills.md；交叉更新 wiki/entities/hermes-agent.md、wiki/entities/mattpocock-skills.md

## [2026-05-21] ingest | sources/repos/boyu_ai_hands_on_rl.md、sources/sites/hrl-boyuai-hands-on-rl.md、sources/courses/boyuai_hands_on_rl_elites_course.md — 接入动手学强化学习（蘑菇书）在线书/代码仓/伯禹视频课并沉淀 wiki/entities/hands-on-rl-book.md；交叉更新 wiki/methods/reinforcement-learning.md、roadmap/depth-rl-locomotion.md、roadmap/motion-control.md、wiki/overview/robot-learning-overview.md

## [2026-05-21] ingest | sources/sites/nvidia-physical-ai-learning.md、sources/courses/nvidia_sim_to_real_so101_isaac.md — 入库 NVIDIA Physical AI 门户与 SO-101 Sim2Real 课；沉淀 wiki/entities/nvidia-physical-ai-learning.md、wiki/entities/nvidia-so101-sim2real-lab-workflow.md；互链 sim2real、lerobot、isaac-lab、vla、sage

## [2026-05-21] structural | wiki/concepts/domain-randomization.md 等 17 页 — 清理 lint 长期 stale 预存量：按 source 给 17 个 wiki 页补 ingest 档案交叉引用

- 触发：`make lint` 报「陈旧页面」17 条（mtime 判定：source 比 wiki 新 ≥ 24h）；预存自 2026-05-19 起多次 ingest 累积，与本批改动前的提交无关
- 影响页面（按 source 分组）：
  - `sources/papers/barkour_arxiv_2305_14654.md` → [`wiki/concepts/domain-randomization.md`](wiki/concepts/domain-randomization.md)、[`wiki/concepts/sim2real.md`](wiki/concepts/sim2real.md)、[`wiki/methods/reinforcement-learning.md`](wiki/methods/reinforcement-learning.md)
  - `sources/papers/bfm_humanoid_arxiv_2509_13780.md` → [`wiki/tasks/teleoperation.md`](wiki/tasks/teleoperation.md)、[`wiki/concepts/privileged-training.md`](wiki/concepts/privileged-training.md)、[`wiki/methods/dagger.md`](wiki/methods/dagger.md)、[`wiki/concepts/curriculum-learning.md`](wiki/concepts/curriculum-learning.md)、[`wiki/entities/amass.md`](wiki/entities/amass.md)、[`wiki/entities/unitree-g1.md`](wiki/entities/unitree-g1.md)
  - `sources/papers/capvector_arxiv_2605_10903.md` → [`wiki/methods/star-vla.md`](wiki/methods/star-vla.md)
  - `sources/papers/defi_arxiv_2604_16391.md` → [`wiki/methods/diffusion-policy.md`](wiki/methods/diffusion-policy.md)、[`wiki/methods/action-chunking.md`](wiki/methods/action-chunking.md)
  - `sources/papers/holomotion_arxiv_2605_15336.md` → [`wiki/methods/imitation-learning.md`](wiki/methods/imitation-learning.md)
  - `sources/papers/physforge_arxiv_2605_05163.md` → [`wiki/entities/sapien.md`](wiki/entities/sapien.md)
  - `sources/papers/robot_link_rotor_inertia_primary_refs.md` → [`wiki/entities/modern-robotics-book.md`](wiki/entities/modern-robotics-book.md)
  - `sources/papers/system_identification.md` → [`wiki/methods/actuator-network.md`](wiki/methods/actuator-network.md)（已有引用，扩写覆盖范围以反映 source 现含 Hwangbo / Gautier–Khalil / Grandia / Peng 等条目）
  - `sources/papers/wm_robot_survey_arxiv_2605_00080.md` → [`wiki/methods/model-based-rl.md`](wiki/methods/model-based-rl.md)
- 改动形态：每页在「参考来源」追加 1 条 ingest 档案行（含一句话提炼），统一与项目约定模式对齐；未做结构/正文重写
- 验证：`make lint` 17 → 0 issues；`make ci-preflight` 通过（同步 `exports/`、`docs/exports/`、`docs/search-index.json` 等，导出质量 12/12）

## [2026-05-21] fix(search): 搜索回归 WBC/MPC 定义页排名 — 条件化 comparison 提权 + 定义页 canonical boost

- `scripts/search_wiki_core.py`：`comparison` 类型仅在查询含「对比/选型」等意图时 ×1.3；WBC/MPC 定义页在缩写命中时 ×1.4 canonical boost
- `scripts/search_indexing.py`：`全身控制` / `模型预测控制` 同义词展开至 wbc/mpc
- 补强 `wiki/concepts/whole-body-control.md`、`wiki/methods/model-predictive-control.md` 标题与 summary 中文检索词
- 验证：`eval_search_quality.py` 37/37（原 35/37）

## [2026-05-21] query | wiki/queries/humanoid-motion-tracking-method-selection.md 等 — V22 方法-Query 闭环：31 条高频 methods 落地预警清零

- 新增 Query：`wiki/queries/humanoid-motion-tracking-method-selection.md`、`manipulation-vla-architecture-selection.md`、`humanoid-contact-character-control-guide.md`、`dexterous-manipulation-data-pipeline.md`
- 新增 Comparison：`wiki/comparisons/amp-add-smp-motion-prior-variants.md`
- 覆盖 methods：`deepmimic`、`beyondmimic`、`amp-reward`、`add`、`smp`、`motionbricks`、`any2track`、`ams`、`gentlehumanoid`、`ase`、`genmo`、`diffusion-motion-generation`、`mimic-video`、`defi`、`dwm`、`star-vla`、`pi07-policy`、`π0-policy`、`pelican-unified-1`、`claw`、`being-h07`、`disney-olaf`、`humanoid-transformer-touch-dreaming`、`hipan`、`zest`、`efgcl`、`auto-labeling-pipelines`、`wilor`、`tactile-impedance-control`、`actuator-network`、`gae`（共 31 页）
- 注册：`wiki/queries/README.md`
- 派生再生成：`make ci-preflight`

## [2026-05-21] ingest | sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md、sources/blogs/wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md — 安装 Agent Reach 抓取具身智能研究室两篇微信公众号长文并消化入库

- 工具：已安装 [Panniantong/Agent-Reach](https://github.com/Panniantong/Agent-Reach) v1.4.0（`pip install` + `agent-reach install --channels=wechat`）；微信正文经 `~/.agent-reach/tools/wechat-article-for-ai`（Camoufox），Jina Reader 对 `mp.weixin.qq.com` 返回 CAPTCHA
- 原始资料：`sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md`（<https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA>）、`sources/blogs/wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md`（<https://mp.weixin.qq.com/s/YZsm3855iP3TNTTt1aou7w>）；索引 `sources/README.md`
- 沉淀页面：`wiki/overview/humanoid-amp-motion-prior-survey.md`；补强 `wiki/overview/humanoid-rl-motion-control-body-system-stack.md`
- 交叉更新：`wiki/methods/amp-reward.md`、`wiki/entities/agent-reach.md`、`sources/repos/panniantong_agent_reach.md`
- 派生再生成：`make ci-preflight` 同步 `exports/`、`docs/exports/`、`docs/search-index.json`、`docs/sitemap.xml`、`README.md`、`docs/index.html`、`index.md` 等

## [2026-05-20] structural | wiki/comparisons/anygrasp-vs-graspnet.md — V22 P2 AnyGrasp vs GraspNet 抓取检测家族对比

- 新增页面：`wiki/comparisons/anygrasp-vs-graspnet.md`，按「一句话定义 + 14 维核心对比表 + Mermaid 数据流并排图（GraspNet 家族白盒基线 / AnyGrasp SDK 工程闭环）+ 三类适用场景 + 6 类常见误判 + 决策矩阵 + 评测指标视角」结构覆盖 GraspNet-1Billion / Contact-GraspNet / GSNet / AnyGrasp 四条子路线；显式区分「白盒改造 vs 工程化交付」「单帧 vs 动态跨帧」「完全开源 vs 二进制 License」三对核心取舍。
- 交叉互链：`wiki/methods/grasp-pose-estimation.md`、`wiki/entities/anygrasp.md`、`wiki/queries/grasp-policy-selection.md`、`wiki/methods/contact-net.md`、`wiki/tasks/manipulation.md` 的 frontmatter `related` 与「关联页面」均加入本页入口，形成「方法谱系页 + 实体页 + Query + 对比页」四级互链闭环。
- 清单推进：`docs/checklists/tech-stack-next-phase-checklist-v22.md` P2「抓取知识链 (+3)」第三项 `anygrasp-vs-graspnet.md` 打勾，整体专题完结进入 `[x]` 完成状态。
- 派生再生成：`make ci-preflight` 同步 `exports/`、`docs/exports/`、`docs/search-index.json`、`docs/sitemap.xml`、`README.md`、`docs/index.html`、`exports/lint-report.md`（图谱节点 399 → 400、边数 2836 → 2850、comparison 类型从 18 → 19；陈旧页面 21 条与本次改动无关，为今日早些 ingest 留下的预存量）。

## [2026-05-20] query | wiki/queries/grasp-policy-selection.md — V22 P2 抓取策略选型 Query 落地

- 新增页面：`wiki/queries/grasp-policy-selection.md`，覆盖三轴选型（物体已知度 / 候选稠密度 / 方法类型）+ TL;DR 决策树 + 四类推荐组合 pipeline（已知物体 / 桌面 bin picking / 动态场景 / 任务级语言指令）+ 关键工程经验 + 常见误区，与 [Grasp Pose Estimation](wiki/methods/grasp-pose-estimation.md) / [AnyGrasp](wiki/entities/anygrasp.md) / [Manipulation](wiki/tasks/manipulation.md) / [Visual Servoing](wiki/methods/visual-servoing.md) / [Contact-Rich Manipulation](wiki/concepts/contact-rich-manipulation.md) 互链。
- 交叉互链：`wiki/queries/README.md` 注册新 Query；`wiki/methods/grasp-pose-estimation.md` frontmatter `related` + 「关联页面」加入本页；`wiki/entities/anygrasp.md`、`wiki/tasks/manipulation.md` 关联页面区块新增 Query 入口。
- 清单推进：`docs/checklists/tech-stack-next-phase-checklist-v22.md` P2「抓取知识链」第二项 `grasp-policy-selection.md` 打勾，附实现摘要。
- 派生再生成：`make ci-preflight`。

## [2026-05-20] ingest | sources/papers/defi_arxiv_2604_16391.md — DeFI 解耦前向/逆动力学 VLA；wiki/methods/defi-decoupled-dynamics-vla.md

## [2026-05-20] structural | wiki/methods/grasp-pose-estimation.md — V22 P2 抓取位姿估计方法谱系页

- 新增 `wiki/methods/grasp-pose-estimation.md`，覆盖 6-DoF/7-DoF 表征、三代谱系（GPD → GraspNet-1Billion → Contact-GraspNet/GSNet/Graspness/AnyGrasp）、点云/RGBD 输入对照、AP/MPPH 评测、下游 cuRobo/视觉伺服/触觉闭环串联与常见误区，含 Mermaid 谱系图。
- 交叉互链：`wiki/entities/anygrasp.md` frontmatter `related` 与「关联页面」回链；`wiki/tasks/manipulation.md` 关联方法区块加入条目；`wiki/methods/contact-net.md` 关联页面新增本页；`references/repos/manipulation-perception.md` 顶部加入「方法谱系总览」指针；`index.md` 重点页面新增条目。
- 清单推进：`docs/checklists/tech-stack-next-phase-checklist-v22.md` P2「抓取知识链」首项打勾，整体专题进入 `[~]` 进行中状态。

## [2026-05-20] ingest | sources/papers/robot_link_rotor_inertia_primary_refs.md — 连杆/转子惯量一手资料入库并沉淀 wiki

- 原始资料：`sources/papers/robot_link_rotor_inertia_primary_refs.md`（URDF / Modern Robotics Ch.8 / Gautier–Khalil 1990 / MuJoCo armature）
- 沉淀页面：`wiki/concepts/robot-link-and-rotor-inertia.md`
- 交叉更新：`wiki/concepts/armature-modeling.md`、`wiki/concepts/system-identification.md`
- 派生再生成：`make ci-preflight`

## [2026-05-20] ingest | sources/repos/mattpocock-skills.md — mattpocock/skills 入库并沉淀 wiki

- 原始资料：`sources/repos/mattpocock-skills.md`（<https://github.com/mattpocock/skills>）；索引 `sources/README.md`
- 沉淀页面：`wiki/entities/mattpocock-skills.md`
- 交叉更新：`wiki/entities/superpowers-obra.md`、`wiki/entities/caveman.md`、`wiki/references/llm-wiki-karpathy.md`
- 派生再生成：`make ci-preflight`

## [2026-05-20] structural | wiki/entities/amp-mjlab.md — 补充 play.py（run_play）详细 Mermaid 流程图

- 页面：`wiki/entities/amp-mjlab.md` 在「训练与回放」下新增 `run_play` 流程图（CLI → play 环境覆盖 → checkpoint 加载 → AMPOnPolicyRunner 推理 → ONNX 导出 → Viewer 主循环），源码依据 [ImChong/AMP_mjlab](https://github.com/ImChong/AMP_mjlab) `scripts/play.py`
- 派生再生成：`make ci-preflight`

## [2026-05-19] structural | wiki/concepts/character-animation-vs-robotics.md — V22 P1「角色化人形边界澄清」落地

- 新增页面：`wiki/concepts/character-animation-vs-robotics.md`（角色动画 vs 机器人控制：六维张力矩阵 + 五案例切片 + 决策矩阵 + Mermaid「角色端→桥接层→机器人端」流程）
- 交叉更新：`wiki/methods/disney-olaf-character-robot.md`、`wiki/entities/botlab-motioncanvas.md`、`wiki/entities/roboto-origin.md`、`wiki/entities/xue-bin-peng.md`、`wiki/concepts/motion-retargeting.md`、`wiki/concepts/reward-design.md` 的 `related` 与「关联页面 / 与其他页面的关系」加入新入口
- 清单同步：`docs/checklists/tech-stack-next-phase-checklist-v22.md` 勾选 P1「角色化人形（Character Humanoid）边界澄清」并补实现摘要
- 派生再生成：`make ci-preflight` 同步 `exports/`、`docs/exports/`、`docs/search-index.json`、`docs/sitemap.xml`、`README.md`、`docs/index.html`、`index.md`、`exports/lint-report.md`（注：lint 报告中 17 条「陈旧页面」均为今日早些 ingest 留下的预存量，与本次改动无直接关系）

## [2026-05-19] ingest | sources/repos/caveman.md — JuliusBrussee/caveman 入库并沉淀 wiki

- 原始资料：`sources/repos/caveman.md`（<https://github.com/JuliusBrussee/caveman>）；索引 `sources/README.md`
- 沉淀页面：`wiki/entities/caveman.md`
- 交叉更新：`wiki/entities/superpowers-obra.md`、`wiki/entities/hermes-agent.md`、`wiki/entities/agent-reach.md`
- 派生再生成：`make ci-preflight` 同步 `exports/`、`docs/exports/`、`docs/search-index.json`、`docs/sitemap.xml`、`README.md`、`docs/index.html`、`index.md` 等

## [2026-05-19] ingest | sources/papers/gentlehumanoid_upper_body_compliance.md、sources/repos/axellwppr_motion_tracking.md — GentleHumanoid / motion_tracking 入库并沉淀 wiki

- 原始资料：`sources/papers/gentlehumanoid_upper_body_compliance.md`（[arXiv:2511.04679](https://arxiv.org/abs/2511.04679)）、`sources/sites/gentle-humanoid-axell-top.md`、`sources/sites/motion-tracking-axell-top.md`、`sources/repos/axellwppr_motion_tracking.md`（<https://github.com/Axellwppr/motion_tracking>）；索引 `sources/README.md`
- 沉淀页面：`wiki/methods/gentlehumanoid-motion-tracking.md`、`wiki/entities/axellwppr-motion-tracking.md`
- 交叉更新：`wiki/concepts/whole-body-control.md`、`wiki/concepts/contact-dynamics.md`、`wiki/concepts/impedance-control.md`、`wiki/overview/humanoid-rl-motion-control-body-system-stack.md`、`wiki/methods/sonic-motion-tracking.md`
- 派生再生成：`make ci-preflight` 同步 `exports/`、`docs/exports/`、`docs/search-index.json`、`docs/sitemap.xml`、`README.md`、`docs/index.html`、`index.md` 等

## [2026-05-19] structural | wiki/overview/motor-drive-firmware-bus-protocols.md — 电机驱动器底软通信协议总览（种类与优缺点）

- 原始资料：`sources/courses/motor_drive_firmware_bus_protocols.md`（CiA CANopen/CiA402、DroneCAN、MIT 紧凑帧与厂商私有等索引）
- 沉淀页面：`wiki/overview/motor-drive-firmware-bus-protocols.md`（物理层 × 应用层 × 控制语义三层；协议族优缺点表；常见组合与选型 Mermaid）
- 交叉更新：`wiki/concepts/can-bus-protocol.md`、`wiki/concepts/ethercat-protocol.md`、`wiki/comparisons/can-vs-ethercat-joint-bus.md`
- 派生再生成：`make ci-preflight`

## [2026-05-19] ingest | sources/sites/cia_can_*、sources/courses/uart_rs485_serial_embedded.md — CiA CAN/CAN FD/CANopen/DroneCAN 与 UART·RS485 一手资料入库

- 原始资料：`sources/sites/cia_can_knowledge_can_classic_and_hs.md`、`sources/sites/cia_can_fd_basic_idea.md`、`sources/sites/cia_canopen_overview.md`、`sources/sites/cia_dronecan_uavcan.md`、`sources/courses/uart_rs485_serial_embedded.md`（CiA [CAN knowledge](https://www.can-cia.org/can-knowledge/)、[DroneCAN](http://dronecan.org/)、TI SLLA383 / Wikipedia UART）；索引 `sources/README.md`
- 沉淀页面：`wiki/concepts/can-bus-protocol.md`、`wiki/concepts/can-fd.md`、`wiki/concepts/uart-serial-communication.md`、`wiki/comparisons/can-vs-ethercat-joint-bus.md`
- 交叉更新：`wiki/concepts/ethercat-protocol.md`、`wiki/concepts/processor-in-the-loop-sim2real.md`、`wiki/formalizations/control-loop-latency-modeling.md`、`wiki/queries/real-time-control-middleware-guide.md`
- 派生再生成：`make ci-preflight` 同步 `exports/`、`docs/exports/`、`docs/search-index.json`、`docs/sitemap.xml`、`README.md`、`docs/index.html`、`index.md` 等

## [2026-05-19] ingest | sources/repos/newton-physics.md、sources/sites/nvidia-newton-physics.md、sources/sites/newton-physics-docs-overview.md — Newton Physics 引擎（NVIDIA / DeepMind / Disney，Warp + MuJoCo Warp）入库

- 原始资料：`sources/repos/newton-physics.md`（<https://github.com/newton-physics/newton>）、`sources/sites/nvidia-newton-physics.md`（<https://developer.nvidia.com/newton-physics>）、`sources/sites/newton-physics-docs-overview.md`（<https://newton-physics.github.io/newton/stable/guide/overview.html>）；索引 `sources/README.md`
- 沉淀页面：`wiki/entities/newton-physics.md`
- 交叉更新：`wiki/entities/mujoco.md`、`wiki/entities/mjlab.md`、`wiki/entities/isaac-gym-isaac-lab.md`、`wiki/queries/simulator-selection-guide.md`
- 派生再生成：`make ci-preflight` 同步 `exports/`、`docs/exports/`、`docs/search-index.json`、`docs/sitemap.xml`、`README.md`、`docs/index.html`、`index.md` 等

## [2026-05-19] ingest | sources/blogs/wechat_embodied_ai_lab_daji_semantic_body_interface.md、sources/papers/daji_arxiv_2605_14417.md — DAJI 预期关节意图（微信精读 / arXiv:2605.14417）入库

- 原始资料：`sources/blogs/wechat_embodied_ai_lab_daji_semantic_body_interface.md`（<https://mp.weixin.qq.com/s/u1ZUaFGYRKXxMcS7-V_2WA>）、`sources/papers/daji_arxiv_2605_14417.md`、`sources/sites/daji-hxxxz0-github-io.md`、`sources/repos/hxxxz0_daji.md`；索引 `sources/README.md`
- 沉淀页面：`wiki/entities/paper-daji-anticipatory-joint-intent.md`
- 交叉更新：`wiki/methods/vla.md`、`wiki/tasks/loco-manipulation.md`、`wiki/overview/humanoid-rl-motion-control-body-system-stack.md`
- 派生再生成：`make ci-preflight` 同步 `exports/`、`docs/exports/`、`docs/search-index.json`、`docs/sitemap.xml`、`README.md`、`docs/index.html`、`index.md` 等

## [2026-05-19] ingest | sources/blogs/wechat_embodied_ai_lab_robot_world_model_training_loop.md、sources/papers/wm_robot_survey_arxiv_2605_00080.md、sources/sites/wm-robot-survey-ntumars.md — 安装 Agent Reach 抓取微信公众号；机器人世界模型综述（arXiv:2605.00080）入库

- 工具：已安装 [Panniantong/Agent-Reach](https://github.com/Panniantong/Agent-Reach) v1.4.0（`pip install` 可编辑包 + `agent-reach install --channels=wechat`）；微信正文经 `wechat-article-for-ai`（Camoufox），Jina Reader 对该 URL 返回 CAPTCHA
- 原始资料：`sources/blogs/wechat_embodied_ai_lab_robot_world_model_training_loop.md`（<https://mp.weixin.qq.com/s/0edW0GhwtyNc5nF6RDIfuw>）、`sources/papers/wm_robot_survey_arxiv_2605_00080.md`、`sources/sites/wm-robot-survey-ntumars.md`；索引 `sources/README.md`
- 沉淀页面：`wiki/overview/robot-world-models-training-loop-taxonomy.md`
- 交叉更新：`wiki/methods/generative-world-models.md`、`wiki/concepts/world-action-models.md`、`wiki/methods/vla.md`、`wiki/overview/humanoid-rl-motion-control-body-system-stack.md`、`wiki/entities/agent-reach.md`、`sources/repos/panniantong_agent_reach.md`
- 派生再生成：`make ci-preflight` 同步 `exports/`、`docs/exports/`、`docs/search-index.json`、`docs/sitemap.xml`、`README.md`、`docs/index.html`、`index.md` 等

## [2026-05-19] ingest | sources/repos/nousresearch_hermes_agent.md、sources/sites/hermes-agent-nousresearch-docs.md — Hermes Agent（NousResearch）仓库与官方文档入库

- 原始资料：`sources/repos/nousresearch_hermes_agent.md`、`sources/sites/hermes-agent-nousresearch-docs.md`（GitHub <https://github.com/NousResearch/hermes-agent>、产品页 <https://hermes-agent.nousresearch.com/>、文档 <https://hermes-agent.nousresearch.com/docs>）；索引 `sources/README.md`
- 沉淀页面：`wiki/entities/hermes-agent.md`
- 交叉更新：`wiki/entities/superpowers-obra.md`、`wiki/entities/agent-reach.md`、`index.md`（Entities 目录补 Hermes Agent 条目）
- 派生再生成：`make ci-preflight` 同步 `exports/`、`docs/exports/`、`docs/search-index.json`、`docs/sitemap.xml`、`README.md`、`docs/index.html`、`index.md` 等

## [2026-05-18] structural | wiki/comparisons/gmr-vs-nmr-vs-reactor.md — V22 P1 动作重定向知识链 (3/3)：新增 GMR / NMR / ReActor 三方对比页

- 沉淀页面：`wiki/comparisons/gmr-vs-nmr-vs-reactor.md`（一句话定义 + 12 维核心对比表 + Mermaid 三路数据流并排图 + 三方适用场景 + 5 类常见误判 + 决策矩阵；强调「误差修补发生位置」（下游 / 离线 / 在线）作为核心选型轴；显式标注 NMR 仍以 GMR 为 CEPR 初值、三者实际常串联而非互斥）
- 交叉更新：`wiki/concepts/motion-retargeting.md`、`wiki/concepts/motion-retargeting-pipeline.md`、`wiki/formalizations/motion-retargeting-objective.md`、`wiki/methods/motion-retargeting-gmr.md`、`wiki/methods/neural-motion-retargeting-nmr.md`、`wiki/methods/reactor-physics-aware-motion-retargeting.md`（关联页面区块回链本对比页）、`index.md`（Wiki Comparisons 目录插入 GMR vs NMR vs ReActor 条目）、`docs/checklists/tech-stack-next-phase-checklist-v22.md`（P1 第 3 项打勾，含实现摘要；至此 V22 P1「动作重定向知识链 (+3)」3/3 全部完成）
- 派生再生成：保持当前状态，待后续 ingest 或 P2 推进时统一 `make ci-preflight` 同步 `exports/`、`docs/exports/`、`docs/search-index.json`、`docs/sitemap.xml`、`README.md`、`docs/index.html`、`index.md` 等

## [2026-05-18] ingest | sources/papers/capvector_arxiv_2605_10903.md、sources/sites/capvector-github-io.md、sources/repos/openhelix_team_capvector.md — CapVector（参数空间 capability vector + 正交正则标准 SFT）arXiv:2605.10903 入库

- 原始资料：`sources/papers/capvector_arxiv_2605_10903.md`、`sources/sites/capvector-github-io.md`、`sources/repos/openhelix_team_capvector.md`（PDF <https://arxiv.org/pdf/2605.10903>、项目页 <https://capvector.github.io/>、代码 <https://github.com/OpenHelix-Team/CapVector>）；索引 `sources/README.md`
- 沉淀页面：`wiki/entities/paper-capvector-capability-vectors-vla.md`
- 交叉更新：`wiki/methods/vla.md`、`index.md`（Entities 目录补 CapVector 条目）
- 派生再生成：`make ci-preflight` 同步 `exports/`、`docs/exports/`、`docs/search-index.json`、`docs/sitemap.xml`、`README.md`、`docs/index.html`、`index.md` 等

## [2026-05-18] ingest | sources/papers/physforge_arxiv_2605_05163.md — PhysForge（VLM 分层物理蓝图 + KVI 扩散、PhysDB）arXiv:2605.05163 入库

- 原始资料：`sources/papers/physforge_arxiv_2605_05163.md`（PDF <https://arxiv.org/pdf/2605.05163>、项目页 <https://hku-mmlab.github.io/PhysForge/>）；索引 `sources/README.md`
- 沉淀页面：`wiki/entities/paper-physforge-physics-grounded-3d-assets.md`
- 交叉更新：`wiki/entities/articraft.md`、`wiki/entities/robotwin.md`、`index.md`（Entities 目录补 PhysForge 条目）
- 派生再生成：`make ci-preflight` 同步 `exports/`、`docs/exports/`、`docs/search-index.json`、`docs/sitemap.xml`、`README.md`、`docs/index.html`、`index.md` 等

## [2026-05-18] ingest | sources/repos/horizon_robotics_holomotion.md、sources/papers/holomotion_arxiv_2605_15336.md — 地平线 HoloMotion-1（混合语料 + 稀疏 MoE Transformer + 序列级 PPO）官方资料入库

- 原始资料：`sources/repos/horizon_robotics_holomotion.md`、`sources/papers/holomotion_arxiv_2605_15336.md`（GitHub / 项目主页 / arXiv:2605.15336 / Hugging Face / Docker Hub）；索引 `sources/README.md`
- 沉淀页面：`wiki/entities/holomotion.md`
- 交叉更新：`wiki/concepts/foundation-policy.md`、`wiki/methods/sonic-motion-tracking.md`、`wiki/entities/paper-behavior-foundation-model-humanoid.md`、`index.md`（Entities 目录补 `holomotion` 条目）
- 派生再生成：`make ci-preflight` 同步 `exports/`、`docs/exports/`、`docs/search-index.json`、`docs/sitemap.xml`、`README.md`、`docs/index.html`、`index.md` 等

## [2026-05-18] ingest | sources/blogs/wechat_jixie_robot_open_source_treasury_issue02_10_robots.md — 微信公众号「机械Robot」开源宝库第02期：10 个机器人/平台实体 + 策展 overview

- 原始资料：`sources/blogs/wechat_jixie_robot_open_source_treasury_issue02_10_robots.md`；索引 `sources/README.md`
- 沉淀页面：`wiki/overview/robot-open-source-wechat-issue02-curator.md`；`wiki/entities/pollen-reachy2.md`、`wiki/entities/poppy-project-robots.md`、`wiki/entities/inmoov-humanoid.md`、`wiki/entities/stanford-doggo-and-pupper.md`、`wiki/entities/elephantrobotics-mycobot-320.md`、`wiki/entities/elephantrobotics-myagv.md`、`wiki/entities/tidybot2.md`、`wiki/entities/kinova-gen3.md`、`wiki/entities/franka-research-3.md`、`wiki/entities/parol6-source-robotics.md`
- 交叉更新：`wiki/overview/robot-open-source-wechat-issue01-curator.md`、`wiki/entities/open-source-humanoid-hardware.md`、`wiki/entities/humanoid-robot.md`
- 派生再生成：`make ci-preflight` 同步 `exports/`、`docs/exports/`、`docs/search-index.json`、`docs/sitemap.xml`、`README.md`、`docs/index.html`、`index.md` 等

## [2026-05-18] ingest | sources/blogs/wechat_jixie_robot_open_source_treasury_issue01_10_robots.md — 微信公众号「机械Robot」开源宝库第01期：10 个机器人/平台独立实体节点 + 策展 overview

- 原始资料：`sources/blogs/wechat_jixie_robot_open_source_treasury_issue01_10_robots.md`；索引 `sources/README.md`
- 沉淀页面：`wiki/overview/robot-open-source-wechat-issue01-curator.md`；`wiki/entities/fourier-grx-n1.md`、`wiki/entities/agibot-lingxi-x1.md`、`wiki/entities/tienkung-humanoid-open-source.md`、`wiki/entities/odri-solo-and-bolt.md`、`wiki/entities/berkeley-humanoid-lite.md`、`wiki/entities/orca-hand.md`、`wiki/entities/turtlebot3.md`、`wiki/entities/robotis-open-manipulator-line.md`、`wiki/entities/robotis-op3.md`、`wiki/entities/robotis-thormang3.md`
- 交叉更新：`wiki/entities/open-source-humanoid-hardware.md`、`wiki/entities/humanoid-robot.md`
- 派生再生成：`make ci-preflight` 同步 `exports/`、`docs/exports/`、`docs/search-index.json`、`docs/sitemap.xml`、`README.md`、`docs/index.html`、`index.md` 等

## [2026-05-18] ingest | sources/repos/mujoco-mjx.md、sources/repos/brax.md、sources/papers/brax_arxiv_2106_13281.md、sources/sites/mujoco-mjx-readthedocs.md — MuJoCo MJX 与 Brax 官方仓/文档/论文入库；新增实体页并交叉更新 MuJoCo / dm_control / 选型指南 / LIFT

- 原始资料：`sources/repos/mujoco-mjx.md`、`sources/repos/brax.md`、`sources/papers/brax_arxiv_2106_13281.md`、`sources/sites/mujoco-mjx-readthedocs.md`；索引 `sources/README.md`
- 沉淀页面：`wiki/entities/mujoco-mjx.md`、`wiki/entities/brax.md`
- 交叉更新：`wiki/entities/mujoco.md`、`wiki/entities/dm-control.md`、`wiki/queries/simulator-selection-guide.md`、`wiki/comparisons/mujoco-vs-isaac-sim.md`、`wiki/comparisons/mujoco-vs-isaac-lab.md`、`wiki/entities/lift-humanoid.md`、`index.md`（重点入口 + Page Catalog）
- 派生再生成：`make ci-preflight` 同步 `exports/`、`docs/exports/`、`docs/search-index.json`、`docs/sitemap.xml`、`README.md`、`docs/index.html` 等

## [2026-05-18] ingest | sources/papers/barkour_arxiv_2305_14654.md、sources/blogs/google-research-barkour-quadruped-agility-2023-05-26.md、sources/repos/google_deepmind_barkour_robot.md、sources/repos/mujoco_menagerie_google_barkour_models.md — Barkour 四足敏捷基准与开源生态入库

- 原始资料：arXiv:2305.14654、Google Research 博客（2023-05-26）、[`google-deepmind/barkour_robot`](https://github.com/google-deepmind/barkour_robot)、[`mujoco_menagerie` 下 `google_barkour_v0` / `google_barkour_vb`](https://github.com/google-deepmind/mujoco_menagerie/tree/main/google_barkour_v0)；OnShape 以 README 中 **gdm.onshape.com / deepmind.onshape.com** 文档链接为准（`cad.onshape.com` 为产品首页）
- 沉淀页面：`wiki/entities/paper-barkour-quadruped-agility-benchmark.md`
- 交叉更新：`wiki/entities/quadruped-robot.md`、`wiki/entities/mujoco.md`、`wiki/tasks/locomotion.md`、`references/papers/locomotion-rl.md`、`sources/README.md`
- 派生再生成：`make ci-preflight` 同步 `exports/`、`docs/exports/`、`docs/search-index.json`、`docs/sitemap.xml`、`README.md`、`docs/index.html`、`index.md` 等

## [2026-05-18] structural | wiki/overview/humanoid-rl-motion-control-body-system-stack.md、wiki/concepts/planetary-roller-screw-humanoid-leg-actuation.md — 参考来源补充微信公众号原文外链

- 更新页面：`wiki/overview/humanoid-rl-motion-control-body-system-stack.md`（`https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA`）、`wiki/concepts/planetary-roller-screw-humanoid-leg-actuation.md`（`https://mp.weixin.qq.com/s/webqJRQJREZdABw8bdl68w`）；保留仓库内 `sources/` 归档链接
- 派生再生成：`make ci-preflight` 同步 `exports/`、`docs/exports/`、`docs/search-index.json`、`docs/sitemap.xml`、`README.md`、`docs/index.html`、`index.md` 等

## [2026-05-18] ingest | sources/sites/wechat-embodied-ai-lab-humanoid-rl-motion-survey-2026-05-18.md、wiki/overview/humanoid-rl-motion-control-body-system-stack.md — 具身智能研究室 42 篇 humanoid RL 运动控制综述入库；新增「身体系统栈」视角 overview 页

- 原始资料：`sources/sites/wechat-embodied-ai-lab-humanoid-rl-motion-survey-2026-05-18.md`（公众号长文，Camoufox 抓取，约 4.5w 字）
- 沉淀页面：`wiki/overview/humanoid-rl-motion-control-body-system-stack.md`（提炼作者的 8 层身体系统栈 + 6 个研究判断；把已有 wiki 实体页 DeepMimic / SONIC / BeyondMimic / Any2Track / AMS / GMR / NMR / DoorMan / VIRAL / BFM / GR00T-WBC / ULTRA 按层挂接；明确「单页未升格论文」候选清单）
- 交叉关联：`wiki/overview/humanoid-motion-control-know-how.md`、`wiki/tasks/humanoid-locomotion.md`、`wiki/tasks/loco-manipulation.md`、`wiki/tasks/ultra-survey.md`、`wiki/tasks/balance-recovery.md` 与多篇 `wiki/methods` / `wiki/entities` 页面通过 related 区块互链
- 派生再生成：`make ci-preflight` 同步 `exports/`、`docs/exports/`、`docs/search-index.json`、`docs/sitemap.xml`、`README.md`、`docs/index.html`、`index.md` 等

## [2026-05-18] ingest | sources/blogs/wechat_zanezhang_tesla_optimus_leg_planetary_roller_screw.md — 微信公众号：Optimus 腿部行星滚柱丝杠解读入库；新增 wiki/concepts/planetary-roller-screw-humanoid-leg-actuation.md；交叉更新 humanoid-robot、locomotion、sources/README.md

- 原始资料：`sources/blogs/wechat_zanezhang_tesla_optimus_leg_planetary_roller_screw.md`；索引 `sources/README.md`
- 沉淀页面：`wiki/concepts/planetary-roller-screw-humanoid-leg-actuation.md`（PRS 原理、反转式布置、连杆映射、与旋转关节权衡、Mermaid 主干流程）
- 交叉更新：`wiki/entities/humanoid-robot.md`（Optimus 备注与参考来源）、`wiki/tasks/locomotion.md`（关联系统与方法回链）
- 派生再生成：`make ci-preflight` 同步 `exports/`、`docs/exports/`、`docs/search-index.json`、`docs/sitemap.xml`、`README.md`、`docs/index.html`、`index.md` 等

## [2026-05-18] ingest | sources/repos/panniantong_agent_reach.md — Panniantong/Agent-Reach 入库；新增 wiki/entities/agent-reach.md；交叉更新 superpowers-obra、index.md、sources/README.md

- 原始资料：`sources/repos/panniantong_agent_reach.md`；索引 `sources/README.md`
- 沉淀页面：`wiki/entities/agent-reach.md`（编码代理互联网接入脚手架：可插拔渠道、`doctor`、上游 CLI/MCP 与本地凭据主张；配流程图）
- 交叉更新：`wiki/entities/superpowers-obra.md`（关联页面回链）、`index.md`（重点入口 + Page Catalog）、`sources/README.md`
- 派生再生成：`make ci-preflight` 同步 `exports/`、`docs/exports/`、`docs/search-index.json`、`docs/sitemap.xml`、`README.md`、`docs/index.html` 等

## [2026-05-18] ingest | sources/papers/bfm_humanoid_arxiv_2509_13780.md、sources/sites/bfm4humanoid-github-io.md — BFM（Behavior Foundation Model，arXiv:2509.13780）入库；新增 wiki/entities/paper-behavior-foundation-model-humanoid.md；交叉更新 foundation-policy / whole-body-control；并对齐 CLAUDE.md 的 PR 截图流程

- 原始资料：`sources/papers/bfm_humanoid_arxiv_2509_13780.md`、`sources/sites/bfm4humanoid-github-io.md`；索引 `sources/README.md`
- 沉淀页面：`wiki/entities/paper-behavior-foundation-model-humanoid.md`（CVAE + 位级二值掩码 + 在线蒸馏 + 潜空间组合 + 残差解码器新技能的人形 WBC 基础模型，配 Mermaid 流程图与 Table III/IV 量化对照）
- 交叉更新：`wiki/concepts/foundation-policy.md`（新增 BFM 子项与回链）、`wiki/concepts/whole-body-control.md`（Learning-based & Generative WBC 段补 BFM；关联页面与 sources 互链）、`index.md`（新增 Entity 条目）、`sources/README.md`
- 规范同步：`CLAUDE.md` 新增「Claude Code Agent：PR 与验证截图」一节，与 `docs/checklists/cloud-agent-pr-workflow.md` 对齐
- 派生再生成：`make ci-preflight` 同步 `exports/`、`docs/exports/`、`docs/search-index.json`、`docs/sitemap.xml`、`README.md`、`docs/index.html`、`index.md` 等

## [2026-05-17] structural | wiki/formalizations/motion-retargeting-objective.md — V22 P1 动作重定向知识链 (2/3)：新增重定向目标函数形式化页

- 沉淀页面：`wiki/formalizations/motion-retargeting-objective.md`（通用目标函数 $\mathcal{L}^{\text{pose}}+\mathcal{L}^{\text{ee}}+\mathcal{L}^{\text{bal}}+\mathcal{L}^{\text{lim}}+\mathcal{L}^{\text{smooth}}$；姿态相似/末端接触/平衡/限位/平滑五大罚项的数学定义；GMR/DeepMimic/ReActor/NMR/SPIDER 五种工程退化形态对照）
- 交叉更新：`wiki/concepts/motion-retargeting.md`、`wiki/concepts/motion-retargeting-pipeline.md`（关联页面区块回链本页）、`docs/checklists/tech-stack-next-phase-checklist-v22.md`（P1 第 2 项打勾，含实现摘要）
- 派生再生成：`make ci-preflight` 同步 `exports/`、`docs/exports/`、`docs/search-index.json`、`docs/sitemap.xml`、`README.md`、`index.md` 等

## [2026-05-17] ingest | sources/repos/lingbot-map.md、sources/papers/lingbot_map_arxiv_2604_14141.md、sources/sites/lingbot-map-technology-robbant.md、sources/sites/businesswire-lingbot-map-2026-04-16.md — LingBot-Map 论文/站点/通稿入库；勘误 byant 误链；扩充 wiki/methods/lingbot-map.md

- 原始资料：`sources/repos/lingbot-map.md`、`sources/papers/lingbot_map_arxiv_2604_14141.md`、`sources/sites/lingbot-map-technology-robbant.md`、`sources/sites/businesswire-lingbot-map-2026-04-16.md`；索引 `sources/README.md`
- 沉淀页面：`wiki/methods/lingbot-map.md`（流程图、工程局限、参考来源；官方仓库为 [Robbyant/lingbot-map](https://github.com/Robbyant/lingbot-map)）
- 交叉更新：`index.md`（由 `make ci-preflight` 同步 `exports/`、`docs/exports/`、`docs/search-index.json`、`docs/sitemap.xml`、`README.md` 等）

## [2026-05-17] ingest | sources/papers/humannet_table1_benchmark_corpora.md — HumanNet Table1 代表性人视频/行为语料官方入口索引；新增对比页 wiki/comparisons/humannet-table1-human-video-corpora.md

- 原始资料：`sources/papers/humannet_table1_benchmark_corpora.md`；索引 `sources/README.md`
- 沉淀页面：`wiki/comparisons/humannet-table1-human-video-corpora.md`
- 交叉更新：`wiki/entities/humannet.md`、`wiki/methods/vla.md`、`index.md`（由 `make ci-preflight` 同步 `exports/`、`docs/exports/`、`docs/search-index.json`、`docs/sitemap.xml`、`README.md` 等）

## [2026-05-17] ingest | sources/papers/egoscale_arxiv_2602_16710.md、sources/sites/nvidia-research-egoscale.md — EgoScale（arXiv:2602.16710）与 NVIDIA GEAR 项目页入库

- 原始资料：`sources/papers/egoscale_arxiv_2602_16710.md`、`sources/sites/nvidia-research-egoscale.md`；索引 `sources/README.md`
- 沉淀页面：`wiki/methods/egoscale.md`
- 交叉更新：`wiki/methods/vla.md`、`wiki/methods/imitation-learning.md`、`wiki/tasks/manipulation.md`、`wiki/entities/humannet.md`、`wiki/concepts/embodied-scaling-laws.md`、`references/papers/imitation-learning.md`、`index.md`（由 `make ci-preflight` 同步）

## [2026-05-17] ingest | sources/papers/mimic_video_arxiv_2512_15692.md、sources/sites/mimic-video-github-io.md、sources/repos/lucidrains_mimic_video.md — mimic-video（VAM，arXiv:2512.15692）入库

- 原始资料：`sources/papers/mimic_video_arxiv_2512_15692.md`、`sources/sites/mimic-video-github-io.md`、`sources/repos/lucidrains_mimic_video.md`；索引 `sources/README.md`
- 沉淀页面：`wiki/methods/mimic-video.md`
- 交叉更新：`wiki/methods/vla.md`、`wiki/methods/imitation-learning.md`、`wiki/concepts/video-as-simulation.md`、`wiki/methods/generative-world-models.md`、`wiki/tasks/manipulation.md`、`references/papers/imitation-learning.md`、`index.md`（由 `make ci-preflight` 同步）

## [2026-05-17] ingest | sources/papers/crisp_real2sim_iclr2026.md、sources/sites/crisp-real2sim-project-github-io.md、sources/repos/crisp_real2sim_repo.md — CRISP（ICLR 2026）Real2Sim 入库

- 原始资料：`sources/papers/crisp_real2sim_iclr2026.md`、`sources/sites/crisp-real2sim-project-github-io.md`、`sources/repos/crisp_real2sim_repo.md`；索引 `sources/README.md`
- 沉淀页面：`wiki/methods/crisp-real2sim.md`
- 交叉更新：`wiki/concepts/sim2real.md`、`wiki/entities/gs-playground.md`、`references/papers/sim2real.md`、`index.md`（新增方法页目录条目）、`README.md` / `exports/` / `docs/exports/` 等（`make ci-preflight`）

## [2026-05-17] ingest | sources/papers/egm_arxiv_2512_19043.md、sources/blogs/egm_themoonlight_literature_review_2512_19043.md — EGM（arXiv:2512.19043）与第三方导读入库

- 原始资料：`sources/papers/egm_arxiv_2512_19043.md`、`sources/blogs/egm_themoonlight_literature_review_2512_19043.md`；索引 `sources/README.md`
- 沉淀页面：`wiki/methods/egm-efficient-general-mimic.md`
- 交叉更新：`wiki/methods/beyondmimic.md`、`wiki/methods/sonic-motion-tracking.md`、`index.md`（由 `make ci-preflight` 同步）

## [2026-05-17] ingest | sources/papers/e_sds_arxiv_2512_16446.md、sds_quadruped_arxiv_2410_11571.md、repos/rpl_cs_ucl_sds.md — E-SDS（arXiv:2512.16446）与 SDS 前序资料入库

- 原始资料：`sources/papers/e_sds_arxiv_2512_16446.md`、`sources/papers/sds_quadruped_arxiv_2410_11571.md`、`sources/repos/rpl_cs_ucl_sds.md`；索引 `sources/README.md`
- 沉淀页面：`wiki/entities/paper-e-sds-environment-aware-humanoid-locomotion-rl.md`
- 交叉更新：`wiki/tasks/locomotion.md`、`wiki/methods/reinforcement-learning.md`、`references/papers/locomotion-rl.md`、`index.md`（由 `make ci-preflight` 同步）

## [2026-05-17] ingest | sources/repos/unitree_ros.md、sources/repos/unitree_ros_to_real.md — 官方 ROS1+Gazebo 与真机 ROS 桥入库

- 原始资料：`sources/repos/unitree_ros.md`、`sources/repos/unitree_ros_to_real.md`；索引 `sources/README.md`
- 沉淀页面：`wiki/entities/unitree-ros.md`
- 交叉更新：`wiki/entities/unitree.md`、`wiki/entities/unitree-rl-mjlab.md`、`wiki/tasks/locomotion.md`、`index.md`（由 `make ci-preflight` 同步）

## [2026-05-17] ingest | sources/papers/urdd_beyond_urdf_arxiv_2512_23135.md、Apollo-Lab-Yale 多仓与 Pages — URDD（arXiv:2512.23135）入库

- 原始资料：`sources/papers/urdd_beyond_urdf_arxiv_2512_23135.md`、`sources/repos/apollo-lab-yale-apollo-resources.md`、`sources/repos/apollo-lab-yale-apollo-rust.md`、`sources/repos/apollo-lab-yale-apollo-three-engine.md`、`sources/repos/apollo-lab-yale-apollo-py.md`、`sources/sites/apollo-lab-yale-apollo-resources-github-io.md`；索引 `sources/README.md`
- 沉淀页面：`wiki/entities/paper-urdd-universal-robot-description-directory.md`
- 交叉更新：`wiki/entities/robot-viewer.md`、`wiki/entities/urdf-studio.md`、`wiki/entities/mujoco.md`、`sources/urdf.md`、`index.md`（由 `make ci-preflight` 同步）

## [2026-05-17] ingest | sources/papers/hy_motion_arxiv_2512_23464.md、sources/repos/tencent_hunyuan_hy_motion_1_0.md — HY-Motion 1.0（arXiv:2512.23464）与官方仓入库

- 原始资料：`sources/papers/hy_motion_arxiv_2512_23464.md`、`sources/repos/tencent_hunyuan_hy_motion_1_0.md`；索引 `sources/README.md`
- 沉淀页面：`wiki/methods/hy-motion-1.md`
- 交叉更新：`wiki/methods/diffusion-motion-generation.md`、`wiki/methods/genmo.md`、`wiki/entities/awesome-text-to-motion-zilize.md`、`index.md`（由 `make ci-preflight` 同步）

## [2026-05-17] ingest | sources/papers/spider_scalable_physics_informed_dexterous_retargeting.md、sources/sites/jc-bao-spider-project-github-io.md、sources/repos/jc-bao-spider-project.md — SPIDER（arXiv:2511.09484）与项目页入库

- 原始资料：`sources/papers/spider_scalable_physics_informed_dexterous_retargeting.md`、`sources/sites/jc-bao-spider-project-github-io.md`、`sources/repos/jc-bao-spider-project.md`；索引 `sources/README.md`
- 沉淀页面：`wiki/methods/spider-physics-informed-dexterous-retargeting.md`
- 交叉更新：`wiki/concepts/motion-retargeting.md`、`wiki/concepts/motion-retargeting-pipeline.md`、`wiki/methods/motion-retargeting-gmr.md`、`index.md`（由 `make ci-preflight` 同步）

## [2026-05-17] ingest | sources/papers/dwm_arxiv_2512_17907.md、sources/sites/snuvclab-dwm-github-io.md、sources/repos/snuvclab_dwm.md — DWM（Dexterous World Models）项目页与论文入库

- 原始资料：`sources/papers/dwm_arxiv_2512_17907.md`、`sources/sites/snuvclab-dwm-github-io.md`、`sources/repos/snuvclab_dwm.md`；索引 `sources/README.md`
- 沉淀页面：`wiki/methods/dwm.md`
- 交叉更新：`wiki/methods/generative-world-models.md`、`wiki/concepts/video-as-simulation.md`、`wiki/tasks/manipulation.md`、`index.md`（由 `make ci-preflight` 同步）

## [2026-05-17] ingest | sources/repos/cyoahs-robot-motion-editor.md、stanford-tml-robot-keyframe-kit.md、project-instinct-robot-motion-editor.md — 机器人关键帧/运动编辑三仓库入库

- 原始资料：`sources/repos/cyoahs-robot-motion-editor.md`、`sources/repos/stanford-tml-robot-keyframe-kit.md`、`sources/repos/project-instinct-robot-motion-editor.md`；索引 `sources/README.md`
- 沉淀页面：`wiki/entities/robot-motion-keyframe-editors.md`
- 交叉更新：`wiki/entities/project-instinct.md`、`wiki/entities/mujoco.md`、`wiki/concepts/motion-retargeting-pipeline.md`、`wiki/tasks/manipulation.md`、`index.md`（由 `make ci-preflight` 同步）

## [2026-05-17] ingest | sources/papers/faststair_arxiv_2601_10365.md、sources/sites/npcliu-faststair-github-io.md — FastStair（arXiv:2601.10365）入库

- 原始资料：`sources/papers/faststair_arxiv_2601_10365.md`、`sources/sites/npcliu-faststair-github-io.md`；索引 `sources/README.md`
- 沉淀页面：`wiki/entities/paper-faststair-humanoid-stair-ascent.md`
- 交叉更新：`wiki/tasks/locomotion.md`、`index.md`（由 `make ci-preflight` 同步）

## [2026-05-17] ingest | sources/repos/obra-superpowers.md、sources/blogs/fsck_superpowers_announcement_2025-10-09.md — Superpowers（obra）编码代理技能与交付工作流入库

- 原始资料：`sources/repos/obra-superpowers.md`、`sources/blogs/fsck_superpowers_announcement_2025-10-09.md`；索引 `sources/README.md`
- 沉淀页面：`wiki/entities/superpowers-obra.md`
- 交叉更新：`wiki/references/llm-wiki-karpathy.md`、`index.md`（由 `make ci-preflight` 同步）

## [2026-05-17] ingest | InterPrior（arXiv:2602.06035）与 sirui-xu.github.io/InterPrior 站点入库

- 原始资料：`sources/papers/interprior_arxiv_2602_06035.md`、`sources/sites/sirui-xu-interprior-github-io.md`；索引 `sources/README.md`
- 沉淀页面：`wiki/entities/paper-interprior.md`
- 交叉更新：`wiki/tasks/loco-manipulation.md`、`wiki/methods/imitation-learning.md`、`wiki/methods/reinforcement-learning.md`、`index.md`（由 `make ci-preflight` 同步）

## [2026-05-17] ingest | sources/repos/leggedrobotics_robotic_world_model.md、leggedrobotics_robotic_world_model_lite.md — ETH RSL 的 RWM / RWM-U（Isaac Lab 扩展与 Lite 离线仓）

- 原始资料：`sources/repos/leggedrobotics_robotic_world_model.md`、`sources/repos/leggedrobotics_robotic_world_model_lite.md`
- 沉淀页面：`wiki/entities/robotic-world-model-eth-rsl.md`
- 交叉更新：`wiki/methods/model-based-rl.md`、`wiki/methods/generative-world-models.md`、`wiki/entities/isaac-gym-isaac-lab.md`、`sources/README.md`、`index.md`

## [2026-05-17] ingest | sources/repos/zilize-awesome-text-to-motion.md — 文本驱动人体运动生成 Awesome 列表入库；新增 wiki/entities/awesome-text-to-motion-zilize.md

- 原始资料：`sources/repos/zilize-awesome-text-to-motion.md`（<https://github.com/Zilize/awesome-text-to-motion>，项目页 <https://zilize.github.io/awesome-text-to-motion/>）
- 沉淀页面：`wiki/entities/awesome-text-to-motion-zilize.md`
- 交叉更新：`wiki/methods/diffusion-motion-generation.md`、`wiki/methods/genmo.md`、`sources/README.md`、`index.md`（由 `make ci-preflight` 同步目录统计）

## [2026-05-17] ingest | sources/repos/sage-sim2real-actuator-gap.md — SAGE（Sim2Real Actuator Gap Estimator）与 README 要点归档

- 原始资料：`sources/repos/sage-sim2real-actuator-gap.md`（<https://github.com/isaac-sim2real/sage> 及 README 公开信息；关联 AMASS、Human2Humanoid、OSMO 工作流线索）
- 沉淀页面：`wiki/entities/sage-sim2real-actuator-gap-estimator.md`
- 交叉更新：`wiki/concepts/sim2real.md`、`wiki/concepts/domain-randomization.md`、`wiki/roadmaps/humanoid-control-roadmap.md`、`wiki/queries/sim2real-gap-reduction.md`、`wiki/methods/actuator-network.md`、`sources/README.md`、`index.md`

## [2026-05-17] ingest | LIFT（arXiv:2601.21363）与 lift-humanoid.github.io、bigai-ai/LIFT-humanoid 归档入库

- 原始资料：`sources/papers/lift_humanoid_arxiv_2601_21363.md`、`sources/sites/lift-humanoid-github-io.md`、`sources/repos/bigai-lift-humanoid.md`
- 沉淀页面：`wiki/entities/lift-humanoid.md`
- 交叉更新：`wiki/tasks/locomotion.md`、`wiki/concepts/sim2real.md`、`wiki/methods/reinforcement-learning.md`、`wiki/methods/model-based-rl.md`、`wiki/queries/rl-algorithm-selection.md`、`sources/README.md`、`index.md`

## [2026-05-17] ingest | DoorMan（arXiv:2512.01061）与 doorman-humanoid 站点、GR00T-VisualSim2Real 归档入库

- 原始资料：`sources/papers/doorman_opening_sim2real_arxiv_2512_01061.md`、`sources/sites/doorman-humanoid-github-io.md`；更新 `sources/repos/gr00t_visual_sim2real.md`
- 沉淀页面：`wiki/entities/paper-doorman-opening-sim2real-door.md`
- 交叉更新：`wiki/entities/gr00t-visual-sim2real.md`、`wiki/entities/paper-viral-humanoid-visual-sim2real.md`、`wiki/tasks/loco-manipulation.md`、`sources/README.md`、`index.md`

## [2026-05-16] structural | wiki/concepts/motion-retargeting-pipeline.md — V22 P1 动作重定向知识链 (1/3)：新增重定向流水线概念页（8 阶段端到端：源归一 → 骨架/DoF 映射 → 体型缩放 → IK/QP → 硬约束与平滑 → 物理可行性筛选 → 可选物理修补 → 离线/在线产物落地）

- 沉淀页面：`wiki/concepts/motion-retargeting-pipeline.md`（含 Mermaid 流程总览、三种工程化形态对比表、常见失败模式表与下游接口契约）
- 交叉更新：`wiki/concepts/motion-retargeting.md`（关联页面回链流水线页）、`docs/checklists/tech-stack-next-phase-checklist-v22.md`（P1 第 1 项打勾、状态置 `[~]`）
- 派生再生成：`make ci-preflight` → 327 Nodes / 2197 Edges / Coverage 325/325；`scripts/lint_wiki.py` 通过（29 条信息型预警不阻塞）

## [2026-05-16] ingest | sources/sites/worldlabs-ai.md — World Labs 官网与 Marble/Spark 归档；新增 wiki/entities/world-labs.md；交叉更新 wiki/methods/generative-world-models.md、wiki/entities/gs-playground.md、index.md

- 原始资料：`sources/sites/worldlabs-ai.md`（<https://www.worldlabs.ai/> 及 About、Marble、Marble Labs、Spark 技术博客等公开链接归档）
- 沉淀页面：`wiki/entities/world-labs.md`
- 交叉更新：`wiki/methods/generative-world-models.md`、`wiki/entities/gs-playground.md`、`index.md`

## [2026-05-16] ingest | sources/papers/pelican_unified_uei_arxiv_2605_15153.md — Pelican-Unified 1.0（UEI，arXiv:2605.15153）技术报告入库；新增 wiki/methods/pelican-unified-1.md；交叉更新 wiki/methods/vla.md、wiki/concepts/world-action-models.md、wiki/methods/being-h07.md、index.md

- 原始资料：`sources/papers/pelican_unified_uei_arxiv_2605_15153.md`（PDF <https://arxiv.org/pdf/2605.15153>；关联 WAM 综述、Awesome-WAM、StarVLA、Being-H0.7 归档链接）
- 沉淀页面：`wiki/methods/pelican-unified-1.md`
- 交叉更新：`wiki/methods/vla.md`、`wiki/concepts/world-action-models.md`、`wiki/methods/being-h07.md`、`index.md`

## [2026-05-16] ingest | sources/papers/ewmbench.md, sources/repos/ewmbench.md, sources/sites/agibot-world.md — EWMBench（arXiv:2505.09694）与 AgibotTech 仓库及 Agibot-World 关联站点入库；新增 wiki/entities/ewmbench.md；交叉更新 wiki/methods/generative-world-models.md、wiki/concepts/video-as-simulation.md

- 原始资料：`sources/papers/ewmbench.md`、`sources/repos/ewmbench.md`、`sources/sites/agibot-world.md`
- 沉淀页面：`wiki/entities/ewmbench.md`
- 交叉更新：`wiki/methods/generative-world-models.md`、`wiki/concepts/video-as-simulation.md`、`index.md`

## [2026-05-16] structural | wiki/methods/genmo.md, docs/main.js, docs/style.css — GENMO 详情页 Mermaid：节点标签改为引号形式并修正边语法，避免 `~`/括号在方括号语法中被误解析；详情页迷你知识地图支持 d3.zoom 平移（禁用滚轮缩放以免抢页面滚动）

- 沉淀页面：`wiki/methods/genmo.md`（`detail.html?id=wiki-methods-genmo` 流程图可渲染）
- 前端：`docs/main.js`（`renderDetailMiniMap`）、`docs/style.css`（grab 光标 / `touch-action`）

## [2026-05-16] ingest | sources/papers/genmo.md, sources/repos/genmo.md — GENMO/GEM（arXiv:2505.01425v1，ICCV 2025 Highlight）论文与 NVlabs/GENMO 仓库入库；扩充 wiki/methods/genmo.md（dual-mode 训练 / multi-text 注入 / NVIDIA 人形栈 / Mermaid 流程图），交叉补 wiki/methods/diffusion-motion-generation.md 参考来源

- 原始资料：`sources/papers/genmo.md`（arXiv abs + HTML v1 摘录）、`sources/repos/genmo.md`（NVlabs/GENMO 仓库、GEM-SMPL HuggingFace 权重、README 时间线）
- 沉淀页面：`wiki/methods/genmo.md`（双模式训练 / multi-text 注入 / NVIDIA 人形栈关联 / Mermaid 流程总览）
- 交叉更新：`wiki/methods/diffusion-motion-generation.md`（参考来源与人体运动域代表实现链接）

## [2026-05-16] ingest | sources/repos/nvlabs-curobo.md — CuRobo / cuRoboV2（curobo.org、GitHub、arXiv:2310.17274、2603.05493）入库；沉淀 wiki/entities/curobo.md

- 原始资料：`sources/repos/nvlabs-curobo.md`（归档 https://curobo.org/、https://github.com/NVlabs/curobo、https://arxiv.org/abs/2310.17274、https://arxiv.org/abs/2603.05493 及 Isaac ROS cuMotion 等关联链接）
- 沉淀页面：`wiki/entities/curobo.md`

## [2026-05-16] ingest | sources/sites/articraft3d-github-io.md, sources/repos/mattzh72-articraft.md — Articraft 项目页与代码仓；新增 wiki/entities/articraft.md；互链 text-to-cad、URDF-Studio

- 原始资料：`sources/sites/articraft3d-github-io.md`、`sources/repos/mattzh72-articraft.md`
- 沉淀页面：`wiki/entities/articraft.md`（交叉更新 `wiki/concepts/text-to-cad.md`、`wiki/entities/urdf-studio.md`、`index.md`）

## [2026-05-16] ingest | sources/repos/jackhan-mujoco-walke3-simulation.md 等六仓 — JackHan-Sdu WalkE3 / HumanoidE3 / FEAP 工具链入库；新增 wiki/entities/jackhan-walke3-e3-ecosystem.md 及六条子实体页（各含 Mermaid 流程图）

- 原始资料：`sources/repos/jackhan-mujoco-walke3-simulation.md`、`sources/repos/jackhan-walke3-dataset.md`、`sources/repos/jackhan-walke3-controller.md`、`sources/repos/jackhan-algorithm-template-for-developer.md`、`sources/repos/jackhan-feap-mujoco-deployment.md`、`sources/repos/jackhan-feapvision-mujoco-deployment.md`
- 沉淀页面：`wiki/entities/jackhan-walke3-e3-ecosystem.md`、`wiki/entities/jackhan-mujoco-walke3-simulation.md`、`wiki/entities/jackhan-walke3-dataset.md`、`wiki/entities/jackhan-walke3-controller.md`、`wiki/entities/jackhan-yobotics-e3-algorithm-template.md`、`wiki/entities/jackhan-feap-mujoco-deployment.md`、`wiki/entities/jackhan-feapvision-mujoco-deployment.md`

## [2026-05-16] structural | wiki/queries/humanoid-rl-cookbook.md, docs/style.css — Humanoid RL Cookbook：「TL;DR」标题改为「快速决策路径」；任务列表内普通子 bullet 增加左缩进避免与 checkbox 同列重叠

## [2026-05-16] structural | wiki/queries/humanoid-rl-cookbook.md, docs/main.js, docs/style.css — Humanoid RL Cookbook：TL;DR 改为 Mermaid 流程图；详情页支持 `- [ ]` / `- [x]` 任务列表复选框渲染

- `wiki/queries/humanoid-rl-cookbook.md`：用 `flowchart TB` 呈现硬件 / 仿真 / 训练路径分支；更新 `updated` 元数据。
- `docs/main.js`：`renderMarkdownContent` 在无序列表解析中识别 GFM task list，输出 `<input type="checkbox" disabled>`；`flushList` 统一列表项结构。
- `docs/style.css`：`.contains-task-list`、`.task-list-item` 与 label 布局，避免与默认列表圆点重叠。

## [2026-05-15] structural | scripts/lint_wiki.py — V22 P0 方法-Query 闭环 Lint：新增 `methods_without_practitioner_query` 检查 + `INFO_ONLY_KEYS` 信息型分类机制

- `scripts/lint_wiki.py`：新增 `_check_methods_without_practitioner_query()`，阈值 `METHOD_PRACTITIONER_INBOUND_THRESHOLD=3`（即 ≥ 4 个 wiki 入链，自链已排除），若入链来源中无任何 `wiki/queries/*` 或 `wiki/comparisons/*` 命中，则标记为"待落地"信息型预警。
- 失败计数机制：抽出 `_failing_total()` / `_info_total()` 辅助函数，新增 `INFO_ONLY_KEYS = {"missing_pages", "methods_without_practitioner_query"}` 让 main 退出码只统计硬错误，避免首次落地即破坏 CI（baseline 28 项以 💡 信息型展示）。
- 测试：新增 `tests/test_lint_wiki_practitioner_query.py` 6 个用例（高入链无 query 命中、queries 命中、comparisons 命中、阈值边界、自链排除、INFO_ONLY 不计失败 total），`PYTHONPATH=scripts pytest --no-cov` 91/91 通过；`ruff check`、`ruff format --check`、`mypy scripts/lint_wiki.py` 均通过；`scripts/lint_wiki.py` 退出码 0，报告含 28 条信息型预警。
- 落地基线：当前 28 条预警覆盖 exoactor / sonic-motion-tracking / amp-reward / beyondmimic / motion-retargeting-gmr / humanoid-transformer-touch-dreaming / deepmimic / auto-labeling-pipelines / pi07-policy / π0-policy 等高频热点，将在 V22 P1（动作重定向）/ P2（抓取）落地 queries 与 comparisons 时同步消减。
- 清单：`docs/checklists/tech-stack-next-phase-checklist-v22.md` P0 "方法-Query 闭环 Lint" 三项全部勾选。

## [2026-05-15] ingest | sources/sites/amass-dataset.md, sources/repos/ubisoft-laforge-animation-dataset.md, sources/sites/mixamo.md — AMASS / LaFAN1 / Mixamo 入库；新增 wiki/entities/amass.md、wiki/entities/lafan1-dataset.md、wiki/entities/mixamo.md；互链 motion-retargeting、wbc-fsm、ProtoMotions

- 原始资料：`sources/sites/amass-dataset.md`、`sources/repos/ubisoft-laforge-animation-dataset.md`、`sources/sites/mixamo.md`
- 沉淀页面：`wiki/entities/amass.md`、`wiki/entities/lafan1-dataset.md`、`wiki/entities/mixamo.md`

## [2026-05-15] ingest | sources/repos/xiaomi-robotics-0.md — 小米 Xiaomi-Robotics-0（官网/GitHub/arXiv:2602.12684）；新增 wiki/entities/xiaomi-robotics-0.md；互链 VLA、Action Chunking

- 原始资料：`sources/repos/xiaomi-robotics-0.md`
- 沉淀页面：`wiki/entities/xiaomi-robotics-0.md`

## [2026-05-15] ingest | sources/papers/viral-humanoid-visual-sim2real.md — VIRAL arXiv:2511.15200：新增 wiki/entities/paper-viral-humanoid-visual-sim2real.md；GR00T-VisualSim2Real 增补 Mermaid 流程图与论文专档互链

- 原始资料：`sources/papers/viral-humanoid-visual-sim2real.md`
- 沉淀页面：`wiki/entities/paper-viral-humanoid-visual-sim2real.md`

## [2026-05-15] ingest | sources/papers/pi07.md — Physical Intelligence π₀.₇（arXiv:2604.15483 + pi.website/blog/pi07），新增 wiki/methods/pi07-policy.md，交叉更新 wiki/methods/π0-policy.md、wiki/methods/vla.md、wiki/concepts/foundation-policy.md

- 原始资料：`sources/papers/pi07.md`
- 沉淀页面：`wiki/methods/pi07-policy.md`

## [2026-05-15] structural | wiki/methods/actuator-network.md, sources/papers/system_identification.md — 补充 ActuatorNet 一手论文 DOI / arXiv、RSS 2018 PDF、Isaac Lab 执行器代码链接；修正错误题名；`system_identification` ingest 映射增加 `actuator-network`

## [2026-05-15] structural | wiki/methods/actuator-network.md — 增加离线辨识训练与仿真步内闭环的 Mermaid 流程图；更新页面 `updated` 元数据

## [2026-05-15] structural | sources/repos/protomotions.md, wiki/entities/protomotions.md — 对照 NVlabs README 与 protomotions.github.io 扩充原始摘录与实体页：能力表、数据—训练—部署与模块化 MDP 双 Mermaid、局限说明；互链 ADD / DeepMimic / xue-bin-peng

## [2026-05-15] ingest | sources/repos/robot-io-rio.md — 收录 RIO 官网、GitHub、Netlify 文档与 arXiv:2605.11564；新建 `wiki/entities/robot-io-rio.md` 并互链 `wiki/methods/vla.md`、`wiki/entities/lerobot.md`、`wiki/tasks/teleoperation.md`

- 原始资料：`sources/repos/robot-io-rio.md`（注明文档站 `/arXiv` 路径 404，以根路径为准）
- 沉淀页面：`wiki/entities/robot-io-rio.md`

## [2026-05-14] structural | scripts/search_wiki_core.py — V22 P0 缩写/别名归一化检索：新增 `WIKI_ABBREVIATIONS`（16 条：WBC/VLA/IL/RL/MPC/PPO/SAC/HQP/CBF/CLF/BC/IK/FK/LIP/ZMP/TSID）与 `expand_query_aliases()`，缩写 ↔ 全称双向展开后同时喂给 BM25 分词与行匹配，并以"缩写归一化：已展开为 …"提示挂到 `semantic_notice`；新增 5 个单测（21/21 通过），`eval_search_quality.py` 36/37 与基线一致

## [2026-05-14] structural | roadmap & docs/main.js — 主路线 ASCII 图换 mermaid + skip-to 改交互按钮 + 4 条 depth 加 mermaid pipeline

- `docs/main.js`：`renderMarkdownContent` 新增原始 HTML block 透传（div/details/summary/section/aside/figure/figcaption），让 markdown 可嵌入交互组件。
- `roadmap/motion-control.md`：L−1 的 4 盒子全景与 L4.0 的方法链 ASCII 图换 mermaid flowchart；资深读者 skip-to 矩阵改为 7 个 grid-style 按钮（auto-fit minmax 260px）。
- `roadmap/depth-*.md`：4 条独立纵深路线页顶部各加专属 mermaid Stage pipeline（不同 stroke 配色区分主题）。

## [2026-05-14] structural | roadmap & docs/roadmap.html — 主路线重构 + 网页正文渲染升级

将 `roadmap.html?id=roadmap-motion-control` 从「阶段树 + 互链 Top10」升级为「mini-map + 完整 markdown 正文 + TOC 侧栏」，复用 detail 页 markdown 渲染管线（`docs/main.js` 新增 `renderRoadmapMarkdownBody`，挂载点 `#roadmapContent` / `#roadmapTocList`）。同时把四条 if-goal 纵深从 `roadmap/motion-control.md` 拆为 `roadmap/depth-{rl-locomotion,imitation-learning,safe-control,contact-manipulation}.md` 四个独立 roadmap 页，主线只留摘要 + 衔接表；旧 `roadmap-if-goal-*` id 由跳锚点改为跳新 roadmap 页。

主路线内容侧从 L0–L6 扩为 L−1 → L7 单主线：
- L−1 序言：感知/规划/控制/执行四盒子全景、为什么人形为主载体、三种读者读法、Modern Robotics 章节映射、25+ 必备术语速查
- 每个 L 加场景隐喻 + 上一层的局限说明，让"为什么这一层存在"显式可见
- L1 / L2 增加里程碑分步说明（L1 三步 / L2 两步），L4 新增 L4.0 桥段（模型粒度 × 控制频率二维表 + 方法链 ASCII 流程图），L4 后新增方法谱系对比表（PID→LQR→LIP→DCM→Centroidal→TO→MPC→TSID/WBC→PPO→BC→DAgger→AMP→Diffusion）
- 每个 L 加 3-4 道自测题，L−1 新增资深读者 skip-to 矩阵
- L7 出口：感知 / 规划 / 操作 / 系统软件栈扫盲 + 2024–2026 前沿地图（Humanoid FM、VLA、World Model、Teacher-Student、AMP、End-to-End、Loco-Manipulation、Tactile）
- Modern Robotics 入口去重：L0–L4 五处 "本阶段入口" 不再重复引入，统一在 L−1 介绍

## [2026-05-14] structural | scripts/generate_link_graph.py — V22 P0 社区粒度二级拆分：保留 Girvan-Newman 一级检测（`PRIMARY_COMMUNITY_CAP=8`），新增纯 Python Louvain（`resolution=1.15`，Reichardt-Bornholdt modularity）对占比 > 40% 的巨型社区二级拆分；`MAX_COMMUNITIES` 提升至 16 容纳子社区。`exports/graph-stats.json`：`largest_community_ratio` 0.651 → 0.138，`community_quality_warning` true → false，Locomotion 拆出 WBC/RL/MPC/IL/Sim2Real/Isaac Gym/Humanoid/Unitree G1 等子社区

## [2026-05-14] structural | wiki — 扩充 `wiki/tasks/locomotion.md`：补充任务边界、闭环 Mermaid、子问题地图、方法选型与工程落地检查

## [2026-05-14] structural | docs/checklists — 新增 `scripts/screenshot_site_detail.sh`（timeout 包裹 headless Chrome、随机 remote-debugging-port），并更新 cloud-agent-pr-workflow 的截图步骤说明

## [2026-05-14] ingest | sources/repos/sonic-humanoid-motion-tracking.md — 对照 NVIDIA GEAR-SONIC 官网扩充 SONIC 原始资料；更新 wiki/methods/sonic-motion-tracking（规模表、VLA/遥操作/规划器接口、双 Mermaid）；互链 foundation-policy、vla、teleoperation 与 tairan-he 项目 URL

## [2026-05-13] structural | references — 扩充 `references/repos/retarget-tools.md`：分组外链、互链 Motion Retargeting / GMR / NMR / ReActor / IL，消除详情页「空正文」观感

## [2026-05-13] structural | roadmap — 将四条 `learning-paths/if-goal-*.md` 全文并入 `roadmap/motion-control.md`「可选纵深」；站点仅保留单一 `roadmap_page`；旧 `roadmap.html?id=roadmap-if-goal-*` 重定向至 `detail.html?id=roadmap-motion-control#...`

## [2026-05-12] structural | wiki — 扩充 `wiki/entities/mimickit.md`：运行时与算法选型 Mermaid 流程图、架构表与局限说明；补强 related 与参考来源

## [2026-05-12] structural | wiki — 扩充 `wiki/methods/motion-retargeting-gmr.md`：命名辨析、双 Mermaid 流程图、开源工程要点与 arXiv；互链 SONIC

## [2026-05-12] ingest | sources/notes/humanoid-parallel-joint-kinematics.md — 新增人形并联关节解算资料索引与 wiki 概念页，互链 Asimov RSU 踝、灵巧手闭链与 Armature

## [2026-05-12] chore | docs/checklists — 明确 PR 验证截图为站点 detail 页而非 GitHub PR 页

## [2026-05-12] structural | docs/checklists — 新增 Cloud Agent PR 推送与验证截图流程说明（cloud-agent-pr-workflow.md），AGENTS 增加指针并忽略 .cursor-artifacts

> V21 里程碑追踪（Current）：
> - P0：基础架构防腐（拼写纠错、公式规范检查、图谱数据精简） ✅ 完成
> - P1：触觉与力觉闭环（Contact Wrench Cone 页面、触觉基础页面、融合范式对比）
> - P2：通信与延迟规范（硬件通信链路设计、延迟来源分析）
> - P3：V21 质量回归验证

> V20 里程碑追踪：
> - P0：检索系统增强（多词组合查询、Tag 提权、摘要匹配优化） ✅ 完成
> - P1：依赖可视化（Graph 页面渲染社区划分、重力物理引擎集成） ✅ 完成
> - P2：基础页补全（新增 14 个基础概念/算法/任务页面并保证链接闭环） ✅ 完成
> - P3：结构瘦身（合并零散页、清理无效引用） ✅ 完成

> V14 里程碑追踪：
> - P1：扩展 `CANONICAL_FACTS` 到 50 条，覆盖 PPO on-policy、CLF/CBF、VLA、contact-rich manipulation、Isaac Lab 并行训练等事实
> - P2：加深 manipulation / contact-rich / terrain / bimanual / sensor-fusion / behavior-cloning / VLA 等薄弱页面
> - P3：规划新增 4 个高价值 Query 页（domain-randomization / clf-cbf / vla-low-level / contact-rich manipulation）
> - P4：lint 增加孤儿节点计数检测，搜索回归扩展到 26 条
> - P5：补齐安全控制与接触操作学习路径，扩展 overview / index / README 入口，并保持日志 append-only 更新

## [2026-05-11] structural | wiki — 新增四足机器人实体页 `wiki/entities/quadruped-robot.md` 并与 humanoid、市面平台纵览、ANYmal、Boston Dynamics、Unitree、locomotion 任务互链

## [2026-05-11] structural | wiki — 新增轮足四足（四轮足）概念页 `wiki/concepts/wheel-legged-quadruped.md`，互链 Hybrid Locomotion、robot_lab、locomotion、Unitree

## [2026-05-11] chore | roadmap — 删除两条空分支学习路径

- 移除 `roadmap/learning-paths/if-goal-generalist.md`、`if-goal-whole-body-control.md`（仅占位列表，无 L 阶段正文）。
- 更新 `roadmap/motion-control.md`、`roadmap/README.md`、`STRUCTURE_v1.md` 引用；`make ci-preflight` 同步索引与导出。

## [2026-05-08] style | docs | 技术路线页 `roadmap.html` emoji 统一为 🚀

- favicon、顶栏站点标题由 🛣️/🧭 改为 🚀，与首页「技术路线指南」CTA 一致；主题切换仍为 ☀️/🌙。
- 同步 `docs/checklists/frontend-optimization-v1.md`。

## [2026-04-20] fix | v14-execution | P0 搜索回归修复：numpy 延迟导入

- V14 P0 完成：`scripts/search_wiki.py` 移除 module 顶部 `import numpy as np`，改为在 `load_vector_resources` / `encode_query_vector` / `search` 内部使用 numpy 的分支做延迟导入
- 修复前：`python3 scripts/search_wiki.py PPO` 直接 `ModuleNotFoundError: No module named 'numpy'`，回归测试 0/26
- 修复后：`python3 scripts/eval_search_quality.py` 通过率 **26/26 (100%)**，BM25 路径不再依赖 numpy
- 无新增依赖、无行为变更；向量搜索分支在 numpy 可用时保持原逻辑

## [2026-04-21] structural | V14 执行清单完整交付 (新增 11 页 + 深度扩充 3 页 + 脚本优化)

## [2026-04-21] structural | V15 执行清单初始化

## [2026-04-21] structural | V15 执行清单完整交付 (聚焦操作社区与软件栈实体)

## [2026-04-21] structural | V16 执行清单初始化 (具身大模型深度化 + 灵巧操作补完)

## [2026-04-21] structural | V16 执行清单完整交付 (具身大模型深度化 + 灵巧操作补完)

## [2026-04-21] structural | V17 执行清单完整交付 & 初始化 V18 (具身数据流与交互闭环)

## [2026-04-21] structural | V20 执行清单完整交付 & 初始化 V21 (具身触觉专题 + 硬件通信链路形式化)

## [2026-04-22] ingest | sources/repos/fusion2urdf.md, sources/repos/marathongo.md — 接入新仓库资料并更新全站索引

## [2026-04-23] ingest | robot_lab (Repo) & CLAW (Blog) — 接入 IsaacLab 扩展框架与 G1 合成数据管线并更新全站索引

## [2026-04-24] ingest | sources/papers/policy_optimization.md — 补充 BRRL/BPO（Bounded Ratio RL）论文、项目页与代码仓库，并更新 Policy Optimization 方法页参考来源

- 新增来源条目：`Bounded Ratio Reinforcement Learning (Ao et al., 2026)`（arXiv / project page / GitHub）
- 更新 `wiki/methods/policy-optimization.md`：新增“BRRL / BPO（2026）”进展说明与参考来源
- 关联强化：将 BRRL 映射到 policy-optimization / reinforcement-learning / ppo-vs-sac / locomotion

## [2026-04-24] structural | V21 P1 推进 | wiki/formalizations/contact-wrench-cone.md

- V21 P1「触觉与力觉闭环」首项：新建 `wiki/formalizations/contact-wrench-cone.md`，把 Friction Cone 从 3D 点接触力推广到 6D 面接触力旋量（CWC / CWS / GWS）
- 涵盖 V-/H- 表示、ZMP/CoP 几何解读、多接触 Minkowski 和 与 Grasp Wrench Space 的统一视角，附最小 Python 骨架与方法局限性
- 回链：`wiki/formalizations/friction-cone.md`、`wiki/concepts/tactile-sensing.md`、`wiki/concepts/contact-dynamics.md` 的关联页面区块新增 CWC 入链
- V21 checklist 对应条目已勾选；follow-up：原同日 log 条目曾在合并 main 时按“冲突以 main 为准”规则被覆盖，此处以 append-only 方式补回

## [2026-04-24] feat | v21-execution | P0 智能拼写纠错（编辑距离）

- V21 P0 第二项推进：`scripts/search_wiki.py` 集成 Levenshtein 编辑距离算法，当查询无结果时自动推荐最接近的 Tag 或 标题
- 新增 `levenshtein_distance` / `collect_known_terms` / `suggest_terms` 三个辅助函数；阈值取 `max(2, ceil(len(query)/2))`，按距离升序返回 Top-5
- `print_results` 增加“您是否想搜索：”分区；`--json` 输出在有 notice 或 suggestions 时切换为 `{notice, suggestions, results}` 字典形式
- `search()` 返回签名保持 `(results, notice)` 不变，`scripts/eval_search_quality.py` 与 `scripts/debug_search.py` 调用方零侵入
- 验证：`python3 scripts/eval_search_quality.py` 通过率 **37/37 (100%)**；`python3 scripts/search_wiki.py "lokomotion" --json` 正确给出 `locomotion`（距离 1）建议
- V21 checklist 对应条目已勾选

## [2026-04-25] lint | 同步 sources/papers/policy_optimization.md 的最新进展（BRRL/BPO 2026）至全站 8 个相关 wiki 页面，消除陈旧预警并更新索引

## [2026-04-25] ingest | sources/repos/roboto_origin.md — 新增 Roboto Origin 及其五个官方模块仓库与文档资料归档，并沉淀 wiki/entities/roboto-origin.md

## [2026-04-25] ingest | sources/repos/atom01_hardware.md, sources/repos/atom01_deploy.md, sources/repos/atom01_train.md, sources/repos/atom01_description.md, sources/repos/atom01_firmware.md — 接入 Roboparty Atom01 五个官方模块仓库原始资料

## [2026-04-25] ingest | wiki/entities/roboto-origin.md — 回链 Atom01 五个 sources 条目并重新导出图谱索引，确保 sources 节点可见

## [2026-04-25] ingest | wiki/entities/atom01-*.md — 新增 Atom01 五个实体页并同步导出图谱（181 nodes / 1012 edges）

## [2026-04-25] feat | v21-execution | P0 自动化背链一致性 Lint（公式变量物理含义检测）

- V21 P0 第三项推进：`scripts/lint_wiki.py` 新增 `formalization_unexplained_vars` 检查；从 `wiki/formalizations/*.md` 的 `$$...$$` 显示公式中抽取单字母拉丁大写变量，逐一验证正文是否给出物理含义解释
- 启发式定义匹配：列表条目冒号、表格行、动词解释（是/为/表示/代表/denote）、其中/where 子句、等式或集合定义（=、\in、\succeq、\succ、\equiv、\triangleq）；并用 `(?![A-Za-z_(])` 排除函数调用形式（如 `R(s,a,s')`）以避免误报
- 修复 6 个被新规则命中的页面，补齐变量物理含义说明：
  - `wiki/formalizations/bellman-equation.md`：Q-learning 更新中的 $R$ 标量
  - `wiki/formalizations/control-lyapunov-function.md`：LQR-CLF 关系中的 $A, B$
  - `wiki/formalizations/ekf.md`：补全 $A, B, C, Q, R, P, K, I$ 整套矩阵物理含义
  - `wiki/formalizations/hjb.md`：LQR 特例段补 $A, B, Q, R, T$
  - `wiki/formalizations/lqr.md`：线性系统模型段补 $A, B, x, u$
  - `wiki/formalizations/tsid-formulation.md`：浮动基座动力学段补执行选择矩阵 $S$
- 验证：`make lint` 0 errors（所有检查通过）；`python3 scripts/eval_search_quality.py` 通过率 **37/37 (100%)**
- V21 checklist 对应条目已勾选

## [2026-04-26] structural | roadmap | 收敛为单一主路线并补齐阶段链接

- 将 `roadmap/route-a-motion-control.md` 从“路线A”调整为唯一主路线，明确 RL/IL/WBC/安全控制/接触操作是目标分支而非并列主路线。
- 删除 `roadmap/route-b-fullstack.md`，避免 roadmap 页面出现 A/B 两套主线。
- 为 L0-L6 阶段补充仓库内概念、方法、形式化、实体、query 页面链接，并同步更新索引与导出。

## [2026-04-26] structural | roadmap | 主路线 ID 去除 A/B 命名

- 将主路线文件从 `roadmap/route-a-motion-control.md` 重命名为 `roadmap/motion-control.md`。
- 将导出 ID 从 `roadmap-route-a-motion-control` 更新为 `roadmap-motion-control`，并同步 README、index、learning paths、wiki 回链与前端入口。
- 在 `docs/main.js` 保留旧 ID 到新 ID 的兼容映射，避免旧 URL 立即失效。

## [2026-04-26] feat | v21-execution | P0 图谱导出数据精简

- V21 P0 第三项收尾：精简 `exports/link-graph.json` 节点字段，去除每个节点上重复的 `community_label`（与 `communities` 数组中的 `label` 完全冗余）。
- `scripts/generate_link_graph.py` 仅在 `communities` 数组保留 `label`，节点端只输出 `community` id；`docs/graph.html` 的 tooltip 与侧栏改为通过 `communityLabelMap.get(d.community)` 查表。
- 体积变化：`exports/link-graph.json` 168 KB → 159 KB（5.7% 缩减），行数 5551 → 5370；`make lint` 与 `check_export_quality.py` 全通过。
- V21 checklist P0 全部完成，进入 P1 触觉与力觉闭环专题阶段。

## [2026-04-27] ingest | sources/repos/lingbot-map.md — 接入 LingBot-Map 流式 3D 重建基础模型方法页

## [2026-04-27] ingest | sources/repos/booster-robocup-demo.md — 接入 Booster Robotics RoboCup 演示框架

## [2026-04-27] ingest | 接入 htwk-gym 与 HumanoidSoccer (PAiD) 足球技能学习框架

## [2026-04-27] ingest | 补全飞书 Wiki 相关的运动控制进阶主题 (BeyondMimic, Any2Track, HAIC, AMP 等)

## [2026-04-27] feat | v21-execution | P1 触觉专题：视触觉融合 (Visuo-Tactile Fusion) 概念页

- V21 P1 触觉学习知识链推进第一项：新增 `wiki/concepts/visuo-tactile-fusion.md`，聚焦“接触瞬间如何在视觉全局与触觉局部之间动态切换权重”。
- 内容覆盖：视觉/触觉互补维度、阶段切换/软门控/注意力级三种融合范式、接触瞬间难点、训练数据采集要点、与现有相关页面的边界澄清（区分 `sensor-fusion`、`multimodal-fusion-tricks`）。
- 同步在 `tactile-sensing`、`contact-rich-manipulation`、`contact-wrench-cone`、`multimodal-fusion-tricks`、`tactile-feedback-in-rl` 中加入回链，避免新页成为孤儿；并刷新 `index.md` Page Catalog。
- 在 `docs/checklists/tech-stack-next-phase-checklist-v21.md` 中将该项打勾。

## [2026-04-28] ingest | 消化 ETH Zurich 关于扩散模型运动生成的论文 (Unitree G1 实时部署)

## [2026-04-28] ingest | 消化 awesome-humanoid-robot-learning 仓库，更新 Loco-Manipulation Wiki 页

## [2026-04-28] ingest | 接入 MimicKit 仓库及其核心算法系列 (DeepMimic, AMP, AWR, ASE, LCP, ADD, SMP)

## [2026-04-28] ingest | 深度更新 SMP (arXiv:2512.03028v3) 技术细节，补充 SDS/ESM/GSI 架构及 Unitree G1 真机验证

## [2026-04-29] ingest | sources/papers/humanoid_touch_dream.md — 消化 HTD / Touch Dreaming 论文并更新触觉增强人形移动操作知识节点

## [2026-04-29] structural | 更新待办列表与主页统计数据 (205 nodes, 1195 edges)

## [2026-04-29] ingest | sources/repos/embodied-ai-guide.md — 接入具身智能全栈百科并补全 RoboTwin/SAPIEN/ALOHA 实体

## [2026-04-29] ingest | sources/repos/xbotics-embodied-guide.md — 接入 Xbotics 工程指南，补全 LeRobot/Genesis 实体与数据飞轮概念

## [2026-04-29] ingest | sources/papers/zest.md — 接入 Boston Dynamics 跨形态技能迁移框架 ZEST

## [2026-04-29] ingest | sources/papers/sumo.md — 接入 RAI Institute 全身移动操作框架 Sumo (MPC-over-RL)

## [2026-04-29] ingest | sources/repos/{robot-explorer,robot-viewer,urdf-studio}.md — 接入 Web 端机器人可视化与设计工具链

## [2026-04-29] ingest | sources/repos/motphys-motrix.md — 接入 Motphys 高性能仿真平台 Motrix (Rust 后端, MJCF 兼容)

## [2026-04-30] ingest | 接入 NVIDIA MotionBricks (GR00T WBC 核心生成式框架)

## [2026-04-30] ingest | 接入 HKUST Switch 框架 (敏捷技能切换与图搜索)

## [2026-05-01] ingest | sources/papers/universal_skeleton.md — Universal Skeleton HOVL 异构骨架开放词汇动作识别，沉淀到 wiki/methods/skeleton-action-recognition.md

## [2026-05-01] ingest | 修复 10 个低健康度 Wiki 节点，并优化 Imitation Learning 与 Contact Dynamics 的搜索关键词以通过质量回归测试。

## [2026-05-01] ingest | sources/books/udl_book.md — 接入 Understanding Deep Learning 教材基础理论，新增 Deep Learning Foundations 和 Generative Foundations wiki 页面

## [2026-05-01] ingest | 扩充 AMP_mjlab 训练命令、指标分析及部署指南

## [2026-05-01] ingest | sources/repos/amp_mjlab.md — 扩充 AMP_mjlab 训练曲线、ONNX 导出与部署检查要点

## [2026-05-01] structural | roadmap/motion-control.md — 在主路线补入 Modern Robotics 章节映射与练习

## [2026-05-01] ingest | sources/repos/wbc_fsm.md — ccrpRepo wbc_fsm：Unitree G1 C++ WBC+FSM 部署框架，新增 wiki/entities/wbc-fsm.md

## [2026-05-01] ingest | sources/repos/gr00t_visual_sim2real.md — NVIDIA GR00T-VisualSim2Real：VIRAL（arXiv:2511.15200）+ DoorMan（arXiv:2512.01061）双 CVPR 2026，PPO Teacher + DAgger Student RGB 蒸馏 Sim2Real，Unitree G1 零样本迁移，新增 wiki/entities/gr00t-visual-sim2real.md，图谱 232 节点

## [2026-05-02] structural | wiki: 把 29 个页面里的 102 处 [[wikilink]] 全部迁移为标准 Markdown [text](path)，含 10 处隐蔽断链（指向 sources/references/roadmap 的非 wiki 路径），脚本 scripts/migrate_wikilinks.py，图谱 233 节点

## [2026-05-03] ingest | sources/papers/multi-gait-learning.md — Multi-Gait Learning for Humanoid Robots Using Reinforcement Learning with Selective Adversarial Motion Priority

## [2026-05-03] ingest | sources/papers/x2n_transformable.md — 新增 X2-N 论文，补充 hybrid-locomotion 和 loco-manipulation

## [2026-05-03] ingest | sources/papers/chasing_autonomy.md — A pipeline to dynamically retarget human motions and use control-guided RL for performant humanoid running. Mapped to motion-retargeting and humanoid-locomotion wiki pages.

## [2026-05-04] ingest | sources/papers/learn_weightlessness.md — 学习失重：人形机器人在非自稳定运动中的模仿机制

## [2026-05-04] ingest | sources/papers/kung_fu_athlete_bot.md — A Kung Fu Athlete Bot That Can Do It All Day: Highly Dynamic, Balance-Challenging Motion Dataset and Autonomous Fall-Resilient Tracking

## [2026-05-06] ingest | sources/papers/lwd.md — AGIBOT《Learning while Deploying: Fleet-Scale Reinforcement Learning for Generalist Robot Policies》, 新增 wiki/methods/lwd.md, 同步刷新 data-flywheel / online-vs-offline-rl / vla / π0-policy

## [2026-05-06] feat | v21-execution | P3 搜索结果按“置信度”分级（精确匹配 vs 潜在关联）

- V21 P3 第二项推进：`docs/main.js` 在搜索结果渲染前对每条命中按 `classifyTier(item, queryTokens)` 分级——命中标签 / 标题 / 路径归为「精确匹配」，仅命中摘要或正文 token 归为「潜在关联」
- 把原 `renderCards` 的卡片 HTML 拼接抽成 `buildResultCardHtml`，避免两个区块各写一份；仅当存在查询 token 时分组，空查询 + 类型筛选场景保持单区块原行为
- `docs/style.css` 新增 `.search-tier-heading` / `.search-tier-exact` / `.search-tier-potential` 样式：跨整行 grid，分隔线 + 小字号大写标签，强调「精确匹配」用 accent 色
- 键盘导航沿用 `getResultCards()`（仅查 `article.card[data-result-url]`），新增的 `<h4>` 区块标题不会进入 ↑↓/Enter 选区
- 验证：`python3 scripts/eval_search_quality.py` 通过率 **37/37 (100%)**；`node --check docs/main.js` 通过；`make lint` 仅余昨日 lwd ingest 留下的 1 条「陈旧页面」预警，与本次改动无关
- V21 checklist 对应条目已勾选

## [2026-05-07] feat | v21-execution | P3 详情页“知识地图”微地图（1-hop 邻居）

- V21 P3 收口：在 `docs/detail.html` 的 detail-hero 顶部新增 `#detailMiniMapWrap` D3 局部图谱容器，并补齐 `<script defer src="https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js"></script>`
- `docs/main.js` 新增 `renderDetailMiniMap(detailPage, detailPages)`：基于 `exports/link-graph.json` 用 `detailPage.path` 定位当前节点，遍历 `edges` 收集 1-hop 邻居（最多 12 个，避免拥挤），D3 force simulation 绘制；点击邻居节点经 `path → detail page id` 反查后跳转 `detail.html?id=...`
- 新增 `TYPE_COLOR_DETAIL_MINI` 与现有 `mini-graph.js` 的色板保持一致；当前节点用 `.mini-node-current` 描边强调；`<title>` 节点 hover 给出完整 label
- `docs/style.css` 新增 `.detail-mini-map` / `.detail-mini-map-head` / `.detail-mini-map-svg` / `.mini-node-current` 样式（高度 180px，跟随 `--bg/--bg-alt/--border` 主题变量）
- 验证：`make lint-js`（eslint）通过；`make test` 55 个测试全通过、覆盖率 57.16%（≥ 52% 阈值）；`make lint` 仅余昨日 lwd ingest 留下的 1 条「陈旧页面」预警，与本次改动无关
- V21 checklist 对应条目已勾选；V21 DoD 中「微地图组件上线」「log.md 记录」两项亦同步勾选

## [2026-05-07] ingest | sources/papers/genesis_gene_ecosystem.md — 收录 GENE-26.5 与 Genesis 仿真论文/链接，新增实体页并补强 genesis-sim 辨析

## [2026-05-07] docs(wiki): wiki/entities/asimov-v1.md — 补充被动趾（主仓弹簧关节 vs mjlab 固连）、RSU 踝机构与公开 MJCF 等效 2-DOF 接口及 asimov_constants 运动学符号说明

## [2026-05-07] docs(wiki): wiki/entities/asimov-v1.md — 补充主仓仿真定位、asimov-mjlab（PPO+imitation）与 Menlo 博文观测/Sim2Real 叙述；新增 sources/repos/asimov-mjlab.md；mjlab 实体页关联 Asimov 训练 fork

## [2026-05-07] ingest | sources/repos/asimov-v1.md — 收录 Asimov v1 全栈开源人形仓库；新增 wiki/entities/asimov-v1.md 并关联开源硬件对比与人形实体页

## [2026-05-07] ingest | sources/repos/1x-technologies.md、sources/repos/figure-ai.md — 沉淀 wiki/entities/1x-technologies.md、figure-ai.md，并更新 humanoid-robot 与硬件选型 query

## [2026-05-07] ingest | sources/repos/notable-commercial-robot-platforms.md — 市面知名人形/四足平台索引与 overview 纵览页

## [2026-05-07] ingest | sources/repos/motioncode.md — 收录 MotionCode 官网资料并新增 wiki/entities/motioncode.md，回链人形机器人 / 动作重定向 / 平台纵览

## [2026-05-07] ingest | sources/repos/sceneverse-pp.md — SceneVerse++ 入库并新增实体页与 3D 空间 VQA / VLN 任务页

## [2026-05-07] docs(wiki): wiki/entities/gene-26-5-genesis-ai.md — 补充 GENE-26.5 官方 YouTube 演示视频链接与参考来源说明

## [2026-05-08] ingest | sources/papers/neural_motion_retargeting_nmr.md — 入库 NMR 论文并新增 wiki/methods/neural-motion-retargeting-nmr.md（含 Mermaid 流程图）；补强 motion-retargeting / GMR 交叉引用；ingest-workflow 增补 Mermaid 步骤说明

## [2026-05-08] chore | merge | 合并 origin/main，解决 PR #134 冲突

- main 已独立实现 V21 P3「搜索结果按置信度分级」（[2026-05-06] feat）与「详情页微地图」（[2026-05-07] feat），与本分支 [2026-05-07] 同名条目重复
- 解决策略：保留 main 的实现（`buildResultCardHtml` / `.search-tier-heading*` / `renderDetailMiniMap` / `.detail-mini-map*`），覆盖本分支 `renderCardItem` / `.search-tier-header*` 变体；checklist v21 P3「详情页微地图」按 main 勾选
- 自动产物（README badge、`exports/*-stats.json`、`exports/lint-report.md`、`docs/exports/*.json`、`exports/index-v1.json`）以 main 为准，再走 `make ci-preflight` 校核
- 最终 PR #134 仅保留：本次 merge 解决冲突的记录

## [2026-05-08] checklist-v21 | schema/canonical-facts.json — 推进 V21「事实库扩展至 ≥ 140 条」DoD

- 围绕 V21 新增的触觉/通信专题与 NMR 入库新页，向 `schema/canonical-facts.json` 追加 10 条矛盾检测规则：Visuo-Tactile Fusion 接触瞬间、Tactile Impedance Control 变阻抗、Contact Wrench Cone 6 维推广、GelSlim 指节级薄型化、控制环路延迟五段分解、UDP 组播四类事件、PTP 时钟同步精度、EtherCAT DC 同步精度、EtherCAT vs EtherNet/IP 选型、NMR CEPR 管线
- 事实库总条目：130 → **140**，达到 V21 DoD「≥ 140 条」目标
- 同步勾选 V21 DoD「知识图谱节点数 ≥ 190」（当前 252，远超阈值）
- `python3 scripts/lint_wiki.py` 矛盾检测：0 个新矛盾（首次回归出现 lcm-basics 误命中后将「UDP 组播」neg_claims 收紧为 `支持.*重传 / 提供.*ACK / 能.*保证.*顺序`，避免误伤合法描述）
- 剩余 DoD「`make lint` 0 errors」单点未达成，其根因是 10 个 sources 比 wiki 新的「陈旧页面」预警，归并到下一日处理

## [2026-05-09] ingest | sources/papers/hipan.md — HiPAN 分层姿态自适应四足导航，新增 wiki/methods/hipan.md 并交叉引用 locomotion、curriculum-learning

## [2026-05-10] ingest | sources/papers/intentional_streaming_rl.md, sources/repos/intentional_rl.md — 流式 RL 意图更新论文与 Intentional_RL 仓库，新增 wiki/methods/intentional-updates-streaming-rl.md 并回链 RL / online-offline / policy-optimization

## [2026-05-10] ingest | sources/blogs/ted_xiao_embodied_three_eras_primary_refs.md — Ted Xiao 访谈相关一手文献索引与 wiki/queries 叙事消化页

## [2026-05-10] structural | wiki — Ted Xiao 一手索引技术线独立节点补全（概念/方法/实体共 14 页）与图谱交叉引用

## [2026-05-11] ingest | sources/repos/rldx-1.md — 收录 RLDX-1 技术报告与仓库；新增 wiki/entities/rldx-1.md，并自 VLA / Manipulation / StarVLA 交叉引用

## [2026-05-11] ingest | sources/repos/dm_control.md, sources/papers/dm_control_suite.md — 接入 Google DeepMind dm_control 与 arXiv:1801.00690，新建 wiki/entities/dm-control 并互链 MuJoCo、仿真器选型与索引

## [2026-05-12] ingest | sources/papers/being_h07.md — Being-H0.7 潜空间世界–动作模型：归档、wiki 方法页，互链 VLA / 生成式世界模型 / 潜空间想象

## [2026-05-12] ingest | sources/repos/april_tag.md — 收录 AprilTag 官方页与 AprilTag 3 C 库并新增实体 wiki 页

## [2026-05-12] ingest | sources/papers/humanoid_parallel_ankle_kinematics_ingest.md — 并联踝文献包（RG+arXiv）并扩充人形并联关节解算页、互链 Sim2Real 与 TO

## [2026-05-12] ingest | sources/sites/botlab_motioncanvas.md — BotLab MotionCanvas 入库与 wiki/entities/botlab-motioncanvas.md

## [2026-05-12] ingest | sources/sites/wuji_robotics.md — 舞肌科技原始资料与实体页、平台纵览互链

## [2026-05-12] ingest | sources/sites/wuji_robotics.md — 增补 Wuji Hand 灵巧手官方文档中心与 NE 时代盘点锚点

## [2026-05-12] ingest | sources/sites/project_instinct.md — Project Instinct 站群入库并升格 wiki/entities/project-instinct.md（含 Mermaid 流程图）

## [2026-05-12] ingest | sources/repos/mjlab_playground.md, sources/repos/freemocap.md — 接入 mjlab_playground 与 FreeMoCap；新增实体页与动捕→mjlab 流程 Mermaid；互链 mjlab / motion-retargeting / unitree_rl_mjlab

## [2026-05-12] ingest | sources/notes/legged_humanoid_rl_pd_gains.md — 腿足/人形 RL 关节 PD 增益资料索引与 wiki 查询页（含 Mermaid）

## [2026-05-12] ingest | sources/papers/rl_pd_action_interface_locomotion.md — RL+PD 人形/双足/四足文献索引与 Kp/Kd query 页增补 Mermaid 与论文共识

## [2026-05-12] structural | wiki — 新增 10 个 RL+PD 论文实体页（paper-*.md）并互链 Kp/Kd query、locomotion、legged_gym 与 sources 索引

## [2026-05-13] ingest | sources/sites/kleiyn-efgcl.md — EFGCL 项目页入库并升格 wiki/methods/efgcl.md

## [2026-05-13] ingest | sources/papers/reactor_rl_physics_aware_motion_retargeting.md — ReActor（arXiv:2605.06593）物理感知双层 RL 重定向入库与 wiki 方法页

## [2026-05-13] ingest | sources/papers/disney_olaf_character_robot.md — Olaf 实机角色（arXiv:2512.16705）入库并新增 wiki/methods/disney-olaf-character-robot.md

## [2026-05-13] ingest | sources/sites/text-to-cad-tools.md — 收录 Zoo 与文字生成 CAD 工具索引并新增 wiki 概念页

## [2026-05-13] structural | wiki — 扩充 text-to-cad 概念页与 sources 索引：成熟度、Adam、Fusion AI、CadQuery/OpenSCAD、网格向工具

## [2026-05-13] checklist-v21 | DoD 收口 & 初始化 V22

- V21 DoD 最后一项「`make lint` 0 errors」收口：`make lint` 输出「✅ 所有检查通过！共发现 0 个问题」，10 个 sources 比 wiki 新的「陈旧页面」预警已全部消化，勾选 `docs/checklists/tech-stack-next-phase-checklist-v21.md` 中对应条目。
- V21 完整交付：图谱 297 节点 / 1933 边，事实库 140 条，触觉与力觉闭环 / 硬件通信链路两条专题全部上线，详情页"知识地图"微地图与搜索结果分级 UI 已在 V20–V21 期间集成。
- 新建 `docs/checklists/tech-stack-next-phase-checklist-v22.md`，专题选定为「动作重定向与角色化人形」，配合「抓取与操作感知」深化；并基于 `exports/graph-stats.json` 中 `community_quality_warning: true`（最大社区占 46.1%）规划 P0 二级社区拆分任务。
- 同步将 README badge、维护看板、`AGENTS.md`、`docs/README.md` 与 `docs/checklists/README.md` 的「当前清单」指针从 V21 切到 V22；V21 进入历史归档区。

## [2026-05-13] ingest | sources/repos/anygrasp-sdk.md — 收录 AnyGrasp SDK / 论文 / GraspNet 生态，新增 wiki/entities/anygrasp、references/repos/manipulation-perception，互链 Manipulation

## [2026-05-14] lint | health-check — 补齐 deep-learning-foundations 与 generative-foundations 元数据，并为 humanoid-robot 增加整体开发流程图

## [2026-05-14] ingest | sources/papers/world_action_models_survey_2605.md、sources/repos/awesome-wam-openmoss.md、sources/sites/awesome-wam-openmoss.md — WAM 综述与 Awesome-WAM 入库，新增 wiki/concepts/world-action-models.md 并交叉 VLA / 生成式世界模型

## [2026-05-14] ingest | sources/sites/tairan-he.md — 收录个人主页并新增 wiki/entities/tairan-he.md，互链 GR00T-VisualSim2Real

## [2026-05-14] ingest | sources/sites/xue-bin-peng.md, sources/sites/zhengyiluo.md — 归档两位学者主页；升格 wiki/entities/xue-bin-peng.md 与 zhengyi-luo.md，并互链 MimicKit、ProtoMotions、SONIC、Tairan He

## [2026-05-14] ingest | sources/repos/kimodo.md、sources/repos/gr00t_wholebodycontrol.md — 入库 Kimodo 与 GR00T WBC 官方仓；新增 wiki/entities/kimodo、gr00t-wholebodycontrol 并互链 diffusion-motion-generation、motionbricks、foundation-policy、SONIC、ProtoMotions、GR00T-VisualSim2Real

## [2026-05-14] ingest | sources/blogs/menlo_noise_is_all_you_need.md — Menlo「处理器在环+总线抖动」博文入库；新增 wiki/concepts/processor-in-the-loop-sim2real；互链 sim2real、asimov-v1

## [2026-05-14] ingest | sources/papers/humannet.md — HumanNet arXiv:2605.06747 入库；新增 wiki/entities/humannet 并互链 VLA / IL / 人形机器人

## [2026-05-15] ingest | sources/personal/humanoid-policy-network-architecture-faq.md — 新建概念页 humanoid-policy-network-architecture 与 tech-map il 节点 policy-network-architecture

## [2026-05-15] structural | schema/search-regression-cases.json — WBC×QP BM25 回归 expected_top_k 3→5（语料扩张后 whole-body-control 概念页常居第 4–5 位）

## [2026-05-15] ingest | sources/repos/pytorch-official.md — 收录 pytorch.org 与 get-started/docs/tutorials；新建 wiki/entities/pytorch.md；互链深度学习基础、Isaac Lab

## [2026-06-20] structural | checklist-v25 P1 数据层专题交叉补强 —— motion-retargeting 与 humanoid-reference-motion-datasets 明示「数据来源 → 质量评估 → 重定向 → 策略输入」四段衔接

- `wiki/comparisons/humanoid-reference-motion-datasets.md` 新增「四段衔接」表与因果判据段，把五集数据落到①数据来源→②质量评估→③重定向→④策略输入四段，并显式回链 `motion-data-quality.md` 与 `humanoid-training-data-pipeline.md`。
- `wiki/concepts/motion-retargeting.md` 新增「上游衔接」表，把重定向定位为链路第③段，明示其触发与补层由 motion-data-quality 四轴（形态差距/接触/物理）决定，与 P1 新页形成双向回链、消除孤儿。
- `make lint` 0 errors（仅 3 条既有信息型预警）；勾选 v25 P1「数据层专题交叉补强」条目。

## [2026-06-23] structural | checklist-v25 P3 详情页「训练数据管线」专题徽标联动 —— 修正分词粒度漏匹配

- 详情页「所属专题」徽标行（`docs/main.js renderMetaTopicBadges`）本就以 `docs/topic-filters.js` 为单一事实源、`topicsForNode` 数据驱动：命中 `data-pipeline` 即渲染「📦 训练数据」徽标并跳 `graph.html?topic=data-pipeline`，空态降级隐藏整行——P3 第①项把 `data-pipeline` 写入单一事实源后，详情页徽标已自动联动，无需二次实现。
- 本次补强 `data-pipeline.segments` +5（`retarget`/`retargeter`/`omniretarget`/`mocap`/`freemocap`），修正纯分词粒度导致的漏匹配（`mocap-retarget`/`soma-retargeter`/`paper-...-omniretarget`/`freemocap` 等重定向与动捕实体此前只命中 `motion-retargeting`）；node 逐页校验后数据集 + 重定向 + mocap 实体 46/46 候选页全部稳定命中专题（全库 47 节点）。
- `make lint` 0 errors（另含 4 条信息型预警，不阻塞 CI）；勾选 v25 P3「详情页『同专题相关页』提示」条目。截图待带 Chrome 的环境补归档。

## [2026-07-07] structural | checklist-v28 P1 具身大模型分类学选型闭环知识链 —— 新建端到端选型 Query

- `wiki/queries/embodied-fm-taxonomy-loop.md`：把 VLM 感知理解 → VLN 空间导航 → VLA 动作执行 → VLX 一体化扩展 → WM 世界模型推演五大家族，从分散的实体/方法/对比页沉淀为一条端到端选型决策链；逐层给出 I/O 边界、数据需求、泛化能力、实时性/控制带宽与闭环稳定性取舍，配「五层选型闭环决策树」Mermaid、家族选型矛盾速查表与按层归因的失败模式速查表。
- `wiki/comparisons/vlm-vln-vla-vlx-world-model-taxonomy.md` 新增回链本 Query，消除新页孤儿并同步消除一条 `embodied_fm_crosslink` INFO 预警（10 → 9）。
- `make lint` 0 errors（另含 9 条信息型预警，不阻塞 CI）；勾选 v28 P1「具身大模型分类学选型闭环知识链」第①项（Query 页），概念页 `embodied-fm-latency-generalization-tradeoff` 待后续推进。

## [2026-07-08] structural | checklist-v28 P1 具身大模型分类学选型闭环知识链 —— 新建「实时性↔泛化取舍」姊妹概念页

- `wiki/concepts/embodied-fm-latency-generalization-tradeoff.md`：把[五层选型闭环 Query] ③ 执行层「泛化↔实时」的工程判据，沉淀为独立概念页；明示**模型规模 / 多模态跨度 / 世界模型推演步长**三个泛化旋钮如何共同推高推理时延 `τ`、压缩可用控制带宽，给出可稳定闭环的充要条件 `τ_total ≤ 1/f_ctrl`，并把这条「可达边界」讲成端到端 vs 分层选型分界的物理根因（分层=频段解耦）；配三旋钮代价表、破边界三条工程路线（动作分块 / 频域意图分词 / 异步双频）与常见误判速查表。
- `wiki/queries/embodied-fm-taxonomy-loop.md` ③ 执行层判据注脚补链新页、`related` 纳入新页，形成双向回链；新页 `tags` 含 vla/world-model，正文回链专题枢纽，不新增 `embodied_fm_crosslink` INFO 预警。
- `make lint` 0 errors（另含 7 条既有信息型预警，均与本次改动无关，不阻塞 CI）；勾选 v28 P1「具身大模型分类学选型闭环知识链」第②项（概念页），该父条目两个子项全部完成。

## [2026-07-19] structural | checklist-v29 P3② 详情页「具身评测基准」专题徽标联动 —— 端到端验证收尾

- 详情页「所属专题」徽标行（`docs/main.js renderMetaTopicBadges` → `docs/topic-filters.js topicsForNode`）本就以单一事实源数据驱动：命中 `embodied-eval-benchmark` 即渲染「🧪 具身评测基准 (Embodied Eval Benchmark)」徽标并跳 `graph.html?topic=embodied-eval-benchmark`，空态降级隐藏整行——P3① 把该专题（3 段 `bench`/`eval`/`benchmark` 干净片段 + 7 页 `ids`）写入单一事实源后详情页已自动联动，无需二次实现。
- node 逐页复核：`robo-bench`/`ewmbench`/`esi-bench`/`paper-gigaworld-1-policy-evaluation`/`simulation-evaluation-infrastructure`/`embodied-eval-benchmark-selection-loop`/`sim-vs-real-eval-gap` 7 页均稳定命中 `embodied-eval-benchmark`，非评测的 `vlm-vln-vla-vlx-world-model-taxonomy` 未命中；端到端截图 `detail.html?id=entity-robo-bench`「所属专题」行同渲「🏋 全身控制 (WBC)」+「🧪 具身评测基准」双徽标，多专题互不干扰。截图归档 `.cursor-artifacts/screenshots/detail-topic-embodied-eval-benchmark.png`（gitignore，不入库）。
- `make export graph` 重生 1687 节点 / 13524 边、0 orphans；`lint_wiki --report` 0 errors（1 条信息型预警，不阻塞 CI）。勾选 v29 P3「详情页『同专题相关页』提示」，**v29 全数完成**。

## [2026-07-20] structural | checklist-v30 主题调整 —— 改为「执行器驱动链选型闭环」

- 应维护者要求更换 v30 优化方向：由「免机器人示教数据采集选型闭环」改为「**执行器驱动链选型闭环**」——与 V28（选哪类具身大模型）/ V29（怎么评测）两条软件链互补，回答「策略力矩指令由什么电子硬件驱动链落地」。
- 四层链：EDA 电路设计（KiCad vs Altium、自研驱动板 vs 一体化关节）→ 电机驱动固件 FOC（SimpleFOC，电流环带宽/编码器分辨率）→ 执行器建模与摩擦辨识（BAM / NeuralActuator / actuator-network / SAGE，理想力矩源假设何时破）→ 实时总线闭环集成（EtherCAT 周期 ≠ 闭环带宽）。P0 巡检 / P1 Query（`actuator-drive-chain-selection-loop`）+ concept（`torque-source-abstraction-gap`）/ P2 事实库 240→250（10 条驱动链选型矛盾）/ P3 专题视图（`actuator-drive-chain`，⚡，第 20 项）+ 详情页徽标结构不变。

## [2026-08-31] ingest | sources/repos/microsoft-ui-xaml.md — WinUI 3 源码仓与 Windows 工控机 HMI 选型

- **意图：** 接入 Microsoft 官方 WinUI 3 开源仓库（`microsoft/microsoft-ui-xaml`），为 Windows 侧遥操作/采数控制台提供 UI 框架实体。
- **开源核查：** **已开源**（MIT，可 `init.cmd` + `build.cmd` 构建）；外部代码 PR 暂不接受（OSS 推进中）。
- **关键页：** [`wiki/entities/winui.md`](wiki/entities/winui.md)；交叉更新 [`wiki/tasks/teleoperation.md`](wiki/tasks/teleoperation.md)。

## [2026-09-01] ingest | sources/blogs/wechat_shenlan_robot_path_planning_five_paradigms.md — 深蓝具身智能路径规划五范式长文

- **意图：** 接入《具身智能基础》专栏第 11 篇，按图搜索、采样、人工势场、最优控制/MPC、强化学习与 AI 五条路线梳理路径规划范式，并以狭窄通道场景对照组合选型。
- **开源核查：** N/A（综述科普文，无单一项目页）。
- **关键页：** [`wiki/comparisons/robot-path-planning-five-paradigms-taxonomy.md`](wiki/comparisons/robot-path-planning-five-paradigms-taxonomy.md)；交叉更新 [`mobile-robot-navigation-planning-methods`](wiki/comparisons/mobile-robot-navigation-planning-methods.md)、控制/学习姊妹 taxonomy。

## [2026-09-05] ingest | sources/repos/robotis-git.md — 复核 ROBOTIS-GIT 组织（154 仓）；新增 cyclo_mjlab / zenoh_ros2_sdk 归档与 wiki/entities/robotis-cyclo-mjlab.md；刷新组织 hub

- **触发：** 用户指定 <https://github.com/ROBOTIS-GIT>；要求自动合入
- **变更：** 组织数 152→154；新增 **cyclo_mjlab**（K1 · mjlab RL/Mimic）、**zenoh_ros2_sdk** + **lerobot_robot_ros2_zenoh**（LeRobot 无 ROS 2 安装路径）；刷新 [`robotis.md`](wiki/entities/robotis.md) 双仿真栈地图
- **开源核查：** 主线仓 **已开源**；Zenoh LeRobot 集成标 **Alpha**

## [2026-09-05] ingest | sources/papers/egoexomocap_arxiv_2607_15868.md — EgoExoMoCap 分布式 HMD ego-exo 野外动捕

- **意图：** 接入 ECCV 2026 Spotlight 论文与 SIPLAB 项目页，沉淀分布式智能眼镜互观测的全身 SMPL 重建管线（EgoNet + ViTPose 射线 + DINOv3 门控）。
- **开源核查：** GitHub `eth-siplab/EgoExoMoCap` **已建仓但 Code coming soon**（MIT，待发布）。
- **关键页：** [`wiki/entities/paper-egoexomocap.md`](wiki/entities/paper-egoexomocap.md)；交叉更新 [`ego-category-04-ego-exo-fusion`](wiki/overview/ego-category-04-ego-exo-fusion.md)、[`paper-notebook-category-14-human-motion`](wiki/overview/paper-notebook-category-14-human-motion.md)。

## [2026-08-22] ingest | sources/blogs/wechat_guyue_rosclaw_ros2_natural_language.md — 古月居 RosClaw / RoboClaw 自然语言控 ROS2 长文

- **意图：** 接入古月居对 RosClaw（OpenClaw × ROS2 插件）三层架构、三种部署模式、工具集与安全的解读；对照 SJTU MINT RoboClaw 跨本体具身助手。
- **开源核查：** RosClaw **已开源**（Apache-2.0，README 注明拆仓迁移中）；RoboClaw **已开源 early stage**。
- **关键页：** [`wiki/entities/rosclaw.md`](wiki/entities/rosclaw.md)、[`wiki/entities/roboclaw.md`](wiki/entities/roboclaw.md)；交叉更新 [`openclaw`](wiki/entities/openclaw.md)。
