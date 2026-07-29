# Data Pyramid for Embodied Manipulation（arXiv:2607.24744）

> 来源归档（ingest）

- **标题：** Data Pyramid for Embodied Manipulation
- **缩写 / 框架：** **Embodied Data Pyramid**（具身数据金字塔）
- **类型：** paper / survey / data-centric / embodied-manipulation
- **arXiv：** <https://arxiv.org/abs/2607.24744>（v1 2026-07-27；PDF：<https://arxiv.org/pdf/2607.24744>）
- **项目页：** <https://jasper-aaa.github.io/embodied-data-pyramid/> — 归档见 [`sources/sites/embodied-data-pyramid.md`](../sites/embodied-data-pyramid.md)
- **代码 / 策展清单：** <https://github.com/worldbench/awesome-embodied-data-pyramid> — 归档见 [`sources/repos/awesome-embodied-data-pyramid.md`](../repos/awesome-embodied-data-pyramid.md)
- **作者：** Yifan Ye*、Yankai Fu*、Yaoxu Lv*、Bohan Hou*、Jun Cen*†（Project Leader）、Lingdong Kong* 等 29 人；通讯 Shanghang Zhang‡（PKU）
- **机构（11 家）：** 北京大学（PKU）、南洋理工大学（NTU）、香港科技大学（HKUST）、新加坡国立大学（NUS）、香港中文大学（CUHK）、香港大学（HKU）、杜克大学（Duke）、加州大学伯克利分校（UC Berkeley）、大湾区大学（GBU）、南京大学（NJU）、上海交通大学（SJTU）
- **入库日期：** 2026-07-29
- **一句话说明：** 数据中心视角的具身操作综述：以 **可扩展性 × 机器人对齐** 两轴（外加质量/多样性/可复用性/物理保真四维）把具身数据生态组织成 **五层金字塔**——真机数据、UMI 式数据、第一/第三人称人类数据、仿真数据、通用视觉语言数据；再从数据配方（data recipe）视角分析具身脑模型、VLA、WAM 三类基础模型如何选择/对齐/混合各层数据，并提出六大开放挑战。

## 开源状态（步骤 2.5）

核查日：**2026-07-29**（项目页 / GitHub / arXiv 首页金属链接）。

| 产物 | 状态 |
|------|------|
| Awesome 策展清单 `worldbench/awesome-embodied-data-pyramid` | **已开源**（数据集/管线按五层分类的持续维护清单；入库时约 70 stars） |
| 项目页数据集表格（按类别 × 任务检索） | **已开放**（jasper-aaa.github.io/embodied-data-pyramid） |
| 训练 / 推理代码、模型权重 | **不适用**——综述论文，无可运行实现 |

**结论：** **资源型开源**（策展清单 + 数据表），非代码型开源；论文承诺 "release and maintain an open-source repository that curates representative datasets"，已兑现为 Awesome 清单。致谢 UMI-Data-Community 与 IRMVLab/awesome-robot-learning-from-human-videos。

## 摘录 1：金字塔组织原则与五层定义（§1）

- **中心问题：** 具身基础模型没有「整个互联网」可吃——需要把观测与物理状态、动作耦合的数据；什么数据该用来训练具身基础模型？
- **两主轴（互相对立）：** **Scalability**（硬件依赖、人力、复位、安全监督、边际生成成本）vs **Robot Alignment**（观测/表示/监督信号对真机学习执行的直接程度）。越对齐越难扩展，越可扩展监督越间接。
- **四个补充维度：** Quality（有效性/一致性/信息量/任务相关）、Diversity（任务/物体/场景/视角/具身/传感覆盖）、Reusability（跨任务/环境/本体/模型族迁移）、Physical Fidelity（接触、摩擦、柔顺、传感噪声、执行延迟的忠实度）。
- **五层（自顶向下 = 对齐↓、可扩展↑）：**
  1. **真机数据**（Real-Robot）：遥操作或脚本化在目标机器人上闭环记录观测/状态/动作/结果；最直接可执行也最昂贵（每小时都要硬件+操作员+复位）。
  2. **UMI 式数据**：手持夹爪 + 腕相机 + 视觉惯性 SLAM 记录末端 6-DoF 相对轨迹，采集合环路无机器人；保留夹爪级监督，丢失关节级本体感知，需重定向。
  3. **第一/第三人称人类数据**（Ego/Exo）：真实物理 + 灵巧手 + 日常多样性，规模取决于可穿戴采集；无机器人参与，动作必须重建并跨人–机鸿沟重定向。
  4. **仿真数据**：物理引擎并行生成，返回可执行动作 + 特权标签（接触/位姿/成功信号），边际成本近零；返回不了它近似的物理本身。
  5. **通用数据**（General）：web 级图像/视频/语言/视觉-语言语料，承载语义、空间结构与常识； ground 感知与推理而非动作，不谈接触与后果。

**对 wiki 的映射：** 升格 [`wiki/entities/paper-data-pyramid-embodied-manipulation.md`](../../wiki/entities/paper-data-pyramid-embodied-manipulation.md)；与模型专属金字塔叙事 [GR00T N1 数据金字塔](../../wiki/entities/paper-hrl-stack-34-gr00t_n1.md)、[mimic 数据金字塔](../../wiki/entities/mimic-hand-m1.md) 互为「类目级系统化 vs 单模型配方」对照；真机层锚定 [Open X-Embodiment](../../wiki/concepts/open-x-embodiment.md)。

## 摘录 2：各层采集管线要点（§2–§6）

- **真机（§2）：** 采集范式三分——脚本化（规则执行 / 轨迹回放 / 自主策略 rollout，QT-Opt、MT-Opt）、遥操作（kinesthetic、leader-follower 含 GELLO/ALOHA、一体化 leader-follower 如 ARX AC-One、设备中介 VR/SpaceMouse、视觉估计 HumanPlus/AnyTeleop、可穿戴动捕/外骨骼 Dexora）、人在环增强（DAgger 谱系 ThriftyDAgger/Sirius/Fleet-DAgger/CR-DAgger）。趋势：单臂→双臂→移动→人形→灵巧手；RGB-D→触觉/力觉/音频/IMU/LiDAR 多模态；规模到 AgiBot World Beta 100 万条 / RoboMIND 2.0 31 万条 739 任务 / OXE 聚合 240 万条。**多样性比条数更关键**（任务/场景/本体/模态/轨迹级五类多样性）。
- **UMI（§3）：** 演进 UMI → FastUMI → LEGATO → FreeTacMan（指戴式）；相对轨迹表示（腕相机 + IMU + 视觉惯性 SLAM 估计 6-DoF，未来目标相对当前末端表示，降低全局漂移）；跨本体部署靠「具身无关任务表示 + IK/运动规划/笛卡尔控制」；灵巧化靠 DexUMI 外骨骼约束 + 视觉 inpainting 换人手机械手。局限：视觉跟踪脆弱、无真机执行难验证质量、缺力反馈。
- **人类 Ego/Exo（§4）：** 采集基础设施按被测物理量组织（视觉传感 / 运动跟踪 / 辅助交互传感——凝视、EMG、力/触觉）；监督构建四类——语义（叙述、verb-noun、程序步骤）、几何（相机/场景、手/体姿态三条路线：标注辅助重建、模型预测+参数拟合、传感捕获、物体交互几何）、多模态（凝视/音频/IMU/EMG/力触觉）、机器人导向（EgoVLA 逆运动学重定向、EgoMimic 对齐、H-RDT 预训练+适配器）。
- **仿真（§5）：** 基础设施三件套（本体-传感系统 / 物体场景资产 / 物理渲染后端）；资产生态 ShapeNet→Objaverse→PartNet-Mobility→PhysX 系列→AIGC ManiTwin 10 万+；合成演示生成四类（人执行、规则执行、回放扩展 MimicGen/DexMimicGen、自主与生成式 rollout 含 GenSim/RoboGen/RoboTwin2.0/InternData）；**世界模型作为仿真器**（策略训练 World4RL/DiWA、策略评估 WorldGym/WorldEval、合成数据引擎 DreamGen/GigaWorld-0）；Sim2Real 差距 = 观测失配 + 交互失配（运动学 gap 可修，动力学 gap 难消）。
- **通用数据（§6）：** 按能力贡献组织——视觉语言（语义/常识）、分割定位（空间接地）、3D（几何）、规划（任务分解）、时序（感知记忆）、物理/因果/失败推理、抓取数据；弱动作接地是根本局限，自动标注幻觉需过滤。

**对 wiki 的映射：** 各层 Takeaway 汇入实体页「核心结构」；与 [Sim2Real](../../wiki/concepts/sim2real.md)、[物理保真度与 Sim2Real 差距](../../wiki/concepts/physics-fidelity-sim2real-gap.md)、[遥操作任务页](../../wiki/tasks/teleoperation.md)、[dagger](../../wiki/methods/dagger.md) 交叉。

## 摘录 3：具身基础模型的数据配方分析（§7）

- **三大趋势：** (i) 配方从单一真机演示走向异构混合（π₀→π₀.₅→π₀.₇ 逐代加层；LingbotVA 2.0 覆盖全部五层）；(ii) 预训练规模陡增（Qwen-RobotManip 约 3.81 万小时多源语料；Xiaomi-Robotics-1 >10 万小时 UMI 预训练 + 约 1 万小时跨本体后训练）；(iii) egocentric 人类数据地位上升（EgoScale 20,854 小时受控缩放；HumanScale 5,000 小时受控预训练）；另有对低质量轨迹容忍度上升（π₀.₇：低质量轨迹 + 清晰 prompt 仍可用）。
- **动作空间对齐三策略：** 具身专属投影（GR00T N1/Octo/SmolVLA）、定长零填充接口（π₀ 等）、**语义动作槽**（Qwen-RobotManip 80 维规范状态-动作向量：双臂各 29 维 + 22 维预留；RDT-1B/Being-H0.5/UniDex/LingbotVLA 2.0 同族）。几何表示三类：机器人中心（OXE）、相机中心（Qwen-RobotManip/OC-VLA）、腕中心（METIS/LDA-1B）；坐标约定（原点/手性/TCP/绝对-vs-delta/旋转参数化/单位）应作一等元数据记录。
- **三类模型消费数据的方式：**
  - **具身脑模型**（RynnBrain、HY-Embodied、Pelican-VL）：action-free 数据支撑理解（含视频预训练 GR-1/V-JEPA 2/Cosmos Predict），action-labeled 数据转为 affordance/轨迹/子任务边界等可迁移监督（RoboBrain/ShareRobot）。
  - **VLA**：action-labeled 主监督从离散 action token（RT-2/OpenVLA）走向扩散/流匹配连续头（RDT/π₀.₅）；action-free 视频经 latent action（LAPA/Moto/UniVLA/Villa-X/CLAP/ConLA）或几何重建（VideoDex/Track2Act/General Flow/VidBot）转为 action proxy；层次化中间监督（π₀.₅ 语义、CoT-VLA 推理链、affordance 感知）。
  - **WAM**：action-labeled 两范式——扩散/流匹配连续动作去噪（Genie Envisioner、Motus MoT、VideoVLA、Cosmos Policy）vs 自回归离散动作 token（WorldVLA）；action-free 视频建立世界先验（UniPi/UniSim/Cosmos/V-JEPA 2 百万小时级），两阶段配方「大规模 action-free 预训练 → 动作条件后训练」成主流。

**对 wiki 的映射：** 数据配方趋势汇入实体页「实验与评测（数据中心分析）」；交叉 [VLA 方法页](../../wiki/methods/vla.md)、[WAM 概念页](../../wiki/concepts/world-action-models.md)、[具身 Scaling Laws](../../wiki/concepts/embodied-scaling-laws.md)、[EgoScale](../../wiki/methods/egoscale.md)、[Xiaomi-Robotics-1](../../wiki/entities/xiaomi-robotics-1.md)。

## 摘录 4：六大开放挑战（§8）

1. **大规模触觉数据集**——RGB-D+本体+语言之外缺「接触层」；触觉未标准化（传感器专属格式、任务覆盖窄）。
2. **失败与恢复数据**——成功偏置丢弃失败轨迹；需要 pre-failure 上下文、失败 onset、类别/原因、恢复动作与结局的结构化标注；失败是恢复行为监督而非废数据。
3. **可扩展采集管线**——降低对昂贵人工遥操作的依赖；可穿戴设备需更轻/无线/模块化 + 自动跨设备标定。
4. **跨本体状态-动作对齐**——统一存储格式 ≠ 一致几何语义；坐标约定/标定参数/控制器模式应作一等元数据。
5. **egocentric 先验用于灵巧操作**——人–机差距是运动学/形态/物理的；人视频应作结构化交互先验（任务意图、affordance、抓取选择、接触序列）而非精确动作标签。
6. **有原则的数据配方**——最优层级比例未确立；缺 compute-matched 单源消融；需架构感知、阶段依赖的固定/课程/自适应混合策略对比。

**对 wiki 的映射：** 汇入实体页「结论」可操作要点；失败恢复轴交叉 [Data Flywheel](../../wiki/concepts/data-flywheel.md)、触觉轴交叉 [T-Rex 触觉 VLA](../../wiki/entities/paper-trex-tactile-reactive-dexterous-manipulation.md)。

## 建议 wiki 动作

- 新建 **`wiki/entities/paper-data-pyramid-embodied-manipulation.md`**（含五层金字塔 Mermaid + 模型数据配方分析 + 结论）。
- 新建 **`sources/sites/embodied-data-pyramid.md`**、**`sources/repos/awesome-embodied-data-pyramid.md`**。
- 交叉更新 [open-x-embodiment](../../wiki/concepts/open-x-embodiment.md)、[paper-hrl-stack-34-gr00t_n1](../../wiki/entities/paper-hrl-stack-34-gr00t_n1.md)、[embodied-scaling-laws](../../wiki/concepts/embodied-scaling-laws.md)、[world-action-models](../../wiki/concepts/world-action-models.md)、[vla](../../wiki/methods/vla.md)、[roadmap/depth-vla](../../roadmap/depth-vla.md)。
