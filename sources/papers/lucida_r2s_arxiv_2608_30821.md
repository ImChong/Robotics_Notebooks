# Lucida: Parse, Generate, and Place for Composable Real-to-Sim Scene Modeling（arXiv:2608.30821）

> 来源归档（ingest）

- **标题：** Lucida: Parse, Generate, and Place for Composable Real-to-Sim Scene Modeling
- **缩写 / 框架：** **Lucida**（组合式 Real-to-Sim 场景建模）；**GizmoAct**（VLM 闭环 gizmo 放置策略）
- **类型：** paper / real2sim / scene-modeling / 3d-grounding / vlm / indoor
- **arXiv：** <https://arxiv.org/abs/2608.30821>（Submitted 2026-08-31；PDF：<https://arxiv.org/pdf/2608.30821>；HTML：<https://arxiv.org/html/2608.30821>）
- **项目页：** <https://lucida-r2s.github.io/>（归档：[lucida-r2s-github-io.md](../sites/lucida-r2s-github-io.md)）
- **作者：** Minghan Qin‡、Yuang Wang‡、Xiuyu Yang‡（实习）、Yushi Long*、Yujian Zhang*、Ruihuan Wang*、Kai Ye*、Yangang Zhang†、Hang Li（‡ 同等贡献；* ByteDance Seed 实习；† 通讯）
- **机构：** 字节跳动 Seed（ByteDance Seed）；北京大学（Peking University）；浙江大学（Zhejiang University）
- **入库日期：** 2026-09-04
- **一句话说明：** 保持 parse–generate–place 顺序，但把「精确实例几何 / 无遮挡视图 / 资产与观测几何一致」等真实室内捕获做不到的前提后移；解析只收多视角证据，生成做 amodal 补全，放置用 GizmoAct 在 3D 编辑器 GUI 上闭环对齐 9-DoF。

## 开源状态（步骤 2.5，2026-09-04）

- **项目页：** <https://lucida-r2s.github.io/> 资源行仅 **arXiv** + **Hugging Face papers**；无 Code / GitHub / 权重 / 数据集按钮。
- **论文：** 摘要与正文只列项目页，**未承诺**「code will be released」。
- **仓库检索：** `github.com/lucida-r2s` 为 Pages 托管；公开检索无实现仓。同名 `Artificial-Humanity/Lucida` 是无关图像/视频生成 CLI。
- **结论：** **确认未开源**（截至入库日）。勿建 `sources/repos/`。wiki 工程实践与源码时序图按「不适用」处理。

## 摘录 1：问题与主张（§1）

- **目标：** 把真实室内场景恢复成**可分离、可编辑的物体资产 + 空间布置**，供机器人仿真 / 具身 AI 消费；相对 NeRF/3DGS 整房单体、CAD 检索（保真度受库覆盖限制）、交互式场景合成（不锚定某次捕获）。
- **痛点：** 现有管线拆成解析 / 生成 / 放置，每步都预设杂乱捕获给不了的输入（精确实例几何、无遮挡中心视图、资产与观测几何一致）。任一步失败会沿固定顺序向后传播。
- **主张：** 保持三步顺序，但**重分配要求**：解析输出带多视角证据的场景图（不必精确实例重建）；生成以证据束为条件做 amodal 补全再抬到 3D；放置接受资产–观测几何不一致，用 GizmoAct 从粗初始化 + referring cue 恢复 9-DoF。精度在闭环末段达成。
- **命名：** Lucida 取自 camera lucida（把实景叠到画纸上的辅助工具）。
- **贡献四点：**（1）杂乱捕获可满足的需求重分配；（2）带 per-instance 多视角证据的场景级 3D 检测（R2S-Scene all 协议 mAP 0.351→0.592 vs Boxer）；（3）GizmoAct：SFT + RL 的 VLM GUI 策略，严格对齐最多 +25.6 个百分点；（4）自建 R2S 基准上检测 / 位姿 / 场景重建三级评测，场景 F-Score 0.924 vs SAM 3D 0.794。

**对 wiki 的映射：** 升格 [`wiki/entities/paper-lucida-r2s.md`](../../wiki/entities/paper-lucida-r2s.md)；与 [Sim2Real](../../wiki/concepts/sim2real.md) Real2Sim 节、[SimFoundry](../../wiki/entities/paper-simfoundry-real2sim-scene-generation.md) / [Agentic Real2Sim](../../wiki/entities/paper-agentic-real2sim.md) / [CRISP](../../wiki/methods/crisp-real2sim.md) / [R2S-EGO](../../wiki/entities/paper-r2s-ego.md) 对照。

## 摘录 2：方法栈（§2）

- **输入：** 带位姿 RGB(-D) \(\mathcal{I}=\{(I_i,D_i,K_i,T_i)\}\)；输出物体级可编辑副本：每实例完整资产 + 9-DoF 位姿 + 场景图。
- **Parse（§2.1）：** 场景图 \(G=(V,E)\)；节点证据 \(\mathcal{E}_o=\{\mathcal{V}_o,\mathcal{M}_o,\mathcal{P}_o,b_o,c_o\}\)（多视角观测、掩码/框、部分点云、代表 3D 框、类别/指称）。边为支撑 / 包含 / 邻接。三子步：几何感知关键帧（共视 + 时间项）+ VLM 发现 + Boxer 类 3D 框；全序列证据巩固（代表框投影、视频分割跟踪补观测）；关系感知细化（拆合并、补缺失支撑）。
- **Generate（§2.2）：** 从 \(\mathcal{E}_o\) 选互补可靠视图 → Set-of-Mark + VLM 编辑指令 → 图像编辑做孤立无遮挡物体图 → **Seed3D 2.0** 抬到资产 \(A_o\)；用 \(b_o\) 粗初始化。
- **GizmoAct（§2.3）：** 把 9-DoF grounding 写成多轮 GUI：状态 \(x_t=(p_t,R_t,s_t)\)；观测渲染点云 + 资产叠加 + 黄框 + 局部系 gizmo（红绿蓝）+ 被点云挡住的半透明绿层 + 辅视角 + 局部轴正交图。动作：`update_pose`（增量、平移/缩放以当前物体尺寸为单位，从不回归绝对位姿或度量尺度）、`stop`、以及大旋转用的 `switch_obs` + `permute_axis`（24 种轴对齐重定向）。XML 标签 + JSON 载荷，每轮一条。SFT：3D-FRONT+MesaTask / FoundationPose / CA-1M Objects 合成专家轨迹，含 DART 式错误注入只监督恢复。RL：同一可执行环境上 GRPO（DAPO 动态采样 \(K=8\)）；终态量化奖励（广义 3D IoU + 测地旋转误差分档），对称物体排除出 RL。
- **后处理（§2.4）：** 支撑 / 碰撞 / 接触一致性与摆放合理性与 GizmoAct **刻意分开**（个体 grounding vs 场景规则）。

**对 wiki 的映射：** 实体页画 parse–generate–place 流程图；强调「增量 gizmo 编辑 + 自学停止」相对 render-and-compare 的读法。

## 摘录 3：评测（§3）

| 设定 | 要点 |
|------|------|
| 检测（Table 1） | CA-1M / R2S-Scene；Boxer / WildDet3D。R2S-Scene `_all`：Boxer 0.351 → **0.592**（相对 +69%）；`_filter` 0.355→0.597。CA-1M `_all` 0.171→0.180，`_filter` 0.373→0.390。仅用 keyframe 提示，仍优于全帧提示的 Boxer。 |
| 位姿（Table 2） | 与上游检测解耦；Boxer 初始化；最多 12 步。R2S-Object max-4-view：ADD-SB 0.038→**0.017**，@0.05 **92.0%**（RecGen 1-view 79.2%），3D IoU 0.500→**0.719**。CA-1M @0.05 **83.4%**（最强基线 57.8%），IoU 0.434→**0.607**。ADT 单视 ADD-SB 最低 0.020；max-4-view @0.05 **90.0%**。 |
| 初始化鲁棒（Table 3） | 同一随机扰动训练策略适配 Boxer / Any6D* / SAM 3D；估计深度上 Boxer 更好，ADT 精确几何上 Any6D*/SAM 3D 更好。 |
| 场景重建（Table 4，R2S-Scene） | Scene CD 0.022→**0.010**，F-Score 0.794→**0.924**，BBox IoU 0.396→**0.495**；物体级归一化 F-Score 0.704→**0.736**（vs SAM 3D；SceneGen 场景 F-Score 仅 0.351）。 |
| 解析消融（Table 5） | 均匀关键帧 / 去掉全序列巩固 / 去掉关系细化均掉点；均匀采样掉得最多（mAP 0.597→0.516，场景 F-Score 0.831→0.727）。 |
| GizmoAct 训练（Table 6 + 难例子集） | RL 训练位姿分布对齐 Boxer 初始化全面更好。110 个大误差非对称物体：RL 旋转误差 **20.98°** vs SFT 45.19° / 无上下文 SFT 27.62°。 |
| 局限（§5） | 解析阶段漏掉的物体后续无法补回；闭环目前只在放置步，作者指向整管线 agentic 扩展。 |

**对 wiki 的映射：** 结论节突出「精度后移 + GizmoAct 吸收资产–观测 mismatch」；注明未开源、评测是几何对齐而非策略 Pearson / 真机操作。

## 建议 wiki 动作

- 新建 **`wiki/entities/paper-lucida-r2s.md`**。
- 交叉更新 [sim2real.md](../../wiki/concepts/sim2real.md) Real2Sim 节、[SimFoundry](../../wiki/entities/paper-simfoundry-real2sim-scene-generation.md)、[Agentic Real2Sim](../../wiki/entities/paper-agentic-real2sim.md)、[CRISP](../../wiki/methods/crisp-real2sim.md)、[R2S-EGO](../../wiki/entities/paper-r2s-ego.md)、[Awesome-Real2Sim2Real](../../wiki/entities/awesome-real2sim2real.md)、[Manipulation](../../wiki/tasks/manipulation.md)。
