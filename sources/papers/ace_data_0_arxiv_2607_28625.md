# ACE-Data-0: Human-Centric Ambient Capture as Embodied Data Engine（arXiv:2607.28625）

> 来源归档（ingest）

- **标题：** ACE-Data-0: Human-Centric Ambient Capture as Embodied Data Engine
- **缩写 / 框架：** **ACE**（Ambient Capture Engine）；**ACE-Data-0**（发布数据集与分层 benchmark）
- **类型：** paper / dataset / benchmark / multimodal / HOI / HSI / egocentric
- **arXiv：** <https://arxiv.org/abs/2607.28625>（v1；HTML：<https://arxiv.org/html/2607.28625>；PDF：<https://arxiv.org/pdf/2607.28625>）
- **项目页：** <https://ace-data-engine.github.io/ACE-Data-0/> — 归档见 [`sources/sites/ace-data-0-github-io.md`](../sites/ace-data-0-github-io.md)
- **数据集：** <https://huggingface.co/datasets/ACERobotics/ACE-Data-0> — 归档见 [`sources/datasets/ace-data-0.md`](../datasets/ace-data-0.md)
- **作者：** Yukang Cao¹\*、Haozhe Xie¹\*、Beichen Wen²\*、Runmao Yao¹、Yinghao Liu²、Yue Huang²、Zhichao Liao²、Yunxiang Wang¹、Haiheng Liu¹、Xingshun Tian²、Dawei Su²、Long Zhuo²、Dacheng Tao²†、Xiaogang Wang²†、Liang Pan²‡、Ziwei Liu¹‡（\* equal；† Project Advisor；‡ Project Lead）
- **机构：** ¹ S-Lab, 南洋理工大学（NTU）；² 大晓机器人（Ace Robotics / ACERobotics）
- **入库日期：** 2026-08-10
- **一句话说明：** 用 Ambient Capture Engine 把真实家居变成时空校准的录制工作室（table-scale + room-scale），发布 **150 h / 17M 帧 / 75k episodes** 的同步多模态家居活动集 ACE-Data-0，并配套「信号→场景组件→交互」三层感知诊断 benchmark。

## 开源状态（步骤 2.5）

核查日：**2026-08-10**（项目页 / Hugging Face API / arXiv 首页金属链接）。

| 产物 | 状态 |
|------|------|
| Hugging Face 数据集 `ACERobotics/ACE-Data-0` | **已发布（gated）** — 非商业学术研究许可；个人申请不可转让 |
| 项目页 | **已开放** — Demo / 模态叙事 / 数据与 benchmark 说明 |
| 训练 / 评测代码、仓库 | **未见** — 项目页未列 GitHub；HF tree 仅 LICENSE + README |
| 模型权重 | **不适用** — 本篇为数据引擎 + 数据集 + 基准，非新架构权重 |

**结论：** **部分开源**——数据与标注已上线（gated 研究许可）；官方训练栈 / 统一 benchmark 跑分代码截至入库日未列。源码运行时序图写「不适用」。

## 摘录 1：问题设定与 ACE 双尺度范式（§1–§3）

- **瓶颈：** 具身学习需要的是「第一人称感知 + 全身运动 + 灵巧操作 + 物体状态 + 声音 + 触觉」随目标推进的**同步演化**；现有集在视角 / 模态 / 空间尺度上碎片化，感知–动作闭环只被部分观测。
- **三条短板（Table 1 对照）：** (1) 模态碎片——大规模 ego 集缺度量运动与外视；(2) 环境不自然——MoCap HOI 多在空旷实验室；(3) 视野短——秒级原子动作 vs 分钟级家居目标链。
- **ACE：** 两互补配置共用同一同步标定管线——
  - **Table-scale：** 近距相机 + 光学动捕 + 触觉手套，解析细粒度手–物接触；
  - **Room-scale：** 铺满家具的公寓（厨 / 餐 / 客 / 卧），宽基线相机跟踪跨房间 locomotion 与家具接触。
- **传感栈：** ≥8 外视 RGB；4 路鱼眼 ego + IMU + 头显位姿；全身 + 关节手 60 Hz；物体 mesh 与 6-DoF；多通道音频；全掌触觉压力图。

**对 wiki 的映射：** 升格 [`wiki/entities/paper-ace-data-0.md`](../../wiki/entities/paper-ace-data-0.md)；金字塔第 ③ 层「人类 Ego/Exo」对照 [Data Pyramid](../../wiki/entities/paper-data-pyramid-embodied-manipulation.md)；与规模向 ego [Ego4D](../../wiki/entities/paper-ego4d.md)、[RekaDaily-10k](../../wiki/entities/rekadaily-10k-dataset.md)、[EgoScale](../../wiki/methods/egoscale.md) 区分「度量同步完备 vs 小时规模」。

## 摘录 2：ACE-Data-0 规模、协议与标注（§3–§4）

- **规模：** 150 小时 · 17M 帧 · 200 任务类 · 50 参与者 · 2 环境 · 75,000 interaction episodes。
- **协议：** **目标级指令**（goal-level）而非逐步脚本——保留路线选择、抓取策略、犹豫与恢复等自然变异；长程行为由任务完成本身涌现，而非拼接短 clip。
- **三类活动：** Atomic HOI（1–3 个日常操作）/ Chain of HOI（连续家务例程，需中间状态跟踪）/ HSI（全身–场景：坐、导航、姿态转换、家具接触）。
- **同步与标定：** 以 OptiTrack 时钟为参考对齐相机 / 音频 / 触觉；静态外视与可穿戴相机经 marker 桥接注册到同一世界系；触觉手套用 IMU 运动模板与 ego 头显对齐。
- **标注：** 相机内外参与时间线；全身 / 手姿态及投影；物体 mesh、6-DoF、框与轨迹；接触标签（运动状态 + 触觉）；对齐时间线的自然语言描述——大量标注由**测量状态自动导出**，而非独立视觉估计管线。

**对 wiki 的映射：** 数据集速查与流程总览进实体页；许可与门控细节见 [`sources/datasets/ace-data-0.md`](../datasets/ace-data-0.md)。

## 摘录 3：三层 benchmark 与评测要点（§5）

层级设计：**信号 → 场景组件 → 交互**（诊断「感知–理解–交互」链条何处断裂）。

1. **触觉 from vision（ego，Table 3）：** TouchAnything 相对最强（Temp Acc 0.71 / C-IoU 0.16），但绝对重叠仍低；PressureVision 几乎失效——**「何时接触」易、「何处加压」难**。
2. **人体运动恢复（room-scale，Table 4，22 方法）：** 局部姿态（PA-MPJPE）可接近常规基准；**世界系轨迹（WA-MPJPE）显著更差**；scene-aware 主要帮全局定位；ego 方法因身体出画最弱。
3. **手运动（ego Table 5 / exo Table 6）：** exo 关节精度更好（WiLoR PA-MPJPE 9.1 mm）；世界系轨迹 exo（HaPTIC 63 mm）明显优于 ego 世界系方法（~98–102 mm）——**egomotion 估计是主要误差源**；双视角互补（ego 近距细节 vs exo 全局与遮挡）。

**对 wiki 的映射：** 汇入实体页「实验与评测」与「结论」可操作要点。

## 摘录 4：局限与伦理（§6）

- 仅 **2** 个站点，布局 / 光照多样性有限。
- GT 限于被仪器化实体：物体需预扫描与贴标；关节机构 / 流体 / 可变形物状态变化未标。
- 动捕服、手套、头显与 marker 在画面中可见，可能引入数据集特有视觉线索。
- 参与者自愿并知情同意录制与研究发布。

**对 wiki 的映射：** 「局限与风险」；选型时勿当「任意家庭 / 任意物体」开箱泛化保证。

## 建议 wiki 动作

- 新建 **`wiki/entities/paper-ace-data-0.md`**（采集管线 Mermaid + 三层 benchmark + 结论；源码时序图「不适用」）。
- 新建 **`sources/sites/ace-data-0-github-io.md`**、**`sources/datasets/ace-data-0.md`**。
- 交叉更新 [paper-data-pyramid-embodied-manipulation](../../wiki/entities/paper-data-pyramid-embodied-manipulation.md)、[paper-ego4d](../../wiki/entities/paper-ego4d.md)、[rekadaily-10k-dataset](../../wiki/entities/rekadaily-10k-dataset.md)、[egoscale](../../wiki/methods/egoscale.md)、[paper-ace-brain-0-5](../../wiki/entities/paper-ace-brain-0-5.md)、[roadmap/depth-vla](../../roadmap/depth-vla.md)。
