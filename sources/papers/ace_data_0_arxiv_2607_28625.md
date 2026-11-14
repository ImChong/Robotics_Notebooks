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
- **入库日期：** 2026-08-10（2026-11-14 核对 v1 全文、项目页及 HF 元数据）
- **核对范围：** 2026-11-14 通读 arXiv:2607.28625v1 全文（摘要、目录、§1–§6、Tables 1–6、Figs 1–12、Refs [1]–[116]），并核对项目页 HTML（1,052 行）及 HF `ACERobotics/ACE-Data-0` API JSON 与 README（379 行）。
- **通读：** 已完成
- **一句话说明：** 用 Ambient Capture Engine 把真实家居变成时空校准的录制工作室（table-scale ~30 m² + room-scale ~200 m²），发布 **150+ h / 17M+ 帧 / 75,000+ episodes** 的同步多模态家居活动集 ACE-Data-0，并配套「信号→场景组件→交互」三层感知诊断 benchmark（10 h 测试集）。

## 开源状态（步骤 2.5）

核查日：**2026-11-14**（实测项目页 HTML / Hugging Face API + README / arXiv v1 全文）。

| 产物 | 状态 |
|------|------|
| Hugging Face 数据集 `ACERobotics/ACE-Data-0` | **已上传且 API 实测未启用平台门控（`gated: false`）** — 仓内已上传 `~22.18 TB`（22,176,330,852,720 bytes）、`≥100,000` 个文件（`lastModified: 2026-10-14`，含 `Room-scale/` 数据目录；API listing 在 100k 条截断），许可为 `ace-data-0-research-license`（非商业学术研究）。**注意：** 其 `README.md` 正文与 YAML 仍残留 `extra_gated_*` 申请表、`Data_Files-Coming_Soon` 徽章与「Data files are not published yet」过期提示 |
| 项目页 | **已开放** — 双尺度硬件/同步叙事、交互示例轮播、任务分类与 benchmark 说明 |
| 训练 / 评测代码、仓库 | **未公开** — 项目页与论文未列 GitHub；HF 仓内除数据集外无统一评测/训练代码（`Room-scale/calibration/` 含标定数据与 Homie 标定脚本说明） |
| 模型权重 | **不适用** — 本篇为数据引擎 + 数据集 + 基准，非新架构权重 |

**结论：** **部分开源（数据主仓已在 HF 上线且 API 当前 `gated: false`，非商业学术研究许可；官方训练与统一评测代码未发布）**。源码运行时序图写 **不适用**。

## 摘录 1：问题设定与 ACE 双尺度范式（§1–§3）

- **瓶颈：** 具身学习需要的是「第一人称感知 + 全身运动 + 灵巧操作 + 物体状态 + 声音 + 触觉」随目标推进的**同步演化**；现有集在视角 / 模态 / 空间尺度上碎片化，感知–动作闭环只被部分观测。
- **三条短板（Table 1 对照）：** (1) 模态碎片——大规模 ego 集缺度量运动与外视；(2) 环境不自然——MoCap HOI 多在空旷实验室；(3) 视野短——秒级原子动作 vs 分钟级家居目标链。
- **ACE：** 两互补配置共用同一同步标定管线——
  - **Table-scale：** 近距相机 + 光学动捕 + 触觉手套，解析细粒度手–物接触；
  - **Room-scale：** 铺满家具的公寓（厨 / 餐 / 客 / 卧），宽基线相机跟踪跨房间 locomotion 与家具接触。
- **传感栈：** ≥8 外视 RGB；4 路鱼眼 ego + IMU + 头显位姿；全身 + 关节手 60 Hz；物体 mesh 与 6-DoF；多通道音频；全掌触觉压力图。

**对 wiki 的映射：** 升格 [`wiki/entities/paper-ace-data-0.md`](../../wiki/entities/paper-ace-data-0.md)；金字塔第 ③ 层「人类 Ego/Exo」对照 [Data Pyramid](../../wiki/entities/paper-data-pyramid-embodied-manipulation.md)；与规模向 ego [Ego4D](../../wiki/entities/paper-ego4d.md)、[RekaDaily-10k](../../wiki/entities/rekadaily-10k-dataset.md)、[EgoScale](../../wiki/methods/egoscale.md) 区分「度量同步完备 vs 小时规模」。

## 摘录 2：ACE-Data-0 规模、协议与标注（§3–§4）

- **规模：** 150+ 小时（1 h 原始约 1 TB）· 17M+ 帧 · 200+ 任务类 · 50+ 3D/2DGS 扫描物体 · 50 参与者（2 天录制）· 2 环境 · 75,000+ interaction episodes。
- **协议：** **目标级指令**（goal-level）而非逐步脚本——保留路线选择、抓取策略、犹豫与恢复等自然变异；长程行为由任务完成本身涌现，而非拼接短 clip。
- **三类活动：** Atomic HOI（30k+ eps，54 h，6.2M+ 帧 = 36.5%，约 3 min）/ Chain of HOI（40k+ eps，64 h，7.3M+ 帧 = 42.9%，20–30 min）/ HSI（30 h，3.5M+ 帧 = 20.6%，约 5 min）。
- **同步与标定（Table 2）：** 共 28 台 OptiTrack PrimeX 22（桌面 16 / 房间 12 @ 60 Hz）；8× ZED One（房间 GMSL2 + Jetson Orin，NatNet + 纳秒 QR 光时钟 <4 ms 残差）与 8× GoPro（桌面音频+QR 时钟）；`ACE-Ego-Head-V02 Lite` 4 鱼眼 1088×1280 @ 20 FPS（<2 ms 偏差）+ IMU + 5 markers（~2 px 手眼重投影）；外视以反光 ArUco 板桥接（<3 px，~1 cm 3D）；`ACE-Sense-Glove Lite` 触觉手套通过 IMU 运动模式同步；房间 Manus 手套 60 Hz，桌面手姿经 8 视角 RANSAC 2D 三角化+人工精修。
- **标注（§4.3）：** 相机内外参、41 关节 SMPL-X、60 Hz 物体 6-DoF、手势与接触标签均直接由测量动捕/标定/触觉传感器导出；**仅自然语言描述**由 `Gemini-3.1-pro-preview` 生成并经人工校验修正。

**对 wiki 的映射：** 数据集速查与流程总览进实体页；许可与门控细节见 [`sources/datasets/ace-data-0.md`](../datasets/ace-data-0.md)。

## 摘录 3：三层 benchmark 与评测要点（§5）

层级设计：**信号 → 场景组件 → 交互**（诊断「感知–理解–交互」链条何处断裂）。

1. **触觉 from vision（ego，Table 3，10 h 留出测试）：** TouchAnything 相对最强（Temp Acc **0.7095** / C-IoU **0.1646** / V-IoU **0.1357** / CoP **6.5846**），但绝对重叠仍低；PressureVision 几乎失效——**「何时接触」易、「何处加压」难**。
2. **人体运动恢复（room-scale，Table 4，5 组 22 方法）：** 局部姿态可接近常规基准（SMPLest-X PA-MPJPE **55.7 mm**；Human3R **60.1 mm**）；**世界系轨迹（WA-MPJPE）显著更差**（Human3R **75.4 mm**，多帧/scene-aware 仍常高出、EgoAllo **220.3 mm**）；ego 方法因身体出画最弱。
3. **手运动（ego Table 5 / exo Table 6）：** exo 关节精度更好（WiLoR PA-MPJPE **9.1 mm**）；世界系轨迹 exo（HaPTIC **10.0 mm** PA-MPJPE / **63.0 mm** Traj err）明显优于 ego 世界系方法（Dyn-HaMR **98.2 mm** / HaWoR **102.1 mm**；WildHands PA-MPJPE **11.2 mm**）——**egomotion 估计是主要误差源**；双视角互补（ego 近距细节 vs exo 全局与遮挡）。

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
