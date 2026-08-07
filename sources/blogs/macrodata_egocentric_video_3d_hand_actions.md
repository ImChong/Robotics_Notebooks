# Macrodata：Turning Egocentric Video into 3D Hand Actions

> 来源归档（blog）

- **标题：** Turning Egocentric Video into Robot Actions / Turning Egocentric Video into 3D Hand Actions
- **类型：** blog / official engineering
- **来源：** Macrodata Labs 官方博客
- **原始链接：** <https://macrodata.co/blog/turning-egocentric-video-into-3d-hand-actions>
- **公司页：** <https://macrodata.co/>（归档见 [macrodata-co.md](../sites/macrodata-co.md)）
- **作者/机构：** Macrodata Labs
- **发布日期：** 2026-08-06
- **入库日期：** 2026-08-07
- **一句话说明：** 用 **RGB-only** 开源组件管线把 egocentric 视频变成 **度量世界系双手 21 关节轨迹**（可作机器人动作监督）；在 10 段 HOT3D Aria episode 上以 **Action MPJPE** + ≥75% 直接覆盖 + ≥15 FPS@H100 为门槛，最终 **52.04 mm / 81.23% / 15.53 FPS**。

## 开源状态（步骤 2.5）

- **博客配方：** **部分可复现** — 组件均为公开上游（WiLoR、HaWoR、VGGT-Omega 等），但 Macrodata **未**发布其端到端编排仓与专有检测器。
- **产品侧：** **确认未开源** — 文中声明另有 proprietary hand-tracking；入口为 Contact / 免费样例标注。详见 [macrodata-co.md](../sites/macrodata-co.md)。

## 核心摘录

### 1) 问题设定与动作表示

- **动机：** 遥操作数据干净但难规模化；普通 egocentric 视频（YouTube 量级）便宜但只有像素、没有策略要预测的 **action**。
- **RGB-only：** 推理不使用测量深度 / LiDAR / 立体 / IMU（即便可用）；相机标定若提供可用。
- **目标表示：** 每手 **21 个度量 3D 关节**（1 腕 + 20 指），机器人无关；可再派生夹爪/重定向表示。对照：HumanEgo（虚拟平行夹爪）、EgoScale（重定向到 22-DoF 机器人手）、ViTra（腕增量 + MANO 参数）。
- **世界系融合：** 逐帧相机系手位姿必须经 **相机轨迹** 变换到共享世界系，否则把头动混进手动。
- **VLA 评测切片：** 从每帧 \(t\) 起取 **1 秒 action chunk**，把未来世界系轨迹变回 \(t\) 时刻相机系，以对齐「观测 → 预测动作」训练接口。

### 2) 基准：HOT3D + Action MPJPE

- **选用 HOT3D**（Banerjee et al., 2025；[HF](https://huggingface.co/datasets/projectaria/hot3d)）：光学标记真值，含度量双手 + 相机位姿 + 可见性；相对 Ego4D / AssemblyHands / H2O / Ego-Exo4D 更匹配长程轨迹评测。
- **评测集：** 10 段完整 Project Aria episode；**19,350** 帧 @30 Hz（约 **10.75 min**）。
- **资格门槛：** (1) HOT3D 可见帧上 **直接预测覆盖 ≥ 75%**；(2) 端到端吞吐 **≥ 15 FPS on H100**（约 ≤2 H100·h / 视频小时）。
- **缺失帧：** 评测用线性插值 / 边界外推填密后再算误差；不奖励「只报简单帧」。
- **Action MPJPE：** 标准 MPJPE 扩展到 1 秒相机相对轨迹；**不**做 Procrustes 对齐/缩放；保留尺度、相机相对运动与手运动误差。单位 mm，越低越好。

### 3) 基线 HaWoR（Zhang et al., 2025）

管线：WiLoR 检测 → 时序 HaWoR MANO 重建 → DROID-SLAM（手像素掩蔽）→ Metric3D 中位数尺度 → 世界系融合 →（可选）learned motion infiller。

| 缺失帧处理 | Action MPJPE | Coverage | FPS |
|------------|--------------|----------|-----|
| 基准 gap-filling | **59.12 mm** | 87.11% | **3.34** |
| HaWoR motion infiller | 60.71 mm | ~99.99% | — |

- 运行瓶颈：**相机重建 + 度量尺度 ≈ 61.7%** 运行时；手重建 24.6%；检测 12.3%。
- Learned infiller 在该集上 **抬高** Action MPJPE（相对线性插值）。

### 4) 改进 Part 1 — 窗口化 VGGT-Omega 世界重建

替换 DROID-SLAM+Metric3D。朴素无重叠 VGGT 拼接 → **90.73 mm**（坐标系/尺度断裂）。经消融选定：

| 设定 | 取值 |
|------|------|
| 窗口长度 | **200** 帧 |
| 重叠 | **40** 帧 |
| 输入分辨率桶 | **416 px** |
| 窗间对齐 | **depth-derived Sim(3)** + 重叠区线性混合 |
| 仅相机阶段 Action MPJPE | **55.60 mm** @ 全管线仍 **≥15 FPS** |

其它相机系统对照（同 HaWoR 手）：ViPE 61.44 mm / 9.58 FPS；MapAnything 61.86 mm / 1.59 FPS；MegaSAM+MoGe-2 71.33 mm / 7.66 FPS。

### 5) 改进 Part 2 — 保守 WiLoR 检测 / 跟踪

- 保留 WiLoR；**不用** HaGRID-YOLOv10n（可见手覆盖仅 51.8%）。
- **置信阈值 0.75**（Action MPJPE 最优且覆盖 >75%）。
- **短间隙恢复：** 同侧 ≤4 帧间隙内，仅当两侧有高置信锚点且弱框与插值框 IoU ≥ **0.20** 才接纳 0.10–0.75 提议。
- ByteTrack / BoT-SORT / SAM2 / EdgeTAM 等更激进关联 **未**改进质量–速度前沿（多找回难帧但 3D 重建更差）。
- 选定覆盖：**81.23%**。

### 6) 手重建模型对照（固定其余管线）

| 方法 | Action MPJPE | Coverage | FPS |
|------|--------------|----------|-----|
| **HaWoR**（时序） | **53.75 mm** | 89.14% | 15.01 |
| WiLoR（逐帧） | 77.35 mm | 89.14% | 21.84 |
| HaMeR | 76.99 mm | 79.10% | 4.28 |
| HaPTIC | 未端到端 | — | <2.08 |
| MediaPipe + DA3 Metric Large | 87.80 mm | 37.00% | 12.7 |

HaWoR 使用 **16 帧窗、8 帧重叠** + 线性/球面混合。

### 7) 后处理（保留投影一致的窄修正）

- **拒绝**直接对手关节做 mean/Gaussian 平滑（降 jitter 但 **抬高** Action MPJPE）。
- **保留：** 相机平移 3 帧二项滤波；clip 级骨长均值尺度（±3.5% 界）+ 沿原相机射线的腕深优化（加速度权重 λ=**0.2**，置信度加权）。
- 后处理：Action MPJPE **53.70 → 52.07 mm**；加速度误差 **12.73 → 7.22 mm/frame²**。
- Learned infiller / EgoInfinity 相对评测器线性插值 **无稳定优势**；缺失帧在导出标注中保持显式缺失。

### 8) 端到端最终结果与误差归因

| 系统 | Action MPJPE | Coverage | FPS |
|------|--------------|----------|-----|
| 官方 HaWoR（infiller off） | 59.12 mm | 87.11% | 3.34 |
| 初始 VGGT-Omega + HaWoR | 90.73 mm | 89.14% | 25.01 |
| **Macrodata 选定开源配方** | **52.04 mm** | **81.23%** | **15.53** |

相对 HaWoR：**误差 −12.0%**，吞吐 **3.34 → 15.53 FPS**。

直接预测子集 Shapley 归因（total 39.20 mm）：

- 相机系手预测 **32.35 mm**（主导）vs 相机轨迹 **6.86 mm**
- 腕平移（含单目深度）**18.08 mm**；深度轴约占直接误差 **~43%**
- 大相机运动显著恶化相机误差（平移 >10 cm 或旋转 >10° 时误差跳升）

### 9) 部署失败模式（客户数据）

HOT3D 未覆盖：穿戴者识别、手套、腕部相机等。开源检测器易把 **非穿戴者手** 检入，或在手套/腕机上完全失败 → Macrodata 自研检测器（未开源）。

## 对 wiki 的映射

- [macrodata-egocentric-hand-action](../../wiki/methods/macrodata-egocentric-hand-action.md) — 方法页（新建）
- [WiLoR](../../wiki/methods/wilor.md) — 检测前端
- [hawor](../repos/hawor.md) — 时序手重建上游
- [EgoScale](../../wiki/methods/egoscale.md) — egocentric → 高 DoF 手监督的另一缩放叙事
- [ViDiHand](../../wiki/entities/paper-vidihand.md) — egocentric 双手重建对照
- [auto-labeling-pipelines](../../wiki/methods/auto-labeling-pipelines.md) — 数据引擎总览（语义分段轴旁路）
- [perceptron-egocentric](../../wiki/entities/perceptron-egocentric.md) — Macrodata WGO 语义标注对照
- [ego-category-01 / 02](../../wiki/overview/ego-category-01-data-collection.md) — Ego 采集与人→机

## 当前提炼状态

- [x] 博客核心数字、配方与开源边界
- [x] wiki 方法页与交叉索引
- [ ] 若 Macrodata 公开编排仓，补源码运行时序图
