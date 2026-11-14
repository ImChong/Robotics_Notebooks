---
type: concept
tags: [bev, bird-eye-view, spatial-representation, perception, sensor-fusion, autonomous-driving, camera, lidar, radar, occupancy]
status: complete
updated: 2026-11-14
related:
  - ./embodied-perception-six-spatial-representations.md
  - ./perception-coordinate-postprocessing.md
  - ./sensor-fusion.md
  - ../formalizations/3d-coordinate-transforms-vision-robotics.md
  - ../entities/paper-msfp-embodied-ai-survey.md
  - ../overview/autonomous-driving-core-algorithms-series.md
  - ../entities/paper-uniad.md
  - ../entities/paper-sparsedrive.md
original:
  - ../../sources/blogs/wechat_shenlan_ai_bev_unified_spatial_fusion_2026_11_14.md
sources:
  - ../../sources/blogs/wechat_shenlan_ai_bev_unified_spatial_fusion_2026_11_14.md
summary: "BEV（Bird's-Eye View）将多相机、LiDAR、毫米波雷达特征落到统一鸟瞰栅格，实现语义、几何、运动信息的空间对齐；解释像素升维、栅格池化、跨模态融合及检测/分割/跟踪，并区分深度、径向速度、跨帧对齐等工程边界。"
---

# BEV：鸟瞰统一空间表征（Bird's-Eye-View Representation）

## 一句话定义

**BEV（Bird's-Eye View，鸟瞰表征）**是在选定的自车/机器人水平参考系中，用二维 \((x,y)\) 栅格及其多通道特征编码周边环境：**同一空间格内对齐相机的语义、激光雷达的几何、雷达的运动信息，让感知头在统一物理平面上工作**。可把它理解为原文所说的「一张棋盘，一套规则」。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| BEV | Bird's-Eye View | 鸟瞰特征地图；将位置对齐到水平空间栅格 |
| LiDAR | Light Detection and Ranging | 激光雷达；测距点云提供几何与反射强度 |
| IPM | Inverse Perspective Mapping | 逆透视映射；通常依赖地面平面假设 |
| LSS | Lift, Splat, Shoot | 图像特征按预测深度提升，再聚合到 BEV |
| IoU | Intersection over Union | 两个目标框/区域的交并比，关联线索之一 |
| MLP | Multi-Layer Perceptron | 多层感知机；可学习融合门控或权重 |

## 为什么重要

- **统一空间基准：** 图像像素的透视尺度、点云的米制坐标、雷达的距离/速度不能直接逐元素相加；先投到同一栅格，可减少空间错位。
- **融合互补线索：** 相机对类别/纹理敏感，LiDAR 对形状与距离敏感，毫米波雷达提供测速信息；BEV 让一个区域汇总多种特征，但不保证每种传感器在所有天气下都可靠。
- **共享多任务骨干：** 同一鸟瞰特征可供车道/可行驶区域分割、3D 目标检测、跟踪/轨迹预测及规划消费；仍需各任务的输出头及必要的时序状态。
- **机器人迁移：** 地面导航、道路感知、动态避障适合水平 BEV；人形/机械臂的接触、伸手、楼梯和空间重叠则可能需要补充 3D 高度/体素信息。

## 核心原理

### 1. 统一坐标，再量化栅格

约定所有输入先转换到同一**当前时刻的自车坐标系**（例如 \(x\) 向前、\(y\) 向左、\(z\) 向上；仅作示例，具体以标定为准）。对于投影得到的点 \((x,y,z)\)，栅格索引为：

\[
i=\left\lfloor\frac{x-x_{\min}}{\Delta x}\right\rfloor,\qquad
j=\left\lfloor\frac{y-y_{\min}}{\Delta y}\right\rfloor.
\]

其中 \(\Delta x,\Delta y\) 为网格米制分辨率；只有落在地图范围内的点参与更新。典型 BEV 张量为 \(F\in\mathbb{R}^{H\times W\times C}\)，二维索引负责位置，**高度、语义、密度、置信度等可放在通道 \(C\) 内**，并不是所有系统都完全丢掉高度。

### 2. 三类传感器如何落到同一「棋盘」

| 分支 | 原始观测 | 坐标/特征提升 | 栅格聚合与输出 | 关键限制 |
|------|----------|----------------|----------------|----------|
| 多相机 | 像素/图像特征 \(f(u,v)\) | 内参 + 外参；深度估计、射线/地面约束或可学习视图变换 | 同格池化/注意力得到语义特征 | 单目像素本身无唯一 3D 位置；遮挡与深度错误会错格 |
| LiDAR | \((x,y,z)\)、反射强度等 | 外参到自车系，离散成柱/体素 | 点数、强度、最大/均值高度及学习特征 | 稀疏、遮挡与空点不可直接等同自由空间 |
| 毫米波雷达 | 距离、方位、径向速度、检测质量 | 极坐标到自车系，时空补偿 | 运动/置信度特征或雷达 token | 径向速度不是天然的完整 \((v_x,v_y)\) |

**相机投影必须说明深度：** 对已去畸变的像素齐次坐标 \(\tilde u=(u,v,1)^\top\)，若已知相机坐标系下沿光轴深度 \(z_c\)，则

\[
p_c=z_cK^{-1}\tilde u,\qquad
p_{\mathrm{ego}}=R_{\mathrm{ego}\leftarrow c}p_c+t_{\mathrm{ego}\leftarrow c}.
\]

若没有 \(z_c\)，式子仅给出一条射线；使用 IPM 时需与**已知地面平面**求交；采用 LSS 等学习式图像到 BEV 的方法时则通过离散深度分布和 splatting 聚合。原文 PDF 第 4 页的逆投影式是教学简写，不能解读成「只有内外参就能还原三维点」。

**栅格聚合的抽象：**

\[
F_m(i,j)=\operatorname{Aggregate}
\left(\left\{f_m(q)\mid \operatorname{grid}(q)=(i,j)\right\}\right).
\]

\(m\) 表示传感器模态；Aggregate 可以是最大/均值池化、PointNet 类聚合、可学习注意力等。不同模态的输出通道未必天然相同，融合前常先对齐维度。

### 3. 跨模态融合：对齐位置不等于简单相加

原文第 6–7 页用三种层级帮助理解：

| 方式 | 示意做法 | 适用读法 |
|------|----------|----------|
| 固定加权 | \(\alpha F_{\rm cam}+\beta F_{\rm lidar}+\gamma F_{\rm radar}\) | 教学/基线；要求通道已对齐，固定权重难应对传感器失效 |
| 可学习加权 | 由网络根据输入和场景预测门控/权重 | 可按区域、通道、模态调整；需看权重学在哪一层 |
| 注意力融合 | 基于位置、查询与特征的自注意力/交叉注意力 | 建模空间/跨模态关联，计算量与位置编码设计需评估 |

**注意：** 此表是融合机制的通用分类，不是对 BEVFusion、UniBEV 等具体论文实现逐层复现；不能从科普文章的示意公式直接推断其网络细节。

### 4. 运行流程总览

\`\`\`mermaid
flowchart TB
    CAM["多相机图像<br/>像素语义特征"] --> LIFT["相机特征提升<br/>深度/地面几何约束"]
    LID["LiDAR 点云<br/>位置·高度·强度"] --> REG["外参/时间同步<br/>统一到自车坐标"]
    RAD["毫米波雷达<br/>位置·径向速度·置信度"] --> REG
    LIFT --> GRID["BEV 栅格化与空间聚合"]
    REG --> GRID
    GRID --> FUSE["多模态融合<br/>拼接/加权/注意力"]
    FUSE --> MAP["语义/可行驶区域分割"]
    FUSE --> DET["3D 目标检测"]
    DET --> TRACK["跨帧补偿与目标关联"]
    FUSE --> TRACK
    MAP --> DOWN["预测/规划/导航"]
    TRACK --> DOWN
\`\`\`

### 5. 多任务如何消费 BEV

- **分割：** 各栅格输出可行驶区域、车道等类别概率；细长线、远距目标的精度依赖栅格分辨率与语义恢复设计。
- **3D 检测：** 从 BEV 特征回归平面中心、尺寸、朝向、高度、类别等，**输出仍是 3D 框**，并非只能预测二维矩形。
- **目标跟踪：** 将上一帧目标通过自车运动补偿到当前坐标，再结合位置、速度、外观/类别、IoU 或运动模型做关联。原文第 9 页的 \(\operatorname{IoU}(Box_t,Box_{t-1})\) 是入门级关联示意，不能代替完整跟踪器。

## 工程实践

### 从零搭建与验证

1. **统一约定：** 写清 \(x/y/z\) 轴、时间戳、坐标系和每格米数；先单独验证 LiDAR 点云落在正确栅格。
2. **校准与同步：** 核对相机内外参、相机畸变、LiDAR/雷达外参与传感器时间差，使用标定物/静态目标做投影检查。
3. **分别调试分支：** 相机看提升深度/地面约束，LiDAR 看点占用与高度统计，雷达看径向速度符号及自车速度补偿。
4. **做缺模态试验：** 分别关闭相机、LiDAR、雷达，检查融合退化；比较固定融合与可学习门控是否对噪声敏感。
5. **检查跨帧：** 匀速直行、转弯时，把同一静态物体投到当前 BEV，测量格间漂移，再测跟踪 ID 切换与延迟。

| 排查现象 | 优先排查 |
|----------|----------|
| 多模态同一辆车出现两处 | 外参、时间戳、坐标系手性及尺度 |
| 图像特征远距被拉长 | 深度估计误差、地面假设和遮挡 |
| 静态路沿跨帧漂移 | ego-motion 补偿、位姿精度与更新时序 |
| 近距离上下叠层混为一格 | 2D BEV 高度压缩；改用高度通道/3D 体素 |
| 雷达目标速度方向不合理 | 径向速度解释、自车速度及坐标轴约定 |

**适用任务：** 轮式无人车道路占据/3D 检测、园区机器人局部导航、具有明显地面拓扑的人形避障；高度复杂的精细操作需要补充三维表征（见下方关联）。

## 局限与风险

- **BEV 是一种坐标组织/特征表征，并非一个固定算法**：相机提升方式、主干与融合头是可替换组件；不同方法未必都依赖 LiDAR/雷达。
- **二维栅格分辨率换取效率：** 格子越大，越难区分细长物体与接近目标；格子越小，显存/计算增大。
- **静态棋盘 ≠ 时序天然对齐：** ego-centric BEV 随自车运动而移动；历史帧需位姿变换、时钟同步与动态物体处理。
- **高空俯视 ≠ 完整三维占据：** BEV 可编码高度特征，但不直接等于具有明确空闲/占据/未知语义的 3D occupancy 或 TSDF/ESDF。
- **科普文章不是评测：** 本页主来源没有可运行实现或比较实验；「最大底牌」是解释性标题，不能据此宣称 BEV 在所有场景优于稀疏实例、体素或透视视图方案。可与 [SparseDrive](../entities/paper-sparsedrive.md) 的稀疏表征选型对照。

## 关联页面

- [具身感知六种空间表征](./embodied-perception-six-spatial-representations.md) — 区分 BEV 特征地图、占据/距离场、语义地图与隐式地图。
- [感知后处理与坐标变换](./perception-coordinate-postprocessing.md) — 像素到射线、深度与外参链。
- [传感器融合](./sensor-fusion.md) — 状态估计融合与感知任务特征融合不能混为一谈。
- [三维坐标变换](../formalizations/3d-coordinate-transforms-vision-robotics.md) — 相机/车体/世界位姿变换。
- [MSFP Survey](../entities/paper-msfp-embodied-ai-survey.md) — 多传感器感知技术分类。
- [自动驾驶核心算法盘点](../overview/autonomous-driving-core-algorithms-series.md) — 3D 检测、跟踪、预测场景。
- [UniAD](../entities/paper-uniad.md) — 规划导向的共享 BEV 感知头。
- [SparseDrive](../entities/paper-sparsedrive.md) — 使用稀疏实例替代稠密 BEV 的对照方向。

## 参考来源

- [深蓝AI：《BEV：统一空间表征，它是多模态融合最大的底牌》归档](../../sources/blogs/wechat_shenlan_ai_bev_unified_spatial_fusion_2026_11_14.md) — 本页主要编译来源；已通读用户提供的 12 页 PDF（2026-11-14），其中第 1–10 页为技术内容。公式边界与工程提醒为入库时的审阅补充。

## 推荐继续阅读

- [原始微信公众号文章（深蓝AI，2026-11-14）](https://mp.weixin.qq.com/s/LBcoaiHJ4X6P-v6hTq8bYA)
- [Lift, Splat, Shoot（Philion & Fidler）](https://arxiv.org/abs/2008.05711) — 视觉特征升维到 BEV 的代表性路线，需以论文原文核对细节。
- [BEVFormer](https://arxiv.org/abs/2203.17270) — 基于空间/时间注意力的多相机 BEV 表征，需以论文原文核对细节。
