# Track4World: Feedforward World-centric Dense 3D Tracking of All Pixels

> 来源归档（ingest）

- **标题：** Track4World: Feedforward World-centric Dense 3D Tracking of All Pixels
- **类型：** paper
- **来源：** arXiv
- **原始链接：**
  - <https://arxiv.org/abs/2603.02573>
  - 项目页：<https://jiah-cloud.github.io/Track4World.github.io/>
  - 代码：<https://github.com/TencentARC/Track4World>
  - 权重：<https://huggingface.co/TencentARC/Track4World>
- **机构：** 香港科技大学（HKUST）；腾讯 ARC Lab（Tencent ARC / PCG）
- **会议：** ECCV 2026（Accepted）
- **入库日期：** 2026-09-15
- **一句话说明：** 在 VGGT 风格全局 3D 场景表示上，用 **2D-to-3D correlation** 前馈估计任意帧对的像素级 2D/3D 稠密流，并融合为世界坐标系下 **每个像素** 的稠密 3D 轨迹。

## 核心论文摘录（MVP）

### 1) 动机：从稀疏/优化式跟踪到前馈全像素 3D 对应

- **链接：** <https://arxiv.org/abs/2603.02573>
- **摘录要点：** 单目视频 **每个像素的 3D 轨迹** 是理解 3D 动力学的关键；既有工作要么只跟踪首帧稀疏点，要么用慢速优化框架做稠密跟踪。Track4World 提出 **前馈模型**，在 **世界坐标系** 下高效完成 holistic 3D tracking。
- **对 wiki 的映射：**
  - [paper-track4world](../../wiki/entities/paper-track4world.md) — 问题定义与定位
  - [D4RT](../../wiki/entities/paper-d4rt.md) — 同为动态时空对应，但 D4RT 用查询解码、截至入库日未开源

### 2) 方法：VGGT 式表示 + 2D-to-3D correlation + 流融合

- **链接：** <https://jiah-cloud.github.io/Track4World.github.io/>
- **摘录要点：** 输入视频 → 提取全局场景表示（几何嵌入、点云、相机位姿）→ **sparse-to-dense scene flow decoder** 用 **2D-to-3D correlation** 同时估计 2D/3D 联合流 → 融合 pairwise flow 得到世界系稠密 3D 跟踪。支持相机系与世界系两套输出。
- **对 wiki 的映射：**
  - [paper-track4world](../../wiki/entities/paper-track4world.md) — 流程总览 Mermaid
  - [state-estimation](../../wiki/concepts/state-estimation.md) — 前馈几何谱系

### 3) 实验：流估计、3D/2D 跟踪与相机位姿

- **链接：** 项目页 Performance Metrics
- **摘录要点：** Kubric-3D 上 EPE3D **0.1537**、AccS **0.5494**（short val）等显著优于 RAFT-3D / POMATO / Any4D；TAPVid-3D 世界系 APD 平均 **0.5636**（L-16）领先 SpatialTrackerV2 / POMATO；2D 跟踪在 Kinetics/RoboTAP/RGB-S 上 AJ **59.1 / 70.9 / 78.2** 超越 CoTracker3；相机位姿 Bonn ATE **0.009**。
- **对 wiki 的映射：**
  - [paper-track4world](../../wiki/entities/paper-track4world.md) — 评测表与结论

### 4) 开源与 WorldTrack 公平对比

- **链接：** <https://github.com/TencentARC/Track4World>
- **摘录要点：** 发布 `demo.py`、评测与可视化；HF 提供 DA3/Pi3/MoGe 三变体权重。README 含与 **OpenD4RT** 在 WorldTrack 子集上的 **同图像信息量** 对比协议（256×256 瓶颈、仅换 predictor）。
- **对 wiki 的映射：**
  - [Track4World 仓库](../repos/track4world.md) — 工程入口
  - [paper-track4world](../../wiki/entities/paper-track4world.md) — 开源状态与复现路径
