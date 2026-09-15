# Track4World — 官方项目页

- **来源**：https://jiah-cloud.github.io/Track4World.github.io/
- **类型**：site（项目页 / ECCV 2026）
- **机构**：香港科技大学（HKUST）；腾讯 ARC Lab（Tencent ARC / PCG）
- **归档日期**：2026-09-15
- **论文**：arXiv:2603.02573 — *Track4World: Feedforward World-centric Dense 3D Tracking of All Pixels*
- **会议**：ECCV 2026（Accepted）
- **代码**：https://github.com/TencentARC/Track4World
- **权重**：https://huggingface.co/TencentARC/Track4World（Tencent Green License）
- **开源结论（步骤 2.5）**：**已开源** — GitHub 推理/评测脚本 + HF 权重（`track4world_da3.pth` / `track4world_pi3.pth` / `track4world_moge.pth`）；依赖 MoGe、Pi3、Depth Anything 3 等第三方子模块

## 一句话说明

**Track4World** 从单目视频 **前馈** 估计 **世界坐标系下每个像素的稠密 3D 轨迹**：在 VGGT 风格 ViT 全局场景表示上，用 **2D-to-3D correlation** 同时预测任意帧对的像素级 2D/3D 稠密流，再融合为 holistic world-centric 3D tracking。

## 为什么值得保留

- 相对 **稀疏首帧跟踪** 或 **慢速优化式稠密跟踪**，把「全像素 3D 对应」写成 **单次前馈**
- 项目页与论文在 **Kubric-3D / KITTI / BlinkVision** 流估计与 **PointOdyssey / ADT / TAPVid-3D** 跟踪上报告 SOTA 级结果
- 提供 **相机系 / 世界系** 两套可视化与 **WorldTrack** 上对 OpenD4RT 的公平对比协议

## 核心能力（项目页归纳）

| 模式 | 说明 |
|------|------|
| `3d_ff` | 首帧 3D 运动重建 |
| `3d_efep` | 稠密每像素 3D 跟踪（camera / world 坐标） |
| `2d` | 标准 2D 稠密跟踪 |
| 骨干变体 | Depth Anything 3（`da3`）、Pi3（`pi3`）、MoGe（`moge`） |

## 对 wiki 的映射

1. **[paper-track4world（论文实体）](../../wiki/entities/paper-track4world.md)** — 方法、评测与开源入口
2. **[D4RT（对照）](../../wiki/entities/paper-d4rt.md)** — 同为动态 4D/3D 跟踪统一接口；D4RT 截至入库日未开源
3. **[state-estimation（概念页）](../../wiki/concepts/state-estimation.md)** — 前馈几何与跟踪谱系

## 关联原始资料

- [Track4World 论文摘录](../papers/track4world_arxiv_2603_02573.md)
- [Track4World 仓库](../repos/track4world.md)
