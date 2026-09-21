# PointZero 项目页

> 来源归档（site）

- **标题：** PointZero: 3D Point Track Completion for Learning Transferable 3D Dynamics
- **类型：** site
- **链接：** https://pointzero-wm.github.io/
- **arXiv：** <https://arxiv.org/abs/2609.19142>
- **代码：** https://github.com/Duisterhof/pointzero
- **入库日期：** 2026-09-20（初稿）；2026-09-21（项目页直 ingest 补全）
- **一句话说明：** 以 RGB-D + 稀疏 3D 点轨迹补全为预训练目标，在无机器人动作标签下学习可迁移 3D 动力学；290 万合成帧 + 真机评测集；后训练用于动作条件动力学与模仿学习。
- **沉淀到 wiki：** [`wiki/entities/paper-pointzero.md`](../../wiki/entities/paper-pointzero.md)

## 机构与作者

| 作者 | 机构 |
|------|------|
| Bardienus P. Duisterhof, Adam Hung, Deva Ramanan, Jeffrey Ichnowski | Carnegie Mellon University |
| Kaifeng Zhang, Yunzhu Li | Columbia University |
| Bowen Wen, Stan Birchfield | NVIDIA |

## 开源状态（步骤 2.5，2026-09-21 再核）

- **代码：** 项目页链至 [Duisterhof/pointzero](https://github.com/Duisterhof/pointzero)，但 README 写 **「Release coming soon」** — 截至入库日 **待发布**。
- **数据集：** 项目页按钮标 **「Dataset (Coming Soon)」** — **待发布**。
- **Checkpoints / 训练配方：** 摘要与项目页宣称将发布，但页上尚无下载链接 — **待发布**。
- **论文 PDF：** [arXiv:2609.19142](https://arxiv.org/abs/2609.19142) 可公开获取。

## 项目页核心摘录

1. **预训练目标：** 给定单帧 RGB-D 与稀疏完整 3D 点轨迹，预测场景中所有观测点的未来 3D tracks；监督仅需 3D 点轨迹，无需机器人动作标签，可用仿真或（原则上）视频点跟踪数据。
2. **架构：** Perceiver-IO 视觉编码器（DINOv2 特征）+ 去噪 Transformer（交替 point-level 与 global attention）；Flow Matching / JiT 优于直接回归。
3. **合成数据：** 290 万帧，密集 per-point 轨迹标注，覆盖可变形 / 关节 / 刚体；随机交互驱动运动。
4. **真机评测集：** 14 物体、124 次交互；轨迹标签来自 FoundationStereo + CoTracker3，用于 zero-shot sim-to-real 评测。
5. **下游 (1) 动作条件动力学：** 微调节点于末端执行器位姿，在 PGND 基准 6 个真机场景中 4 个优于强基线；同架构从零训练（Scratch）显著落后。
6. **下游 (2) 模仿学习：** 轻量 action head；每任务 20 条带动作示范 + 100 条无动作示范视频；7 个仿真+真机任务中 6 个达或超基线，全面优于 DP3。
7. **预训练消融：** 有下游 track 监督时平均成功率 88.2% vs Scratch 80.5%；冻结预训练点流时 80.0% vs 74.1%。

## 对 wiki 的映射

- [paper-pointzero](../../wiki/entities/paper-pointzero.md)
