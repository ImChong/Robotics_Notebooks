# PointZero（arXiv:2609.19142）

> 来源归档（paper）

- **标题：** PointZero: 3D Point Track Completion for Learning Transferable 3D Dynamics
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2609.19142>
- **PDF：** <https://arxiv.org/pdf/2609.19142>
- **代码：** https://github.com/Duisterhof/pointzero
- **项目页：** https://pointzero-wm.github.io/
- **入库日期：** 2026-09-20（初稿）；2026-09-21（项目页直 ingest 补全）
- **一句话说明：** 3D 点轨迹补全预训练可迁移 3D 动力学；290 万合成帧；后训练用于 PGND 动作条件动力学与 7 任务模仿学习。

## 开源状态

- **待发布**（步骤 2.5，2026-09-21）：项目页链 GitHub 但 README 写 release coming soon；Dataset 按钮 Coming Soon。

## 核心摘录

1. **问题：** 现有动作条件 3D 动力学方法依赖机器人动作标签，无法利用 web 视频；PointZero 用点轨迹补全作 robot-free 预训练目标。
2. **输入/输出：** 单帧 RGB-D + 少量完整 3D point tracks → 预测所有观测点的未来 3D 轨迹。
3. **架构：** DINOv2 → Perceiver-IO 编码 + 去噪 Transformer（point/global attention）；Flow Matching 变体在合成 held-out 上全面优于 GBND / ParticleFormer / PGND / PTv3 等同数据基线。
4. **数据：** 290 万合成帧（deformable / articulated / rigid）；真机评测 14 物体 124 交互（FoundationStereo + CoTracker3 标注）。
5. **PGND 后训练：** PointZero-FT 在 6 个真机–物体场景中 4 个 MDE/CD/EMD 最优；Scratch 同架构显著更差。
6. **IL 后训练：** 20 labeled + 100 actionless demos/任务；Blockstack 99.8%、Drawer/Cup 100% 等；7 任务中 6 个 best/joint-best，全面优于 DP3。
7. **预训练价值：** 同架构 Scratch vs Pretrained，有/无下游 track 监督均显示预训练增益。

**对 wiki 的映射**

- [paper-pointzero](../../wiki/entities/paper-pointzero.md)
