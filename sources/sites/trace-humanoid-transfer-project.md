# TRACE 项目页（bfm-trace.github.io）

> 来源归档

- **标题：** TRACE: Efficient Transfer Learning for Neural Whole-Body Control Across Humanoid Embodiments
- **类型：** site（项目页）
- **URL：** <https://bfm-trace.github.io/>
- **论文：** <https://arxiv.org/abs/2610.17999>（归档：[trace_humanoid_transfer_arxiv_2610_17999.md](../papers/trace_humanoid_transfer_arxiv_2610_17999.md)）
- **代码：** <https://github.com/QinnyuLan/SONIC_unitree_h2>（见 [trace-sonic-unitree-h2.md](../repos/trace-sonic-unitree-h2.md)）；无 Hugging Face 链接
- **入库日期：** 2026-10-17
- **一句话说明：** 把预训练 SONIC 从 Unitree G1 迁移到 H2，约 8 GPU 小时，PHUMA 成功率 96.08%（源 G1 为 97.59%）。

## 页面结构

| 区块 | 内容要点 |
|------|----------|
| Method | 对齐本体 → 筛选运动 → 高效适配（蒸馏 + LoRA + 两阶段 PPO）→ 跨任务部署 |
| Results | 成功率、消融（87.29 / 93.80 / 95.64%）、训练曲线、误差阈值 |
| Real robot | VR 遥操作与 GR00T-N1.7 后训练五任务（65–90%，每任务 20 次） |
| Demo / Video | 在线 demo 页与 2:45 概览视频 |

## 开源核查结论

截至 2026-10-17：列出官方 GitHub 仓库，未写 "coming soon"；仓库细节见 repos 归档。

## 对 wiki 的映射

- [wiki/entities/paper-trace-humanoid-transfer.md](../../wiki/entities/paper-trace-humanoid-transfer.md)
- [sources/papers/trace_humanoid_transfer_arxiv_2610_17999.md](../papers/trace_humanoid_transfer_arxiv_2610_17999.md)
