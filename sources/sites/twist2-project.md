# TWIST2 项目页（yanjieze.com/projects/TWIST2）

> 来源归档

- **标题：** TWIST2: Scalable, Portable, and Holistic Humanoid Data Collection System
- **类型：** site（项目页 + 复现教程 + 实验视频）
- **URL：** <https://yanjieze.com/projects/TWIST2/>（旧路径 <https://yanjieze.com/TWIST2/> 重定向至此）
- **论文：** <https://arxiv.org/abs/2505.02833>
- **代码：** <https://github.com/amazon-far/TWIST2>
- **数据集：** <https://twist-data.github.io/>
- **入库日期：** 2026-06-12
- **一句话说明：** Amazon FAR 官方页：便携可扩展全身遥操作 + 2-DoF 颈增广 egocentric 感知 + PICO4U 全身流 + 分层 visuomotor（System1 跟踪 RL + System2 模仿学习）；全栈开源（颈 BOM/训练/部署/控制器权重）；ICRA 2026 接收。

## 页面结构（维护索引）

| 区块 | 内容要点 |
|------|----------|
| Introducing TWIST2 | 便携 / 整体 / 可扩展 / 自主四支柱；长时程全身灵巧 egocentric loco-manipulation 数据采集 |
| Reproduce on G1 | 五步：颈组装 → PICO 配置 → 控制器训练部署 → 遥操作采集 → 自主 visuomotor 策略 |
| Open Dataset | 全身 loco-manipulation 数据集开源；社区贡献入口 twist-data.github.io |
| TWIST2 Neck | 2-DoF 颈增广主动 egocentric 感知；MuJoCo 颈建模 |
| PICO Teleop | PICO 4 Ultra + 2 Motion Trackers；XRoboToolkit 统一 egocentric 视觉与全身姿态流 |
| Hierarchical Policy | 低层 tracking（sim2real RL，System 1）+ 高层 visuomotor 模仿（System 2，如 Diffusion Policy） |
| Scalable Collection | 15 min 内 128 次双手灵巧 pick-place；15 min 内 50 次移动 pick-place |
| Long-Horizon | 连续叠毛巾、找布折叠、踢球/篮球、穿门搬运、地面捡砖、绕圈等 |
| Autonomy | Kick-T、WB-Dex 等 visuomotor 自主技能；扩散策略预测全身关节位置 ghost 轨迹 |
| Related Works | TWIST（CoRL 2025 跟踪控制器）、GMR（重定向）、iDP3（3D 扩散 visuomotor 基座） |
| BibTeX | `@article{ze2025twist2,...}`（项目页标注 arXiv:2511.02832；知识库主索引沿用 2505.02833） |

## 对 wiki 的映射

- 主实体：[TWIST2（论文实体）](../../wiki/entities/paper-twist2.md)
- 论文摘录：[bfm_awesome_twist2_arxiv_2505_02833.md](../papers/bfm_awesome_twist2_arxiv_2505_02833.md)、[humanoid_rl_stack_10_twist2_scalable_portable_and_holistic_humanoid_d.md](../papers/humanoid_rl_stack_10_twist2_scalable_portable_and_holistic_humanoid_d.md)
- 代码仓库：[twist2.md](../repos/twist2.md)
