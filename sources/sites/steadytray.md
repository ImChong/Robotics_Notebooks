# SteadyTray 项目页

- **URL：** <https://steadytray.github.io/>
- **关联论文：** [steadytray_arxiv_2603_10306](../papers/steadytray_arxiv_2603_10306.md)
- **实体页：** [paper-notebook-steadytray](../../wiki/entities/paper-notebook-steadytray.md)
- **代码：** <https://github.com/AllenHuangGit/steadytray> — 归档见 [`sources/repos/steadytray.md`](../repos/steadytray.md)
- **IsaacLab 依赖 fork：** <https://github.com/AllenHuangGit/IsaacLab_SteadyTray> — 归档见 [`sources/repos/isaaclab-steadytray.md`](../repos/isaaclab-steadytray.md)
- **核查日期：** 2026-09-14

## 开源状态（步骤 2.5）

| 组件 | 状态 |
|------|------|
| 项目页 | 已上线（方法图、真机演示、扰动恢复） |
| 训练管线 | **已开源** — `AllenHuangGit/steadytray`（四阶段 RSL-RL + PPO） |
| IsaacLab 环境 | **已开源** — `AllenHuangGit/IsaacLab_SteadyTray` fork |
| 预训练权重 | **已提供** — `model/model_9999.pt`（学生策略） |
| MuJoCo sim2sim | **已开源** — `deploy/deploy_mujoco/` |
| 真机部署 | README 侧重 sim2sim；真机零样本由论文/项目页展示 |

**结论：已开源**（训练 + sim2sim + 预训练 checkpoint；真机栈以论文为准）。
