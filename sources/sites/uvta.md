# UVTA 项目页（Unified Visual-Tactile-Action）

- **标题：** UVTA — Unified Visual-Tactile-Action Modeling
- **类型：** site / project-page
- **URL：** <https://uni-vta.github.io/>
- **配套论文：** <https://arxiv.org/abs/2609.34182>
- **代码：** <https://github.com/uni-vta/UVTA>（项目页提供 GitHub / Code 链接；公开仓库，MIT）
- **数据集：** <https://huggingface.co/datasets/Chopper233/UVTA>（项目页提供 Dataset 链接）
- **入库日期：** 2026-10-16

## 项目页核查与开放状态

截至 2026-10-16，项目页明确列出论文、GitHub 代码仓和 Hugging Face 数据集；官方代码仓 README 确认训练、离线评估与机器人 rollout 入口，并标注 MIT 许可证。数据集 README 说明共 28 个 Zarr v2 tar 分片、约 482 GB。

| 资源 | 状态 | 核查结果 |
|------|------|----------|
| 训练 / 评估代码 | **已开源** | 官方仓库 uni-vta/UVTA，含 scripts/train.py、scripts/eval_action_quality.py |
| 数据集 | **已公开** | Hugging Face Chopper233/UVTA，28 个 tar 分片约 482 GB |
| 机器人部署 | **部分可复用** | 提供 cet/relative_policy_rollout.py 与 cet/robot_env.py；RobotEnv、机械臂运动学和厂商 SDK 需目标硬件侧自行适配 |
| 许可证 | **代码 MIT** | 以仓库 README / LICENSE 为准；数据集许可未在本次核查中独立确认 |

## 项目要点

- 被动 22-DoF 手部外骨骼同步采集腕部 RGB、腕姿、手指动作和五指指尖触觉。
- 人类和机器人数据共同监督动作生成与未来触觉预测；部署只执行机器人动作。
- 项目页展示五项真机任务，并报告 UVTA 平均成功率 70%，相较最强基线高 41 个百分点。
- 页面披露的边界：当前触觉只覆盖指尖；更大的人类数据量和模型规模尚未评测。

## Robotics_Notebooks 映射

- 论文来源：[sources/papers/uvta_arxiv_2609_34182.md](../papers/uvta_arxiv_2609_34182.md)
- 官方实现：[sources/repos/uni-vta-uvta.md](../repos/uni-vta-uvta.md)
- Wiki 实体：[wiki/entities/paper-uvta-human-tactile-dexterous-manipulation.md](../../wiki/entities/paper-uvta-human-tactile-dexterous-manipulation.md)

## 一手来源

- 项目页：<https://uni-vta.github.io/>
- 代码：<https://github.com/uni-vta/UVTA>
- 数据集：<https://huggingface.co/datasets/Chopper233/UVTA>
- 论文：<https://arxiv.org/abs/2609.34182>
