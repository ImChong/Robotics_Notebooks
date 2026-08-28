# LAC Project Page

> 来源归档

- **标题：** LAC — Linear and Angular Compliance
- **类型：** site / project page
- **URL：** <https://lac-humanoid.github.io/>
- **论文：** <https://arxiv.org/abs/2608.25405>
- **代码：** <https://github.com/lac-humanoid/lac-code>
- **Pages 仓：** <https://github.com/lac-humanoid/lac-humanoid.github.io>（含 in-browser demo）
- **机构：** 日本东北大学 Neuro-Robotics Laboratory
- **入库日期：** 2026-08-28
- **一句话说明：** 官方项目页：上身线/角柔顺叙事、浏览器交互 demo；训练代码不在此仓。

## 开源核查（2026-08-28）

| 入口 | 状态 |
|------|------|
| 项目页 | 可打开；描述含 in-browser demo |
| Code | [lac-humanoid/lac-code](https://github.com/lac-humanoid/lac-code) — sim2sim + ckpt + ROS 2，MIT |
| 训练 | **未发布**（论文用 Isaac Lab PPO，仓库无 train 入口） |

结论：**部分开源**。可复现推理 / MuJoCo 扰动实验，不可复现 378k 增强数据与四卡训练。

## 对 wiki 的映射

- 论文摘录：[lac_arxiv_2608_25405.md](../papers/lac_arxiv_2608_25405.md)
- 仓归档：[lac-code.md](../repos/lac-code.md)
- 实体：[paper-lac.md](../../wiki/entities/paper-lac.md)
