# Causal Analysis of Robot Foundation Models 论文来源归档

## 元数据

- **题目：** Causal Analysis of Robot Foundation Models Enables Efficiency Gains
- **作者：** Aiden Swann, Mathilde Papillon, Mozes Jacobs, Siddharth Boppana, Tal Haklay, Eric Bigelow, Andrew Lee, Conor Watts, Thomas Icard, Mac Schwager, Owen Lewis, Jack Merullo, Ekdeep Singh Lubana, Atticus Geiger, Thomas Fel, Matthew Kowal
- **预印本：** https://arxiv.org/abs/2610.20750
- **代码：** https://github.com/goodfire-ai/robotics_causal_analysis
- **版本：** v1，2026-10-14，cs.RO
- **入库日期：** 2026-10-17
- **一句话说明：** 对 VLA 的视觉语言骨干与动作专家之间的连接做因果干预，定位动作条件信息的关键层带，并据此裁剪计算。

## 论文要解决的问题

VLA 的视觉语言表征如何转化为动作仍不透明。论文通过 activation patching，将供体回合不同层的激活注入接收回合，测量目标物体选择和到达位置如何随之变化，从而分析图像、指令语义与空间信息影响动作的因果路径。

## 方法与主要发现

- 研究 π₀.₅、MolmoAct2、MolmoBot 三种模型，逐连接及以三层窗口为单位干预视觉语言模型（VLM）到动作专家（AE）的信息传递。
- 三种模型都呈现狭窄的中后层作用带，作者称为 **action-conditioning interface**。
- 目标名词较早层的表征主要决定“选哪个物体”；中后层文本 token 则携带影响“往哪里移动”的空间信息。不同架构中空间信息所在 token 位置不同。
- MolmoBot 仅保留 36 个连接中的 3 个用于 AE 条件化，论文报告推理 FLOPs 最多降低 33%，3,000 个评估 episode 的成功率为 46.7% 对 46.9%；同一思路也在 π₀.₅ 真机部署中验证。

## 开源状态与复现边界

截至 2026-10-17，官方 GitHub 仓库公开，MIT 许可（仓库内派生的 `franka_droid_kinematics.xml` 单独遵循 Apache-2.0）。仓库包含 `vla_patching/`、`data/`、`docs/` 与 README；指标是特定模型、连接裁剪方案与任务下的结果，不能直接外推到其他 VLA 架构或任务。

## 对 wiki 的映射

- [论文实体页](../../wiki/entities/paper-robot-foundation-model-causal-analysis.md)
- [官方实现归档](../repos/goodfire-robotics-causal-analysis.md)
