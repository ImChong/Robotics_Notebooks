# PROOF（arXiv:2611.18305）原文归档

- **标题：** PROOF: Perceptive Robustness via Out-of-Distribution Fallback for Legged Locomotion
- **类型：** paper
- **作者：** Siyuan Wang, Dongjie Zhu, Jinxuan Lu, Wenjun Xu, Jiankun Wang
- **机构：** 南方科技大学、鹏城实验室、清华大学、香港科技大学（广州）
- **发表：** CoRL 2026；arXiv v1，2026-11-13
- **论文：** https://arxiv.org/abs/2611.18305
- **HTML：** https://arxiv.org/html/2611.18305
- **PDF：** https://arxiv.org/pdf/2611.18305
- **项目页：** https://syw-robotics.github.io/proof-loco/
- **代码：** https://github.com/syw-robotics/proof-loco
- **入库日期：** 2026-11-17
- **核对范围：** arXiv v1 HTML 正文 §1–§6、附录 A–C（含训练算法、网络参数、评测设置），官方项目页与代码仓 README；未运行训练与真机实验。
- **通读：** 已完成

## 核心摘录

1. **问题：** 深度图在遮挡、反射、强光等分布外情形下会误导端到端感知运动策略，单纯增加感知噪声鲁棒性不等于能在视觉失效时安全降级。
2. **教师：** Isaac Lab 中以高程图和本体历史为输入，通过 PPO 训练；非跑酷地形随机屏蔽高程图 latent，让同一 actor 学到有视觉与无视觉两种动作行为。
3. **学生：** 在线 DAgger 蒸馏；CNN 编码深度帧，GRU 估计教师外感知 latent 均值与对角 log-variance，NLL 学习输入相关不确定性；动作模仿和 gate BCE 联合监督。
4. **回退：** 训练时视觉损坏 burst 对齐教师的外感知 mask；部署时 gate 使用硬二值滞回：关闭状态 ≥0.8 才打开，打开状态 <0.5 才关闭。
5. **仿真：** 强视觉扰动下非跑酷地形测试，PROOF 在 rough flat/uphill/downhill/upstairs/downstairs 的成功率分别为 100/99/100/95/100%；无教师 mask 基线均为 0%（项目页结果）。
6. **真机：** Unitree Go2 + RealSense D435；演示 70 cm 沟、50 cm 平台以及遮挡/弱光回退。论文和项目页同时披露稀疏草地 gate 误判与长时间极限运动电机过热问题。
7. **复现：** 官方代码已发布训练入口，README 标明这是基于 Isaac Lab 3.0、LocoLab 和 Z_RL 的后续重实现；部署代码仍待发布，不应将其与论文实验实现完全等同。

## 对 wiki 的映射

- [PROOF 论文实体](../../wiki/entities/paper-proof-loco.md)
