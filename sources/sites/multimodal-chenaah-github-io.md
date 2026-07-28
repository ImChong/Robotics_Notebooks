# chenaah.github.io/multimodal（Multi-Modal Legged Locomotion 项目页）

- **标题：** Multi-Modal Legged Locomotion Framework with Automated Residual Reinforcement Learning
- **类型：** site / project-page
- **URL：** <https://chenaah.github.io/multimodal/>
- **配套论文：** [arXiv:2202.12033](https://arxiv.org/abs/2202.12033)，IEEE RA-L / IROS 2022（vol. 7 no. 4, pp. 10312–10319）
- **代码：** GitHub 三仓（见下）— 归档见 [`sources/repos/cheetah-trainer.md`](../repos/cheetah-trainer.md)
- **入库日期：** 2026-07-28

## 一句话摘要

Chen Yu、Andre Rosendo（上海科技大学）的多模态腿足项目页：Mini Cheetah 加装 3D 打印支撑结构后，手工过渡动作序列 + Automated Residual RL（ARRL）学习双足行走；ARRL 用黑箱优化器（ES/BO）与 RL（TD3/SAC）**同时**训练基础 PD 控制器与残差策略；页面含过渡方案、六组 ARRL 组合对比与代码入口。

## 公开信息要点（截至入库日）

- **机构：** ShanghaiTech University（School of Information Science and Technology）。
- **代码开放度：** **已开源**，页面 Code 区列出三部分：
  - <https://github.com/Chenaah/Cheetah-Gym> — PyBullet 仿真环境
  - <https://github.com/Chenaah/Cheetah-Software-RL> — 训练/测试代码（TensorFlow / Python）
  - <https://github.com/Chenaah/Cheetah-Trainer> — 真机程序（TensorFlow / C++）
  - 附支撑结构 STL 文件。
- **关键结论（页内摘要）：** 除「TD3 + Line 步态」外，六组 ARRL 组合均优于对应纯 RL；正弦/玫瑰/三角步态上 TD3 系组合普遍优于 SAC 系与纯黑箱优化。
- **Sim2Real：** 页面给出 sim-to-real 迁移技术说明与真机双足行走/模式切换演示。

## 为何值得保留

ARRL 是 Residual RL 的**自动化变体**：连基础控制器的参数也不再手调，而是由黑箱优化器与 RL 残差**联合**训练——回答「手工动作/控制器不够稳定时，哪些阶段需要多强的残差修正」；三仓开源使其成为四足改双足研究的可复现基线。

## 对 wiki 的映射

- 实体页：[paper-multimodal-legged-arrl](../../wiki/entities/paper-multimodal-legged-arrl.md)
- 方法页：[residual-policy-learning](../../wiki/methods/residual-policy-learning.md)
