# sites.google.com/view/learning-to-jump（Continuous Versatile Jumping 项目页）

- **标题：** Continuous Versatile Jumping Using Learned Action Residuals
- **类型：** site / project-page
- **URL：** <https://sites.google.com/view/learning-to-jump>
- **配套论文：** [PMLR 211 (L4DC 2022), yang23b](https://proceedings.mlr.press/v211/yang23b.html)；[arXiv:2304.08663](https://arxiv.org/abs/2304.08663)
- **代码：** 页面与 PMLR 条目均**未列代码仓库**（截至 2026-07-28 核查，按「未开源」处理）
- **入库日期：** 2026-07-28

## 一句话摘要

Yang, Meng, Yu, Zhang, Tan, Boots（University of Washington / Google）的四足连续跳跃项目页：高层 stance 控制器 = 手工加速度控制器 + 学习残差策略（ARS 训练），低层 WBC 转电机指令；Go1 真机实现全向跳跃最高约 50 cm、最远约 60 cm、单次转向跳约 90°，页面提供真机视频。

## 公开信息要点（截至入库日）

- **机构：** University of Washington（Yang、Meng、Boots）；Google（Yu、Zhang、Tan）。
- **页面内容：** 真机演示视频为主（全向跳、转向跳、连续跳）；无 Code 区。
- **硬件：** Unitree Go1（15 kg，12 DoF）；仿真 PyBullet；控制管线 500 Hz。
- **注意：** 用户清单中标注「CoRL 2022 / PMLR 2023」；正式出处为 **L4DC 2022**（Proceedings of Machine Learning Research vol. 211, 2023 年出版）。

## 为何值得保留

该工作是「传统控制器打底、RL 学残差」在**真实腿足机器人**上的教科书案例：加速度控制器保证可达起跳速度（warm start），残差只修稳定性；消融显示端到端 RL 需要约 10× 训练样本且回报更低。项目页是核查其开源状态（未开源）的一手依据。

## 对 wiki 的映射

- 实体页：[paper-versatile-jumping-action-residuals](../../wiki/entities/paper-versatile-jumping-action-residuals.md)
- 方法页：[residual-policy-learning](../../wiki/methods/residual-policy-learning.md)
