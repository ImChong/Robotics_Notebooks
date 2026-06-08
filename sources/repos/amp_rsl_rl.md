# AMP-RSL-RL

> 来源归档

- **标题：** AMP-RSL-RL
- **类型：** repo
- **链接：** https://github.com/gbionics/amp-rsl-rl
- **PyPI：** `pip install amp-rsl-rl`
- **维护者：** Istituto Italiano di Tecnologia (IIT)（Giulio Romualdi、Giuseppe L'Erario）
- **上游论文：** AMP (Peng et al., 2021)
- **License：** BSD-3-Clause
- **入库日期：** 2026-06-08
- **一句话说明：** 在 rsl_rl 的 PPO 上扩展 AMP（对抗式运动先验），用对抗模仿让人形从动捕数据学运动技能；含对称性数据增广，可 pip 安装。
- **沉淀到 wiki：** 是 → [`wiki/entities/amp-rsl-rl.md`](../../wiki/entities/amp-rsl-rl.md)

## 生态位置

- 框架：构建于 [rsl_rl](https://github.com/leggedrobotics/rsl_rl)（ETH RSL 的 GPU PPO 库）。
- 与 [AMP_for_hardware](amp_for_hardware.md)、[amp_mjlab](amp_mjlab.md) 同属「重定向/动捕产物 → AMP 风格策略」的消费侧；本仓偏人形、框架无关、可 pip 安装，接入现代 rsl_rl 训练栈成本低。
