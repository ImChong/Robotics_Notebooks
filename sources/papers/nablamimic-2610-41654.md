# ∇Mimic: Ultra-Efficient Humanoid Control with Differentiable Simulation

- **论文：** <https://arxiv.org/html/2610.41654v1>（v1，2026-10-30）
- **作者和机构：** Weyl Lu、Yubei Chen，加州大学戴维斯分校
- **项目页：** <https://sailab.ai/nablamimic/>
- **代码：** <https://github.com/OamIcs/NablaMimic>；[NablaEngine](../repos/nablaengine.md)
- **入库日期：** 2026-11-03；核对日期：2026-11-04

## 原文核对

- 全文的任务、方法、实验及附录涉及 G1 29 关节、LAFAN1 40 段、50 Hz 控制、64 步反传窗口与 TD(lambda) 价值引导。
- 对反向传播的每样本动作梯度按批次中位数裁剪；用裁剪后的梯度形成动作目标并多步回归，抑制极端接触梯度的影响。
- v1 报告 RTX PRO 6000 上 0.34 h 达到归一化回报 0.82；SuperMimic 的 1.2 h 是另一论文在 H100 上报告的结果。训练时间排除编译与评估。
- 结果为平地 G1 仿真；训练默认梯度在特定病态几何与执行器饱和处做特殊处理，裁剪也有偏差。未部署真机。
- [知识节点与流程图](../../wiki/entities/paper-nablamimic.md)。项目主页和仓库的更新配置与 v1 结果需分开理解。
