# ∇Mimic: Ultra-Efficient Humanoid Control with Differentiable Simulation

> 来源归档（草稿；尚未完成全文与源码核查）

- **论文：** https://arxiv.org/abs/2610.41654 （v1，2026-10-30）
- **作者：** Weyl Lu、Yubei Chen
- **项目页：** https://sailab.ai/nablamimic/；论文还列出 https://oamics.github.io/NablaMimic/
- **代码：** https://github.com/OamIcs/NablaMimic；[NablaEngine](../repos/nablaengine.md)
- **入库日期：** 2026-11-03

## 已核实线索

- 论文摘要提出使用逐样本梯度裁剪与多步回归处理接触梯度不稳定，并使用 G1 专用可微仿真器 NablaEngine。
- 论文报告在 LAFAN1 的 40 个动作上训练一个策略；不同方法的训练时间涉及不同 GPU，比较时应保留实验条件。

## 待完成

- PDF 已可访问，但尚未通读方法、实验和附录；项目页当前无法访问。补核后再建立或更新 wiki 实体，并绘制项目流程图及基于实际入口的源码运行时序图。
