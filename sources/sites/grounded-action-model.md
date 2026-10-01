# Grounded Action Model 项目页

> 来源归档（site）

- **标题：** Grounded Action Model: 3D Grounding as a Foundation for Robotics
- **类型：** site / project
- **链接：** <https://grounded-action-model.github.io/>
- **论文：** [arXiv:2609.23863](https://arxiv.org/abs/2609.23863)（cs.RO，v2 2026-09-25）
- **机构：** 西北大学（Northwestern University）；华盛顿大学（University of Washington）；新加坡国立大学（National University of Singapore）
- **作者：** Gehao Zhang, Weikai Huang, Shailesh Shailesh, Yiyan Peng, Jiafei Duan, Ranjay Krishna
- **入库日期：** 2026-10-01
- **一句话说明：** 以可提示的 3D grounding 骨干（WildDet3D）把语言/点/框解析为对象中心视觉特征与度量几何，再经多流 MM-DiT + flow matching 预测动作块；支持自主运行或与 Molmo2 等高层规划器组合。
- **沉淀到 wiki：** [`wiki/entities/paper-grounded-action-model-3d-grounding.md`](../../wiki/entities/paper-grounded-action-model-3d-grounding.md)

## 开源状态（步骤 2.5，2026-10-01）

| 核查项 | 结论 |
|--------|------|
| **项目页 Code 按钮** | 链 [GehaoZhang6/Grounded-Action-Model](https://github.com/GehaoZhang6/Grounded-Action-Model) |
| **GitHub README** | 标题区 **「Code coming soon」**；训练/推理代码、预训练权重与真机 setup **准备发布**，需 Watch 仓库 |
| **Hugging Face / 权重** | 项目页与 README **未列** HF 或 checkpoint 直链 |
| **结论** | **待发布** — 官方仓已公开但 **无可运行** 训练/推理入口（截至入库日）；勿与「已开源可复现」等同 |

## 对本库的意义

- 把 **度量 3D grounding** 从 VLA/WAM 的隐式演示学习里 **显式化** 为机器人 foundation model 接口（语言 / 2D 点 / 2D 框 → 统一对象表示）。
- 与 [Spatial Forcing](../../wiki/entities/paper-rcl-ref-0f2536c81a1992e3c3b8-spatial-forcing-implicit-spatial-representation.md)（隐式空间对齐）、[π0.5](../../wiki/entities/paper-pi05-open-world-vla.md)（开放世界 VLA）在 **RoboTwin 2.0 / LIBERO-PRO** 上形成可对照读法。
