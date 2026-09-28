# ScaleDP：十亿参数可扩展扩散 Transformer 策略（arXiv:2409.14411）

> 论文来源归档（ingest）

- **标题：** Scaling Diffusion Policy in Transformer to 1 Billion Parameters for Robotic Manipulation
- **作者：** Minjie Zhu, Yichen Zhu, Zhiyuan Xu, Jinming Li, Junjie Wen, Ning Liu, Ran Cheng, Chaomin Shen, Yaxin Peng, Feifei Feng, Jian Tang（* 共一）
- **类型：** paper / imitation-learning / diffusion-policy / transformer / manipulation
- **arXiv：** <https://arxiv.org/abs/2409.14411> · PDF：<https://arxiv.org/pdf/2409.14411.pdf>
- **会议：** IEEE ICRA 2025（Accepted）
- **项目页：** <https://scaling-diffusion-policy.github.io/> — [`sources/sites/scaling-diffusion-policy-github-io.md`](../sites/scaling-diffusion-policy-github-io.md)
- **入库日期：** 2026-09-28
- **一句话说明：** **ScaleDP** 用 **AdaLN 条件融合 + 非因果（unmasking）动作自注意力** 稳定 **DP-T** 训练，把扩散 Transformer 策略从 **10M 扩到 1B** 参数并获 MetaWorld / 真机增益。

## 核心摘录（面向 wiki 编译）

### 1)  vanilla DP-T 难以随深度/头数缩放

- **要点：** MetaWorld 上增加 Transformer 层数或 head 数，成功率反降；观测 cross-attention 融合带来 **层间梯度方差过大**，深网络训练不稳定。
- **对 wiki 的映射：** [`wiki/methods/diffusion-policy.md`](../../wiki/methods/diffusion-policy.md)、[`wiki/entities/paper-scaledp-scaling-diffusion-transformer-policy.md`](../../wiki/entities/paper-scaledp-scaling-diffusion-transformer-policy.md)

### 2) AdaLN 观测融合 + 非因果动作注意力

- **要点：** 用 **AdaLN** 从时间步与观测 embedding 回归 scale/shift，替代 cross-attention 条件注入；去掉 action token 的 **因果 mask**，让去噪网络在训练时看见 **chunk 内未来动作**，减轻只执行首步时的复合误差。
- **对 wiki 的映射：** [`wiki/concepts/diffusion-transformer.md`](../../wiki/concepts/diffusion-transformer.md)

### 3) 规模–性能曲线与真机七任务

- **要点：** ScaleDP-Ti/S/B/L/H（约 10M–1B）；MetaWorld 50 任务最大 ScaleDP 相对 **DP-T** 平均 **+21.6%**；7 项真机（Franka 单臂 + 双臂 UR5）随规模成功率上升（项目页表：ScaleDP-H 平均 **92.14%** vs DP-T **39.28%**）。
- **对 wiki 的映射：** [`wiki/tasks/manipulation.md`](../../wiki/tasks/manipulation.md)

## 开源状态（步骤 2.5，2026-09-28）

| 核查项 | 结论 |
|--------|------|
| 项目页 Code 按钮 | 指向 **第三方** [juruobenruo/DexVLA](https://github.com/juruobenruo/DexVLA)，**非** ScaleDP 官方实现 |
| arXiv / 项目页其他区 | **无** ScaleDP 训练/推理仓库、权重或数据链接 |
| 判定 | **截至入库日未开源**；复现需自研或等待作者发布 |

## 当前提炼状态

- [x] 要点摘录与 wiki 映射
- [x] 升格实体：[`wiki/entities/paper-scaledp-scaling-diffusion-transformer-policy.md`](../../wiki/entities/paper-scaledp-scaling-diffusion-transformer-policy.md)
