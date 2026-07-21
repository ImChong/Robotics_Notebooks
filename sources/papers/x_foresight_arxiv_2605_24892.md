# X-Foresight: A Joint Vision-Action Causal Forecasting Network via Predictive World Modeling（arXiv:2605.24892）

> 来源归档（ingest）

- **标题：** X-Foresight: A Joint Vision-Action Causal Forecasting Network via Predictive World Modeling
- **缩写：** **X-Foresight**
- **类型：** paper / vla / predictive-world-model / autonomous-driving / causal-forecasting
- **arXiv：** <https://arxiv.org/abs/2605.24892>
- **项目页：** <https://x-foresight-1.github.io/en/>
- **代码：** 截至 2026-07-21 项目页仅链 arXiv，**未列 GitHub**
- **机构：** 小鹏（XPeng）PWM Team
- **作者（前若干）：** Baolu Li, Jingyu Qian, Rui Guo, Yilun Chen, Hanpeng Liu, Yuan Lin, Junhong Zhou, Ruixin Liu 等
- **入库日期：** 2026-07-21
- **一句话说明：** 把 **预测式世界模型** 直接嵌进驾驶 VLA：用 **长视界 chunk-wise 自回归**（块内稠密、块间稀疏）学因果，辅以课程视界、时序重要性采样与扩散多视角 Renderer；在约 **28 万小时** 车端数据上改善规划与碰撞率。

## 摘录 1：朴素 next-frame 的两个失败模式

- 视频 token 低熵冗余 → 预测塌缩为平凡外推。
- **时间困境：** 稠密帧捕瞬时动力学，却难覆盖长程因果；稀疏则丢运动线索。

**对 wiki 的映射：** [`wiki/entities/paper-x-foresight.md`](../../wiki/entities/paper-x-foresight.md)；挂 [VLA](../../wiki/methods/vla.md) 与 [WAM](../../wiki/concepts/world-action-models.md)「联合建模」对照轴。

## 摘录 2：架构与训练

- **Large Drive Model（LDM）：** 自回归 Transformer；每步预测 ego 动作、BEV scene plot、每摄未来外观 latent tokens。
- **Vision Renderer：** 基于 X-World 视角–时序注意力与 3D causal VAE 的 DiT rectified-flow 解码器；只条件于 LDM camera tokens（无动作捷径），渲染帧反馈闭合 AR 环。
- **Chunk-wise foresight：** 预测语义上更远的 chunk，而非相邻帧；课程逐步加长视界；TIS 把监督集中到安全关键 chunk。
- **三阶段训练：** 分训 LDM/Renderer → 冻结 LDM、用预测 camera tokens 对齐 Renderer。
- **数据：** ~280k h / 34M clips / 7 摄 / 13.8T tokens；训练 4 Hz（原始 12 Hz）。

## 摘录 3：结果与开源

- 生产规模（1024 GPU，H=21+CLEF+TIS）：横向 ADE **0.1675→0.1567**，碰撞率 **0.228%→0.191%**（相对 −16.2%），CCES Total **3.8296→3.6535**。
- **开源：** 截至入库日 **未开源**；内部数据不可复现。

**对 wiki 的映射：** 与 X-Mind 对照「稠密像素想象 vs 抽象 sketch Visual CoT」。
