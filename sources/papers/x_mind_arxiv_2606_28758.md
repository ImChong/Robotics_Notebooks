# X-Mind: Efficient Visual Chain-of-Thought via Predictive World Model for End-to-End Driving（arXiv:2606.28758）

> 来源归档（ingest）

- **标题：** X-Mind: Efficient Visual Chain-of-Thought via Predictive World Model for End-to-End Driving
- **缩写：** **X-Mind**
- **类型：** paper / vla / visual-cot / predictive-world-model / autonomous-driving / efficiency
- **arXiv：** <https://arxiv.org/abs/2606.28758>
- **项目页：** <https://xp-x-mind.github.io/en/>（PDF：项目页 `X_Mind.pdf`）
- **代码：** 截至 2026-07-21 项目页仅链 ArXiv / Tech Report PDF，**未列 GitHub**
- **机构：** 小鹏（XPeng）PWM Team
- **作者（前若干）：** Bohao Zhao, Chengrui Wei, Guangfeng Jiang, Ruixin Liu, Xuejie Lv, Liu Liang, Sutao Deng, Xiuyang Fan 等
- **入库日期：** 2026-07-21
- **一句话说明：** 把 PWM **内化** 为驾驶 VLA 的 **Visual CoT**：先 rollout 未来再规划；用 **抽象 sketch（BEV+驾驶先验）+ DC-AE（12 帧→96 token）** 与 **Recurrent Block Diffusion（单次前向完成去噪）** 解决车载时延。

## 摘录 1：为何需要 Visual CoT

- 现有驾驶 VLA 多为感知→动作直映，缺显式预测。
- 外挂级联 PWM 时延不可接受；浅层终端任务又无法把前瞻推理注入 LLM 深骨干。

**对 wiki 的映射：** [`wiki/entities/paper-x-mind.md`](../../wiki/entities/paper-x-mind.md)；与 [X-Foresight](../../wiki/entities/paper-x-foresight.md) 对照效率/表示粒度。

## 摘录 2：机制

- **Abstract sketch：** BEV 拓扑 + 动态体 + 红绿灯 + 导航意图 + 速度合规剖面；非稠密未来图像。
- **DC-AE：** 12 帧未来 rollout → **96 tokens**（对比 dense image 3584 / 3DGS 3072）。
- **RBD：** 去噪步沿 LLM 层级展开；推理用 Euler 积分在一次前向内完成。
- **联合损失：** L_total = λ_WM L_WM + λ_plan L_plan；层间 latent flow matching + 稀疏图像重建。
- **数据协议：** 与 X-World / X-Foresight 同 7 摄；实验用总数据 1/8 子集。

## 摘录 3：结果与开源

- Sketch 表示 ADE 最优且几乎不增推理（1.1× vs Base）；RBD FID **9.59** vs 单步 **67.30**。
- 预测未来 12 帧比重建当前帧更利轨迹（尽管 FID 略差）——优势来自 **预测** 而非视觉保真。
- **开源：** 截至入库日 **未开源**。

**对 wiki 的映射：** 工程选型：「车载要 Visual CoT 时，优先压缩思考表示，而非级联全像素 WM」。
