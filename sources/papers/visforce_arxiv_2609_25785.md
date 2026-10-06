# VisForce（arXiv:2609.25785）

> 来源归档（ingest）

- **标题：** VisForce: Visual Grounding of Current and Desired Forces for Goal-Conditioned Dexterous Manipulation
- **arXiv：** <https://arxiv.org/abs/2609.25785> · <https://arxiv.org/pdf/2609.25785>
- **作者：** Jung-Woo Lee、Soo-Chul Lim
- **入库日期：** 2026-10-05
- **摘要：** 当前指尖力和目标力映射到视觉图像，并以目标条件策略生成动作；实验用 UR10 与灵巧手。
- **详情：** [VisForce](../../wiki/entities/paper-visforce-force-grounding.md)

## 核心摘录

> 2026-10-05 据 arXiv HTML 全文（v1）核对。

- 骨干为 π0.5；逐指执行器力在 MuJoCo 中渲染为指尖视觉线索并叠加到对齐后的真实腕部图像，期望力渲染进按子任务检索的目标图，经带前景掩码的目标条件交叉注意力融合。
- 真机平台：UR10 + Inspire RH56F1 灵巧手 + 两台 RealSense D405；T1–T4 示教数 30 / 25 / 30 / 30。
- T1 中档期望力下鸡蛋 / 牙膏管抓起成功率 70% / 80%（Visual Force + Text Force Goal 40% / 20%）。
- T2 / T3 / T4 最终成功率 70% / 55% / 40%（每任务 20 次）；去掉交叉注意力的消融仅 10% / 5% / 15%。
