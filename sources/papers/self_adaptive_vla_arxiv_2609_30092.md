# Self-Adaptive VLA（arXiv:2609.30092）

> 来源归档（ingest）

- **标题：** Self-Adaptive VLA for Robust Robot Deployment
- **缩写：** **Self-Adaptive VLA**
- **类型：** paper / vla / deployment / sim2real / hardware-drift
- **arXiv：** <https://arxiv.org/abs/2609.30092>
- **PDF：** <https://arxiv.org/pdf/2609.30092>
- **项目页：** <https://icefoxzhx.github.io/self-adaptive-vla/> — 归档见 [`sources/sites/self-adaptive-vla-github-io.md`](../sites/self-adaptive-vla-github-io.md)
- **作者：** Hongxin Zhang*、Chunru Lin*、Tsun-Hsuan Wang、Zhenjia Xu、Chuang Gan（* 共同一作）
- **机构：** 马萨诸塞大学阿默斯特分校（UMass Amherst）；创世纪 AI（Genesis AI）
- **入库日期：** 2026-09-26
- **开源状态（步骤 2.5，2026-09-26）：** 项目页 **未列 GitHub/HF**；arXiv 摘要链项目页视频 → **待发布 / 未开源**。

## 核心论文摘录（MVP）

### 1) 问题：无记忆 VLA 对硬件漂移脆弱

- **链接：** <https://arxiv.org/abs/2609.30092>
- **核心贡献：** 磨损、标定误差等 **hardware shift** 在部署期出现，而标准 VLA **无法在站点持续自适应** 且不宜频繁人工重标定。Self-Adaptive VLA 是 **post-training recipe**：用 **策略自身 rollout** 作 context，迭代适应未知 shift。
- **对 wiki 的映射：**
  - [Self-Adaptive VLA 论文实体](../../wiki/entities/paper-self-adaptive-vla.md)
  - [VLA 方法页](../../wiki/methods/vla.md)

### 2) Shift-conditioned 专家数据 + Context encoder

- **核心贡献：** 在 **已知注入 shift** 下 rollout **冻结 base policy** 得 context；将原训练示范的 **专家动作预补偿（pre-compensate）** 同一 shift，保证监督仍来自专家而非次优 rollout；**轻量 plug-in context encoder** 把多视角视频、本体与 rollout 动作压成 **单个 context token**，经 **AdaLN** 调制冻结 VLM 上的 DiT 策略；**flow-matching loss** 与 base 相同，VLM 冻结。

### 3) 测试时 token ensemble

- **核心贡献：** 多次失败 trial 的 context token **直接求和 ensemble**；新失败揭示先前 masked 的 shift，逐步收敛补偿；每 trial **算一次 token**，闭环 **零额外延迟**。
- **评测（Abstract / 项目页）：** 四项 **精密双臂/灵巧手** 任务；hardware shift 下恢复 base **>80%** 性能；新工位 **Station 2** 上 base 0/5 → 1 次失败 context 2/5 → 2 次失败 context **5/5**（Assemble Ring 示例）。
