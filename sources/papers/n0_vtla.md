# 𝒩₀-VTLA：Scaling Vision-Tactile-Language-Action Model with Latent Tactile Tokens

> 来源归档（ingest）

- **标题：** N0-VTLA: Scaling Vision-Tactile-Language-Action Model with Latent Tactile Tokens
- **类型：** paper / technical-report / vtla / tactile / offline-rl / contact-rich
- **项目页：** <https://research.neoteai.com/n0-vtla/>
- **PDF：** <https://research.neoteai.com/assets/n0-vtla-report.pdf>
- **代码 / Checkpoints：** <https://github.com/neoteai/N0-VTLA>（截至入库日占位）
- **机构：** NeoteAI × Fudan TEAI
- **日期：** 2026-07-25
- **入库日期：** 2026-07-26
- **一句话说明：** 在 NeoData 上预训练的 VTLA：预测 **未来动作块上的潜空间触觉 token** 条件 flow-matching 动作专家，并用 **ALTER** 从部署语料做 advantage 条件离线 RL。

## 摘要级要点

- **问题：** 触觉帧远离接触时近空、接触切换时极短窗高信息；当额外相机浪费视觉前缀；仅注入当前触觉帧是 **反应式**（描述已发生接触）。
- **方法：** 预测下一动作块期望的净触觉变化 latent \(z\)，prepend 到 noisy action suffix；当前接触 token **不直接** 进 VL 前缀或动作专家。
- **三阶段训练：** (1) predictor 对比匹配 + 空间重建；(2) 掩码 VL 注意力迫使动作专家消费 \(z\)；(3) 恢复 VL 注意力端到端，冻结触觉编码器骨干。
- **ALTER：** Advantage Labeling from Trajectory Events and Relative Progress；清洁示教 + 自主 rollout + HIL 纠正 + 分段恢复；阶段内 top 30% → Advantage:positive。
- **主结果：** NeoReal 九任务均 **47.2%**（π₀.₅ **29.4%**），progress 42.3→56.8；UniVTAC+NeoSim 均 **63.8%** vs 最强基线 **44.0%**；ALTER 下毛巾 **95%** / 装袋 **80%** / 纸箱折叠 **75%**。
- **开源状态：** **部分 / 待发布** — 仓存在但代码与权重 roadmap **By July 31, 2026**。

## 核心论文摘录（MVP）

### 1) Predictive Touch：差分接触 token → 未来 latent

- **摘录要点：** 相对 episode 起始零接触基线做差分；冻结触觉视觉编码器 + 可训投影；对称对比匹配正确未来；辅助 L1 锚定空间布局。
- **对 wiki 的映射：** [paper-n0-vtla.md](../../wiki/entities/paper-n0-vtla.md)、[visuo-tactile-fusion.md](../../wiki/concepts/visuo-tactile-fusion.md)

### 2) 统一动作容器

- **摘录要点：** 固定宽度 state/action；相对首姿态的 action chunk；单臂/双臂/手持共用目标。
- **对 wiki 的映射：** [VLA](../../wiki/methods/vla.md)、[𝒩₀-Foundation](../../wiki/entities/paper-n0-foundation.md)

### 3) ALTER 离线改进

- **摘录要点：** 触觉接触变化助分段；掉落与 HIL 纠正作稀疏偏好；部署始终用 positive condition。
- **对 wiki 的映射：** [TACO](../../wiki/entities/paper-taco-tactile-wm-vla-posttrain.md)、[safe-real-world-rl](../../wiki/concepts/safe-real-world-rl-fine-tuning.md)

### 4) NeoReal / UniVTAC / NeoSim 结果

- **摘录要点：** Socket Plugging 85% vs 60%；Board Insertion 25%（基线 0）；行为上插入碰沿后抬升重试、柔顺物调节开度等。
- **对 wiki 的映射：** [contact-rich-manipulation.md](../../wiki/concepts/contact-rich-manipulation.md)

## 对 wiki 的映射（汇总）

- 实体：[paper-n0-vtla.md](../../wiki/entities/paper-n0-vtla.md)
- 交叉：[methods/vla.md](../../wiki/methods/vla.md) · [paper-n0-foundation.md](../../wiki/entities/paper-n0-foundation.md) · [paper-n0-twam.md](../../wiki/entities/paper-n0-twam.md)
- 站点 / 仓：[research-neoteai-com.md](../sites/research-neoteai-com.md) · [n0-vtla.md](../repos/n0-vtla.md)

## BibTeX

```bibtex
@article{n0vtla2026,
  title   = {N0-VTLA: Scaling Vision-Tactile-Language-Action Model with Latent Tactile Tokens},
  author  = {NeoteAI Team and Fudan TEAI Team},
  journal = {Technical Report},
  year    = {2026}
}
```
