# Shallow-π: Knowledge Distillation for Flow-based VLAs

- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2601.20262>
- **项目页：** <https://icsl-jeon.github.io/shallow-pi/>
- **代码：** 截至入库日项目页未列独立 GitHub（待发布）
- **会议：** IROS 2026 Best Paper / Best Student Paper 候选（文内）
- **入库日期：** 2026-09-25
- **索引来源：** [AI科技评论 IROS 六趋势](../blogs/wechat_ai_tech_review_iros_2026_six_trends_2026-09-25.md)
- **一句话说明：** 对 flow-based VLA（π 类）做 VLM+动作头联合层蒸馏 18→6 层；推理 >2× 加速，成功率绝对降幅 <1%；Jetson Orin/Thor 真机验证。

## 核心摘录

1. 系统压缩 **transformer 深度**（相对 token 剪枝较少被研究的路径），中间层注入 conditioning。
2. 42dot、首尔大学、Samsung Research；IROS 2026 最佳论文候选。
3. **待发布** 权重/代码（步骤 2.5：2026-09-25 仅项目页 + arXiv）。

## 对 wiki 的映射

- 实体页：[`wiki/entities/paper-shallow-pi.md`](../../wiki/entities/paper-shallow-pi.md)
