# FlowVLA（Visual CoT）（arXiv:2508.18269）

> 来源归档（paper）

- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2508.18269>
- **PDF：** <https://arxiv.org/pdf/2508.18269>
- **作者：** Zhong, Yan, Li, Liu, Gong 等（arXiv 摘要页前 5 位）
- **提交：** 2025-08-25；（arXiv，v3 更名）
- **项目页：** <https://irpn-lab.github.io/FlowVLA/>
- **入库日期：** 2026-11-04
- **线索来源：** [wechat_xinjiqishijue_embodied_algorithm_stack_2026-11-04.md](../blogs/wechat_xinjiqishijue_embodied_algorithm_stack_2026-11-04.md)

## arXiv 摘要要点（人工转述）

摘要称：许多 VLA 内置的世界模型靠下一帧预测 v_t → v_{t+1}，缺少显式的运动推理，常导致物理上不合理的视觉预测与低效的策略学习；FlowVLA 是自回归 Transformer，显式物化 v_t → f_t → v_{t+1}，使「动力学预测」的预训练目标与「动作生成」的下游任务对齐；在操作基准与真机评测上，既生成更连贯、物理合理的预测，又取得 SOTA 策略性能与显著提升的样本效率。（WebSearch 补：早期版本标题为 FlowVLA: Thinking in Motion with a Visual Chain of Thought；作者来自 HKUST(GZ) 与上海交大，未在摘要页核对机构。）

## 对 wiki 的映射

- [paper-flowvla-visual-cot](../../wiki/entities/paper-flowvla-visual-cot.md)
