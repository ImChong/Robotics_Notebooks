# MiniMax H3 官方博客介绍

> 来源归档

- **标题：** MiniMax H3: An Open Model Breaking the Boundaries Between Tasks and Modalities
- **类型：** blog / 官方发布
- **URL：** <https://www.minimax.io/blog/minimax-h3>
- **发布日期：** 2026-07-31（文内）
- **在线体验：** <https://hailuoai.video/zh-Intl/tools/minimax-h3>
- **入库日期：** 2026-09-29
- **一句话说明：** MiniMax 发布 **通用多模态生成系统 H3**：统一理解文本/图像/视频/音频上下文，生成 **原生立体声** 视频（最长 **15 s**、**2K**）；强调任务泛化（T2I/T2V/T2A/参考与编辑合一）、**Contextual Omni Representation**、**H3-VAE**、**H3-Omni Transformer**、**In-Context Regeneration**；计划开放权重并兼顾 AI 硬件兼容。

## 核心摘录

- **任务边界打破：** 不再拆 T2I、I2V、动作参考、风格参考、配音/音效/音乐等孤立专家；用自然语言描述上下文与目标关系（示例：参考 Video1 希区柯克运镜 + Image2 人物 + Audio3 人声）。
- **预训练范式：** 文本–图像、文本–视频（含联合音频、原生立体声、多镜头）、文本–音频；广义参考/编辑（图/视/音/视音联合）；真实数据 + 自然语言表达参考关系。
- **架构训练：** 尽早融合多模态与多任务；理解/生成分工训练架构，吞吐约 **+30%**；2K 通过 **In-Context Regeneration** 而非独立超分模块。
- **商业与定价叙事：** 2K 默认；宣称 2K 单价低于主流约 **1/3**，768p 低于主流 720p 约 **1/2**（文内营销口径）。
- **开源承诺：** 博客称将在合规前提下 **开放模型权重**；硬件兼容自设计早期纳入考量。

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [MiniMax H3 实体](../../wiki/entities/minimax-h3.md) | wiki 主节点 |
| [MiniMax-H3 仓库](../repos/minimax-h3.md) | 权重与推理实现 |
| [HarnessEval-W](../../wiki/entities/paper-harnesseval-w.md) | 论文榜 MiniMax H3 Overall **74.3**（Prompt I2V） |
