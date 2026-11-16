# Self-Refine: Iterative Refinement with Self-Feedback｜基于自反馈的迭代改进

> 来源归档（论文原文，NeurIPS 2023）

- **标题：** Self-Refine: Iterative Refinement with Self-Feedback
- **类型：** paper / llm-agents / iterative-refinement / prompting
- **作者：** Aman Madaan、Niket Tandon、Prakhar Gupta、Skyler Hallinan 等（共 16 位作者）
- **机构：** Carnegie Mellon University、Allen Institute for AI、University of Washington、NVIDIA、UC San Diego、Google Research / Brain Team
- **版本：** arXiv:2303.17651v2，2023-05-25；NeurIPS 2023
- **论文：** <https://arxiv.org/abs/2303.17651v2>
- **全文（可读 HTML）：** <https://arxiv.org/html/2303.17651v2>
- **会议归档：** <https://papers.nips.cc/paper/2023/hash/91edff07232fb1b55a505a9e9f6c0ff3-Abstract-Conference.html>
- **项目主页：** <https://selfrefine.info/>
- **官方代码：** <https://github.com/madaan/self-refine>
- **源码归档：** [madaan/self-refine](../repos/self-refine-madaan.md)
- **项目归档：** [Self-Refine 官方主页](../sites/self-refine.md)
- **收录日期：** 2026-11-11
- **核对范围：** 核对 arXiv v2 全文 §1–7（算法、七任务评测、消融、相关工作、局限）及附录 A–S（任务数据、模型与人工/GPT-4 评测、置信区间、各任务提示词示例）；对照官方项目页、GitHub README 和 src/acronym、src/responsegen 运行入口及三个 prompt 模块。
- **通读：** 已完成

## 核心摘录

1. **方法（§2，Algorithm 1）：** 同一 LLM M 用三类提示词执行 Init、Feedback、Refine；初稿 y0=M(p_gen||x)，反馈 fb_t=M(p_fb||x||y_t)，下一稿以 x、此前草稿和反馈历史为条件生成。停止可按最大迭代数或模型输出的质量信号；模型权重在迭代中不变。
2. **为什么能起作用（§2、§4）：** 反馈需要具体定位缺陷并提出可执行的修改，而非只说「再优化」。历史上下文帮助下一轮避免重复失误。多轮通常有边际收益递减；多指标任务可能非单调改进。
3. **评测（§3、表 1）：** 七任务覆盖 sentiment reversal、dialogue response、code optimization、code readability、math reasoning、acronym generation、constrained generation。基线为同一 GPT-3.5 / ChatGPT / GPT-4（代码任务还测 Codex）的单轮生成。评价混合任务特定指标、人工盲测偏好和 GPT-4 偏好，不能把所有数值视为同一成功率。
4. **GPT-4 量化例子（表 1）：** 代码优化 27.3→36.0（+8.7 个百分点）；受约束生成 15.0→45.0（+30.0）；对话回答偏好 25.4→74.6（+49.2）；数学推理 92.9→93.1（仅 +0.2）。论文摘要报告跨任务平均约 +20 个百分点，取决于所列任务和指标。
5. **消融（§4）：** Code Optimization 使用具体反馈 27.5、泛泛反馈 26.0、无反馈 24.8；说明批评信息的质量比纯粹多生成一轮重要。数学推理中模型自检常未发现错误，借助外部 oracle 反馈可进一步提高。
6. **局限（§6）：** 依赖模型的 few-shot 与 instruction-following 能力；实验主要用闭源且收费的模型，数据集仅英文；方法本身不包含安全防护，也不能保证每轮指标单调上升。
7. **实现边界：** 论文的抽象算法按反馈决定停止，但仓库的 src/acronym/run.py 示例采用 max_attempts 固定循环（默认 5），并返回所有候选与分数；不能将理论停止条件当作已在每个任务脚本实现。

## 与 Robotics_Notebooks 的关系

- **统一项目节点：** [Self-Refine：基于自反馈的迭代改进](../../wiki/entities/paper-self-refine-2303-17651.md)；论文、主页、源码只对应一个实体。
- **方法边界：** 这是 **推理期输出改写**，不是强化学习、权重自训练或能自主设计下一代模型的 RSI；对照 [递归自改进](../../wiki/concepts/recursive-self-improvement.md) 与 [RSI Survey](../../wiki/entities/paper-rsi-survey-2607-07663.md)。
- **机器人启发（非论文验证）：** 语言规划、代码草稿等可借鉴自反馈，但论文没有在真实机器人运动控制/闭环操作中验证成功率或安全性。

## 参考来源

- 原始论文全文：<https://arxiv.org/html/2303.17651v2>
- 会议论文记录：<https://papers.nips.cc/paper/2023/hash/91edff07232fb1b55a505a9e9f6c0ff3-Abstract-Conference.html>
- 官方项目与提示词示例：<https://selfrefine.info/>
- 官方实现：<https://github.com/madaan/self-refine>
