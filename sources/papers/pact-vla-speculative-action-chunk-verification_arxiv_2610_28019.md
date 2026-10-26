# PACT-VLA（arXiv:2610.28019）

> 来源归档（paper）

- **标题：** PACT-VLA: Speculative Action-Chunk Verification with Pre-Commit Observations
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.28019>
- **PDF：** <https://arxiv.org/pdf/2610.28019>
- **作者：** Yixiang Zhu, Ruilong Wu, Yonghao Chen, Jiaxiang Zou, Yusong Hu, Xinyu Chen
- **arXiv 评论栏：** （无）

- **入库日期：** 2026-10-26
- **一句话说明：** PACT-VLA：利用执行前观测进行推测式动作块验证。

## 开源状态

- arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布）（2026-10-26，仅核对 arXiv 摘要页）

## 核心摘录

1. **策展来源：** [【10.19-10.23前沿论文动态】Manipulation 37篇](../blogs/wechat_senlanke_weekly_manipulation_2026-10-19_23.md)
2. **周报所述问题：** VLA 生成动作块后，连续执行期间的环境变化可能使后续动作失效。固定长度的开环执行难以及时处理状态偏移。
3. **周报所述创新点：** 将执行前观测与动作块验证结合，在提交动作之前检查候选动作与当前环境状态的一致性，研究降低动作块执行风险的方法。
4. **arXiv 摘要（英文节选）：** Vision-language-action (VLA) policies are commonly deployed with asynchronous inference in real-world settings, where the next action chunk is generated while the robot is still executing the previous one. By the time the new chunk is committed, the robot and scene have changed, so the chunk is conditioned on a stale observation and task success drops sharply as this inference delay grows. Generating multiple candidate chunks could let newer observations inform the final choice, but existing multi-candidate methods either rank candidates with the stale inference-start observation or repeatedly switch between candidates during execution, which can make the executed action sequence discontinuous.

**对 wiki 的映射**

- [paper-pact-vla-speculative-action-chunk-verification](../../wiki/entities/paper-pact-vla-speculative-action-chunk-verification.md)
