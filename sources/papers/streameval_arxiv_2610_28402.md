# StreamEval: Evaluating Interactive Streaming Video Generation Models

> 来源归档（paper）；依据 arXiv v3 摘要页与官方项目/代码仓核对。

- **作者：** StreamEval Team
- **版本/日期：** arXiv:2610.28402v3，cs.CV；v1 于 2026-10-20 提交，v3 于 2026-10-26 更新
- **论文：** <https://arxiv.org/abs/2610.28402>
- **PDF：** <https://arxiv.org/pdf/2610.28402>
- **项目页 / Leaderboard：** <https://streameval.ai>
- **代码：** <https://github.com/StreamEval/StreamEval>
- **数据：** <https://huggingface.co/datasets/StreamEval/StreamEval>；<https://www.modelscope.ai/datasets/StreamEval/StreamEval>
- **入库日期：** 2026-10-27

## 摘要式摘录

StreamEval 提出交互式流式视频生成统一基准，评测用户请求逐步到达时，系统是否能持续生成视频并响应修改，同时满足状态一致与内容保留要求。基准含 300 个脚本化测试用例，每段 15–300 秒，覆盖数字人对话、数字人表演、世界生成、指令编辑与参考图编辑五类任务。

评测按任务执行、状态一致性、视觉保真度、视听协调四个维度组织十项任务相关指标，分数范围 0–100。论文评测 37 个模型；视觉质量和身份保持较好，并不意味着可靠执行实时请求或长期维持状态。论文报告 avatar performance 的 Following 最高 36.90、文本条件 world generation 的 Following 最高 25.93、reference editing 的 Edit Success 最高 15.00；这些是各自任务中的最高分，不是跨任务总分。

## 关键事实

- 每个 case 定义初始素材、逐步到达的多模态请求，以及状态变化与内容保留要求。
- Overall 是各任务适用指标的等权均值；五类任务不会合并为单一总分。
- 评测重点是持续生成过程中的请求执行和状态保持，不能用单帧画质替代。

## 开放状态

官方代码仓提供评测协议、运行框架、模型适配入口和基准文件；README 提供 Python 3.10+ 的 CPU mock 快速路径。Hugging Face 与 ModelScope 发布相同的基准定义和媒体，README 标注约 3.1 GB。代码仓许可证为 Apache-2.0。

## 对 wiki 的映射

- [StreamEval 实体页](../../wiki/entities/paper-streameval.md)
- [具身评测基准选型闭环](../../wiki/queries/embodied-eval-benchmark-selection-loop.md)
