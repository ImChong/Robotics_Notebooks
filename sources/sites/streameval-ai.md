# StreamEval 官方项目页与排行榜

- **类型：** site（项目主页、公开排行榜与评测说明）
- **URL：** <https://streameval.ai>
- **论文：** <https://arxiv.org/abs/2610.28402>
- **代码：** <https://github.com/StreamEval/StreamEval>
- **Hugging Face 数据：** <https://huggingface.co/datasets/StreamEval/StreamEval>
- **ModelScope 数据：** <https://www.modelscope.ai/datasets/StreamEval/StreamEval>
- **核查日期：** 2026-10-27
- **一句话说明：** 交互式流式视频生成的公开评测与排行榜，覆盖五类任务、十项指标和 15–300 秒持续会话。

## 核查结果

- 官方 GitHub README 将 streameval.ai 链接为 leaderboard。
- 测试用户请求陆续到达时，模型能否跟随请求并保留已有内容与状态。
- Hugging Face 与 ModelScope 提供相同的 benchmark 数据；官方 README 标注约 3.1 GB。
- **开放状态：** 评测代码、协议与数据已公开；代码仓标注 Apache-2.0。

## 边界

该项目评测视频生成系统的交互质量，不直接测试机器人动作可执行性、控制闭环或仿真到真机迁移；不能把视觉或请求跟随分数解读为机器人策略成功率。

## 对 wiki 的映射

- [StreamEval 实体页](../../wiki/entities/paper-streameval.md)
- [论文来源](../papers/streameval_arxiv_2610_28402.md)
- [官方代码仓](../repos/streameval.md)
