# StreamEval 官方代码仓

- **仓库：** <https://github.com/StreamEval/StreamEval>
- **项目页 / Leaderboard：** <https://streameval.ai>
- **论文：** <https://arxiv.org/abs/2610.28402>
- **数据：** <https://huggingface.co/datasets/StreamEval/StreamEval> · <https://www.modelscope.ai/datasets/StreamEval/StreamEval>
- **许可证：** Apache-2.0
- **核查日期：** 2026-10-27
- **一句话说明：** 交互式流式视频生成评测基准的代码、协议、模型适配接口与测试工具。

## 仓库与运行入口

仓库包含基准定义、评测协议、运行包、模型适配器、脚本和示例。README 提供 Python 3.10+ CPU mock 路径，可不依赖 GPU、API key 或完整媒体数据，运行一个 case 并用 validate-run 校验产物。完整数据可从 Hugging Face 或 ModelScope 下载，约 3.1 GB。

1. 创建虚拟环境并安装项目 runtime、judge 和 dev extras。
2. 生成 synthetic smoke benchmark，运行 mock model。
3. 使用 validate-run 校验运行产物；下载完整数据后可校验 benchmark。
4. 注册或连接模型服务，按 evaluation protocol 评测并保存结果。

## 开源边界

评测代码与协议已公开，完整基准定义和媒体由 Hugging Face / ModelScope 分发。特定模型服务可能有独立依赖或凭据要求；榜单数值应结合任务、模型版本和协议解读。

## 对 wiki 的映射

- [StreamEval 实体页](../../wiki/entities/paper-streameval.md)
- [官方项目页与排行榜](../sites/streameval-ai.md)
- [论文来源](../papers/streameval_arxiv_2610_28402.md)
