# Murea-InWorld

> 来源归档（ingest · 官方代码仓）

- **标题：** Murea-InWorld
- **类型：** repo
- **机构：** Murea Labs
- **链接：** <https://github.com/Murea-Labs/InWorld>
- **项目页：** <https://murea.ai/>
- **技术报告：** <https://murea.ai/page/paper/InWorld-technical-report.pdf>
- **Hugging Face：** <https://huggingface.co/MureaLabs/InWorld>
- **许可：** Apache-2.0（代码）
- **入库日期：** 2026-10-18
- **一句话说明：** 交互式视听世界模型开源仓，提供 Murea-Mind 故事 Agent 与 Murea-Fast 推理服务；Murea-Fast 权重已发布。

## 开源范围核查

官方 README 将系统分为 Murea-Mind（故事 Agent）和 Murea-Fast（视频生成推理服务）。仓库公开这两部分的代码；README 指向 Hugging Face 上的 `murea-fast.safetensors`，部署指南还要求准备 LTX-2.5 基础组件。技术报告评测的 Murea-Pro 权重不在 README 声明的本次开源组件内，不能据此认为 Pro 也已开放。

仓库 README 指向项目主页和技术报告。当前抓取渠道无法读取 `murea.ai` 正文，因此这里的代码与权重状态以可读取的官方 GitHub README、仓库 LICENSE 和 README 中的 Hugging Face 链接为依据。

## 运行与资源线索

- 推理服务文档：`inference/README.md`、`inference/docs/deployment.md)、`inference/docs/api.md`
- Agent 文档：`agent/README.md`
- README 报告：8 步采样；5 秒视频在 4 张 H20 上约需 17 秒（启用 sequence parallelism）。
- README 摘要的评测：Murea-Pro 在 WBench Full-289 得分 81.13；Murea-Fast 在 OpenS2V-Eval 得分 64.98、OS2V-Pro 得分 69.9。以上为仓库 README 转述的报告结果，不代表独立复核。

## 对 wiki 的映射

- [Murea-InWorld 实体页](../../wiki/entities/paper-murea-inworld.md)
- [生成式世界模型](../../wiki/methods/generative-world-models.md)
