# SO-Arena 官方项目主页（System One 决策评测）

- **官方主页：** <https://lichenlillc.github.io/SO-Arena/>
- **论文：** <https://arxiv.org/abs/2611.15777v1>
- **代码：** <https://github.com/holdrain/SO-Arena>
- **数据集：** <https://huggingface.co/datasets/SO-Arena/SO-Arena>
- **代码归档：** [holdrain/SO-Arena](../repos/so-arena.md)
- **论文归档：** [arXiv:2611.15777](../papers/so_arena_arxiv_2611_15777.md)
- **wiki 映射：** [SO-Arena](../../wiki/entities/paper-so-arena.md)
- **核对日期：** 2026-11-19
- **核对范围：** 官方项目页 Overview、Leaderboard、Reliability、Domains、Benchmark、Evaluate、Cite，及论文、代码、数据三条对外链接。
- **通读：** 已完成

## 开源状态与版本界限

- **代码已开源：** GitHub 提供顶层 CLI、evaluator、models、reliability、latency、tests；MIT 只适用于代码。
- **数据已发布：** 正式 HF 数据集为 SO-Arena/SO-Arena（Hugging Face 的 SO-Arena 根 URL 是组织页），数据混合许可证且部分来源要求非商业使用。
- **模型不随 benchmark 打包：** 用户需部署自有模型服务，通过 HTTP 或 Python adapter 评测。
- **项目页补充项：** 页脚说明 Gemini Decision 1 是 2026-11-13 **论文之后补评**，不可并入 arXiv v1 的 21 模型主实验。
- **复现门槛：** Linux x86-64、Python 3.12，锁定数据下载、校验、独立模型推理。
