# DeliverBench｜官方源码

> 来源归档（公开代码仓，静态核对）

- **类型：** repo
- **链接：** <https://github.com/theseus-labs-rsi/deliverbench>
- **项目页：** <https://deliverbench.github.io/>
- **论文：** <https://arxiv.org/abs/2611.03412>
- **入库日期：** 2026-11-09
- **核对范围：** README、入口脚本和文中列出的主要运行模块；没有在本次环境运行完整依赖或评测。
- **通读：** 未完成

## 已核查入口

- **运行入口：** `harbor run -c harbor_job.yaml -a deliverbench_harbor.agents:Codex -m MODEL --resume-trajectory --plugin deliverbench`。
- **核心模块（静态核对）：** `deliverbench-harbor/src/deliverbench_harbor/lifecycle.py` 的 `Plugin.on_job_start` 检查原生 Harbor 版本、候选会话恢复、Verifier 及 Judge 配置；`agents.py` 包含受约束的原生 agent 封装；`verifier.py` 对提交结果验收。
- **数据入口：** `scripts/download_data.py` 根据 `data_release.json` 锁定 HF 提交、检验 SHA256 和文件集合，拒绝覆盖变更数据。
- **许可：** 软件 Apache-2.0；数据/呈现素材为 CC BY 4.0，数据包里可执行文件遵循其单列许可。

## 工程复现建议

1. 在 Python 3.12 环境按仓库 README 安装 deliverbench-harbor 和 Harbor 0.20.0，并准备 Docker（建议至少 8 GiB 内存配额）。
2. 用 `python scripts/download_data.py --project all --output-dir data` 拉取固定修订的三例数据并核验文件 SHA256。
3. 对单项目运行 `harbor run -c harbor_job.yaml -a deliverbench_harbor.agents:Codex -m <candidate-model> --resume-trajectory --plugin deliverbench --job-name case-01-my-model`。
4. 用 `deliverbench-check jobs/case-01-my-model --verify-evidence` 验证完整性；synthetic Judge/Responder 的 pipeline probe 仅是协议自检，不是正式模型成绩。

## 局限

三例公开数据不覆盖论文中的全部研究项目；运行需要 Python 3.12、Harbor 0.20.0、Docker 与候选/Judge/Responder 配置；授权材料和未来输入必须对被评 agent 隐藏。

## 对 wiki 的映射

- [DeliverBench 主实体](../../wiki/entities/paper-deliverbench.md)
- [论文归档](../papers/deliverbench_arxiv_2611_03412.md)
- [官方项目页归档](../sites/deliverbench.md)
