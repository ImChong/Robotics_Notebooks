# UniPat-AI/SPU：开放任务贡献仓库

> 依据官方 README 及公开目录归档；区别于 Hugging Face 分发的 Harbor benchmark 任务包。

- **类型：** repo / benchmark infrastructure / submission workflow
- **GitHub：** https://github.com/UniPat-AI/SPU
- **论文：** https://arxiv.org/abs/2611.15049
- **项目主页：** https://spu.unipat.ai/
- **公开数据：** https://huggingface.co/datasets/UnipatAI/SPU
- **许可：** Apache-2.0（GitHub 代码）；公开数据 CC BY 4.0（HF）
- **入库日期：** 2026-11-14
- **核对范围：** 仓库 README、根目录、CONTRIBUTING 与官方数据卡的运行说明；未逐行审阅全部脚本与提交 CI 逻辑。
- **通读：** 已完成

## 仓库身份与能力边界

本仓面向新实验任务的**贡献、模板与审核**，可见 `benchmark/`、`templates/submission/`、`quality/`、`scripts/`、`tests/`、`.github/`、`docs/` 等目录。任务 PR 使用 `spu/pr-format` 和 `spu/clarity-audit` 检查提交完整性，科学有效性仍需维护者审阅。

**已发布的可运行 benchmark 位于 [UnipatAI/SPU](https://huggingface.co/datasets/UnipatAI/SPU)，不是本 GitHub 仓库包含论文全部 agent 框架。** HF 任务有 `task.toml`、`instruction.md`、`environment/`、`tests/`、`solution/`；公开部分数据与 grader、oracle 均可获取，私有实验仍隐藏。

## 官方运行入口（Harbor oracle 自检）

官方数据卡要求 Harbor 0.25.0（Python 3.12+）与 Docker：

```bash
hf download UnipatAI/SPU --repo-type dataset --local-dir SPU
cd SPU
harbor run -p tasks/fluorene-dehydration-concentrations -a oracle -e docker \
  --override-gpus 0 --override-cpus 1 --override-memory-mb 1024 -o ../jobs
```

- `-a oracle` 重放公开参考答案，测试 grader 能否正常工作；并非调用论文所评的 13 种智能体。
- 真实 agent 应只看到 `instruction.md` 与 `environment/public/`；`tests/`、`solution/`、`SOURCES.json` 必须隔离。
- 官方明确：论文所用的审计网络工具和来源识别筛查器不在本次开源包中，重新评测不能直接宣称复现论文同口径分数。

## 对 wiki 的映射

- [SPU 论文实体](../../wiki/entities/paper-science-prediction-universe.md)
- [论文来源](../papers/science_prediction_universe_arxiv_2611_15049.md)
- [项目入口](../sites/science-prediction-universe.md)
