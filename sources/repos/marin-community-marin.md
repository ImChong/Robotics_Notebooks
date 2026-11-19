# marin-community/marin｜开放 LLM 研究与训练栈

- **仓库：** <https://github.com/marin-community/marin>
- **官网：** <https://marin.community/>
- **文档：** <https://marin.readthedocs.io/en/latest/>
- **核查日期：** 2026-11-19（读取 GitHub 默认分支）
- **核对范围：** 仓库 `README（定位、Delphi / 8B / 32B、教程与 example）；`experiments/tutorials/train_tiny_model.py 全文；`lib/marin/src/marin/experiment/cli.py 的 Click 入口、BuildContext、版本参数、lower / run 及 plan 分支；`docs/tutorials/first-experiment.md 的 lazy dependency、tokenize、train、StepRunner 说明
- **通读：** 已完成
- **开源状态：** 公开可运行的示例与训练代码；**本次为静态阅读**，未安装依赖、未实际执行训练或复现 535B Mochi。
- **wiki 映射：** [Marin 开放式大模型预训练](../../wiki/entities/marin-open-llm-pretraining.md)

## 与文章关系

官方代码仓库包含数据管理、训练实验与基础设施实现。README 明确可从官方教程训练小模型，并提供 Delphi scaling recipe、训练语料配方、checkpoint 与实验记录的入口。**不能从 TinyStories 的教学示例推断它等同于 Mochi 535B 的真实生产训练脚本。**

### 核实的最小运行入口

`experiments/tutorials/train_tiny_model.py 定义 Click `main(device, data)`，经 `@build_options` 注册 `--version`（必传）、`--run`（否则只输出计划）、`--device`、`--dataset`：

```bash
# 仅输出依赖计划（默认 CPU、TinyStories）
uv run python -m experiments.tutorials.train_tiny_model --device cpu --dataset tinystories --version dev
# 真正执行小模型样例（需按官方安装说明配置）
uv run python -m experiments.tutorials.train_tiny_model --device cpu --dataset tinystories --version dev --run
```

**实际源码路径（不采用通用猜测）**

1. `dataset()` 根据 `tinystories / wikitext / fineweb-edu` 返回 `tokenized()` 或 `pretokenized()` 的惰性 `ArtifactStep`。
2. `build()` 根据设备资源 `ResourceConfig` 和模型 `llama_nano / llama_150m` 构造 `train_lm()` checkpoint artifact；优化器 `AdamConfig`，示例 `num_train_steps=100`。
3. `@build_options` 在 `BuildContext` 中构建依赖图；不加 `--run` 时 lower 并显示 plan / resolved versions；加入 `--run` 调用 `marin.execution.lazy.run` 执行依赖。
4. 官方教程说明：tokenization 在训练前按依赖图执行并可缓存，`train_lm` 通过 Fray / Levanter 管理训练硬件与 checkpoint；此为文档所述预期执行路径，本次**没有运行验证**。

## 开放边界与工程风险

- 仓库公开源码与教程，是**可查验的训练软件栈**；运行需要相应依赖/资源，不能据此认定超大规模运行无需专门集群。
- 官方 2026-11-17 公告称其开始公开 Mochi **57** 个预训练 checkpoints，但它们不是做过指令/安全后训练的部署版助手。
- 需分别检验训练数据来源与再分发条件；「源可追踪」不等于 25T 原始数据都在仓库里。

## 官方入口

- [仓库 README](https://github.com/marin-community/marin/blob/main/README.md)
- [教学训练脚本](https://github.com/marin-community/marin/blob/main/experiments/tutorials/train_tiny_model.py)
- [CLI 实现](https://github.com/marin-community/marin/blob/main/lib/marin/src/marin/experiment/cli.py)
- [第一个实验教程](https://github.com/marin-community/marin/blob/main/docs/tutorials/first-experiment.md)
- [Delphi 技术长文](https://openathena.ai/blog/delphi/)
- [项目主页归档](../sites/marin-community.md)
- [Open Athena 文章归档](../blogs/openathena-scaling-marins-open-llm-pretraining-2026-11-05.md)
