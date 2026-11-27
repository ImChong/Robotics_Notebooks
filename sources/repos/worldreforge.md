# WorldReforge：Nano 推理与评测源码

- **源码：** <https://github.com/aim-uofa/WorldReforge>
- **论文：** [arXiv 原文归档](../papers/worldreforge_arxiv_2611_33925.md)
- **项目页：** [项目主页归档](../sites/worldreforge.md)
- **许可：** 代码 Apache-2.0；第三方依赖、权重和数据各用其许可。
- **入库日期：** 2026-11-27
- **核对范围：** 静态检查 README、pyproject、公开命令及 CLI 和 pipeline 的调度；未下载数据或运行 GPU 推理。
- **Wiki：** [WorldReforge](../../wiki/entities/paper-worldreforge.md)

[`scripts/evaluate.sh`](https://github.com/aim-uofa/WorldReforge/blob/main/scripts/evaluate.sh) 启动 [`worldreforge run`](https://github.com/aim-uofa/WorldReforge/blob/main/src/worldreforge/cli.py)，[`run_pipeline`](https://github.com/aim-uofa/WorldReforge/blob/main/src/worldreforge/evaluation/pipeline.py) 校验文件和模型后冻结清单，通过独立子进程运行 `infer`，核对推理覆盖，调用跟踪/评分并验收。公开环境描述依赖带 Cosmos3 Nano 的 CUDA Cosmos Framework，使用 Wan2.2 VAE 和 Qwen3-VL-8B-Instruct tokenizer 文件（不是 Qwen 模型权重）。CoTracker3 用于运动评分；`--pixels-only` 可跳过跟踪。当前实现覆盖推理与评测，不把论文的训练/采集流程写成已发布入口。
