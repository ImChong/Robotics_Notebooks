# LAW 官方代码仓库

- **仓库：** https://github.com/hey-cjj/LAW
- **默认分支：** main
- **关联论文：** [arXiv:2610.23334](https://arxiv.org/abs/2610.23334)
- **项目页：** https://hey-cjj.github.io/LAW-project/
- **模型 / 数据：** [Hugging Face 模型与数据入口](https://github.com/hey-cjj/LAW#2-models-and-data)
- **核查日期：** 2026-10-20
- **一句话说明：** LAW 的官方训练、推理与 RoboTwin 全任务评测实现，覆盖 FastWAM 和 OpenWAM 两条 WAM 后端。

## 开源范围

官方 README 公开四类模型（LAW-FastWAM、LAW-VLM-FastWAM、LAW-OpenWAM、LAW-VLM-OpenWAM）、训练初始化权重以及 WAM/VLM 训练数据，并提供下载校验脚本、训练入口和推理/评测流程。仓库包含 scripts/run.py 的 precompute、train-wam、train-vlm、serve、eval 子命令；完整训练与 50 任务评测要求四节点、每节点八张 GPU。论文数值可通过已发布检查点复测。

许可需按资产分别判断：根代码 MIT、作者模型权重 Apache-2.0、作者标注 CC-BY-4.0；上游代码、媒体与资产保留各自许可。请按仓库 NOTICE 和 docs/artifact_licenses.md 核对下载与再分发边界。

## 运行入口

- **准备：** 按 environments/README.md 安装 Python 3.10、模型与 RoboTwin 环境，设置 LAW_* 路径及各推理器解释器。
- **下载与校验：** scripts/download.py models --profile all、foundations、assets 与 datasets --dataset all，随后运行 scripts/verify.py --profile all --scan-source。
- **WAM：** scripts/run.py precompute --kind latent 后使用 scripts/run.py train-wam --profile 16d（FastWAM）或 --profile 20d（OpenWAM）。
- **VLM：** scripts/run.py train-vlm --stage q1q2 --profile 16d，之后以 --stage q3 训练相应 profile。
- **评测：** 单任务使用 scripts/run.py eval --profile 16d --task <task> --episodes 1；完整 RoboTwin 评测在四节点上运行 eval --profile 16d 或 20d。

- **归档：** [项目页核查](../sites/law.md) · [论文摘录](../papers/law_arxiv_2610_23334.md)
- **wiki：** [LAW 实体页](../../wiki/entities/paper-law.md)
