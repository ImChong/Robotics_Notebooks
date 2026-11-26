# T2-z-latent-optimization/T2-z

- **类型：** 官方代码仓库
- **仓库：** <https://github.com/T2-z-latent-optimization/T2-z>
- **项目页：** <https://t2-z-latent-optimization.github.io/>
- **论文：** [arXiv:2610.22364](https://arxiv.org/abs/2610.22364)
- **数据集：** <https://huggingface.co/datasets/huagailuowen/T2-z-datasets>
- **许可：** Apache-2.0
- **源码状态：** 已公开；数据与模型权重独立于代码仓。2026-11-26 静态核对 [`c72f051`](https://github.com/T2-z-latent-optimization/T2-z/tree/c72f051f5cffcb1f372a32bc488df273556d39b9) 的 README、LICENSE、训练与推理入口、摩擦配置及仿真评测入口；未运行训练或推理。
- **知识节点：** [T2-z](../../wiki/entities/paper-t2-z-environment-latents.md)

## 运行入口

- 安装环境：Linux，README 推荐 Python 3.11 与 CUDA 12.8 对应的 PyTorch。
- 训练：`t2z/train.py`，配置位于 `configs/ours/` 与 `configs/baselines/`；仓库提供任务 launcher。
- 推理：`t2z/infer.py`，配置位于 `configs/inference/`；输入 checkpoint、context table、metadata 及 support/query 索引。
- 数据：`scripts/download_datasets.py` 拉取 HF 的 `T2-z-datasets.zip` 并解包。训练任务目录应有 `train.jsonl`、`sampling.json`、`action_stats.json` 与 `data/`。
- 权重：还需 Wan2.2-TI2V-5B 基座模型和 BWM 初始化 checkpoint；在线仿真另需 simulator dependencies。

## 静态调用链与复现边界

- `t2z/train.py` 读取 YAML，准备环境分组，`--check-data` 输出数据描述后返回；训练派发至 `t2z.training.ours.main()`。启用该课程时，`launch_curriculum_grouped_stage1()` 切换参数梯度，并用 checkpoint logger 写 `.safetensors` 及配套 `.context_table.json`。
- `t2z/infer.py` 读取行号，`group_samples()` 按环境 ID 分组，写出 `runtime/support.jsonl`、`runtime/config.yaml`、`provenance.json` 后派发至 `t2z.inference.ours.main()`。通用入口不应被当成严格切分验证器；用户须事先保证 support/query 不仅行号互斥，也不来自同一原始轨迹。
- 摩擦配置 `configs/inference/sim_pushbox_friction.yaml` 的 ours 路径调用 `_freeze_pipe()` 与 `_adapt_ttt_state()`，常规初始化取已学习 context table 的均值；40 次 latent 更新和 25 次生成去噪是两件事。实现还有可配置正则、裁剪及初始化选项；不同于论文用公式概括的单一抽象流程。环境分组缓存复用优化后的 z；无特殊覆盖时 `_run_autoregressive()` 生成查询预测。
- 在线仿真入口为 `python -m wan_video_action.evaluation.sim_action_selection`。它起模拟器 / 模型 worker、采集独立交互、预测候选、执行选中动作并写评测结果；不同于 `t2z/infer.py` 的离线视频输出，并不等于真机控制入口。对照实验应记录是否允许使用已观察的 support 结果参与筛选（`--no-use-observed-support-outcomes` 可关闭）。
- 已核对到官方公开的代码、数据和两个上游初始化权重。README 仍要求自备训练得到的 T2-z checkpoint 与匹配 context table；本次未确认官方成品下载，不把初始化权重计作成品。Apache-2.0 覆盖此代码仓，不自动覆盖外部模型与数据。

## 关联资料

- [项目页归档](../sites/t2-z-latent-optimization.md)
- [论文摘录](../papers/t2_z_arxiv_2610_22364.md)
