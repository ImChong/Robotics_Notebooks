# OpenWAM（Stanford）GitHub 仓库归档

> 来源归档

- **标题：** OpenWAM/OpenWAM
- **类型：** repo
- **链接：** <https://github.com/OpenWAM/OpenWAM>
- **代码：** <https://github.com/OpenWAM/OpenWAM>
- **论文：** <https://arxiv.org/abs/2610.07922>
- **项目页：** <https://openwam.stanford.edu>（[归档](../sites/openwam-stanford.md)）
- **许可：** AGPL-3.0-only（NOTICE 与 THIRD_PARTY_NOTICES.md 另述第三方组件）
- **入库日期：** 2026-10-12（浅克隆核查，tip 提交日期 2026-10-10；约 186 stars）
- **沉淀到 wiki：** [paper-openwam-composable-world-action-models.md](../../wiki/entities/paper-openwam-composable-world-action-models.md)
- **区分：** 与 [OpenWAM-Official/OpenWAM](openwam.md)（arXiv:2609.07398）无关，请勿混用。

## 定位

README 自述「可扩展、可组合的世界–动作模型框架」，Alpha 阶段研究软件（Linux，Python 3.11/3.12）：因果视频预训练、视频–动作策略、独立训练的动力学模型，数据接口含 LIBERO、RoboTwin、CALVIN、LeRobot。

## 目录与入口（浅克隆核对）

| 路径 | 内容 |
|------|------|
| `src/open_wam/cli/` | `train.py`、`eval.py`、`sanity.py`、`sim_rollout.py`、`inspect_config.py`、`validate_config.py`（对应 pyproject 的 `openwam-train/eval/sanity/sim-rollout/inspect-config/validate-config`） |
| `src/open_wam/models/` | `video_backbone`、`visual_tower`、`policy_variants`、`action_decoders` |
| `src/open_wam/training/`、`evals/` | 训练循环；LIBERO 滚动评测、指标与产物 |
| `configs/experiments/` | `dual_expert_libero_{joint,video_then_action,action_then_video,decoupled_same_step,conditional_dynamics,...}.yaml`、`parallel_stream_libero_*.yaml`、`causal_video_prediction_*.yaml`、RoboTwin smoke 配置 |
| `configs/local_paths.sample.yaml` | 数据/检查点/仿真器路径注册表 |
| `scripts/` | 数据构建，如 `build_libero_fdm_counterfactual_demo_dataset.py` |
| `docs/` | `running_experiments.md`、`video_action_composition.md`、`extension_sdk.md`、`reproducibility.md` 等 |

## 运行方式（README）

- 安装：`uv sync --frozen --group dev --extra train --extra eval`（或 `pip install 'openwam[train,eval]'`）。
- 无 GPU 冒烟：`openwam-sanity --cfg configs/examples/public_tiny_synthetic_contract.yaml --device cpu --max-batches 1 --rollout-steps 1`（合成数据，不复现论文结果）。
- 训练：参考双专家 LIBERO 配置，4×48GB GPU，`torchrun --standalone --nproc-per-node=4 -m open_wam.cli.train --cfg configs/experiments/dual_expert_libero_joint.yaml`；`--set policy_variant.program=video_then_action` 切 VTA。

## 发布状态（2026-10-12）

- 已发布：预训练视频权重（HF，含参考配置与校验和）。
- 未含：视频前端与文本编码资产、数据集、训练状态。
- 待发布：策略检查点、LIBERO-Long-CF 数据集、IDM 检查点（README 称准备中）。
