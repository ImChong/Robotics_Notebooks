# SLA / OpenHelix-Team/Spatial-Forcing 官方代码子目录

- **仓库：** https://github.com/OpenHelix-Team/Spatial-Forcing
- **SLA 独立实现：** https://github.com/OpenHelix-Team/Spatial-Forcing/tree/master/SLA
- **运行说明：** https://github.com/OpenHelix-Team/Spatial-Forcing/blob/master/SLA/README.md
- **论文：** https://arxiv.org/abs/2611.08373
- **检查日期：** 2026-11-08
- **官方项目主页：** https://sla-team.github.io/omnisf_projectpage/
- **作者资源合集：** https://huggingface.co/collections/haofuly/sla-collection
- **开放状态：** 部分开源；已有训练、推理与评估入口以及预训练/后训练模型资源；完整持续预训练原始数据和 RoboDojo-Real 训练与评估仍在 TODO。
- **核对范围：** 阅读 SLA/README.md、scripts/train_align.py、scripts/serve_policy.py、examples/libero/main.py、src/openpi/models/pi0_align.py、src/openpi/training/align_utils.py 的相关实现，以及 external_config/pretrain.py、external_config/train_libero_plus.py。非全仓代码审计，未实际训练/运行。

## 关键路径（基于源码）

| 组件 | 文件及职责 |
|---|---|
| 训练入口 | [scripts/train_align.py](https://github.com/OpenHelix-Team/Spatial-Forcing/blob/master/SLA/scripts/train_align.py)：读取训练配置/批次，计算 action + alignment 损失，求导和更新 |
| 空间对齐策略 | [src/openpi/models/pi0_align.py](https://github.com/OpenHelix-Team/Spatial-Forcing/blob/master/SLA/src/openpi/models/pi0_align.py)：Pi0Align 扩展 Pi0，抓取中间视觉特征，计算动作及空间特征损失 |
| 空间教师与数据对齐 | [src/openpi/training/align_utils.py](https://github.com/OpenHelix-Team/Spatial-Forcing/blob/master/SLA/src/openpi/training/align_utils.py)：对观测提取并整理教师目标特征 |
| 训练配置 | [src/openpi/external_config/pretrain.py](https://github.com/OpenHelix-Team/Spatial-Forcing/blob/master/SLA/src/openpi/external_config/pretrain.py)、[train_libero_plus.py](https://github.com/OpenHelix-Team/Spatial-Forcing/blob/master/SLA/src/openpi/external_config/train_libero_plus.py) |
| 推理服务 | [scripts/serve_policy.py](https://github.com/OpenHelix-Team/Spatial-Forcing/blob/master/SLA/scripts/serve_policy.py)：从指定 checkpoint 建立 Policy，经 WebsocketPolicyServer 服务 |
| 仿真评测 | [examples/libero/main.py](https://github.com/OpenHelix-Team/Spatial-Forcing/blob/master/SLA/examples/libero/main.py)：WebsocketClientPolicy 请求动作块，在 LIBERO 环境执行并统计成功率 |

## 作者当前发布的模型与数据入口

| 资源 | 官方地址 | README 中角色 |
|---|---|---|
| SLA_base | https://huggingface.co/haofuly/SLA_base | 持续预训练后的基础 checkpoint |
| SLA_training_component | https://huggingface.co/haofuly/SLA_training_component | OmniVGGT、DINOv2 等对齐训练辅助组件 |
| SLA_benchmark_ckpts | https://huggingface.co/haofuly/SLA_benchmark_ckpts | 已后训练的 benchmark checkpoint |
| SLA_real_world_demo | https://huggingface.co/datasets/haofuly/SLA_real_world_demo | 真实世界演示数据，不是全量持续预训练语料 |

截至 2026-11-08，官方 README TODO 将「Continual pre-training corpus」列为已完成，但「Continual pre-training data」仍列待发布；两者分别对应代码/处理工作与数据开放程度，不能用 ✅ 误证全量数据可下载。RoboDojo-Real post-training and evaluation 仍为 ❌。作者仓库根目录同时包含 Spatial Forcing 旧版；**SLA 特有实现位于 `SLA/`**，不可将两项工作合并为一个论文实体。

## README 中可复现的最小操作路径

1. 使用 uv 初始化环境；准备官方说明中的 LIBERO 数据、SLA_base 和 SLA_training_component。
2. 运行 `uv run scripts/compute_norm_stats.py --config-name train_libero_plus`。
3. 运行 `uv run --no-sync scripts/train_align.py train_libero_plus --exp-name=experiments`。
4. 推理时一端运行 `uv run scripts/serve_policy.py policy:checkpoint --policy.config=train_libero_plus --policy.dir=<CKPT_DIR> --port 8000`；另一端运行 `python examples/libero/main.py --args.task-suite-name libero_spatial --args.num_trials_per_task 1 --args.port 8000`。

> 源码核查提醒：公开配置中的 vla_adapt_layers_align=False、vla_layers_align=12，是固定层配置；论文所述门控优势不可直接套到默认配置上。此条不代表已完成任何训练复现实验。

## 对 wiki 的映射

- [SLA 实体页](../../wiki/entities/paper-sla-spatial-language-action.md)
- [原始论文归档](../papers/sla_spatial_language_action_arxiv_2611_08373.md)
- [项目主页核查](../sites/sla_spatial_language_action.md)
