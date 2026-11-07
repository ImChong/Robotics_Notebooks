# SLA / OpenHelix-Team/Spatial-Forcing 官方代码子目录

- **仓库：** https://github.com/OpenHelix-Team/Spatial-Forcing
- **SLA 独立实现：** https://github.com/OpenHelix-Team/Spatial-Forcing/tree/master/SLA
- **运行说明：** https://github.com/OpenHelix-Team/Spatial-Forcing/blob/master/SLA/README.md
- **论文：** https://arxiv.org/abs/2611.08373
- **检查日期：** 2026-11-07
- **开放状态：** 部分开源；公开训练、推理入口和可下载的部分模型权重/组件；部分数据和 RoboDojo-Real 复现仍在 TODO。
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
