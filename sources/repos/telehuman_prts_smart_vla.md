# SMART-VLA 权重与 TeleHuman/PRTS 共享模型实现

- **官方 SMART-VLA 权重：** https://huggingface.co/TeleEmbodied/SMART-VLA
- **共享代码仓库：** https://github.com/TeleHuman/PRTS
- **关联项目论文：** https://arxiv.org/abs/2610.07652
- **PRTS 自身独立论文：** https://arxiv.org/abs/2604.27472（与 SMART 不是同一篇论文）
- **核对日期：** 2026-11-11
- **开放状态：** 已发布权重、通用后训练与 WebSocket 推理入口；SMART-Sim 专属合成引擎未见公开。

## 实际源码入口（本轮阅读）

1. `scripts/compute_stats.py`：通过训练配置与 LeRobotDataset 元数据加载多数据集并计算归一化统计，入口为 `compute(args)`。
2. `scripts/ft/launch_finetune.sh`：配置 `dataset_config_path`、`embodiment_tag`、`chunk_size`、模型路径，调用 `torchrun ... scripts/train.py`。
3. `scripts/serve_policy.py`：`tyro.cli(Args)` → `create_default_policy` → 加载 `PRTS_Qwen3VL` / processor → `PRTSInterface` → `WebsocketPolicyServer.serve_forever()`。
4. 自定义权重需在该脚本注册 `EnvMode` 与 `DEFAULT_CHECKPOINT`（没有现成 `--ckpt-path` 参数）；外部机器人客户端与动作维数/归一化需自行对齐。

## 复现注意

- PRTS README 的示例数据为 LeRobot v2.1，SMART-Data 发布为 LeRobot v3.0；尚未验证直接兼容。
- 通用 PRTS 后训练/推理代码不等同于 SMART 论文的第一阶段全量合成预训练配方；尤其 SMART 论文不使用 PRTS 论文的 CRL 对比预训练目标。
- 未读取到 SMART-Sim 资产处理、agent 自动任务生成、96 worker 分布式任务调度的官方可运行代码，因此不绘制这些模块的“源码级时序图”。

## 关联

- [SMART 论文归档](../papers/gamma_art_smart_arxiv_2610_07652.md)
- [SMART 官方主页归档](../sites/gamma_art_smart_project.md)
- [SMART 实体详情](../../wiki/entities/paper-gamma-art-smart.md)
