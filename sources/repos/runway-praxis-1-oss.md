# runwayml/praxis-1-oss（Praxis-1 官方实现）
> 来源归档（repo） · 读取日期：2026-11-12

- **GitHub：** https://github.com/runwayml/praxis-1-oss
- **项目研究报告：** https://runway.com/research/praxis-1-video-pretraining-to-robot-control
- **权重：** https://huggingface.co/collections/runwayml/praxis-1
- **许可证：** Apache-2.0
- **核对范围：** 仓库 README 的模型说明、环境安装、输入输出、采样与量化说明；`praxis/train/README.md` 与 `praxis/serve/README.md` 的微调和部署章节；源码 `scripts/run_sim.py`、`scripts/run_inference.py`、`praxis/sim/loop.py`、`praxis/pipeline.py`、`praxis/model.py`、`praxis/sampler.py` 的运行入口、关键接口及采样路径。
- **通读：** 已完成

## 实际运行入口与调用链

| 工作 | 入口 | 关键实现 |
|---|---|---|
| MuJoCo 交互仿真 | `python scripts/run_sim.py --config configs/sim_droid.yaml` | `praxis.sim.config.load_config` → `praxis.sim.loop.run` → `SimEnv` / `History` / policy session |
| 离线动作与视频预测 | `python scripts/run_inference.py --config configs/inference_droid.yaml` | `PraxisPipeline`、`Session.step`、`praxis.model.PraxisPolicy` |
| 服务化控制 | `python -m praxis.serve --config configs/serve_droid.yaml --host 0.0.0.0 --port 8000` | WebSocket `/api/v1/session` → 单会话 policy，逐块返动作 |
| 全参数微调 | `torchrun ... scripts/finetune.py --config configs/finetune_base.yaml`（具体参数按 README） | 新 rig 的 normalization / action heads + FSDP |
| 同 rig 任务 LoRA | `scripts/finetune.py` + `configs/finetune_droid_lora.yaml` | 只训练 DROID 动作专家的 LoRA，不适用于新本体 |
| 仿真评测 | `scripts/run_sim_eval.py --suite hard --checkpoint ...` | `praxis/sim/EVAL.md` 定义的 11 个 harder tasks（352 episodes） |

**源码调用依据：** `scripts/run_sim.py.main` 先读取配置再调用 `praxis.sim.loop.run`；loop 的实现注释明确按 reset→observe→`Session.step`→15 Hz 逐行动作执行→历史相机帧回传闭环。 `Session`（`praxis/pipeline.py`）组合 `Autoencoder`、`TextEncoder` 和 `PraxisPolicy.sample_window_joint`；后者调用 `VideoBackbone.forward_chunk_causal`，并将 taps 提交给 `ActionExpert`，`sampler.py` 生成时间步与 CFG 组合。

## 工程注意

- 默认分块每 24 行动作执行后重新观测；首次调用 21 行。不是每 15 Hz 时刻都重新做 27B 推理。
- `praxis/train/README.md` 明确：新 rig 需全参数微调和新动作头；DROID 同 rig 的 LoRA 可单 80 GB GPU，不能直接迁移其他机器人。
- 主 README 对 NVFP4 路由专家是实验性选项，量化后闭环评测应自行验证；不能以单个 NVFP4 QAT 结果替代目标硬件验证。

## 对 wiki 的映射

- [Praxis-1 统一实体](../../wiki/entities/runway-praxis-1.md)
- [技术报告归档](../papers/runway_praxis_1_technical_report_2026_11.md)
- [官方研究页](../sites/runway-praxis-1.md)
