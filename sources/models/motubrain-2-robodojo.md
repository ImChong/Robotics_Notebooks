# Motubrain 2.0 RoboDojo 权重发布 | RoboDojo Checkpoint

- **模型卡：** [Motubrain/MotuBrain2.0-Robodojo](https://huggingface.co/Motubrain/MotuBrain2.0-Robodojo)
- **核验版本：** [e7422d94f9c6c29f266211a86c219921333b21b4](https://huggingface.co/Motubrain/MotuBrain2.0-Robodojo/tree/e7422d94f9c6c29f266211a86c219921333b21b4)，2026-11-26 核对。
- **许可证：** 模型卡标注 CC BY-NC-SA 4.0，商用须获权利方事先书面授权；第三方组件另遵各自条款。
- **核对范围：** 读模型卡、推理配置、动作统计文件和版本清单；未下载大权重或执行推理。
- **通读：** 已完成

版本目录含 `step-5000.safetensors`（39,048,437,614 字节，清单记录）、`config.yaml`、`action_stats.json`、模型卡与许可证。不含模型类实现或启动脚本。配置含 `falcon.models.motubrain.MotubrainModel`、24 层、55 维动作与 mask、三路相机名和 `causal_9_to_3` 稀疏历史；类名是复现线索，不是可运行入口。目录未确认此权重对应报告的所有部署设置或评测配置。

同期官方组织的公开模型 API 及用户指定搜索只返回该项；Base、Distill、RoboCasa365、EBench、LIBERO-Plus 的公开直达候选没有返回可读模型卡，不能据此证明永久未发布。未把别的托管方条目作为官方权重依据。

## 对 Wiki 的映射

- [Motubrain 2.0 技术报告](../../wiki/entities/paper-motubrain-2.md)
