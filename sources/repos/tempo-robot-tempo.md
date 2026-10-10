# tempo-robot/TEMPO

> 来源归档

- **标题：** TEMPO: Learning Temporal Context for Dynamic Robot Manipulation
- **类型：** repo
- **组织 / 作者：** tempo-robot；Zhenyang Feng、Jimin Heo、Erik B. Sudderth、Unnat Jain
- **代码：** <https://github.com/tempo-robot/TEMPO>
- **论文：** <https://arxiv.org/abs/2609.16864>
- **项目页：** <https://tempo-robot.github.io/>
- **许可证：** Apache-2.0（仓库含独立 Gemma 许可说明）
- **入库日期：** 2026-10-10
- **一句话说明：** 基于 openpi / π0.5 的 TEMPO 官方实现，包含 SAM 2 特征预处理、PyTorch 微调与策略服务入口。

## 开源核查（2026-10-10）

| 项 | 状态 |
|----|------|
| 仓库可见 | 是，公开 GitHub 仓库 |
| 可运行代码 | 是；README 给出 `uv sync` 环境、模型加载验证、训练与 `serve_policy.py` 推理命令 |
| 主要入口 | `tools/precompute_sam2_tokens.py`、`scripts/train_pytorch.py`、`scripts/serve_policy.py`、`src/openpi/` |
| 数据 | README 说明使用 LeRobot v2.1 / v3.0 布局；训练数据 TODO 尚未解除 |
| 权重 | README TODO 标记训练 checkpoints 待发布；从 π0.5 基础权重微调 |
| 部署 | TODO 中 I2RT YAM 部署待发布；推理接口可通过 openpi-client 调用 |
| 结论 | **实现代码已开源，数据、训练后权重与特定机器人部署仍有缺口** |

README 对应论文的标题说明为 “Closing the Representational Gap for VLAs in Dynamic Settings”，同时指向该 arXiv 与项目页；按论文编号与作者确认，这与 arXiv 当前标题 “TEMPO: Learning Temporal Context for Dynamic Robot Manipulation” 是同一工作，并非另一个同名项目。

## 仓库结构与运行链

| 路径 | 用途 |
|------|------|
| `tools/precompute_sam2_tokens.py` | 对数据集视频帧预计算 SAM 2 token |
| `scripts/compute_norm_stats.py` | 计算训练配置所需的归一化统计 |
| `scripts/train_pytorch.py` | 按配置微调 PyTorch VLA |
| `scripts/serve_policy.py` | 启动 websocket 策略服务 |
| `packages/openpi-client` | 发送 observation 与获取动作的客户端接口 |
| `src/openpi/training/config.py` | TEMPO 及 ablation 配置 |

## 对 wiki 的映射

- [论文归档](../papers/tempo_arxiv_2609_16864.md)
- [项目页归档](../sites/tempo-dynamic-manipulation.md)
- [TEMPO 动态操作实体](../../wiki/entities/paper-tempo-dynamic-manipulation.md)
