# tempo-robot/TEMPO

> 来源归档

- **标题：** TEMPO: Learning Temporal Context for Dynamic Robot Manipulation
- **类型：** repo
- **组织 / 作者：** tempo-robot；Zhenyang Feng、Jimin Heo、Erik B. Sudderth、Unnat Jain
- **代码：** <https://github.com/tempo-robot/TEMPO>
- **论文：** <https://arxiv.org/abs/2609.16864>
- **项目页：** <https://tempo-robot.github.io/>
- **许可证：** Apache-2.0（仓库含独立 Gemma 许可说明；配套 `TempoBench` 仓库 `LICENSE` 目前为占位文本）
- **入库日期：** 2026-10-10
- **通读：** 已完成
- **核对范围：** 已核查 `tempo-robot/TEMPO` 目录树 `_tree.txt`（L1–100）、`README.md`（L1–203）、`tools/{sam2_encoder,precompute_sam2_tokens}.py`、`scripts/{train_pytorch,serve_policy}.py` 与 `src/openpi/models_pytorch/pi0_pytorch.py`（L331–430 及 `SAM2CrossAttnFusion`、`ActionHistoryCondMLP`、`ActionHistoryTokens`、`UtActionDecoder` 钩子）；并核查配套基准仓库 `tempo-robot/TempoBench` 的 `_tree.txt` 与 `README.md`（L1–189）。
- **一句话说明：** 基于 openpi / π0.5 的 TEMPO 官方实现，包含 SAM 2 特征预处理、PyTorch 微调与策略服务入口；配套 `tempo-robot/TempoBench` 仓库已公开 LeRobot v2.1 基准数据集。

## 开源核查（2026-11-14 复核）

| 项 | 状态 |
|----|------|
| 仓库可见 | 是，公开 GitHub 仓库（`tempo-robot/TEMPO` 与 `tempo-robot/TempoBench`） |
| 可运行代码 | 是；`TEMPO` README 给出 `uv sync` 环境、模型加载验证、训练与 `serve_policy.py` 推理命令 |
| 主要入口 | `tools/sam2_encoder.py`、`tools/precompute_sam2_tokens.py`、`scripts/train_pytorch.py`、`scripts/serve_policy.py`、`tools/ut/*`、`src/openpi/models_pytorch/pi0_pytorch.py` |
| 数据 | **基准数据已公开**：配套仓 `tempo-robot/TempoBench` 通过 Git LFS（约 750 MB）发布 257 episodes（182 已标注、97,265 总帧 ≈ 54 分钟 @ 30fps、55,482 定位帧）的 LeRobot v2.1 数据；但策略微调真机示范数据仍在 `TEMPO` TODO 中 |
| 权重 | `TEMPO` README TODO 仍标记训练 checkpoints 待发布；需自备 `pi05_base` 权重微调 |
| 部署 | TODO 中 I2RT YAM 硬件部署待发布；推理服务端可通过 `openpi-client` 调用 |
| 结论 | **模型训练/推理代码与 TEMPO-Bench 基准数据集均已开源；训练后 VLA 权重与 YAM 真机部署仍有缺口** |

README 对应论文的标题说明为 “Closing the Representational Gap for VLAs in Dynamic Settings”，同时指向该 arXiv 与项目页；按论文编号与作者确认，这与 arXiv 当前标题 “TEMPO: Learning Temporal Context for Dynamic Robot Manipulation” 是同一工作，并非另一个同名项目。

## 仓库结构与运行链

| 路径 | 用途 |
|------|------|
| `tools/sam2_encoder.py` / `tools/precompute_sam2_tokens.py` | 提取并缓存 SAM 2.1-Tiny post-memory-attention 池化特征 `(64, 256)` |
| `tools/ut/*` | 可选 `UtActionDecoder` 相关的 MVAE 训练与窗口编码工具 |
| `scripts/compute_norm_stats.py` | 计算训练配置所需的归一化统计 |
| `scripts/train_pytorch.py` | 按配置微调 PyTorch VLA（注入 `SAM2CrossAttnFusion`、`ActionHistoryTokens` / `ActionHistoryCondMLP`） |
| `scripts/serve_policy.py` | 启动 websocket 策略服务 |
| `packages/openpi-client` | 发送 observation 与获取动作的客户端接口 |
| `src/openpi/models_pytorch/pi0_pytorch.py` | `SAM2CrossAttnFusion`（`tanh(gate) * out_proj` 融合主视角 `vis_tokens` 与 SAM2 记忆 token）、可选 `temporal_attn_period` 及 `ActionHistoryCondMLP` / `UtActionDecoder` |
| `src/openpi/training/config.py` | TEMPO 及 ablation 配置 |
| `tempo-robot/TempoBench`（配套仓） | 257 episodes LeRobot v2.1 数据集、`tempobench` 读取器及 `scripts/render_mujoco.py` / `visualize_object_motion.py` |

## 对 wiki 的映射

- [论文归档](../papers/tempo_arxiv_2609_16864.md)
- [项目页归档](../sites/tempo-dynamic-manipulation.md)
- [TEMPO 动态操作实体](../../wiki/entities/paper-tempo-dynamic-manipulation.md)
