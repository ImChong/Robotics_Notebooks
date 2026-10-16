# Thinker-WM 官方代码仓库

> 来源归档（repo；核查日期：2026-10-16）

- **标题：** Thinker-WM: A Unified World Model with Shared Multimodal Backbone for Embodied Intelligence
- **组织：** UBTECH-Robot / Thinker Team
- **代码：** <https://github.com/UBTECH-Robot/Thinker-WM>
- **论文：** <https://arxiv.org/abs/2610.18447>（[论文归档](../papers/thinker_wm_arxiv_2610_18447.md)）
- **项目页：** <https://thinkercosmos.ubtrobot.com/thinker-wm/>（[站点归档](../sites/thinker-wm.md)）
- **模型：** <https://huggingface.co/UBTECH-Robotics/Thinker-WM>（[模型归档](../models/thinker-wm.md)）
- **许可：** CC BY-NC-SA 4.0（仓库 LICENSE）
- **发布边界：** README 自称官方推理发布，支持加载 checkpoint 后运行六种推理模式；不是训练代码完整开放的声明。
- **核查日期：** 2026-10-16

## README 所述能力

| 模式 | 输入 | 输出 |
|------|------|------|
| `wam` | 文本提示、首帧、机器人（状态可选） | 视频 + 动作（可选深度） |
| `wm` | 文本提示、首帧 | 视频（可选深度） |
| `vla` | 文本提示、首帧、机器人 | 动作 |
| `a2v` | 首帧、动作、机器人 | 视频 |
| `v2a` | 视频、机器人 | 动作 |
| `uncond` | 首帧、机器人 | 视频 + 动作 |

README 还提供命令行 / Python 推理入口和示例，要求 Python 3.10+、NVIDIA CUDA GPU，且将 `flash-attn` 列为硬依赖。机器人名称用于动作归一化和维度映射；状态可作为相对末端动作恢复到绝对量的锚点。

## 开放边界与核查备注

- README 称主 checkpoint 约 24 GB，VAE 权重为 GB 级；模型文件由独立权重目录管理。
- README 的 “HuggingFace Download” 小节仍保留 `<your-org>/thinkerwm` 占位说明，代码示例也未替换成用户提供的模型仓库地址。故将模型仓库单独归档；实际文件、可下载性与版本应以 Hugging Face 当前页面为准。
- README 标题使用 “Shared Multimodal Backbone”，arXiv 题名为 “Shared Multimodal Representations”；README 内引用块还留有旧 eprint `2607.15330`。论文标识以 arXiv:2610.18447 为准。
- README 概述称预训练数据含 20K 小时第一视角数据和 10K 小时机器人数据，并提到约 4K 小时深度监督；论文摘要另报告约 1K 小时仿真数据。论文数值以论文归档为准。
