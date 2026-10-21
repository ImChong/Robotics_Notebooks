# EdenLive 项目页（VAST / Project Eden）

- **类型：** 项目主页 / 技术报告入口
- **入口：** <https://vast-eden.ai/?p=edenlive>
- **主体：** VAST / Project Eden
- **代码：** <https://github.com/VAST-AI-Research/EdenLive>（已开源，见[仓库归档](../repos/edenlive-vast-ai-research.md)）
- **技术报告：** <https://assets.vast-eden.ai/assets/edenlive/EdenLive-tech-report.pdf>
- **模型权重：** <https://huggingface.co/VAST-AI/EdenLive>
- **收录日期：** 2026-10-18
- **核查依据：** 项目页和报告作为官方入口；源码、推理方式、开放权重及许可边界按官方仓库 README 与许可证核对。

## 一句话

**EdenLive** 是面向交互式世界的实时生成渲染器：上游程序负责维护状态并产生参考视频，模型负责将其转换成更高细节的视觉流，目标是保持交互逻辑与生成式外观解耦。

## 为什么值得保留

- 它提供了**用传统引擎维持可执行状态、用视频生成模型补充画面细节**的系统范例，可与纯生成式世界模型区分。
- 官方 README 将游戏世界重渲染、视频外观转换，以及把人类视频转成机器人视角用于具身数据增强列为目标用途。
- 仓库、实时推理代码和 5B / 14B 权重均已公开，能进一步检查从参考帧到流式生成的工程路径。

## 开源核查（2026-10-18）

| 项目 | 状态 |
|------|------|
| 代码 | 已开源：离线推理、实时服务、prompt-to-game skill 与试玩前端 |
| 权重 | Hugging Face 已公开 5B / 14B，含双向与少步 chunk-AR 变体 |
| 主仓许可证 | Apache-2.0；工作台首帧生成所用 FLUX.2 [klein] 9B 为非商业许可，工作台使用受其自身条款约束 |
| 数据 | 仓库含两段 proxy 示例视频及对应首帧 / 提示词；未据此推断训练数据集已公开 |

## 参考链接

- [官方项目页](https://vast-eden.ai/?p=edenlive)
- [技术报告](https://assets.vast-eden.ai/assets/edenlive/EdenLive-tech-report.pdf)
- [代码仓库](https://github.com/VAST-AI-Research/EdenLive)
- [Hugging Face 权重](https://huggingface.co/VAST-AI/EdenLive)
- [站内实体页](../../wiki/entities/edenlive.md)
