# VAST-AI-Research / EdenLive

- **标题：** EdenLive: Real-Time Generative Rendering from Any Visual Reference
- **类型：** repo
- **来源：** VAST / Project Eden
- **链接：** <https://github.com/VAST-AI-Research/EdenLive>
- **项目页：** <https://vast-eden.ai/?p=edenlive>
- **技术报告：** <https://assets.vast-eden.ai/assets/edenlive/EdenLive-tech-report.pdf>
- **模型权重：** <https://huggingface.co/VAST-AI/EdenLive>
- **入库日期：** 2026-10-18
- **一句话说明：** 面向交互式世界的实时生成渲染器：上游引擎或程序维护状态、相机与运动，EdenLive 把参考视频渲染为更高细节的视频流；可用于游戏外观重渲染、视频风格化，也包括将人类视频转成机器人视角以扩充具身数据。
- **沉淀到 wiki：** 是 → [`wiki/entities/edenlive.md`](../../wiki/entities/edenlive.md)

## 开源状态核查（2026-10-18）

| 项 | 结论 |
|----|------|
| **推理代码** | **已开源** — 仓库包含离线推理与实时流式推理服务代码 |
| **模型权重** | **已公开** — Hugging Face 页面发布 5B / 14B 双向与少步（chunk-AR）权重；README 估计全量模型文件约 210 GB |
| **prompt-to-game** | **已开源** — 仓库包含 `game-prototype-turbo` skill 与试玩前端 |
| **许可证** | 仓库主代码及实时服务采用 Apache-2.0；工作台首帧生成使用 FLUX.2 [klein] 9B，该组件采用非商业许可，因此工作台路径有单独限制。权重及各依赖组件的具体许可应按其随附许可证逐项核查，不将主仓许可证外推到所有权重。 |

## README 摘要

EdenLive 将**世界状态 / 规则**和**外观生成**分离：上游 coding agent 或游戏引擎按动作推进状态并输出参考视频；EdenLive 在目标首帧和文本提示下生成细节更丰富的视频。README 报告 24 FPS 或更高的实时目标、480p / 720p 输出；5B 支持单 GPU 720p，另报告 RTX 5090 上约 24.7 FPS（1280×704）的模型推理吞吐。

公开配置包括 Wan2.1-I2V-14B（480p）与 Wan2.2-TI2V-5B（720p）的双向推理和少步 chunk-AR 推理。实时服务接收游戏帧并通过 WebRTC 输出生成视频。仓库安装说明列出 Linux x86-64、NVIDIA GPU、CUDA 13 驱动等已验证环境；README 标注的 80 GB GPU 环境与 5B 单卡 720p 示例需结合各自配置理解。

## 参考链接

- 官方代码：<https://github.com/VAST-AI-Research/EdenLive>
- 项目页：<https://vast-eden.ai/?p=edenlive>
- 技术报告：<https://assets.vast-eden.ai/assets/edenlive/EdenLive-tech-report.pdf>
- 权重：<https://huggingface.co/VAST-AI/EdenLive>
