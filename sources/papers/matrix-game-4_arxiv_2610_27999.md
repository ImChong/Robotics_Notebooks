# Matrix-Game 4.0 — 游戏引擎执行、生成模型渲染

- **类型：** paper / 原论文 HTML 与官方源码核对
- **论文：** <https://arxiv.org/abs/2610.27999>
- **代码：** <https://github.com/SkyworkAI/Matrix-Game>
- **版本：** arXiv 2026-10-20
- **核对范围：** arXiv HTML 的引言/方法/数据/实验/局限与主要实现附录，结合官方 README 和运行入口核对；未复现实验与 GPU benchmark。
- **通读：** 已完成

## 问题与方法

将「规则正确」与「视觉逼真」拆分：原生游戏引擎管理物理、碰撞、角色、事件和持久状态，输出 low-poly RGB。条件视频扩散/自回归模型结合场景文本与参考帧做视觉翻译；多模态 coding agent 在研究路线中负责编写游戏规则和生成动态描述，不能与已经部署的渲染推理模块混同。推理栈采用轻量 TAEHV、少步蒸馏、FP8、CUDA Graph、环形 KV cache 与异步媒体 IO。

## 输入、输出与复现路径

- **输入与输出：** 游戏引擎低多边形帧 + 固定或动态场景描述 + 参考帧；模型输出视觉增强后的连续帧（HUD / 音频由外部层提供）。
- **实际入口：** Matrix-Game-4/scripts/serve.sh、matrix_game/cli.py、README Offline Rendering / Browser Demo
- **官方复现：** 子目录 `Matrix-Game-4/` 安装后，离线执行 `matrix-game render --scene ancient-egypt --input lowpoly.mp4 --resources resources --preset realtime --prompt-mode fixed --output outputs/egypt`；在线用 `mediamtx configs/mediamtx.yml` 和 `bash scripts/serve.sh --scene ancient-egypt --resources resources --preset realtime --fps 40`。RTX 5090 / 5080 / 5070 Ti 需显式选择对应 preset。

## 实验观察与适用边界

- **作者报告：** 论文在 RTX 5090、864×480、单步模型推理下报告稳态 83.4 FPS、key-to-frame 112 ms、峰值约 18.2 GiB；5080 / 5070 Ti 固定提示词下模型稳态约 67 / 58 FPS。上述指标主要是模型侧，不是浏览器 WebRTC 端到端指标。
- **论文/实现局限：** 生成渲染不负责碰撞/游戏逻辑；HTTP/RTSP/WebRTC 示例仅适合可信单用户环境，不能未经鉴权直接暴露公网。源代码 Apache-2.0，模型和游戏资源按各自许可。README 的 `--fps 40` 是采集目标，不保证端到端吞吐。

## 对 wiki 的映射

- [独立论文详情](../../wiki/entities/paper-matrix-game-4.md)
- [11 项开源世界模型技术对照](../../wiki/comparisons/open-source-world-models-11-2026.md)
