# Matrix-Game 4.0 — 官方代码与运行入口

- **标题：** Matrix-Game 4.0
- **类型：** repo / 官方 README + 运行入口
- **机构：** 昆仑万维 SkyworkAI（联合高校研究团队）
- **代码：** <https://github.com/SkyworkAI/Matrix-Game>
- **项目页：** <https://matrix-game-v4-0.github.io/page/>
- **论文/技术报告：** <https://arxiv.org/abs/2610.27999>
- **读入日期：** 2026-11-20
- **核对范围：** 官方代码仓 Matrix-Game-4/README.md 全文、Matrix-Game-4/scripts/serve.sh、matrix_game/cli.py、README Offline Rendering / Browser Demo；核对模型发布说明、安装/推理入口及许可证说明（未实际下载权重或跑 GPU）。
- **通读：** 已完成

## 可复核的实现事实

- **功能：** 游戏引擎维护确定的世界规则和 low-poly 帧，因果视频模型将其实时神经渲染为高质量画面。
- **主要组件：** 原生游戏引擎、因果 DiT、TAEHV、MediaMTX/WebRTC
- **公开边界：** 模型/游戏资产和推理代码已开放；源码与资产许可分开核对
- **官方实现入口：** Matrix-Game-4/scripts/serve.sh、matrix_game/cli.py、README Offline Rendering / Browser Demo

## 数据与控制流

1. 玩家输入和游戏脚本
2. 原生引擎更新碰撞/游戏状态
3. 引擎输出 low-poly RGB 与场景描述
4. 少步因果视频渲染模型进行神经翻译
5. 媒体服务同步叠加 HUD 和音频

## 复现步骤

子目录 `Matrix-Game-4/` 安装后，离线执行 `matrix-game render --scene ancient-egypt --input lowpoly.mp4 --resources resources --preset realtime --prompt-mode fixed --output outputs/egypt`；在线用 `mediamtx configs/mediamtx.yml` 和 `bash scripts/serve.sh --scene ancient-egypt --resources resources --preset realtime --fps 40`。RTX 5090 / 5080 / 5070 Ti 需显式选择对应 preset。

## 可信边界与限制

生成渲染不负责碰撞/游戏逻辑；HTTP/RTSP/WebRTC 示例仅适合可信单用户环境，不能未经鉴权直接暴露公网。源代码 Apache-2.0，模型和游戏资源按各自许可。README 的 `--fps 40` 是采集目标，不保证端到端吞吐。

## 关联节点

- [Matrix-Game 4.0 详情](../../wiki/entities/paper-matrix-game-4.md)
- [11 项世界模型技术对照](../../wiki/comparisons/open-source-world-models-11-2026.md)
