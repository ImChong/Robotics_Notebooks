# Matrix-Game 4.0 — 官方主页与开源核验

- **名称：** Matrix-Game 4.0
- **类型：** site / project
- **项目页：** <https://matrix-game-v4-0.github.io/page/>
- **代码：** <https://github.com/SkyworkAI/Matrix-Game>
- **技术原文：** <https://arxiv.org/abs/2610.27999>
- **机构：** 昆仑万维 SkyworkAI（联合高校研究团队）
- **核对日期：** 2026-11-20
- **核对范围：** 官方仓库 README 链接、发布内容及对应介绍
- **通读：** 已完成

## 技术定位

将「规则正确」与「视觉逼真」拆分：原生游戏引擎管理物理、碰撞、角色、事件和持久状态，输出 low-poly RGB。条件视频扩散/自回归模型结合场景文本与参考帧做视觉翻译；多模态 coding agent 在研究路线中负责编写游戏规则和生成动态描述，不能与已经部署的渲染推理模块混同。推理栈采用轻量 TAEHV、少步蒸馏、FP8、CUDA Graph、环形 KV cache 与异步媒体 IO。

## 开放状态和复现提示

- 模型/游戏资产和推理代码已开放；源码与资产许可分开核对。
- 生成渲染不负责碰撞/游戏逻辑；HTTP/RTSP/WebRTC 示例仅适合可信单用户环境，不能未经鉴权直接暴露公网。源代码 Apache-2.0，模型和游戏资源按各自许可。README 的 `--fps 40` 是采集目标，不保证端到端吞吐。
- 入口与源码分层见 [归档](../repos/matrix-game-4.md)。

## 映射

- [项目/论文详情](../../wiki/entities/paper-matrix-game-4.md)
