# Awesome Touch README 通读（sun254667）

> 来源归档（原文通读 · 支撑 `wiki/entities/awesome-touch.md`）

- **标题：** Awesome Touch（upstream README.md）
- **作者 / 维护方：** Yuanliang Sun（sun254667）
- **类型：** repo / awesome-list
- **原文：** <https://github.com/sun254667/awesome-touch>（2026-11 通过 raw.githubusercontent.com 取得 main 分支 README 全文，527 行）
- **一句话说明：** 2025.01–2026.07 触觉操作精选清单，按 Surveys / Tactile VLA / 视触觉 WM / Tactile WAM / 策略学习 / 仿真与 Sim2Real / 硬件 / 数据集 / 关键挑战 组织，CC0-1.0。
- **核对范围：** 通读 README 全文（纳入标准、目录、全部 9 个分区的条目标题与 Highlights、数据集表、Key Research Challenges、Contact、License）。清单所列各论文本身未逐篇阅读，仅依据 README 的一句话摘要引用。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **纳入标准：** 优先 ICRA / IROS / CoRL / RSS / NeurIPS / ICML / ICLR / CVPR 等顶会顶刊；核心方向为 Tactile VLA、视触觉 WM、Tactile WAM、触觉模仿学习、触觉仿真与 Sim2Real、数据集与基准；时间窗 2025.01–2026.07。
- **结构：** 1 Surveys（3 条）；2 Tactile VLA（2.1 统一 VTLA 框架，如 UniTacVLA、OmniVTLA、TacMamba、ResTacVLA、AT-VLA；2.2 基础模型增强，如 HapticVLA、DreamTacVLA、VLH）；3 视触觉世界模型（VT-WM、ContactWorld、TouchWorld、TacForeSight、ViTacWorld、OmniVTA、ViTacPhys）；4 Tactile WAM（VTAM、Dream-Tac、Tactile-WAM、VT-WAM、N0-TWAM、HiTac-WAM、TacPAC）；5 策略学习（5.1 模仿学习、5.2 强化学习、5.3 扩散策略）；6 仿真与 Sim2Real（TacSL、Taccel、SimTac、UniVTAC、Tac2Real、ETac 等）；7 传感器与硬件（PolyTouch、OSMO、FreeTacMan、ART-Glove 等）；8 数据集与基准表（OmniViTac、ContactWorld、ManiFeel、WorldArena 2.0、UniVTAC、ObjTac、Tactile MNIST、ManiSkill-ViTac 2025）；9 Key Research Challenges。
- **条目格式：** 每条含发表处、Highlights、论文链接；标 🆕 表示新增。
- **TacMamba（2.1）：** 异步架构，触觉编码器（System 1）100Hz 连续流式推理，VLA 规划器（System 2）按需查询压缩触觉历史。
- **OmniVTA（3）：** OmniViTac 数据集 21,000+ 任务、86 物体；两路视触觉 WM + 接触感知融合策略 + 60Hz 反射控制器。
- **Tactile-WAM（4）：** Tactile Asymmetric Attention（TAAM）+ VideoClean mask 防止 tactile pollution。
- **关键挑战（9）：** Tactile Pollution、Modality Collapse、数据稀缺、触觉 Sim2Real 差距、跨模态表征兼容、时空结构、接触中心保真度、实时推理（约 60Hz）、预测与执行的时序错位、传感器泛化。
- **许可（README 末尾）：** CC0 1.0 Universal。

## 对 wiki 的映射

- [`wiki/entities/awesome-touch.md`](../../wiki/entities/awesome-touch.md)
