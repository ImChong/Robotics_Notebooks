# zkf1997.github.io/DART（DartControl 项目页）

- **标题：** DartControl — A Diffusion-Based Autoregressive Motion Model for Real-Time Text-Driven Motion Control
- **类型：** site / project-page
- **URL：** <https://zkf1997.github.io/DART/>
- **配套论文：** [DartControl（arXiv:2410.05260）](https://arxiv.org/abs/2410.05260) — 归档见 [`sources/papers/dart_control_arxiv_2410_05260.md`](../papers/dart_control_arxiv_2410_05260.md)
- **代码：** <https://github.com/zkf1997/DART> — 归档见 [`sources/repos/zkf1997_dart.md`](../repos/zkf1997_dart.md)
- **入库日期：** 2026-06-09

## 一句话摘要

ETH Zürich **DART** 官方站点：展示 **自回归运动原语潜扩散** 在 **在线文本流** 下的实时长序列合成，以及 **潜噪声优化 / RL** 驱动的 **in-between、路点到达、人体–场景交互（SDF + 目标骨盆位置）**；并演示与 **PHC** 物理跟踪组合以减轻运动学 artifact。

## 公开信息要点（截至入库日）

- **方法板块：**
  - **Autoregressive Motion Primitive** — $H$ 历史 + $F$ 未来重叠原语；SMPL-X 276 维/帧（旋转、位置、一阶差分）
  - **DART Architecture** — VAE 编码/解码 + 冻结权重下训练去噪器；在线 text-to-motion rollout 示意与示例视频
  - **Latent space control** — 目标最小化公式；噪声优化算法图；RL 策略控制示意
- **结果板块：**
  - **Text-conditioned temporal composition** — 多段文本驱动长动作
  - **CLI interactive demo** — 命令行实时文本驱动
  - **Latent optimization** — in-between（对比 DNO、OmniControl）；human-scene interaction（walk-turn-sit、上下楼梯等，红球标目标骨盆）
  - **RL waypoint reaching** — walk/run/hop 等文本条件动态路点（黄环目标），**~240 FPS** 宣称
  - **PHC 组合** — 爬行序列：DART 穿模 → PHC 跟踪后足地接触改善
  - **Perpetual rollout** — 单 prompt 分钟级 jog/cartwheel/dance；边界动作（kneel down）在边界态附近波动
  - **Semantic ambiguity** — 粗整句标签导致动作随机跳转；拆分为逐动作 prompt 可恢复顺序
  - **Dense/sparse joint control** — 手腕点/轨迹约束示例
- **引用：** ICLR 2025 BibTeX（`Zhao:DartControl:2025`）

## 对 wiki 的映射

- [dart-control](../../wiki/methods/dart-control.md)
- [phc](../../wiki/entities/phc.md) — 物理后处理组合演示
