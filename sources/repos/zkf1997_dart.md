# zkf1997/DART

- **URL（仓库）**: https://github.com/zkf1997/DART
- **URL（项目页）**: https://zkf1997.github.io/DART/
- **维护方**: Kaifeng Zhao 等（ETH Zürich）
- **定位**: **DartControl（ICLR 2025）** 官方代码：自回归潜扩散 **文本→人体运动** 与 **潜空间空间控制**（噪声优化 / RL）实现
- **论文**: arXiv:2410.05260 — 归档见 [`dart_control_arxiv_2410_05260.md`](../papers/dart_control_arxiv_2410_05260.md)

## 仓库要点（维护者速览）

- **核心能力**：运动原语 VAE + 文本/历史条件潜扩散；在线自回归 rollout；latent noise optimization 与 RL-based waypoint control 脚本/模块（以 README 与 `docs/` 为准）。
- **表示**：SMPL-X 系 **276 维/帧** 过参数化原语（与 HumanML3D 263 维不同）；默认 $H{=}2$、$F{=}8$ 帧原语。
- **性能宣称**：单卡 RTX 4090 **>300 FPS** 生成；与 FlowMDM 等离线组合基线相比强调 **实时在线**。
- **生态演示**：项目页展示与 **PHC** 物理跟踪组合、命令行交互 demo、人体–场景 SDF 交互 preliminary 结果。
- **许可**：以仓库 `LICENSE` 为准。

## 对 wiki 的映射

- 方法页：[dart-control](../../wiki/methods/dart-control.md)
- 论文摘录：[dart_control_arxiv_2410_05260.md](../papers/dart_control_arxiv_2410_05260.md)
- 项目页归档：[dart-control-project.md](../sites/dart-control-project.md)
