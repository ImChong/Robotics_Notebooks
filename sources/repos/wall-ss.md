# X-Square-Robot/wall-ss

> 来源归档

- **标题：** WALL-SS（官方仓库）
- **类型：** repo
- **组织：** X-Square-Robot（自变量机器人）
- **代码：** <https://github.com/X-Square-Robot/wall-ss>
- **arXiv：** <https://arxiv.org/abs/2608.26239>
- **论文 PDF：** <https://github.com/X-Square-Robot/wall-ss/blob/main/wall-ss-paper.pdf>
- **项目页：** <https://wall-ss.ws-robotics.workers.dev/>（原官网：<http://x2robot.com/pages/ss>）
- **许可：** MIT（LICENSE 已在；README 写 *Code will be released under the MIT License*）
- **入库日期：** 2026-08-28
- **复核日期：** 2026-11-21（检出 `X-Square-Robot/wall-ss`，确认 README 新增 arXiv:2608.26239 链接，但 TODO `Release the training and inference code` 仍未勾选，仓内仍无训练/推理源码与权重）
- **核对范围：** 通读 `X-Square-Robot/wall-ss` 仓库全部文件（`README.md`、`LICENSE`、`assets/`、`wall-ss-paper.pdf` 确认）
- **通读：** 已完成
- **一句话说明：** 截至入库日为 **论文 + 项目页占位仓**：无训练/推理脚本、无权重；README 声明四能力（统一 / 可变长度 / 流式 / 可奖励优化）与三项关键配方。

## 入口速查（对齐 README，2026-08-28）

| 路径 / 声明 | 作用 |
|-------------|------|
| `wall-ss-paper.pdf` | 论文全文（约 6.2 MB） |
| `assets/` | 框架图与评测可视化 |
| `LICENSE` | MIT |
| TODO: Release the paper | **已勾选**（2026-08-26） |
| TODO: Release the training and inference code | **未勾选** |

## 开源边界

- **已发布：** 论文 PDF、项目页链接、MIT 许可文件、配图。
- **未发布：** 训练 / 推理代码、tokenizer、权重、数据预处理脚本。
- **复现入口：** 入库日 **无可运行 CLI**；后续应以 README TODO 变更为准，勿把占位仓写成可复现基线。

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [WALL-SS](../../wiki/entities/paper-wall-ss.md) | 实体归纳：next-scale AR WM、60 s 流式、虚实校准 |
| [Generative World Models](../../wiki/methods/generative-world-models.md) | 自回归 next-scale vs 扩散 clip 生成 |
| [Ctrl-World](../../wiki/entities/paper-ctrl-world.md) | 对照：多视角 VLA 闭环；Ctrl-World **已开源** |
| [world-models-route-03-virtual-sandbox](../../wiki/overview/world-models-route-03-virtual-sandbox.md) | 虚拟策略评估沙盒 |

## 对 wiki 的映射

- 论文摘录：[`sources/papers/wall_ss_x_square_2026.md`](../papers/wall_ss_x_square_2026.md)
- 项目页：[`sources/sites/x2robot-wall-ss.md`](../sites/x2robot-wall-ss.md)
- 沉淀 **[`wiki/entities/paper-wall-ss.md`](../../wiki/entities/paper-wall-ss.md)**
