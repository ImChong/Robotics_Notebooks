# R³ — 官方项目页

> 来源归档（ingest · 步骤 2.5 · 2026-09-09）

- **标题：** R³: 3D Reconstruction via Relative Regression
- **类型：** site（作者项目页）
- **发布方：** University of Michigan × Westlake University × NVIDIA
- **原始链接：** <https://kevinxu02.github.io/r3-site/>
- **论文：** <https://arxiv.org/abs/2605.26519>
- **代码：** <https://github.com/KevinXu02/R3>
- **权重：** <https://huggingface.co/KevinXu02/R3>
- **入库日期：** 2026-09-09
- **一句话说明：** 在线流式重建演示（输入流 → 渲染 flythrough + 点云）；372M 参数、40 FPS* 流式、单 checkpoint 双模式；与 InfiniteVGGT / TTT3R 定性对比；置信门控防污染。

## 步骤 2.5 开源核查（2026-09-09）

| 项 | 项目页 / 关联链接 |
|----|-------------------|
| **代码** | **已开源** — 页眉链 GitHub |
| **权重** | **已发布** — HF checkpoints |
| **论文** | arXiv [2605.26519](https://arxiv.org/abs/2605.26519) |
| **评测代码** | GitHub README 标 **TODO** |

## 主页摘录

### 关键数字

- **372M** 参数（约 1B 级前馈基线 ⅓）
- **40 FPS\*** 流式（\*RTX PRO 6000）
- **1 checkpoint** 支持 causal streaming + full-context offline

### 机制三步

1. **Predict relative poses** — 成对相机运动回归，视频变长目标稳定
2. **Fuse with confidence** — 旋转/平移置信加权轨迹装配
3. **Stream with memory** — 有界 keyframe bank 回连历史

### 鲁棒性

置信度兼作 **outlier gate**：新帧平均置信低于基线 → 抑制位姿、作废 KV、跳过 keyframe 入库。

## 对 wiki 的映射

- 论文实体：[`wiki/entities/paper-r3-relative-regression.md`](../../wiki/entities/paper-r3-relative-regression.md)
- 代码归档：[`sources/repos/kevinxu02_r3.md`](../repos/kevinxu02_r3.md)
- 论文摘录：[`sources/papers/r3_arxiv_2605_26519.md`](../papers/r3_arxiv_2605_26519.md)
