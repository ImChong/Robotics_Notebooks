# DynaWM: A Base-VLA-Guided World Foundation Model for Moving-Object Manipulation

- **类型：** paper
- **原始链接：** <https://arxiv.org/abs/2607.02604>
- **全文版本：** <https://arxiv.org/pdf/2607.02604v1>
- **核对日期：** 2026-11-13
- **核对范围：** arXiv v1 PDF，全文 §1–§5（14 页正文），图 1–3、表 1–5；无方法附录，参考文献浏览。图表以文字层核对，未观看演示视频，未运行训练或真机实验。
- **通读：** 已完成

## 通读核对

- §3：输入是冻结 VLA 的 10×7 动作块、5 帧×3 视角历史、9D 状态；Mamba-3 编码动作，V-JEPA 2.1 提供历史特征，8 层 DiT 从高斯噪声用 10 步 Euler 重生成，非残差修正。
- §3.4–3.5：V-JEPA 属于可训练 DynaWM；只有 base VLA 冻结。各基础 VLA 分别拟合 DynaWM 权重，跨 VLA 共享的是架构/配方，不是一个通用 checkpoint。
- §4：32 个 Isaac Sim 任务，各 50 次；1600 条仿真遥操作 / 约 510K 步 / 1.53M 图像。原文把硬件验证作为后续，不报告 Franka 真机成功率。
- 表 2：fine-tuned π₀ 总计 14 个任务退步，不是 Amount 类有 14 个任务（Amount 仅 6 任务）。八组均值 50.36%→76.15%。
- 表 4–5：39.94%→67.44% 的视觉消融基于 Qwen 内部融合参照；29% 是「去动作编码器并改为 base-action flow 初始化」的联合改动，不能当作纯粹移除 action conditioning 的单变量消融。

## 对 wiki 的映射

- [DynaWM: A Base-VLA-Guided World Foundation Model for Moving-Object Manipulation](../../wiki/entities/paper-dynawm-vla-online-correction.md)
