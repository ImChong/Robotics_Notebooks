# vla.simd（arXiv:2609.24274）

> 来源归档（paper）

- **标题：** vla.simd: Efficient CPU Inference for Language-Conditioned Manipulation
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2609.24274>
- **PDF：** <https://arxiv.org/pdf/2609.24274>
- **项目页：** <https://vla-simd.github.io/>
- **入库日期：** 2026-10-18
- **线索来源：** [机器人会写控制程序之后，能否自己造训练数据？](../blogs/wechat_embodied_station_26_papers_embodiedswe_2026-10-18.md)（具身智能小站，2026-10-18）
- **一句话说明：** 用 SIMD 内核与计算复用加速 CPU 推理，并把查询延迟和动作块长度一起分析。

## 开源状态

- **项目页未列 GitHub**（步骤 2.5 核查，2026-10-18）：项目页 HTTP 200，未见 GitHub 链接或 coming soon 字样。

## 核心摘录（公众号转述，未核验原文表格）

- CPU 部署先算动作供给，再看反馈频率；动作供给率不等于闭环反馈频率。
- SIMD 内核 + 计算复用；联合分析查询延迟与动作块长度。
- 指标：公众号提示：树莓派部署数据需连同热负载和执行方式阅读；未给具体数字。

## 对 wiki 的映射

- [paper-vla-simd](../../wiki/entities/paper-vla-simd.md)
- [编程智能体造数据与 VLA 闭环评测：具身智能小站 10-18 期 26 篇阅读坐标](../../wiki/overview/embodied-station-2026-10-18-26-papers-technology-map.md)
