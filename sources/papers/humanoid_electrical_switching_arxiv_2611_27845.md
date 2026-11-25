# Humanoid robots for safety-critical electrical switching: A perception-informed runtime constraint filter

- **类型：** paper
- **作者：** Songyang Liu、Yu Hou、Yuqing Hu、Shuai Li
- **机构：** 佛罗里达大学、北卡罗来纳大学夏洛特分校、宾夕法尼亚州立大学
- **版本及日期：** arXiv:2611.27845v1，2026-11-20 提交；预印本
- **原文：** [arXiv](https://arxiv.org/abs/2611.27845) · [HTML（含原图）](https://arxiv.org/html/2611.27845v1) · [PDF](https://arxiv.org/pdf/2611.27845v1)
- **入库日期：** 2026-11-25
- **核对范围：** PDF 正文 §§1–6，重点交叉核对 §3 的场景建模、调度、越界恢复和仿真扩展，§4–5 及表 3–5 的分母、实测间距、保持事件、实际控制时延；另核对声明与数据可用性。
- **通读：** 已完成
- **开放状态：** 原文及 arXiv 未列本项目的官方代码或公开复现实验包，数据声明为按请求提供。论文引用的 XRoboToolkit 和求解器不等于作者滤波器开源。

## 核心摘录

1. 头部 RGB-D 建模在试验前完成，用户指定目标，运行时用固定几何、编码器和最近下发的参考姿态过滤关节指令。滤波不替代目标确认。
2. 真机在断电实验架上进行，固定目标 10/10，全部尝试 10/12；12 次均未观察到误触或误动作。仿真另用三指手与仿真真值，9/9 对 5/9 只属于配对仿真实验。
3. 最长约束保持 61.38 s，论文将其归因于几何检查拒绝建约束而非 QP 求解慢；编码器模型间距最小 49.04 mm，不能把 50 mm 参考模型边界读成物理保证。

- **对 wiki 的映射：** [人形机器人电气开关遥操作安全滤波](../../wiki/entities/paper-humanoid-electrical-switching-runtime-filter.md)
