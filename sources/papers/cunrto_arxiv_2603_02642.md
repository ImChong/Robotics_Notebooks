# cuNRTO: GPU-Accelerated Nonlinear Robust Trajectory Optimization

> 来源归档

- **标题：** cuNRTO: GPU-Accelerated Nonlinear Robust Trajectory Optimization
- **类型：** paper
- **出处：** 2026 · RSS 2026 Finalist · arXiv preprint
- **arXiv：** <https://arxiv.org/abs/2603.02642>
- **论文 HTML：** <https://arxiv.org/html/2603.02642>
- **项目页：** <https://cunrto.github.io/>
- **代码：** 截至 2026-07-20，项目页**未列出 GitHub 链接**，代码尚未公开发布
- **作者：** Jiawei Wang, Arshiya Taj Abdul; Evangelos A. Theodorou（Georgia Tech）
- **入库日期：** 2026-07-20
- **一句话说明：** CUDA 并行化 NRTO：Douglas-Rachford SOCP 投影 + 反馈增益联合优化全量迁移 GPU；规划时间从数小时压缩至秒级；不确定性感知轨迹规划服务 sim2real 鲁棒性；RSS 2026 Finalist。

---

## 核心摘录（策展，非全文）

### 问题与动机

- 非线性鲁棒轨迹优化（NRTO）能生成对不确定性鲁棒的轨迹，但传统 CPU 实现规模大、代价高（数小时级别），无法支持实时规划或快速迭代开发。
- 核心问题：如何将 NRTO 的核心计算（SOCP 投影 + 反馈增益优化）重构为大规模 GPU 并行，突破计算瓶颈。

### 关键贡献

1. **CUDA NRTO 框架（cuNRTO）：** 将 Douglas-Rachford SOCP 投影算子与反馈增益优化全量实现为 CUDA kernels。
2. **规划加速：** 秒级 vs 数小时级，使 NRTO 进入实时可用范围。
3. **不确定性感知规划：** 将 sim2real gap 建模为不确定性集合，生成在真机上鲁棒的轨迹。

### 方法要点

| 维度 | cuNRTO |
|------|--------|
| 优化框架 | NRTO（最坏情况鲁棒，SOCP 约束） |
| 核心算子 | Douglas-Rachford 分裂 → SOCP 投影并行化 |
| 反馈增益 | 联合优化 K；多候选并行评估 |
| 硬件 | CUDA GPU（单卡） |
| 加速比 | 数小时 → 秒级（具体比值见论文 Table） |
| 应用 | 飞行/操作轨迹规划、sim2real 鲁棒性 |

### 实验摘要

- 与传统 CPU NRTO 对比：相同规划质量下，cuNRTO 秒级完成。
- 演示：飞行器轨迹规划、机器人手臂任务；鲁棒性 vs 名义轨迹 baseline 验证。
- 不确定性集合涵盖动力学参数误差、外部扰动等 sim2real gap 建模场景。

### 代码状态

- 项目页：<https://cunrto.github.io/>，截至核查日未列出 GitHub 链接，代码未公开。

### 局限（论文自述）

- SOCP 凸松弛在强非线性下存在近似误差。
- GPU 内存限制对超长 horizon 系统。

### 对 wiki 的映射

- [paper-cunrto-gpu-robust-trajectory-optimization](../../wiki/entities/paper-cunrto-gpu-robust-trajectory-optimization.md)
- [trajectory-optimization](../../wiki/methods/trajectory-optimization.md)
- [sim2real](../../wiki/concepts/sim2real.md)

## 参考来源（原始）

- 论文：<https://arxiv.org/abs/2603.02642>
- 项目页：<https://cunrto.github.io/>
