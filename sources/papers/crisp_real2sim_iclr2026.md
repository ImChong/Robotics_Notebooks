# crisp_real2sim_iclr2026

> 来源归档（ingest）

- **标题：** Contact-guided Real2Sim from Monocular Video with Planar Scene Primitives（CRISP）
- **类型：** paper
- **会议：** ICLR 2026（Poster）
- **OpenReview：** <https://openreview.net/forum?id=xlr3NqxUqY>
- **项目页：** <https://crisp-real2sim.github.io/CRISP-Real2Sim/>
- **代码：** <https://github.com/crisp-real2sim/CRISP-Real2Sim>
- **入库日期：** 2026-05-17
- **一句话说明：** 从单目 RGB 视频恢复**可物理仿真**的人形运动与场景几何：深度点云上拟合**仿真就绪的凸平面原语**，用人–场景接触补全遮挡几何，再用 RL 驱动人形控制器做物理一致性闭环，显著降低运动跟踪失败率并提高仿真吞吐。

## 核心论文摘录（MVP）

### 1) 问题设定与总贡献（Abstract）

- **链接：** <https://openreview.net/forum?id=xlr3NqxUqY>
- **核心贡献：** 现有「人–场景联合重建」要么强依赖数据驱动先验、联合优化但**无物理在环**，要么几何噪声大、伪影多，导致后续**带场景交互的运动跟踪策略**易失败。CRISP 主张把场景拟合成**可放进物理引擎的凸平面片集合**，并把人与场景一起用于 RL 人形控制，使重建结果在**接触与动力学**上更可用。
- **对 wiki 的映射：**
  - [CRISP（Contact-guided Real2Sim）](../../wiki/methods/crisp-real2sim.md)
  - [Sim2Real](../../wiki/concepts/sim2real.md)

### 2) 场景几何：平面拟合与聚类管线（站点 Method + OpenReview 叙述）

- **链接：** <https://crisp-real2sim.github.io/CRISP-Real2Sim/>（Method 区 pipeline 图）、<https://openreview.net/forum?id=xlr3NqxUqY>
- **核心贡献：** 在基于深度的点云重建上，通过 **depth / normals / optical flow** 上的简单聚类流程，将场景近似为**仿真就绪的凸平面原语**（论文 Fig.2 强调平面拟合细节），降低「稠密但不可仿真」网格带来的接触不稳定性。
- **对 wiki 的映射：**
  - [CRISP](../../wiki/methods/crisp-real2sim.md)

### 3) 接触引导与遮挡补全 + RL 闭环（Abstract / Keywords）

- **链接：** <https://openreview.net/forum?id=xlr3NqxUqY>
- **核心贡献：** 用 **人–场景接触建模**（例如由人体姿态推断被遮挡的座椅支撑面）补全交互中不可见结构；再用重建的人与场景驱动 **RL 人形控制器**，把「几何 + 运动」一起拉向**物理可信**区域。论文报告在人-centric 视频基准（**EMDB、PROX**）上运动跟踪失败率从 **55.2% 降至 6.9%**，RL 仿真吞吐提高约 **43%**（相对其对比设置，见 OpenReview 摘要数字）。
- **对 wiki 的映射：**
  - [Whole-Body Control](../../wiki/concepts/whole-body-control.md)
  - [Reinforcement Learning](../../wiki/methods/reinforcement-learning.md)

## BibTeX（项目页提供）

```bibtex
@inproceedings{
wang2026contactguided,
title={Contact-guided Real2Sim from Monocular Video with Planar Scene Primitives},
author={Zihan Wang and Jiashun Wang and Jeff Tan and Yiwen Zhao and Jessica K. Hodgins and Shubham Tulsiani and Deva Ramanan},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=xlr3NqxUqY}
}
```

## 当前提炼状态

- [x] 摘要与站点 Method 区对齐
- [x] wiki 页面映射确认
- [x] 站点 / OpenReview 交叉索引见 `sources/sites/crisp-real2sim-project-github-io.md`
