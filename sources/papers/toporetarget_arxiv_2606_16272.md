# TopoRetarget: Interaction-Preserving Retargeting for Dexterous Manipulation（arXiv:2606.16272）

> 来源归档（ingest · 全文消化）

- **标题：** TopoRetarget: Interaction-Preserving Retargeting for Dexterous Manipulation
- **类型：** paper / dexterous hand motion retargeting + reference-based RL + sim2real
- **arXiv abs：** <https://arxiv.org/abs/2606.16272>
- **arXiv HTML：** <https://arxiv.org/html/2606.16272v1>
- **PDF：** <https://tsinghua-mars-lab.github.io/toporetarget-web/static/images/paper.pdf>（MARS Lab 项目页镜像）；<https://arxiv.org/pdf/2606.16272>
- **项目页：** <https://tsinghua-mars-lab.github.io/toporetarget-web/>（归档见 [`sources/sites/toporetarget-github-io.md`](../sites/toporetarget-github-io.md)）
- **作者：** Jielin Wu*, Shenzhe Yao*, Guanqi He*‡, Xiaohan Liu*‡, Zhaoqing Zeng, Xiangrui Jiang, Han Yang, Wentao Zhang, Hang Zhao†（* equal；‡ project lead；† corresponding）
- **机构：** IIIS, Tsinghua University
- **硬件：** Wuji Hand（主真机实验）；亦展示 MANO、Leap 等跨 embodiment 增广
- **入库日期：** 2026-06-17
- **一句话说明：** 基于 **稀疏 hand–object interaction graph** 与 **距离加权 Laplacian 形变能** 的交互保留灵巧重定向：骨方向一致性初始化 + 共享拓扑 Delaunay mesh + 穿透软/硬约束；**单组固定参数**覆盖多物体/尺度/手型；下游 **轻量 PPO 参考跟踪**（残差动作 + 4 项 reward + DR）在 ContactPose / Ho-cap / Pen-Spin 上显著优于 OmniRetarget、Mink、DexPilot、GeoRT，并 **零样本** 部署到 Wuji Hand 转笔与魔方重定向。

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| 项目页 | <https://tsinghua-mars-lab.github.io/toporetarget-web/> | 真机视频、基线对比、增广演示 |
| 全身交互近邻 | [OmniRetarget（arXiv:2509.26633）](https://arxiv.org/abs/2509.26633) | 同人族 interaction mesh，面向人形全身 loco-manipulation |
| 物理采样灵巧 | [SPIDER（arXiv:2511.09484）](https://arxiv.org/abs/2511.09484) | 仿真采样 refinement + 虚拟接触引导 |
| 遥操基线 | DexPilot [8]、GeoRT [13] | 手部中心目标 vs 极速几何重定向 |
| 评测数据 | ContactPose [26]、Ho-cap [31] | 接触保真 / 下游 tracking 评测 |
| 下游 RL 先例 | DexTrack、Dexplore、DexMachina、ManipTrans | 参考跟踪范式对照 |

## 摘要级要点

- **痛点：** 人手演示富含接触结构，但 naive 指尖/关节角匹配会破坏 **hand–object 局部交互**；伪影（漏接触、穿透、不可行姿态）会传导到 RL tracking，降低策略成功率。
- **方法：** MediaPipe 21 关键点 + 物体 mesh 表面采样 → Delaunay 四面体 **interaction mesh**；源帧固定拓扑与距离衰减权重，机器人侧匹配 **加权 Laplacian 坐标**；叠加骨方向先验、时序正则与 SDF 穿透 slack 约束。
- **实时性：** ContactPose 评测平均 **4.70 ms/帧**（OmniRetarget 40.96 ms；GeoRT 1.17 ms 但接触误差大）。
- **下游 RL：** 残差动作 $q^{\text{tar}}_t = q^{\text{ref}}_{k_t} + a_t$；观测含本体、物体轴点与 base 系前瞻参考；**4096** 并行环境 PPO；物体质量/COM、接触、执行器、阻尼、惯量 DR + 外力扰动。
- **真机：** 仿真训策略 **零样本** 上 Wuji Hand 完成魔方重定向与转笔（5/5 trial 保持转笔）。

## 核心摘录（面向 wiki 编译）

### 1) 优化管线（Sec. 3）

1. **骨方向初始化（Eq. 1–2）：** 在腕部手坐标系匹配相邻骨段方向差 $E_{\text{bone}}$，加时序 warm-start 平滑。
2. **Interaction mesh（Eq. 3–4）：** $V^s_t = [P^h_t; O_t]$，$V^r_t(q) = [P^r_t(q); O_t]$；源顶点 Delaunay 得边集 $\mathcal{I}_t$，机器人复用同一连通性。
3. **拓扑感知 Laplacian（Eq. 5–8）：** 源帧距离衰减权重 $w_{ij,t}$；最小化 $\| \Delta_t(V^r_t(q))_i - \Delta_t(V^s_t)_i \|^2$ 均值 + $\lambda_{\text{bone}} E_{\text{bone}} + E_{\text{reg}}$，约束 $\phi_i(q) \geq -\tau$（穿透 slack）。

### 2) 重定向质量（Table 1 · ContactPose）

| 方法 | 接触精度 (mm) ↓ | 接触对齐 (°) ↓ | 最大穿透 (mm) ↓ | >2mm 穿透帧 % ↓ | 求解 (ms) ↓ |
|------|----------------|---------------|----------------|----------------|------------|
| **TopoRetarget** | **7.71** | **15.67** | **1.07** | **0.00** | **4.70** |
| OmniRetarget | 14.15 | 30.80 | 1.15 | 0.00 | 40.96 |
| Mink | 14.12 | 37.36 | 20.12 | 84.00 | 4.37 |
| DexPilot | 14.13 | 33.71 | 11.87 | 88.00 | 1.74 |
| GeoRT | 26.77 | 25.74 | 22.22 | 96.00 | 1.17 |

### 3) 下游 RL tracking（Table 2）

| 数据集 | 指标 | TopoRetarget | OmniRetarget | Mink | DexPilot | GeoRT |
|--------|------|-------------|-------------|------|----------|-------|
| Ho-cap (32 clips) | 成功率 ↑ | **84.4%** | 56.2% | 75.0% | 75.0% | 75.0% |
| Ho-cap | 物体位置误差 (cm) ↓ | **0.87** | 1.07 | 0.91 | 0.92 | 0.90 |
| Pen-Spin (32 clips) | 成功率 ↑ | **87.5%** | 46.9% | 21.9% | 40.6% | 31.2% |
| Pen-Spin | 物体位置误差 (cm) ↓ | **0.98** | 1.45 | 1.61 | 1.29 | 1.19 |

Pen-Spin 相对最强基线（OmniRetarget 46.9%）提升 **+40.6 百分点**（项目页与摘要口径）。

### 4) 贡献三元组

1. 交互保留重定向框架（局部 hand–object 交互 + 运动学/穿透约束）。
2. 轻量参考式 RL tracking → contact-rich 灵巧技能 **零样本 sim2real**（转笔、魔方重定向）。
3. 灵巧 hand–object 交互数据集（重定向轨迹、任务参考、已训策略，论文承诺可复现）。

### 5) 局限（Sec. 7）

- 依赖上游人手参考质量；可修正源穿透，但对 **虚拟接触**（手指应接触但未触物体）效果有限，需预处理。

## 对 wiki 的映射

- 新建方法页：[`wiki/methods/toporetarget-interaction-preserving-dexterous-retargeting.md`](../../wiki/methods/toporetarget-interaction-preserving-dexterous-retargeting.md)
- 交叉：[`wiki/concepts/motion-retargeting.md`](../../wiki/concepts/motion-retargeting.md)、[`wiki/tasks/manipulation.md`](../../wiki/tasks/manipulation.md)、[`wiki/entities/wuji-robotics.md`](../../wiki/entities/wuji-robotics.md)、[`wiki/methods/spider-physics-informed-dexterous-retargeting.md`](../../wiki/methods/spider-physics-informed-dexterous-retargeting.md)

## 关联原始资料

- 项目页：[`sources/sites/toporetarget-github-io.md`](../sites/toporetarget-github-io.md)
