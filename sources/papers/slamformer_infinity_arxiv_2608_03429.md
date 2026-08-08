# slamformer_infinity_arxiv_2608_03429

> 来源归档（ingest）

- **标题：** SLAMFormer-∞: Infinite SLAM Transformer for Unbounded Frontend and Backend Processing
- **短名：** SLAMFormer-∞ / SLAMFormer-Infinity
- **类型：** paper
- **来源：** arXiv abs / PDF
- **原始链接：**
  - <https://arxiv.org/abs/2608.03429>
  - <https://arxiv.org/pdf/2608.03429>
- **项目页：** <https://tsinghua-mars-lab.github.io/SLAMFormer-Infinity> — 归档见 [`sources/sites/tsinghua-mars-lab-slamformer-infinity.md`](../sites/tsinghua-mars-lab-slamformer-infinity.md)
- **作者：** Zhijian Fang*, Weicheng Zheng*, Yijun Yuan*†, Weibang Wang, Zhuoguang Chen, Chang Sun, Junhao Huang, Kenan Li, Minghui Qin, Hang Zhao†（* 同等贡献；† 通讯）
- **机构：** IIIS, Tsinghua University（清华大学交叉信息研究院）
- **版本：** arXiv:2608.03429v1（Submitted 2026-08-04）
- **入库日期：** 2026-08-08
- **一句话说明：** 面向 **无界长程单目稠密 SLAM** 的几何 Transformer：以 **memory condition** 定义局部坐标系/尺度，前端保局部算力；后端 **PGGO** 在同一 transformer 先验下联合迭代精炼长程位姿与 dense pointmap；KITTI/Waymo 相对 VGGT-Long 降 ATE，并演示 **>17 km** 城市场景。

## 核心摘录

### 1) 问题与动机
- 经典几何 foundation（DUSt3R / MASt3R / VGGT / π³ 等）多绑定 **有界上下文** 与训练可见轨迹尺度；长序列要么靠 **状态管理/拼接**，要么 **只对齐位姿、冻结局部几何**（如 VGGT-Long）。
- [SLAM-Former](https://arxiv.org/abs/2509.16909) 把 frontend/backend 收进单一 transformer，但长程仍受 **训练轨迹分布与增长序列状态** 约束。
- 目标：同一模型同时支撑 **无界前端跟踪** 与 **无界后端联合位姿–几何优化**，去掉显式距离上界。

### 2) 方法要点
1. **Memory-conditioned 推理：** 建模 \(p(\mathcal{X},\mathcal{P}\mid\mathcal{I},\mathcal{I}_C,\mathcal{X}_C)\)，条件块定义参考坐标系与尺度；frontend/backend 均在 **条件局部系** 推理，而非首帧锚定全局系。
2. **Conditional Frontend：** 关键帧 + 有界局部上下文 + 邻域条件；KV cache 来自此前 frontend/backend。
3. **Local Backend：** 每 \(c_w\) 个关键帧，对最近窗口做条件后端精炼，并回写 memory。
4. **PGGO（Pose-Geometry Graph Optimization）：** 节点为位姿∪几何；显式相对位姿边 + transformer 注意力隐式几何相关；迭代 \(\hat{\mathbf{x}}^{k+1}=f_\psi\circ f_\theta(\cdot)\)，pointmap 直接更新，位姿用阻尼 SE(3) 插值（\(\alpha\)）。
5. **测试时节奏：** Streaming frontend → 周期 local backend → 回环/序列末 **Global PGGO**（户外 fine stage 位姿初值与 VGGT-Long 一致的 pose-graph）。
6. **训练：** 自预训练 SLAM-Former 初始化；四模式共享权重（frontend / backend w/o condition / backend w/ condition / fine-stage）；室内 12 帧·长边 518，户外 36 帧·长边 224；**48×A100，10 epochs**。

### 3) 实验（论文报告摘要）
| 基准 | 指标 | VGGT-Long / VGGT-SLAM | SLAMFormer-∞ | 读法 |
|------|------|------------------------|--------------|------|
| KITTI 00–10 | Avg ATE RMSE (m) ↓ | 26.358 | **23.011** | 全序列跟踪；calibration-free dense |
| Waymo 城市场景 | Avg ATE RMSE (m) ↓ | 1.996 | **1.813** | 多速度/车流片段 |
| Waymo pointmap | Acc / Comp / Chamfer ↓ | 1.182 / 2.860 / 2.021 | **0.949 / 2.777 / 1.863** | 相对 LiDAR |
| 7-Scenes | ATE / Acc/Comp/Chamf ↓ | 0.068 / 0.054/0.060/0.057（VGGT-SLAM） | **0.046 / 0.029/0.049/0.039** | 短轨迹仍竞争 |
| Replica（消融） | ATE；fine vs w/o | — | **0.052** vs 0.061 | fine stage 数值边际、定性表面更净 |
| 自采城市场景 | 定性 | VGGT-Long 崩溃 | **一致大尺度地图** | 项目页宣称 **17 km · 45 min** |

- 室内短序列上 **SLAM-Former** 仍常领先（端到端学图连接）；∞ 明确优先 **无界长程全局联合优化**。
- **局限（§5）：** PGGO 依赖 frontend/回环给出的 **预定义图连通**；连通质量未端到端学习。

### 4) 开源核查（步骤 2.5）
- **项目页：** 有 Paper / Demo / Explore / BibTeX；**未列** 可运行训练或推理 Code 按钮。
- **GitHub：** [`Tsinghua-MARS-Lab/SLAMFormer-Infinity`](https://github.com/Tsinghua-MARS-Lab/SLAMFormer-Infinity) — `main` **仅 README**（链到项目页与论文）；`gh-pages` 托管站点与 demo 视频 → **占位仓 / 无可辨识训练·推理入口**。
- **前作对照：** [`Tsinghua-MARS-Lab/SLAM-Former`](https://github.com/Tsinghua-MARS-Lab/SLAM-Former)（ECCV 2026，arXiv:2509.16909）**已开源** demo/训练分支与权重；**不可**等同于 Infinity 已可复现。
- **结论：** **部分开源（项目页 + 占位仓）/ Infinity 推理与训练待发布** → wiki `## 源码运行时序图` 标不适用。

## 对 wiki 的映射

- 升格 [SLAMFormer-∞ 论文实体](../../wiki/entities/paper-slamformer-infinity.md)
- 更新 [导航·SLAM 栈](../../wiki/overview/navigation-slam-autonomy-stack.md)、[状态估计知识链](../../wiki/overview/hub-state-estimation.md)、[State Estimation](../../wiki/concepts/state-estimation.md)、[Glob3R](../../wiki/entities/paper-glob3r.md)、[LingBot-Map](../../wiki/methods/lingbot-map.md)

## 当前提炼状态

- [x] 摘要 + 方法 + KITTI/Waymo/室内表 + 开源边界
- [x] wiki 实体页与交叉引用
- [x] `sources/sites/` + `sources/repos/`（占位仓）
