# iCrowdNav（arXiv:2606.26047）

> 来源归档（ingest）

- **标题：** Learning Robot Visual Navigation in Crowds via Intention-Aware Scene Representations
- **缩写：** **iCrowdNav**
- **类型：** paper / visual-crowd-navigation / deep-rl / intention-aware / bev / sim2real
- **arXiv：** <https://arxiv.org/abs/2606.26047>
- **期刊：** IEEE Robotics and Automation Letters（RA-L），2026，Vol. 11, No. 5, pp. 6186–6193；doi:[10.1109/LRA.2026.3677748](https://doi.org/10.1109/LRA.2026.3677748)
- **项目页：** <https://broln7.github.io/socialbev.io/>
- **代码：** <https://github.com/BRoln7/icrowdnav>（截至 2026-07-28：**TODO Release codes**，无可运行入口）
- **视频：** <https://www.youtube.com/watch?v=8q0dhAiWCEA>
- **作者：** Han Bao†, Bingyi Xia†, Hanjing Ye, Yu Zhan, Hao Cheng, Baozhi Jia, Wenjun Xu, Jiankun Wang（† equal contribution）
- **机构：** 南方科技大学（SUSTech）；锐冠信息（Reconova）；鹏城实验室（Peng Cheng Laboratory）
- **入库日期：** 2026-07-28
- **一句话说明：** 面向 **密集人群视觉导航** 的 DRL 方法：用 **时空 BEV 编码器** 提取占据特征，用 **I²Former（Intent-Interact Former）** 从 **3D 人体姿态** 推断行人意图，融合机器人状态后以 **PPO** 训练；Isaac Sim **SocNav-Gym** 上优于 DRL-VO / SARL\*-OM / ViNT / DWA，并在健身房 / 地铁站 / 商场 **零样本** 真机部署（Clearpath Dingo，双 RealSense，板载 RTX 2060 ~15 Hz）。

## 核心论文摘录（MVP）

### 1) 问题与总贡献（Abstract / Introduction）

- **链接：** <https://arxiv.org/abs/2606.26047>；项目页 <https://broln7.github.io/socialbev.io/>
- **核心贡献：** 既有人群导航 DRL 多用 **2D 行人点 / 占据栅格 / 单线激光** 作状态，丢掉姿态、注视与环境语义等视觉线索。**iCrowdNav** 从 **第一人称 RGB-D** 学习 **intention-aware scene representations**：
  1. **Spatio-temporal encoder**（FIERY 风格 lift→BEV + 时序 3D CNN，nuScenes 预训练后冻结）提取场景占据；
  2. **I²Former**：YOLO 检测 2D 姿态 → 深度抬升为 17 关节 3D → IntentFormer（关节 MHSA）+ InteractFormer（机器人状态 query 的 MHCA）；
  3. 与机器人状态 MLP 融合后进 **PPO**；奖励刻意保持简单（到达 / 碰撞 / 私人空间惩罚 / 进度 / 角速度平滑），依赖表征而非精细手调社交奖励。
- **对 wiki 的映射：**
  - [iCrowdNav 论文实体](../../wiki/entities/paper-icrowdnav.md)
  - [DWA](../../wiki/methods/dwa.md)（经典局部避障基线）
  - [PPO](../../wiki/methods/ppo.md)
  - [Sim2Real](../../wiki/concepts/sim2real.md)

### 2) I²Former 与 BEV 融合（§III-B）

- **链接：** arXiv HTML Methodology
- **核心贡献：**
  - BEV：多相机 ResNet-18 特征 + 深度 lift，\(H_b=120,W_b=200\)；历史帧按 ego-motion 对齐后 3D 卷积聚合，再 2D CNN 得 \(\mathbf{z}^t_{\mathrm{bev}}\)。
  - 姿态：最多 \(N_p\) 行人，\(\boldsymbol{\Theta}^t\in\mathbb{R}^{N_p\times17\times3}\)，遮挡关节零填充；Transformer 注意力自然偏向可见关节。
  - InteractFormer 以机器人状态 embedding 为 query，跨注意力汇聚行人意图，再与 BEV、目标方向/距离/速度拼接。
- **对 wiki 的映射：**
  - [iCrowdNav 论文实体](../../wiki/entities/paper-icrowdnav.md) — 流程总览
  - [SRU](../../wiki/entities/paper-sru-spatially-enhanced-recurrent-memory.md) — 同为视觉 RL 导航，但 SRU 侧重长程空间记忆而非社交意图

### 3) SocNav-Gym 与评测协议（§IV）

- **链接：** Simulation Experiments
- **核心贡献：** Clearpath Dingo（1.0 m/s）；双 RealSense D435，合成 FoV ~140°；行人 **Social Force Model** + Isaac Sim 动画。测试按场景宽度与密度分层：办公大厅 7.0 m、医院走廊 4.0 m、仓库 2.5 m；密度 low 0.1 / high 0.2 ped/m²。指标：**SR / NT / PL / TPZ**（私人空间 <0.8 m 时间）。长程任务接 **广义 Voronoi 拓扑图**，路径 >20 m。
- **消融：** w/o I² → SR 下降、TPZ 上升；w/o BEV（改 CNN 编码占据图）→ 灵活性与私人空间合规变差。仅 BEV 版已可比肩需完整行人状态 + LiDAR 的 **DRL-VO**。
- **对 wiki 的映射：**
  - [移动机器人导航规划方法对比](../../wiki/comparisons/mobile-robot-navigation-planning-methods.md)（DWA 对照）
  - [导航纵深路线 Stage 3](../../roadmap/depth-navigation.md)

### 4) 真机零样本（§V）

- **链接：** Real-World Validation；项目页 Real-World Experiment
- **核心贡献：** 健身房（非合作阻挡）、地铁站（密集窄通道）、商场（**109.49 m** 长程，均速 **0.76 m/s**）；板载 **RTX 2060** 推理 **~15 Hz**。作者强调严重遮挡与有限 FoV 仍是超密场景瓶颈。
- **对 wiki 的映射：**
  - [Sim2Real](../../wiki/concepts/sim2real.md)
  - [社会导航笔记实体](../../wiki/entities/paper-notebook-learning-social-navigation-from-positive-and-neg.md)（另一社交导航路线：正负示范 + 规则）

## 对 wiki 的映射（汇总）

- 沉淀实体页：[`wiki/entities/paper-icrowdnav.md`](../../wiki/entities/paper-icrowdnav.md)
- 仓库归档：[`sources/repos/icrowdnav.md`](../repos/icrowdnav.md)
- 项目页：[`sources/sites/broln7-socialbev-io.md`](../sites/broln7-socialbev-io.md)
- 互链参考：[DWA](../../wiki/methods/dwa.md)、[PPO](../../wiki/methods/ppo.md)、[Sim2Real](../../wiki/concepts/sim2real.md)、[SRU](../../wiki/entities/paper-sru-spatially-enhanced-recurrent-memory.md)、[NavWAM](../../wiki/entities/paper-navwam-goal-conditioned-visual-navigation-wam.md)、[社会导航笔记](../../wiki/entities/paper-notebook-learning-social-navigation-from-positive-and-neg.md)、[导航·SLAM 栈总览](../../wiki/overview/navigation-slam-autonomy-stack.md)、[导航纵深](../../roadmap/depth-navigation.md)

## BibTeX（项目页 / 仓库）

```bibtex
@ARTICLE{11456337,
  author={Bao, Han and Xia, Bingyi and Ye, Hanjing and Zhan, Yu and Cheng, Hao and Jia, Baozhi and Xu, Wenjun and Wang, Jiankun},
  journal={IEEE Robotics and Automation Letters},
  title={Learning Robot Visual Navigation in Crowds via Intention-Aware Scene Representations},
  year={2026},
  volume={11},
  number={5},
  pages={6186-6193},
  doi={10.1109/LRA.2026.3677748}
}
```
