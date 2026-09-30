# Echo in the Steps: Learning Perceptive Humanoid Parkour with Gated Memory（arXiv:2609.28960）

> 来源归档（ingest）

- **标题：** Echo in the Steps: Learning Perceptive Humanoid Parkour with Gated Memory
- **类型：** paper / humanoid / perceptive-locomotion / parkour / reinforcement-learning / sim2real
- **arXiv abs：** <https://arxiv.org/abs/2609.28960>
- **PDF：** <https://arxiv.org/pdf/2609.28960>
- **项目页：** <https://echo-in-the-steps.github.io/> — 归档见 [`sources/sites/echo-in-the-steps-github-io.md`](../sites/echo-in-the-steps-github-io.md)
- **代码：** **待发布** — 项目页 Code 为 Coming soon（2026-09-30）
- **机构：** 清华大学 — Ming-Ju Lee、Zizhuo Wang、Shaoting Zhu、Haozhe Lou、Hang Zhao、Yiming Li
- **会议：** CoRL 2026
- **入库日期：** 2026-09-30（深读升格，原 2026-09-26 清单占位）
- **一句话说明：** 机载深度 PPO 人形跑酷：**垂直深度差显著性先验** 引导 **门控记忆** 跨帧保留踏点线索；**交替损失** 抑制同步双足跳；Isaac Lab 2048 并行 G1，相对 Hiking in the Wild 平均 SR **+16.7 pt**。

## 核心摘录（面向 wiki 编译）

### 1) 仿真主结果 vs Hiking in the Wild [45]（Table 1 均值）

| 指标 | Hiking | Echo（Ours） |
|------|--------|----------------|
| Success Rate | — | **92.7%** 均值（五地形） |
| 相对提升 | baseline | **+16.7 pt SR**，**+7.8 pt FA** |

单地形 SR 例：Boxes **98.1%**、Trapezoids **97.1%**、Stakes **88.1%**。

### 2) 消融（Table 2，五地形均值）

| 配置 | SR | FA |
|------|-----|-----|
| 均匀平均 latent | 52.12 | 70.80 |
| + learnable weight | 40.65 | 82.65 |
| + saliency prior | 88.14 | 90.63 |
| + gated memory | **98.27** | **91.10** |

### 3) 对称正则（Table 3，box）

- w/o sym：**85.0%** SR → w/ sym：**96.7%**；交替损失使 double-support ratio **0.806→0.308**

### 4) 真机

- **G1 + Jetson Orin NX**；D435i 深度 **480×270→64×36→32×18** crop；**50 Hz** 推理
- 每地形 **10 trials**（项目页 Repeated-Trial 视频）

### 5) 感知模块

- 显著性：\(R_t(u,v)=\frac{v}{H}|D_t(u,v)-D_t(u,v-1)|\)；标量 prior \(r_t\) 为 map 均值
- 历史 **4 帧** 深度 + 本体 **8 帧**；门控残差融合 \(\tilde z = z^c + \sigma(\beta)\sum w_t(z^h_t-z^c)\)

## 对 wiki 的映射

- 升格：[paper-echo-in-the-steps](../../wiki/entities/paper-echo-in-the-steps.md)
- 交叉：[paper-hrl-stack-22-perceptive_humanoid_parkour](../../wiki/entities/paper-hrl-stack-22-perceptive_humanoid_parkour.md)、[stair-obstacle-perceptive-locomotion](../../wiki/tasks/stair-obstacle-perceptive-locomotion.md)、[unitree-g1](../../wiki/entities/unitree-g1.md)

## 当前提炼状态

- [x] arXiv + 项目页深读（2026-09-30）
- [x] 开源：代码 **待发布**
- [ ] 代码发布后补 `sources/repos/` 与时序图
