# Holo-M: Humanoid Loco-Manipulation With Discrete VLA Model（arXiv:2609.35709）

> 来源归档（ingest）

- **标题：** Humanoid Loco-Manipulation With Discrete VLA Model
- **类型：** paper / vla / humanoid / discrete-action / loco-manipulation
- **arXiv abs：** <https://arxiv.org/abs/2609.35709>
- **PDF：** <https://arxiv.org/pdf/2609.35709>
- **项目页：** <https://horizonrobotics.github.io/gail/Holo-M/> — 归档见 [`sources/sites/holo-m-horizon-github-io.md`](../sites/holo-m-horizon-github-io.md)
- **代码：** **待发布** — 论文与项目页写明将发布 **全部代码与权重**；截至 2026-09-30 项目页无 GitHub / HF 按钮
- **机构：** 地平线机器人（Horizon Robotics）— Wenxin Shao、Siqi Chai、Kun Li 等
- **入库日期：** 2026-09-30
- **一句话说明：** 首个 **离散 token VLA** 人形 loco-manipulation：**四部件 tokenizer**（EEF/body/hand/kinematics，共 208 token）扩展 VLM 词表；**分组离散 diffusion** 解码（部件内并行、部件间自回归）；SIMPLE **163/180 specialist、143/180 generalist** SOTA。

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| 项目页 | <https://horizonrobotics.github.io/gail/Holo-M/> | SIMPLE 全表、真机 demo、延迟表 |
| 基准 | SIMPLE | 6 任务 × 3 随机化 × 10 ep = 180 |
| 低层 | Ψ0 同款 decoupled WBC | 隔离高层策略对比 |
| 对照 | Ψ0、π0.5、GR00T、WholeBodyVLA 系 | 项目页 specialist/generalist 表 |

## 摘要级要点

- **动机：** 臂级离散 VLA 难直接扩到 **96-D 异构全身**；连续 action expert 有 **knowledge insulation**。
- **Tokenizer：** EEF 48D→100 tok、Body 29→62、Hand 14→32、Kinematics 5→14；缺失部件 **mask loss** 以混源训练（遥操作 / ego 人视频 / 仿真）。
- **解码：** Grouped discrete diffusion — **32 步** demask vs 208 步自回归；500 ms 观测周期、1 s action chunk；G1 + RTX 5090 上 **4-step 276 ms**。
- **训练：** 四阶段 progressive（跨具身预训练 → 任务 specialist）。

## 核心摘录（面向 wiki 编译）

### 1) SIMPLE Specialist overall（/180）

| Method | Overall |
|--------|---------|
| **Holo-M** | **163** |
| Holo-M AR | 157 |
| Ψ0 | 154 |
| ACT | 149 |
| π0.5 | 92 |

### 2) SIMPLE Generalist overall

| Method | Overall |
|--------|---------|
| **Holo-M** | **143** |
| Holo-M AR | 138 |
| Ψ0 | 114 |

### 3) 部署

- Unitree G1-comp、Dex3-1 双手、640×360@30Hz head cam
- 推理：**2/4/8** diffusion steps → **176 / 276 / 477 ms** per 1 s chunk

## 对 wiki 的映射

- 新建：[paper-holo-m](../../wiki/entities/paper-holo-m.md)
- 交叉：[vla](../../wiki/methods/vla.md)、[loco-manipulation](../../wiki/tasks/loco-manipulation.md)、[paper-loco-manip-161-075-simple](../../wiki/entities/paper-loco-manip-161-075-simple.md)、[paper-loco-manip-161-156-psi0](../../wiki/entities/paper-loco-manip-161-156-psi0.md)

## 当前提炼状态

- [x] arXiv + 项目页核查（代码待发布）
- [ ] lint 跟进：仓库链接出现后补 `sources/repos/`
