# HOTICE: Whole-Body Humanoid Object Transportation in Cluttered Environments（arXiv:2609.25363）

> 来源归档（ingest）

- **标题：** HOTICE: Whole-Body Humanoid Object Transportation in Cluttered Environments
- **类型：** paper / humanoid / loco-manipulation / rl / cluttered-navigation
- **arXiv abs：** <https://arxiv.org/abs/2609.25363>
- **PDF：** <https://arxiv.org/pdf/2609.25363>
- **项目页：** <https://hotice2027.github.io> — 归档见 [`sources/sites/hotice2027-github-io.md`](../sites/hotice2027-github-io.md)
- **代码：** **未列链接** — 项目页 Anonymous Authors，无 GitHub/HF（2026-09-24）
- **机构：** 南加州大学（University of Southern California, USC）— Toan Nguyen、Weiduo Yuan、Siheng Zhao、Yue Wang、Daniel Seita
- **入库日期：** 2026-09-24
- **一句话说明：** HOD-PF 解耦人形/物体势场 + dual-agent RL 全身 loco-manipulation + specialist→generalist 蒸馏；MuJoCo SR 88.5%、未见场景 avg 80.1%；G1 真机 side 6/6、overhead/ground 5/6。

## 核心摘录（面向 wiki 编译）

### 1) 三组件

1. **HOD-PF（Humanoid-Object Decoupled Potential Fields）** — SGF 引导 head/pelvis/feet；OGF（膨胀障碍 margin 8 cm）引导物体 keypoints + 双手，保持 robot–object 同步避障。
2. **Dual-Agent RL** — 上身 / 下身独立 actor-critic + 共享观测与 whole-body reward，应对高维全身动作空间。
3. **Specialist→Generalist** — 75 场景 privileged teacher → DAgger + RL fine-tune 单一 deployable student。

### 2) headline 数字（仿真 20 场景 × 20k trials）

| 方法 | SR | mDist |
|------|-----|-------|
| Single-Agent & SGF | 58.7% | 0.296 m |
| Dual-Agent & HOD-PF **(Ours)** | **88.5%** | **0.130 m** |

- **未见 50 场景 generalist：** avg SR **80.1%**（floor ~70%），avg mDist **0.27 m**。
- **形状泛化（cuboid/cylinder/sphere）：** SR **88.5 / 86.7 / 87.2%**。

### 3) 真机（G1，每类障碍 6 trials）

| 方法 | Side | Overhead | Ground |
|------|------|----------|--------|
| Dual-Agent & HOD-PF **(Ours)** | **6/6** | **5/6** | **5/6** |

- SLAM 重建场景 + AprilTag 估计 box pose；50 Hz；含 pickup policy 完整 pipeline。

### 4) 开源状态（2026-09-24）

- 项目页 demo/方法图公开；**无** 官方代码仓库链接 → **待发布 / 未列链接**。

## 对 wiki 的映射

- 新建：[paper-hotice](../../wiki/entities/paper-hotice.md)
- 交叉：[loco-manipulation](../../wiki/tasks/loco-manipulation.md)、[whole-body-control](../../wiki/concepts/whole-body-control.md)、[paper-tango-vla](../../wiki/entities/paper-tango-vla.md)、[sim2real](../../wiki/concepts/sim2real.md)
