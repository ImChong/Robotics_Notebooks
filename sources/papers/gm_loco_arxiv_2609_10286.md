# GM-Loco（arXiv:2609.10286）

> 来源归档（深度 ingest）

- **标题：** GM-Loco: Terrain-Adaptive Humanoid Locomotion on Granular Media
- **曾用 arXiv 标题：** Learning Terrain-Adaptive Humanoid Locomotion on Granular Terrain（v1）
- **类型：** paper / humanoid-locomotion / granular-terrain / reinforcement-learning
- **arXiv：** <https://arxiv.org/abs/2609.10286>（v2，2026-09-14 更新标题）
- **PDF：** <https://arxiv.org/pdf/2609.10286>
- **项目页：** <https://humanoid-gm-locomotion.github.io/HUMANOID-GM/>（见 [humanoid-gm-loco.md](../sites/humanoid-gm-loco.md)）
- **机构：** 佐治亚理工学院 IRIM + School of Physics；东北大学
- **作者：** Junnosuke Kamohara、Feiyang Wu、Andy Ningan Zong、Daniel I. Goldman、Yashwanth Nakka、Seth Hutchinson、Ye Zhao
- **平台：** Unitree G1（29 DoF 关节位置残差动作）
- **开源：** **待发布** — 项目页「Code (Coming Soon)」；截至 2026-09-21 无官方仓库
- **入库日期：** 2026-09-21（深度 ingest 覆盖 2026-09-14 浅层公众号映射）

## 核心论文摘录

### 1) 3D RFT 颗粒接触求解器（无启发式切向力）

- 基于 Agarwal et al. 3D resistive force theory，将足端网格点侵入深度、速度角、姿态映射为反力/力矩，再汇总为 whole-body dynamics 接触 wrench；对比 Heuristic-RFT、Cone-RFT、Rigid，在 Newton MPM 参考下水平位移、沉陷与 GRF 最接近 MPM。
- **对 wiki 的映射：** [../../wiki/entities/paper-gm-loco.md](../../wiki/entities/paper-gm-loco.md) — 「核心原理 / 3D RFT」

### 2) Isaac Lab + Warp 大规模 RL 训练管线

- 4096 并行环境、单卡 RTX 4090；RL 50 Hz、PD+物理 200 Hz；软颗粒接触经 NVIDIA Warp 实现；课程学习：先硬地后软地、命令速度线性 ramp。
- **对 wiki 的映射：** [../../wiki/entities/paper-gm-loco.md](../../wiki/entities/paper-gm-loco.md) — 「工程实践」

### 3) Teacher-Student + VAE 地形刚度估计

- Teacher 非对称 actor-critic + MLP VAE encoder/decoder，重建归一化介质缩放系数 η=ξ/ξ_max；Student TCN encoder 从本体历史推断 ẑ，DAgger+PPO 蒸馏动作与 latent/reconstruction 对齐。
- **对 wiki 的映射：** [../../wiki/entities/paper-gm-loco.md](../../wiki/entities/paper-gm-loco.md) — 「方法 / 地形表示」

### 4) MPM 仿真与真机零样本验证

- Newton MPM 评测 basalt / sand / poppy seed：Ours（3D-RFT+TS）SR 100% 且 ev/eθ 优于 PPO-Rigid；真机 basalt、排球场沙、海滩沙走跑至 2.5 m/s；地形过渡时摆腿 clearance 随 η 自适应（仿真 9→16 cm）。
- **对 wiki 的映射：** [../../wiki/entities/paper-gm-loco.md](../../wiki/entities/paper-gm-loco.md) — 「实验与评测 / 结论」

### 5) 动态跳跃与合规踝关节控制

- 3D-RFT 训练的策略可在模拟颗粒面上完成跳跃（Rigid 起飞前滑移失败）；真机 dry sand 1.5 m/s 行走时 Ours 踝 pitch 力矩峰值低于 Rigid/3D-RFT baseline，更少「跺脚扬沙」。
- **对 wiki 的映射：** [../../wiki/entities/paper-gm-loco.md](../../wiki/entities/paper-gm-loco.md) — 「与其他工作对比」

## 步骤 2.5 开源核查（2026-09-21）

| 核查项 | 结论 |
|--------|------|
| 项目页 Code 按钮 | 「Code (Coming Soon)」，无 GitHub URL |
| arXiv PDF / 摘要 | 仅链项目页，无 code/data 链接 |
| 可运行实现 | **无** — 待官方发布后再建 `sources/repos/` |

## 当前提炼状态

- [x] 项目页核查与 `sources/sites/` 归档
- [x] 深度 wiki 实体页 `paper-gm-loco.md`
- [ ] 官方代码发布后补 `sources/repos/` 与「源码运行时序图」
