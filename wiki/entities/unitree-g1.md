---
type: entity
tags: [hardware, humanoid, platform, unitree]
status: complete
updated: 2026-06-05
related:
  - ./humanoid-robot.md
  - ./unitree.md
  - ./unitree-ros.md
  - ./botlab-motioncanvas.md
  - ../roadmaps/humanoid-control-roadmap.md
sources:
  - ../../sources/papers/humanoid_hardware.md
summary: "Unitree G1 是一款由宇树科技推出的入门级教育科研用人形机器人，以其极高的性价比、高集成度以及对仿真学习框架的良好支持而备受关注。"
---

# Unitree G1 (人形机器人)

**Unitree G1** 是宇树科技 (Unitree) 在 H1 之后推出的一款量产型、高性价比的人形机器人平台。其设计初衷是降低人形机器人研究的门槛，使其能够大规模进入实验室、高校和家庭场景。

## 核心特性

1. **高集成度与便携性**：G1 的体型较 H1 更小，支持折叠收纳，单人即可搬运和部署。
2. **丰富的感知方案**：集成了 3D 激光雷达 (LiDAR) 和深度相机，原生支持 [地形自适应](../concepts/terrain-adaptation.md)。
3. **力控能力**：全关节支持高带宽力控，极其适配 [WBC](../concepts/whole-body-control.md) 与强化学习。
4. **生态支持**：完美适配 [Isaac Lab](../entities/nvidia-omniverse.md)、[robot_lab](../entities/robot-lab.md) 和 `legged_gym`；同时作为 NVIDIA [MotionBricks](../methods/motionbricks.md) 生成式运动框架的首批验证硬件。
5. **高效数据生成**：支持 [CLAW](../methods/claw.md) 等合成数据管线，通过网页交互快速生成带语言标签的全身动作数据。
6. **足球技能研究**：作为 [PAiD](../methods/paid-framework.md) 框架的主要实验平台，证明了其在执行类人化踢球动作方面的卓越物理特性。
7. **敏捷技能切换**：[Switch](../methods/switch-framework.md) 框架在 G1 上实现了 100% 的跨技能切换成功率与极强的抗扰动能力。
8. **视频生成驱动的零样本控制**：作为 [ExoActor](../methods/exoactor.md)（BAAI, 2026）的端到端验证平台，把第三人称视频生成 + 通用动作跟踪整合为零真实数据的交互行为生成系统。
9. **浏览器侧策略–仿真编排（生态周边）**：地瓜机器人 [BotLab / MotionCanvas](./botlab-motioncanvas.md) 在网页中提供面向 G1 / Go2 的 ONNX + MuJoCo 节点图实验台，便于对照训练侧 obs 堆叠语义与推理同步策略。
10. **无机器人全身示范部署**：[BifrostUMI](./paper-bifrost-umi.md)（BAAI Aether, 2026）在 G1 上验证 Pico + 双腕夹爪采集数据经扩散策略与 SKR 的杂乱桌面与桌下全身操作。
11. **统一走跑起身（SD-AMP）**：[SD-AMP](./paper-unified-walk-run-recovery-sdamp.md)（HKU, arXiv:2605.18611）在 G1 真机用三条 LAFAN1 参考 + 双 AMP 判别器实现 recovery→walk→run 无部署模式切换。
12. **感知跑酷（PHP）**：[Perceptive Humanoid Parkour](./paper-hrl-stack-22-perceptive_humanoid_parkour.md)（arXiv:2602.15827，RSS 2026）在 G1 上仅用机载深度与 2D 速度指令完成 1.25 m 攀墙与长程多障碍跑酷。
13. **显式楼梯几何爬梯**：[Explicit Stair Geometry Conditioning](./paper-explicit-stair-geometry-humanoid-locomotion.md)（arXiv:2605.09944）在 G1 上零样本部署 BEV 点云 → 楼梯几何 token → PPO，户外连续 33 级上楼。
14. **多姿态起身（HoST）**：[HoST](./paper-host-humanoid-standingup.md)（arXiv:2502.08378，RSS 2025 系统论文 finalist）在 G1 上从零 RL 学习跨地面/平台/墙/坡及俯仰卧、室内外场景的起身，官方 [InternRobotics/HoST](https://github.com/InternRobotics/HoST) 开源。
15. **感知 loco-manipulation LLC（PILOT）**：[PILOT](./paper-pilot-perceptive-loco-manipulation.md)（arXiv:2601.17440，上海交大）在 G1 上用 LiDAR 高程图 + MoE 单阶段全身策略完成楼梯/高台等非结构化 **边走边操作**（VR 遥操作 + 分层 RL）。

## 在具身智能中的作用

G1 的出现极大地加速了大规模数据的采集。由于其成本低廉，研究者可以构建“机器人机房（Robot Farms）”，利用海量实体机器人通过 [自动化标注](../methods/auto-labeling-pipelines.md) 或利用 [CLAW](../methods/claw.md) 等仿真合成手段快速生成训练基础策略（Foundation Policies）所需的真实数据。

## 关联页面
- [smp](../methods/smp.md) (基于得分匹配的运动先验，已在 G1 完成验证)
- [人形机器人 (Humanoid Robot)](./humanoid-robot.md)
- [Unitree 品牌主页](./unitree.md)
- [robot_lab (IsaacLab 扩展框架)](./robot-lab.md)
- [PAiD Framework (足球技能学习)](../methods/paid-framework.md)
- [Humanoid Soccer (足球任务)](../tasks/humanoid-soccer.md)
- [CLAW (宇树 G1 全身动作数据生成管线)](../methods/claw.md)
- [LEGS（论文实体）](./paper-legs-embodied-gaussian-splatting-vla.md) — G1 上 3DGS 合成 loco-manip VLA 数据（arXiv:2606.01458）
- [Humanoid Control Roadmap](../roadmaps/humanoid-control-roadmap.md)
- [ExoActor](../methods/exoactor.md) — G1 上的视频生成驱动的零样本交互控制系统。
- [BifrostUMI（论文实体）](./paper-bifrost-umi.md) — 无机器人示范 + SKR + WBC 的全身 visuomotor 管线。
- [BotLab / MotionCanvas](./botlab-motioncanvas.md) — 浏览器内 G1 相关策略与 MuJoCo 可视化编排入口。
- [显式楼梯几何条件化（论文实体）](./paper-explicit-stair-geometry-humanoid-locomotion.md) — G1 楼梯几何 token + PPO（arXiv:2605.09944）。
- [PILOT（论文实体）](./paper-pilot-perceptive-loco-manipulation.md) — G1 感知统一 loco-manipulation 低层控制器（arXiv:2601.17440）。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| G1 | Unitree G1 Humanoid | 宇树入门级教育科研人形平台 |
| WBC | Whole-Body Control | 全关节力控，适配全身协调控制 |
| RL | Reinforcement Learning | 常见在 Isaac Lab / legged_gym 训练 |
| LiDAR | Light Detection and Ranging | 机载 3D 激光，支撑地形感知 |
| SD-AMP | Selective Domain AMP | 走跑起身统一策略的代表工作线 |
| PILOT | Perceptive Loco-Manipulation | 感知移动操作 LLC 在 G1 上的验证 |

## 参考来源
- Unitree G1 官方规格书。
- [sources/papers/exoactor.md](../../sources/papers/exoactor.md) — ExoActor 在 G1 上的端到端实现。
- [sources/papers/bfm_humanoid_arxiv_2509_13780.md](../../sources/papers/bfm_humanoid_arxiv_2509_13780.md) — BFM：在 G1 上做 motion tracking / VR 遥操作 / locomotion 的统一条件生成策略。
- [sources/papers/bifrost_umi_arxiv_2605_03452.md](../../sources/papers/bifrost_umi_arxiv_2605_03452.md) — BifrostUMI：G1 真机全身 loco-manipulation（arXiv:2605.03452）。
- [sources/papers/php_parkour_arxiv_2602_15827.md](../../sources/papers/php_parkour_arxiv_2602_15827.md) — PHP：G1 感知跑酷（arXiv:2602.15827）。
- [sources/papers/pilot_arxiv_2601_17440.md](../../sources/papers/pilot_arxiv_2601_17440.md) — PILOT：G1 感知 loco-manipulation LLC（arXiv:2601.17440）。
