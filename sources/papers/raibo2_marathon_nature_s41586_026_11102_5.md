# A quadruped robot designed to complete a marathon on a single battery charge（Nature 2026）

> 来源归档（ingest）

- **标题：** A quadruped robot designed to complete a marathon on a single battery charge
- **平台名：** **RAIBO2**（KAIST RaiLab 高能效四足）
- **类型：** paper / quadruped / locomotion / energy-efficiency / hardware-software co-design / reinforcement-learning
- **期刊：** *Nature*（2026-09-23 在线发表）
- **DOI：** <https://doi.org/10.1038/s41586-026-11102-5>
- **Nature 页：** <https://www.nature.com/articles/s41586-026-11102-5>
- **PDF：** <https://www.nature.com/articles/s41586-026-11102-5.pdf>
- **机构：** KAIST Robotics and Artificial Intelligence Laboratory（RaiLab）；通讯作者 Jemin Hwangbo
- **资助：** Samsung Research Funding and Incubation Center（SRFC-IT2002-02）
- **数据：** Zenodo <https://doi.org/10.5281/zenodo.14825866> — 归档见 [`sources/sites/zenodo_raibo2_marathon_dataset.md`](../sites/zenodo_raibo2_marathon_dataset.md)
- **代码：** <https://github.com/railabatkaist/raisimGym_nature> — 归档见 [`sources/repos/raisimGym_nature.md`](../repos/raisimGym_nature.md)
- **实机演示：** [YouTube — RAIBO2 全马 42.195 km / 4:19:52](https://youtu.be/-HMFoa3g9nA)（2024，RaiLab KAIST）
- **入库日期：** 2026-09-25
- **一句话说明：** 通过 **整机能量损耗模型** 协同设计 **力透传轻量化腿足、低损耗电机驱动与低耗散 RL 步态**，**RAIBO2** 在 **单次充电（约 1,447 Wh）** 完成 **全程马拉松（4 h 19 min 52 s）**，**CoT 0.25** 优于人类参考 **0.37**，续航约为现有四足的 **3 倍以上**。

## 开源状态（Nature Code/Data availability + GitHub，2026-09-25）

| 组件 | 状态 |
|------|------|
| 论文图表与马拉松评测 **数据集** | ✅ Zenodo [10.5281/zenodo.14825866](https://doi.org/10.5281/zenodo.14825866) |
| **Locomotion 策略训练** + **奖励消融评测** 代码 | ✅ [railabatkaist/raisimGym_nature](https://github.com/railabatkaist/raisimGym_nature)（MIT）；依赖 sibling **`raisimLib`** |
| RAIBO2 **整机 CAD / 驱动固件 / 部署栈** | ❌ 未随 Nature 仓发布 |
| **硬件复现** | 需自研或等待后续发布；论文侧重 **设计原则 + 损耗分解 + 策略奖励项** |

→ wiki 归类：**部分开源**（仿真训练与数据可复现能效策略与消融；不可一键复现马拉松硬件）。

## 摘要级要点

- **痛点：** 四足相对轮式 **关节持重 + 间歇触地动能损失** 导致 **单次充电航程短**；能效优化涉及 **机械、电气、控制多物理耦合**，既往工作多 **单点** 减损（齿隙、弹性腿、能耗奖励等）。
- **方法：** 建立 **RAIBO2 总损耗模型**（电机/驱动器/机械/控制相关项），硬件侧 **力透传轻量化结构 + 低阻 MOSFET/电流采样/门驱优化**，软件侧 **PPO 类 RL** + **碰撞/触地速度/Joule 损耗/身体高度** 等 **能效导向奖励** 与地形课程。
- **结果：** **42.195 km** 马拉松 **4:19:52**；**CoT 0.25**；相对文献中四足 **航程 >3×**；补充材料含与动物、EV、乘用车等的 **CoT 对照表**（Supplementary Data 1）。
- **团队脉络：** 与 Hwangbo 组 **RaiSim / raisimGym**、[并发策略–估计器](../papers/concurrent_policy_estimator_locomotion_arxiv_2202_05481.md)、SciRobo 地形行走等同源。

## 核心论文摘录（MVP）

### 1) 整机能量损耗模型与三管齐下减损

- **链接：** Abstract；Fig. 2–3
- **摘录要点：** 不孤立优化某一损耗项；仿真 sensitivity（Extended Data Fig. 3）与实测对齐 **驱动器 ablation**（Hall vs shunt、门阻、开关频率等）。
- **对 wiki 的映射：**
  - [RAIBO2 马拉松实体页](../../wiki/entities/paper-raibo2-marathon-energy-efficient-quadruped.md)
  - [Locomotion 任务页](../../wiki/tasks/locomotion.md) — 能效作为户外部署硬指标

### 2) 低耗散 locomotion policy（RL + 奖励设计）

- **链接：** Extended Data Fig. 4–5；Supplementary Video 2–3
- **摘录要点：** **碰撞奖励** 降低触地机械损耗；**Joule 损耗奖励** 重塑膝力矩分布；**身体高度** 影响电气损耗；跑步机 **4 m/s** 对照波形验证。
- **对 wiki 的映射：**
  - [RAIBO2 实体页](../../wiki/entities/paper-raibo2-marathon-energy-efficient-quadruped.md)
  - [Sim2Real](../../wiki/concepts/sim2real.md) — RaiSim 训练 → RAIBO2 实机马拉松

### 3) 马拉松级系统验证与 CoT 对标

- **链接：** Fig. 1, 4；Ref. 7 YouTube
- **摘录要点：** 真实马拉松赛道 **单次电池** 完赛；CoT **0.25** vs 人类 **0.37**（Tucker 等经典参照）；强调 **户外长时任务** 而非短距冲刺。
- **对 wiki 的映射：**
  - [四足机器人实体](../../wiki/entities/quadruped-robot.md)
  - [Walk These Ways / Learning to Adapt](../../wiki/entities/paper-walk-these-ways-quadruped-mob.md) — 能力轴不同（敏捷/多步态 vs **续航/CoT**）

## BibTeX

```bibtex
@article{lee2026quadrupedmarathon,
  title={A quadruped robot designed to complete a marathon on a single battery charge},
  author={Lee, Choongin and Youm, Donghoon and Park, Jeongsoo and others and Hwangbo, Jemin},
  journal={Nature},
  year={2026},
  doi={10.1038/s41586-026-11102-5},
  url={https://doi.org/10.1038/s41586-026-11102-5}
}
```

## 对 wiki 的映射

- 主实体页：[`wiki/entities/paper-raibo2-marathon-energy-efficient-quadruped.md`](../../wiki/entities/paper-raibo2-marathon-energy-efficient-quadruped.md)
- 代码：[`sources/repos/raisimGym_nature.md`](../repos/raisimGym_nature.md)
- 数据：[`sources/sites/zenodo_raibo2_marathon_dataset.md`](../sites/zenodo_raibo2_marathon_dataset.md)
- 互链：[四足机器人](../../wiki/entities/quadruped-robot.md)、[RSS 2018 敏捷四足 sim2real（Hwangbo）](../../wiki/entities/paper-quadruped-agile-sim2real-rss2018.md)、[并发策略–估计器](../../wiki/entities/paper-concurrent-policy-estimator-locomotion.md)
