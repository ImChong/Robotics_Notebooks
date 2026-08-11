# Fault-Tolerant Locomotion（arXiv:2608.07328）

> 来源归档（ingest）

- **标题：** Learning Fault-Tolerant Locomotion with Adaptive Gait Timing
- **缩写：** **FTL** / Fault-Tolerant Locomotion（本文语境）
- **类型：** paper / quadruped / locomotion / fault-tolerant / reinforcement-learning / sim2real / actuator-power-loss
- **arXiv：** <https://arxiv.org/abs/2608.07328>
- **HTML：** <https://arxiv.org/html/2608.07328>
- **PDF：** <https://arxiv.org/pdf/2608.07328>
- **项目页：** <https://gianni0907.github.io/fault_tolerant_locomotion/> — 归档见 [`sources/sites/fault-tolerant-locomotion-github-io.md`](../sites/fault-tolerant-locomotion-github-io.md)
- **代码：** 截至 **2026-08-11** 项目页与论文 **未列 GitHub / 训练仓**
- **演示视频：** <https://youtu.be/x4paP49SKuY>
- **作者：** Giovanbattista Gravina、Luca Rossini、Carlo Rizzardo、Arturo Laurenzi、Nikos Tsagarakis
- **机构：** 意大利技术研究院（Istituto Italiano di Tecnologia, IIT）Humanoids and Human-Centered Mechatronics；EU Horizon 2020 EuROBIN（No.101070596）
- **入库日期：** 2026-08-11
- **一句话说明：** 面向 **68 kg KYON 四足** 的 **执行器功率损失（power loss）容错步态**：非对称 actor–critic + **latent-alignment**，动作为关节位置偏差 + **可学习步态频率**，仿真崎岖地形与真机平地零样本验证。

## 核心论文摘录（MVP）

### 1) 问题与总贡献（Abstract / Introduction）

- **链接：** <https://arxiv.org/abs/2608.07328>
- **核心贡献：** 大质量四足在功率损失下难靠小平台常见的高频率激进补偿。本文用 **单阶段 PPO + 非对称 actor–critic**：critic 见故障掩码等特权信息，actor 仅用本体历史重建 latent；辅以 **可学习 gait frequency**，无需按故障腿预定义策略。
- **对 wiki 的映射：**
  - [Fault-Tolerant Locomotion 实体](../../wiki/entities/paper-fault-tolerant-locomotion.md)
  - [Locomotion](../../wiki/tasks/locomotion.md)
  - [Sim2Real](../../wiki/concepts/sim2real.md)

### 2) 非对称 actor–critic 与 latent-alignment（§III）

- **链接：** Method
- **核心贡献：**
  - Critic encoder：\(o^p=\langle o,e\rangle\)（含关节故障 mask \(m_J\)）→ latent \(r\)；Actor encoder：历史 \(h_t=\langle o_t,\ldots,o_{t-H+1}\rangle\) → \(\hat r\)。
  - 总损失 \(\mathcal{L}=\mathcal{L}^{\mathrm{PPO}}+\lambda_3\mathcal{L}^{\mathrm{MSE}}(\hat r,r)\)。
  - 动作 \(a=\langle a^q,a^\nu\rangle\)：关节相对默认姿态偏差 + 标量步态频率；PD 跟踪；相位 \(\phi\) 由 \(\nu^{\mathrm{ref}}\) 推进并进入 feet-phase 奖励。
  - 故障：扭矩效率 \(k_\tau\in[0,1]^{n_J}\) 缩放；课程从部分失效递进到完全功率损失；故障腿不计入 feet-phase 奖励。
- **对 wiki 的映射：**
  - [Fault-Tolerant Locomotion 实体](../../wiki/entities/paper-fault-tolerant-locomotion.md) — 流程总览
  - [Reinforcement Learning](../../wiki/methods/reinforcement-learning.md)
  - [Teacher–Student / DAgger 训练](../../wiki/methods/teacher-student-dagger-training.md) — 对照：本文走单阶段非对称而非两阶段蒸馏

### 3) 评测与消融（§IV）

- **链接：** Experiments
- **核心贡献：**
  - 变体：Oracle / w/o latent alignment / w/o history / Ours（\(H=3,\lambda_3=1\)）；五次独立训练；故障于 \(t=5\mathrm{s}\) 随机关节。
  - 指标：故障下线/角速度跟踪误差与生存时间（Fig. 5）；膝关节故障通常最难。
  - Sim-to-Sim：MuJoCo + **XBot2** 1 kHz 异步；未见楼梯（10 cm / 0.7 m）与 13° 坡；膝故障常切 **三足**，髋故障可继续利用残余自由度。
  - Sim-to-Real：平地零样本；演示后左膝 pitch 功率损失下行走。
  - 消融：历史 \(H=1\to 2\) 收益最大，更长边际小，取 \(H=3\)；可学习频率相对 free-gait 更周期、更平滑（加速度与动作变化更低）。
- **对 wiki 的映射：**
  - [RL 运动控制纵深](../../roadmap/depth-rl-locomotion.md)
  - [执行器约束 RL 高速四足](../../wiki/entities/paper-actuator-constrained-rl-high-speed-quadruped-locomotion.md) — 同为重型四足执行器边界，问题轴不同（工作区 vs 故障）

### 4) 开源边界（步骤 2.5）

- **链接：** <https://gianni0907.github.io/fault_tolerant_locomotion/>
- **核心贡献：** 项目页含 Abstract / Architecture / MuJoCo 演示 / 真机片段与 YouTube；**无 Code / GitHub / 权重** 入口。截至入库日按 **确认未开源** 处理。
- **对 wiki 的映射：**
  - [项目页归档](../sites/fault-tolerant-locomotion-github-io.md)
  - [Fault-Tolerant Locomotion 实体](../../wiki/entities/paper-fault-tolerant-locomotion.md) — 源码运行时序图不适用

## 对 wiki 的映射（汇总）

- 沉淀实体页：[`wiki/entities/paper-fault-tolerant-locomotion.md`](../../wiki/entities/paper-fault-tolerant-locomotion.md)
- 项目页归档：[`sources/sites/fault-tolerant-locomotion-github-io.md`](../sites/fault-tolerant-locomotion-github-io.md)
- 互链参考：[Locomotion](../../wiki/tasks/locomotion.md)、[Sim2Real](../../wiki/concepts/sim2real.md)、[Reinforcement Learning](../../wiki/methods/reinforcement-learning.md)、[RL 运动控制纵深](../../roadmap/depth-rl-locomotion.md)、[执行器约束 RL 高速四足](../../wiki/entities/paper-actuator-constrained-rl-high-speed-quadruped-locomotion.md)

## BibTeX（arXiv）

```bibtex
@misc{gravina2026faulttolerant,
  title={Learning Fault-Tolerant Locomotion with Adaptive Gait Timing},
  author={Gravina, Giovanbattista and Rossini, Luca and Rizzardo, Carlo and Laurenzi, Arturo and Tsagarakis, Nikos},
  year={2026},
  eprint={2608.07328},
  archivePrefix={arXiv},
  primaryClass={cs.RO}
}
```
