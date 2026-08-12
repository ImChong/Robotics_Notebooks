# HIL-HARC（arXiv:2608.09762）

> 来源归档（ingest）

- **标题：** Efficient Real-World Online Reinforcement Learning for Robot Manipulation via Centralized Training and Critic Decomposition
- **缩写：** **HIL-HARC**（Human-in-the-Loop · Hybrid Actors with Reward-decomposed Critic）
- **类型：** paper / online-rl / human-in-the-loop / manipulation / ctde / hybrid-action
- **arXiv：** <https://arxiv.org/abs/2608.09762>
- **HTML：** <https://arxiv.org/html/2608.09762>
- **PDF：** <https://arxiv.org/pdf/2608.09762>
- **项目页：** <https://hil-harc.github.io/> — [`sources/sites/hil-harc-github-io.md`](../sites/hil-harc-github-io.md)
- **代码：** 截至 **2026-08-12** 项目页 **未列训练仓**；GitHub 仅有静态页仓 `HIL-HARC/HIL-HARC.github.io`
- **作者：** Changhao Li\*、Yifang Zhang、Heng Zhang、Davide Torielli、Damiano Gasperini、Arturo Laurenzi、Luca Muratore、Arash Ajoudani、Nikos Tsagarakis
- **机构：** 意大利技术研究院（IIT）HHCM / HRI²；热那亚大学（University of Genova）DIBRIS；代尔夫特理工大学（TU Delft）Cognitive Robotics
- **入库日期：** 2026-08-12
- **一句话说明：** 真机在线 RL：连续臂（SAC）+ 离散夹爪（categorical SAC）在 **CTDE** 下共享集中式多头 critic；**HRA** 把稀疏任务奖励与 potential-based 抓取奖励拆成 task/grasp 头。相对 HIL-SERL，在约 **5–25×** 更大域随机下真机平均成功率 **40%→75%**（160 min），网球 **60%→80%**、香蕉 **60%→90%**、锅复位 **0%→55%**；仿真 G1 搬块 **25%→95%**；收敛后干预率 **0%**。投稿后增补 bottle stowing **85%**（17/20）。

## 核心论文摘录（MVP）

### 1) 问题与总贡献（Abstract / §I）

- **链接：** <https://arxiv.org/abs/2608.09762>
- **核心贡献：** HIL-SERL 等在小随机与混合动作（连续臂 + 离散夹爪）下仍受限：独立训练两策略引入非平稳；单 monolithic critic 在噪声 RGB 上难回归长程回报。HIL-HARC = RLPD 先验数据 + CTDE 混合动作 + HRA 分解 critic。
- **对 wiki 的映射：**
  - [HIL-HARC 实体](../../wiki/entities/paper-hil-harc.md)
  - [Reinforcement Learning](../../wiki/methods/reinforcement-learning.md)
  - [Online vs Offline RL](../../wiki/comparisons/online-vs-offline-rl.md)

### 2) CTDE + HRA 机制（§III）

- **链接：** Methodology
- **核心贡献：**
  - RLPD：50% demo / 50% online；高 UTD；人干预写入 policy buffer。
  - CTDE：连续 Cartesian SAC + 离散 gripper SAC；训练时联合评价值，执行时本地观测。
  - HRA：\(r=r_{\mathrm{task}}+r_{\mathrm{grasp}}\)；grasp 为 \(\gamma\Phi(s')-\Phi(s)+P\)（力矩/开口势 + 切换惩罚）；多头 TD 目标与 actor 加权组合。
  - 异步：机器人上 actors 采集，远端 learner 更新后周期同步参数。
- **对 wiki 的映射：**
  - [HIL-HARC 实体](../../wiki/entities/paper-hil-harc.md) — 流程总览
  - [Safe Real-World RL Fine-tuning](../../wiki/concepts/safe-real-world-rl-fine-tuning.md)

### 3) 大随机真机 / 仿真结果（§IV / 项目页）

- **链接：** Experiments
- **核心贡献：**
  - 随机：网球 \(50\times40\) cm、香蕉 \(30\times30\) cm + \(360^\circ\)、锅 \(40\times40\) cm + \(90^\circ\)；相对 HIL-SERL 常见 \(2\text{–}8\) cm 约 **5–25×**。
  - 160 min 真机预算：平均成功率 **75%** vs 基线 **40%**；干预率降至 **0%**。
  - 专家等价 episode：本文 69/76/111/115 vs HIL-SERL 80/102/132/189。
  - 仿真 Unitree G1 双块搬移 **95%** vs **25%**；投稿后 bottle stowing **85%**（未入正文表）。
- **对 wiki 的映射：**
  - [Manipulation](../../wiki/tasks/manipulation.md)
  - [Sim2Real](../../wiki/concepts/sim2real.md)

### 4) 开源边界（步骤 2.5）

- **链接：** <https://hil-harc.github.io/>
- **核心贡献：** Resources 仅 Citation 占位（「official BibTeX will be added when published」）；**无 Code / HF**。`HIL-HARC.github.io` 为静态 Pages。截至入库日 **确认未开源**。
- **对 wiki 的映射：**
  - [项目页归档](../sites/hil-harc-github-io.md)
  - [HIL-HARC 实体](../../wiki/entities/paper-hil-harc.md) — 源码运行时序图不适用

## 对 wiki 的映射（汇总）

- 实体：[`wiki/entities/paper-hil-harc.md`](../../wiki/entities/paper-hil-harc.md)
- 交叉：[Reinforcement Learning](../../wiki/methods/reinforcement-learning.md)、[Online vs Offline RL](../../wiki/comparisons/online-vs-offline-rl.md)、[Safe Real-World RL Fine-tuning](../../wiki/concepts/safe-real-world-rl-fine-tuning.md)、[Manipulation](../../wiki/tasks/manipulation.md)、[Sim2Real](../../wiki/concepts/sim2real.md)、[ROVE](../../wiki/entities/paper-rove-humanoid-vla-intervention.md)

## BibTeX（arXiv；项目页尚未给官方条）

```bibtex
@article{li2026hilharc,
  title={Efficient Real-World Online Reinforcement Learning for Robot Manipulation via Centralized Training and Critic Decomposition},
  author={Changhao Li and Yifang Zhang and Heng Zhang and Davide Torielli and Damiano Gasperini and Arturo Laurenzi and Luca Muratore and Arash Ajoudani and Nikos Tsagarakis},
  journal={arXiv preprint arXiv:2608.09762},
  year={2026},
  url={https://arxiv.org/abs/2608.09762}
}
```
