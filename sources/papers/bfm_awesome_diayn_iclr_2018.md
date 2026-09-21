# Diversity is All You Need: Learning Skills without a Reward Function

> 来源归档（深读 · ingest · awesome-bfm-papers 第 30/41）

- **标题：** Diversity is All You Need: Learning Skills without a Reward Function
- **作者：** Benjamin Eysenbach, Abhishek Gupta, Julian Ibarz, Sergey Levine
- **类型：** paper / unsupervised-rl / skill-discovery
- **BFM 分类：** 03 Intrinsic reward 预训练（[awesome-bfm-papers](https://github.com/friedrichyuan/awesome-bfm-papers)）
- **出处：** 2018 · ICLR
- **论文链接：** <https://arxiv.org/abs/1802.06070>
- **代码/项目：** 官方 <https://github.com/ben-eysenbach/sac/blob/master/DIAYN.md> · 项目页 <https://sites.google.com/view/diayn/>
- **入库日期：** 2026-05-26
- **深读更新：** 2026-09-21
- **一句话说明：** DIAYN 用 **互信息 + 最大熵策略** 在无任务奖励下发现可区分技能；Ant 等环境涌现走/跳/翻，且 **单技能零-shot 可解 benchmark** 并加速下游微调。

## 核心摘录（面向 wiki 编译）

### 1) 目标与伪奖励

- **要点：** 最大化 $\mathcal{G}=\mathbb{E}[\log q_\phi(z|s')-\log p(z)] + \mathcal{H}[a|s,z]$（互信息 + 动作熵）。实现：discriminator $q_\phi(z|s)$ 与 SAC 策略 $\pi_\theta(a|s,z)$；$r_z(s)=\log q_\phi(z|s)-\log p(z)$。
- **对 wiki 的映射：** [`wiki/entities/paper-bfm-30-diayn.md`](../../wiki/entities/paper-bfm-30-diayn.md)

### 2) 三原则

- **技能决定所访状态**（非动作标签）；**用状态而非动作区分技能**（外力不可见动作）；**高熵探索** 迫使技能占据不同状态分区。
- **对 wiki 的映射：** 同上

### 3) 下游用法

- **Policy init：** 最高奖励技能作 warm-start，加速 task RL。
- **Hierarchy：** 随机/学习 meta-policy 选 $z$，短 horizon 解稀疏奖励。
- **Imitation：** 匹配专家状态分布选 $z$ 或微调。
- **对 wiki 的映射：** [`wiki/concepts/behavior-foundation-model.md`](../../wiki/concepts/behavior-foundation-model.md)

### 4) 实验摘要

- **技能：** 2D navigation 六技能分区；Ant **走/跳/翻/滑** 等无奖励涌现；Mountain Car / Pendulum **多解**。
- **Benchmark：** 多个环境 **未见过 task reward 训练** 却由某技能 **直接高回报**。
- **稳定性：** 相对 adversarial unsupervised RL，DIAYN 为 **合作博弈**；gridworld 有解析最优（均匀分区）。
- **对 wiki 的映射：** [`wiki/entities/paper-bfm-30-diayn.md`](../../wiki/entities/paper-bfm-30-diayn.md)

## 开源边界（步骤 2.5）

| 状态 | 说明 |
|------|------|
| **已开源** | 官方 SAC 仓库 `DIAYN.md` + 项目页视频 |
| **第三方** | [DIAYN-PyTorch](https://github.com/alirezakazemipour/DIAYN-PyTorch) 便于现代栈 |
| **环境** | 经典 MuJoCo/Gym；与 Isaac 人形栈非直接互换 |

## 对 wiki 的映射

- [paper-bfm-30-diayn.md](../../wiki/entities/paper-bfm-30-diayn.md)
- [behavior-foundation-model.md](../../wiki/concepts/behavior-foundation-model.md)
- [diayn_sac.md](../repos/diayn_sac.md)

## 参考来源（原始）

- 论文：<https://arxiv.org/abs/1802.06070>
- 官方代码：<https://github.com/ben-eysenbach/sac/blob/master/DIAYN.md>
- 策展列表：<https://github.com/friedrichyuan/awesome-bfm-papers>
