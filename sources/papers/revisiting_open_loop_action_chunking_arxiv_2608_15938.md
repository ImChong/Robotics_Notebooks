# Revisiting Open-Loop Execution in Robotics（arXiv:2608.15938）

> 来源归档（ingest）

- **标题：** Revisiting Open-Loop Execution in Robotics: Toward Reactive, Higher-Performing Policies
- **类型：** paper / imitation-learning / action-chunking / analysis
- **arXiv：** <https://arxiv.org/abs/2608.15938>
- **项目页：** <https://revisiting-open-loop-action-chunking.github.io/>
- **机构：** 麻省理工学院（MIT）；加州大学伯克利分校（UC Berkeley）
- **入库日期：** 2026-08-22
- **一句话说明：** 系统实验表明：长 **open-loop execution horizon** 主因是 **短上下文策略模仿非马尔可夫专家**；足够长的观测上下文下 **闭环 reactive 策略最优**；提出 **double encoder** 稳定长上下文 Diffusion Policy 训练。

## 开源状态（步骤 2.5，2026-08-22）

| 资源 | 状态 |
|------|------|
| 项目页 / arXiv | **已发布** |
| 完整策略代码 | **未列主仓链接** |
| Section 4 自动化 Markov/非 Markov 专家策略 | 论文称 **publicly release**（以作者后续发布为准） |

**结论：** **部分/待跟进** — 机制论文以 arXiv + 项目页为准；主实验代码未在项目页挂链。

## 核心论文摘录

### 1) 核心论断（Abstract）

- **核心贡献：** Action chunking 的 **open-loop 执行前缀**（execution horizon \(T_{\mathrm{exec}}\)）降低反应性。本文认为：在 **短 context**（常见 \(T_o=1\)–\(2\)）下，长 \(T_{\mathrm{exec}}\) 主要补偿 **专家演示的非马尔可夫性**（隐式记忆、暂停、模式承诺），而非单独由复合误差或推理延迟解释；**复合误差有影响，但弱于非马尔可夫性**。
- **对 wiki 的映射：**
  - [Revisiting Open-Loop 论文实体](../../wiki/entities/paper-revisiting-open-loop-action-chunking.md)
  - [Action Chunking](../../wiki/methods/action-chunking.md)
  - [Why Action Chunking Improves BC](../../wiki/entities/paper-why-action-chunking-improves-bc.md)（并发机制对照）

### 2) 实验设计（§3–4）

- **变量：** 固定 prediction horizon \(T_p\)，扫 \(T_{\mathrm{exec}}\) 与 \(T_o\)（context length）；success-horizon 曲线。
- **任务：** FurnitureSimOneLeg、GearInsertion、PushT-D、Kitchen（仿真）；SinglePillDispense、SlipIntoBaggie（双 ARX-5 真机）。
- **干预：** HG-DAgger 测试复合误差假说；Markov vs 非 Markov 专家对照。
- **对 wiki 的映射：**
  - [Diffusion Policy](../../wiki/methods/diffusion-policy.md)
  - [Behavior Cloning](../../wiki/methods/behavior-cloning.md)

### 3) 长上下文与 double encoder（§5 / Appendix F）

- **发现：** 增大 \(T_o\)（如 8–20）逐步消除长 \(T_{\mathrm{exec}}\) 收益；数据充足时长上下文 **reactive** 策略优于短上下文长执行。
- **方法：** **Double encoder** — 短程 \(E_S\) 与长程 \(E_L\) 分离最近帧与历史；短程 dropout 防忽略长程特征。
- **对 wiki 的映射：**
  - [LIBERO benchmark](../../wiki/entities/libero-benchmark.md)（对照生态）

### 4) 与 Why AC Improves BC 的分界

- **Revisiting（本文）：** 聚焦 **execution horizon** vs **context length**；主张足够上下文下 **不必长 open-loop 执行**。
- **Why AC（CoRL 2026）：** 聚焦 **chunk 训练目标** vs **Delay/RDE 部署**；强调隐式集成与延迟条件化。
- **对 wiki 的映射：**
  - 两文互补写入 [action-chunking](../../wiki/methods/action-chunking.md) 机制节
