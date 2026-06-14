# From AGI to ASI

> 来源归档（ingest）

- **标题：** From AGI to ASI
- **类型：** paper（技术报告 / 立场综述）
- **venue：** arXiv preprint
- **原始链接：**
  - arXiv：<https://arxiv.org/abs/2606.12683>
- **机构：** Google DeepMind（Tim Genewein, Matija Franklin, Alexander Lerchner, Laurent Orseau, Samuel Albanie, Adam Bales, Cole Wyeth, Stephanie Chan, Iason Gabriel, Joel Z. Leibo, Allan Dafoe, Marcus Hutter, Thore Graepel, Shane Legg 等）
- **入库日期：** 2026-06-14
- **一句话说明：** DeepMind 技术报告：在 **后 AGI** 假设下刻画 **ASI** 与 **Universal AI (AIXI)** 上界，归纳 **四条并行技术路径**（算力/数据/模型 scaling、范式演进、递归自改进、多智能体集体涌现）及 **六类瓶颈**（数据墙、经济资源、神经范式不足、研究变难、抽象壁垒、刻意减速）；对 **有效算力增长**、**测试时 scaling**、**仿真/交互数据** 与 **具身概念发现** 给出可检验研究议程。

## 核心论文摘录（MVP）

### 1) AGI / ASI / UAI 的定性刻画与数字智能优势

- **链接：** <https://arxiv.org/abs/2606.12683> §3–4
- **摘录要点：**
  - **AGI**：约 **中位人类** 在多数认知任务上的通用智能（Morris et al.「Competent AGI」量级）。
  - **ASI**：在 **几乎所有人类关心领域** 上显著超越 AGI；单实例或 **百万级并行副本集体** 均可构成 ASI；门槛设为超越 **数万专家、十年协作** 的大型人类集体（非单点窄域超人类如 AlphaGo）。
  - **UAI**：Legg–Hutter 分数上的理论极限，由 **AIXI** 形式化；不可计算，只能自下而上近似。
  - **数字智能相对生物的六项 scaling 优势**（Table 1）：I/O 带宽、内部思考速度、工作记忆、基质无关、无损复制、经验高带宽共享——均随 **有效算力** 放大。
  - **ASI 非全知全能**：受物理、实时性、物质操控、可观测性、复杂度理论、哥德尔/停机限制（Table 2）。
- **对 wiki 的映射：**
  - [paper-from-agi-to-asi](../../wiki/entities/paper-from-agi-to-asi.md) — 术语框架与机器人读者导读

### 2) 有效算力增长与「量化 scaling 是否够」

- **链接：** <https://arxiv.org/abs/2606.12683> §2, §5.1, §6
- **摘录要点：**
  - 过去十年 **有效算力** 约 **10×/年**（硬件 ~1.5×、投资 ~2.5×、算法效率 ~3× 复合；Epoch AI 保守估计）。
  - 若 AGI 单模型能力平台期但有效算力仍指数增长：**并行实例数** 与 **思考速度** 可在数年内从千级扩到亿级——集体能力可能构成 ASI，即使单实例仍处人类中位。
  - **Scaling 路径核心不确定性**：算力→能力 是平滑还是尖峰涌现？ diminishing returns 何时主导？ **测试时 scaling**（CoT、采样、搜索）headroom 有限时，多实例集体可能是主要增益通道。
  - 与 **bitter lesson** 对照：朴素暴力搜索不够，**先验/归纳偏置** 决定搜索效率；强偏置会封顶单模型泛化，需定性创新突破。
- **对 wiki 的映射：**
  - [paper-from-agi-to-asi](../../wiki/entities/paper-from-agi-to-asi.md) — 与具身 scaling 的对照节
  - [embodied-scaling-laws](../../wiki/concepts/embodied-scaling-laws.md) — 交叉引用宏观算力叙事

### 3) 四条 AGI→ASI 技术路径（可并行）

- **链接：** <https://arxiv.org/abs/2606.12683> §5, Table 3
- **摘录要点：**
  1. **Scaling 算力/模型/数据**：延续 power-law / Chinchilla 共缩放；数据墙可能靠合成、仿真、RL 交互、test-time 蒸馏（AlphaZero 式）缓解。
  2. **算法范式演进/跃迁**：当前范式 = 大 Transformer 预训练 + 多阶段微调 + 测试时 scaling + 检索/工具 + agent 脚手架；演进方向包括 **无限上下文/工作记忆**、**持续学习**、**交互环境 RL**、**内部世界模型**（Dreamer、MuZero、扩散决策等）；真范式跃迁（神经形态、RL 预训练等）难预测。
  3. **递归（自）改进**：代码/硬件/数据/分工四类 RSI；弱形式已现（NAS、超参搜索、AI 辅助芯片设计、FunSearch/AlphaEvolve）；全自主闭环动力学不确定（可能双曲增长或快速衰减）。
  4. **多智能体协调与集体智能**：Group Agent、虚拟 agent 经济、认知分工；**多智能体 scaling law** 为开放问题。
- **对 wiki 的映射：**
  - [paper-from-agi-to-asi](../../wiki/entities/paper-from-agi-to-asi.md) — Mermaid 四路径总览
  - [world-action-models](../../wiki/concepts/world-action-models.md)、[generative-world-models](../../wiki/methods/generative-world-models.md) — 世界模型作为范式演进实例
  - [data-flywheel](../../wiki/concepts/data-flywheel.md) — 递归数据改进与飞轮

### 4) 六类瓶颈与具身相关「抽象壁垒」

- **链接：** <https://arxiv.org/abs/2606.12683> §5.5, Table 4
- **摘录要点：**
  - **数据墙**：高质量文本预训练数据本十年可能耗尽；对抗：合成数据、高保真仿真、agent 交互、自博弈；具身侧强调 **仿真与 RL 交互** 可随算力扩展采集。
  - **经济/自然资源**：能源、芯片互联、数据中心选址；与 AI 经济回报反馈耦合。
  - **神经范式不足**：纯 log-loss 预训练 + 脚手架可能不够；候选硬问题：幻觉、prompt 注入、风险敏感决策、第三人称数据的因果不足、**抽象壁垒**。
  - **研究变难**：Bloom 等「低垂果实」效应；AI 自动化研究可能对冲。
  - **抽象壁垒（Abstraction Barrier）**：主要吃人类符号/文本抽象的产品，可能难以 **从原始传感数据发现新概念**；若成立，单实例或达 AGI 平台，但集体 scaling 仍可能推进 ASI；突破或需 **具身接地交互** 验证假说——**具身瓶颈** 把递归硬件自改进锚定在物理实验速度上。
  - **刻意减速**：监管、事故、社会反弹 vs 军经竞争压力。
- **对 wiki 的映射：**
  - [paper-from-agi-to-asi](../../wiki/entities/paper-from-agi-to-asi.md) — 机器人研究启示节
  - [sim2real](../../wiki/concepts/sim2real.md)、[video-as-simulation](../../wiki/concepts/video-as-simulation.md) — 仿真数据对抗数据墙

### 5) 对当前范式的理论支撑（预训练 ≈ 有界通用压缩）

- **链接：** <https://arxiv.org/abs/2606.12683> §4
- **摘录要点：**
  - 互联网规模 **log-loss 预训练** 可视为 **资源有界下的通用压缩近似**（Genewein et al. 2026 论点）；叠加 **显式规划/决策脚手架** 或 RL 隐式决策，或可向通用智能推进。
  - AIXI 实践鸿沟仍在：**持续学习、超长上下文、稳健规划** 为活跃短板。
  - 评测饱和（GPQA、SWE-bench 等）推动 **ARC-AGI 类** 与 **多智能体 scaling 评测** 需求。
- **对 wiki 的映射：**
  - [foundation-policy](../../wiki/concepts/foundation-policy.md)、[vla](../../wiki/methods/vla.md) — 机器人侧范式实例
  - [robot-learning-three-eras-narrative](../../wiki/queries/robot-learning-three-eras-narrative.md) — 产业 scaling 叙事对照

## 对 wiki 的映射（汇总）

- [paper-from-agi-to-asi.md](../../wiki/entities/paper-from-agi-to-asi.md) — 主沉淀页
- 交叉更新：[embodied-scaling-laws.md](../../wiki/concepts/embodied-scaling-laws.md)、[data-flywheel.md](../../wiki/concepts/data-flywheel.md)、[robot-learning-three-eras-narrative.md](../../wiki/queries/robot-learning-three-eras-narrative.md)

## 引用（arXiv BibTeX）

```bibtex
@article{genewein2026agitoasi,
  title={From {AGI} to {ASI}},
  author={Genewein, Tim and Franklin, Matija and Lerchner, Alexander and Orseau, Laurent and Albanie, Samuel and Bales, Adam and Wyeth, Cole and Chan, Stephanie and Gabriel, Iason and Leibo, Joel Z. and Dafoe, Allan and Hutter, Marcus and Graepel, Thore and Legg, Shane and others},
  journal={arXiv preprint arXiv:2606.12683},
  year={2026}
}
```
