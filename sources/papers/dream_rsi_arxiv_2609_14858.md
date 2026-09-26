# Dream-RSI: Recursive Self-Improvement through Evolving Worlds

> 来源归档（ingest）

- **标题：** Dream-RSI: Recursive Self-Improvement through Evolving Worlds
- **类型：** paper / recursive-self-improvement / llm-agents / ai-auto-research / meta-exploration / world-models-analogy
- **arXiv：** <https://arxiv.org/abs/2609.14858>（PDF：<https://arxiv.org/pdf/2609.14858>）
- **Hugging Face Papers：** <https://huggingface.co/papers/2609.14858>
- **项目页：** <https://dream-rsi.com/>（归档见 [sources/sites/dream-rsi-com.md](../sites/dream-rsi-com.md)）
- **代码：** <https://github.com/zhengkid/Dream-RSI>（归档见 [sources/repos/dream-rsi.md](../repos/dream-rsi.md)）
- **作者：** Tong Zheng 等（Google · UMD · Google DeepMind · UVA）
- **入库日期：** 2026-09-26
- **一句话说明：** 把 **已完成 discovery 的树形历史** 当作 **精确 replay 模拟器**（非学习 WM），在 **零额外 execution** 下筛选 meta-**exploration policy**；在线部署赢家策略再录新树 → **evolving worlds pool**，形成 exploration 层的 RSI 闭环。

## 开源状态（步骤 2.5）

- **项目页（2026-09-26）：** 链 **Paper PDF**、**GitHub**、**Live demo**；arXiv 按钮暂指向站内 PDF（页内 TODO：待 arXiv 上线后改 abs 链）。
- **GitHub [zhengkid/Dream-RSI](https://github.com/zhengkid/Dream-RSI)：** 公开仓含 README、PDF、`assets/`；README **Release plan** 写明 **Full codebase / Reproduction scripts / Discovered programs ⏳ Being prepared**；顶部 NOTE：**Code is being prepared for release**。
- **结论：** **部分开源 / 待发布** — 论文 PDF + 项目页 + 交互 demo **可用**；**尚无可 clone 即跑** 的完整复现代码（截至入库日）。

## 为什么值得保留

- **RSI 瓶颈重定位：** 长 horizon discovery（数千 proposal–evaluation）的瓶颈是 **exploration policy**（分支、并行、截断），而非底层 coding agent 权重；固定 exploration 无法从累积失败中学习，在线 meta-优化又面临 **延迟贵反馈 + 巨大 meta 策略空间**。
- **History-as-simulator 洞察：** 完成 run 的 **discovery tree**（决策 + 已落盘 execution 结果）允许 **alternative exploration policy** 以不同遍历顺序/并行/停止 **replay**，评分 **只读日志、零重跑** — 类比 model-based RL 的「world」，但是 **exact** 于已探索子空间，非神经网络预测。
- **Evolving worlds：** 每轮在线赢家带回 **新树** → simulator **pool 增长**；跨 pool dreaming 的策略优于只拟合单次 run 的 luck — 与 [MetaRSI-v1](../../wiki/entities/paper-metarsi-v1.md) 的 Data/Harness/Model 算子 **正交**：Dream-RSI 专改 **exploration orchestration**，不改 coding agent 梯度。
- **实证域：** algorithm engineering、mathematical optimization、GPU kernel engineering（8 tasks / 3 domains）；相对 Recursive Fixed Exploration、SimpleTES 等报告 **更少 discovery compute / 更少 agent calls**（项目页与 README 统计条，细节以 PDF 为准）。

## 核心摘录（面向 wiki 编译）

### 三阶段 loop（Fig.1 / 项目页）

1. **Online Explore** — 当前 exploration policy 驱动 coding agent，扩展 discovery tree 并 log traces。
2. **Construct Replay Simulator** — 树转为可复用 simulator（pool 条目）。
3. **Dreaming-based Policy Improvement** — 大量 candidate exploration policies 在 pool 上 **dream replay** 得 immediate off-policy feedback；赢家 **redeploy online**，pool 持续扩大。

### 与 learned world model 的对比（项目页 Insight）

| 读法 | Dream-RSI history replay | 学习型 WM |
|------|--------------------------|-----------|
|  fidelity | 对已探索节点 **exact** | 近似预测 |
| 成本 | 已付 discovery 的 **沉没成本** | 训练 + rollout |
| 覆盖 | 仅 history 到达过的分支 | 可 extrapolate（亦可能 hallucinate） |
|  forcing recursion | 每 lap 必须 online 扩边界 | 可选纯 offline |

###  highlight 指标（README stats 图 / 项目页，归档口径）

- **Algorithm engineering（Gemini-3.1-Pro 等）：** 相对 fixed exploration 约 **1.22×** 更快 downstream runtime、**1.74×** 更少 discovery compute；相对 SimpleTES 约 **162×** 更少 discovery-agent calls（Lasso 等设定，见 PDF）。
- **Mathematical optimization：** 3 tasks 中 **2** 达到或超过所选 baseline。
- **GPU kernel engineering：** 4/4 kernels 改进；等预算约 **2.09×** 性能；等性能约 **2.43×** 更少 generations。
- **Coding agent：** **Zero gradient steps** on the underlying coding agent（改 exploration 层，非权重 RSI）。

## 对 wiki 的映射

- **升格实体页：** [`wiki/entities/paper-dream-rsi.md`](../../wiki/entities/paper-dream-rsi.md)
- **站点 / 仓库：** [`sources/sites/dream-rsi-com.md`](../sites/dream-rsi-com.md)、[`sources/repos/dream-rsi.md`](../repos/dream-rsi.md)
- **概念互链：**
  - [`wiki/concepts/recursive-self-improvement.md`](../../wiki/concepts/recursive-self-improvement.md) — exploration 层 RSI 实例
  - [`wiki/entities/paper-metarsi-v1.md`](../../wiki/entities/paper-metarsi-v1.md) — harness/data/model 算子 RSI（不同改进面）
  - [`wiki/entities/paper-rsi-survey-2607-07663.md`](../../wiki/entities/paper-rsi-survey-2607-07663.md) — 全谱系 taxonomy
  - [`wiki/methods/generative-world-models.md`](../../wiki/methods/generative-world-models.md) — 类比读法；Dream-RSI **非** 生成式 WM 训练
