# MetaRSI-v1 / RSI2：A Meta-Recursive Self-Improving System for Recursive Self-Improving Systems Themselves

> 来源归档（ingest）

- **标题：** MetaRSI / RSI2: A Meta-Recursive Self-Improving System for Recursive Self-Improving Systems Themselves（项目页亦称 **MetaRSI-v1**）
- **标语：** One Kernel, Two Axes, Three Operators, Every Domain
- **类型：** paper / llm-agents / recursive-self-improvement / agent-harness / ai-auto-research
- **arXiv：** <https://arxiv.org/abs/2609.06396>（Submitted 2026-09-06；Online 2026-09-09；PDF：<https://arxiv.org/pdf/2609.06396>）
- **项目页：** <https://www.cosmosmind.ai/research/metarsi-v1>
- **机构站：** <https://cosmosmind.ai/>
- **代码（Harness-RSI）：** <https://github.com/CosmosMind-ai/RSI-Harness> — 归档见 [`sources/repos/rsi-harness.md`](../repos/rsi-harness.md)
- **Hugging Face：** <https://huggingface.co/CosmosMind/RSI-Harness>
- **作者：** Zihan Tan*、Leixin Sun*†（project leads）、Guancheng Wan‡（corresponding）等 31 人（* equal · † leads · ‡ corresponding）
- **机构：** 宇宙心智（CosmosMind AI Lab）
- **入库日期：** 2026-09-14
- **一句话说明：** 把 RSI 从「单表面、可形式化评测域」推进为 **同一 loop kernel 上 Data-RSI / Harness-RSI / Model-RSI 三算子的可组合调度**；两轴优化器联合决定算子顺序与各算子 proposal policy，meta-level policy 跨 term 修订 schedule；在 code 与 closed-form science 标准评测上 **无外部 teacher** 自举验证。

## 开源状态（步骤 2.5）

- **项目页（2026-09-14）：** 列出 PDF + GitHub（RSI-Harness）+ Hugging Face — **Harness-RSI 实现已公开**。
- **RSI-Harness 仓：** 可安装 `rsih` / `gee` / Genome 工具链；README 与 HF 均写明 **不含** benchmark、数据生成、训练或评测代码 → **Model-RSI / Data-RSI 全栈训练环未随官方 harness 仓发布**（论文实验栈待论文/后续仓跟进）。
- **结论：** **部分开源** — Harness-RSI **已开源**；MetaRSI 完整三算子闭环的 **训练 / benchmark 代码** 截至入库日 **未在 RSI-Harness 仓提供**。

## 摘录 1：问题与主张（Abstract）

- **痛点：** RSI 迄今几乎只在 **coding** 与 **可机器检验** 的 formal benchmark（science QA、数学）上验证 — **format bound** 把改进锁在「答案可自动判对错」的切片，而非开放问题、靠论证/复现/测量定真伪的通用能力。
- **主张：** RSI 下一步应跨 **真实多元的科学、工程与 meta-scientific 域** 运作，而非只在 formal evaluation 易做的域。
- **MetaRSI-v1：** 改进 = 三 **typed operator** 在 **统一范式** 上的 **scheduled composition**：
  - **Data-RSI** — 放大既有能力并标记边界；
  - **Harness-RSI** — 编辑 **五槽 scaffold**，**不改权重**；
  - **Model-RSI** — 经 **有界训练** 把能力 **内化进参数**。
- **共享：** 同一 loop kernel + artifact vocabulary → data / scaffold / model 改动 **可组合** 而非互斥。
- **优化：** **两轴 optimizer** — 算子 **顺序** + 各算子 **proposal policy**；**meta-level policy** 跨 term 修订 schedule。
- **验证设定：** 领域标准评测；code + closed-form science；**无 external teacher** — target model 在 loop 内扮演 **全部角色**。
- **两条路径：** **model route**（训练内化）与 **harness route**（权重不动 → 可延伸到 **任意经 interface 可达的模型**）；**Data-RSI** 重定义为 **喂给两条路线的共享底物**。
- **理论产出：** 关于 loop 存在条件、算子如何 compose、supervision 买到什么的 **可反驳定律**。

**对 wiki 的映射：** 升格 [`wiki/entities/paper-metarsi-v1.md`](../../wiki/entities/paper-metarsi-v1.md)；Harness 实现链 [`wiki/entities/rsi-harness.md`](../../wiki/entities/rsi-harness.md)；概念互链 [`wiki/concepts/recursive-self-improvement.md`](../../wiki/concepts/recursive-self-improvement.md)、[`wiki/concepts/ai-auto-research.md`](../../wiki/concepts/ai-auto-research.md)。

## 摘录 2：Harness-RSI 与 RSI-Harness  artifact（项目页 + 官方仓）

- **RSIH：** Pi coding agent + **Genome** 配置层 — harness 变成 **可版本、可分享、可自动生成** 的目录对象（12 组件：prompt / tools / skills / MCP / runtime / …）。
- **GEE（Genome Expression Engine）：** `gee` → `harness-rsi` Genome — 读 **真实 session 史**（RSIH / Pi / Claude Code stores），聚合 tool histogram、bash、热文件与重复纠错，经用户确认后写出通过 `rsih genome validate` 的 Genome。
- **自指：** 「RSI」在此 **不是** 模型改权重，而是 **用与普通 Genome 相同的手段编辑自己的 harness**；`src/` 无专供 `harness-rsi` 的硬编码分支。
- **仓边界：** **无** benchmark / training / eval — 仅是 **Harness-RSI 工程载体**，不是 MetaRSI 全论文复现包。

**对 wiki 的映射：** [`sources/repos/rsi-harness.md`](../repos/rsi-harness.md) + 实体页 **源码运行时序图**（install → gee → validate → launch）。

## 摘录 3：与 harness 自进化文献的对照读法

| 维度 | MetaRSI Harness-RSI | HarnessBank | SoL-Pi | karpathy/autoresearch |
|------|---------------------|-------------|--------|----------------------|
| 改什么 | 五槽 scaffold / Genome 全表面 | prompt·runtime·config 等 | Pi extension 效率机制 | 仅 `train.py` |
| 是否动权重 | **否**（harness route） | **否** | **否** | **是**（训练代码） |
| 组合性 | 与 Data / Model 算子 **同一 kernel** | 独立 harness 进化 | 固定 4 机制插件 | 单文件环 |
| 官方代码 | RSI-Harness（harness 栈） | 待发布 | NVlabs/SoL-Pi | karpathy/autoresearch |

**对 wiki 的映射：** 实体页「与其他工作对比」；互链 [HarnessBank](../../wiki/entities/paper-harnessbank.md)、[SoL-Pi](../../wiki/entities/sol-pi.md)、[autoresearch](../../wiki/entities/karpathy-autoresearch.md)。

## 建议 wiki 动作

- 新建 **`wiki/entities/paper-metarsi-v1.md`**（含流程总览 mermaid + 结论 + Harness 时序图）。
- 新建 **`wiki/entities/rsi-harness.md`**（Harness-RSI 工程选型页）。
- 注册机构 **`cosmosmind`** → `schema/institutions.json`。
- 轻量互链 [`wiki/concepts/recursive-self-improvement.md`](../../wiki/concepts/recursive-self-improvement.md) 的 `related`（可选）。
