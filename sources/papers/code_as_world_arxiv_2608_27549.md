# Code as Worlds: Agentic Discovery of Executable World Representations for Physical Reasoning（arXiv:2608.27549）

> 来源归档（ingest）

- **标题：** Code as Worlds: Agentic Discovery of Executable World Representations for Physical Reasoning
- **缩写：** **Code-as-World** / **CaW**
- **类型：** paper / 物理推理 / 可执行世界表示 / VLM
- **arXiv：** <https://arxiv.org/abs/2608.27549>（PDF：<https://arxiv.org/pdf/2608.27549>）
- **项目页：** <https://mirros-lab.github.io/code-as-world/>
- **代码：** <https://github.com/MirroS-Lab/Code-as-World>
- **权重：** [Code-as-World-VL-4B](https://huggingface.co/MirroS-Lab/Code-as-World-VL-4B)、[Code-as-World-VL-9B](https://huggingface.co/MirroS-Lab/Code-as-World-VL-9B)
- **作者：** MirroS Team（Wang Hanyang, Cai Yimo, Chen Weiliang 等；含 Long Mingsheng, Liu Ziwei, Duan Yueqi 等）
- **入库日期：** 2026-09-21
- **一句话说明：** 用 **可执行代码** 表示物理世界（实体、状态、动力学、渲染），经 **agentic abductive discovery loop** 从语言/视频提出–执行–渲染–验证–迭代假设；以 verified executable worlds 作 scalable 物理监督，训练 **Code-as-World-VL** 在 **QuantiPhy** 上 SOTA 并超越 leading proprietary models。

## 摘要级要点

- **问题：** 现代 VLM 能识别/解释物理事件，但常缺 **机制级** 显式表示（物体状态、物理参数、 governing dynamics），难可靠推理演化与干预。
- **范式：** **Code-as-World** — 物理组成、动态演化与视觉外观均表达为 **可执行代码**；强调 **Abstraction / Compositionality / Executability / Controllability** 四性质。
- **发现循环：** 受 abductive reasoning 启发 — agent **提出** 可执行世界假设 → **执行** 仿真 → **渲染** 观测 → **验证** 与证据一致性 → **迭代 refine**。
- **应用：** 用 verified executable worlds 为 VLM 提供 **定量物理监督**；发布 **Code-as-World-VL**（4B/9B/27B 规模曲线）。
- **QuantiPhy：** 4B MRA **50.6**、9B **55.4**、27B **58.6**（项目页）；开源仓含 **QuantiPhy 评测脚本** 与 **video-driven simulation 示例**。
- **开源（2026-08-27）：** GitHub 推理 + QuantiPhy eval + MuJoCo simulation example；HF 4B/9B 权重；技术报告与项目页上线。

## 核心论文摘录（MVP）

### 1) 可执行代码作为世界本体

- **链接：** 项目页 Representing the physical world through code
- **摘录要点：** 像素是 evidence 非 ontology；代码分离 task-relevant structure 与 incidental pixels；simulator 可执行使表示可检验。
- **对 wiki 的映射：**
  - [Code-as-World](../../wiki/entities/paper-code-as-world.md) — 四性质表与 discovery loop。

### 2) Agentic discovery loop

- **链接：** arXiv abstract；README Overview
- **摘录要点：** propose → execute → render → verify → refine；从自然语言或真实视频构造 executable world hypotheses。
- **对 wiki 的映射：**
  - [Code-as-World](../../wiki/entities/paper-code-as-world.md) — mermaid 流程图与源码时序图。

### 3) QuantiPhy 与 Code-as-World-VL

- **链接：** 项目页 Parameter scaling；GitHub Get started
- **摘录要点：** 规模越大 MRA 越高；官方 `python -m code_as_world.evaluation` 对接 QuantiPhy CSV 与 validation videos。
- **对 wiki 的映射：**
  - [Code-as-World](../../wiki/entities/paper-code-as-world.md) — 工程实践与开源表。

## 对 wiki 的映射

- 沉淀实体页：[`wiki/entities/paper-code-as-world.md`](../../wiki/entities/paper-code-as-world.md)
- 项目页归档：[`sources/sites/code-as-world-project.md`](../sites/code-as-world-project.md)
- 官方仓库：[`sources/repos/code-as-world.md`](../repos/code-as-world.md)
