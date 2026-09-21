# Code-as-World 项目页

> 来源归档（site）

- **标题：** Code as Worlds: Agentic Discovery of Executable World Representations for Physical Reasoning
- **类型：** site
- **链接：** https://mirros-lab.github.io/code-as-world/
- **arXiv：** <https://arxiv.org/abs/2608.27549>
- **代码：** <https://github.com/MirroS-Lab/Code-as-World>
- **权重：** [4B](https://huggingface.co/MirroS-Lab/Code-as-World-VL-4B)、[9B](https://huggingface.co/MirroS-Lab/Code-as-World-VL-9B)
- **入库日期：** 2026-09-21
- **一句话说明：** 可执行代码表示物理世界 + agentic discovery；Code-as-World-VL 在 QuantiPhy 上 SOTA。
- **沉淀到 wiki：** [`wiki/entities/paper-code-as-world.md`](../../wiki/entities/paper-code-as-world.md)

## 机构

- **MirroS Lab**（镜界）

## 开源状态（步骤 2.5，2026-09-21）

- **代码：** **已开源** — [MirroS-Lab/Code-as-World](https://github.com/MirroS-Lab/Code-as-World)：推理、QuantiPhy 评测、`code_as_world.simulation` 示例。
- **权重：** **已发布** — Hugging Face 4B/9B；README 含 vLLM serving 配方。
- **训练数据 / 完整 discovery pipeline：** 技术报告描述 agentic loop，公开仓以 **推理与评测** 为主；大规模 discovery 训练细节见论文。

## 项目页核心摘录

1. **四性质：** Abstraction、Compositionality、Executability、Controllability。
2. **QuantiPhy MRA：** 4B 50.6 → 9B 55.4 → 27B 58.6。
3. **核心 claim：** 像素是证据；可执行代码提供 compact、quantitative、controllable 物理抽象。

## 对 wiki 的映射

- [`wiki/entities/paper-code-as-world.md`](../../wiki/entities/paper-code-as-world.md)
- [`sources/repos/code-as-world.md`](../repos/code-as-world.md)
