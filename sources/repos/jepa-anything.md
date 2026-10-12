# Gen-Verse/JEPA-Anything

> 来源归档（官方代码仓）

- **仓库：** <https://github.com/Gen-Verse/JEPA-Anything>
- **类型：** PyTorch 核心库 / 任务设计工具 / 结构示例
- **许可证：** Apache-2.0
- **关联论文：** [JEPA-Anything](../papers/jepa_anything_arxiv_2609_20800.md)（arXiv:2609.20800）
- **项目页：** [PhAI Labs 项目稿](../sites/jepa-anything.md)
- **入库日期：** 2026-10-12
- **一句话说明：** 可复用的 orthogonal predictive factorization 核心原语与跨域任务脚手架。

## 仓库边界

仓库包含 typed PyTorch OPF primitives、任务契约/验证器、脚手架生成器及 synthetic structural example。示例用来验证数据和接口结构，不会训练模型或复现论文基准。仓库未包含领域数据集或训练权重；PhAI Labs 页面另列的 Hugging Face model collection 是独立发布资源。

架构文档中的 expected checks 是执行期检查定义，不应当写成已观测的实验结果。项目提供的是实现骨架和研究工具，不等于七个领域模型的成套训练与评测流水线。

## 对 wiki 的映射

- [JEPA-Anything 实体页](../../wiki/entities/paper-jepa-anything.md)
- [生成式世界模型](../../wiki/methods/generative-world-models.md)
- [RSI-Anything 实体页](../../wiki/entities/paper-rsi-anything.md) — 后续工作把 JEPA-Anything 作为预测模型基础。
