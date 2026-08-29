# CLAP 项目页（omni-clap.github.io）

> 来源归档

- **标题：** CLAP: Cross-Embodiment Action-Conditioned Video World Models are Zero-Shot Physical Simulators
- **类型：** site（项目页）
- **URL：** <https://omni-clap.github.io/>
- **论文：** <https://arxiv.org/abs/2608.27406>
- **代码：** <https://github.com/omni-CLAP/clap>
- **权重：** <https://huggingface.co/omni-CLAP/CLAP>
- **入库日期：** 2026-08-29
- **一句话说明：** 官方项目页核对结论：**已开源**（GitHub + HF 权重）；展示 EE / LAM / language 条件、课程配方、DROID/Bridge 对比，以及 YAM / G1 适配样例。

## 页面结构（维护索引）

| 区块 | 内容要点 |
|------|----------|
| Abstract / Overview | 跨本体视频 WM；课程先潜动作后末端接地 |
| Action-conditioning | `ee` 7-D、`lam` 32-D、`language` CLIP 文本 |
| Models | `clap-curr` 默认；`adapt-yam` / `adapt-g1`；DROID/Bridge 后训练 |
| Comparisons | 对 Ctrl-World / Bridge-Base；零样本规划；few-shot 适配 |
| BibTeX | 页上仍写 Coming soon；引用以 arXiv:2608.27406 为准 |

## 开源核查（步骤 2.5，2026-08-29）

- **代码：** 页头列出 GitHub `omni-CLAP/clap`。
- **权重：** 指向 `omni-CLAP/CLAP`。
- **结论：** **已开源**（训练/推理入口 + 检查点）。

## 对 wiki 的映射

- 主实体：[CLAP](../../wiki/entities/paper-clap-cross-embodiment.md)
- 论文摘录：[clap_arxiv_2608_27406.md](../papers/clap_arxiv_2608_27406.md)
- 代码仓库：[omni-clap.md](../repos/omni-clap.md)
