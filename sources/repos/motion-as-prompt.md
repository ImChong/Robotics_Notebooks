# SunVictor23/MaP

- **标题：** Motion-as-Prompt 官方实现（training-free）
- **类型：** repo
- **URL：** <https://github.com/SunVictor23/MaP>
- **配套论文：** [arXiv:2608.11655](https://arxiv.org/abs/2608.11655) — [`sources/papers/motion_as_prompt_arxiv_2608_11655.md`](../papers/motion_as_prompt_arxiv_2608_11655.md)
- **入库日期：** 2026-08-19

## 一句话说明

轨迹引导 cross-frame visual prompting；不改 MLLM 权重；依赖 CoTracker3 + 可选 Qwen3-VL。

## 仓库状态（2026-08-19 核查）

| 项 | 内容 |
|----|------|
| 核心 | `map_kit/` 轨迹恢复与标注 |
| 评测 | CLEVRER / SSv2 / TempCompass runners |
| 权重 | 无 MaP 训练权重；用外部 tracker/VLM |

## 与 wiki 的关系

- 实体页：[paper-motion-as-prompt](../../wiki/entities/paper-motion-as-prompt.md) — 含源码运行时序图。
