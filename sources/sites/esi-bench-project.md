# ESI-Bench 项目页（esi-bench.github.io）

> 来源归档

- **标题：** ESI-Bench: Towards Embodied Spatial Intelligence that Closes the Perception–Action Loop
- **类型：** site（项目页 + 演示视频）
- **URL：** <https://esi-bench.github.io/>
- **论文：** <https://arxiv.org/abs/2605.18746>（v2，2026-05-25）
- **代码：** <https://github.com/ESI-Bench/ESI-Bench>（官方仓库，MIT，已开源）
- **数据集：** <https://huggingface.co/datasets/esi-bench/ESI-Bench>；单视图集 <https://huggingface.co/datasets/ESI-Bench/esi-bench-passive-single>
- **核查日期：** 2026-11-07
- **入库日期：** 2026-05-22
- **一句话说明：** 官方项目页：摘要、10/29/3081 规模统计、完整 **Task Taxonomy**（含每子类一句话能力说明）、相对既有空间基准的 **三大 departure**、三条 **Key Findings**（含 emergent 策略示意与数值案例）、BibTeX。

## 页面结构（维护索引）

| 区块 | 内容要点 |
|------|----------|
| Abstract | 感知–行动环；主动揭示 unseen；MLLM 实验与行动盲 / 3D / 元认知结论 |
| 统计 | 10 类、29 子类、3081 实例 |
| Task Taxonomy | 01–10 类及子类（Physical Capacity … Action Sequencing） |
| Three Departures | 从 sensing→competence；选择性传感；消解 perceptual mirages |
| Key Findings | 行动盲与级联；3D 完美 vs 重建有害；人类 vs 模型元认知 |
| Citation | `@article{hong2026esibench,...}` |

## 对 wiki 的映射

- 主实体：[ESI-Bench（具身空间智能基准）](../../wiki/entities/esi-bench.md)
- 论文摘录：[esi_bench_arxiv_2605_18746.md](../papers/esi_bench_arxiv_2605_18746.md)

## 2026-11-07 核查补充

- **开放结论：** 项目页顶部提供 Paper / Code / Dataset 导航；官方 GitHub README 明确列出数据、`dataset/json_clean`、`src/main.py` 运行说明与 MIT License。因此是**已开源的评测代码及任务数据**，但依赖外部 OmniGibson / BEHAVIOR-1K 环境和模型 API；开源不等于即装即跑。
- **术语映射：** 项目页使用 `Physical Capacity`、`Temporal Scene`；论文 v2 与代码 JSON 分类对应 `Physical Structure`、`Temporal Understanding`，共计仍为 10 大类、29 子类。
- **源码入口：** [`src/main.py`](https://github.com/ESI-Bench/ESI-Bench/blob/main/src/main.py) 转发到 [`src/active_explore/pipeline.py`](https://github.com/ESI-Bench/ESI-Bench/blob/main/src/active_explore/pipeline.py)。
- **代码与论文归档：** [源码归档](../repos/esi_bench.md)、[论文归档](../papers/esi_bench_arxiv_2605_18746.md)。
