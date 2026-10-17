# LADDER 项目页（ladder.insait.ai）

> 来源归档

- **标题：** LADDER: A Long-Horizon Robot Manipulation Benchmark
- **类型：** site（项目页）
- **URL：** <https://ladder.insait.ai/>
- **论文：** <https://arxiv.org/abs/2610.18053>（归档：[ladder_arxiv_2610_18053.md](../papers/ladder_arxiv_2610_18053.md)）
- **代码：** <https://github.com/insait-institute/ladder>（页面列出；2026-10-17 WebFetch 返回 404，内容未核实，见 [ladder-insait.md](../repos/ladder-insait.md)）
- **数据集 / 模型：** Hugging Face 集合 <https://huggingface.co/collections/INSAIT-Institute/ladder-6abbbddc46691080f5d954c8>（页面列出，未打开核验）
- **入库日期：** 2026-10-17
- **一句话说明：** 145 个任务、1–10 个物体 / 子目标的长程操作基准；最佳系统约 6 成任务完成，8–10 物体仅 34.8%。

## 页面结构

| 区块 | 内容要点 |
|------|----------|
| 设计 | 每个额外物体多一次失败机会；任务仅在全部物体到位才算成功 |
| 排行 | GPT-6 Astra 63.3%；8–10 物体 34.8%；成本 6.49 美元 / 完成任务；PSP-Agent 经 Gemini 3.7 Flash 42.8% / 0.33 美元 |
| LADDER-Lite | 10 任务子集，三个模型对比 |
| 真机 | Franka 与 WidowX；与仿真排序一致 |
| 资源 | 论文、GitHub、HF 集合、排行榜提交指南、details.html |

## 开源核查结论

截至 2026-10-17：页面列出 GitHub 与 HF 集合，但 GitHub 链接实测 404，状态按"宣称已开放、未核实"处理。

## 对 wiki 的映射

- [wiki/entities/paper-ladder-long-horizon-benchmark.md](../../wiki/entities/paper-ladder-long-horizon-benchmark.md)
- [sources/papers/ladder_arxiv_2610_18053.md](../papers/ladder_arxiv_2610_18053.md)
