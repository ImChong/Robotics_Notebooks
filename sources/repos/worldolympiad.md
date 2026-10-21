# WorldOlympiad 官方代码仓库

> 来源归档（官方仓库 README 与项目页核查）

- **标题：** WorldOlympiad: Can Your World Model Survive a Triathlon?
- **类型：** repo / benchmark
- **代码：** <https://github.com/alibaba-damo-academy/WorldOlympiad>
- **项目页：** <https://alibaba-damo-academy.github.io/WorldOlympiad/>
- **论文：** <https://arxiv.org/abs/2606.11129>
- **数据集：** <https://huggingface.co/datasets/ziplab/WorldOlympiad>
- **许可证：** 官方项目页声明项目网站与代码采用 MIT License；数据集以 Hugging Face 页面和数据文件许可为准。
- **开源状态：** 已开放评估代码与数据入口；官方仓库 README 提供环境、权重、数据目录和批量评估运行说明。
- **一句话说明：** 长视频世界模型诊断基准，统一运行物理、几何与长程交互三条评测管线。
- **入库日期：** 2026-10-16

## 仓库入口摘要

- `worldeval/physical/`：物体级物理规则证据与裁判流程。
- `worldeval/3d_metrics/`：DA3 / Gaussian-splat 几何评估。
- `worldeval/interaction/`：VLM 与 CLIP 的片段级、全局交互评估。
- README 说明批量输入需包括参考视频、生成视频、`prompt.json`，并可记录生成片段时间戳；输出逐例 judge JSON，可跳过已完成样本以恢复批处理。
- 官方项目页列出 1,000 段视频，涵盖 robotics、gaming、general 三个域。

## 对 wiki 的映射

- [WorldOlympiad](../../wiki/entities/worldolympiad.md)
- [WorldArena](../../wiki/entities/worldarena.md) — 对照另一种具身世界模型评测分解方式。
