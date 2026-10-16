# VT-Bridge: Bridging Pretrained Foundation VLAs to VTLAs via Lightweight Residual Adaptation

> 来源归档（论文）

- **类型：** paper（方法 / 实体）
- **arXiv：** [2609.22606](https://arxiv.org/abs/2609.22606)
- **版本：** arXiv v1，2026-09-18
- **作者：** Yansong Wu、Tuo Yang、Rongping Zhao、Lingyun Chen、Xiao Chen、Junnan Li、Fan Wu、Alois Knoll
- **机构：** Technical University of Munich；Lingyun Chen 另有 Mohamed bin Zayed University of Artificial Intelligence；Fan Wu 为 Shanghai University
- **项目页：** https://hoxnocha.github.io/vt-bridge-web/
- **项目页仓库：** https://github.com/hoxnocha/vt-bridge-web
- **研究代码：** 截至 2026-10-16 未公开；项目页 Code 按钮显示 “Coming soon”

## 内容摘要

VT-Bridge 保留预训练 VLA 主干，并在其 action chunk 上增加轻量触觉残差适配器：较慢的 VLA 负责语义理解与粗动作计划，适配器以机器人执行频率读取近期外力/关节力矩反馈，对每一步动作做接触调节。论文先用 LoRA 微调 VLA，再冻结主干训练残差适配器；残差目标由示范中的期望命令减去对齐后的 VLA 名义动作构造，非接触样本的目标残差设为零。

作者在 π₀、π₀.₅ 与 SmolVLA 三种主干、四项真实接触丰富操作任务上评估。每项任务最多使用 50 条视触觉示范，适配器为 0.98M 参数；平均任务完成率从仅任务微调 VLA 的 11.7% 提高到 62.9%。该方法展示的是一种将触觉反馈接入现有 VLA 的轻量残差路径，不是重新训练完整 VTLA。

## 对 wiki 的映射

- [VT-Bridge 实体页](../../wiki/entities/paper-vt-bridge.md) — 训练目标、慢快推理、真机评测与开源边界
- [视触觉融合](../../wiki/concepts/visuo-tactile-fusion.md) — 残差适应范式补充
- 开源状态依据：[项目页归档](../sites/vt-bridge-project-page.md) 与[项目页源码仓库归档](../repos/hoxnocha-vt-bridge-web.md)
