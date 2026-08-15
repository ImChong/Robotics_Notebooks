# Grounded Semantic Re-Binding for Robust Instruction Generalization in VLAs（arXiv:2608.02497）

> 来源归档（ingest）

- **标题：** Grounded Semantic Re-Binding for Robust Instruction Generalization in Vision-Language-Action Models
- **缩写 / 框架：** **GSR**；原生解耦模型 **ParaVLA**（0.33B）
- **类型：** paper / vla / instruction-generalization / libero
- **arXiv：** <https://arxiv.org/abs/2608.02497>
- **代码：** <https://github.com/AutoLab-SAI-SJTU/GSR-ParaVLA>（归档见 [`sources/repos/gsr-paravla.md`](../repos/gsr-paravla.md)）
- **权重：** <https://huggingface.co/AutoLab-SJTU/GSR>
- **作者：** Zhaokai Yin、Zhipeng Zhang†
- **机构：** 上海交通大学人工智能学院 AutoLab（SJTU）；Anyverse Dynamics Research Lab
- **入库日期：** 2026-08-15
- **一句话说明：** VLA 改写指令崩溃时内部仍保有正确任务身份，失败来自动态视觉与文本的 joint encoding；GSR 用冻结 T5 抽纯语义、再绑回原生视觉，并从零训动作专家。仅用 canonical 演示，LIBERO-Para 上 SmolVLA **4.47%→49.12%**（+44.6 pp），\(\pi_{0.5}\) PRIDE **70.4**。

## 开源状态（步骤 2.5）

- **仓库核查（2026-08-15）：** [AutoLab-SAI-SJTU/GSR-ParaVLA](https://github.com/AutoLab-SAI-SJTU/GSR-ParaVLA) MIT；`recipes/train_*.sh` 与 `eval_*_libero_para.sh` 齐备；HF 发布 ParaVLA / VLA-Adapter GSR / SmolVLA GSR / \(\pi_{0.5}\) GSR。**已开源、可运行。**
- **结论：** 实体页须含源码运行时序图。

## 摘录 1：诊断

- 改写后 VLA-Adapter / SmolVLA / \(\pi_{0.5}\) 成功率可掉最多 67.53 pp，但 Retrieval@1 与语义保留 \(R\) 仍高于随机。
- 在 VLA-Adapter 最后 Bridge-Attention 前，把 paraphrased 语言特征换成 canonical，成功率 **60%→96%**。
- 给 Qwen 辅支固定假图：Full Para **46.82%→61.58%**。说明 joint V-L 路由引入可分离的 wording shift。

## 摘录 2：修法

- 冻结 T5-large 只看指令；脆弱骨干把原生指令中性化为 “perform the task”；可靠骨干（\(\pi_{0.5}\)）保留原生指令、T5 作补充。
- **必须重初始化动作专家**；只加 T5 而保留原 Qwen 指令几乎无效（46.82→47.31）。
- ParaVLA：DINOv2-Large + T5-Large + 解耦动作专家，0.33B，改写近无掉点。

## 摘录 3：LIBERO-Para（4092 episode）

| 模型 | Goal SR | Full Para SR | PRIDE |
|------|--------:|-------------:|------:|
| SmolVLA Native | 72.0 | 4.47 | 2.6 |
| SmolVLA GSR | 78.0 | **49.12** | 41.4 |
| VLA-Adapter Native | 98.2 | 46.82 | 36.7 |
| VLA-Adapter GSR | 98.0 | **70.94** | 62.0 |
| \(\pi_{0.5}\) Native | 93.0 | 73.60 | — |
| \(\pi_{0.5}\) GSR | 91.0 | **75.59** | **70.4** |
| Xiaomi-Robotics-0（报告） | 98.8 | 76.0 | 69.2 |

\(\pi_{0.5}\) 标准日程 Goal SR 略降，加倍步数恢复到 96.0。

**对 wiki 的映射：** [`wiki/entities/paper-gsr-paravla.md`](../../wiki/entities/paper-gsr-paravla.md)；交叉 [VLA](../../wiki/methods/vla.md)、[LIBERO](../../wiki/entities/libero-benchmark.md)、[Xiaomi-Robotics-0](../../wiki/entities/xiaomi-robotics-0.md)、[π₀.₅](../../wiki/methods/pi07-policy.md)。

## 当前提炼状态

- [x] 论文摘要填写
- [x] wiki 页面映射确认
- [x] 开源状态核查（已开源）
