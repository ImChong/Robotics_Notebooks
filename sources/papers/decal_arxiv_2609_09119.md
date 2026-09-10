# DeCAL: Towards Physically-Grounded Dexterous Vision-Language-Action Models via Contact-Aware Latent Co-Imagination（arXiv:2609.09119）

> 来源归档（ingest）

- **标题：** DeCAL: Towards Physically-Grounded Dexterous Vision-Language-Action Models via Contact-Aware Latent Co-Imagination
- **简称：** DeCAL
- **类型：** paper / vla / tactile / dexterous-manipulation / world-model
- **arXiv：** <https://arxiv.org/abs/2609.09119>
- **PDF：** <https://arxiv.org/pdf/2609.09119>
- **项目页：** <https://aureleopku.github.io/DeCAL/> — 归档见 [`sources/sites/decal.md`](../sites/decal.md)
- **代码：** <https://github.com/AureleoPKU/DeCAL> — 归档见 [`sources/repos/decal.md`](../repos/decal.md)
- **数据/权重：** <https://www.modelscope.cn/datasets/Aureleo/DeCAL_dataset>
- **会议：** CoRL 2026
- **机构：** 北京大学、北京智源人工智能研究院（BAAI）
- **入库日期：** 2026-09-10
- **一句话说明：** MoT 架构统一理解/视触想象/动作；接触感知门控融合触觉 + 视触 latent co-imagination；六项真机任务 mean SR 71%、PSR 83.4%。

## 开源状态（步骤 2.5，2026-09-10）

- **结论：** **已开源** — GitHub 含安装、训练（`launch/decal_finetune.sh`）、评测说明；ModelScope 数据集；预训练骨干 InternVLA-A1-3B 自 Hugging Face 下载。

## 核心摘录（面向 wiki 编译）

### 摘录 1：架构与触觉

- **Mixture-of-Transformers（MoT）** 分专家：理解、想象、动作生成，信息在专家间流动。
- **Adaptive Visuo-Tactile Fusion：** 接触感知门控动态调节触觉注入时机与强度。
- **Visuo-Tactile Latent Co-Imagination：** 联合建模视觉与触觉动力学，为策略注入隐式物理知识。

**对 wiki 的映射：** [paper-decal](../../wiki/entities/paper-decal.md)

### 摘录 2：评测

- 六项真实世界接触丰富灵巧任务：**71% 平均成功率**、**83.4% progress success rate**；四类 OOD 设定仍具泛化。

**对 wiki 的映射：** [paper-decal](../../wiki/entities/paper-decal.md)

## 当前提炼状态

- [x] 项目页与 GitHub 核查（2026-09-10）
- [x] wiki 实体页已建
