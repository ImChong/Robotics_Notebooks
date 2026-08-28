# ConfAL-WM: Confidence-Guided Active Learning for Action-Conditioned World Models

> 来源归档（ingest）

- **标题：** ConfAL-WM: Confidence-Guided Active Learning for Action-Conditioned World Models
- **短名：** ConfAL-WM
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2608.25572>
- **PDF：** <https://arxiv.org/pdf/2608.25572>
- **项目页：** <https://ConfAL-WM.github.io>
- **代码：** <https://github.com/ConfAL-WM/ConfAL-WM>
- **权重 / 数据：** <https://huggingface.co/anonymous89793/ConfAL-WM> ；<https://huggingface.co/datasets/anonymous89793/ConfAL-WM-Dataset>
- **入库日期：** 2026-08-28
- **索引来源：** [具身智能小站 9 篇盘点](../blogs/wechat_embodied_station_wam_vla_cross_embodiment_9_papers_2026-08-28.md)（<https://mp.weixin.qq.com/s/FNhRO3KOm8k8CkJEqystQQ>）
- **一句话说明：** 用稠密置信度风险图同时决定世界模型该学哪些数据、重点修正哪些区域。

## 开源状态（步骤 2.5）

- **已开源**：[`ConfAL-WM/ConfAL-WM`](https://github.com/ConfAL-WM/ConfAL-WM) 含 `al_pipeline/`、`trainer/train_evac_with_al.py`、`eval/`；HF 发布 EVAC warmup / v2 权重、置信度探针与预计算评测产物。构建于 [EnerVerse-AC (EVAC)](https://github.com/AgibotTech/EnerVerse-AC)。

## 核心摘录（面向 wiki 编译）

### 摘录 1：置信度探针驱动的主动后训练

- 在 EVAC UNet 解码特征上加轻量置信度探针，预测潜空间稠密置信度图，聚合为任务 / 帧 / 图块三级评分。
- 流程：少量目标域数据重训探针并预热模型 → 任务级预筛选分配采样预算 → 已选数据 + 可选帧/图块加权增强训练。

**对 wiki 的映射：** [paper-confal-wm](../../wiki/entities/paper-confal-wm.md)、[生成式世界模型](../../wiki/methods/generative-world-models.md)

### 摘录 2：评测

- RoboTwin 2.0：置信度选择提高后训练效率；稠密加权优于标量奖励、进度及评审式评分基线。

**对 wiki 的映射：** [World Action Models](../../wiki/concepts/world-action-models.md)

## 对 wiki 的映射

- 升格 [`wiki/entities/paper-confal-wm.md`](../../wiki/entities/paper-confal-wm.md)

## 当前提炼状态

- [x] 方法要点与开源核查
- [x] wiki 实体与技术地图回链
