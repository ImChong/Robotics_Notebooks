# Zero-WAM: In-Context World-Action Modeling from Human Videos for Open-Ended Task Generalization

> 来源归档（ingest）

- **标题：** Zero-WAM: In-Context World-Action Modeling from Human Videos for Open-Ended Task Generalization
- **短名：** Zero-WAM
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2608.26103>
- **PDF：** <https://arxiv.org/pdf/2608.26103>
- **项目页：** <https://robbyant-research.github.io/Zero-WAM/>
- **代码：** <https://github.com/robbyant-research/Zero-WAM>
- **入库日期：** 2026-08-28
- **索引来源：** [具身智能小站 9 篇盘点](../blogs/wechat_embodied_station_wam_vla_cross_embodiment_9_papers_2026-08-28.md)（<https://mp.weixin.qq.com/s/FNhRO3KOm8k8CkJEqystQQ>）
- **一句话说明：** 把人类视频当 in-context 任务规格，因果视频-动作模型零样本执行未见操作任务。

## 开源状态（步骤 2.5）

- **待发布**：[`robbyant-research/Zero-WAM`](https://github.com/robbyant-research/Zero-WAM) 仓已建（Apache-2.0）；README 写明代码 / 模型 / 数据计划 **2026-09-15 前**发布。截至入库日仅 LICENSE、README 与 `docs/assets`，无可运行训练入口。

## 核心摘录（面向 wiki 编译）

### 摘录 1：人类视频作任务规格

- 零样本跨任务泛化被改写成 in-context 任务指定；语言不足以描述物体交互过程，人类视频提供任务演化线索。
- HumanGen：自动把任务采样的机器人轨迹转成语义对齐人类视频，**7.42 万**人机配对、覆盖 **8600** 个任务。
- IFP（in-context future chunk prediction）抑制从已见任务抄近道，迫使模型从视频提示抽取任务信息。

**对 wiki 的映射：** [paper-zero-wam](../../wiki/entities/paper-zero-wam.md)、[World Action Models](../../wiki/concepts/world-action-models.md)

### 摘录 2：评测

- RoboTwin 2.0 七个未见任务平均成功率 **47.0%**（相对最强视频动作基线 LingBot-VA **+29.5 pp**）。
- 真机展示多物体、长时程与精细插入任务泛化，无需对应机器人数据或参数更新。

**对 wiki 的映射：** [manipulation](../../wiki/tasks/manipulation.md)

## 对 wiki 的映射

- 升格 [`wiki/entities/paper-zero-wam.md`](../../wiki/entities/paper-zero-wam.md)

## 当前提炼状态

- [x] 方法要点与开源核查
- [x] wiki 实体与技术地图回链
