# A Taxonomy of Construction Task Activities for Robot Workers

> 来源归档（ingest）

- **标题：** A Taxonomy of Construction Task Activities for Robot Workers
- **短名：** TARCAT
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2608.25395>
- **PDF：** <https://arxiv.org/pdf/2608.25395>
- **代码 / 标注：** <https://github.com/AICPS/TARCAT-Taxonomy>
- **入库日期：** 2026-08-28
- **索引来源：** [具身智能小站 9 篇盘点](../blogs/wechat_embodied_station_wam_vla_cross_embodiment_9_papers_2026-08-28.md)（<https://mp.weixin.qq.com/s/FNhRO3KOm8k8CkJEqystQQ>）
- **一句话说明：** 用 41 个动作原语建立建筑作业共同词表，把人类工种活动连接到机器人技能库。

## 开源状态（步骤 2.5）

- **已开源（标注与分类体系）**：[`AICPS/TARCAT-Taxonomy`](https://github.com/AICPS/TARCAT-Taxonomy) 发布 `primitives.json`、`composite/` 技能族与视频标注；README 写 TARCAT v1.0（2026-08-25）。**不是**可训练 VLA / 控制策略仓。作者还在搭载 CRAFT 手的 DOBOT CR3 上演示部分原语。

## 核心摘录（面向 wiki 编译）

### 摘录 1：职业任务驱动的分类

- 来源：7 个高就业建筑工种的 **91** 项 O\*NET 任务 + **30** 段实体作业教学视频。
- **41** 个动作原语，组织为 **12** 个组与 **3** 个类别；带参数的原语序列可组合成可复用技能。

**对 wiki 的映射：** [paper-tarcat](../../wiki/entities/paper-tarcat.md)、[VLA](../../wiki/methods/vla.md)

### 摘录 2：用途

- 整理示范、定义机器人能力需求、支持编码智能体检索和扩展技能库。

**对 wiki 的映射：** [manipulation](../../wiki/tasks/manipulation.md)

## 对 wiki 的映射

- 升格 [`wiki/entities/paper-tarcat.md`](../../wiki/entities/paper-tarcat.md)

## 当前提炼状态

- [x] 方法要点与开源核查
- [x] wiki 实体与技术地图回链
