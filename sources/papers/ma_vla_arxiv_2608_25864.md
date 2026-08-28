# MA-VLA: Multi-Arm Vision-Language-Action Model for Collaboration and Compositional Generalization

> 来源归档（ingest）

- **标题：** MA-VLA: Multi-Arm Vision-Language-Action Model for Collaboration and Compositional Generalization
- **短名：** MA-VLA
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2608.25864>
- **PDF：** <https://arxiv.org/pdf/2608.25864>
- **项目页：** <https://github.com/zhangzaibin/future-robots>
- **代码：** <https://github.com/zhangzaibin/future-robots>
- **入库日期：** 2026-08-28
- **索引来源：** [具身智能小站 9 篇盘点](../blogs/wechat_embodied_station_wam_vla_cross_embodiment_9_papers_2026-08-28.md)（<https://mp.weixin.qq.com/s/FNhRO3KOm8k8CkJEqystQQ>）
- **一句话说明：** 用逐臂原子动作分配与 Arm Shuffle，让多臂协作摆脱固定执行角色。

## 开源状态（步骤 2.5）

- **已开源**：[`zhangzaibin/future-robots`](https://github.com/zhangzaibin/future-robots) 含 `scripts/train.py`、数据转换、统一/分臂训练与部署；README 将 MA-VLA 标为已完成（ECCV 2026）。许可证文件为 Apache-2.0（README 徽章写 MIT，以 LICENSE 为准）。论文声明开放代码、模型与数据。

## 核心摘录（面向 wiki 编译）

### 摘录 1：原子动作分配 + Arm Shuffle

- 现有 VLA 把语言写成一条全局指令，缺少向不同机械臂分配并组合专属行为的机制。
- MA-VLA 把协作拆成中层原子提示并分配给各臂；Arm Shuffle 同步置换每条臂的观察、状态和原子提示。
- 作者构建测试协作模式不出现在训练集中的 MACG 基准。

**对 wiki 的映射：** [paper-ma-vla](../../wiki/entities/paper-ma-vla.md)、[VLA](../../wiki/methods/vla.md)

### 摘录 2：评测

- 仿真与真机：既有先进 VLA 在未见协作模式下大多失败，MA-VLA 能持续完成任务。

**对 wiki 的映射：** [manipulation](../../wiki/tasks/manipulation.md)

## 对 wiki 的映射

- 升格 [`wiki/entities/paper-ma-vla.md`](../../wiki/entities/paper-ma-vla.md)

## 当前提炼状态

- [x] 方法要点与开源核查
- [x] wiki 实体与技术地图回链
