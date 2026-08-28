# StreamPI: Streaming Multimodal Temporal Modeling for Vision-Language-Action Models

> 来源归档（ingest）

- **标题：** StreamPI: Streaming Multimodal Temporal Modeling for Vision-Language-Action Models
- **短名：** StreamPI
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2608.26067>
- **PDF：** <https://arxiv.org/pdf/2608.26067>
- **项目页：** <https://happinesslz.github.io/projects/StreamPI>
- **代码：** 官方仓计划 2026-08-30 公开（项目页 News）
- **入库日期：** 2026-08-28
- **索引来源：** [具身智能小站 9 篇盘点](../blogs/wechat_embodied_station_wam_vla_cross_embodiment_9_papers_2026-08-28.md)（<https://mp.weixin.qq.com/s/FNhRO3KOm8k8CkJEqystQQ>）
- **一句话说明：** 不增加参数，为单帧 VLA 注入可流式推理的时间记忆与异步部署能力。

## 开源状态（步骤 2.5）

- **待发布**：项目页写明基于 openpi 的官方实现含 LIBERO / CALVIN / ALOHA 式真机流程，公开日计划 **2026-08-30**。截至入库日（2026-08-28）未检索到可 clone 的训练仓。

## 核心摘录（面向 wiki 编译）

### 摘录 1：指令锚定的流式注意力

- 每个（视觉观察，语言指令）对是原子时间单元：单元内双向注意力做跨模态融合，单元间因果注意力维持流式推理；指令持续充当语义锚点。
- 随机间隔流式训练（间隔如 U[3,7]）弥合同步训练与异步真机执行的差距。
- 零新增参数，可继承单帧预训练权重；支持单帧或多帧推理。

**对 wiki 的映射：** [paper-streampi](../../wiki/entities/paper-streampi.md)、[VLA](../../wiki/methods/vla.md)

### 摘录 2：评测

- LIBERO：StreamPI (T=5) 平均 **98.3%** vs π0.5 **96.9%**（+1.4 pp）；Goal / Long 增益最大。
- 真机 AgileX PiperX：Cup Insertion 60%→92%，Shell Game 46.7%→80%。
- CALVIN ABC→D 平均链长 4.547 vs π0.5 的 4.313。

**对 wiki 的映射：** [libero-benchmark](../../wiki/entities/libero-benchmark.md)

## 对 wiki 的映射

- 升格 [`wiki/entities/paper-streampi.md`](../../wiki/entities/paper-streampi.md)

## 当前提炼状态

- [x] 方法要点与开源核查
- [x] wiki 实体与技术地图回链
