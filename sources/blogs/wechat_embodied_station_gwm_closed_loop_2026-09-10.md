# 世界模型的定义虽然很乱，但闭环正在收拢

> 来源归档（blog / 微信公众号）

- **标题：** 世界模型的定义虽然很乱，但闭环正在收拢
- **类型：** blog
- **作者：** 具身智能之心（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/2J1bmGFOL2yC8IvUURBAAg
- **发表日期：** 2026-09-10（入库日）
- **入库日期：** 2026-09-10
- **抓取方式：** WebFetch（桌面 UA 返回微信验证页；正文由 WebFetch 可读通道获取）
- **原始抓取落盘：** [`sources/raw/wechat_embodied_station_gwm_closed_loop_2026-09-10.md`](../raw/wechat_embodied_station_gwm_closed_loop_2026-09-10.md)
- **一句话说明：** 盘点 Fei-Fei 功能分类与生数 GWM 第一性原理报告，串起 Motubrain / RTC 异步部署 / Motus2 闭环自进化；**5/5 独立详情节点**（1 新建 + 4 复用；0 重复 arXiv）。

## 文内点名 → 本库节点

| # | 资料 | 身份 | 开源结论（入库日） | wiki |
|---|------|------|-------------------|------|
| 01 | A Functional Taxonomy of World Models | 博客 / 概念文（Fei-Fei / World Labs，2026-06） | **确认未开源** | [functional-taxonomy-world-models](../../wiki/concepts/functional-taxonomy-world-models.md) **复用** |
| 02 | General World Models from First-Principles | 技术报告 / 手稿（生数 / 清华，2026-08） | **部分** — 战略页 + 演讲/PDF；**无可运行训练代码** | [paper-gwm-first-principles](../../wiki/entities/paper-gwm-first-principles.md) **新建** |
| 03 | Motubrain | 论文 arXiv:2604.27792 | **部分开源** — 官方仓 README 占位 | [paper-motubrain](../../wiki/entities/paper-motubrain.md) **复用** |
| 04 | World Action Models in Real Time | 论文 arXiv:2608.01880 | **已开源** 博客 + 论文；Motubrain 部署对照 | [paper-wam-realtime-async](../../wiki/entities/paper-wam-realtime-async.md) **复用** |
| 05 | Motus2 | 论文 arXiv:2608.30237 | **未开源**（项目页无代码仓） | [paper-motus2](../../wiki/entities/paper-motus2.md) **复用** |

**产品提及（非独立 paper 节点）：** Vidu Q3（L1）、Vidu S1（L2）— 见 [paper-gwm-first-principles](../../wiki/entities/paper-gwm-first-principles.md) 路线图表。

## 文内要点速记

1. **术语过载：** 视频生成 / 3D / 自驾 / 机器人都在叫「世界模型」，缺共同能力坐标；具身让闭环问题变紧迫。
2. **功能 vs 第一性原理：** Fei-Fei 按 **输出** 分 Renderer / Simulator / Planner；生数报告从 **理解–想象–行动** 闭环 + L1–L5 自主性分级定义 GWM。
3. **数据 recipe：** D1 互联网视频 → D2 教学视频 → D3 第一视角人视频 → D4 带动作记录的人示范 → D5 真机轨迹；Motubrain 称 **50–100 条** 目标机轨迹可适配。
4. **架构：** MoT 让视觉 / 语言 / 动作共享注意力上下文，避免模块化快照过期。
5. **实时：** RTC 在执行当前 chunk 时滚动更新未完成部分，处理新旧动作块衔接。
6. **Motus2：** 同一模型 policy + simulator + evaluator；常规数据训 WAM、次优/失败数据扩仿真与评估；MBRL + Best-of-N 五任务 **65%→75%**；仍属 L3 口径、向 L4 自进化迈步。

## 对 wiki 的映射

- **5/5 独立详情节点**；**0 重复 arXiv 节点**。
- 阅读坐标：[GWM 闭环 5 篇技术地图](../../wiki/overview/gwm-closed-loop-5-papers-technology-map.md)
- 交叉：[World Action Models](../../wiki/concepts/world-action-models.md)、[功能分类](../../wiki/concepts/functional-taxonomy-world-models.md)、[生成式世界模型](../../wiki/methods/generative-world-models.md)

## 当前提炼状态

- [x] 公众号正文抓取与 raw 归档
- [x] 5 篇独立节点核查（1 新建 / 4 复用 / **0 重复 arXiv**）
- [x] 项目页与开源状态核查（步骤 2.5）
