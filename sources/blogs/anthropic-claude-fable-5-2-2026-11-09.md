# Claude Fable 5.2：Anthropic 官方发布博客

- **英文标题：** Claude Fable and Mythos 5.2
- **中文标题：** Claude Fable 5.2 与 Mythos 5.2 发布
- **类型：** Anthropic 官方博客
- **原文：** <https://www.anthropic.com/claude-fable-and-mythos-5-2>
- **日期：** 2026-11-09
- **核验日期：** 2026-11-11
- **核对范围：** Anthropic 2026-11-09 官方发布博客全篇：模型引言、基准图表、Coding/Research/Hardware and robotics/安全与可用性等章节、RoboDojo Sim 说明、真机演示的正文及页末说明；以公开网页版本为准。
- **通读：** 已完成

## 从一手来源确认的要点

1. 官方将 RoboDojo Sim **64.9%** 归于 Fable 5.2 的 **最高努力档（Max effort）**；该指标是指定评测设置下作者报告的成绩，不是跨全部机器人任务的通用成功率。
2. 博客展示真机魔方、旋瓶盖、绘画等机器人演示；演示与 RoboDojo 仿真评测不是同一评测协议，不能把视频当成 64.9% 的独立复现。
3. 官方页面是模型能力与应用演示，不是机器人策略训练源码或全套 RoboDojo 可复现协议发布。

## 待核验

- 官网全文及附录 / benchmark 细项、统计置信区间、机器人的控制接口与实验次数。
- 是否公开特定机器人任务的完整复现实验脚本。

## 全文核验补充（2026-11-11）

- 官方将 RoboDojo Sim 定义为**从相机图像控制模拟双臂机器人**。其 64.9% 结果基于 Anthropic 的**内部运行**与 **Max effort**，对照 GPT-6 Astra 29.8%、Fable 5.1 27.6%；分数不属于实机演示。
- 真机视频展示魔方、旋瓶盖、绘画，但原文没有提供这些视频每任务的重复次数、独立成功率或控制 API 运行代码。不能将演示视频当作可复现的统计实验。
- 机器人只是该发布博客 Hardware and robotics 的子部分，全文还包含代码、科学研究、硬件设计和安全评估。独立研究复现仍需等待明确任务接口及评测协议。

## 对 wiki 的映射

- [Claude Fable 5.2 主节点](../../wiki/entities/claude-fable-5-2.md)
