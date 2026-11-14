# DemoStream: Batched Task-and-Motion-Plan Optimization for Automated Learning from Demonstration

> 来源归档（ingest），来自 2026-11-13「具身智能小站」18 篇论文阅读清单；以原始 arXiv 及官方项目和仓库为核查依据。

## 元数据

- **标题：** DemoStream: Batched Task-and-Motion-Plan Optimization for Automated Learning from Demonstration
- **中文：** DemoStream：批量 TAMP 优化自动示范数据
- **arXiv：** 2610.22020
- **版本：** 2026-11-14 检查 arXiv HTML；具体修订号待逐项补齐
- **入库日期：** 2026-11-14
- **核对范围：** 已阅读 arXiv HTML 的摘要、背景、方法、实验、结论、局限及可用附录；公开源码仅核查官方 README 与实际入口，未独立运行。
- **通读：** 已完成
- **论文：** https://arxiv.org/abs/2610.22020
- **HTML：** https://arxiv.org/html/2610.22020
- **PDF：** https://arxiv.org/pdf/2610.22020
- **官方项目页：** https://demostream.github.io/

## 论文机制

任务约束和几何约束进入批量 TAMP；batched streams、计算图优化并行探索可行解，按任务空间代价选择动作与轨迹，再通过行为克隆训练视觉运动策略。

## 实验与证据边界

真机示范收集平均约每小时 66 条；论文图示 75 条真机示范训练得到 67% 的 visuomotor 策略成功率。物体复位和成功标注仍需要人工支持，不代表完全无人工采数。

## 复现与开放状态

项目页提供论文与视频；代码入口未列出可用仓库链接，暂不声称源码可复现。

## 局限

TAMP 优化数据的质量受约束模型、碰撞判定与成本函数影响；示范可行性不保证克隆策略成功。

## 对 wiki 的映射

- [DemoStream：批量 TAMP 优化自动示范数据](../../wiki/entities/paper-demostream-batched-tamp.md)

## 参考来源

- [原论文 HTML](https://arxiv.org/html/2610.22020)
- [转载阅读线索（仅作索引）](https://mp.weixin.qq.com/s/gatO-d6IxsemBSk62EvS1w)
