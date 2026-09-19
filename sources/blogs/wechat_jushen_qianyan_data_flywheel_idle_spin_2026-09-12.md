# Ego、遥操、仿真都在采，为什么具身数据飞轮仍会空转？

> 来源归档（blog / 微信公众号）

- **标题：** Ego、遥操、仿真都在采，为什么具身数据飞轮仍会空转？
- **类型：** blog（数据飞轮 / 闭环工程）
- **作者：** 具身智能前沿（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s?__biz=Mzg5OTY3ODkzNg==&mid=2247494430&idx=1&sn=a7590eafcec35ef4e52cbb18e738392e
- **发表日期：** 2026-09-12
- **入库日期：** 2026-09-19
- **抓取方式：** wechat-article-for-ai（Camoufox）
- **原始抓取落盘：** [`sources/raw/wechat_jushen_qianyan_data_flywheel_idle_spin_2026-09-12.md`](../raw/wechat_jushen_qianyan_data_flywheel_idle_spin_2026-09-12.md)
- **一句话说明：** 飞轮≠堆数据源；最小闭环是「可执行起点—部署反馈—系统更新与回归测试」最短信号链，Ego/OXE/仿真是加速器而非门票。

## 对 wiki 的映射

| 主题 | wiki |
|------|------|
| 概念页 | [embodied-data-flywheel-minimal-closed-loop](../../wiki/concepts/embodied-data-flywheel-minimal-closed-loop.md) |
| 广义飞轮 | [data-flywheel](../../wiki/concepts/data-flywheel.md) |
| 系列专辑 | [embodied-data-collection-to-flywheel-album](../../wiki/overview/embodied-data-collection-to-flywheel-album.md) |
| RECAP / π* | [droid-policy-learning](../../wiki/entities/droid-policy-learning.md) 等 |

## 核心摘录（MVP）

- **空转判据：** 数据必须改变下一轮系统的具体部分；无结果/版本/场景信息的失败不可复现。
- **冷启动：** 观测—动作—时序—后果可靠连接；teleop 最直接，UMI/仿真/VLA 预训练可改路径但需真机验收。
- **部署反馈：** 记录完成与否、偏离步骤、人工纠正；RECAP 样本。
- **失败去处：** 分类到补数据/重训/修控制/改仿真/回归测试；需策略版本与任务条件。
- **加速器：** OXE、EgoMimic、仿真扩覆盖；非最小闭环必要条件。
