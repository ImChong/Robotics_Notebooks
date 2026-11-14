---
type: entity
project_id: demostream-batched-tamp
tags: [paper, robotics, ingest-2026-11]
status: draft
updated: 2026-11-14
arxiv: "2610.22020"
project: https://demostream.github.io/
related:
  - ./paper-coda-coevolving-data-policy.md
  - ./paper-submillisecond-reflex-engine.md
original:
  - ../../sources/papers/demostream-batched-tamp_arxiv_2610_22020.md
sources:
  - ../../sources/papers/demostream-batched-tamp_arxiv_2610_22020.md
  - ../../sources/sites/demostream-batched-tamp.md
summary: "批量优化 TAMP 生成更易模仿的示范"
---

# DemoStream：批量 TAMP 优化自动示范数据

## 一句话定义

**使用 GPU 批量约束采样与任务运动规划搜索，在生成自动化示范的同时优化轨迹一致性，便于行为克隆学习。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| TAMP | Task and Motion Planning | 任务与运动联合规划 |
| BC | Behavior Cloning | 行为克隆 |
| GPU | Graphics Processing Unit | 并行计算处理器 |

## 为什么重要

批量优化 TAMP 生成更易模仿的示范，关注现有控制、数据或机器人学习方法在复杂任务中的失效点。这里单独建论文实体，避免与其他论文共用详情页；当前是待完成全文与源码复核的草稿。

## 核心机制

任务约束和几何约束进入批量 TAMP；batched streams、计算图优化并行探索可行解，按任务空间代价选择动作与轨迹，再通过行为克隆训练视觉运动策略。

## 方法流程（待全文逐节复核）

以下流程仅依据已核对的论文 HTML 方法描述，不代替源代码调用图。

```mermaid
flowchart LR
  goal[任务与场景几何] --> tamp[批量 TAMP 采样与规划]
  tamp --> cost[轨迹代价优化]
  cost --> demos[自动执行与示范筛选]
  demos --> bc[行为克隆训练]
  bc --> policy[视觉运动策略评测]
```

## 源码运行时序图

**待核实，不绘制未经代码阅读确认的时序图。** 目前未确认可运行官方源码，待项目源文件发布或访问恢复后补充。

## 实验与评测

真机示范收集平均约每小时 66 条；论文图示 75 条真机示范训练得到 67% 的 visuomotor 策略成功率。物体复位和成功标注仍需要人工支持，不代表完全无人工采数。

## 局限与工程实践

- TAMP 优化数据的质量受约束模型、碰撞判定与成本函数影响；示范可行性不保证克隆策略成功。
- **源码状态：** 项目页提供论文与视频；代码入口未列出可用仓库链接，暂不声称源码可复现。
- **原文状态：** 本轮已核查 arXiv 部分章节，尚未通读全部正文/附录，保持 draft，不宣称健康度满分。

## 结论

**自动示范不只要能规划成功，还要优化动作空间和轨迹方差，否则学习效果可能很差。**

1. 使用与论文一致的任务、硬件和评测协议复验，不跨实验口径比较数据。
2. 把方法改善的指标与额外的数据、计算或控制成本分开记录。
3. 对未经独立核查的复现入口，暂不声称完成验证。

## 关联页面

- [CoDA：失败驱动的数据引擎与动作策略协同进化](./paper-coda-coevolving-data-policy.md)
- [亚毫秒反射：连续人机物理交互的无锁控制切换](./paper-submillisecond-reflex-engine.md)

## 参考来源

- [论文归档](../../sources/papers/demostream-batched-tamp_arxiv_2610_22020.md)
- [官方项目页归档](../../sources/sites/demostream-batched-tamp.md)

## 推荐继续阅读

- [arXiv 原文](https://arxiv.org/abs/2610.22020)
- [官方项目页](https://demostream.github.io/)
