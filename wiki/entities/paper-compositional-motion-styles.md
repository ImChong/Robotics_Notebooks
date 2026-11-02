---
type: entity
project_id: compositional-motion-styles
tags:
  - paper
  - humanoid
  - locomotion
status: complete
updated: 2026-11-01
arxiv: "2610.24008"
project: https://www.proroklab.org/compositional-motion-styles/
related:
  - ../tasks/locomotion.md
  - ../concepts/motion-retargeting.md
  - ../concepts/sim2real.md
  - ../methods/reinforcement-learning.md
  - ../overview/embodied-station-2026-11-01-56-papers-technology-map.md
sources:
  - ../../sources/papers/compositional_motion_styles_arxiv_2610_24008.md
  - ../../sources/sites/compositional-motion-styles.md
  - ../../sources/blogs/wechat_embodied_station_56_papers_libero_dynamic_latency_eval_2026-11-01.md
summary: "Compositional Motion Styles（arXiv:2610.24008）：CODES从运动记录中发现风格，再让控制器组合风格与独立速度指令；跨仿真器和四足真机展示迁移，风格保真与指令跟踪之间仍存在取舍。"
---

# Compositional Motion Styles（arXiv:2610.24008）

**Compositional Motion Styles**（*Weakly Supervised Discovery of Motion Styles for Compositional Robot Control*，[arXiv:2610.24008](https://arxiv.org/abs/2610.24008)，[项目页](https://www.proroklab.org/compositional-motion-styles/)）来自 [具身智能小站 2026-11-01 盘点](../../sources/blogs/wechat_embodied_station_56_papers_libero_dynamic_latency_eval_2026-11-01.md)，属「人形与腿足控制」组。

## 一句话定义

**CODES从运动记录中发现风格，再让控制器组合风格与独立速度指令；跨仿真器和四足真机展示迁移，风格保真与指令跟踪之间仍存在取舍。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Compositional Motion Styles | Weakly Supervised Discovery of Motion Styles for Compositional Robot Control | 本文方法 / 基准的简称 |

## 为什么重要

- 速度指令之外，还能独立选择步态风格。
- 开源结论：**待发布**（步骤 2.5，2026-11-01）。项目页 HTTP 200，未列 GitHub，页面含 soon 字样（脚本扫描）。
- 同期同组工作见 [动态评测、技能编译与环境潜变量适配：具身智能小站 11-01 期 56 篇阅读坐标](../overview/embodied-station-2026-11-01-56-papers-technology-map.md)。

## 流程总览

```mermaid
flowchart TD
    A["多源运动录像：Kine2Go 策略回放（狗 / 马 / 其他机器人）+ Go2 遥操作动捕"] --> B["Retargeting Ti：统一状态坐标与采样率，并保证对目标机器人动力学可行"]
    B --> C["切 2 s 滑窗 X_i（L=120 帧）+ 粗标签 y_i（含 unknown）"]
    C --> D["M 个 TCN 编码器集成：重建 + 实例对比 + 粗标签吸引 / 分离损失"]
    D --> E["拼接得单位范数共识码 z̃_i，内积等于平均余弦亲和度"]
    E --> F["HDBSCAN 密度聚类（不使用标签，K 自动涌现）"]
    F --> G["风格组 C_k 与参考库 R（-1 为支持不足窗口）"]
    G --> H["冻结编码器，预存参考码 r_t"]
    H --> I["PPO 训练 style-conditioned 策略 π(a|o, u, r)"]
    J["任务指令 u_t：vx, vy, ωz"] --> I
    K["条件 AMP 判别器：参考 vs 策略转移"] --> L["门控奖励 r = r_aux + w_AMP·r_AMP + w_cmd·g(r_AMP)·r_cmd"]
    L --> I
    I --> M["Go2 执行：指令与风格参考可独立选择"]
```

图依据：论文 §III-A–D / Fig.1、式 (13)–(14)

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2610.24008](https://arxiv.org/abs/2610.24008) |
| **开源** | **待发布** |
| **要点** | CODES从运动记录中发现风格，再让控制器组合风格与独立速度指令；跨仿真器和四足真机展示迁移，风格保真与指令跟踪之间仍存在取舍。 |
| **文内指标** | 公众号未给具体数字，本页不引用。 |

## 源码运行时序图

**不适用（待发布）** — 项目页 HTTP 200，未列 GitHub，页面含 soon 字样（脚本扫描）。 不编造运行脚本；代码公开后再补。

## 实验与评测

- 公众号未给具体数字，本页不引用。
- **读法：** 本页为索引级摘要，依据公众号转述的官方摘要与仓库信息整理；逐项数字与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [动态评测、技能编译与环境潜变量适配：具身智能小站 11-01 期 56 篇阅读坐标](../overview/embodied-station-2026-11-01-56-papers-technology-map.md)；同 arXiv 不重复造页。

## 结论

**速度指令之外，还能独立选择步态风格：CODES从运动记录中发现风格，再让控制器组合风格与独立速度指令；跨仿真器和四足真机展示迁移，风格保真与指令跟踪之间仍存在取舍。**

1. 开源边界：**待发布** — 项目页 HTTP 200，未列 GitHub，页面含 soon 字样（脚本扫描）。
2. 核心机制：CODES从运动记录中发现风格，再让控制器组合风格与独立速度指令；跨仿真器和四足真机展示迁移，风格保真与指令跟踪之间仍存在取舍。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [locomotion](../tasks/locomotion.md)
- [motion-retargeting](../concepts/motion-retargeting.md)
- [sim2real](../concepts/sim2real.md)
- [reinforcement-learning](../methods/reinforcement-learning.md)
- [embodied-station-2026-11-01-56-papers-technology-map](../overview/embodied-station-2026-11-01-56-papers-technology-map.md)

## 参考来源

- [compositional_motion_styles_arxiv_2610_24008.md](../../sources/papers/compositional_motion_styles_arxiv_2610_24008.md)
- [wechat_embodied_station_56_papers_libero_dynamic_latency_eval_2026-11-01.md](../../sources/blogs/wechat_embodied_station_56_papers_libero_dynamic_latency_eval_2026-11-01.md)
- [arXiv:2610.24008](https://arxiv.org/abs/2610.24008)
- [arXiv PDF](https://arxiv.org/pdf/2610.24008)：已通读全文：摘要、§I–§IV（问题设定、retargeting、弱监督表示、共识几何与聚类、条件 AMP 组合控制、Go2 案例设置）；§V 评估结果仅浏览

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2610.24008)
- [项目页](https://www.proroklab.org/compositional-motion-styles/)
