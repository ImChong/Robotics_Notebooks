# gentle-humanoid-axell-top

> 来源归档（ingest）

- **标题：** GentleHumanoid 项目页（gentle-humanoid.axell.top）
- **类型：** site
- **链接：** https://gentle-humanoid.axell.top/
- **论文：** [arXiv:2511.04679](https://arxiv.org/abs/2511.04679)
- **入库日期：** 2026-05-19
- **一句话说明：** GentleHumanoid 官方展示站：浏览器在线 demo、真实人机/人物交互视频、与 baseline 的接触力/气球任务对比，以及阻抗集成运动跟踪方法图解。

## 核心摘录

### 公开能力叙事

- **上半身柔顺**：肩–肘–腕运动链协调响应；**统一弹簧式交互力**（resistive + guiding）；**可调力阈值**（部署示例：握手 5 N、拥抱 10 N、坐–站辅助 15 N）。
- **真机**：Unitree G1；仿真 + 实物验证「通用 whole-body policy + 上半身 compliance」。
- **浏览器 Demo**：站点提供一键在线体验 GentleHumanoid（具体运行时依赖以页面为准）。

### 演示任务（站点分类）

| 类别 | 示例 |
|------|------|
| 通用策略 | 任意参考运动 + compliance；力限 5 N / 15 N 对比 |
| 人机交互 | 握手、坐–站辅助、掰手腕 |
| 人物体交互（视频参考） | 枕头、篮子、袋子、气球；单目 RGB → [PromptHMR](https://yufu-wang.github.io/phmr-page/) → [GMR](https://github.com/YanjieZe/GMR) → 跟踪 |
| 视觉自主拥抱 | 帽式 mocap 定位 + 头部 RGB → 人体 shape 估计 → 个性化拥抱姿态 |

### 实验展示

- **人台拥抱测力**：腰侧 40 路电容 taxel 压力垫；对齐 / 错位拥抱下 GentleHumanoid 接触压更均匀，baseline 峰值升高或不稳定。
- **气球操作**：baseline 易失败，本方法可安全操控可变形物体。

## 对 wiki 的映射

- [GentleHumanoid（方法页）](../../wiki/methods/gentlehumanoid-motion-tracking.md)
- [论文 sources 摘录](../papers/gentlehumanoid_upper_body_compliance.md)
