---
type: query
tags: [iros, survey, vla, embodied-ai, world-model, system-engineering]
status: complete
updated: 2026-09-25
summary: "从 AI科技评论 IROS 2026（1933 篇）策展文提炼：机器人学进入系统问题时代——AI 嵌入而非取代传统模块；VLA 竞争转向效率/几何/记忆/系统壳；WM 量少但更贴控制。"
related:
  - ../overview/iros-2026-six-trends-technology-map.md
  - ../methods/vla.md
  - ../concepts/world-action-models.md
  - ../tasks/loco-manipulation.md
  - ../entities/paper-geovla.md
  - ../entities/paper-shallow-pi.md
sources:
  - ../../sources/blogs/wechat_ai_tech_review_iros_2026_six_trends_2026-09-25.md
  - ../../sources/papers/iros_2026_ai_tech_review_six_trends_cited_papers_catalog.md
---

# IROS 2026：1933 篇论文释放的六条信号

> **Query 产物**：用户 ingest [AI科技评论 · 六变化](https://mp.weixin.qq.com/s/XvdbbidbJKBszMQwVzvD0A) 后，将文内 **多标签统计 + 六段论证** 沉淀为可检索结论；细节论文见 [技术地图](../overview/iros-2026-six-trends-technology-map.md) 与 [catalog](../../sources/papers/iros_2026_ai_tech_review_six_trends_cited_papers_catalog.md)。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作统一策略 |
| WM | World Model | 预测未来状态/观测的模型 |
| TAMP | Task and Motion Planning | 符号任务 + 连续运动规划 |
| FM | Foundation Model | 大基础模型（VLM 等） |
| SR | Success Rate | 任务成功率 |

## 1. 核心结论（可行动）

1. **没有「AI 吃掉机器人学」**：Learning×Manipulation/Perception/Planning/Control 交叉篇数均在 **200+** 量级（多标签，不可加总）。
2. **VLA 主线分化**：~82 篇明确 VLA；热点从 **更大** 转向 **更快（Shallow-π）、更几何（GeoVLA）、更视角鲁棒（AnyCamVLA）、更长记忆、更完整系统（RoboBRIDGE 类）**。
3. **中间层回来**：Reasoning/Memory ~119 篇 — 长程任务需要 **3D 一致记忆 + 显式推理/符号约束**，端到端 alone 不够。
4. **操作仍是落地主战场**：Manipulation ~520；触觉子线 ~91 — **TacVLA（实时触觉） vs HapticVLA（训练用触觉、推理免触觉）** 代表两种部署哲学。
5. **人形问题结构变了**：loco-manipulation ~89 — 负载改变步态、托盘/网球等 **whole-body** 任务不能「先走再贴手」。
6. **WM 热但少（19 篇）**：价值在 **动作条件下的可信未来**（深度去噪、插接、2.5D 动态），不是视频好看。

## 2. 对选型/研究的读法

| 你在做什么 | 优先跟哪条 IROS 信号 |
|------------|----------------------|
| 边缘部署 VLA | Shallow-π 类 **层蒸馏** + [VLA 部署指南](./vla-deployment-guide.md) |
| 固定/变化相机 | AnyCamVLA（推理期） vs GeoVLA（训练期 3D） |
| 长程 household | Memory/3D GS/Temporal KV — catalog §03 |
| Contact-rich | VTAP 硬件 + Tac/Haptic VLA + [world-action-models](../concepts/world-action-models.md) |
| 人形搬物 | ULTRA / SteadyTray / DreamMimic 索引 + [loco-manipulation](../tasks/loco-manipulation.md) |

## 3. 局限

- 统计来自 **公众号二次策展**，非官方 PC 程序；数字用于 **趋势**，不用于精确引文。
- 多数举例论文 **尚未** 独立 wiki 实体，见 catalog「待建」；勿与已有 arXiv 节点重复造页。

## 关联页面

- [iros-2026-six-trends-technology-map](../overview/iros-2026-six-trends-technology-map.md)
- [GeoVLA](../entities/paper-geovla.md) · [Shallow-π](../entities/paper-shallow-pi.md)

## 参考来源

- [wechat_ai_tech_review_iros_2026_six_trends_2026-09-25.md](../../sources/blogs/wechat_ai_tech_review_iros_2026_six_trends_2026-09-25.md)
