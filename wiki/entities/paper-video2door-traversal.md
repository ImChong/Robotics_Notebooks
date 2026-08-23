---
type: entity
tags: [paper, loco-manipulation, real-to-sim-to-real, door-traversal, mobile-manipulation, sjtu]
status: complete
updated: 2026-08-22
arxiv: "2608.20251"
related:
  - ../tasks/loco-manipulation.md
  - ../concepts/sim2real.md
  - ../tasks/manipulation.md
  - ./paper-smpc2rl-loco-manipulation.md
  - ../overview/video-contact-control-10-papers-technology-map.md
sources:
  - ../../sources/papers/video2door_traversal_arxiv_2608_20251.md
  - ../../sources/sites/video2door-traversal.md
  - ../../sources/blogs/wechat_embodied_station_video_contact_control_10_papers_2026-08-22.md
summary: "Video2DoorTraversal（arXiv:2608.20251）：单 RGB 视频 → DoorTwin 门孪生 → ArticuACT 双深度闭环穿门；五扇真门 96.57%，未见门 80.95%；代码待发布。"
---

# Video2DoorTraversal

**Video2DoorTraversal: Push Door Traversal via Simulated Door Twins**（[arXiv:2608.20251](https://arxiv.org/abs/2608.20251)，[项目页](../../sources/sites/video2door-traversal.md)）——上海交通大学（SJTU）；山东大学；NeoWa Robotics。

## 一句话定义

**把一次门观察变成可反复试错的仿真资产，再学全身穿门闭环。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| R2S2R | Real-to-Sim-to-Real | 单视频重建仿真再回真机 |
| RGB-D | Red-Green-Blue + Depth | 双深度策略输入 |
| LoCo | Loco-Manipulation | 移动操作联合控制 |

## 为什么重要

- 纳入 [具身智能小站 2026-08-22 十篇盘点](../../sources/blogs/wechat_embodied_station_video_contact_control_10_papers_2026-08-22.md) 的「视频→接触→控制→VLA 持续学习」主线。
- 开源状态（入库日）：**待发布**。

## 核心信息

| 项 | 内容 |
|----|------|
| **机构** | 上海交通大学（SJTU）；山东大学；NeoWa Robotics |
| **出处** | arXiv:2608.20251（2026-08） |
| **开源** | **待发布** |

### 流程总览

```mermaid
flowchart LR
  vid[单段 RGB 视频] --> twin[DoorTwin 关节门孪生]
  twin --> sim[仿真闭环技能程序 + 修复 rollout]
  sim --> train[ArticuACT 训练]
  train --> real[轮足移动操作真机]
```

## 评测

| 项 | 内容 |
|----|------|
| **真机训练门** | 五扇真实门，平均成功率 **96.57%** |
| **未见门 zero-shot** | 结构相近未见门 **80.95%** |
| **单次穿越耗时** | 全流程约 **13 s** |
| **策略输入** | ArticuACT 双深度 + 本体相机条件化，输出底盘/臂/夹爪 |

- 数据出处：[ingest 摘录「ArticuACT」](../../sources/papers/video2door_traversal_arxiv_2608_20251.md) 与 [具身智能小站 10 篇盘点](../../sources/blogs/wechat_embodied_station_video_contact_control_10_papers_2026-08-22.md)。
- 评测口径注意：训练门与未见门均为 **结构相近** 的门类，跨结构（如推拉/闭门器差异大）泛化未在摘录中给出。

## 结论

**门操作的关键是把一次观察变成可迭代试错的仿真孪生体，而非端到端黑盒策略。**

- DoorTwin 实例对齐 + 关节 + 可仿真外观
- 仿真 agent 将失败 rollout 修复为可执行演示
- ArticuACT 双深度 + 本体相机条件化
- 代码 **Coming soon**（项目页 2026-08-22）

## 源码运行时序图

**不适用**（截至 **2026-08-22**）：官方训练/推理入口尚未公开发布。

## 与其他页面的关系

- [loco-manipulation](../tasks/loco-manipulation.md)
- [sim2real](../concepts/sim2real.md)
- [manipulation](../tasks/manipulation.md)
- [paper-smpc2rl-loco-manipulation](./paper-smpc2rl-loco-manipulation.md)
- [视频–接触–控制 10 篇技术地图](../overview/video-contact-control-10-papers-technology-map.md)

## 参考来源

- [video2door_traversal_arxiv_2608_20251](../../sources/papers/video2door_traversal_arxiv_2608_20251.md)
- [video2door-traversal](../../sources/sites/video2door-traversal.md)
- [wechat_embodied_station_video_contact_control_10_papers_2026-08-22](../../sources/blogs/wechat_embodied_station_video_contact_control_10_papers_2026-08-22.md)

## 推荐继续阅读

- [arXiv:2608.20251](https://arxiv.org/abs/2608.20251)
