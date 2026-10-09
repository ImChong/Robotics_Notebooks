---
type: overview
tags: [loco-manipulation, humanoid, category-hub, survey]
status: complete
updated: 2026-10-09
summary: "人形 Loco-Manip 161 篇 · 05 动捕、人类视频与交互动作规划（11 篇）— 人类动作数据转成机器人可用的运动和交互先验。"
related:
  - ./humanoid-loco-manip-161-papers-technology-map.md
  - ../entities/cartwheel-comic.md
sources:
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md
  - ../../sources/papers/humanoid_loco_manip_161_catalog.md
  - ../../sources/sites/cartwheel-comic.md
  - ../../sources/repos/cartwheel-mcp.md
---

# Loco-Manip 161 分类 05：动捕、人类视频与交互动作规划

> **图谱分类节点**：**05 动捕、人类视频与交互动作规划**；总地图见 [人形 Loco-Manip 161 篇技术地图](./humanoid-loco-manip-161-papers-technology-map.md)。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Loco-Manip | Loco-Manipulation | 行走与操作动力学耦合的全身任务 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制层 |
| VLA | Vision-Language-Action | 视觉-语言-动作多模态策略 |
| WM | World Model | 学习环境动态以供想象/规划的世界模型 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |

## 核心问题

人类动作数据转成机器人可用的运动和交互先验

## 本组论文（11 篇）

| # | 工作 | Wiki 实体 |
|---|------|-----------|
| 109 | FALCON | [paper-loco-manip-161-109-falcon](../entities/paper-loco-manip-161-109-falcon.md) |
| 110 | HDMI | [paper-hrl-stack-06-hdmi](../entities/paper-hrl-stack-06-hdmi.md) |
| 111 | HITTER | [paper-loco-manip-161-111-hitter](../entities/paper-loco-manip-161-111-hitter.md) |
| 112 | HumanX | [paper-loco-manip-161-112-humanx](../entities/paper-hrl-stack-05-humanx.md) |
| 113 | Humanoid | [paper-loco-manip-161-113-humanoid](../entities/paper-amp-survey-13-humanoid_goalkeeper.md) |
| 114 | OmniRetarget | [paper-loco-manip-161-114-omniretarget](../entities/paper-hrl-stack-03-omniretarget.md) |
| 115 | ResMimic | [paper-loco-manip-161-115-resmimic](../entities/paper-resmimic.md) |
| 116 | WoCoCo | [paper-loco-manip-161-116-wococo](../entities/paper-loco-manip-161-116-wococo.md) |
| 117 | 多阶段强化学习的人形全身羽毛球 | [paper-loco-manip-161-117-n117](../entities/paper-loco-manip-161-117-n117.md) |
| 118 | 腿式机械手全身动态投掷 | [paper-loco-manip-161-118-n118](../entities/paper-loco-manip-161-118-n118.md) |
| 119 | 迈向多样化人形乒乓球：具有预测增强的统一强化学习 | [paper-loco-manip-161-119-n119](../entities/paper-loco-manip-161-119-n119.md) |

## 工程动捕工具（非论文）

- [Cartwheel Comic](../entities/cartwheel-comic.md) — 单目视频人体动捕与足部接触估计；可将人类动作导出后接入动画或机器人重定向流程。官方开放的是 API/MCP 接入方式，模型本身为云端服务。

## 关联页面

- [人形 Loco-Manip 161 篇技术地图](./humanoid-loco-manip-161-papers-technology-map.md)
- [Loco-Manipulation 任务页](../tasks/loco-manipulation.md)

## 参考来源

- [wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md)
- [humanoid_loco_manip_161_catalog.md](../../sources/papers/humanoid_loco_manip_161_catalog.md)
- [Cartwheel Comic 产品资料](../../sources/sites/cartwheel-comic.md)
- [Cartwheel MCP 源码资料](../../sources/repos/cartwheel-mcp.md)

## 推荐继续阅读

- [运动小脑 64 篇技术地图](./humanoid-motion-cerebellum-technology-map.md)
