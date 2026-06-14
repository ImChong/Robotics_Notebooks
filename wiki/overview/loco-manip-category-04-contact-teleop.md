---
type: overview
tags: [loco-manipulation, tactile, teleoperation, category-hub, survey]
status: complete
updated: 2026-06-14
summary: "Loco-Manip 8 篇周报 · 04 触觉与跨本体遥操作（2 篇）— 力/触觉如何补视觉盲区（WT-UMI），MPC 重定向如何让 teleop 成为跨机器人数据入口（X-OP）？"
related:
  - ./loco-manip-8-papers-technology-map.md
  - ./loco-manip-category-03-command-controller.md
  - ../tasks/teleoperation.md
  - ../entities/paper-loco-manip-07-wt-umi.md
  - ../entities/paper-loco-manip-08-x-op.md
  - ../entities/paper-hrl-stack-03-omniretarget.md
sources:
  - ../../sources/blogs/wechat_embodied_ai_lab_loco_manip_8_papers_survey.md
  - ../../sources/papers/loco_manip_8_papers_catalog.md
---

# Loco-Manip 分类 04：触觉与跨本体遥操作

> **图谱分类节点**：对应 [具身智能研究室 · Loco-Manip 8 篇周报](https://mp.weixin.qq.com/s/Ez87ljBYmCyIpLKjMjEyaQ) 的 **04 触觉与跨本体遥操作** 分组；总地图见 [Loco-Manip 8 篇技术地图](./loco-manip-8-papers-technology-map.md)。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| UMI | Universal Manipulation Interface | 便携数据采集接口族（本组 WT-UMI 强调触觉扩展） |
| MPC | Model Predictive Control | 滚动优化，用于跨本体动作重定向 |
| WBC | Whole-Body Control | 全身协调控制层 |

## 核心问题

**真实 loco-manip 数据如何补接触、并跨机器人复用？** 视觉难判压住/滑动/偏心载荷；单机体 teleop 难形成 **跨平台数据资产**。

## 本组论文（2 篇）

| # | 工作 | Wiki 实体 | Source |
|---|------|-----------|--------|
| 07 | WT-UMI | [paper-loco-manip-07-wt-umi.md](../entities/paper-loco-manip-07-wt-umi.md) | [source](../../sources/papers/loco_manip_survey_07_wt_umi.md) |
| 08 | X-OP | [paper-loco-manip-08-x-op.md](../entities/paper-loco-manip-08-x-op.md) | [source](../../sources/papers/loco_manip_survey_08_x_op.md) |

## 关联页面

- [Teleoperation](../tasks/teleoperation.md)
- [Contact-Rich Manipulation](../concepts/contact-rich-manipulation.md)
- [OmniRetarget](../entities/paper-hrl-stack-03-omniretarget.md)

## 参考来源

- [wechat_embodied_ai_lab_loco_manip_8_papers_survey.md](../../sources/blogs/wechat_embodied_ai_lab_loco_manip_8_papers_survey.md)
- [loco_manip_8_papers_catalog.md](../../sources/papers/loco_manip_8_papers_catalog.md)

## 推荐继续阅读

- [WT-UMI 项目页](https://wt-umi.github.io/WTUMI/)
