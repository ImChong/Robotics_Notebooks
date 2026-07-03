---
type: overview
tags: [loco-manipulation, contact-rich, category-hub, survey, data-pipeline]
status: complete
updated: 2026-07-03
summary: "Loco-Manip 接触专题 · 01 接触数据（8 篇）— 带物体状态、场景约束、接触时序与本体可执行性的交互数据从哪来？"
related:
  - ./loco-manip-contact-technology-map.md
  - ./loco-manip-contact-category-02-contact-representation.md
  - ../concepts/motion-retargeting.md
  - ../tasks/loco-manipulation.md
sources:
  - ../../sources/blogs/wechat_embodied_ai_lab_loco_manip_contact_survey.md
---

# Loco-Manip 接触分类 01：接触数据从哪来

> **图谱分类节点**：对应 [具身智能研究室 · Loco-Manip 接触专题](https://mp.weixin.qq.com/s/UjShbwl8p1h9ukymfiRNaw) 的 **01 接触数据** 段；总地图见 [接触五段链路技术地图](./loco-manip-contact-technology-map.md)。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| HOI | Human-Object Interaction | 人-物交互，含接触与物体运动 |
| Retarget | Motion Retargeting | 将人体/异源运动映射到机器人关节空间 |
| UMI | Universal Manipulation Interface | 轻量遥操作/示范接口范式（HumanoidUMI 借名） |
| Ego-Exo | Egocentric + Exocentric | 第一视角与外视角联合采集 |

## 核心问题

**策略能学到什么，取决于数据里有没有真实的接触结构。** 普通动作数量再多，若只剩人体姿态而无物体状态、场景约束、接触时序与本体可执行性，策略很难凭空学出稳定接触。

## 本组工作（8 篇）

| 工作 | Wiki 实体（复用） | 文内角色 |
|------|-------------------|----------|
| OmniRetarget | [paper-loco-manip-161-114-omniretarget](../entities/paper-loco-manip-161-114-omniretarget.md) | 重定向时保持脚底支撑、手物接触与场景约束 |
| HumanX | [paper-loco-manip-161-112-humanx](../entities/paper-loco-manip-161-112-humanx.md) | 人类视频 → 可学习人形交互技能 |
| HDMI | [paper-loco-manip-161-110-hdmi](../entities/paper-loco-manip-161-110-hdmi.md) | RGB 视频恢复人-物轨迹 |
| SUGAR | [paper-loco-manip-161-076-sugar](../entities/paper-loco-manip-161-076-sugar.md) | 运动轨迹与接触先验 → 技能优化与蒸馏 |
| Human-as-Humanoid | [paper-human-as-humanoid](../entities/paper-human-as-humanoid.md) | Ego-Exo 人类视频 → 人形动作标签 |
| HumanoidUMI | [paper-humanoidumi](../entities/paper-humanoidumi.md) | 轻量跨本体示范 → 全身控制器 |
| HumanoidMimicGen | [paper-humanoidmimicgen](../entities/paper-humanoidmimicgen.md) | 全身规划合成 loco-manip 示范 |
| VLK | [paper-vlk-synthetic-loco-manipulation](../entities/paper-vlk-synthetic-loco-manipulation.md) | 重建场景中合成视觉-语言-运动三元组 |

## 策展判断

- **共同难题：** 视频里有丰富接触，却没有天然的机器人动作标签。
- **Human-as-Humanoid / HumanoidUMI：** 降低采集成本，也把 **跨本体误差** 带入控制链路。
- **HumanoidMimicGen / VLK：** 偏 **数据制造**——规划或重建场景批量产出可训练轨迹。

## 关联页面

- [接触表示](./loco-manip-contact-category-02-contact-representation.md)
- [运动重定向](../concepts/motion-retargeting.md)
- [161 篇 · 05 人类视频](./loco-manip-161-category-05-mocap-human-video.md)

## 参考来源

- [wechat_embodied_ai_lab_loco_manip_contact_survey.md](../../sources/blogs/wechat_embodied_ai_lab_loco_manip_contact_survey.md)

## 推荐继续阅读

- [OmniRetarget 数据集](../entities/omniretarget-dataset.md)
- [Loco-Manip 8 篇 · 第一视角数据](./loco-manip-category-01-egocentric-data.md)
