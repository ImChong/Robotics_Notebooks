---
type: overview
tags: [loco-manipulation, contact-rich, category-hub, survey, whole-body-control]
status: complete
updated: 2026-07-03
summary: "Loco-Manip 接触专题 · 02 接触表示（8 篇）— 接触被写成标签、接触流、身体-手协同、主动感知、触觉力还是全身控制接口？"
related:
  - ./loco-manip-contact-technology-map.md
  - ./loco-manip-contact-category-01-contact-data.md
  - ./loco-manip-contact-category-04-post-contact-stability.md
  - ../entities/paper-scenebot.md
  - ../entities/paper-omnicontact-humanoid-loco-manipulation.md
sources:
  - ../../sources/blogs/wechat_embodied_ai_lab_loco_manip_contact_survey.md
---

# Loco-Manip 接触分类 02：接触怎么进入策略

> **图谱分类节点**：对应 [具身智能研究室 · Loco-Manip 接触专题](https://mp.weixin.qq.com/s/UjShbwl8p1h9ukymfiRNaw) 的 **02 接触表示** 段；总地图见 [接触五段链路技术地图](./loco-manip-contact-technology-map.md)。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| CF | Contact Flow | OmniContact 的稀疏关键体轨迹 + 接触信号接口 |
| WBC | Whole-Body Control | 全身协调控制，接触不宜只压在手部局部 |
| HOI | Human-Object Interaction | 人-物交互与接触规划 |
| EE | End-Effector | 末端执行器；CEER 等用 EE-根接口 |

## 核心问题

**接触被写成什么信号？** 接触标签、接触流、身体-手协同、头手主动感知、触觉力与全身控制接口代表不同路线，尚无统一答案。

## 本组工作（8 篇）

| 工作 | Wiki 实体（复用） | 接口类型 |
|------|-------------------|----------|
| SceneBot | [paper-scenebot](../entities/paper-scenebot.md) | 逐身体部位接触标签 + 场景交互图 |
| OmniContact | [paper-omnicontact-humanoid-loco-manipulation](../entities/paper-omnicontact-humanoid-loco-manipulation.md) | 接触流（建立/保持/转移/断开时序） |
| CoorDex | [paper-coordex-dexterous-humanoid-loco-manipulation](../entities/paper-coordex-dexterous-humanoid-loco-manipulation.md) | 身体-手协同残差策略 |
| HALOMI | [paper-halomi-humanoid-loco-manipulation](../entities/paper-halomi-humanoid-loco-manipulation.md) | 头手目标与主动感知 |
| WT-UMI | [paper-loco-manip-07-wt-umi](../entities/paper-loco-manip-07-wt-umi.md) | 全身触觉图像 + 力监督规划 |
| CEER | [paper-motion-cerebellum-ceer](../entities/paper-motion-cerebellum-ceer.md) | 末端执行器-根部接口 |
| Pro-HOI | [paper-loco-manip-161-074-pro-hoi](../entities/paper-loco-manip-161-074-pro-hoi.md) | 根节点轨迹 |
| CWI | [paper-cwi-composite-humanoid-whole-body-imitation](../entities/paper-cwi-composite-humanoid-whole-body-imitation.md) | 复合式全身模仿 |

## 接口拆解（策展）

1. **SceneBot**：接触标签让参考动作带物理含义（踩台阶、双手持续接触箱子）。
2. **OmniContact**：长程 loco-manip 的技能衔接里 **接触时序** 最易丢失，接触流显式写出。
3. **CoorDex**：边走边抓时失败常来自 **身体支撑未跟上** 手部接触。
4. **HALOMI**：头部视角与手部目标是接触建立的 **前置条件**。
5. **WT-UMI**：接触质量、受力与分布 **直接改变** 后续动作。
6. **CEER / Pro-HOI / CWI**：接触属于 **全身控制接口**，不能只压在手部局部。

## 关联页面

- [SceneBot](../entities/paper-scenebot.md)
- [OmniContact](../entities/paper-omnicontact-humanoid-loco-manipulation.md)
- [运动小脑 · I 柔顺接触](./motion-cerebellum-category-09-compliance-contact.md)

## 参考来源

- [wechat_embodied_ai_lab_loco_manip_contact_survey.md](../../sources/blogs/wechat_embodied_ai_lab_loco_manip_contact_survey.md)

## 推荐继续阅读

- [OmniContact 数据集](../../sources/datasets/omnicontact-dataset.md)
- [全身协调](../concepts/whole-body-coordination.md)
