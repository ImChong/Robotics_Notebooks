---
type: entity
project_id: dynamic-creatures-hugo
project: https://www.dynamiccreatures.com/
tags: [company, robot, mobile-robot, human-robot-interaction, entertainment, hospitality, character-robot, snowjay]
status: complete
updated: 2026-10-14
related:
  - ./boston-dynamics.md
  - ./humanoid-robot.md
  - ../methods/humanoid-voice-interaction.md
sources:
  - ../../sources/sites/dynamic-creatures-hugo.md
summary: "Dynamic Creatures 的 Hugo 是首款公开的活体角色机器人原型，使用 SnowJay 平台并配有 43 个执行器，面向主题公园、度假村、赌场与零售等访客场景；官方称其情境理解和实时回应能力仍在开发中。"
---

# Dynamic Creatures：Hugo 与 SnowJay

## 一句话定义

**Hugo** 是 Dynamic Creatures 公布的首款“活体角色机器人”原型：以拟人化角色设计结合移动机器人能力，计划在主题公园、度假村、赌场与零售环境中与访客进行面对面互动；其机器人软件平台为公司自研的 **SnowJay**。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| AI | Artificial Intelligence | SnowJay 页面所称的机器人 AI 能力集合 |
| HRI | Human-Robot Interaction | 面向游客的面对面互动与行为响应 |
| OS | Operating System | 官方用语将 SnowJay 称为机器人平台 / 操作系统 |
| RL | Reinforcement Learning | SnowJay 页面提及的学习式动作技能背景；未披露 Hugo 训练细节 |

## 为什么重要

- **机器人从完成任务转向扮演角色：** Hugo 面向游客互动，角色表达、身体动作与环境移动共同构成体验；它代表娱乐与酒店业中介于固定机械动画和任务型服务机器人之间的一种产品方向。
- **重视全身表达与交互闭环：** 官方将 SnowJay 描述为协调移动、感知与行为的平台，并把手势、全身动作、表情和对访客的回应视为角色体验的一部分。
- **适合观察商用人机交互边界：** 目标场景包含主题公园、度假村、赌场与零售等高人流环境，后续能否稳定导航、理解互动并安全回应，比单条展示视频更能说明其部署成熟度。

## 核心信息

| 字段 | 内容 |
|------|------|
| **公司** | Dynamic Creatures；Boston Dynamics 官方酒店与娱乐合作伙伴 |
| **机器人** | Hugo，首款公开的活体角色机器人原型 |
| **执行器** | 官方公告称配有 43 个执行器；未披露具体关节 / 执行器配置 |
| **平台** | SnowJay；公司称其协调运动、感知、行为与技能执行 |
| **目标场景** | 主题公园、度假村、赌场与零售；公司整体市场还包括酒店、景点、博物馆、邮轮与现场娱乐 |
| **公开状态** | 官方资料未列 Hugo / SnowJay 的公开代码仓库、数据集或规格书链接 |

## 系统与能力（按官方公开信息）

SnowJay 不只是动作播放器。公司平台页将其定位为整合运动、感知和行为的机器人控制平台，列出实时策略与控制器推理、技能之间的切换、行为编排、日志与诊断、系统健康监控，以及安全停止和受控关机等能力。平台还支持以动捕或动画创建新动作，并描述了自然移动、全身角色手势、摔倒恢复和多机器人编舞等技能方向。

这些是 **SnowJay 平台层** 的公开能力说明，不能逐项推定 Hugo 已具备。针对 Hugo 的公告只确认其原型、43 个执行器、SnowJay 平台与目标场景；公司明确表示 Hugo 理解互动情境并实时回应的能力仍在开发。

## 工程实践与开放状态

- **场景拆解：** 实际部署需分别验证在人群中移动、访客接近时的速度 / 停止策略、角色动作与底盘运动的协调、对话或手势触发，以及故障和急停处理。
- **平台与本体分层：** SnowJay 是公司的软件与行为平台；Hugo 是一个角色原型。官方资料未说明 Hugo 是否使用 Boston Dynamics 的现成底盘，也没有公开其机构图、传感器配置、导航栈或控制频率。
- **复现边界：** 截至 2026-10-14，官方公告、官网、SnowJay 介绍页和新闻页均未提供 Hugo / SnowJay 的代码或数据下载入口。若要评估开发者可复现性，应等待官方资源链接，而不要把平台功能介绍当作已公开 SDK 或开源实现。

## 局限与风险

目前公开信息以公司介绍与展示为主，未见可核查的连续运行时间、拥挤场景导航成功率、交互延迟、系统安全评估或真实部署数据。43 个执行器是官方披露的数量，不能单独推出自由度、负载、动作范围或硬件平台。面向近距离访客的机器人还需重点评估碰撞风险、急停机制、儿童接近时的行为约束，以及服装 / 角色外壳对感知和散热的影响；这些细节尚未由公开资料回答。

## 关联页面

- [Boston Dynamics](./boston-dynamics.md) — Dynamic Creatures 的官方合作伙伴与技术生态背景
- [人形机器人](./humanoid-robot.md) — 对照双足平台的本体与开发流程；Hugo 的公开资料尚不足以确认其完整机构形态
- [人形语音交互](../methods/humanoid-voice-interaction.md) — 面向游客的语音交互设计可参考的相关方向

## 参考来源

- [Hugo 与 SnowJay 官方资料归档](../../sources/sites/dynamic-creatures-hugo.md)

## 推荐继续阅读

- [Dynamic Creatures 官方 SnowJay 平台页](https://www.dynamiccreatures.com/snowjay)
- [Dynamic Creatures 官方新闻页](https://www.dynamiccreatures.com/news)
