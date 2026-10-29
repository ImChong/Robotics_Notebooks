---
type: entity
project_id: every-joule
project: https://iamtechartist.com/every-joule/
tags:
  - education
  - visualization
  - threejs
  - webgl
  - llm-agents
status: complete
topic: [ecosystem, ai-agents]
updated: 2026-10-27
related:
  - ./claude-shaped-science.md
  - ./threejs-game-skills.md
sources:
  - ../../sources/sites/every-joule.md
summary: "Every Joule 是 Techartist 制作的 GCSE/IGCSE 能量主题互动 3D 物理书；作者称使用 Claude Opus 5.5 与 Three.js 构建，官方站已提供十课内容入口。"
---

# Every Joule：互动 3D 物理课本

**Every Joule** 是 Techartist 制作的浏览器互动物理书，用十节课程讲解 GCSE/IGCSE 物理中的能量主题；作者称项目使用 Claude Opus 5.5 与 Three.js 构建。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| GCSE | General Certificate of Secondary Education | 英格兰等地区的中学阶段资格考试 |
| IGCSE | International General Certificate of Secondary Education | 国际版中学阶段资格课程 |
| AI | Artificial Intelligence | 作者使用的辅助开发技术之一 |
| WebGL | Web Graphics Library | 浏览器图形 API；项目 3D 体验需要图形加速 |

## 为什么重要

Every Joule 是一个可直接体验的交互式 STEM 教学案例：它把能量主题从静态文字和图示转换为可操作的 3D 场景，并以预测、测试、揭示答案的节奏组织学习。对于机器人研究者，它可作为理解 Three.js 教学演示和 AI 辅助前端制作的参照；它本身不是机器人仿真器或控制工具。

## 核心原理

官方站将全章拆成十课，覆盖能量储存与转化、动能和势能、弹性势能、耗散、比热容、功率、效率、电网和食物能量等主题。作者介绍的互动节奏是让学习者先作预测、在 3D 场景中测试，再翻开答案并获得印章。项目页还列出教师说明和课程来源，支持教学者将互动演示与课文配合使用。

## 工程实践

| 项目 | 当前可确认内容 |
|------|------|
| 技术 | 作者在 X 帖子中称使用 Claude Opus 5.5 与 Three.js |
| 课程结构 | 官方站描述为 GCSE/IGCSE 能量主题十课 |
| 体验入口 | 官方站提供整套书入口；GitHub Pages 提供第 1 课 |
| 运行条件 | 浏览器需支持 JavaScript 与 WebGL / 图形加速 |
| 源码复用 | 截至 2026-10-27 未发现官方公开源码仓库链接；不能仅凭在线页面推断源码可复用 |

## 局限与风险

- 官方站展示的是教学资源，不是经独立评估的物理课程或学习效果研究；公式、数值与课程适配仍应对照教师材料和权威课程标准。
- 网页可体验不等于源码开源。当前未发现官方仓库链接，二次开发应先确认授权；第三方复刻指南不等同于原作者代码或授权。
- 3D 效果依赖浏览器图形能力，低性能设备或未启用图形加速时可能无法正常运行。

## 关联页面

- [Claude-shaped science](./claude-shaped-science.md) — 关于 AI 能力与实际工作匹配的实践文章。
- [Three.js Game Skills](./threejs-game-skills.md) — 浏览器 Three.js 交互内容的代理开发工作流；用途与本项目不同。

## 参考来源

- [Every Joule 项目页与源码状态核查](../../sources/sites/every-joule.md)

## 推荐继续阅读

- [Every Joule 官方站](https://iamtechartist.com/every-joule/) — 十课目录、体验入口与教师说明。
- [作者首发帖](https://x.com/techartist_/status/2113310927093436587) — 构建说明与第 1 课。
- [作者十课发布帖](https://x.com/techartist_/status/2114035465875824813) — 全部课程体验入口。
