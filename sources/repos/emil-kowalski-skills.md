# Emil Kowalski Skills（emilkowalski/skills）

> 来源归档

- **标题：** Skills for Designers and Engineers
- **类型：** repo
- **作者：** Emil Kowalski
- **链接：** https://github.com/emilkowalski/skills
- **主页 / 分发：** https://aiforui.dev/skills · `npx skills@latest add emilkowalski/skills`
- **代码：** MIT；本次核查的仓库 HEAD 为 [e8a175d](https://github.com/emilkowalski/skills/commit/e8a175de22ae1e49370fc144c1f3bb9aeedf988d)（2026-10-02）
- **入库日期：** 2026-10-09
- **一句话说明：** 面向设计与前端工程的 Agent Skills 集合，把动效、界面设计、原型、多端适配、Swift 和 UI 压力测试经验写成可按需安装的 `SKILL.md`。
- **为什么值得保留：** 它将作者的设计工程判断转为可在 coding agent 中复用的工作规约，覆盖「设计—实现—审查—反例验证」；适合与通用 frontend-design、工程技能集对照，而不是当作 UI 组件库或自动设计器。
- **沉淀到 wiki：** 是 → [Emil Kowalski Skills](../../wiki/entities/emil-kowalski-skills.md)

## README 要点（截至 2026-10-02 快照）

- **定位：** 设计与工程能力的副产品；强调 AI 放大专业判断，而非替代设计经验。
- **安装：** `npx skills@latest add emilkowalski/skills`；目标仓库中由 agent 按技能说明工作。
- **技能清单：** README 当时列出 14 项：`emil-design-eng`、`animate`、`animate-expo`、`review-animations`、`improve-animations`、`find-animation-opportunities`、`animation-vocabulary`、`apple-design`、`write-swift`、`pick-ui-library`、`prototype`、`mobile-native`、`break-ui`、`ask-sonner`。清单会随上游更新变化。
- **重点技能：** `emil-design-eng` 提供整体设计/动效判断；`animate` 从目标到曲线、时长与属性实现动效；`review-animations` 和 `improve-animations` 分别偏严格审查与批量审计；`break-ui` 用现实极端数据寻找布局缺陷；`prototype` 生成多种 UI 方案并用切换器比较。
- **协议：** 根目录 MIT License。

## 面向实践的理解

该仓库提供的是**可安装的提示规约与操作流程**，不是可独立调用的设计模型、渲染引擎或组件实现。运行效果取决于 agent 是否能读取目标仓库、运行应用并检查结果；设计指导不能替代浏览器/设备验证，也不保证输出必然符合产品品牌和无障碍要求。技能目录属于快速迭代内容，使用前应核对上游当前版本与所选技能正文。

## 参考来源

- [上游 README](https://github.com/emilkowalski/skills/blob/e8a175de22ae1e49370fc144c1f3bb9aeedf988d/README.md)
- [上游 MIT License](https://github.com/emilkowalski/skills/blob/e8a175de22ae1e49370fc144c1f3bb9aeedf988d/LICENSE)
- [作者技能介绍页](https://aiforui.dev/skills)
- [上游提交 e8a175d](https://github.com/emilkowalski/skills/commit/e8a175de22ae1e49370fc144c1f3bb9aeedf988d)
