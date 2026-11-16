# Hallmark — Nutlope / Together AI 前端设计 Agent Skill

> 来源归档（官方仓库和关键源码）

- **标题：** Hallmark — Anti-AI-slop design skill
- **类型：** repo / Agent Skill / Node.js 工具
- **维护方：** Together AI（官方 README 署名）；GitHub 仓库 Nutlope/hallmark
- **代码：** https://github.com/Nutlope/hallmark
- **项目页：** https://www.usehallmark.com/
- **核对版本：** [04977e15f3dfd6ee4b74c887dac3eb52665d6c92](https://github.com/Nutlope/hallmark/tree/04977e15f3dfd6ee4b74c887dac3eb52665d6c92)（main，2026-11-12 获取）
- **核对日期：** 2026-11-12
- **核对范围：** 通读官方 README.md、`skills/hallmark/SKILL.md`；阅读 `references/gates.md` 的分层和规则定义，`references/flow.md` 的预检及组件分支；核对 `scripts/theme.mjs`、`scripts/sloplint.mjs`、`scripts/sloplint-render.mjs`、`scripts/install.mjs` 的实际 CLI 入口、读写对象和主要调用链；核查官方项目站及 package.json。未逐个通读全部 50 份主题说明、完整图片库和所有测试文件，未本地安装/执行 UI 截图或质量门禁。
- **通读：** 已完成
- **开放状态：** 已开源；仓库 MIT。核心技能文档、主题参考、Node.js 校验与安装脚本公开；无模型权重或训练数据集的交付目标。
- **映射至 wiki：** [Hallmark 主节点](../../wiki/entities/hallmark.md)

## 官方机制与核验事实

1. **设计约束：** `SKILL.md` 定义不可豁免的 Floor、允许有理由豁免的 Reflex 与非阻塞 Finish，涉及 CSS token、交互焦点、`prefers-reduced-motion`、真实性、视觉排版和移动端。
2. **任务驱动选型：** 官方 v2 README 与 `SKILL.md` 说明主题目录有 50 个主题；`theme.mjs deal` 给出 3 个候选，`draw` 根据任务选页面结构，`scaffold` 生成 CSS token 和基础样式。也允许 custom 绕开主题但不绕开 Floor。
3. **两层检查：** `sloplint.mjs` 对 HTML/CSS 和部分前端组件源码做静态检查；`--render` 走 `sloplint-render.mjs`，用 Puppeteer + Chrome/Chromium 在 320、375、414、768、1024、1280、1440、1920 像素等视口测页面并保存截图。SKILL 另要求人工/子代理评审，不能把其视为 `sloplint` 自动功能。
4. **已有界面的边界：** `redesign <path>` 与 `audit <path>` 必须由使用者明确指定，默认构建流程不会自行重做已有界面。
5. **安装方式：** 官方提供 `npx skills add nutlope/hallmark -g`（`-g` 全局，去掉为当前项目），Node.js ≥18；浏览器检查另需 Chrome 和 `puppeteer-core`。仓库 `install.mjs` 也提供从本地克隆安装/校验的替代路径，与 `skills` CLI 不是同一个命令。

## 工程适用范围

适合 Robot/AI 团队的静态展示站、项目介绍页、仿真 UI 和可视化仪表板的设计/质量校验；不包含机器人控制器、Sim2Real 或真机安全验证。

## 局限与复现状态

- 50 个主题是本次 README v2 快照，未来可能增减。
- `--render` 依赖额外浏览器工具；没有这些依赖时静态检测仍可运行。
- 本归档验证的是**源码逻辑与官方文档**，不是一次实际安装成功、浏览器可用性或设计质量的独立实测。

## 一手来源

- [官方 README（固定提交）](https://github.com/Nutlope/hallmark/blob/04977e15f3dfd6ee4b74c887dac3eb52665d6c92/README.md)
- [官方 SKILL.md](https://github.com/Nutlope/hallmark/blob/04977e15f3dfd6ee4b74c887dac3eb52665d6c92/skills/hallmark/SKILL.md)
- [Gates（规则与分层）](https://github.com/Nutlope/hallmark/blob/04977e15f3dfd6ee4b74c887dac3eb52665d6c92/skills/hallmark/references/gates.md)
- [theme.mjs（主题 CLI）](https://github.com/Nutlope/hallmark/blob/04977e15f3dfd6ee4b74c887dac3eb52665d6c92/skills/hallmark/scripts/theme.mjs)
- [sloplint.mjs（静态检查）](https://github.com/Nutlope/hallmark/blob/04977e15f3dfd6ee4b74c887dac3eb52665d6c92/skills/hallmark/scripts/sloplint.mjs)
- [sloplint-render.mjs（浏览器检查）](https://github.com/Nutlope/hallmark/blob/04977e15f3dfd6ee4b74c887dac3eb52665d6c92/skills/hallmark/scripts/sloplint-render.mjs)
- [install.mjs（本地安装）](https://github.com/Nutlope/hallmark/blob/04977e15f3dfd6ee4b74c887dac3eb52665d6c92/skills/hallmark/scripts/install.mjs)
