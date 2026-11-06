# Superpowers README 通读

> 来源归档（原文通读 · 支撑 `wiki/entities/superpowers-obra.md`）

- **标题：** obra/superpowers README（main 分支）
- **作者：** Jesse Vincent / Prime Radiant
- **类型：** repo / readme
- **原文：** <https://github.com/obra/superpowers>（2026-11 通过 raw.githubusercontent.com 获取 README.md 全文，约 400 行）
- **一句话说明：** 面向多种编码代理 harness 的技能库与启动指令，把头脑风暴、计划、子代理实现、TDD、评审、收尾写成"自动触发的强制工作流"。
- **核对范围：** 通读 README 全部章节（How it works、Commercial Services、16 种 harness 安装、Basic Workflow、When Something Goes Wrong、Community、What's Inside、Philosophy、Contributing、Updating、License、telemetry）。未逐个阅读仓库内各 `skills/*/SKILL.md`、`docs/README.*.md` 及 superpowers-evals，具体技能正文细节不在本次核对范围。
- **通读：** 已完成

## 核心摘录（均出自 README）

- **定位：** 一套"完整的软件开发方法论"，由可组合 skills 加启动初始指令构成；技能自动触发，用户无需特别操作。
- **流程：** 先追问真实目标 → 分段展示规格供人批准 → 计划写到"热情但无品味、无判断、无项目背景、不爱测试的初级工程师"也能执行 → 用户说 go 后进入 subagent-driven-development，代理可连续自主工作数小时。
- **基本工作流 7 步：** brainstorming；using-git-worktrees（批准后建隔离分支并验证干净测试基线）；writing-plans（2-5 分钟粒度任务，含精确路径、完整代码、验证步骤）；subagent-driven-development（每任务一个新子代理并逐任务评审，最彻底）或 executing-plans（当前会话内联执行，末尾对整分支做一次评审，最省）；test-driven-development（RED-GREEN-REFACTOR，先于测试写的代码会被删除）；requesting-code-review（按严重度分级，Critical 阻断）；finishing-a-development-branch（验证测试、给出 merge/PR/保留/丢弃选项、清理 worktree）。
- **强制性：** 代理在任何任务前检查相关技能，"强制工作流而非建议"。
- **技能库：** Testing（TDD）；Debugging（systematic-debugging 四阶段、verification-before-completion、diagnosing-superpowers）；Collaboration（brainstorming、writing-plans、executing-plans、dispatching-parallel-agents、requesting/receiving-code-review、using-git-worktrees、finishing-a-development-branch、subagent-driven-development 两阶段评审：先规格符合、再代码质量）；Meta（writing-skills、using-superpowers）。
- **诊断：** 会话行为异常时可让代理调用 diagnosing-superpowers 读取会话转录，给出带行级证据的报告，并可打包脱敏材料用于 bug 报告。
- **安装：** 各 harness 需分别安装（Claude Code 官方/自有市场、Codex、Cursor、Gemini CLI、Copilot CLI、OpenCode、Pi、Hermes Agent、Muse 等）；Pi 与 Hermes 说明了启动引导注入与压缩后丢失引导的差异。
- **哲学：** TDD、系统化优于临时摸索、降低复杂度、证据优先于声明。
- **贡献：** 一般不接受新增技能；技能改动须跨所支持的代理可用；技能行为测试用 superpowers-evals 的 drill 评测框架。
- **遥测：** brainstorming 可选可视化伴随功能默认从官网加载 logo（带版本号），可用 `SUPERPOWERS_DISABLE_TELEMETRY` 等关闭。
- **协议：** MIT。

## 对 wiki 的映射

- [`wiki/entities/superpowers-obra.md`](../../wiki/entities/superpowers-obra.md)

## 参考来源

- 仓库：<https://github.com/obra/superpowers>
