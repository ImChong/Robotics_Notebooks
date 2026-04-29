# AGENTS.md - Robotics_Notebooks

本仓库从“思维导图资源集合”升级为“机器人研究与工程知识库”。

## 项目定位

- **主定位：** 面向机器人研究与工程的 markdown wiki
- **内容核心：** 机器人技术栈、控制、强化学习、模仿学习、Sim2Real、系统设计
- **展示层：** 当前思维导图网页先保留，不优先改渲染
- **知识层：** 逐步迁移到 `wiki/`，并通过 `exports/` 给网页提供数据

## 仓库分层

### 1. `sources/`
原始资料层。只收集，不做强结构化改写。

包括但不限于：
- 论文
- 博客
- 课程
- 视频
- 外部仓库
- 旧版笔记原稿

### 2. `wiki/`
结构化知识层。是本仓库的核心。

目标：
- 将分散资料整理成概念页、方法页、任务页、路线页、对比页
- 强调交叉引用
- 强调面向研究和工程使用

### 3. `schema/`
知识库维护规则。

包括：
- 页面类型定义
- 命名规范
- 链接规范
- ingest 流程

### 4. `exports/`
给网页/思维导图使用的导出层。

当前先占位，后续再把 `wiki/` 映射到导图数据。

### 5. `docs/checklists/`
项目执行清单与阶段性维护看板。

包括：
- 当前技术栈执行清单：[`docs/checklists/tech-stack-next-phase-checklist-v21.md`](docs/checklists/tech-stack-next-phase-checklist-v21.md)
- 前端体验优化清单：[`docs/checklists/frontend-optimization-v1.md`](docs/checklists/frontend-optimization-v1.md)
- 历史执行清单索引：[`docs/checklists/README.md`](docs/checklists/README.md)

这些文件用于记录阶段性工程计划、验收标准和历史推进过程；不要把它们当作 wiki 知识页。若修改前端体验、导出链路或阶段性目标，应同步更新对应 checklist。

## 写作原则

1. **原始资料和知识归纳分开**
   - 原始来源进 `sources/`
   - 归纳后的页面进 `wiki/`

2. **优先按知识实体组织，而不是按时间堆砌**
   - 概念、方法、任务、系统、对比、路线

3. **允许图结构，不强制单父节点树**
   - 同一个概念可以被多个页面引用
   - 底层是知识图，前端才是树/导图

4. **避免和 `Humanoid_Robot_Learning_Paper_Notebooks` 重复**
   - [`Humanoid_Robot_Learning_Paper_Notebooks`](https://github.com/ImChong/Humanoid_Robot_Learning_Paper_Notebooks) 负责单篇论文深读
   - [`Robotics_Notebooks`](https://github.com/ImChong/Robotics_Notebooks) 负责跨主题知识组织

5. **优先可维护性，不追求一次性完美迁移**
   - 先搭 schema 和 wiki 骨架
   - 再做试点迁移
   - 最后再考虑前端导出

## 页面风格要求

所有 wiki 页面优先简洁、直接、可交叉引用。

推荐包含：
- 一句话定义/总结
- 为什么重要
- 核心结构/机制
- 常见误区或局限
- 与其他页面的关系
- 推荐继续阅读

## 第一阶段目标

第一阶段只做：
- 建立新目录结构
- 建立 schema 文件
- 建立 index/log
- 建立一批 MVP wiki 页面
- 不大动旧网页渲染

## 禁止事项

- 不要把网页渲染逻辑和知识结构强绑定
- 不要把所有内容继续塞回一个大纲式 README
- 不要为了兼容旧导图而牺牲 wiki 结构设计

## 迁移策略

旧内容暂时保留，逐步迁移。

迁移顺序建议：
1. humanoid control
2. reinforcement learning / imitation learning
3. sim2real
4. locomotion / manipulation

## 对 LLM / 维护者的要求

在新增或修改页面时：
- 优先复用现有页面与链接
- 若知识点已存在，补充而不是重复造页
- 若是新外部资料，先进入 `sources/`，再决定是否沉淀到 `wiki/`
- 若会影响学习路径，应同步更新 `index.md` 或相关 roadmap 页面

### 浏览器验证工具

后续 agent 如果需要修改或验证 `docs/` 下的前端页面，尤其是 `graph.html` 这类交互页面，应优先安装并使用 Chrome DevTools MCP：

- 项目地址：<https://github.com/ChromeDevTools/chrome-devtools-mcp>
- Codex CLI 可用示例：`codex mcp add chrome-devtools -- npx -y chrome-devtools-mcp@latest`
- 用途：打开本地页面、检查 console/network、模拟点击/键盘交互、截图或验证图谱节点状态。

### LLM Wiki Ops 规范（必须遵守）

本知识库采用 **Karpathy LLM Wiki 模式**：LLM 是维护者，人类是 curator。

核心操作规范在 `schema/ingest-workflow.md`，每次维护本仓库前必须先读该文件。

三种核心 Op：
- **`ingest`** — 新资料进入知识库（先 `sources/`，再判断是否升格 `wiki/`）
- **`query`** — 向知识库提问，结果写回 wiki 而非留在聊天记录；独立洞见写入 `wiki/queries/`
- **`lint`** — 定期健康检查（orphan pages、矛盾、缺失 cross-reference、缺失参考来源等）

关键约束：
- 不要把 source 直接复制成 wiki — 要提炼，不是转存
- 不要把 wiki 页写成纯外链列表 — 要有知识归纳
- 不要为了收集而收集 — 优先服务学习与研究主线
- 不要在 ingest 时一次性做太多事 — 一次一条资料，深度到位再推进
- 每次 ingest 都要追加到 `log.md`
- 每次 query 有好结果都要写回 wiki
- **每个 wiki 页面必须包含 `## 参考来源` 区块**，标注该页知识编译自哪些原始资料
  （这是 Karpathy"compilation beats retrieval"的核心体现：页面本身即溯源）
- **CI 质量网关（必须通过）**：
  - 提交前必须本地运行 `make lint`，确保 0 issues。
  - **严禁使用 `[[...]]` 语法**进行内链（代码块内除外），必须使用标准 `[text](path)` 格式，以确保 `lint_wiki.py` 的入链统计与断链检查准确。
  - **同步统计数据**：若新增/删除了 wiki 页面，必须按顺序运行 `make catalog`、`make graph` 和 `make export`，并手动更新 `README.md` 与 `docs/index.html` 中的节点/边统计数值，否则 GitHub Actions 会因数据不一致而报错。

### Git 提交规范 (Git Commit Convention)

为保持仓库历史清晰，所有提交必须使用 **中文** 描述，并遵循以下格式：

除非用户明确要求不要提交或不要推送，否则 agent 完成修改后应：
- 只 stage 本次任务相关文件，避免带入无关工作区变化。
- 按本节格式创建中文 commit，commit 消息参考近期历史提交风格。
- 将当前分支推送到 GitHub 远端（通常是 `origin main` 或当前工作分支）。

1. **知识入库提交 (Ingest)**：
   格式：`[YYYY-MM-DD] ingest | <源文件路径> — <中文描述内容>`
   示例：`[2026-04-23] ingest | sources/repos/robot_lab.md — 接入 IsaacLab 扩展框架并同步全站索引`

2. **结构/功能/修复提交**：
   格式：`<类型>(范围): <中文描述内容>`
   - 类型 (type)：feat, fix, chore, docs, refactor, style, test。
   - 范围 (scope)：可选（如 ux, actions, wiki）。
   示例：`fix(actions): 修复 CLAW 页面格式缺失主要技术路线的问题`
   示例：`chore: 更新主页统计数据与图谱 (172 nodes, 955 edges)`
