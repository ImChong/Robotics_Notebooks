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

### LLM Wiki Ops 规范（必须遵守）

本知识库采用 **Karpathy LLM Wiki 模式**：LLM 是维护者，人类是 curator。

核心操作规范在 `schema/ingest-workflow.md`，每次维护本仓库前必须先读该文件。

三种核心 Op：
- **`ingest`** — 新资料进入知识库（先 `sources/`，再判断是否升格 `wiki/`）
- **`query`** — 向知识库提问，结果写回 wiki 而非留在聊天记录
- **`lint`** — 定期健康检查（orphan pages、矛盾、缺失 cross-reference 等）

关键约束：
- 不要把 source 直接复制成 wiki — 要提炼，不是转存
- 不要把 wiki 页写成纯外链列表 — 要有知识归纳
- 不要为了收集而收集 — 优先服务学习与研究主线
- 不要在 ingest 时一次性做太多事 — 一次一条资料，深度到位再推进
- 每次 ingest 都要追加到 `log.md`
- 每次 query 有好结果都要写回 wiki
