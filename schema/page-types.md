# Page Types

其它 `schema` 文件索引见 [README.md](README.md)。

本文件定义 `wiki/` 中的页面类型。

---

## YAML Frontmatter 规范

每个 wiki 页面顶部应包含 YAML frontmatter，供 Obsidian Dataview 插件查询：

```yaml
---
type: concept        # concept | method | task | comparison | formalization | entity | overview | query
tags: [locomotion, control, dynamics]
status: complete     # stub | draft | complete
---
```

**字段说明：**
- `type`：页面类型，对应下方各页面类型定义
- `tags`：相关主题标签，用于 Dataview 过滤和图谱分析
- `status`：`stub`（只有骨架）/ `draft`（基本完成但待完善）/ `complete`（满足最低质量标准）

**Dataview 查询示例（在 Obsidian 中）：**
```dataview
TABLE type, status, tags
FROM "wiki"
WHERE status = "stub"
SORT type ASC
```

> 为什么需要 frontmatter：Karpathy LLM Wiki 模式中明确提到，通过 YAML frontmatter + Dataview 可以对 wiki 做动态查询，
> 避免维护静态目录，让知识库的元数据也保持"活的"状态。

---

## 1. Overview Page
回答：一个领域整体是什么。

适用场景：
- robot learning overview
- humanoid control overview
- sim2real overview

推荐结构：
1. 一句话总结
2. 这个领域解决什么问题
3. 核心组成
4. 和其他领域的关系
5. 推荐起步阅读

## 2. Concept Page
回答：一个概念是什么。

适用场景：
- Sim2Real
- Whole-Body Control
- MDP
- Domain Randomization

推荐结构：
1. 一句话定义
2. 为什么重要
3. 核心组成
4. 常见误区
5. 参考来源
6. 关联页面
7. 推荐继续阅读

## 3. Method Page
回答：一种方法怎么工作。

适用场景：
- Reinforcement Learning
- Imitation Learning
- MPC
- TSID

推荐结构：
1. 解决什么问题
2. 输入/输出
3. 核心思想
4. 优势
5. 局限
6. 在机器人中的典型应用
7. 参考来源
8. 关联页面

## 4. Task Page
回答：一个任务方向需要什么能力。

适用场景：
- locomotion
- manipulation
- loco-manipulation
- teleoperation

推荐结构：
1. 任务定义
2. 关键挑战
3. 常见方法路线
4. 评价指标
5. 关联系统/方法
6. 参考来源

## 5. Comparison Page
回答：两个或多个方法/框架如何选型。

适用场景：
- WBC vs RL
- PPO vs SAC
- MPC vs policy learning

推荐结构：
1. 比较对象
2. 核心差异
3. 各自适用场景
4. 常见误判
5. 结论
6. 参考来源

## 6. Roadmap Page
回答：学习或研究该主题的顺序。

适用场景：
- humanoid control roadmap
- sim2real roadmap
- control engineer to robot learning roadmap

推荐结构：
1. 适合谁
2. 先修知识
3. 学习阶段划分
4. 每阶段推荐资源
5. 常见卡点

## 7. Source Summary Page
回答：某篇论文/博客/课程讲了什么。

注意：
- 这里只做简版摘要与定位说明
- 深度论文拆解尽量放在 [`Humanoid_Robot_Learning_Paper_Notebooks`](https://github.com/ImChong/Humanoid_Robot_Learning_Paper_Notebooks)

## 8. Query Page（wiki/queries/）
回答：一个高价值问题的综合分析，将探索结论写回 wiki 实现知识复利。

适用场景：
- 跨多个 wiki 页的综合分析（不适合放在单个 concept/method 页）
- 工程选型决策（solver 选哪个 / 算法怎么选）
- 完整 checklist / 操作指南

**必须满足的特殊格式要求**（区别于其他 wiki 页）：

```markdown
> **Query 产物**：本页由以下问题触发：「<问题一句话>」
> 综合来源：<精读的 wiki 页面列表>
```

推荐结构：
1. TL;DR 决策路径（决策树 / 快速结论）
2. 详细分析（对比表 / 分阶段步骤 / 代码示例）
3. 参考来源（链接到 sources/papers/*.md）
4. 关联页面
5. 一句话记忆

**质量要求**：
- 300+ 字（有实质内容，非简单重复 wiki 页内容）
- 至少包含一个对比表格或决策树
- 精读来源不少于 2 个 wiki 页面
- 在 `wiki/queries/README.md` 的产物列表中注册

---

## 新增页面最低质量标准

每个新增 wiki 页面在合入 main 之前，必须满足以下标准：

### 必须满足

1. **一句话定义在最前面**（读完后 5 秒内知道这页在讲什么）
2. **为什么重要**（动机清晰，能说服人为什么要学这个）
3. **核心内容**（定义、公式直觉、典型应用、局限，至少覆盖 3 项）
4. **参考来源**（至少 1 条，指向 `sources/` 中的对应文件或外部链接）
5. **关联页面**（至少链接到 2 个相关 wiki 页）
6. **推荐继续阅读**（至少 1 个外部资源）

**参考来源区块格式示例：**

```markdown
## 参考来源

- [Schulman et al. - PPO 论文](../sources/papers/ppo.md)
- [Karpathy RL 博客](https://karpathy.github.io/2016/05/31/rl/)
```

> 目的：让每个 wiki 页面自身就能追溯知识来源，而不仅依赖 log.md。
> 这是 Karpathy LLM Wiki 模式中"compilation beats retrieval"的核心体现。

### 结构性要求

- 所有页面必须被 `index.md` 或对应模块页索引
- 所有 wiki 页面必须位于 `wiki/concepts/`、`wiki/methods/`、`wiki/tasks/`、`wiki/overview/` 之一
- 不能出现无分类的孤立页面
- 不能把多个不相关主题塞进同一个文件

### 互链要求

- 每页末尾必须有 `关联页面` 和 `推荐继续阅读` 区块
- 同类页面之间要尽量双向链接（如果 A 链接 B，B 也应链接 A）
- roadmap 页面必须链接到对应的 wiki 知识页

### 不满足标准的后果

不满足以上标准的页面不允许合入 main。这不是过度要求，是为了保证知识库的可用性而不是变成资源垃圾场。
