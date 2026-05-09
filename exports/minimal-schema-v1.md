# Exports Minimal Schema v1

本文档定义 `Robotics_Notebooks` 当前阶段的**最小可用导出层 schema**。

目标不是一步做成完整知识图数据库，而是先解决：

> 现有 wiki / roadmap / tech-map / entities / references 内容，如何以统一字段导出给网页、搜索和可视化层消费。

---

## 设计原则

### 1. 最小可用
只导出当前网页和导航层最需要的字段，不追求一次覆盖所有细节。

### 2. 内容优先于形式
导出层不是新知识层，而是已有 markdown 内容的结构化镜像。

### 3. 兼容现有目录结构
尽量直接映射当前：
- `wiki/`
- `roadmap/`
- `tech-map/`
- `references/`
- `sources/`

### 4. 先服务导航，再服务分析
先保证：
- 首页能列内容
- 模块页能显示关系
- 路线页能展示阶段与入口
- references 能做跳转

---

## 顶层对象类型

当前最小 schema 先覆盖 5 类对象：

1. `wiki_page`
2. `roadmap_page`
3. `entity_page`
4. `reference_page`
5. `tech_map_node`

后续如需要，再扩：
- `source_page`
- `benchmark_page`
- `paper_entry`
- `repo_entry`

---

## 通用字段（所有对象共享）

所有导出对象至少应包含：

```json
{
  "id": "unique-stable-id",
  "type": "wiki_page | roadmap_page | entity_page | reference_page | tech_map_node",
  "title": "页面标题",
  "path": "仓库内相对路径",
  "summary": "一句话摘要",
  "tags": ["tag1", "tag2"],
  "related": ["other-id-1", "other-id-2"],
  "source_links": ["https://..."],
  "status": "active"
}
```

### 字段说明

| 字段 | 含义 |
|------|------|
| `id` | 稳定唯一 ID，供网页、关系图、搜索引用 |
| `type` | 对象类型 |
| `title` | 页面标题 |
| `path` | markdown 文件相对路径 |
| `summary` | 一句话摘要 |
| `tags` | 标签，用于检索和过滤 |
| `related` | 关联对象 ID |
| `source_links` | 外部链接（docs / repo / papers 等） |
| `status` | 当前默认 `active` |

---

## 1. wiki_page

用于导出：
- concepts
- methods
- tasks
- comparisons
- overview

最小字段：

```json
{
  "id": "wiki-concepts-centroidal-dynamics",
  "type": "wiki_page",
  "page_type": "concept | method | task | comparison | overview",
  "title": "Centroidal Dynamics",
  "path": "wiki/concepts/centroidal-dynamics.md",
  "summary": "用机器人整体质心的线动量和角动量来描述全身动力学的一种中层建模方式。",
  "tags": ["humanoid", "locomotion", "dynamics", "control"],
  "related": ["wiki-concepts-lip-zmp", "wiki-concepts-tsid"],
  "source_links": [],
  "status": "active"
}
```

### 额外建议字段
- `page_type`
- `module`
- `difficulty`
- `reading_stage`

---

## 2. roadmap_page

用于导出：
- motion-control
- learning-paths 下各页面

最小字段：

```json
{
  "id": "roadmap-motion-control",
  "type": "roadmap_page",
  "title": "主路线：运动控制算法工程师成长路线",
  "path": "roadmap/motion-control.md",
  "summary": "从机器人基础出发，逐步成长为能做人形机器人运动控制、强化学习与模仿学习相关工作的算法工程师。",
  "tags": ["roadmap", "motion-control", "humanoid"],
  "related": ["wiki-roadmaps-humanoid-control-roadmap"],
  "source_links": [],
  "status": "active",
  "stages": [
    {"id": "l0", "title": "数学与编程基础"},
    {"id": "l1", "title": "机器人学骨架"}
  ]
}
```

### 额外建议字段
- `stages`
- 路线图页 **流程图** 为浏览器内用 `roadmap_pages[].stages` 生成的 **Mermaid** 总图（非 `site-data` 字段）。
- `target_user`
- `prerequisites`
- `outputs`

---

## 3. entity_page

用于导出：
- MuJoCo
- Isaac Gym / Isaac Lab
- Pinocchio
- Crocoddyl
- Unitree

最小字段：

```json
{
  "id": "entity-mujoco",
  "type": "entity_page",
  "title": "MuJoCo",
  "path": "wiki/entities/mujoco.md",
  "summary": "机器人与控制领域最经典的物理引擎之一，由 Google DeepMind 维护并开源。",
  "tags": ["simulator", "rl", "control", "robotics"],
  "related": ["wiki-methods-reinforcement-learning", "entity-isaac-gym-isaac-lab"],
  "source_links": ["https://mujoco.org/", "https://github.com/google-deepmind/mujoco"],
  "status": "active",
  "entity_kind": "simulator"
}
```

### 额外建议字段
- `entity_kind`（simulator / library / hardware / framework）
- `official_site`
- `repo_url`
- `ecosystem`

---

## 4. reference_page

用于导出：
- papers/ 下主题页
- repos/ 下主题页
- benchmarks/ 下主题页

最小字段：

```json
{
  "id": "reference-papers-locomotion-rl",
  "type": "reference_page",
  "reference_kind": "papers | repos | benchmarks",
  "title": "Locomotion RL",
  "path": "references/papers/locomotion-rl.md",
  "summary": "聚焦人形/腿足机器人 locomotion 中的强化学习论文。",
  "tags": ["rl", "locomotion", "papers"],
  "related": ["wiki-tasks-locomotion", "wiki-methods-reinforcement-learning"],
  "source_links": [],
  "status": "active"
}
```

### 额外建议字段
- `reference_kind`
- `mapped_modules`
- `entry_points`

---

## 5. tech_map_node

用于导出：
- tech-map 中的模块
- dependency graph 中的节点

最小字段：

```json
{
  "id": "tech-node-centroidal-dynamics",
  "type": "tech_map_node",
  "title": "Centroidal Dynamics",
  "path": "tech-map/dependency-graph.md",
  "summary": "机器人整体质心与动量的中层建模模块。",
  "tags": ["tech-map", "control", "dynamics"],
  "related": ["wiki-concepts-centroidal-dynamics"],
  "source_links": [],
  "status": "active",
  "node_kind": "concept",
  "parents": ["tech-node-lip-zmp"],
  "children": ["tech-node-tsid", "tech-node-trajectory-optimization"]
}
```

### 额外建议字段
- `node_kind`
- `parents`
- `children`
- `layer`

---

## 当前最小导出目标

第一阶段不生成复杂图数据库，只先确保：

1. **首页能列内容卡片**
2. **模块页能显示 related links**
3. **路线页能显示 stages**
4. **tech-map 能显示父子关系**
5. **references 能被按主题聚合显示**

也就是说：
- 先解决导航
- 再解决搜索
- 最后再考虑复杂关系分析

---

## 生成策略建议

### 第一步
先做人工维护的 schema 文档（当前就是这一步）。

### 第二步
再写一个脚本，把 markdown 头部信息或正文规则提取成最小 JSON。

### 第三步
最后再决定：
- 存成单个 `exports/index.json`
- 还是分类型导出多个 JSON 文件

---

## id / tags / related 生成规则

这是当前阶段最关键的三条规则。只要这三条定清楚，后面无论是导出 JSON、做网页还是做关系图，都会稳定很多。

### 1. `id` 生成规则

原则：
- 稳定优先，不要依赖页面标题
- 直接基于**类型 + 相对路径**生成
- 保持全小写、连字符风格

推荐规则：

```text
wiki/concepts/centroidal-dynamics.md
→ wiki-concepts-centroidal-dynamics

wiki/methods/trajectory-optimization.md
→ wiki-methods-trajectory-optimization

wiki/entities/mujoco.md
→ entity-mujoco

roadmap/motion-control.md
→ roadmap-motion-control

references/papers/locomotion-rl.md
→ reference-papers-locomotion-rl

tech-map/dependency-graph.md 中的节点“Centroidal Dynamics”
→ tech-node-centroidal-dynamics
```

#### 规则摘要
- `wiki/concepts/*` → `wiki-concepts-*`
- `wiki/methods/*` → `wiki-methods-*`
- `wiki/tasks/*` → `wiki-tasks-*`
- `wiki/comparisons/*` → `wiki-comparisons-*`
- `wiki/overview/*` → `wiki-overview-*`
- `wiki/entities/*` → `entity-*`
- `roadmap/*` → `roadmap-*`
- `references/papers/*` → `reference-papers-*`
- `references/repos/*` → `reference-repos-*`
- `references/benchmarks/*` → `reference-benchmarks-*`
- `tech-map` 中的抽象节点 → `tech-node-*`

### 2. `tags` 生成规则

当前阶段先不做全自动 NLP 标签抽取，避免不稳定。

采用**半规则化生成**：

#### 第一层：目录标签（强规则）
根据文件目录直接生成：
- `wiki/concepts/*` → `concept`
- `wiki/methods/*` → `method`
- `wiki/tasks/*` → `task`
- `wiki/comparisons/*` → `comparison`
- `wiki/entities/*` → `entity`
- `roadmap/*` → `roadmap`
- `references/papers/*` → `papers`
- `references/repos/*` → `repos`
- `references/benchmarks/*` → `benchmarks`

#### 第二层：主线标签（弱规则）
根据文件路径名和标题进行人工维护映射：
- `humanoid`
- `locomotion`
- `control`
- `dynamics`
- `optimization`
- `rl`
- `il`
- `sim2real`
- `wbc`
- `tooling`
- `hardware`

#### 当前建议
第一版导出时：
- 至少保留 1 个目录标签
- 再补 2–4 个主线标签
- 总标签数控制在 3–6 个之间

### 3. `related` 生成规则

当前阶段也不做复杂自动图推理，采用**显式链接优先**策略。

#### 优先级 1：页面中的“关联页面”区块
如果页面末尾有 `关联页面`，直接把这些页面对应的 `id` 放进 `related`。

#### 优先级 2：页面中的“继续深挖入口”区块
如果页面有 `继续深挖入口`，其中链接到的 references / repos / benchmarks 页面也进入 `related`。

#### 优先级 3：同主题显式互链
例如：
- `Centroidal Dynamics` ↔ `TSID`
- `Locomotion` ↔ `Reinforcement Learning`
- `MuJoCo` ↔ `Isaac Gym / Isaac Lab`

只要页面正文里有明确链接，也可进入 `related`。

#### 当前不做
- 不做纯 embedding 相似度生成 related
- 不做复杂图算法自动补全 related
- 不做无显式依据的“猜测性关联”

原因：
- 当前内容还在快速演化
- 先保证 related 可解释、稳定

## 当前最小落地建议

如果现在就开始做导出脚本，建议第一版规则是：

1. `id`：完全按路径规则生成
2. `tags`：目录标签 + 少量人工主线标签
3. `related`：只取显式 markdown 链接（重点取“关联页面”和“继续深挖入口”）
4. `summary`：优先取“一句话定义”或首段摘要

这样做的好处是：
- 简单
- 稳定
- 可维护
- 后面容易扩展

## 推荐下一步

1. 为现有页面补最小可导出的元信息约束（哪怕先是人工规则）
2. 按本文档规则生成第一版 `id / tags / related`
3. 再决定是否写自动导出脚本
4. 最后再考虑是否引入更复杂的自动推理
