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
- route-a-motion-control
- route-b-fullstack
- learning-paths 下各页面

最小字段：

```json
{
  "id": "roadmap-route-a-motion-control",
  "type": "roadmap_page",
  "title": "路线A：运动控制算法工程师成长路线",
  "path": "roadmap/route-a-motion-control.md",
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

## 推荐下一步

1. 为现有页面补最小可导出的元信息约束（哪怕先是人工规则）
2. 确定 `id` 命名规则
3. 确定 tags 和 related 的最小生成策略
4. 再决定要不要做自动导出脚本
