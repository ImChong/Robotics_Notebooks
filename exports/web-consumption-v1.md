# Web Consumption Schema v1

本文档定义：网页层应该如何消费 `exports/index-v1.json`。

目标不是先写前端，而是先把**页面层的数据接口**讲清楚。

一句话：

> 导出层解决“仓库里的 markdown 如何变成结构化数据”，网页消费层解决“页面应该从这些结构化数据里拿什么字段来渲染”。

当前对应产物：
- 对象池：`exports/index-v1.json`
- 页面级聚合导出：`exports/site-data-v1.json`

---

## 设计原则

### 1. 页面优先，不是对象优先
导出层按对象组织，网页消费层按页面组织。

也就是说：
- `index-v1.json` 是全量对象池
- `site-data-v1.json` 是第一阶段页面级聚合结果
- 网页层需要的是：主页数据、模块页数据、路线页数据、关系页数据

### 2. 先支持静态页面，再支持复杂交互
第一阶段先服务：
- 首页
- 模块入口页
- 路线页
- 技术栈页
- 单页面详情页

### 3. 尽量复用同一批字段
避免前端为不同页面维护太多特例字段。

---

## 当前建议的网页页面类型

第一阶段建议网页层至少支持 5 类页面：

1. `home_page`
2. `module_page`
3. `roadmap_page`
4. `tech_map_page`
5. `detail_page`

---

## 1. home_page

首页的职责不是展示所有内容，而是回答：

- 这个项目是什么
- 适合谁
- 从哪里开始
- 当前主线是什么
- 现在最值得点进去的入口有哪些

### 首页最少需要的数据

```json
{
  "hero": {
    "title": "Robotics_Notebooks",
    "subtitle": "机器人技术栈知识库 / Robotics research and engineering wiki."
  },
  "quick_entries": [
    {"title": "主路线：运动控制成长路线", "id": "roadmap-motion-control"},
    {"title": "RL Locomotion 学习路径", "id": "roadmap-if-goal-locomotion-rl"}
  ],
  "featured_chain": [
    "wiki-concepts-lip-zmp",
    "wiki-concepts-centroidal-dynamics",
    "wiki-methods-trajectory-optimization",
    "wiki-methods-model-predictive-control",
    "wiki-concepts-tsid",
    "wiki-concepts-whole-body-control",
    "wiki-concepts-sim2real"
  ],
  "featured_modules": [
    "tech-node-control-mpc",
    "tech-node-control-whole-body-control",
    "tech-node-rl-humanoid-rl"
  ]
}
```

### 首页实际依赖字段
- `title`
- `summary`
- `id`
- `type`
- `tags`
- `related`

### 首页当前最适合展示的内容来源
- `README.md`
- `index.md`
- `roadmap/motion-control.md`
- `tech-map/overview.md`

---

## 2. module_page

模块页的职责是回答：

- 一个模块整体在解决什么问题
- 这个模块下最核心的页面有哪些
- 这个模块和其他模块怎么连接

适合展示的模块：
- 控制
- RL
- IL
- sim2real
- locomotion
- humanoid

### 模块页最少需要的数据

```json
{
  "module_id": "control",
  "title": "控制与优化主链",
  "summary": "从 LIP / ZMP 到 MPC、TSID、WBC 的控制主干。",
  "entry_items": [
    "wiki-concepts-lip-zmp",
    "wiki-concepts-centroidal-dynamics",
    "wiki-methods-trajectory-optimization",
    "wiki-methods-model-predictive-control",
    "wiki-concepts-tsid",
    "wiki-concepts-whole-body-control"
  ],
  "related_modules": ["sim2real", "rl"],
  "references": [
    "reference-papers-whole-body-control",
    "reference-repos-utilities"
  ]
}
```

### 模块页实际依赖字段
- `title`
- `summary`
- `tags`
- `related`
- `type`

### 备注
模块页不一定要求仓库里先有一篇对应 markdown，第一阶段可以由前端根据 tags 聚合生成。

---

## 3. roadmap_page

路线页的职责是回答：

- 应该按什么顺序学
- 每个阶段是什么
- 每个阶段该点进哪些知识页

### 路线页最少需要的数据

```json
{
  "id": "roadmap-motion-control",
  "title": "主路线：运动控制算法工程师成长路线",
  "summary": "从机器人基础出发，逐步成长为能做人形机器人运动控制、强化学习与模仿学习相关工作的算法工程师。",
  "stages": [
    {"id": "l0", "title": "数学与编程基础"},
    {"id": "l1", "title": "机器人学骨架"},
    {"id": "l2", "title": "动力学与刚体建模"}
  ],
  "related_items": [
    "wiki-concepts-lip-zmp",
    "wiki-concepts-centroidal-dynamics",
    "wiki-methods-reinforcement-learning"
  ]
}
```

### 路线页实际依赖字段
- `title`
- `summary`
- `stages`
- `related`
- `source_links`

---

## 4. tech_map_page

技术栈页的职责是回答：

- 当前技术栈有哪几个模块
- 模块之间的关系是什么
- 这个节点该先看什么、后看什么

### 技术栈页最少需要的数据

```json
{
  "graph_meta": {
    "overview_id": "tech-node-overview",
    "dependency_graph_id": "tech-node-dependency-graph"
  },
  "nodes": [
    {
      "id": "tech-node-control-mpc",
      "title": "MPC",
      "layer": "control",
      "related": ["wiki-methods-model-predictive-control"]
    },
    {
      "id": "tech-node-rl-ppo",
      "title": "PPO",
      "layer": "rl",
      "related": []
    }
  ]
}
```

### 技术栈页实际依赖字段
- `id`
- `title`
- `summary`
- `layer`
- `node_kind`
- `related`

### 第一阶段建议
技术栈图不用一上来就做成力导图。先做：
- layer 分组
- 节点卡片
- 点击后跳对应 detail page

---

## 5. detail_page

详情页是最通用的页面类型，适用于：
- wiki 页面
- entity 页面
- reference 页面
- tech-map 单节点页面

### detail_page 最少需要的数据

```json
{
  "id": "wiki-concepts-centroidal-dynamics",
  "title": "Centroidal Dynamics",
  "type": "wiki_page",
  "summary": "用机器人整体质心的线动量和角动量来描述全身动力学的一种中层建模方式。",
  "tags": ["concept", "dynamics", "control"],
  "related": [
    "wiki-concepts-lip-zmp",
    "wiki-concepts-whole-body-control",
    "reference-papers-whole-body-control"
  ],
  "source_links": []
}
```

### detail_page 实际依赖字段
- `id`
- `type`
- `title`
- `summary`
- `tags`
- `related`
- `source_links`
- `path`

### 注意
第一阶段的 detail page 可以只用导出层做卡片信息，正文仍然直接回源 markdown。

也就是说：
- 导出层负责页面元信息
- markdown 负责正文内容

这是当前最稳的方案。

---

## 当前建议的前端消费方式

### 最简单方案
前端直接加载：
- `exports/index-v1.json`

然后按：
- `type`
- `tags`
- `id`
- `related`

做页面聚合。

### 为什么先这样做
因为当前项目还在快速演化：
- 页面结构在变
- tags 还会继续修
- relations 还会继续变细

这时候先用一个总 JSON 最稳。

---

## 暂时不建议过早拆分的东西

当前先不要急着拆：
- `home.json`
- `roadmaps.json`
- `tech-map.json`
- `references.json`

理由很简单：
- 现在还在验证 schema
- 先有一个稳定总对象池更灵活
- 等前端页面结构稳定了，再做二次聚合导出更合理

---

## 下一步建议

如果继续往下做，最自然的下一步是：

1. 定义一个最小网页信息架构（首页 / 模块页 / 路线页 / 技术栈页）
2. 指定每个页面怎么从 `index-v1.json` 选数据
3. 如有必要，再补一层前端友好的聚合导出格式
