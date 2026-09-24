# Naming Rules

其它 `schema` 文件索引见 [README.md](README.md)。

## 总原则

命名追求：
- 清晰
- 可搜索
- 稳定
- 尽量避免中英文混杂在文件名中

## 目录命名

统一使用小写英文 + 连字符：

- `overview/`
- `concepts/`
- `methods/`
- `tasks/`
- `comparisons/`
- `roadmaps/`

## 文件命名

统一使用小写英文 + 连字符 + `.md`：

示例：
- `robot-learning-overview.md`
- `sim2real.md`
- `whole-body-control.md`
- `reinforcement-learning.md`
- `wbc-vs-rl.md`

## 页面标题

文件名用英文，页面内容里可以同时保留中英文标题。

推荐格式：
```md
# Sim2Real
**仿真到现实迁移**
```

## 图谱社区命名

知识图谱（`exports/link-graph.json`）中的「社区」是 [`schema/topics.json`](topics.json) 登记的 **20 个固定主题**，不再由 Louvain 结构聚类决定。展示名由 `scripts/generate_link_graph.py` 按主题 `label` 追加后缀生成，**统一格式**：

```text
中文（English） 社区
```

### 格式要求

| 部分 | 规则 | 示例 |
|------|------|------|
| 中文主名 | 放在最前，用简体中文概括主题 | `强化学习`、`硬件与执行器` |
| 英文副名 | 放在**全角括号** `（）` 内；有通用缩写时优先 **全称, 缩写**（如 `Reinforcement Learning, RL`）；专有名可仅写全称 | `（Reinforcement Learning, RL）`、`（Hardware and Actuators）` |
| 后缀 | 固定为半角空格 + `社区` | ` 社区` |

完整示例：`强化学习（Reinforcement Learning, RL） 社区`、`导航与 SLAM（Navigation and SLAM） 社区`。

### 节点归属规则

每个节点最多两个主题（主 + 次）：主主题 → `node.community`（图谱着色、图例、首页 chip 规模），次主题 → `node.community_secondary`（详情页「所属社区」第二枚徽标、路线视图命中）。优先级：

1. frontmatter `topic:` 显式声明（1–2 个主题 id，整体覆盖以下规则），如 `topic: [ecosystem]`；
2. `seeds` 种子页固定主主题（`seeds[0]` 为锚点页，首页 chip 的搜索别名挂在它上面）；
3. frontmatter `tags` 按**书写顺序**精确匹配各主题 `tags`：第一个命中为主，第二个不同主题为次；
4. 仍无主题的节点按已定主题邻居投票传播（仅主主题；平票取注册表靠前者）；
5. 与任何已定主题节点都不连通的节点归 `其他（Other） 社区`（`community-other`）。

社区 id 固定为 `community-<topic-id>`（如 `community-vla`），前端可直接引用。

### 维护方式

1. 新增或调整主题：改 `schema/topics.json`（`id` / `label` / `seeds` / `tags`）；一个 tag 只能归属一个主题，主题数保持 20。
2. 单页归属不对：优先在该页 frontmatter 加 `topic:`；批量不对时调整 tags 或注册表 `tags`。
3. `make topic-diagnose` 用 Louvain 结构聚类对照，列出「结构上更像属于另一主题」的可疑页面，供人工复核。
4. `make lint` 会报出不在注册表中或超过 2 个的 `topic:`（阻塞 CI）；主题 label 不符合 `中文（…）` 模式时 `make graph` 打印 `WARNING`。

### 命名反例（勿用）

- `SONIC（规模化运动跟踪人形控制） 社区` — 英文在前、中文在括号内
- `Robot Learning Overview 社区` — 纯英文、无中文主名
- `Humanoid Hardware 101：七类子系统技术地图 社区` — 英文主名 + 中文副标题，未遵循「中文（English）」
- `BFM 技术地图（Behavior Foundation Model） 社区` — 英文缩写开头、中文不在主位

## 研究机构命名

知识图谱与详情页「所属机构」徽标使用的展示名来自 [`schema/institutions.json`](institutions.json) 的 `label` 字段，**统一格式**：

```text
中文（English）
```

### 格式要求

与上文「图谱社区命名」的基名规则相同（不含 ` 社区` 后缀）：

| 部分 | 规则 | 示例 |
|------|------|------|
| 中文主名 | 放在最前，用简体中文概括机构 | `英伟达`、`清华大学` |
| 英文副名 | 放在**全角括号** `（）` 内；可为品牌、缩写或官方英文名 | `（NVIDIA）`、`（Tsinghua）` |

完整示例：`英伟达（NVIDIA）`、`地平线（Horizon Robotics）`、`清华大学（Tsinghua）`。

### 维护方式

1. 新增机构时在 `schema/institutions.json` 的 `registry` 追加 `id`、`label`、`aliases`。
2. `aliases` 为 frontmatter `tags` 的精确匹配 token（小写）；`label` 仅用于展示，不参与匹配。
3. 运行 `make graph` 时，若某 `label` 不符合 `INSTITUTION_LABEL_RE`，脚本会打印 `WARNING`；CI 不因此失败，但维护者应修正 `label`。

## 避免事项

不要：
- 用空格做文件名
- 用时间戳做知识页文件名
- 把多个不相关主题塞进同一个文件
- 把 README 继续当总索引和总内容的混合垃圾场
- 在 `schema/topics.json` 中写不符合「中文（English）」格式的主题 label
- 让机构 `label` 使用纯英文或「English（中文）」颠倒格式
