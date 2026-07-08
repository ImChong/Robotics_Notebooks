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

知识图谱（`exports/link-graph.json`）中每个社区的展示名由 `scripts/generate_link_graph.py` 生成，**统一格式**：

```text
中文（English） 社区
```

### 格式要求

| 部分 | 规则 | 示例 |
|------|------|------|
| 中文主名 | 放在最前，用简体中文概括主题 | `强化学习`、`人形硬件技术地图` |
| 英文副名 | 放在**全角括号** `（）` 内；有通用缩写时优先 **全称, 缩写**（如 `Reinforcement Learning, RL`）；产品名或专有名可仅写全称 | `（Reinforcement Learning, RL）`、`（Simulation and Platform Ecosystem）` |
| 后缀 | 固定为半角空格 + `社区` | ` 社区` |

完整示例：`强化学习（Reinforcement Learning, RL） 社区`、`规模化运动跟踪（Supersizing Motion Tracking for Natural Humanoid Control, SONIC） 社区`。

### 维护方式

1. **优先**在 `scripts/generate_link_graph.py` 的 `COMMUNITY_NAME_OVERRIDES` 中为枢纽页（hub）显式指定 `中文（English）` 基名；脚本会自动追加 ` 社区` 后缀。
2. 未命中 override 时回退为枢纽页 H1 标题 + ` 社区`，但 H1 风格不一（纯英文、英文在前等），**新增或变更社区划分后应检查并补 override**。
3. 兜底桶 `community-other` 固定为 `其他（Other） 社区`，与命名社区共用同一格式。
4. 运行 `make graph` 或 `make ci-preflight` 时，若某社区基名不符合 `中文（…）` 模式，脚本会打印 `WARNING`；CI 不因此失败，但维护者应补 override。

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
- 让图谱社区名直接沿用 wiki H1 而不检查是否符合「中文（English） 社区」格式
- 让机构 `label` 使用纯英文或「English（中文）」颠倒格式
