# 技术栈项目下一阶段执行清单 v2

最后更新：2026-04-12
项目仓库：<https://github.com/ImChong/Robotics_Notebooks>
关联项目：<https://github.com/ImChong/Humanoid_Robot_Learning_Paper_Notebooks>
上一版清单：[`docs/tech-stack-next-phase-checklist-v1.md`](tech-stack-next-phase-checklist-v1.md)

## 当前项目状态判断

和 v2 刚建立时相比，项目状态又往前推进了一大步。

现在更准确的判断是：

1. **核心主干 wiki、roadmap、references、exports 已经形成可串联的知识骨架**
2. **页面级聚合导出不再只是 schema / preview，而是已经开始驱动真实页面**
3. **`detail.html?id=...` 和 data-driven `tech-map.html` 已经跑通，页面层闭环初步建立**
4. **当前最主要的短板，已经从“有没有页面级导出”切换成“如何继续补真实页面能力与正文消费层”**

一句话：

> 现在项目的主要矛盾，已经从“把知识结构导出来”切换成“如何把导出层真正推进为稳定、可扩展、可继续深挖的网站页面体系”。

## 已完成的主干内容

### 已完成的核心概念 / 方法页
- [x] `LIP / ZMP`
- [x] `Centroidal Dynamics`
- [x] `TSID`
- [x] `State Estimation`
- [x] `System Identification`
- [x] `Trajectory Optimization`
- [x] `Floating Base Dynamics`
- [x] `Contact Dynamics`
- [x] `Capture Point / DCM`

### 已完成的第一批关键实体页
- [x] `Isaac Gym / Isaac Lab`
- [x] `MuJoCo`
- [x] `legged_gym`
- [x] `Pinocchio`
- [x] `Crocoddyl`
- [x] `Unitree`

### 当前主线已经能串起来
当前已经初步具备这条知识链：

```text
LIP / ZMP
  ↓
Floating Base Dynamics
  ↓
Contact Dynamics
  ↓
Capture Point / DCM
  ↓
Centroidal Dynamics
  ↓
Trajectory Optimization / MPC
  ↓
TSID / WBC
  ↓
State Estimation / System Identification / Sim2Real
```

这意味着：
- 再继续线性补散页，收益已经明显下降
- 接下来更该做的是“让已有主线连接到论文、代码、工具、资源入口”

## 下一阶段总目标

把 `Robotics_Notebooks` 从“已经有页面级导出能力的技术栈知识库”，推进到：

> **一个已经有统一 detail route、技术栈页、后续还能继续扩展正文与导航能力的数据驱动机器人知识网站。**

## 下一阶段优先级重排

## P0，最高优先级，继续完善页面消费层

### 1. 巩固 detail page，决定正文消费策略
目标：让 `detail.html?id=...` 不只是 metadata 演示，而是稳定的一类真实页面。

待办：
- [x] 明确 detail page 不再停留在纯 metadata-first，而是进入 content-backed detail page
- [x] 采用最小前端 markdown 渲染方案，继续直接消费 `content_markdown`
- [x] 明确正文同步后的最小模板：无构建链路、无额外依赖、先支持基础 markdown 可读性
- [x] 给 detail page 增加最小公式显示，提升含公式条目的可读性
- [x] 给长文 detail page 增加最小目录导航与标题锚点
- [~] 明确哪些页面类型优先支持正文（当前已覆盖 wiki / entity / reference，tech-map 节点仍允许为空正文）

完成标准：
- detail page 的下一阶段路线被明确，不再停留在试验态
- detail page 的正文展示从“能同步”推进到“基础可读”
- 含公式的 detail page 已具备最小公式高亮显示能力
- 长文 detail page 已具备最小 TOC + 锚点跳转能力

### 2. 继续把 tech-map / roadmap / module page 做成真实页面体系
目标：不只验证单页能渲染，而是形成真正可浏览的网站结构。

待办：
- [x] 为 tech-map 增加最小 layer filter / 分组导航
- [x] 让 tech-map 当前筛选状态同步到 URL 查询参数，支持刷新与分享链接保留状态
- [x] 让 tech-map 节点按 layer 分组并支持最小折叠
- [x] 将 module page 从 preview 升级为真实页面
- [x] 将 roadmap page 从 preview 升级为真实页面
- [x] 统一 detail / tech-map / roadmap / module 之间的跳转方式

完成标准：
- 页面之间不是松散 demo，而是统一的信息架构

## P1，第二优先级，继续补 references / sources / wiki 联动

### 3. 把 references / sources 的深挖入口补到更多关键页
目标：保证页面层增长的同时，知识层不会断开到外部资源。

待办：
- [ ] 为当前控制主线补齐剩余 references 入口
- [ ] 为关键实体页补齐 docs / repo / project 入口
- [ ] 为学习主链补 benchmark / project / course 延伸入口
- [ ] 给 `sources/` 增加“已提炼 / 待提炼”的状态感

完成标准：
- 用户不只会“看页面”，还能顺着进入论文 / repo / docs 继续深挖

### 4. 稳定导出层与页面层的部署链路
目标：保证页面和数据不是本地能跑、上线就断。

待办：
- [ ] 持续维护 `docs/exports/` 镜像导出
- [ ] 明确页面统一读取 `docs/exports/site-data-v1.json`
- [ ] 检查后续新增页面类型是否仍能复用现有导出结构
- [ ] 必要时补充导出字段而不是临时在前端硬编码

## P2，第三优先级，继续第二批扩展内容

### 5. 第二批实体页 / 方法页 / 概念页补充
目标：在不打散主线的前提下继续扩展完整度。

待办：
- [ ] `Policy Optimization Methods (PPO / SAC / TD3)`
- [ ] `Diffusion Policy`
- [ ] `Model-Based RL`
- [ ] `Privileged Training`
- [ ] `Loco-manipulation`
- [ ] `Balance Recovery`
- [ ] `Jump / Hopping`

## 当前最明确待办
- [x] 审视 `references/` 当前结构
- [x] 审视 `sources/` 当前结构
- [x] 明确 references / sources / wiki 三层职责边界
- [~] 为当前控制主线建立 references 入口
- [~] 为当前工具实体页建立 docs / repo / project 入口
- [x] 提升 `references/` 目录自身的总入口可导航性
- [x] 设计导出层最小 schema 草案
- [x] 明确 `id / tags / related` 的生成规则
- [x] 生成第一版导出样例
- [x] 实现第一版最小导出脚本并批量导出核心页面
- [x] 将 `tech-map` 节点正式纳入导出层
- [x] 优化第一版导出质量（summary / tags / related）
- [x] 定义网页消费层字段设计
- [x] 生成页面级聚合导出 `site-data-v1.json`
- [x] 建立 `docs/exports/` 镜像导出，打通 GitHub Pages 消费链路
- [x] 新增 `docs/detail.html`，跑通 metadata-first detail page
- [x] 将 `docs/tech-map.html` 推进为 data-driven 页面，并接入 detail route
- [x] 决定 detail page 下一阶段先支持最小 markdown 正文同步，并升级为基础 markdown 渲染
- [x] 为 detail page 增加最小 TOC / 锚点导航，提升长文可用性
- [x] 将 module page 从 preview 升级为真实页面（`module.html?id=...`）
- [x] 将 roadmap page 从 preview 升级为真实页面（`roadmap.html?id=...`）
- [x] 为 tech-map 增加最小 layer filter / 分组导航
- [x] 让 tech-map 当前筛选状态同步到 URL 查询参数，支持刷新与分享链接保留状态
- [x] 让 tech-map 节点按 layer 分组并支持最小折叠

## 我建议的实际执行顺序
1. 先明确 detail page 是否继续 metadata-first
2. 如果需要正文，再设计 markdown 正文同步到 `docs/` 的最小方案
3. 然后给 tech-map 增加最小 filter / 分组导航
4. 再决定 module page / roadmap page 是否从 preview 升级为真实页面
5. 页面层稳定后，再回头继续补 references / docs / repo 深挖入口

## 当前阶段变化说明
- v1 的重点：补主干 wiki 页面
- v2 前半段的重点：结构联动、入口优化、路线执行化
- v2 中段的重点：把主干内容接到 references / sources / tools / projects
- v2 当前阶段的重点：**把页面级导出真正推进为 detail page / tech-map page 等真实数据驱动页面**

## references / sources / wiki 三层职责边界

经过当前审视，三层职责可以明确为：

### `sources/` = 原始资料输入层
作用：
- 新资料先扔这里
- 原始课程、论文、视频、教程、博客、数据、工具入口的原材料层
- 偏“待提炼、待吸收”

不做什么：
- 不负责给出结论
- 不负责组织完整知识结构
- 不直接承担主导航职责

### `references/` = 继续深挖入口层
作用：
- 已经知道一个概念后，下一步应该看哪些论文、benchmark、repo、生态项目
- 偏“按主题整理的外部延伸入口”
- 服务于已有 wiki / roadmap 主线

不做什么：
- 不重复讲 wiki 里的概念解释
- 不退化成纯链接堆
- 不承担原始资料归档职责

### `wiki/` = 结构化知识层
作用：
- 解释概念 / 方法 / 任务 / 比较 / 路线
- 给出主线、定义、关系、理解框架
- 是当前项目最核心的一层

不做什么：
- 不把资源全堆进正文
- 不把页面写成 bibliography

## 当前审视结论

### `references/` 当前状态
优点：
- 已经分成 `papers/`、`repos/`、`benchmarks/` 三个大类
- 结构方向是对的

当前短板：
- 和现有 wiki 主线的映射还不够显式
- 用户还不能从某个概念页自然跳到对应 references 入口

### `sources/` 当前状态
优点：
- 已经有输入资料层意识
- 也明确了“先 sources，再 wiki”的原则

当前短板：
- 主题命名还偏原材料收纳
- 和 references / wiki 的界面还没完全打清楚
- 缺少“哪些 sources 已经被提炼、哪些还没被提炼”的状态感

## 本次推进记录
- 2026-04-11：已完成 `tech-map/overview.md` 重写。
- 2026-04-11：已完成 `tech-map/dependency-graph.md` 重写。
- 2026-04-11：已完成 `roadmap/route-a-motion-control.md` 重写。
- 2026-04-11：已完成两个 learning path 重写。
- 2026-04-11：已完成 README 重写为真正入口页。
- 2026-04-11：已完成 index.md 重写为真正导航页。
- **V2 P0（结构联动）已全部完成。**
- 2026-04-11：已建 `docs/content-backlog.md`（P0/P1/P2 分级待办 + 质量标准）。
- 2026-04-11：已建 `docs/change-log.md`（与旧 `log.md` 区分的维护日志）。
- 2026-04-11：已在 `schema/page-types.md` 末尾新增“新增页面最低质量标准”章节。
- **V2 P1（规范化扩展）已全部完成。**
- 2026-04-11：已补齐第一批关键实体页：Isaac Gym / Isaac Lab、MuJoCo、legged_gym、Pinocchio、Crocoddyl、Unitree。
- 2026-04-11：已补齐第一批关键缺失概念页：Floating Base Dynamics、Contact Dynamics、Capture Point / DCM。
- 当前阶段判断已更新：下一步应从“继续补概念”切换到“建立 references / sources / wiki 的三层联动”。
- 2026-04-11：已完成对 `references/` 与 `sources/` 现状的审视，并明确三层职责边界：`sources/` 负责原始输入，`references/` 负责继续深挖入口，`wiki/` 负责结构化知识主线。
- 2026-04-11：已为第一批主线页补“继续深挖入口”，当前覆盖 `Locomotion`、`Sim2Real`、`Reinforcement Learning`、`Whole-Body Control`、`Isaac Gym / Isaac Lab`、`MuJoCo`。
- 2026-04-11：已为第二批主线页补“继续深挖入口”，当前新增覆盖 `Centroidal Dynamics`、`TSID`、`Trajectory Optimization`、`System Identification`。
- 当前主线的大部分关键页已经能顺着跳到 references / repos / benchmarks。
- 2026-04-11：已重写 `references/README.md`，从目录说明升级为 references 总入口页（快速入口 / 三个子目录职责 / 主线深挖入口 / 和 wiki / sources 的边界）。
- 2026-04-11：已重写 `references/papers/README.md`、`references/repos/README.md`、`references/benchmarks/README.md`，让 references 三个子目录都具备“适合谁看 / 快速入口 / 主线对应关系”。
- 当前 `references/` 已从目录树升级为初步可导航层。
- 2026-04-11：已建立 `exports/minimal-schema-v1.md`，定义当前阶段最小可用导出层 schema，覆盖 `wiki_page`、`roadmap_page`、`entity_page`、`reference_page`、`tech_map_node` 五类对象。
- 2026-04-11：已在导出层 schema 草案中明确 `id / tags / related` 的生成规则，当前采用“路径稳定命名 + 半规则标签 + 显式链接优先 related”的最小策略。
- 2026-04-11：已生成 `exports/sample-export-v1.json`，用一批核心页面验证最小 schema 的可落地性，当前覆盖 `Centroidal Dynamics`、`Reinforcement Learning`、`Locomotion`、`MuJoCo`、`Route A`、`Locomotion RL references`。
- 2026-04-11：已实现 `scripts/export_minimal.py`，并生成 `exports/index-v1.json`，当前可批量导出核心 markdown 页面为结构化数据。
- 2026-04-11：已将 `tech-map` 正式纳入导出层，当前 `index-v1.json` 共导出 63 个对象，其中包含 18 个 `tech_map_node`。
- 2026-04-11：已优化第一版导出质量，当前 `summary` 优先取定义句，`tags` 从标题+路径扩展到正文线索，`related` 优先提取“关联页面 / 继续深挖入口 / 关联任务”区块，再补一般显式链接。
- 2026-04-11：已建立 `exports/web-consumption-v1.md`，定义网页消费层的 5 类页面（`home_page`、`module_page`、`roadmap_page`、`tech_map_page`、`detail_page`）以及它们如何从 `exports/index-v1.json` 取字段。
- 2026-04-11：已扩展 `scripts/export_minimal.py`，在对象池 `exports/index-v1.json` 之外新增页面级聚合导出 `exports/site-data-v1.json`。
- 2026-04-11：`site-data-v1.json` 当前已覆盖首页、6 个模块页、全部路线页、tech-map 页和全量 detail pages，可直接作为前端第一阶段消费层。
- 2026-04-12：已新增 `docs/site-data-preview.html`，开始直接消费 `site-data-v1.json` 验证首页 / 模块页 / 路线页的最小网页渲染。
- 2026-04-12：已继续扩展预览页，开始验证 `detail_pages` 和 `tech_map_page` 的最小网页渲染。
- 2026-04-12：已在 `scripts/export_minimal.py` 中增加 `docs/exports/` 镜像导出，修正 GitHub Pages 只部署 `docs/` 时的导出数据可访问问题。
- 2026-04-12：已新增 `docs/detail.html`，并扩展 `docs/main.js` / `docs/style.css`，跑通 metadata-first detail page。
- 2026-04-12：已新增 `tests/test_detail_page.py`，验证 detail page 骨架与渲染器接入点。
- 2026-04-12：已将 `docs/tech-map.html` 从静态说明页推进为真正消费 `tech_map_page` 的 data-driven 页面，并统一接到 `detail.html?id=...`。
- 2026-04-12：已新增 `tests/test_tech_map_page.py`，验证 tech-map 页面挂载点与渲染器接入点。
- 2026-04-12：已为 `detail_pages` 新增 `content_markdown` 字段，并让 `detail.html` 开始展示来自源 markdown 的最小正文同步内容。
- 2026-04-12：已新增 `tests/test_content_sync.py`，验证正文同步字段、挂载点与前端渲染器接入点。
- 2026-04-12：已新增 `docs/module.html`，开始直接消费 `module_pages`，建立真实的 `module.html?id=...` 路由。
- 2026-04-12：已扩展 `docs/main.js`，新增 module page 渲染器，并让首页模块入口跳到统一 module route。
- 2026-04-12：已新增 `tests/test_module_page.py`，验证 module page 挂载点与渲染器接入点。
- 2026-04-12：已重写 `docs/roadmap.html`，开始直接消费 `roadmap_pages`，建立真实的 `roadmap.html?id=...` 路由。
- 2026-04-12：已扩展 `docs/main.js`，新增 roadmap page 渲染器，并让首页“开始看路线”跳到统一 roadmap route。
- 2026-04-12：已新增 `tests/test_roadmap_page.py`，验证 roadmap page 挂载点与渲染器接入点。
- 2026-04-12：已为 `docs/tech-map.html` 增加最小 layer filter / 分组导航，并扩展 `docs/main.js` 支持按 layer 过滤节点。
- 2026-04-12：已新增 `tests/test_tech_map_filter.py`，验证 tech-map filter 挂载点与渲染函数接入点。
- 当前阶段判断已更新：detail / module / roadmap / tech-map 四类核心页面都已进入可浏览状态；下一步重点转向是否还要细化筛选能力与信息架构。

## 维护规则
以后优先维护这个 v2 文件。
当项目阶段发生明显变化时，再继续升级版本，而不是不断把不同阶段的目标硬塞在同一版清单里。
README 应固定链接到当前生效版本。

## 状态约定
- `[ ]` 未开始
- `[~]` 进行中
- `[x]` 已完成
- `[-]` 暂缓
