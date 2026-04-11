# 技术栈项目下一阶段执行清单 v2

最后更新：2026-04-11
项目仓库：<https://github.com/ImChong/Robotics_Notebooks>
关联项目：<https://github.com/ImChong/Humanoid_Robot_Learning_Paper_Notebooks>
上一版清单：[`docs/tech-stack-next-phase-checklist-v1.md`](tech-stack-next-phase-checklist-v1.md)

## 当前项目状态判断

和 v2 刚建立时相比，项目状态已经明显变化了。

现在更准确的判断是：

1. **核心主干 wiki 已经不仅是“基本成型”，而是已经能串出完整主线**
2. **tech-map / roadmap / README / index 已经形成可用入口体系**
3. **第一批关键实体页和缺失概念页也已补齐**
4. **当前最主要的短板，开始从“缺概念页”转向“缺继续深挖入口”和“缺 references / sources 联动”**

一句话：

> 现在项目的主要矛盾，已经从“主干内容不够”切换成“已有内容如何继续向论文、代码、工具和资源层展开”。

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

把 `Robotics_Notebooks` 从“已经有主干内容的技术栈知识库”，推进到：

> **一个不仅能讲清主线，还能把用户自然带到 references / sources / tools / papers 继续深挖的导航系统。**

## 下一阶段优先级重排

## P0，最高优先级，先做 references / sources / wiki 联动

### 1. 建立 references / sources 与现有主线模块的映射关系
目标：让已有 wiki 页面不只停在概念解释，而是能继续往下挖论文、开源代码、课程和工具。

待办：
- [ ] 审视 `references/` 当前结构
- [ ] 审视 `sources/` 当前结构
- [ ] 为当前主线模块建立 references 映射入口
- [ ] 为关键实体页建立工具 / repo / docs 映射入口
- [ ] 让 `references/` 和 `sources/` 明确区分职责

完成标准：
- 用户从任一主线概念页或实体页，都能顺着进入对应的论文 / 开源项目 / 资源入口

### 2. 建立“主线深挖入口”页面
目标：不是让用户自己到处找，而是给出下一步看什么。

待办：
- [ ] 按当前主线补“继续阅读入口”结构
- [ ] 为控制主链建立 paper / repo / tool 三类延伸入口
- [ ] 为学习主链建立 benchmark / project / course 三类延伸入口

完成标准：
- 任意一条主链都不止有 wiki 页面，还有继续深挖的结构化出口

## P1，第二优先级，准备导出层与网页消费层

### 3. 准备最小可用导出层
目标：为网页、脑图、未来可视化展示打基础。

待办：
- [ ] 设计最小可用导出 schema
- [ ] 导出 wiki 元信息
- [ ] 导出 roadmap 数据
- [ ] 导出 dependency 数据
- [ ] 导出 entity 元信息

### 4. 评估网页消费层需要什么字段
目标：不是急着做页面，而是先把数据层准备对。

待办：
- [ ] 列出网页首页需要字段
- [ ] 列出模块页需要字段
- [ ] 列出路线页需要字段
- [ ] 对照现有 markdown 看缺什么

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
- [~] 提升 `references/` 目录自身的总入口可导航性
- [x] 设计导出层最小 schema 草案
- [x] 明确 `id / tags / related` 的生成规则
- [x] 生成第一版导出样例
- [x] 实现第一版最小导出脚本并批量导出核心页面
- [x] 将 `tech-map` 节点正式纳入导出层

## 我建议的实际执行顺序
1. 先看 `references/` 现状
2. 再看 `sources/` 现状
3. 明确这两个目录和 `wiki/` 的职责边界
4. 先把控制主链接到 references / repo / docs
5. 再把实体页接到工具文档与开源项目
6. 然后开始做导出层 schema

## 当前阶段变化说明
- v1 的重点：补主干 wiki 页面
- v2 前半段的重点：结构联动、入口优化、路线执行化
- v2 当前阶段的重点：**把主干内容继续接到 references / sources / tools / projects**

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
- 下一步建议：继续提高导出质量，例如优化 `summary / tags / related` 的精度，或者开始定义网页消费层该如何读取 `index-v1.json`。

## 维护规则
以后优先维护这个 v2 文件。
当项目阶段发生明显变化时，再继续升级版本，而不是不断把不同阶段的目标硬塞在同一版清单里。
README 应固定链接到当前生效版本。

## 状态约定
- `[ ]` 未开始
- `[~]` 进行中
- `[x]` 已完成
- `[-]` 暂缓
