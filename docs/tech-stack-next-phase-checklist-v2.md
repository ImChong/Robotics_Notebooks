# 技术栈项目下一阶段执行清单 v2

最后更新：2026-04-11
项目仓库：<https://github.com/ImChong/Robotics_Notebooks>
关联项目：<https://github.com/ImChong/Humanoid_Robot_Learning_Paper_Notebooks>
上一版清单：[`docs/tech-stack-next-phase-checklist-v1.md`](tech-stack-next-phase-checklist-v1.md)

## 当前项目状态判断

和 v1 阶段相比，项目已经不是“只有骨架”的状态了。

现在更准确的判断是：

1. **核心主干 wiki 已经基本成型**
2. **页面数量已经开始形成主线知识链**
3. **真正的短板，开始从“缺页面”转向“缺结构联动”和“缺执行入口”**

一句话：

> 现在项目的主要矛盾，已经从“内容不够”切换成“结构怎么联起来、入口怎么更好用、路线怎么更可执行”。

## 已完成的主干内容

### 已完成的核心概念 / 方法页
- [x] `LIP / ZMP`
- [x] `Centroidal Dynamics`
- [x] `TSID`
- [x] `State Estimation`
- [x] `System Identification`
- [x] `Trajectory Optimization`

### 当前主线已经能串起来
当前已经初步具备这条知识链：

```text
LIP / ZMP
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
- 再继续无脑补散页，收益已经下降
- 接下来更该做的是“把已有内容组织成可导航、可执行、可扩展的系统”

## 下一阶段总目标

把 `Robotics_Notebooks` 从“已经有主干内容的知识库草稿”，推进到：

> **一个能指导学习顺序、能提供模块导航、能继续稳定扩展的机器人技术栈系统。**

## 下一阶段优先级重排

## P0，最高优先级，先做结构联动

### 1. 打通 tech-map 和 wiki 的映射
目标：让技术栈地图不只是标题列表，而是真正能跳到知识页。

待办：
- [x] 更新 `tech-map/overview.md`，补一级模块说明
- [x] 更新 `tech-map/dependency-graph.md`，明确主线依赖关系
- [x] 为关键模块补对应 wiki 链接
- [ ] 明确模块卡片与 wiki 页的双向映射规则

完成标准：
- 用户从 tech-map 任一主模块出发，都能跳到对应的 wiki / roadmap / references

### 2. 强化 roadmap 的执行性
目标：把路线图从“知识导航”升级成“学习执行路线”。

待办：
- [ ] 更新 `roadmap/route-a-motion-control.md`
- [ ] 为阶段补齐：前置知识 / 核心问题 / 推荐阅读 / 产出物
- [ ] 更新 `roadmap/learning-paths/if-goal-locomotion-rl.md`
- [ ] 更新 `roadmap/learning-paths/if-goal-imitation-learning.md`

完成标准：
- 路线图能直接回答：现在先学什么，下一步学什么，学完输出什么

### 3. 升级 README 和 index 为真正入口
目标：让项目首页第一次打开就知道怎么用。

待办：
- [ ] 重写 README 的“适合谁看 / 怎么使用 / 从哪里开始”
- [ ] 更新 `index.md`，让它更像总导航页而不是简单索引
- [ ] 明确 wiki / roadmap / tech-map / references 的边界

完成标准：
- 新读者打开 README 和 index，就能迅速找到起点和路径

## P1，第二优先级，开始规范化扩展

### 4. 建立统一页面模板规范
目标：控制后续扩展质量，避免页面风格越来越散。

待办：
- [ ] 建 wiki 页面模板规范
- [ ] 建 roadmap 页面模板规范
- [ ] 建 tech-map 模块卡片模板规范

建议统一字段：
- 一句话定义
- 它解决什么问题
- 为什么重要
- 前置知识
- 相关方法
- 相关任务
- 推荐资源
- 相关项目/工具
- 延伸阅读

### 5. 建立 backlog 和迭代记录
目标：让后续扩展变成可持续维护，而不是想到哪写到哪。

待办：
- [ ] 建 `docs/content-backlog.md`
- [ ] 建 `docs/change-log-next.md`
- [ ] 定义新增页面的最低质量标准
- [ ] 定义页面之间必须互链的规则

### 6. 开始补关键实体页
目标：把“概念和方法”继续接到“工具和生态”。

待办：
- [ ] `Isaac Gym / Isaac Lab`
- [ ] `MuJoCo`
- [ ] `Unitree`
- [ ] `legged_gym`
- [ ] `OpenLoong / 人形机器人开源生态`

## P2，第三优先级，准备导出和展示层

### 7. 准备最小可用导出层
目标：为网页、脑图、未来可视化展示打基础。

待办：
- [ ] 设计最小可用导出 schema
- [ ] 导出 wiki 元信息
- [ ] 导出 roadmap 数据
- [ ] 导出 dependency 数据

### 8. 评估网页消费层需要什么字段
目标：不是急着做页面，而是先把数据层准备对。

待办：
- [ ] 列出网页首页需要字段
- [ ] 列出模块页需要字段
- [ ] 列出路线页需要字段
- [ ] 对照现有 markdown 看缺什么

## 当前最明确待办
- [x] 更新 `tech-map/overview.md`
- [x] 更新 `tech-map/dependency-graph.md`
- [ ] 更新 `roadmap/route-a-motion-control.md`
- [ ] 更新 `roadmap/learning-paths/if-goal-locomotion-rl.md`
- [ ] 更新 `roadmap/learning-paths/if-goal-imitation-learning.md`
- [ ] 更新 `README.md`
- [ ] 更新 `index.md`
- [ ] 建 `docs/content-backlog.md`
- [ ] 建统一模板规范文件

## 我建议的实际执行顺序
1. 先改 `tech-map/overview.md`
2. 再改 `tech-map/dependency-graph.md`
3. 再改 `roadmap/route-a-motion-control.md`
4. 然后补两个 learning path
5. 再统一 README / index 入口
6. 之后再做模板和 backlog
7. 最后再考虑导出层

## 本次版本升级说明
- v1 的重点是补主干 wiki 页面
- v2 的重点切换为：**结构联动、入口优化、路线执行化、维护规范化**
- 以后默认维护 v2，不再以 v1 作为主要执行看板

## 本次推进记录
- 2026-04-11：已完成 `tech-map/overview.md` 重写，从简单列表升级为“模块总览 + 知识入口 + 当前主攻主线”页面。
- 2026-04-11：已完成 `tech-map/dependency-graph.md` 重写，从箭头列表升级为“主线依赖 + 分层关系 + 横向桥接 + 推荐阅读顺序”页面。
- V2 第一项中的“overview / dependency / 关键模块链接”已基本打通。
- 下一步建议继续推进 `roadmap/route-a-motion-control.md`，把结构联动进一步落到学习执行路线。

## 维护规则
以后优先维护这个 v2 文件。
当项目阶段发生明显变化时，再继续升级版本，而不是不断把不同阶段的目标硬塞在同一版清单里。
README 应固定链接到当前生效版本。

## 状态约定
- `[ ]` 未开始
- `[~]` 进行中
- `[x]` 已完成
- `[-]` 暂缓
