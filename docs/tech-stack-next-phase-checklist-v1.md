# 技术栈项目下一阶段执行清单 v1

最后更新：2026-04-11
项目仓库：<https://github.com/ImChong/Robotics_Notebooks>
关联项目：<https://github.com/ImChong/Humanoid_Robot_Learning_Paper_Notebooks>

## 一句话目标
把 `Robotics_Notebooks` 从“有结构的知识草稿”继续推进成“可持续迭代的机器人技术栈导航系统”，优先服务你当前的人形机器人运动控制、强化学习、模仿学习主线。

## 当前判断
目前项目的骨架已经有了，但还缺 3 个真正决定可用性的东西：
1. 核心 wiki 覆盖度还不够
2. tech-map / roadmap / wiki 三层联动还不够强
3. 对外展示层还没形成稳定输出

## 下一阶段总原则
- 先补主干，不急着铺太宽
- 先打通导航关系，再继续堆内容
- 优先围绕“人形机器人运动控制 / RL / IL”主线
- 每一项都尽量产出可复用模板，而不是一次性内容

## P0，先做，最高优先级

### 1. 补齐核心 wiki 主干页
目标：把当前主线讲清楚，形成连续知识路径。

待办：
- [x] 补 `LIP / ZMP`
- [x] 补 `Centroidal Dynamics`
- [ ] 补 `TSID`
- [ ] 补 `State Estimation`（面向运动控制）
- [ ] 补 `System Identification`（面向 sim2real）
- [ ] 补 `Trajectory Optimization`

完成标准：
- 每页都有：一句话定义、为什么重要、核心公式/概念、在机器人里的作用、和已有页面的关联跳转

### 2. 打通 tech-map 和 wiki 的映射
目标：不是只有页面，而是形成“从地图到知识页”的导航闭环。

待办：
- [ ] 给 `tech-map/overview.md` 补一级模块说明
- [ ] 给 `dependency-graph.md` 明确主线依赖关系
- [ ] 为每个关键模块补对应 wiki 链接
- [ ] 定义“模块卡片”和 wiki 页的双向关系

完成标准：
- 用户从 tech-map 点进任一主模块，都能跳到对应 wiki 或 roadmap

### 3. 强化 roadmap 的执行性
目标：路线图不是概念图，而是能直接指导学习顺序。

待办：
- [ ] 给 `route-a-motion-control.md` 增加阶段划分
- [ ] 每阶段补“前置知识 / 核心问题 / 推荐阅读 / 输出物”
- [ ] 给 `if-goal-locomotion-rl.md` 增加更明确的顺序建议
- [ ] 给 `if-goal-imitation-learning.md` 增加与运动控制主线的衔接

完成标准：
- 每条路线都能回答“现在该学什么，下一步学什么，学完输出什么”

## P1，紧接着做

### 4. 增加关键实体页
目标：把理论和工具生态连起来。

待办：
- [ ] 补 `Isaac Gym / Isaac Lab`
- [ ] 补 `MuJoCo`
- [ ] 补 `Unitree`
- [ ] 补 `legged_gym`
- [ ] 补 `OpenLoong / 人形机器人开源生态`（如适合）

### 5. 建立统一页面模板
目标：后续新增内容时不散、不乱。

待办：
- [ ] 给 wiki 页面模板补固定字段
- [ ] 给 roadmap 页面模板补固定字段
- [ ] 给 tech-map 模块卡片补固定字段

建议固定字段：
- 一句话定义
- 它解决什么问题
- 为什么重要
- 前置知识
- 相关方法
- 相关任务
- 推荐资源
- 相关项目/工具
- 延伸阅读

### 6. 首页文案和入口整理
目标：让项目首页第一次看就知道怎么用。

待办：
- [ ] 重写 README 的“适合谁看 / 怎么使用 / 从哪里开始”
- [ ] 把 `index.md` 做成真正的总导航页
- [ ] 明确区分 wiki、roadmap、tech-map、references 的职责

## P2，后续推进

### 7. 准备导出层
目标：以后能稳定喂给网页、脑图或别的展示层。

待办：
- [ ] 设计最小可用导出 schema
- [ ] 先导出 wiki 元信息
- [ ] 再导出 roadmap / dependency 数据
- [ ] 为网页展示准备统一字段

### 8. 建立内容维护机制
目标：不是做一波就停，而是能持续演进。

待办：
- [ ] 建 `docs/content-backlog.md`
- [ ] 建 `docs/change-log-next.md` 或类似迭代记录
- [ ] 定义新增页面时的最低质量标准
- [ ] 定义页面之间必须互链的规则

## 我建议的实际执行顺序
1. 先补 6 个核心 wiki 页
2. 再改 tech-map 映射关系
3. 再改 roadmap 可执行性
4. 然后补实体页
5. 最后再做导出层和展示层

## 当前明确待办
- [x] 产出 `LIP / ZMP` 页面
- [x] 产出 `Centroidal Dynamics` 页面
- [ ] 产出 `TSID` 页面
- [ ] 产出 `State Estimation` 页面
- [ ] 产出 `System Identification` 页面
- [ ] 产出 `Trajectory Optimization` 页面
- [ ] 更新 `tech-map/overview.md`
- [ ] 更新 `tech-map/dependency-graph.md`
- [ ] 更新 `roadmap/route-a-motion-control.md`
- [ ] 更新 `roadmap/learning-paths/if-goal-locomotion-rl.md`
- [ ] 更新 `roadmap/learning-paths/if-goal-imitation-learning.md`
- [ ] 建立统一模板规范
- [ ] 建立 backlog 文件

## 本次推进记录
- 2026-04-11：已新增 `wiki/concepts/lip-zmp.md`，并同步更新 `index.md` 索引。
- 2026-04-11：已新增 `wiki/concepts/centroidal-dynamics.md`，并同步更新 `index.md` 索引。
- 下一步建议直接继续补 `TSID`，这样 `Centroidal Dynamics → TSID / WBC` 这段控制落地链条就完整了。

## 维护规则
以后我会优先更新这个文件，而不是把方向散落在聊天里。
每次你让我继续推进 `Robotics_Notebooks`，我会默认先看并更新这份清单，再执行具体工作。

## 状态约定
- `[ ]` 未开始
- `[~]` 进行中
- `[x]` 已完成
- `[-]` 暂缓
