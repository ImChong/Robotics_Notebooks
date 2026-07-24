---
type: comparison
tags: [path-planning, navigation, local-planning, motion-planning, mobile-robot, comparison, engineering-selection]
status: complete
updated: 2026-07-23
related:
  - ../methods/a-star.md
  - ../methods/dwa.md
  - ../methods/smooth-navigation-path-generation.md
  - ../entities/navigation2.md
  - ../entities/python-robotics.md
  - ../entities/humanoid-system-curriculum.md
sources:
  - ../../sources/courses/shenlan_humanoid_system_theory_practice.md
  - ../../sources/repos/python_robotics.md
  - ../../sources/repos/navigation2.md
  - ../../sources/blogs/wechat_shenlan_ai_ad_planning_control.md
  - ../../sources/courses/numerical_optimization_foundations_robotics.md
summary: "移动机器人分层导航规划选型：全局 A* 搜折线、局部 DWA 跟踪避障、平滑优化补动力学可行性三层如何各司其职并组合落地。"
---

# 移动机器人导航规划方法对比：全局搜索 · 局部避障 · 路径平滑

## 一句话定义

移动机器人的导航规划不是单一算法，而是一条 **分层流水线**：全局层用 [A\*](../methods/a-star.md) 在静态占据栅格上搜出起点到终点的最优 **折线**；局部层用 [DWA](../methods/dwa.md) 在速度空间内反应式地跟踪该折线并躲开瞬态/动态障碍；后处理层用 [平滑路径生成](../methods/smooth-navigation-path-generation.md) 把折线优化成曲率连续、车/人形真正跟得动的曲线。三层分别对应深蓝学院人形系统课第 4.2、4.3 节与数值优化课 2.4–2.5 节，职责互补而非互相替代——本页帮你判断某个导航问题该动哪一层。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| A\* | A-star | 带可采纳启发 \(f=g+h\) 的最优图搜索，全局层基线 |
| DWA | Dynamic Window Approach | 速度空间采样打分的反应式局部规划 |
| DWB | DWB Controller（Nav2） | DWA 的 Nav2 工程继承，生成器 + 批评器插件化 |
| RRT | Rapidly-exploring Random Tree | 采样式全局规划，高维/无栅格时替代 A\* |
| TEB | Timed Elastic Band | 优化式局部规划，狭窄通道/强约束下对照 DWA |
| Dijkstra | Dijkstra's Algorithm | \(h=0\) 的 A\* 特例，无启发引导 |
| Costmap | Cost Map | 带膨胀层的代价地图，全局与局部层共享 |
| NLP | Nonlinear Programming | 平滑层的控制点/样条系数优化建模 |
| SDF | Signed Distance Field | 平滑优化中障碍距离软惩罚的来源 |
| cmd_vel | Command Velocity | DWA 输出、下发运控的速度指令 |

## 核心维度对比

| 维度 | A\*（全局搜索） | DWA（局部避障） | 平滑优化（后处理） |
|------|----------------|-----------------|--------------------|
| **层次** | 全局层 | 局部层 | 后处理层 |
| **输入** | 静态占据栅格 + 起终点 | 全局折线 + 当前位姿 + costmap | 折线/采样几何路径 |
| **输出** | 拓扑可行折线路径 | 本周期 \((v,\omega)\) 速度指令 | 曲率连续的平滑轨迹 |
| **是否考虑动力学** | 否（纯几何格点） | 是（加速度/制动窗口 \(V_r\)） | 是（曲率/jerk 惩罚） |
| **是否避动态障碍** | 否（静态图上一次成型） | 是（每周期重规划、反应式） | 否（沿用输入路径几何） |
| **最优性** | 可采纳 \(h\) 下全局最优 | 仅当前窗口局部最优 | 代价意义下的软最优 |
| **计算模式** | 一次性搜索（十～百毫秒级小图） | 高频循环（常 10–20 Hz） | 离线/低频批优化 |
| **典型失败模式** | 代价地图脏、终点落膨胀区、忽略动力学 | U 形陷阱局部最优、贴墙切弯、打滑失配 | 权重失衡致偏离原路径或过度收缩 |

## 何时用哪层 / 分层配合

三层的关系是 **串联接力**，不是择一：

1. **[A\*](../methods/a-star.md) 给骨架**：在建好的静态占据栅格上搜出一条几何连通的折线，负责「大方向对不对、有没有绕死」。它不管机器人转弯半径，也不管路上突然走过来的行人——这些交给下游。若全局层直接失败，先查代价地图是否干净、终点是否落在膨胀层内，而不是怀疑 A\* 本身。

2. **[DWA](../methods/dwa.md) 跟线并兜住瞬态**：把 A\* 折线当参考，在当前速度可达的动态窗口 \(V_r=V_s\cap V_d\cap V_a\) 内采样 \((v,\omega)\) 轨迹，用朝向、间隙、速度多目标打分，每个控制周期完全重规划。它才是真正把 **加速度限幅、碰撞预测、动态障碍** 纳入同一评分的一层，也是「A\* + DWA」这个移动机器人最小可运行栈里负责安全执行的部分。遇到 U 形陷阱等局部最优，需回到全局层重规划或触发恢复行为。

3. **[平滑路径生成](../methods/smooth-navigation-path-generation.md) 补可行性**：A\* 折线在拐点处曲率不连续，差速/Ackermann/人形都跟不干净。平滑层通过无约束或软约束优化最小化 jerk/曲率/能量，把折线变成曲率连续、可跟踪的曲线，再交给 DWA 或 MPC 类跟踪器。它是几何后处理，本身不引入新的避障语义——避障仍靠 DWA 的实时窗口。

工程上三者常在 [Navigation2](../entities/navigation2.md) 里落地：`planner_server` 承载 A\*（NavFn/Smac），`controller_server` 承载 DWB（DWA 继承），平滑既可作为 planner 的后处理插件，也可内嵌进 Smac。教学直觉则可在 [PythonRobotics](../entities/python-robotics.md) 用动画分别建立 \(f=g+h\)、速度窗口采样与样条平滑的感觉。

## 常见误区

- **想用一层搞定全部**：把「搜索 + 避障 + 平滑」压成一步端到端，会同时丢掉全局最优性、实时性与可解释性。分层的意义正是让每层的失败模式可定位。
- **误以为 DWA 能替代建图/全局层**：没有静态地图和 A\* 骨架，DWA 只有局部视野，远程目标不可达，还容易陷 U 形陷阱。DWA 是「跟线 + 兜瞬态」，不是「找路」。
- **把平滑当避障**：平滑层只改几何、让路径可跟踪，不理解动态障碍。动态避障始终是 DWA（或 TEB/MPC）每周期重规划的职责，别指望一次平滑消除碰撞。
- **归因错层**：全局穿墙常是 costmap 未膨胀或坐标系错，而非 A\* 逻辑坏；DWA 撞障常是 footprint 偏小或 costmap 未更新，而非评分公式错。先分清是哪一层的输入脏了。

## 参考来源

- [深蓝学院人形系统课程大纲](../../sources/courses/shenlan_humanoid_system_theory_practice.md) — 第 4.2（A\*）、4.3（DWA）节导航规划
- [PythonRobotics 归档](../../sources/repos/python_robotics.md) — A\* / DWA / 路径平滑教学实现
- [Navigation2 归档](../../sources/repos/navigation2.md) — planner_server / DWB 工程栈
- [深蓝AI：规划与控制篇](../../sources/blogs/wechat_shenlan_ai_ad_planning_control.md) — 全局/局部分层背景
- [数值优化基础课程](../../sources/courses/numerical_optimization_foundations_robotics.md) — 第 2.4–2.5 节平滑导航路径

## 关联页面

- [A\* 全局路径规划](../methods/a-star.md) — 全局层：静态图上搜最优折线，本对比的「找路」一层
- [DWA 局部路径规划](../methods/dwa.md) — 局部层：速度窗口采样跟线并避动态障碍，本对比的「执行」一层
- [平滑路径生成](../methods/smooth-navigation-path-generation.md) — 后处理层：把折线优化成曲率连续可跟踪曲线
- [Navigation2](../entities/navigation2.md) — 三层落地的默认工程框架（planner_server + DWB + 平滑插件）
- [PythonRobotics](../entities/python-robotics.md) — 三层算法的教学级动画实现入口
- [人形系统课程策展](../entities/humanoid-system-curriculum.md) — A\* + DWA 分层导航所属的课程主线
