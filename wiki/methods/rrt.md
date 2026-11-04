---
type: method
tags: [path-planning, sampling-based, rrt, motion-planning]
status: complete
updated: 2026-11-04
related:
  - ./rrt-star.md
  - ./rrt-connect.md
  - ./informed-rrt-star.md
  - ../concepts/configuration-space.md
  - ../entities/ompl.md
  - ../entities/moveit2.md
  - ../methods/a-star.md
  - ../comparisons/robot-path-planning-five-paradigms-taxonomy.md
sources:
  - ../../sources/blogs/wechat_shenlan_rrt_rrt_star_explainer_2026-11-04.md
summary: "RRT（快速探索随机树）在连续高维配置空间里随机采样、向最近树节点扩展并做碰撞检测，先解决「有没有可行路径」；概率完备但不保证最优、不保证固定时间内找到。"
---

# RRT（Rapidly-exploring Random Tree）

> 依据 [深蓝AI《彻底搞懂RRT与RRT*》](https://mp.weixin.qq.com/s/EeLwGT_pBUoEnIlsQObYdg)（2026-11-04）整理；一手论文为 LaValle 1998（文章列出，PDF 文本不可机读，细节未独立核对）。

## 一句话定义

**RRT 不预先把整张图搭好，而是从起点让一棵树在空间里随机探索、逐步长向目标，先解决「找到一条可行路」而非「找到最短路」。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RRT | Rapidly-exploring Random Tree | 快速探索随机树 |
| C-space | Configuration Space | 配置空间，规划变量是各关节角等状态而非末端位置 |
| Goal Bias | Goal Bias | 偶尔直接把终点作采样目标 |

## 为什么重要

- 多关节机器人在配置空间里难以枚举：文章举例每关节取 20 个角度、6 个关节就有约 6400 万状态（**仅为帮助理解规模的假设**）；RRT 提供「只探索一部分」的思路，适用于连续、高维、甚至带运动约束的问题（并不是说网格搜索失效）。

## 主要技术路线

### 四步循环（文章）

1. **Sample**：在允许空间随机采样一个点（不必是目标）；
2. **Nearest**：在已有树上找按规划空间距离度量最近的节点（不一定是二维直线距离）；
3. **Steer**：从最近节点朝采样点走一小段，控制扩展尺度；带运动约束时需用局部运动模型或控制输入生成可行运动；
4. **Collision Check**：新运动无碰撞才把新状态与边加入树，否则放弃重采。

某分支连到目标区域后，沿父节点回溯得到路径。

### 性质（文章表述）

- **Voronoi 偏置**：面向未探索区域的节点吸引更大范围的随机样本，因此「随机采样 + 最近节点扩展」自然鼓励树向空白区生长。
- **概率完备性**：在常见可行性与采样条件下，若存在足够稳健的可行路径，采样越多找到解的概率趋近 1；**不保证固定时间内一定找到，更不保证第一条最优**。
- **狭窄通道**是典型难题：门缝存在，但随机样本很难落在正确位置与方向。
- **目标偏置**能减少对运气的依赖，但过强会让树反复向被障碍挡住的方向尝试、削弱探索。
- 局限：连上之后通常不再重新审视「把老树枝接到别处会不会更短」，因此路径可能绕远、拐弯多。

## 部署要补上的东西（文章）

- 状态空间：位置 + 朝向、关节角、再加速度等；「距离最近」「向采样点扩展」都依赖状态表示。
- 碰撞检测：新节点无碰撞不等于整段运动安全，需检查连续运动或足够可靠的离散近似，并考虑机器人尺寸与安全余量。
- 运动约束：车辆最小转弯半径、关节速度加速度限制需要符合运动学 / 动力学的局部扩展器（文章配图用 Dubins 曲线连边的 Dubins-RRT）。
- 路径 → 轨迹：路径平滑、时间参数化、速度规划与反馈控制；环境变化时重规划或动态避障。

## 关联页面

- [RRT*](./rrt-star.md) — 在 RRT 基础上持续改进路径代价
- [RRT-Connect](./rrt-connect.md)、[Informed RRT*](./informed-rrt-star.md)
- [A*](./a-star.md) — 离散图搜索对照
- [配置空间](../concepts/configuration-space.md)
- [OMPL](../entities/ompl.md)、[MoveIt 2](../entities/moveit2.md)
- [路径规划五范式分类](../comparisons/robot-path-planning-five-paradigms-taxonomy.md)

## 参考来源

- [文章归档](../../sources/blogs/wechat_shenlan_rrt_rrt_star_explainer_2026-11-04.md)
- LaValle, *Rapidly-Exploring Random Trees: A New Tool for Path Planning*, 1998：<https://msl.cs.illinois.edu/~lavalle/papers/Lav98c.pdf>
- [OMPL 库 README 列出的 RRT-Connect 等 40 多种采样规划算法](../entities/ompl.md)
