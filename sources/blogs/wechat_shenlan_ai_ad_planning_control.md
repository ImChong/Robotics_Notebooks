# 自动驾驶核心算法盘点｜规划与控制篇

> 来源归档（blog / 微信公众号）

- **标题：** 自动驾驶核心算法盘点｜规划与控制篇
- **类型：** blog
- **作者：** 深蓝AI / 深蓝学院（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s?__biz=MzY4NjA5NTgyMQ==&mid=2247602818&idx=1&sn=e6a0f914dcdd7878d5f4993d247fb85c
- **专栏专辑：** [《自动驾驶核心算法盘点》](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzY4NjA5NTgyMQ==&action=getalbum&album_id=4596755873481310212)（第 3 篇）
- **发表日期：** 2026-07-09
- **入库日期：** 2026-07-21
- **抓取方式：** [Agent Reach](https://github.com/Panniantong/Agent-Reach) v1.5.0 + [wechat-article-for-ai](https://github.com/bzd6661/wechat-article-for-ai)（Camoufox）；直连 `mp.weixin.qq.com` 遇「环境异常」CAPTCHA，经 **搜狗微信** `weixin.sogou.com` 中转签名链成功取正文；Jina Reader 不可用
- **一句话说明：** 按「感知 → 定位 → 规划 → 控制」栈定位规控；规划侧盘点 Hybrid A*、Frenet/Lattice、Apollo EM Planner，控制侧盘点 PID、LQR、MPC，强调量产是经典算法的场景化组合而非单一最优解。

## 核心摘录（归纳，非全文）

### 规控在栈中的位置

- 规划接收感知障碍物 + 高精地图全局路线，输出时空轨迹；控制输出转角/油门/刹车。
- 规划内部分层：全局路由 → 行为决策 → 运动规划（本文重点）。

### 规划三件套

| 算法 | 核心突破 | 适用场景 |
|------|----------|----------|
| **Hybrid A***（Dolgov et al., DARPA 2008） | 状态含航向；按车辆运动学扩展边；网格剪枝 + 非完整约束启发 | APA 泊车、狭窄掉头、非结构化避障 |
| **Lattice / Frenet**（Werling et al., ICRA 2010） | 沿参考线拉直道路；横/纵采样候选轨迹族；代价函数打分 | 车道保持、高速变道、跟车 |
| **EM Planner**（Apollo, arXiv:1807.08048） | 路径–速度解耦；SL/ST 上 DP 粗搜 + QP 精修 | 城市动态交通工业级规控 |

### 控制三件套

| 算法 | 角色 | 要点 |
|------|------|------|
| **PID** | 纵向速度基线 | P/I/D 直观；高速横向易蛇行 |
| **LQR** | 横向主力 | 二自由度模型 + Riccati 反馈；跟线精度 vs 打方向平稳 |
| **MPC** | 约束极限工况 | 滚动时域在线优化；硬约束友好，算力要求高 |

### 收束判断

- 量产常见组合：低速 Hybrid A*；巡航 Lattice/EM；纵向 PID 兜底、横向 LQR、极限避障 MPC。
- 端到端/RL 在渗透规控，但安全舒适的物理约束与经典折中仍是必修。

## 一手论文索引（文内）

1. Dolgov et al., Practical Search Techniques in Path Planning for Autonomous Driving, AAAI Workshop 2008（Hybrid A*）
2. Werling et al., Optimal trajectory generation for dynamic street scenarios in a Frenet frame, ICRA 2010
3. Fan et al., Baidu Apollo EM Motion Planner, arXiv:1807.08048

## 对 wiki 的映射

- [autonomous-driving-core-algorithms-series](../../wiki/overview/autonomous-driving-core-algorithms-series.md)（专辑父节点）
- [navigation-slam-autonomy-stack](../../wiki/overview/navigation-slam-autonomy-stack.md)
- [lqr-ilqr](../../wiki/methods/lqr-ilqr.md)、[model-predictive-control](../../wiki/methods/model-predictive-control.md)
- [python-robotics](../../wiki/entities/python-robotics.md)（文内配图引 Atsushi Sakai LQR 示例）

## 可信度与使用边界

- 微信「盘点」策展体例；算法细节与引用量以论文原文为准。
- 原始抓取正文见 [wechat_shenlan_ai_ad_planning_control_2026-07-09.md](../raw/wechat_shenlan_ai_ad_planning_control_2026-07-09.md)。
