# SANDO 论文来源归档

## 元数据

- **题目：** SANDO: Safe Autonomous Trajectory Planning for Dynamic Unknown Environments
- **作者：** Kota Kondo、Jesús Tordesillas、Juan Rached、Lili Sun、Yixuan Jia、Jonathan P. How
- **机构：** 麻省理工学院航空航天系 / Aerospace Controls Laboratory（MIT ACL）；Jesús Tordesillas 另属 Comillas Pontifical University 的 TRAIL
- **预印本：** https://arxiv.org/abs/2604.07599
- **HTML：** https://arxiv.org/html/2604.07599v3
- **版本：** v3，2026-09-25，cs.RO
- **项目页：** https://kotakondo.github.io/projects/sando/
- **代码：** https://github.com/mit-acl/sando
- **演示视频：** https://youtu.be/_T10DJiLQXg
- **发表状态：** Kota Kondo 个人主页的 2026 年 7 月新闻称 T-RO 有条件接收；截至 2026-10-17，仓库 README 写作“Accepted to IEEE Transactions on Robotics”。保留两处来源表述，最终卷期 / DOI 以正式出版记录为准。
- **一句话说明：** 面向 3D 动态未知环境的 UAV 规划栈，以时空安全飞行走廊、热图全局规划与硬约束 MIQP 兼顾连续时间碰撞安全和在线重规划。

## 论文的核心问题

未知环境中的障碍位置与运动不能预先完整给定；动态障碍还会让刚生成的无碰撞路径很快失效。软惩罚规划器虽然快，但通常不能给出严格安全保证；硬约束方法虽能保证安全，却可能计算慢或过度保守。SANDO 针对狭窄、拥挤且有动态障碍的三维场景，将感知、跟踪、全局路径、时空走廊和轨迹优化组成闭环。

## 方法摘要

1. **动态障碍追踪：** 从时序占据栅格检测与聚类目标，用自适应扩展卡尔曼滤波器估计状态并预测未来运动。
2. **热图引导全局规划：** 静态与动态障碍周围构造风险代价；A* 将占据约束作为硬限制、将热图作为软代价，偏向有更大安全空间的路径。
3. **时空安全飞行走廊（STSFC）：** 沿路径和时间分层生成凸多面体；对动态障碍按各时间层的最坏可达集合膨胀，而不是对整个规划时域统一采用最坏情况。
4. **硬约束轨迹优化：** 在走廊内用 MIQP 进行轨迹段与多面体分配。变量消元减少决策量；论文报告优化耗时最高可加速 7.4 倍。
5. **失败回退：** 重规划无可行解时继续执行旧轨迹；若已没有可用轨迹则悬停。论文讨论移动子目标导致的递归可行性问题。

## 论文报告的证据与边界

- 标准化静态基准、障碍密集森林和动态障碍仿真中，作者报告 SANDO 在森林及动态基准的所有难度下均达到 100% 成功率，且未出现约束违反；感知闭环实验不使用障碍真值。
- 四旋翼真机使用机载规划、感知与定位。静态场景完成 6 次飞行，动态场景完成 12 次飞行；动态实验中没有发生碰撞。
- 动态真机平均重规划时间约 18–24 ms。实现以 100 Hz 触发重规划，但每轮完成后才开始下一轮，因此实际频率由总延迟限制。
- **形式保证有前提：** 论文定理依赖障碍速度/尺寸、估计误差和跟踪误差等有界假设，保证范围是单个规划时域的连续时间避碰。
- **真机不等于定理条件全部满足：** 作者报告机载目标估计误差约 0.30 m（行人实验 0.43 m），跟踪 RMS 误差约 0.21–0.57 m；据论文讨论，硬件上界与理想假设并不完全成立。因此应分别陈述“定理条件下保证”与“12 次动态真机无碰撞的经验结果”。
- **局限：** 移动子目标下尚无递归可行性保证；最坏可达集膨胀在密集环境中可能过度保守；对传感器估计与实际跟踪误差的建模还需加强。

## 对 wiki 的映射

- 主实体页：[SANDO](../../wiki/entities/paper-sando.md)
- 对照：[EGO-Planner Swarm](../../wiki/entities/ego-planner-swarm.md)、[MIGHTY](../../wiki/entities/paper-mighty-hermite-spline-trajectory-planning.md)
- 系统背景：[多旋翼仿真—规划—飞控栈总览](../../wiki/overview/multirotor-simulation-planning-control-stack.md)

## 一手来源

- arXiv v3 HTML：<https://arxiv.org/html/2604.07599v3>
- 官方项目页：<https://kotakondo.github.io/projects/sando/>
- 官方代码与运行说明：<https://github.com/mit-acl/sando>
- 官方演示：<https://youtu.be/_T10DJiLQXg>
