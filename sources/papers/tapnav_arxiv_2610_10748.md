# TAPNAV 论文来源归档

## 元数据

- **题目：** TAPNAV: Humanoid Navigation through Tactile Active Perception
- **方法：** Tactile Active Perception Navigation（TAPNAV / TapNav）
- **作者：** Huaze Liu, Zhenyu Wu, Jaehwi Jang, Junjie Sheng, Andrew Collins, Aaron Xie, Zhaoyuan Gu, Kaijie Zhu, Ding Jiang, Kevin Cai, Ye Zhao
- **单位：** Georgia Institute of Technology；Harvey Mudd College；Walton High School（HTML 版署名分组，逐人映射页面未完全明示）
- **预印本：** https://arxiv.org/abs/2610.10748
- **HTML：** https://arxiv.org/html/2610.10748v1
- **版本：** v1，2026-10-07，cs.RO；9 页 10 图
- **项目页：** https://tapnav.github.io/TAPNAV-Project/（归档见 [tapnav-project.md](../sites/tapnav-project.md)）
- **代码：** 项目页写 “Code: Coming soon”；截至 2026-10-12 未发布
- **入库日期：** 2026-10-12

## 论文要解决的问题

人形机器人在黑暗、烟雾或相机失效时无法依赖视觉定位；仅靠里程计与 IMU 会因漂移使位姿不确定性快速增长。TAPNAV 的思路是把“触碰已知结构”当作主动定位手段：在已知、静态平面图上，机器人用手部触觉探头接触墙面或障碍，用接触/未接触结果校正位姿。

## 方法与训练信息

1. **状态估计：** SE(2) 位姿（x, y, yaw）粒子滤波，N_p = 600；初始化围绕已知起点（位置 0.03 m/轴、偏航 1.5° 标准差）。预测步用带高斯噪声的里程计；重加权用触觉似然 + 偏航的 wrapped-Gaussian IMU 似然；N_eff < 0.5·N_p 时重采样。规划时取粒子的高斯近似（均值+协方差）。
2. **触觉似然：** 接触时惩罚停止点到地图表面的距离 |δ|；未接触时仅惩罚预测穿透（δ < 0）。
3. **触觉硬件：** 每只手装自制末端，TouchTronix SensX 160（20×8 taxel，有效面积 20×50 mm，上限 100 N）外覆 Ecoflex 00-50 硅胶垫；总激活量经逆指数模型（沿用 WT-UMI）换算法向力后阈值化为二值接触结果（阈值未给出）。
4. **全局规划：** 0.1 m 网格上的约束 A*；墙体最小间隙 0.2 m、单次位移 ≤ 0.4 m；每次位移约束位置不确定性 ≤ 0.18 m、航向 ≤ 20°；对每个候选传播协方差，在探测范围内有平面时做简化卡尔曼更新；估计位姿偏离 > 0.1 m 时重规划。
5. **局部规划（信息增益探触）：** 动作 a = (j, α)，j ∈ {左, 右}，α 在 [−40°, 110°] 离散并受手臂运动学约束；每个子目标最多 4 次探触。接触概率 p = Φ(−μ/σ)，σ² = HΣHᵀ + σ_control²；IG_c = ½ log(1 + HΣHᵀ/σ_c²)；EIG = p·IG_c + (1−p)·IG_f；最大 EIG < 0.15 nats 时停止探触。σ_c = 0.018 m，σ_control = 0.045 m。
6. **全身控制器：** PPO 训练的解耦式上/下半身策略，共享本体感知。上半身两阶段课程（先跟踪 IK 参考的末端位姿，再加入沿探触方向的弹簧-阻尼力场做力跟踪）；下半身跟踪落脚点指令，参考由 3D-LIPM + 三次样条摆动脚生成；控制频率 200/50 Hz。
7. **平台：** Unitree G1；不使用相机或 LiDAR。

## 论文报告的结果

- **仿真（MuJoCo）：** 四张地图（弯曲走廊、分隔房间、之字走廊、散落障碍房间），每图每方法 20 次试验（seed 0–19），1800 s 超时；基线为 Odometry-only、Random-touch、Sweep-touch。
- **成功率（地图 a/b/c/d）：** TapNav 95/100/100/85；Random-touch 90/70/75/75；Sweep-touch 50/50/30/25；Odometry-only 35/5/5/45。平均 95%、77.5%、38.8%、22.5%。
- **位置误差（cm，仅统计成功试验）：** TapNav 6.5/9.4/5.9/16.8；Random-touch 9.6/11.4/8.7/16.6；Sweep-touch 11.7/11.2/10.9/13.7；Odometry-only 23.4/36.5/13.2/28.5。地图 d 上 Sweep-touch 的位置误差反而更小（但成功率仅 25%）。
- **IMU 航向噪声敏感性：** 仅给图，正文定性称 TapNav 位置/偏航误差最低、成功率最高。
- **落脚点控制器 vs 速度控制器：** 各 30 次仿真试验，定性报告速度跟踪控制器随航向误差累积而漂移；指令速度上限 0.2 m/s。
- **真机：** 办公室走廊与厨房中 Unitree G1 绕过台面并连续转弯到达目标；**无试验次数、无误差/成功率统计**。

## 复现与解释边界

- 假设已知且静态的平面图；规划期把粒子信念近似成单峰高斯；作者列为未来工作：多峰信念、部分已知地图。
- 真机验证只是定性演示，不能据此外推成功率。
- 项目页代码“Coming soon”，触觉接触阈值、PPO 超参等未在可抓取的正文给出，复现需等待代码。

## Robotics_Notebooks 映射

- 详情节点：wiki/entities/paper-tapnav-humanoid-tactile-navigation.md
- 汇总页：wiki/overview/frontier-humanoid-quadruped-2026-10-05-10-09.md

## 一手来源

- arXiv 摘要：<https://arxiv.org/abs/2610.10748>
- arXiv HTML：<https://arxiv.org/html/2610.10748v1>
- 项目页：<https://tapnav.github.io/TAPNAV-Project/>
