# LACART（arXiv:2611.09640）

> 来源归档（paper · 原文通读 · 支撑 `wiki/entities/paper-lacart.md`）

- **标题：** LACART: Lookahead-Augmented Collision Avoidance and Reference Tracking Strategies for Safety-Critical Robots
- **作者 / 机构（以论文首页为准）：** Pedro Lopes（INESC TEC，葡萄牙波尔图）；Amit Shivam、Luís Tiago Paiva（SYSTEC, ISR-Porto, ARISE-LA；波尔图大学工程学院 FEUP）
- **类型：** paper（cs.RO，arXiv v1，2026-11-06；正文约 4 页，体例为演示 / 摘要性短文，含「演示方案」一节）
- **arXiv：** <https://arxiv.org/abs/2611.09640>
- **PDF：** <https://arxiv.org/pdf/2611.09640v1>
- **代码：** <https://github.com/Kanestrene/autonomous-racing-cbf>（仅静态读 README，未核对与论文实验的对应关系；论文本身给出的是 github.up2speed.pt，未访问）
- **入库日期：** 2026-11-10
- **一句话说明：** 在名义路径跟踪器之后加一层前视点 CBF + 松弛 CLF 的二次规划安全滤波，把圆形障碍、赛道边界和参考跟踪放进同一个小 QP；仿真与 Huang 等 [9] 比较中位计算时间略低，并引用硕士论文中的 1:43 微型车实验。
- **核对范围：** 通读 arXiv v1 PDF 全部正文（摘要、§I–§VII、图 1–4、表 I、参考文献），图 1–2 另渲染页面图查看；全文约 4 页，没有附录或补充材料。未读被引的会议论文 [1]（APCA 2026，论文标注「已接收」）与硕士论文 [2]，所以硬件数据（表 I）与图 4 只能按本文转述核对，未回到原始来源；未读对比对象 [9]（Huang 等，IECON 2023）原文。
- **通读：** 已完成

## 核心摘录（均出自原文）

1. **定位（摘要、§I）：** 作者自述本文是对会议论文 [1] 与硕士论文 [2]（UP2Speed）的综合，并与 [9] 作比较；实际贡献写为「在一个小 QP 里集成转向敏感的障碍约束、赛道边界约束和松弛的参考跟踪，并给出硬件实现和对感知 / 执行限制的明确讨论」。
2. **前视点模型（§II，式 1）：** 单轮车模型，输入 u=(v, ω)；轴上前方距离 l>0 的点 p_l = p + l·t，其速度 ṗ_l = v·t + l·ω·n，因此角速度 ω 通过 l·ω·n 进入点的速度。
3. **障碍屏障（§II，式 2）：** h_i = ‖d_i‖² − r_s,i²，d_i = p_l − o_i，r_s,i = r_obs,i + r_rob + d_min；约束 2 d_iᵀ(v t + l ω n) ≥ −α h_i 对 (v, ω) 是仿射的。作者说明，当 d_i 的横向分量非零时 ω 才能影响安全，这与以轴心为参考点的距离屏障不同。
4. **赛道边界（§II）：** 用附近折线段的最近点、障碍半径取零来表示；活动线段需随机器人运动更新。
5. **QP（§II，式 3）：** min ½(u−u_nom)ᵀW(u−u_nom) + p_s δ²，约束为式 2、松弛 CLF V̇ ≤ −εV + δ（δ≥0）和 u∈U；CLF 松弛使屏障约束在跟踪与避障冲突时优先。前视距离、屏障速率、跟踪速率和松弛权重用于调节「早干预」与「平滑恢复」。
6. **仿真对比（§III，图 1–2）：** 对象为 Huang 等 [9]（时变 CBF + CLF-QP、后轴偏置点）的「supplied implementation」。图 1 为整圈赛道路径，两者轨迹相近（文字称「similar / qualitatively similar」，无定量跟踪误差）。图 2 中位数（图上取整标注）：每步总计算 0.31 ms 对 0.33 ms，QP 求解 0.023 ms 对 0.025 ms；作者自算相对降低 (0.33−0.31)/0.33≈6.1%、(0.025−0.023)/0.025=8.0%，绝对约 20 µs 与 2 µs。
7. **未归因的内容（§III）：** 硕士论文还含离线参数调优与多车相对运动约束（[2] 第 5–6 章），作者明确说不归属于图 1–2 的比较。
8. **硬件（§IV，图 3–4，表 I）：** 1:43 比例车，后轮直流电机、前轮转向舵机、ESP32-C3；头顶 RunCam Pro 5 相机，OpenCV HSV 分割，标记点质心给位置、相邻位置差分给航向；状态经 UDP 送外部控制器，指令经 BLE 回车。表 I（转引自 [2] 表 7.1 与 §7.2.3）：图像采集 6.76 ms、图像处理 27.39 ms、状态获取（UDP）28.29 ms、QP 求解 7.49 ms、BLE 通信 4.34 ms、控制周期 40 ms（25 Hz）、平均状态估计延迟 62.44 ms。作者指出各阶段均值之和超过控制周期，不是串行时间分配；QP 均值约占 40 ms 周期的 19%。
9. **硬件结果（§IV，图 4）：** 展示参考跟随、静态障碍避让与回到路径；硕士论文还报告「难放置的障碍」下偶发赛道边界碰撞（归因于转向极限与延迟）以及多个边界约束同时激活时的停车。没有给出碰撞次数、最小间距、成功率等定量数据。
10. **安全条件（§V，式 4）：** 圆盘模型下若 ‖p_l−o‖ ≥ r_obs+r_rob+d_min，则由三角不等式 ‖p−o‖ ≥ r_obs+r_rob+d_min−l；故 d_min ≥ l 是车身净空的充分几何条件；前视点位置误差有界 η 时需 d_min ≥ l+η。状态估计延迟 τ 使前视点位移至多 τ·√(v_max²+l²ω_max²)（另加估计误差）；平均延迟 62.44 ms 不是最坏界，1 m/s 时仅平移分量就是 6.24 cm。
11. **作者承认的局限（§I、§V）：** CBF 滤波可能出现不希望的平衡点 [7][8]，避障成功不等于无死锁；正对正的几何下 ω 的瞬时影响消失，前视点不能消除所有死锁；实车在原地不能转向，固定最大转向角下最大偏航率随速度下降；安全结论依赖可行的执行、足够准确且及时的状态估计、以及可行的屏障问题；安全停车需要足够净空以容纳实测停车距离。
12. **规划中、未完成的内容（§V、§VI）：** 运行时监测（运动学合理性检查、过期 / 不合理状态更新请求停车）在图 3 中标为「planned」；演示方案场景 3（受控帧延迟）是「Planned extension」。拟议的是录制演示，软件在环回放为后备。

## 对 wiki 的映射

- [paper-lacart](../../wiki/entities/paper-lacart.md)

## 参考来源

- Lopes, P., Shivam, A., Paiva, L. T. (2026). *LACART: Lookahead-Augmented Collision Avoidance and Reference Tracking Strategies for Safety-Critical Robots*. arXiv:2611.09640v1. <https://arxiv.org/abs/2611.09640>
- 代码仓库 README（静态读）：<https://github.com/Kanestrene/autonomous-racing-cbf>
