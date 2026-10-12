# TAME-Nav 论文来源归档

## 元数据

- **题目：** TAME-Nav: Temporal Asymmetric Mixture-of-Experts Distillation for Agile and Versatile Robot Navigation
- **作者：** Lingjie Zhang, Jun Xue, Shanze Wang, Haoyu Xi, Xinming Zhang, Xiaoyu Shen, Wei Zhang
- **单位：** 上海交通大学；宁波东方理工大学（Eastern Institute of Technology, Ningbo）；香港理工大学航空及航空工程系；中国科学技术大学（通讯作者 Wei Zhang）
- **预印本：** https://arxiv.org/abs/2610.12877
- **HTML：** https://arxiv.org/html/2610.12877v1
- **版本：** v1，2026-10-09，cs.RO
- **项目页：** https://tame-nav.github.io/ （归档见 sites/tame-nav-project.md）
- **代码：** 论文称项目页提供视频与代码；项目页 Code 链接（/code/）截至 2026-10-12 仅为 “Coming soon” 占位页。
- **读取范围：** arXiv 摘要页与 HTML 全文抽取；若干超参数文中未给出，见下。

## 论文要解决的问题

单一导航策略难以同时满足不同场景的决策需求，尤其是敏捷机器人在拥挤空间、含动态障碍时（局部陷阱、狭窄通道、突发动态障碍）。

## 方法与训练信息

1. **观测：** 机器人观测 55 维（41 维 LiDAR + 14 维非 LiDAR：目标、速度、上一指令、投影重力）；学生使用 10 帧 × 55 维历史（LiDAR 10 Hz，约 0.9 s），本体与上一指令 50 Hz 更新。
2. **教师（特权）：** 时序 MLP 将 10 帧干净历史编码为 16 维潜变量；最多 5 条动态障碍轨迹，每条 8 维特征（相对位置、速度、半径、不确定度、年龄、可见性），经自注意力与交叉注意力（2 头、64 维 query）得到 16 维动态潜变量；以带可学习门标量（初值 1）的归一化残差和融合。
3. **学生（可部署）：** 3 层 1D CNN 处理 LiDAR（每槽 256 维）、2 层 MLP 处理状态（64 维）；单层 GRU 跨 10 个槽得 256 维摘要；软 MoE 共 8 个专家，每个输出 16 维，稠密 softmax 路由，路由输入为 GRU 摘要 + 11 维导航上下文（共 267 维）。
4. **Actor/Critic：** 共享 Actor 输出名义速度均值与 CBF 系数（softplus），基于 LiDAR 的可微 CBF（来自 SEA-Nav）修正均值；高斯策略，对 (vx, vy, ωz) 使用可学习的状态无关标准差；Critic 输入 104 维含 32 维障碍集合编码，仅训练时使用。
5. **损失：** PPO 更新教师编码器、Actor、Critic；学生对 stop-gradient 教师潜变量做潜在蒸馏（MSE，按 1/16 缩放）加专家均值权重的均衡损失；rollout 中 75% 并行环境跑教师、25% 跑学生。
6. **仿真与随机化：** 自研 GPU 向量化 2D 导航仿真器（命令响应模型，不含关节动力学）；600 个程序化生成的布局（6 个场景族各 100：密集杂物、多方向动态穿越、凹陷阱、不规则窄走廊、遮挡动态交互、复合室内）。随机化：LiDAR 额外延迟 0–20 ms、距离噪声标准差 1–2 cm、射线丢失 0–3%；里程计延迟 0–40 ms；执行端前向延迟 0–40 ms、响应时间常数 0.08–0.16 s、增益 0.80–0.95、启动死区 0.22–0.28 m/s。学生接收被扰动的观测，教师与 Critic 接收干净观测。
7. **机器人：** Unitree Go2，Livox MID-360 LiDAR，板载 NVIDIA Jetson Orin，指令 50 Hz；仿真速度指令由另行训练的运动控制器执行。
8. **文中未给出：** 均衡损失权重、折扣因子、并行环境数。

## 论文报告的结果

- **指标：** 成功率 SR、碰撞率 CR、BARN 风格导航得分（v_max = 2.0 m/s）。基线：SEA-Nav、NavRL、DRL-DCLP；(OM) 表示在作者地图上重训。DCLP 代码与地图未公开，仅为作者复现。
- **分布内（600 个未见布局）：** TAME-Nav 在 6 个场景族均为最高 SR，密集杂物与凹陷阱领先超过 35 个百分点；各族具体数值只在图 3，文本未给出。
- **序贯组合场景（64 布局 / 512 回合）：** SR/CR/Score：SEA-Nav 47.27/52.73/0.1934；NavRL 0.00/75.39/0.0000；SEA-Nav(OM) 66.80/33.01/0.2393；DCLP(OM) 59.96/40.04/0.2998；NavRL(OM) 34.96/60.94/0.1743；TAME-Nav 81.64/18.16/0.4068。
- **多房间组合场景（64 布局 / 512 回合）：** SEA-Nav 48.24/37.70/0.2234；NavRL 3.91/42.97/0.0195；SEA-Nav(OM) 66.41/22.85/0.2535；DCLP(OM) 60.16/35.16/0.2951；NavRL(OM) 67.58/15.62/0.3115；TAME-Nav 75.20/13.48/0.3448。
- **消融（分布内）：** 去 MoE（参数匹配 MLP）SR −4.25 pp；去教师轨迹 −6.75 pp（最大）；去时序历史（当前观测重复 10 次）−5.59 pp。完整模型绝对 SR 文本未给。
- **真机（每方法每场景 10 次，一次不撞倒障碍的轻微接触视为允许）：** 局部陷阱脱困 TAME-Nav 100%、碰撞 0.10/次，SEA-Nav 0%，NavRL 0%，SEA-Nav(OM) 0%，DCLP(OM) 0%，NavRL(OM) 50%；窄通道+突发动态障碍 TAME-Nav 100%、碰撞 0.00/次，SEA-Nav 80%，NavRL 0%，SEA-Nav(OM) 90%，DCLP(OM) 80%，NavRL(OM) 90%。TAME-Nav 20 次试验中唯一一次轻微接触出现在局部陷阱场景。

## 复现与解释边界

- 无显式局限章节；真机仅两个场景、一台机器人、每场景 10 次。
- 训练与评测在 2D 抽象仿真器中，腿式运动由独立控制器承担。
- 基线比较含改造：NavRL 原为空中机器人，DCLP 为作者复现。
- 关键超参数未报告，仓库未发布，难以完整复现。

## Robotics_Notebooks 映射

- 详情节点：wiki/entities/paper-tame-nav.md
- 汇总页：wiki/overview/embodied-station-34-papers-spatial-grounding-technology-map.md
- 概念入口：wiki/concepts/hierarchical-quadruped-navigation-stack.md
- 方法入口：wiki/methods/teacher-student-dagger-training.md

## 一手来源

- arXiv：<https://arxiv.org/abs/2610.12877>
- arXiv HTML：<https://arxiv.org/html/2610.12877v1>
- 项目页：<https://tame-nav.github.io/>
