# An Empirical Study of Reinforcement Learning for Keypoint Positioning of Deformable Linear Objects（arXiv:2611.07261）

> 来源归档（paper · 原文通读）

- **标题：** An Empirical Study of Reinforcement Learning for Keypoint Positioning of Deformable Linear Objects
- **作者：** Vinay Rodrigues, Sara Zeynalpour, Gokhan Solak, Sebastian Hjorth, Arash Ajoudani
- **机构（论文首页）：** 意大利技术研究院（IIT）Human-Robot Interfaces and Interaction（HRII）实验室，热那亚；第一作者同时隶属热那亚大学 DRIM 国家机器人博士项目
- **资助（论文首页脚注）：** 欧盟 Horizon 项目 TORNADO（GA 101189557）
- **类型：** paper
- **链接：** <https://arxiv.org/abs/2611.07261>（PDF：<https://arxiv.org/pdf/2611.07261v1>）
- **版本 / 日期：** arXiv v1，cs.RO，2026-11-05
- **入库日期：** 2026-11-09
- **代码：** 论文正文未给出代码或项目页链接（未发现）
- **核对范围：** 下载 v1 PDF（首页标题与编号核对一致），通读摘要、§I 引言、§II（A 闭环架构、B 视觉系统、C 任务与 RL 框架含观测/奖励公式、D 训练策略）、§III 任务描述、§IV（A 计算加速含 Table I/II 与 Fig. 4、B 观测消融 Fig. 5、C 仿真鲁棒性含 Table III 与 Fig. 6–8、D 零样本真机含 Table IV 与 Fig. 9）、§V 结论与未来工作；参考文献仅浏览。论文无附录与补充材料页（v1 PDF 文本中未见）。图 1–3、9 为示意/照片，仅按图注与正文理解，未逐像素核对；Fig. 4–8 的曲线数值以正文与表格为准。未运行任何代码。
- **通读：** 已完成

## 核心摘录（均出自原文）

1. **问题设定（§II-A、§III）：** 平面内、一端被末端执行器（EE）刚性抓住的可变形线状物体（DLO，如缆线、软管、绳）；任务空间动作为 EE 平面平移 (x, y) 与绕 z 轴偏航增量，归一化到 [-1, 1]^3；准静态假设，忽略动力学效应。策略 10 Hz，物理仿真 1 kHz，单回合最多 200 步（20 s）（§II-D）。
2. **观测设计（§II-C 式 1–3）：** DLO 用沿中心线的关键点表示；actor 只看 Nπ = 5 个关键点（含 EE 端 katt 与自由端 ktip）的相对 EE 位置/速度加自由端绝对位置、目标误差与上一动作；critic 额外得到 NV = 12 个关键点的稠密状态与限幅/边界违例指示 zt（特权信息），即非对称 actor–critic。
3. **奖励（§II-C 式 4）：** 目标跟踪、进度奖励（λprog = 10）、平滑正则、事件项（βs = βT = 150，βleave = 16）；双点任务先追自由端，进入 ε = 0.03 m 后再追 EE 端关键点。
4. **训练（§II-D）：** MuJoCo Playground 内 JAX 版 PPO；actor 2 层、critic 3 层、每层 256 单元、SiLU；随机化含 DLO 平面滑动摩擦 U(0.5, 0.8)、关节刚度/致动器阻尼/被动阻尼缩放各 U(0.95, 1.05)、观测高斯噪声、至多一个控制步的延迟、扰动外力；弯曲与扭转刚度因 MP 插件不支持运行时随机化而固定。
5. **三套训练栈对比（§IV-A，Table I/II，Fig. 4）：** SB3+mj（PyTorch SB3 + CPU MuJoCo，4 个并行环境）、SBX+mj（JAX SBX + CPU MuJoCo）、MP（MuJoCo Playground，MJX，1024 个 GPU 并行环境）；每套用 Optuna（TPE）单独调参，目标 0.90·s̄final + 0.10·AUC − 0.35·σ(sfinal)，每个 trial 3 个调参种子；调参用 8 核 CPU + A100 80GB，其余实验用 8 核 CPU + V100 32GB。5 个留出种子从零训练：Final SR 为 SB3+mj 0.860±0.108、SBX+mj 0.880±0.110、MP 0.950±0.032；SR AUC 为 0.529、0.532、0.735。约 240 min 的共同预算内 MP 收集的环境交互约为 CPU 栈的 13 倍。
6. **观测消融（§IV-B，Fig. 5）：** 单点任务、约 2000 万步、2 个随机种子；相对 EE 的关键点表示优于机器人基座系绝对表示，非对称 actor–critic 优于对称；相对+非对称组合最优并用于后续所有实验。正文只给曲线结论，未给数值表。
7. **仿真结果（§IV-C，Table III，Fig. 6）：** 每个任务-条件 5 个评测种子 x 50 回合 = 250 回合。单点：Basic 249/250（99.6%）、N+DR 241/250（96.4%）、N+DR+D（含外部扰动）245/250（98.0%）。双点：Basic 231/250（92.4%）、N+DR 222/250（88.8%）。双点失败分解：Basic 19 次（Type-1 为 9、Type-2 为 1、Type-3 为 9），N+DR 28 次（19 / 2 / 7），N+DR 下 78.6% 的失败出现在靠近工作空间边界（0.05 m 内）的目标对，此类目标对占 44.4%。目标间距 ≥0.20 m 时 N+DR 成功率 83.1%（<0.10 m 时 92.7%）。
8. **变长度（§IV-C-e，Fig. 8）：** 单一单点策略在 0.300 / 0.375 / 0.450 m 三种长度上联合训练，成功率 211/250（84.4%）、215/250（86.0%）、216/250（86.4%）；论文明确这是训练分布内的长度变化，不是外推到未见长度。
9. **零样本真机（§IV-D，Table IV）：** Franka Emika Panda，RealSense D435i 俯视 30 Hz，ArUco 外参标定，ROS 集成；直径 0.004 m 硅胶导管；YOLO26-seg 分割 + 骨架化 + 等间距取 5 个关键点 + 深度投影到桌面。分割指标：mask 精度 98.81%、召回 99.13%、F1 98.97%、mAP@0.5 99.48%、mAP@0.5:0.95 79.30%、29.8 FPS。每任务 10 次：单点 9/10（中位时间 9.30 s）、扰动恢复 10/10（6.55 s）、双点 8/10（任务中位 9.05 s；自由端先到中位 8.55 s）。
10. **真机失败（§IV-D-4）：** 机器人造成的遮挡使关键点估计退化、导管高柔顺性释放弹性能造成紧凑双点构型下的突然偏转、感知延迟造成过冲；双点两次失败均为自由端到位后又离开目标（Type-3）。
11. **作者自述的未来工作（§V）：** 更复杂的形状控制任务、三维操作、对未见 DLO / 环境 / 任务配置的泛化。

## 对 wiki 的映射

- [paper-dlo-keypoint-rl-empirical-study](../../wiki/entities/paper-dlo-keypoint-rl-empirical-study.md)
