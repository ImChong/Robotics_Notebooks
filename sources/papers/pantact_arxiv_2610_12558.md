# PanTact 论文来源归档

## 元数据

- **题目：** PanTact: Teaching Whole-Body Dexterity with a Wearable Contact Interface
- **作者：** Xiaomeng Xu, Han Zhang, Zhian Ruan, Xiaofeng Guo, Jinkun Cao, Ruihan Yang, Carmelo Sferrazza, Shuran Song, Guanya Shi（† 共同指导）
- **单位（HTML/项目页）：** 斯坦福大学；Amazon FAR；卡内基梅隆大学（作者与单位的逐一对应在抽取文本中不清）
- **预印本：** https://arxiv.org/abs/2610.12558
- **HTML：** https://arxiv.org/html/2610.12558v1
- **版本：** v1，2026-10-08，cs.RO
- **项目页：** https://pantact.github.io/（见 [pantact-project.md](../sites/pantact-project.md)）
- **代码：** arXiv 页未列仓库；项目页 Code 按钮 href 为空，截至 2026-10-12 未发布
- **读取方式：** 摘要页 + HTML 全文，2026-10-12

## 要解决的问题

人形机器人有大量自由度和接触面，但多数 loco-manipulation 把“操作”限制在双手。PanTact 要从无需机器人的人类示范中学习用全身表面操作。核心判断：接触的位置与时机可跨本体迁移，决定任务需要什么；怎样用机器人身体去实现这些接触，由机器人自己在物理约束内协调。

## 方法与训练信息

**数据采集（穿戴式、无机器人）：** 模块化接触贴片，每块 44×44 mm，内含 1 个力敏电阻（FSR），自制 50×50 mm PCB 最多 16 路、1 kHz，12 位 ADC，RS-485 菊花链；磁吸连接，可装 HTC Vive tracker 提供 6-DoF；接触按各通道自适应基线二值化。贴片覆盖演示者前臂与膝盖（传感器总数原文未给）。另有头戴 ZED X Mini 立体相机、两个 ZED X One S 鱼眼腕部相机（60 fps）、Vive tracker（头与双腕，250 Hz）、双臂 UMI 夹爪（宽度 1 kHz），日志在 Jetson AGX Orin。采集数据：Grasp-and-Embrace 100 条、Sweep-and-Stow 150 条、Carry-and-Push 110 条。

**策略：** 扩散策略，DiT 去噪头（10 层、宽度 768、12 头），微调 DINOv3 ViT；10 Hz，动作视野 H=16，观察历史 h=2。输入：去除机身区域的第一视角点图、左右腕图像、各腕末端位姿与夹爪宽度。人机观察差异处理：由夹爪坐标系与接触贴片定义“朝向身体的半空间”，其交集近似身体区域，对该区域的点在编码器注意力里遮挡；图像 patch 特征加 3D 正弦位置编码；所有量以最新左夹爪位姿为参照。输出：头的 3D 注视点、目标末端位姿与夹爪宽度、各传感器接触目标（二值接触状态 + 3D 位置，仅追踪激活目标）。训练：100 步扩散，推理 DDIM 16 步，ε 预测，500 epoch，有效 batch 48，学习率 7.5e-5（编码器 7.5e-6），cosine + warmup，EMA。

**全身控制（G1，29 DoF）：** 以平滑最小距离（温度 0.02 m）度量命令接触点到候选机器人本体的距离并驱向零，激活目标沿物体方向加虚拟位移以产生按压力（力/刚度数值原文未给）。三种实现：
1. Upper-Body IK：QP，权重 EEF 位置 500、姿态 200、接触 250，腿部站立控制，无需训练；
2. WBIK + SONIC：全身 QP + 步进逻辑（骨盆漂移 0.12 m 持续 0.2 s 触发），稠密输出交 SONIC 跟踪器执行；
3. Sparse Contact Tracker：PPO，30,000 次迭代、4096 并行环境、50 Hz 输出 29 关节 PD 目标；上身 8 / 全身 16 个接触槽；用 SONIC rollout 的事后标注接触目标训练，并对摩擦、恢复系数、质量、PD 增益、骨盆速度扰动做域随机化。
定位使用 G1 自带头部 LiDAR-惯性单元上的 Super Odometry；推理异步，过期动作丢弃。

## 论文报告的结果

- **Grasp-and-Embrace（每方法 20 次）：** Upper-Body IK 85%；Sparse Contact Tracker 80%；Humanoid UMI（仅末端跟踪）无法支撑大体积物体，原文未给数值（项目页写 0%）。
- **Sweep-and-Stow（16 次 = 4 条件×4）：** WBIK + SONIC 端到端 90.63%；Humanoid UMI++（连续跟踪膝盖位姿）原文未给数值（项目页写 0%）。
- **Carry-and-Push（10 次）：** WBIK + SONIC 90%（成功 = 手持箱不掉且地面箱推动超过 20 cm）；Sparse Tracker 与 Humanoid UMI++ 掉箱更多，正文无数值（项目页写 Humanoid UMI++ 40%）。
- **控制器回放比较：** 加入接触目标后各控制器接触距离均下降且末端代价小；Upper-Body IK 上身末端误差最低；Sparse Tracker 接触距离最低，并在上身两项指标上优于 WBIK+SONIC；全身任务上其接触距离更低但末端误差更高。数值仅在图中。

## 复现与解释边界

- 无独立“局限”节。Upper-Body IK 仅适用上身任务；WBIK+SONIC 的稠密参考含与任务无关的姿态细节，且不建模 SONIC 的执行极限；Sparse Tracker 需仿真训练、末端误差更高。
- 失败模式：夹爪抓空、柜门只关一部分、追踪死区、臂接近关节极限、累积漂移；人机比例差异使部分膝部目标难实现。
- 每条件 10–20 次、无多种子统计；传感器总数、虚拟目标力/刚度参数、传感器漂移鲁棒性未报告。
