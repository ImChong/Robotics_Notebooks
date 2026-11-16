# prism_real2sim2real_arxiv_2609_38172

> 来源归档（paper · 原文通读，支撑 `wiki/entities/paper-prism-real2sim2real.md`）

- **标题：** Counterfactual Video Generation Enables Scalable Humanoid Loco-Manipulation（PRISM）
- **类型：** paper
- **会议：** CoRL 2026（正文页脚：10th Conference on Robot Learning, Austin TX, USA）
- **作者 / 机构：** Zihan Wang、Zhen Wu、Pieter Abbeel、Rocky Duan、Jitendra Malik、Carmelo Sferrazza、C. Karen Liu、Guanya Shi、Angjoo Kanazawa；Amazon FAR、UC Berkeley、CMU、Stanford（首页脚注：后 7 位为 FAR Team Co-Leads）
- **arXiv：** <https://arxiv.org/abs/2609.38172>（本次读 v1，cs.RO，2026-09-29；首页标题与编号已核对）
- **项目页：** <https://prism-real2sim2real.github.io/>
- **代码（项目页）：** <https://github.com/amazon-far/PRISM-Real2Sim2Real>（2026-09-30 入库时匿名 GET 404；2026-11-12 经代理访问返回 403，无法据此判断是否公开，仍待复核）
- **入库日期：** 2026-09-30；**原文通读升级：** 2026-11-12
- **核对范围：** 读 arXiv v1 PDF 全文文字：摘要、§1 引言、§2 相关工作、§3（3.1 counterfactual V2V，3.2 接触锚定 Real2Sim 与重定向，式 1）、§4（4.1 teacher，式 2；4.2 蒸馏，式 3）、§5（5.1–5.3，表 1–3）、§6 结论、§7 局限，以及附录 A–G（V2V 细节与 prompt、真机物体、训练数据与评测、超参表 4–6、物体物理表 7–8、V2V 对几何增广表 9、接触锚可靠性、运行时间表 10、额外交互示例）和参考文献列表（只看条目，未读被引论文）。图 1–8 只读了图注与文字抽取内容，未逐张看图像本身；未看项目页视频，未运行也未下载任何代码；未读 OpenReview 评审页（被验证挑战挡住）。
- **通读：** 已完成
- **一句话说明：** 用 V2V counterfactual 视频把 4 条真人搬箱种子扩成 256 条交互 clip，经接触锚定 Real2Sim 与重定向训统一机载深度加摇杆的 G1 策略，真机零样本 pick–carry–drop 多类物体。

## 核心论文摘录

### 1) 问题与 counterfactual V2V 数据范式（Abstract / §1 / §3.1 / 附录 A）

- 动机：互联网视频内容与取景受人类观看偏好左右，筛选「全身清晰、交互无遮挡、视角稳定」的片段代价高（§1、§3.1）。
- 做法：给定 seed 视频与类别级文本 prompt，让模型替换被操作物体，同时保留背景、光照、机位和粗任务结构（§3.1）。作者声称物体变化会连带人的行为变化（弯腰、手距、搬运方式），这是作者自述，论文未对「行为适配」做量化评估。
- 规模：4 条真人搬箱 seed，每 seed 对 box / bin / barrel / ball 各 16 个样本，共 64 条每 seed、256 条总计；生成模型 SeedDance 2.0，经其网页界面调用（附录 A）。Prompt 模板见附录图 6，仅一句，类别用 `<CLS>` 占位。
- 注意：§5.3 与附录 F 写明「提交时用了 8 条 seed」做 V2V 对几何增广的消融（Seed-1X 即 8 条），与引言/§7 的 4 seed → 256 条口径并存，论文未统一说明。

### 2) 接触锚定 Real2Sim 与重定向（§3.2、式 1、附录 G.1）

- 后端 CRISP（HMR 加视觉 SLAM）恢复相机、公制场景与 4D SMPL-X；本文扩展到动态物体：SAM 2 掩码，SAM3D 出网格与初始位姿（§3.2）。
- 不用 FoundationPose 独立跟踪（遮挡敏感，表 3 支撑）；接触相位 [t1, t2] 内把物体固定在 SMPL-X 掌坐标系的相对变换上，接触由人体运动线索检测，无需人工接触标注（§3.2）。
- 接触锚：两掌心连线段与物体网格求交；重定向沿用 OmniRetarget 的 interaction mesh 约束 IK，加一项末端到锚点的二次项 wc‖x_eef + J·dq − c‖²（式 1）。
- 自陈：无形式化接触或可行性保证；位姿或网格误差会造成穿透与 IK 失败，只保留不穿透的解（附录 G.1）。

### 3) 统一深度策略（§4、式 2–3、附录 D）

- Teacher 为特权 co-tracking 策略，同时跟踪机器人与物体参考；奖励含机器人/物体位姿跟踪、动作率、关节限位、碰撞项，另加接触奖励 R_contact（式 2，末端到目标点的指数项乘受力阈值的 min 项）。表 5 中接触相关项为「腕目标引导」权重 5.0 与「力门控接触引导」权重 10.0。
- Student 输入：本体（角速度、关节、上一动作）、摇杆指令（相对根位移 Δx、Δy、Δyaw 加二值 drop 键）、机载深度（58×87 经小 CNN 压成 32 维）；训练中名义相机俯仰 37°，随机化相机并注入噪声、丢点、空洞、边缘伪影与偏移（§4.2、表 6）。
- 蒸馏损失 L = λ·L_PPO + (1−λ)·L_D（式 3）；表 6 写 PPO 系数 0.1 到 0.9，每 500 iter 升 0.1，末 20K iter 保持 0.9。Teacher 40K iter，student 28K iter，3 层 MLP，8 张 L40S × 每卡 4096 环境（§4.2、附录 D）。
- 暖启动：先用仅 box 数据训 23K iter，只恢复 actor 权重；从零训练也成功，暖启动只加速收敛并用于发布的 checkpoint（§4.2、附录 D）。

### 4) 数据漏斗与评测口径（§5、附录 C、表 1–3、表 8–9）

- 漏斗：256 条生成视频中 137 条得到可行的机器人–物体轨迹（其余因求解器失败，主因严重碰撞，部分物体对 G1 过大）；teacher 训练后 129 条成功 rollout，作为 student 的 129 条演示（附录 C、§7）。
- **关键口径：** §5.1 的表 1 与表 9 用的是「更早的 student」（由 20K iter teacher 的 80 条成功 rollout 蒸馏），PRISM-ID 评测集就是这 80 条训练交互；更新后的 129 条版本不在该比较里（附录 C、F）。PRISM-OOD 为 48 条重建交互（table / chair / lamp / monitor 各 12 个实例），类别未出现在生成数据中。
- 表 1（MuJoCo sim-to-sim，成功 = 完成 pick–carry–drop 且物体落在目标位置）：OMOMO(90) 训练 → OMOMO Train 94.44%，OMOMO Test 91.67%，PRISM-ID 23.75%，PRISM-OOD 12.50%；PRISM-ID(80) 训练 → 98.89%、100%、96.25%、72.92%。
- 表 3（消融，基线为 SAM3D + FoundationPose + OmniRetarget）：22.50/12.50 → +锚定物体位姿 58.75/29.17 → +接触感知重定向 86.25/68.75 → +接触奖励 96.25/72.92（ID/OOD）。
- 表 9（V2V 对几何增广）：Seed-1X(8 条) 在 PRISM-ID/OOD 为 17.50/6.25；Seed-17X(103 条，由 8 seed 各加 16 个 yaw/xy/scale 变体，深度相机可见性过滤后留 103) 为 36.25/22.92；V2V(80 条) 为 96.25/72.92。该比较共享训练预算与评测协议，但评测集由 V2V 数据本身构成，对几何增广方有分布偏向（推测）。
- 表 8（载荷敏感度，80 条版 student，仿真）：0.1–1.0 kg 到 4.0–5.0 kg，PRISM-ID 98.75 降到 91.25，PRISM-OOD 77.08 降到 64.58（非单调：3–4 kg 为 62.50，4–5 kg 为 64.58）。

### 5) 真机（§5.2、表 2、附录 B、图 5）

- 平台 Unitree G1 29-DoF，50 Hz，PD 增益参照 BeyondMimic；头载 D435i 30 Hz 双目，离板电脑经以太网跑 Fast-FoundationStereo 估深度；人类操作员摇杆；手动把颈部上仰约 10°（§5.2）。
- 判据：够到并抓起，稳定搬运 ≥ 3 m 不掉不倒；每物体 5 次试验，物体位姿与距离变化。注意这个真机判据不含「放到目标位置」，与仿真判据不同（附录 C 明说）。
- 表 2：in-domain 每类 3 个不同物体共 15 次——Ball 12/15（80%）、Bin 14/15（93%）、Barrel 14/15（93%）、Box 15/15（100%）；OOD 各 5 次——纸巾 5/5、头盔 3/5、桌子 5/5、背包 5/5、灯 3/5、椅子 4/5、水壶 4/5、小鸡玩具 3/5。OOD 每类只有一个物体、5 次试验。
- 零样本高处拾取：35° 斜坡与 0.43 m 平台成功，45° 与 0.45 m 失败；作者推测与训练物体高抓取高度重叠有关（图 5、§5.3）。
- 颈部偶尔复位或偏离目标角，造成 OOD 深度（§5.2）。

### 6) 作者自述局限（§7、附录 E、G）

- 单目 4D 人–物–场景恢复困难，误差沿重定向传播；4 seed → 256 → 137 → 129 的规模不足以刻画 scaling 行为；物体物理只用类别名义质量（球 0.5、桶 1.5 kg 等，缩放因子 U(0.33, 3.0)）与随机摩擦，未辨识实例属性；物体按刚体仿真，折叠椅等关节物体会不稳。
- 运行时间（表 10，每 8 秒 clip 均值）：V2V 生成 191.76 s，重建 414.5 s，重定向 71.6 s（重建与重定向用单张 L40S）。
- 附录 G.3 的开门与滚球只是重建的定性例子，不是策略或真机评测。

## 对 wiki 的映射

- [PRISM（Real2Sim2Real loco-manip）](../../wiki/entities/paper-prism-real2sim2real.md)
- [Loco-Manipulation 任务页](../../wiki/tasks/loco-manipulation.md)
- [CRISP Real2Sim 方法页](../../wiki/methods/crisp-real2sim.md)
- [Motion Retargeting 概念](../../wiki/concepts/motion-retargeting.md)
- [DAgger 方法](../../wiki/methods/dagger.md)
- [VideoMimic 实体](../../wiki/entities/videomimic.md)

## 当前提炼状态

- [x] 论文全文（正文加附录文字）核对，见「核对范围」
- [x] 开源状态：项目页 Code 链至 `amazon-far/PRISM-Real2Sim2Real`，仓库可访问性待复核
- [x] wiki 实体页 [`paper-prism-real2sim2real.md`](../../wiki/entities/paper-prism-real2sim2real.md)
