# CoorDex: Coordinating Body and Hand Priors for Continuous Dexterous Humanoid Loco-Manipulation

> 来源归档（ingest）

- **标题：** CoorDex: Coordinating Body and Hand Priors for Continuous Dexterous Humanoid Loco-Manipulation
- **类型：** paper
- **来源：** arXiv abs / HTML；项目页交叉核对
- **原始链接：**
  - <https://arxiv.org/abs/2606.23680>
  - <https://arxiv.org/html/2606.23680v1>
  - <https://skevinci.github.io/coordex/>
- **作者：** Sikai Li, Shuning Li, Zhenyu Wei, Yunchao Yao, Chenran Li, Mingyu Ding
- **机构：** 北卡罗来纳大学教堂山分校（UNC Chapel Hill）、加州大学伯克利分校（UC Berkeley）
- **入库日期：** 2026-06-25
- **一句话说明：** 将高维 **全身 + 灵巧手** 控制分解为 **冻结的 proprio 条件潜空间先验**（body 16-D / hand 12-D）与 **协调潜残差 PPO**：特权 tracking teacher → VAE 蒸馏（PULSE 同族）→ **共享任务上下文 + 分体 body/hand 残差头** 在 Isaac Lab 上学会 **边走边抓/开门/转身持物**；G1+20-DoF WUJI 仿真三项任务，WalkGrab 消融显示关节空间 PPO、单体式潜残差均失败。

## 核心论文摘录（MVP）

### 1) 问题：停走式 loco-manip + 低 DoF 夹爪范式

- **链接：** <https://arxiv.org/abs/2606.23680> §I
- **摘录要点：** 人形 loco-manipulation 常被简化为 **走到物体 → 停下操作 → 再走**；末端也多为 **开–合夹爪** 级低维接口。高 DoF 灵巧手在 **持续行走** 中需同时协调平衡、腕位、指形与接触，直接在全关节空间探索极难。现有手部方法多假设腕轨迹由遥操作/规划/固定基座提供；行走人形上腕位由步态、躯干、全身到达共同涌现，若手先验还要解释 6D 腕运动，容量被身体侧占用而非指间协调。
- **对 wiki 的映射：**
  - [CoorDex（论文实体）](../../wiki/entities/paper-coordex-dexterous-humanoid-loco-manipulation.md) — 问题定位与「连续 dexterous loco-manip」定义。
  - [Loco-Manipulation](../../wiki/tasks/loco-manipulation.md) — 停走式 vs 连续移动操作子路线。

### 2) 不对称 body–hand 先验构造

- **链接：** arXiv §3.1；Appendix A
- **摘录要点：**
  - **示范采集：** Isaac Lab 仿真遥操作——下身 **AGILE 步态控制器**，上身/右手由 **Apple Vision Pro + CloudXR** 追踪；腕目标经 **Pink IK**，人手经 **dex-retargeting** 映射到目标灵巧手。
  - **Body prior：** BeyondMimic 式 **29-DoF G1 全身 tracking teacher**（特权参考 + PPO）→ 蒸馏为 **16-D** proprio 条件潜先验 + 解码器；部署观测 **去掉基座线速度**（no-state-estimator）。
  - **Hand prior：** **腕位运动学写入仿真** 的 floating-hand / wrist-stabilized 环境，teacher 只控 **20 主动指关节**（WUJI）；ManipTrans 风格 MANO 关键点跟踪 → 蒸馏 **12-D** 手先验。
  - **蒸馏：** encoder–prior–decoder VAE（PULSE 动机）：动作重建 + 潜码时序平滑 + KL；训练后 **冻结** $\mathcal{R}_x, D_x$，下游 RL 在潜均值上加残差。
- **对 wiki 的映射：**
  - [CoorDex](../../wiki/entities/paper-coordex-dexterous-humanoid-loco-manipulation.md) — 先验管线 Mermaid。
  - [Privileged Training](../../wiki/concepts/privileged-training.md)、[Whole-Body Tracking Pipeline](../../wiki/concepts/whole-body-tracking-pipeline.md)。

### 3) 协调潜残差策略（Coordinated Latent Residual Policy）

- **链接：** arXiv §3.2–3.3
- **摘录要点：**
  - 每步从冻结先验得 $\boldsymbol{\mu}^{b,p}_t, \boldsymbol{\mu}^{h,p}_t$；actor 输出 **28-D** 残差 $\Delta z = [\Delta z^b, \Delta z^h]$。
  - **共享协调 trunk** $f_{\mathrm{coord}}$ 读 body/hand proprio、任务状态、手–物几何、接触、先验均值、上一步残差 → 特征 $\mathbf{c}_t$。
  - **分体头** $f_b, f_h$（tanh）分别预测 body/hand 残差；$\tilde z = \mu + \Delta z$ 经冻结解码器 → 关节位置目标 + PD 执行。
  - 手目标经 **EMA 0.4** 平滑；探索发生在 **28-D 潜空间** 而非 49-D 关节空间。
- **对 wiki 的映射：**
  - [CoorDex](../../wiki/entities/paper-coordex-dexterous-humanoid-loco-manipulation.md) — 残差架构表。
  - [ResMimic](../../wiki/entities/paper-resmimic.md) — 另一路「先验 + 残差」全身 loco-manip 对照。

### 4) 三项仿真任务与定量结果

- **链接：** arXiv §4；项目页
- **摘录要点：**
  - **平台：** 29-DoF **Unitree G1** + **20-DoF WUJI** 右手；Isaac Lab **4096** 并行环境，60 Hz 控制。
  - **WalkGrab：** 边走边从侧桌 **抓瓶–抬起–搬运**；成功率 **55%**（5 万 eval episodes），**零摔倒**，非停速度近瓶约 **0.25 m/s**。
  - **OpenFridge：** 抓冰箱把手 **边后退边开门**；成功率 **66%**，门角 **57.76° / 60°** 阈值。
  - **WalkPickTurn：** 走近取立方体 **转身 180° 持物**；成功率 **89%**，最小航向误差 **9.98°**；用 **NoDemoRSI**（无专家状态的参考状态初始化）辅助长时程探索。
  - **WalkGrab 动作空间消融（同奖励预算）：** All Joint Space **0%**（常停走 **86%**、零抓取）；Body Prior + Hand Joint **0%**（到达但停住抓）；Monolithic Latent Residual **0%**（更高 action rate **0.40**）；CoorDex **55%**。
- **对 wiki 的映射：**
  - [Unitree G1](../../wiki/entities/unitree-g1.md)、[舞肌 Wuji Hand](../../wiki/entities/wuji-robotics.md)。

### 5) 真机定性回放与手型可替换性

- **链接：** arXiv Appendix C；项目页 Real-world demos
- **摘录要点：** 主文定量实验为 **G1+WUJI**；真机可视化为 **G1+Dex3-1（7-DoF）** 上 **回放记录关节轨迹**（非同配置成功率报告）。因子化设计允许 **换 hand morphology**：重训 hand tracking teacher 并蒸馏新手先验，body 先验与下游残差接口结构不变。当前策略使用 **特权物体位姿与接触** 观测，尚未做视觉 sim2real。
- **对 wiki 的映射：**
  - [CoorDex](../../wiki/entities/paper-coordex-dexterous-humanoid-loco-manipulation.md) — 局限与硬件注记。
  - [Manipulation](../../wiki/tasks/manipulation.md) — 高 DoF 灵巧操作语境。

### 6) 与相邻 loco-manipulation / motion prior 路线对照

- **链接：** arXiv §2
- **摘录要点：** 相对 **ResMimic**（GMT 全身残差）、**HALOMI**（稀疏头手 + VLA）、**VIRAL/视觉分层** 等，CoorDex 隔离 **连续高 DoF 指级接触 + 行走** 的控制问题，强调 **body 腕位涌现 vs hand 指协调** 的潜空间分解与 **协调残差头**，而非停走式或夹爪级接口。
- **对 wiki 的映射：**
  - [Loco-Manipulation](../../wiki/tasks/loco-manipulation.md) — 路线 §20。
  - [ResMimic](../../wiki/entities/paper-resmimic.md)、[HALOMI](../../wiki/entities/paper-halomi-humanoid-loco-manipulation.md)。
