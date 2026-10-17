# PACT: Posture-Aligned Co-Design Technique for Hand Configuration and Retargeting in Multi-Finger Imitation Learning（arXiv:2610.18443）

> 来源归档（ingest）

- **标题：** PACT: Posture-Aligned Co-Design Technique for Hand Configuration and Retargeting in Multi-Finger Imitation Learning
- **arXiv：** <https://arxiv.org/abs/2610.18443>（HTML：<https://arxiv.org/html/2610.18443v1>），cs.RO，2026-10-13 提交，9 页 6 图 2 表；已投 IEEE ICRA 2027
- **代码：** 项目页标注 "upon publication"，无可用链接（2026-10-17 核查）
- **项目页：** <https://pact-hand-2027.github.io>（[sources/sites/pact-hand-codesign-project.md](../sites/pact-hand-codesign-project.md)）
- **作者：** Koyo Sasazaki, Satoshi Funabashi, Pranav Ponnivalavan, Alexander Schmitz, Tetsuya Ogata, Shigeki Sugano；早稲田大学（Waseda University，东京新宿）
- **入库日期：** 2026-10-17
- **一句话说明：** 在可装配约束下，联合优化 17 舵机机械手的几何参数与非线性姿态重定向器，使人手抓取姿态对应误差更低、载重更大，并用遥操作与 ACT 模仿学习验证。

## 开源状态（步骤 2.5）

- **项目页核查（2026-10-17）：** 页面有 "Open source (Release)" 区块，但零件清单、可打印 CAD、装配与接线指南、重定向器与遥操作软件均标 "upon publication"；无任何链接。项目页作者匿名（双盲）。
- **结论：** **宣称将开源（待论文发表），当前无链接**。

## 摘录 1：方法

- 硬件：17 舵机拟人手，食指、中指、无名指各 4 个驱动，拇指 5 个；设计变量含连杆长度、关节偏置、安装角、基座位置；手相对人手放大 1.56x，手指链长 153–161 mm；舵机 Dynamixel XC330-M288-T。
- 重定向器：线性人到机关节映射给出种子，再用阻尼最小二乘姿态 IK 迭代 20 次；每关键点 IK 权重与阻尼也是设计变量。
- 目标：MCP、PIP、DIP、指尖全指对应，加制造约束：舵机干涉、装配间隙、平手突出、MCP 轴锚点。
- 分阶段优化 S0 到 S6，每阶段加一条约束并 warm start，设 10% 精度守卫；每阶段 32 个随机重启 x 400 次迭代，PyTorch float64 可微实现。
- 数据：DexYCB，234 条抓取序列、10 名受试者、8 个 YCB 物体，过滤后 5,117 个姿态；最终设计用 1,788 个姿态训练，277 个留出测试。

## 摘录 2：结果

- 全关键点 RMSE：PACT 3.90 mm（训练）/ 3.88 mm（测试）；LEAP Hand 12.82 / 13.09；仅指尖消融 23.66 / 22.79。指尖 RMSE（训练）：PACT 1.84，LEAP 13.45，仅指尖 2.46 mm。
- 10 mm 内命中率 HPCR：PACT 100%（训练与测试）；LEAP 21.6% / 7.9%；仅指尖 99.5% / 100%。12 个未见 YCB 物体（1,421 次抓取）：PACT 4.06 mm、HPCR 99.6%；LEAP 11.16 mm。
- 载重（关节力矩上限匹配 0.067 至 0.266 N·m，质量 400–2000 g，每组合至少 5 次）：0.167 N·m 下 PACT 举 2 kg，两基线约 1 kg；1300 g：PACT 5/5，LEAP 0/6，仅指尖 0/5。
- 纸杯圆度（1.000 为未变形，每手 25 次）：PACT 0.9847 ± 0.0020，人手 0.9777 ± 0.0089，LEAP 0.8752，仅指尖 0.8272。
- 10 个 YCB 物体各 5 次：PACT 43/50（86%），LEAP 32/50（64%），仅指尖 16/50（32%）；成功试验平均遥操作时间 10.6 s / 21.8 s / 43.5 s。
- 模仿学习：ACT 策略用 54 条遥操作罐子放托盘示范；原放置位置 94%（17/18），留出位置 75%（9/12）。遥操作：单 RGB 相机 + WiLoR，约 23 Hz，UR5e 臂 + GELLO 主手。
- 局限：测试姿态来自相同受试者与均值手形，操作者手尺寸鲁棒性未测；物理评估为开环遥操作；规则族为线性种子加阻尼 IK；训练集中没有的把手抓取代价最大（水壶 4.74 mm）；放大 1.56x 后小特征精度弱（细汤罐、自上而下抓杯）。

## 建议 wiki 动作

- 新建 `wiki/entities/paper-pact-hand-codesign.md`。
