# OmniGuide: Universal Guidance Fields for Enhancing Generalist Robot Policies（arXiv:2603.10052）

> 来源归档（paper）

- **标题：** OmniGuide: Universal Guidance Fields for Enhancing Generalist Robot Policies（arXiv v1 首页标题如此；CoRL 2026 阅读清单写作 "Universal Spatial Guidance Fields"，正文 III-C 节小标题为 "A Universal Spatial Guidance Field"）
- **类型：** paper
- **作者 / 机构：** Yunzhou Song*, Long Le*, Yong-Hyun Park, Jie Wang, Junyao Shi, Lingjie Liu, Jiatao Gu, Eric Eaton, Dinesh Jayaraman, Kostas Daniilidis（*同等贡献）；University of Pennsylvania（论文首页）
- **arXiv：** <https://arxiv.org/abs/2603.10052>（读取版本 v1，2026-03-09，cs.RO）
- **PDF：** <https://arxiv.org/pdf/2603.10052v1>
- **项目页：** <https://omniguide.github.io/>
- **代码：** https://github.com/daniilidis-group/OmniGuide （项目页 Code 按钮的外链；本环境访问该仓库被代理拒绝，仓库内容与是否已发布代码均未核实）
- **入库日期：** 2026-11-12
- **一句话说明：** 不重训、不加数据：把 3D 碰撞、VLM 语义目标、人类手部轨迹统一写成笛卡尔空间的可微能量场，经可微运动学反传到 flow-matching VLA 的去噪步，在测试时引导动作采样。

## 核对范围与通读状态

- **核对范围：** 读了 arXiv v1 PDF 全文：摘要、I-V 节（引言、相关工作、方法、实验、结论）、附录 A-E（仿真设置、真机硬件与 VGGT 尺度对齐、基线实现、数学细节、朝向引导）、图 1-14 及图注；图 4-10 渲染成图像核对了柱状 / 折线走势（图 5、7、8、9、10 的数值是从图上读的大致值，非表格）；项目页静态读了页面文本与链接（视频、交互内容未看）。未读：被引文献原文（π0.5、GR00T N1.6、HaPTIC、DemoDiffusion、F3RM、cuRobo 等）；未运行任何代码；未拿到代码仓库内容。
- **通读：** 已完成

## 方法要点（作者自述，论文 III 节）

1. **后验速度场（III-A，式 6-9）：** 在 flow-matching 速度场上叠加能量梯度 v_θ − λ∇L，沿用速度与 score 的闭式关系和 Tweedie 估计的干净动作块 Ã^τ = A^τ + (1−τ)v_θ（式 5）；全程用常数引导强度 λ。
2. **笛卡尔空间求能量（III-B，式 10-12）：** 每个去噪步先估计干净动作块，必要时从潜空间解码到关节空间，经可微动力学 / 运动学得到末端轨迹 X，算能量 L_y(X)，梯度反传回噪声动作块；梯度截断到常数 α。
3. **初始噪声引导（式 13）：** 采样 N 个初始噪声，少步去噪后选能量最低者，解释为能量重加权先验的零温度极限的蒙特卡洛近似。
4. **三类场（III-C）：** 碰撞排斥场（占据栅格的 SDF，L_C = −log SDF + 常数，式 14-16）；语义吸引场（VLM 指出目标像素，深度反投影到 3D，高斯负对数似然，式 17-18，另有朝向项）；人类轨迹吸引场（HaPTIC 提腕部位置，Alg. 1 的单调最近邻匹配近似 DTW，只对匹配上的点施加，式 19）。
5. **真机感知（附录 B）：** VGGT 从多视角 RGB 出点云，用已标定相机对的真实距离重标尺度；PyBullet 渲染机器人 URDF 去除机器人本体点；patch 级 CLIP 与指令的余弦相似度前 95% 分位点判为任务相关并从碰撞点云剔除；Gemini-2.5-Flash 先列出候选点再逐点询问是否相关。

## 实验摘录（论文 IV 节与附录）

- **仿真（IV-A，RoboCasa，GR00T N1.6-3B，Franka Panda，地面真值深度与物体位姿）：** 碰撞任务为带额外杂物的拾放与 TurnSinkSpout，另有 Multi-Choice 语义任务；每设置 50 次试验，报成功率与安全率（一次碰撞即安全失败）。超参 d=0.15 m，λ_C=0.02，λ_S=5.0，朝向项权重 0.02λ_S。
- **消融（IV-B，图 5）：** 仅初始化引导成功率 +8%、碰撞 −18%；仅去噪引导 +20% / −34%；两者合用 +26% / −46%（文中原话）。基线成功率约 67%、安全率约 27%（按图 5 读数，大致值）。
- **引导强度（图 7）：** λ 增大安全率升高，过大则成功率下降；作者称存在较宽的可用区间。
- **组合（图 8）：** 杂乱 Multi-Choice 场景下语义 + 碰撞同时提升成功率与安全率（图上读数：成功率约 31% 到 48%，安全率约 43% 到 62%，大致值）。
- **真机（IV-C，图 9，DROID 平台，Franka Research 3，π0.5，Gemini-2.5-Flash）：** 3 类引导 × 3 任务共 9 个配置，每配置 5 次试验；对比 cuRobo 后处理（碰撞）、F3RM（语义）、DemoDiffusion（人类模仿）。图上 OmniGuide 在碰撞三任务与人类模仿三任务均约 100%，语义三任务约 100% / 100% / 80%；π0.5 基线多数为 0 到 60%（图上读数）。
- **延迟（图 10）：** 单张 RTX 5090 上基础 π0.5 约 0.25 s、OmniGuide 约 0.53 s；控制频率由 30 Hz 降到约 15 Hz。
- **摘要 / 引言（I 节）的头条数字：** 「成功率 24.2% 升到 92.4%、避碰率 7.0% 升到 93.5%」，正文与图 5、9 中找不到对应的汇总，口径不明。

## 作者自述的局限（论文 V 节与 III-C）

- 引导场是确定性、任务特定的，可能有局部极小与接触动力学建模不全（III-C 末段）；引导无法自行解决人到机器人重定向的运动学失配、势场的局部极小与复杂接触（V 节）。
- 展望：引入以物体为中心的表示、力传感器、点轨迹、AI 生成视频与检索动作作为新的引导（V 节）。

## 对 wiki 的映射

- [paper-omniguide-spatial-guidance-fields](../../wiki/entities/paper-omniguide-spatial-guidance-fields.md)
