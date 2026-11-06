# Contact-Rich 模仿学习综述通读（arXiv 预印本）

> 来源归档（原文通读 · 支撑 `wiki/concepts/contact-rich-manipulation.md`）

- **标题：** A Survey on Imitation Learning for Contact-Rich Tasks in Robotics
- **作者：** Toshiaki Tsuji, Yasuhiro Kato, Gokhan Solak, Heng Zhang, Tadej Petrič, Francesco Nori, Arash Ajoudani
- **类型：** paper / survey / imitation-learning / contact-rich
- **原文：** <https://arxiv.org/abs/2506.13498>（v1，2025-06；2026-11 通过 arXiv PDF 获取全文并 pdftotext 通读）；同一综述的期刊版见 [contact_rich_imitation_learning_survey_doi_10_1177_02783649261417694.md](contact_rich_imitation_learning_survey_doi_10_1177_02783649261417694.md)
- **一句话说明：** 首篇专门梳理「接触丰富任务 × 模仿学习」的综述：示教采集（模态、示教方式、合成数据、数据集）、学习方法（BC / DMP / 生成式 / IRL / 多模态 / 离线 RL / 其他）、应用（工业 / 家庭 / 医疗）与未来方向。
- **核对范围：** 通读摘要与 §1 引言、§2 预备（接触丰富机器人背景、运动控制启示、关键挑战）、§3 示教采集（3.1 数据模态、3.2 示教方式、3.3 合成数据、3.4 数据集）、§4 学习方法（4.1–4.7）、§5 应用案例、§6 结论；参考文献列表仅浏览，未逐条核对被引原文。本档只依据该预印本，未与期刊版逐字比对。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **定义与难点（§1、§2.1）：** 接触丰富操作指机器人与环境存在连续、复杂交互、需在一至多个接触点上精细控制力的任务；摩擦、弹性、塑性形变等使动力学强非线性，微小位姿偏差即可改变行为。经典路线是操作空间 / 混合力位 / 阻抗控制与接触建模、多接触控制（Khatib、Neville、Raibert & Craig 等）；其可扩展性受限（如 peg-in-hole 的力 / 阻抗需求随位姿、材料、间隙变化；擦拭、打磨需随表面几何、摩擦持续调整）。
- **RL 与 IL 的取舍（§1、§2.3）：** RL 可自主获得含接触切换的动作，但需大量试错，真机磨损与安全风险大，仿真中接触模型不准导致 sim-to-real 差距；IL 能高效学到人类对接触力与位置的细微调节，但受限于缺少适合接触任务的数据采集技术。
- **运动控制启示（§2.2）：** 人类在接触任务中靠柔顺、预测控制、学习与鲁棒反馈整合，而非精确最优控制；阻抗 / 导纳控制是对应的机器人手段；示教天然含姿态微调与力调制。
- **关键挑战（§2.3）：** 接触动力学难以用经典模型刻画；触觉传感器与可穿戴触觉多停留在研究阶段；视觉下接触任务「非完全可观」——交互力无法从图像直接测得，接触区常被遮挡；第三人称示教存在感知差距与视角 / 外观差异；数据效率与泛化更难。
- **数据模态（§3.1）：** 位置（轨迹跟踪）、力（装配 / 插入的力调节）、视觉（语境，但有遮挡且间接推力）、触觉（摩擦、滑移、纹理、形变）；多模态融合提升鲁棒与泛化；新兴模态含 EMG、软传感器、手持无机器人力示教设备。
- **示教方式（§3.2，图 1）：** 在线示教（kinesthetic、遥操作、VR）与离线示教（观察式）× 在线 / 离线学习，得四类：交互式模仿（如 DAgger 类）、示教增强 RL（GAIL、DDPGfD、DAPG）、直接模仿（BC、DMP）、观察学习（大规模数据集 BC、视频模仿）。遥操作常用含力反馈的双向控制，可同时示教力与位置，需保证透明性；观察式示教无机器人干扰、质量高，但人机动力学不同。接触任务需在工具中嵌力传感器或触觉手套才能取力。
- **合成数据（§3.3）：** 仿真器（MuJoCo、PyBullet、Isaac Gym、Drake 的 hydroelastic 接触）各有速度 / 真实度权衡；域随机化、物理增强、可微仿真、合成触觉数据、混合数据；sim-to-real 差距随接触数量与复杂度增大。
- **数据集（§3.4）：** TVL、Touch100k、VisGel、Sparsh / TacBench、BridgeData V2、DROID、Open X-Embodiment，以及 Ablett 等基于 kinesthetic 的视触觉开门数据集。
- **BC（§4.1）：** 监督回归 $J(\theta)=E\|\pi_\theta(s_t)-a_t\|^2$；RNN / LSTM / Seq2Seq 用于布料、刮取、开门；ACT + ALOHA（约 50 条示教）、Diffusion、Mamba（MaIL）、隐式 BC（对不连续切换有利）。关键局限：缺乏适应性与复合误差，action chunking 会放大误差；缓解：残差 RL（Ankile 等）、CCIL 纠正标签、人类反馈 / ThriftyDAgger / LazyDAgger。
- **DMP（§4.2，式 2–4）：** 弹簧阻尼变换系统 + 相位变量 + 高斯基函数强迫项；扩展：四元数 / 黎曼流形 G-DMP、ProMP、KMP、ProDMP；接触任务中多模态靠「并行力轨迹（DMP/GMM）+ 阻抗自适应」或「重构（CMP 加关节力矩、FA-ProDMP 力感知、可逆 DMP、网格 DMP）」；常用 IL 初始化后接 RL 精修。
- **生成式（§4.3）：** VAE / CVAE（含 ACT、MaIL 的 CVAE 式结构）；基础模型 / VLM / VLA（RT-2、OpenVLA 类）——作者指出现有 FM 多在准静态位置控制任务上成功，需调力的动态接触任务仍是挑战；Diffusion Policy、Consistency Policy、Bi-ACT（位置 + 力）、GR-1、Elemental。
- **IRL / 对抗（§4.4）：** 从示教反推奖励；GAIL / AIL 以判别器作隐式奖励，有 6 µm 间隙 peg-in-hole 等成功案例。
- **多模态 IL（§4.5）：** 力 / 本体 / 触觉 / 视觉 / 语言 / 视频的组合；视触觉对比预训练即使部署时去掉触觉编码器也提升性能；无触觉时可用阻抗控制间接调力；「统一大模型」vs「多模型组合」（如 PoCo 在推理时对多个扩散策略乘积采样）；Liu 等 (2025) 把力加入点云扩散策略。
- **离线 RL（§4.6）：** 预训练 + 微调（PTR、IQL+SAC）、Q-Transformer、状态条件动作量化、少量专家数据内在奖励；朴素微调会引入分布偏移。
- **其他（§4.7）：** 世界模型、零 / 单样本 IL、黎曼流形 IL、次优 / 少量示教、无监督 + IL（LOTUS）、优化型 IL（双层优化 / 约束层）。
- **应用（§5）：** 工业（装配、插入、去毛刺、打磨、可变形物）；家庭（擦拭、搅拌、开门 / 抽屉、切菜、倒液体，多为实验室通用平台，灵巧硬件成本高）；医疗（手术如 SRT / 打结、康复）。
- **结论与方向（§6）：** 三个方向：①基于双过程（System 1 / 2）的分层架构；②多模态感知（尤其触觉）；③sim-to-real；基础模型预计影响这三者。

## 对 wiki 的映射

- [`wiki/concepts/contact-rich-manipulation.md`](../../wiki/concepts/contact-rich-manipulation.md) — 核心原理（RL / IL 取舍、模态、示教、方法族）、工程实践与局限三节的原文依据。
- [`wiki/entities/paper-imitation-learning-contact-rich-tasks-survey.md`](../../wiki/entities/paper-imitation-learning-contact-rich-tasks-survey.md)
