# BFM 综述 arXiv:2506.20487 通读（支撑 BFM 41 篇技术地图）

> 来源归档（原文通读 · 支撑 `wiki/overview/bfm-41-papers-technology-map.md`）

- **标题：** A Survey of Behavior Foundation Model: Next-Generation Whole-Body Control System of Humanoid Robots
- **作者：** Mingqi Yuan, Tao Yu, Wenqi Ge 等（PolyU、LimX Dynamics、东方理工、HKU、EPFL 等）
- **类型：** paper / survey / humanoid whole-body control
- **原文：** <https://arxiv.org/abs/2506.20487>（2026-11 通过 arXiv PDF v5，2026-02-09 版，19 页，pdftotext 获取全文）
- **一句话说明：** 首篇 BFM 综述：把人形 WBC 从模型法、任务专用学习法演进到 BFM；预训练分 goal-conditioned / intrinsic reward / forward-backward 三线，适应分微调与层次化控制；并讨论应用、局限、机会与风险。
- **核对范围：** 通读正文 §I–§VI（引言、背景、§III 预训练与适应、§IV 应用与局限含 Table III 数据集、§V 机会与风险、结论）及图表说明；参考文献列表（[1]–[222+]）仅浏览，未逐条核对；未阅读综述所引 41 篇单篇论文原文，各篇细节仅依赖本库各自的 source 页。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **演化三段（Fig.2）：** 模型法（MPC、WBOSC，2006 起）、学习式任务专用控制器（DeepMimic、AMP 等，2018 起）、BFM（Motivo、BFM-Zero、SONIC 等，2024 起）。模型法局限：任务设计与调参费人力、实时 MPC 难处理高维、难做高动态技能、鲁棒性弱。学习法局限：RL 样本效率低且依赖奖励设计、Sim2Real 缺口、IL 受数据与示教者偏差影响、泛化弱。
- **BFM 定义：** 一类控制智能体行为的基础模型子类，在大规模行为数据（人类演示、交互轨迹）上预训练，编码广泛行为谱而非单任务专精；与 VLA 的区别是 BFM 直接控制 locomotion / manipulation / interaction，面向全尺寸人形的 WBC，而多数 VLA 面向机械臂或轮式平台。术语 BFM 最早出自 FB-IL 工作。
- **预训练三线（Table II，带 * 为真机部署）：**
  - goal-conditioned：TeamPlay、ASE、PHC、CALM、MTM、CASE、InterMimic、MoConVQ、H-GAP、MaskedMimic、HugWBC*、HOVER*、ModSkill、CLONE*、TWIST*、TWIST2*、AMS*、Any2Track*、BFM4Humanoid*、SONIC*。动机是学跟踪单个姿态比模仿整段动作更易泛化；HugWBC 把 WBC 改写为自监督命令跟踪，无需动作数据。
  - intrinsic reward：ICM、DIAYN、RND、APS、ProtoRL、RE3、ODPP；多在单环境训练，先验较弱，被视为概念相关的原型，而非严格 BFM。
  - forward-backward：FB、FB-IL、FB-AW、FB-ARE、FB-AWARE、Motivo、BFM-Zero*。核心是后继测度的有限秩近似 M(X|s,a)≈∫F(s,a)^T B(s')ρ(ds')，给定奖励 r 推 z=E[B(s)r(s)]，Q=F^T z，实现零样本策略推断；目标跳到 z_s=B(s) 即目标到达。
- **FB 的局限与改进：** 有限维 d 下的低秩假设与离线数据覆盖差会导致塌缩；Motivo 的 FB-CPR 用判别器正则（把 KL 当作回报）使策略对齐无标签人类动作数据，文中报告 motion tracking 成功率 83%、奖励优化达 top-line 的 61%、300 步 episode 约 12 秒；BFM-Zero 引入双 critic（对抗模仿 + 稳定性辅助奖励）、域随机化、历史策略、大残差网络，零微调部署到 Unitree G1 与 Booster T1。
- **适应：** 预训练自带表征级泛化（给目标、潜变量或奖励嵌入即可）；微调线含 ReLA / LoLA（最多提升约 40%）、TaskTokens（长跳等任务最高 99.75%）、Belief-FB（最多约 2 倍）、TokenHSI；层次化线含 UniHSI、CLoSD、UniPhys、LangWBC*、LeVERB*、BeyondMimic*、SENTINEL*，由高层规划（LLM / 扩散）生成子任务、BFM 作低层执行。
- **应用（§IV-A）：** 人形通用加速器、虚拟角色与游戏、Industry 5.0、医疗与辅助机器人。
- **局限（§IV-B）：** Sim2Real（文中称当前 BFM 基本局限于仿真，尚无大规模真机部署记录）、计算与系统集成（控制频率 50-100Hz；BFM-Zero 约 44.5M 参数、SONIC 约 42M）、数据瓶颈（Table III：Humanoid-X 163800 clips / 240h，Motion-X++ 120462 / 180.9h，PHUMA 76000 / 73h，AMASS 11265 / 40h 等，规模远小于 LLM 数据；缺少本体感觉 + 视觉 + 接触对齐的大规模数据）、具身泛化（形态、执行器、传感器不匹配）。
- **机会与风险（§V）：** 多模态 BFM、与 LLM 的高层系统、scaling law（FB-CPR 参数 25M 到 288M 有提升；数据 scaling 更关键但未被充分研究）、后训练（RLHF、测试时扩展）、多智能体、评测机制缺失；风险为伦理（数据偏见、隐私）与安全（对抗扰动、跨模态混淆）。
