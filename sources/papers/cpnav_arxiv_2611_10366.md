# CPNAV: Calibrated Vision–Language Navigation for Object-Goal Search in Unseen Environments（arXiv:2611.10366）

> 来源归档（paper）

- **标题：** CPNAV: Calibrated Vision–Language Navigation for Object-Goal Search in Unseen Environments
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2611.10366>（v1，2026-11-07，cs.RO）
- **PDF：** <https://arxiv.org/pdf/2611.10366v1>
- **代码：** https://github.com/practice-lab-ucla/CPnav
- **项目页：** <https://practice-lab-ucla.github.io/papers/cpnav/>
- **作者与机构（论文首页）：** Qizhao Chen、Shoh Nishino、Anushri Dixit（加州大学洛杉矶分校 UCLA 机械与航空航天工程系）；Yuanhong Zeng（UCLA 电气与计算机工程系）
- **入库日期：** 2026-11-10
- **一句话说明：** 用保形预测（conformal prediction）把 VLM 给导航候选动作打的分数离线校准成阈值，在线过滤低置信动作并配合树状探索历史回退；Habitat 仿真与公寓真机均优于未校准基线。
- **核对范围：** 通读 arXiv v1 PDF 正文全部章节（摘要、I 引言、II 相关工作、III 方法含公式 1–9、IV 实验设置含 Table I、V 结果含 Table II–IV 与 Fig. 4、VI 结论与局限、参考文献列表）以及附录 A（命题 1 的证明）、B（硬件环境 Fig. 5）、C（受限步数 Table V）、D（token 消耗 Table VI）、E（定性分析 Fig. 6）；图仅看图注与图中文字，未逐像素核对。同时静态阅读了 GitHub 仓库 README 与项目页文字；未下载或运行代码，未核对仓库源码、DATASET.md、配置文件与论文的一致性，也未阅读论文引用的原始文献。
- **通读：** 已完成

## 开源状态

- 项目页与仓库均可访问（2026-11-10）：仓库 README 描述为 CPNAV 的 conda 环境、OpenAI API key 配置、HM3D/MP3D 数据集准备，以及先离线校准 / 搜索再在线运行的两阶段用法。
- 许可证、模型权重与校准数据是否随库发布：**未核实**。

## 核心摘录

1. **问题（引言）：** 在无地图、未见过的室内环境做物体目标导航；VLM 对动作的置信度未校准，常给不安全、冗余或与目标无关的动作不低的分数。
2. **方法（III）：** RGB-D 经导航性掩码（高度变化小于 5 cm 视为可通行）生成碰撞过滤后的候选动作（距离短于 0.75 m 的剔除，长于 1.7 m 的截断，另加原地掉头动作 a0）；两个 VLM 查询：stopping VLM 给出是否到达 done∈{0,1} 与全局语义值 GSV∈[0,1]，action VLM 给每个候选动作打 [0,1] 分；融合分数 s = GSV × 动作分（式 1–2 之后）。
3. **树状探索（III-C）：** 探索历史 H_k={(x_t, A_t, S_t)} 是「探索历史结构而非规划树」，不做 rollout 或价值估计；局部回退（回到父节点执行未试过的最高分动作）与全局回退（在全历史中找仍有未试动作的节点）。
4. **离线 oracle 路径（III-D）：** 以「路径上最小动作分（瓶颈）最大」为准则选 oracle 路径 P*（式 4），用修改的 BFS 搜索，并在大于当前瓶颈的未试动作上继续细化（式 5）。
5. **校准（III-E）：** n=500 个 i.i.d. 校准 episode；非一致性分数 E_j = 1 − b(P*_j)；用 δ=0.01 的 Beta 分位数修正得到调整后风险 α̂（式 6），再取共形分位数 q̂（式 7）；在线预测集 C(ξ_t)={a | s ≥ 1−q̂}（式 8），为空则触发掉头回退，否则执行最高分动作；保证为轨迹层面的 P(a*_t∈C(ξ_t), ∀t>0) ≥ 1−α（式 9，要求校准与部署 episode 可交换）。
6. **设置（IV，Table I）：** VLM 为 GPT-5 Nano；仿真步数上限 200、目标风险 α=0.3；硬件步数上限 100、α=0.45；δ=0.01；校准 / 评测 episode 数仿真 500/200、硬件 500/24；相机高度仿真 0.9 m，水平视场仿真 131°、硬件 101°。评测基准 HM3D v0.2（主）与 MP3D（跨数据集）；成功判据为人工标准（最终观测中目标可见且路径无遮挡，停在错误目标附近算失败）。
7. **仿真结果（V-A，Table II，200 episodes/基准）：** HM3D：CPNAV SR 86.0 / SPL 44.9 / 距离 29.2 m，无校准基线 81.0 / 40.3 / 35.4 m（相对 SR +6.2%，SPL +11.4%，距离 −17.5%）；MP3D：CPNAV 77.0 / 46.8 / 32.1 m，无校准 68.5 / 43.0 / 36.4 m（相对 SR +12.4%，SPL +8.8%）。HM3D 上 STRIVE 82.5 / 41.5 / 17.6 m，WMNav 80.0 / 39.9 / 18.2 m；MP3D 上 STRIVE 71.5 / 42.3 / 21.0 m，WMNav 75.0 / 45.2 / 18.5 m。论文自述 CPNAV 在两者之上 SR / SPL 略高，但行走距离明显更长（STRIVE 短 39.7% / 34.6%，WMNav 短 37.7% / 42.4%），并说明步数定义不可比。
8. **校准阈值（V-A1）：** 仿真 q̂=0.940（α̂=0.25）；硬件 q̂=0.936（α̂=0.39）。
9. **敏感性（V-B，Table III，HM3D）：** 相机高度 0.45 m → SR 67.0；1.70 m → 79.3；HFoV 69° → SR 40.0；101° → 72.7；换 GPT-4o-mini → 71.3；去掉体素地图 → 66.0（默认 86.0）。
10. **硬件（V-C，Table IV）：** 两间公寓共 24 个随机 episode：无校准 SR 66.7 / SPL 42.0 / 距离 9.0 m；CPNAV 83.3 / 48.4 / 6.3 m（相对 SR +24.9%，SPL +15.2%，距离 −30.0%）。
11. **受限步数（附录 C，Table V，HM3D）：** 50 / 100 / 200 步下 CPNAV SR 73.0 / 80.0 / 86.0，无校准 60.3 / 75.7 / 81.0。
12. **token（附录 D，Table VI，MP3D 200 episodes）：** CPNAV 总计 201.6K，无校准 231.4K（少 12.9%），Ensemble Set 1803.3K，STRIVE 99.3K，WMNav 182.1K。
13. **作者自述局限（VI）：** 树节点仅按空间邻近定义、不含朝向；硬件校准使用仿真 episode，保证依赖仿真与真实公寓的可交换性；依赖经 ChatGPT API 访问的 VLM，其随机性会引入打分与校准变异。

## 一致性提示（读者核对时发现）

- 论文正文称模型为 GPT-5 Nano（Table I），第 V-B 节又写 GPT-5o-nano，应为同一模型的笔误（推测）。
- Table II 的「Dist.」一列在正文被称为 mean distance；HM3D 上 STRIVE / WMNav 的距离远小于 CPNAV，说明 CPNAV 的 SR 优势并非来自更短路径。
- 硬件「24 episodes」：正文有「24 trials each in two apartment environments」与「24 randomized episodes across two layouts」两种说法，是总共 24 还是每环境 24 不明确；按 Table I 与 V-C 取总计 24。

## 对 wiki 的映射

- [paper-cpnav](../../wiki/entities/paper-cpnav.md)
