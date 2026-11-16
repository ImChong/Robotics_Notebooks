# X-NavDP：GQRM 跨本体导航扩散策略 RL 后训练（arXiv:2607.28560）

> 来源归档（ingest）

- **英文标题：** X-NavDP: Generalizing Navigation Diffusion Policy to Novel Behavior and Embodiments with Group Q-score Reweighted Matching
- **类型：** paper
- **作者：** Tianyu Yang, Yiming Zeng, Wenzhe Cai, Yuqiang Yang, Jiaqi Peng, Hui Cheng, Jiangmiao Pang, Tai Wang（前三位同等贡献；Tai Wang 通讯）
- **机构：** Fudan University；Shanghai AI Laboratory；Sun Yat-sen University；Tsinghua University
- **版本：** arXiv:2607.28560v1 [cs.RO]，2026-07-30
- **arXiv：** <https://arxiv.org/abs/2607.28560>
- **PDF：** <https://arxiv.org/pdf/2607.28560>
- **Hugging Face Paper：** <https://huggingface.co/papers/2607.28560>
- **项目页：** <https://yty-sky.github.io/x-navdp-project-page/>
- **代码 / 权重：** <https://github.com/InternRobotics/NavDP>（`baselines/x-navdp`）；资产与 checkpoint：<https://huggingface.co/InternRobotics/X-NavDP>
- **项目页源码（静态站）：** <https://github.com/yty-sky/x-navdp-project-page>
- **开源：** **已开源**（MIT，X-NavDP baseline；步骤 2.5 核查见 [项目页归档](../sites/x-navdp-project-page.md)）
- **入库日期：** 2026-09-28；通读升级 2026-11-12
- **核对范围：** arXiv v1 PDF 全文 20 页逐段读完：摘要、1 引言、2 相关工作、3 预备（扩散策略与 DPMD）、4 方法（4.1 至 4.4）、5 实验（表 1 至 3、图 4）、6 结论与局限、附录 A（GQRM 推导）、附录 B（训练细节：奖励、调度、回放、MPC）、附录 C（表 5 至 7）；图 1 至 4 仅读到图注与坐标数字，Fig. 3 定性轨迹图未能看图。未读：参考文献所引各篇原文；未运行官方代码；项目页与 HF README 未在本次重读（沿用此前核查）。
- **通读：** 已完成

## 核心论文摘录（均为论文原文数字，章节在括号内）

### 1) 问题与动机（§1）

- 预训练导航扩散策略依赖全知 oracle 规划器生成的专家轨迹（针对单一名义机器人）；部署只有局部 RGB-D，全局监督与局部观测不匹配，抑制探索与脱困；数据生成对本体不敏感。
- 已有扩散 RL：策略梯度类（DDPO、DPPO）经全去噪链计算似然而不稳；潜空间法（DSRL）冻结核心参数而探索受限；奖励加权 score matching（DPMD）可保留先验，但对高维视觉与长时序导航尚未被定制（§1）。
- 最相关工作：Sheng 等（NavDP-RL）同样在线 RL 微调 NavDP，但探索只靠扩散采样随机性（§2.2）。

### 2) GQRM 与自举扰动（§4.2、§4.3、附录 A）

- 自举扰动：同一观测下采样 goal 条件轨迹与 goal 无关（no-goal）轨迹，按式 (7)(8) 做 `τ_mixed = s ⊙ (τ_pointgoal + λ τ_nogoal)`，s 以概率 ε 对整条 chunk 的 x、y 坐标独立翻转符号。
- GQRM：按同一状态的候选组内 Q 做均值方差归一化，乘 c、裁剪到 [-h, h]（式 10），用 `exp(Q̃_G/λ)` 作为 score matching 权重（式 11）；推导见附录 A。仅保留 top-k 且 Q̃_G>0 的候选。
- 附录 B.2 实际取值：每状态 64 个候选，取 5 个最高；裁剪区间 [-3, 3]；10 步去噪中只微调最低噪声的 6 步；使用学习得到的温度。

### 3) 跨本体（§4.1、§4.4、附录 B、C.1）

- IsaacLab，「500+ 机器人、50+ 场景」同时交互（§4.1）；附录 B 给出具体：56 个训练场景，每场景一个进程、每进程 8 个环境共 448 个并行环境，三种本体平均分配并周期轮换（首次 30,000 步后切换，之后每 24,000 步）。
- 分层控制：扩散策略出 3 秒轨迹，统一 MPC 转底层速度，本体专属预训练步态策略执行；MPC 与步态 25 Hz。
- 本体条件：学得的机器人嵌入既加到动作 token（式 12），又经 FiLM 调制解码后特征（式 13）；附录 C.1 对比 Soft-Prompt Concat。
- 部署期 RTC 式时序一致性引导（式 14、15），仅部署时启用。

### 4) 实验结果（§5、附录 C）

- 仿真：GRScenes-100，56 训练 / 40 留出场景（20 住宅 + 20 商业），三本体 Dingo、Unitree Go2、Unitree G1；表 1：NavDP 61.20/58.95 → X-NavDP 84.28/77.19（SR/SPL，%）；iPlanner 33.84/32.60，ViPlanner 43.87/42.84。人形 NavDP 商业 64.35、住宅 50.70 → X-NavDP 84.25、84.50（§5.2）。
- 真机：Turtlebot、Go2、G1，三个场地 lab / hall / office，每格 10 次（§5.1，表 2）；NavDP 九格为 0/40/0/0/30/0/0/20/0（均值 10%），X-NavDP 为 60/70/60/80/60/70/50/50/80（均值约 64.4%，摘要与引言写 65%）。无 sim-to-real 微调。
- RL 方法对比（24 场景、表 3）：SFT 53.50、DPPO 33.98、DSRL 13.65、DPMD-original 71.57、X-NavDP 80.48（SR）。
- 探索消融（表 6，24 场景）：去掉 no-goal 扰动与反向后 SR 6.23；仅反向 77.89；仅 no-goal 65.51；去外推 77.84；完整 80.48。
- 本体注入（表 5）：56 场景下 FiLM 84.28 vs Soft-Prompt Concat 82.84；24 场景下两者差仅 0.24 SR。
- RTC（表 7）：无 RTC 82.17/75.81，完整 84.28/77.19。
- 后训练耗时：「仅 12 小时」（§1）。

### 5) 作者自述的失败与局限（§5.3、§6）

- G1 窄通道难通过；绕长障碍时因记忆有限可能沿原路返回；透明 / 镂空障碍物难感知；仅依赖短时上下文；仅验证三种本体且需对应预训练底层控制器；真机评测范围有限。

## 对 wiki 的映射

- [../../wiki/entities/paper-x-navdp.md](../../wiki/entities/paper-x-navdp.md)
