# ReCoVLA: VLM-Guided Reward Compilation for Failure Recovery in Vision-Language-Action Policies

- **作者：** Haodi Hu, Chung-Ta Huang, Jing Liu, Ye Wang, Kei Suzuki, Matthew Brand, Toshiaki Koike-Akino
- **机构：** University of Southern California (USC), USA; Harvard University, USA; Mitsubishi Electric Research Laboratories (MERL), USA
- **版本 / 载体：** arXiv:2606.09630v1 [cs.RO]（2026-06-08 提交，19 pages, 7 figures，arXiv HTML 全文）
- **原文：** https://arxiv.org/html/2606.09630 （摘要页：https://arxiv.org/abs/2606.09630）
- **项目页 / 代码：** 未确认开源代码（截至 2026-11-27，arXiv:2606.09630 原文与摘要页均未公布 ReCoVLA 官方源码仓库或独立项目主页；Appendix A.1 脚注引用的 `https://github.com/mli0603/openpi-comet` 为 Behavior-1K 对比基线 OpenPI Comet 的仓库，并非本文官方实现）
- **核对范围：** 已通读 arXiv:2606.09630v1 HTML 全文，覆盖 Abstract、Section 1–6（含方法 3.1–3.3、公式 Eq. 1–10、Fetch 仿真/真机/OOD 评测与局限性分析）及 Appendix A.1–A.4（Behavior-1K Table 1、RLinf/RACER 对比 Table 2、任务提示 Table 3、Algorithm 1 确定性奖励编译、主定量结果 Table 4、VLM 检测混淆矩阵 Figure 7 与训练超参数 Table 5）。
- **通读：** 已完成

## 原文核对笔记

- **核心定位与问题：**
  - 预训练 VLA（如 $\pi_{0.5}$、OpenVLA）在分布内标称任务上先验强，但一旦因操作偏差进入非标称/失败状态（如物体掉落、放错容器、夹爪持物未释放、工具箱盖未合上），模仿学习策略往往缺乏针对性恢复能力。
  - 直接微调 VLA 存在恢复演示采集成本高、遗忘原有能力的问题；直接用 RL 优化流匹配 VLA 缺乏动作似然，且通用任务级奖励稀疏或同时激活多项奖励易引发目标冲突与 reward hacking。
- **三阶段框架设计：**
  1. **失败目录构建（3.2）：** 在仿真（OmniGibson）中运行冻结基础 VLA（M1），由外部 VLM（`Qwen3-VL-8B-Instruct`）分析完整失败轨迹 $\tau_i^{\mathrm{M1}}$ 与任务指令 $\ell_i$，输出结构化恢复描述 $\xi_i=(c_i,z_i,E_i,\rho_i,m_i)$（失败类别、恢复阶段、活跃实体、置信度、奖励掩码），过滤 $\rho_i \ge \tau_{\mathrm{cat}}$ 得到可恢复失败目录 $\mathcal{C}_{\mathrm{train}}$。
  2. **确定性奖励编译与分失败类别残差 RL 训练（3.3 & Algorithm 1）：** VLM 不直接生成动作或自由形式数值奖励，仅充当语义奖励选择器。编译器对标签规范化、解析并绑定实体语义角色，从原语库 $\{\varphi_{\mathrm{dist}}, \varphi_{\mathrm{grasp}}, \varphi_{\mathrm{place}}, \varphi_{\mathrm{close}}\}$ 抽取势能差 $\Delta\varphi_k$，并按阶段挂接二值阶段门控 $g_{c,k}(s_t; E_c)$（例如先抓取再放置、先接触再铰接闭合），叠加残差正则惩罚 $\Omega_t(a^r)$、动作变化与符号翻转惩罚后，使用 PPO 在冻结 VLA 隐特征 $h_t$ 上训练各失败类别的 MLP 残差策略 $\pi_\theta^r(\cdot \mid h_t)$，以 $a_t = \Pi_{\mathcal{A}}(a_t^b + \beta_t a_t^r)$ 执行。
  3. **零样本真机部署与调度（4.1 & A.4）：** 在实体 Fetch 机器人上默认执行冻结基础 VLA；外部 VLM 并行监测 5 帧稀疏 RGB 历史（间隔 $\Delta=15\mathrm{s}$），仅当 $\rho_t \ge \tau_{\mathrm{deploy}}$ 且 $c_t \in \mathcal{C}_{\mathrm{train}}$ 时置 $\beta_t=1$ 并调度对应残差专家策略；若遇到未见过类别或描述校验被拒绝则回退基础 VLA。
- **主要实验结果（每任务每方法 20 次试验）：**
  - **Fetch 仿真均值（M1–M6，Table 4）：** M4（$\pi_{0.5}$ + ReCoVLA 阶段门控编译）平均成功率/Q-score 为 `0.67 / 0.83`，高于 M1 无恢复 `0.37 / 0.56`、M2 任务级残差 `0.40 / 0.55` 与 M3 无门控失败掩码 `0.48 / 0.63`；在 OpenVLA 上 M6 将 M5 从 `0.23 / 0.33` 提升至 `0.45 / 0.55`。
  - **Fetch 真机零样本均值（Table 4）：** M4 平均达到 `0.62 / 0.75`（整理工具箱 `0.50 / 0.69`、蔬菜分拣 `0.60 / 0.72`、易拉罐回收 `0.75 / 0.83`），优于 M1 `0.27 / 0.40`、M2 `0.27 / 0.40`、M3 `0.40 / 0.51`；OpenVLA 从 M5 `0.15 / 0.25` 提升至 M6 `0.35 / 0.42`。
  - **真机 OOD 物体替换压力测试（Figure 5 / Table 4）：** 南瓜→番茄、易拉罐→高罐、线缆→胶带下，M4 平均 `0.53 / 0.65`，M1 为 `0.10 / 0.22`。
  - **扩展对比（Appendix A.1）：** 仿真三任务对比 RLinf (`0.12 / 0.19`) 与 RACER (`0.43 / 0.60`)，M4 达 `0.67 / 0.83`（Table 2）；在 Behavior-1K 的 Galaxea R1 Pro 三任务上，ReCoVLA 平均 `0.37 / 0.60`，高于 OpenPI Comet 的 `0.22 / 0.40`（Table 1）。
  - **VLM 检测器真机混淆矩阵（Figure 7）：** 共 117 次真机检测，总体聚合准确率 `84.6%`，宏平均准确率 `85.0%`（类间标准差 `6.0%`）。
- **局限性核对：** 需先在仿真中复现对应失败状态分布并离线训练残差策略，暂不支持真机单次试验中在线合成或训练新恢复策略；当前真机评测规模限于 Fetch 的 3 类桌面/移动操作任务各 20 次试验；零样本迁移仍受 sim-to-real 偏差、接触变化及 VLM 细粒度误分类影响。
- **对 wiki 的映射：** [ReCoVLA: VLM-Guided Reward Compilation for Failure Recovery in Vision-Language-Action Policies](../../wiki/entities/paper-agentic-2606-09630.md)
