# MaCoPlanner: LLM-Assisted Manual-Compiled Task Planning with Proactive Safety Verification for Robotic Industrial Panel Operation

- **作者：** Guipeng Xin, Jiahe Xu, Mohammad Deghat, Chenhui Wan, Jie Liu, Youmin Hu, Zhongxu Hu
- **机构：** 华中科技大学（Huazhong University of Science and Technology, HUST）；新南威尔士大学（University of New South Wales, UNSW）
- **版本 / 载体：** arXiv:2608.28300v1 [cs.RO]（2026-08-28，HTML 全文，License: CC BY-NC-ND 4.0）
- **原文：** [arXiv 摘要](https://arxiv.org/abs/2608.28300) · [arXiv HTML 全文](https://arxiv.org/html/2608.28300)
- **项目页 / 代码：** 已开源 — 官方仓库 [XinGP/MaCoPlanner](https://github.com/XinGP/MaCoPlanner)（MIT License）；包含 `src/macoplanner` 统一规划 API/CLI、3 类设备（`engine` / `generator` / `vfd`）编译知识与规则、400 条检索基准、1,440 条结构化规划评测记录及单元测试；原始 PDF 手册、ROS 构建树、厂商驱动与硬件标定文件未随仓发布（见仓库 `docs/open-source-audit.md`）
- **核对范围：** 通读 arXiv HTML 全文（§1 引言、§2 相关工作、§3 MaCoPlanner 框架、§4 手册引导定位与设备记忆、§5 实验 §5.1–§5.10、§6 结论、附录 A–D 及全部表 Table 1–18；Fig. 1–15 仅依据正文与图注核对，未查看图像文件）以及官方 GitHub 仓库主页、README 与目录结构列表（静态核对）
- **通读：** 已完成

## 原文核对笔记

### 1. 研究动机与核心框架（§1–§3）

- **问题背景：** 工业设备操作面板（HMI）包含按钮、旋钮、开关与指示灯，其规划不仅受物理定位精度限制，更受分散在操作手册、安全规范与参数说明书中的流程顺序、模式互锁（interlock）与设备状态约束支配。直接将原始长文本手册塞入 VLM 上下文难以稳定捕获跨步骤依赖。
- **离线手册编译（§3.2.1）：** 将原始手册集合 $D$ 编译为可追溯的四元组类型化中间表示（IR）：
  $$\mathcal{I} = \operatorname{Compile}(D) = (\mathcal{T}, \mathcal{S}, \mathcal{P}, \mathcal{G})$$
  其中 $\mathcal{T}$ 为流程骨架集（步节点 $s_i = \langle \phi_i^{pre}, a_i, \phi_i^{post}, \beta_i, \rho_i \rangle$），$\mathcal{S}$ 为触发-禁止安全条款集（$k_j: \phi_j^{trig}(x) \Rightarrow \neg \psi_j^{forbid}(x, a)$），$\mathcal{P}$ 为参数规范集（$p_k = \langle u_k, [l_k, r_k], \epsilon_k, g_k(\cdot) \rangle$），$\mathcal{G} = (V, E)$ 为控件-步骤-参数结构关系图；每条编译项通过 $\ell(i) = (d, \omega)$ 回指原始手册片段。
- **在线查询分解与证据门控（§3.2.2）：** 针对指令 $q$ 与设备记忆 $m_t$，先由 `gpt-4o` 按四类槽位分解为子查询 $Z = \bigcup_{c \in \{\mathrm{proc,safe,param,rel}\}} Z_c$（每类至多 1 条，并经确定性模式校验），再用共享 `intfloat/e5-base-v2`（L2 归一化、512-token 分块、64-token 重叠）向对应类型索引各检索 $L = 5K$ 候选，最后执行可用性门控：
  $$Score(e, q_{(c,i)}, m_t) = \alpha\,Sim(e, q_{(c,i)}) + (1 - \alpha)\,App(e, m_t), \quad \alpha = 0.70,\ \tau = 0.55$$
  其中硬状态冲突 $C_h = 1$ 直接一票否决（$App=0$），且不因语义高分被补偿。
- **双层主动安全验证与有界修复（§3.3，Algorithm 1）：**
  1. 依据门控证据 $E_t$ 与 $m_t$ 生成初始符号动作原语序列 $P^{(0)}$；
  2. 通过抽象转移算子 $\hat{f}$ 将计划展开为符号轨迹 $\xi^{(k)}$；
  3. 同时执行 **全局时序检查**（有限迹线性时序逻辑 LTLf，$\sigma^{(k)} \models \Phi$）与 **局部状态机检查**（Safety FSM，检查模式切换与互锁合法性，如换向必须先经 `OFF`、关机前必须先断负载）；
  4. 若违例，定位最早违例步 $i_{viol}$ 及规则/原因并构造结构化反馈 $F^{(k)}$，调用 VLM 做局部化修复，最多 $K_{\max} = 3$ 轮；若预算耗尽仍未同时通过两层检查，则 **fail-closed 拒止**（`RejectTask`），不输出也不下发物理动作。

### 2. 物理定位与设备记忆接口（§4）

- 视觉与几何管线：以 SAM2 分割掩码、DINO-X 开放词汇检测与 OCR 文本锚点做语义粗对齐；结合手册空间关系 $\mathcal{R}_s = \{\text{left\_of}, \text{right\_of}, \text{above}, \text{below}, \text{near}, \dots\}$ 用匹配代价 $S_{\mathrm{match}}(c) = \sum \|p_i^{(c)} - p_i^*\|^2 + \lambda \sum g_r(p_a^{(c)}, p_b^{(c)})$（$\lambda=0.5$，阈值 $0.65$）筛选候选，仅在语义歧义或平局时调用 VLM。
- 操作参数提取：使用 Intel RealSense D435i 深度图经 FoundationStereo 恢复三维点云，滤波后通过 RANSAC 拟合面板基准平面（阈值 `0.005 m`），对齐坐标系后用 DBSCAN（$\epsilon_{\mathrm{db}} = 0.015\text{ m}$、`MinPts = 10`）聚类控件点云，针对旋钮拟合圆柱/圆盘半径与高度（$(r_{\text{knob}})$），针对按钮估计按压行程与触发阈值（$\delta_{\text{button}}, \eta_{\text{button}}$）。
- 设备记忆：以 JSON 槽位 `{mode, control_states, parameters, last_action, verification_status, confidence, timestamp}` 维护状态，设置 `5 s` 陈旧阈值（staleness threshold），避免未刷新推断覆盖已验证状态。

### 3. 关键实验结论与数字核对（§5 & 附录 B–D）

- **全语料与编译保真度（Table 2 & Table 16）：** 4 类设备手册（Motor-drive controller simulator、Generator control system、Laser controller、Engine controller）共 304 页 / 347 块，编译得 463 项（$|\mathcal{T}|=50, |\mathcal{S}|=107, |\mathcal{P}|=70, |\mathcal{G}|=236$）。独立标注的 350 单元审计集上，总体完整性 **97.14%**（340/350），语义一致性 **97.32%**（$\mathcal{S}$ 完整性 100.00%、$\mathcal{T}$ 94.59%、$\mathcal{P}$ 98.15%、$\mathcal{G}$ 95.53%）。
- **端到端规划与安全评测（480 任务：L1=80, L2=200, L3=200，16,000-token 预算，Table 3 & §5.6）：**
  - MaCoPlanner：L1 **93.5%**、L2 **84.4%**、L3 **43.2%**、独立安全 oracle 违例率 **2.7%**；进入规划修复环的 357 次运行中 **26.3%** 在耗尽 $K_{\max}=3$ 预算后被 fail-closed 拒止。
  - 对比基线（Table 3）：Raw-Manual Long-Context（88.2% / 62.8% / 25.9% / 违例 11.6%）、Prompt-Safety（93.1% / 70.5% / 33.8% / 8.3%）、ISR-LLM（93.6% / 74.2% / 35.7% / 6.8%）、$\text{LLM}^3$（93.4% / 76.8% / 37.1% / 5.2%）、SafePlan（92.7% / 79.3% / 39.6% / 3.5%）。
- **形式化规则审计与 PDDL 诊断（§5.7 & Table 18）：** 81 条 LTL 规则中 75 条语义准确、6 条为保守近似；48 条 FSM 转移/守卫中 45 条语义准确；50 任务子集上 Fast Downward（`lama-first`）全部可求得非空符号解；L3 动作中 42% 为状态确认动作。
- **物理感知与长时程真机实验（§5.8–§5.9，Table 4–5，无工业负载控制器面板模拟器 + Dobot CR3 + AG95 夹爪 + RealSense D435i）：**
  - 5 视角控件测量与重复操作成功率（Table 4）：Buttons **96.0%**、Small buttons **93.0%**、Knobs **95.0%**、Small round knob **93.0%**、Emergency stop **80.0%**（中心偏差 1.20±0.06 mm、按压偏差 3.50±0.18 mm）。
  - 5 个长时程任务各 5 次共 25 次试验（Table 5）：总完成 **23/25**，首次尝试完成 **19/25**，平均规划/API 耗时 **4.2 s**、物理执行 **3.9 s**、平均重规划 **0.36** 次、平均安全触发 **0.16** 次；12 例失败归因中检索漏召回占比最高（§5.10）。

对 wiki 的映射：[MaCoPlanner](../../wiki/entities/paper-agentic-2608-28300.md)
