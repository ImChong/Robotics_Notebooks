# From Reaction to Anticipation: Proactive Failure Recovery through Agentic Task Graph for Robotic Manipulation

- **作者：** Sheng Xu, Ruixing Jin, Huayi Zhou, Bo Yue, Guanren Qiao, Yueci Deng, Yunxin Tai, Kui Jia, Guiliang Liu
- **机构：** 香港中文大学（深圳）数据科学学院（School of Data Science, The Chinese University of Hong Kong, Shenzhen）、跨维智能（DexForce Technology）、深圳河套研究院（Shenzhen Loop Area Institute）
- **版本 / 载体：** arXiv:2605.11951v1 [cs.RO]（2026-05-12 提交）；18 页，已录用于 Robotics: Science and Systems (RSS) 2026
- **原文：** [arXiv:2605.11951 (HTML v1)](https://arxiv.org/html/2605.11951) / [arXiv 摘要页](https://arxiv.org/abs/2605.11951) / [官方项目页](https://shengxu.net/AgentChord/)
- **项目页 / 代码：** 已开源 — 官方代码仓库 [EDEM-AI/AgentChord](https://github.com/EDEM-AI/AgentChord)（Apache-2.0，76 stars，README 所列 `git clone` 入口及项目页 [edem-ai.github.io/AgentChord](https://edem-ai.github.io/AgentChord)）；论文 Appendix F 另列出 [Jasonxu1225/AgentChord](https://github.com/Jasonxu1225/AgentChord)（同描述与 Apache-2.0 协议）。仓库基于 EmbodiChain 发布层次化智能体（`task_agent.py`、`recovery_agent.py`、`compile_agent.py`）、监控函数、原子动作、三项仿真任务及 CobotMagic 真机部署脚本；README 标注更多失效监控函数与恢复模板仍在扩展中。
- **核对范围：** 通读 `arXiv:2605.11951v1` HTML 全文（Abstract、§I–§VI 正文、Table I–V、Fig. 1–8 图注、Algorithm 1、Appendix A–F 全部内容），同步核对官方项目页文本（Overview / Method / Results / Videos）与官方仓库 [EDEM-AI/AgentChord](https://github.com/EDEM-AI/AgentChord) 元数据、`README.md` 及核心文件清单。
- **通读：** 已完成

## 原文核对笔记

- 对 wiki 的映射：[From Reaction to Anticipation: Proactive Failure Recovery through Agentic Task Graph for Robotic Manipulation](../../wiki/entities/paper-agentic-2605-11951.md)

### 1. 研究动机与核心思想（§I–§II）

- **现有范式痛点：** 现有基于 MLLM 的失败处理多采用“检测—推理—恢复（detect-reason-recover）”反应式流水线：直接 VQA 推理存在 API/本地计算延迟及语言指令空间粒度不足的问题（如 Inner Monologue、DoReMi）；即便引入代码化监控（如 ReKep、Code-as-Monitor），失败后仍多通过整条流水线重新调用（高延迟）或向后回溯重放（在异步/同步双臂长时程任务中不可靠）来恢复。
- **AgentChord 方案：** 提出以“恢复增强有向任务图（recovery-augmented directed task graph）” $\mathcal{G}_{\text{aug}}$ 为统一表征，在执行前即完成失败模式预判与恢复分支编译，使在线监控触发时无需重新调用 MLLM 即可立即切换至前向恢复路径。

### 2. 系统接口、三智能体编排与求解机制（§III, Appendix C）

- **工具与感知特征接口（§III-A, Appendix C-D）：**
  - 三类功能工具：感知工具 $\mathcal{F}_{\text{perc}}$（SAM3 开放词汇分割与追踪、AnyGrasp 6-DoF 抓取生成、FoundationPose）、原子动作库 $\mathcal{F}_{\text{action}}$（Table IV 共 10 个原语：`drive`、`grasp`、`open_gripper`、`close_gripper`、`move_to_obj`、`move_to_target`、`move_by_offset`、`rotate_eef`、`place`、`back`）、多模态模型调用 $\mathcal{F}_{\text{mllm}}$（实验统一采用 GPT-5）。
  - 任务相关特征向量 $z_t = \Phi(\hat{x}_t, \xi_t)$：融合 RGB-D 感知输出 $\hat{x}_t = \Psi(o_t)$（物体掩码、深度融合点云 $\mathcal{P}_t^k$ 质心/极值点/PCA 主轴/比例位置）与机器人本体信号 $\xi_t = (q_t, g_t)$（关节配置、夹爪开合宽度/状态）。
- **Task Structuring Agent（Composer，§III-B）：** 由指令 $\mathcal{I}$ 和初始观测 $o_0$ 构建名义有向任务图 $\mathcal{G} = (V, E)$，节点 $v^i$ 绑定子目标约束 $\mathcal{C}_{\text{sub}}^i$，边 $\varepsilon^{i\rightarrow j}$ 绑定路径约束 $\mathcal{C}_{\text{path}}^{i\rightarrow j}$。
- **Recovery Orchestration Agent（Arranger，§III-C）：** 为每条名义边 $\varepsilon^{i\rightarrow j}$ 预测潜在失败模式 $\mathcal{F}^{i\rightarrow j} = \{f_1^{i\rightarrow j}, f_2^{i\rightarrow j}, \dots\}$，注入恢复节点 $V_{\text{rec}}$ 与恢复边 $E_{\text{rec}}$ 得到 $\mathcal{G}_{\text{aug}} = (V \cup V_{\text{rec}}, E \cup E_{\text{rec}})$。通过**前向恢复原则** $\mathrm{Dist}(v_m^{\text{rec}(i,j)}, v^N) \le \mathrm{Dist}(v^{\text{fail}}, v^N)$（Eq. 8）过滤会导致剩余目标距离增加的回退分支。
- **Execution Compilation Agent（Conductor，§III-D, Appendix C-E–C-G）：**
  - **节点求解（Eq. 9）：** 采用 Dual Annealing 全局搜索 + SLSQP 局部精修，在硬约束 $\mathcal{C}_{\text{sub}}^i$ 下最小化碰撞（ESDF/SDF）、IK 可达性、位姿正则、时序一致性与双臂防自碰代价。
  - **边求解（Eq. 10）：** 滚动时域（receding-horizon）优化未来 $H$ 步末端位姿序列并执行前 $M=5$ 步后重规划，惩罚平滑度、安全距离与 IK 残差。
  - **编译期监控与触发（Eq. 11–19）：** 为失败模式编译标量违背函数 $f_m^{i\rightarrow j}(z_t) \le 0$（涵盖位移漂移 $f_{\text{shift}}^k$、倾倒 $f_{\text{tilt}}^k$、未抓牢/滑脱 $f_{\text{grasp}}, f_{\text{attach}}^k$、遮挡/追踪丢失 $f_{\text{vis}}^k$、相对关系失准 $f_{\text{rel}}^{k,l}$）。仅当违背超过阈值 $\epsilon$ 且持续 $K$ 步（$\mathcal{M}_m^{i\rightarrow j}(t)=1$）时，立即按映射 $\rho(\varepsilon^{i\rightarrow j}, f_m^{i\rightarrow j}) = \varepsilon^{i\rightarrow \text{rec}(i,j,m)}$ 切换至预编译恢复边，不触发 MLLM 重规划。

### 3. 实验评测与消融发现（§IV, Appendix A–E）

- **硬件与任务设置（§IV-A, Appendix A–B）：** 平台为 AgileX CobotMagic 固定底座双臂机器人（双 6-DoF 臂 + 最大张开 10 cm 平行夹爪）+ 2 台 10 Hz 双目立体 RGB-D 相机（俯视+前视）。评测 6 类任务（每类 20 次试验，变换物体实例并随机采样初始位姿）：Single-arm pour water、Dual-arm pour water、Rearrange table、Handover block、Fold towel、Setup coffee tray。
- **仿真（EmbodiChain，Table I，每步原子动作以概率 $p \in \{0.05, 0.10\}$ 随机掉落手中物体）：** 3 项任务平均成功率 AgentChord 达 **99.2%**（IM 79.2%、DRM 92.5%、ReKep 92.5%、CaM 97.5%），平均执行时间 **41.5 s**（DRM 105.9 s、ReKep 54.4 s、CaM 78.1 s），平均步数 **352 步**（DRM 373、ReKep 391、CaM 356）。
- **真机（Table II，6 项任务每回合施加 1–2 次相同扰动）：** 平均成功率 AgentChord 达 **77.5%**（IM 59.2%、DRM 66.7%、ReKep 65.0%、CaM 72.5%），平均耗时 **92.2 s**（DRM 143.5 s、ReKep 107.1 s、CaM 130.9 s）。其中可变形物体 Fold towel 仅 55%（各基线 40%–50%），Setup coffee tray 为 60%。
- **前向恢复图消融（Table V，Appendix E-C）：** 在 3 项真机任务上对比保留编译监控但改为回退重放的变体 `AC-BT` 与 `ReKep`：平均成功率 `ReKep` 68.3%、`AC-BT` 73.3%、`AC` **86.7%**；平均耗时 `ReKep` 122.4 s、`AC-BT` 128.4 s、`AC` **105.6 s**。证明前向恢复分支是提升成功率与缩短耗时的核心。
- **恢复轨迹用于策略学习（Table III，§IV-E）：** 在仿真单臂倒水（抓瓶后移动杯子扰动，50 次试验）上，将 Sim2Real-VLA 的 40 条微调演示中的一半替换为 AgentChord 生成的可恢复轨迹，成功率从 `26/50` 提升至 **`39/50`**（对比 RDT `16/50`、$\pi_0$ `20/50`、GR00T N1.5 `15/50`）。
- **失效分析与局限（§IV-D, §V）：**
  1. 长时程推理遗漏（未预判的失效模式或选错并回节点，如交接位移需后退重张夹爪）；
  2. 倒地/歪斜物体重抓时受限于 IK 可解性与工作空间边界；
  3. 感知掩码/点云噪声或编译期选错监控特征（如用物体-夹爪距离代替更可靠的夹爪开合度）；
  4. 依赖 PCA 主轴的特征在透明、可变形或不规则物体上易失效。
