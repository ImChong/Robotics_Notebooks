# Zero-Shot Mission-Level Evaluation for Aerial MLLM Agents

- **作者：** Suman Navaratnarajah, Taehyoung Kim, Jona Ruthardt, Ishaan Bhimwal, Ryousuke Yamada, Yannik Blei, Wolfram Burgard, Yuki M. Asano
- **机构：** University of Technology Nuremberg（UTN）、Fraunhofer IVI、THWS、AIST（产业技术综合研究所）
- **版本 / 载体：** arXiv:2607.22014v1 [cs.AI]（2026-07-24 提交），HTML 全文（CC BY 4.0）
- **原文：** [arXiv 摘要](https://arxiv.org/abs/2607.22014) · [arXiv HTML 全文](https://arxiv.org/html/2607.22014) · [项目主页](https://gomtae.github.io/publications/missionbench)
- **项目页 / 代码：** **已开源**（闭环无人机任务评测框架、LangGraph 推理图与策略、提示词模板、实验自动化脚本及 YAML 配置已完整公开于 `FraunhoferIVI/MissionBench`；打包的 UE5/Cosys-AirSim 仿真环境二进制与完整测试集在首次运行 `main_driver.py` 时按需下载，采用 CC BY 4.0 协议。注：仓库 README 中 arXiv 徽章显示文字为 `2604.02327` 而超链接指向 `https://arxiv.org/abs/2607.22014`，引用 BibTeX 条目 key 为 `ruthardt2026steervit`） — [GitHub: FraunhoferIVI/MissionBench](https://github.com/FraunhoferIVI/MissionBench)
- **核对范围：** 已通读 arXiv:2607.22014v1 HTML 全文 §1–§6（Abstract、Introduction、Related Work、MissionBench、Experiments、Conclusion、Acknowledgements）与 Supplementary Appendix A–D（A.1–A.3 局限与伦理、B.1–B.4 扩展实验与失败模式定量拆解、C.1–C6 仿真构建与评测指标、D.1–D.3 推理参数、API 成本与完整提示词），逐项核对 Tables 1–4、Tables B1–B5、Table C6、Table D7 与 Figures 1–3、C1、D2 图注；同步核对官方仓库 `FraunhoferIVI/MissionBench` 主页、`README.md` 与 `files_sample` 文件清单。
- **通读：** 已完成

## 原文核对笔记

- 对 wiki 的映射：[MissionBench 论文实体页](../../wiki/entities/paper-agentic-2607-22014.md)

### 1. 问题定义与基准构建（§3, Appendix C）

- **单指令任务级闭环设定：** 不同于只给逐步路径描述的 VLN（如 AerialVLN、AVDN）或孤立子技能评测，MissionBench 在 Unreal Engine 5 + Cosys-AirSim 构建的 5 个高保真 3D 场景（`Neighborhood`、`City`、`Forest`、`Savannah`、`AirSimNH`）中评测冻结（frozen）、未针对空中任务微调的通用多模态大模型（MLLM）。
- **120 条任务分布与划分（Table 2）：** 共 120 条人类编写并经多人审查的任务指令，覆盖 4 类任务家族：Reporting（57）、Inspection（38）、Manipulation（16）、Patrol（9）。参考轨迹长度 13 m 至 4,171 m（中位数 93 m，均值 260 m）。数据集划分为 4 个子集：Proxy（10 条，Neighborhood，空间感知对比）、Ablation（5 条，Proxy 中较易子集，框架消融）、Test（30 条：17 R / 7 I / 3 M / 3 P，跨 N/F/C 三环境，主评测）、Held-out（80 条，跨全部五环境，预留供未来微调）。
- **输入与 4-DoF 连续步长控制接口（§3.1, §4.1, §D.3）：**
  - 输入为自然语言任务指令 $\mathcal{M}$、无人机 NED 坐标与偏航角状态，以及最近 3 帧 $1920 \times 1080$、水平 FOV $90^\circ$、固定俯仰角 $-45^\circ$ 的第一人称 RGB 图像（历史窗口含当前帧与前 2 帧）及过去动作历史。
  - 每步模型输出 XML 结构化字段：`<BoundingBox>`（归一化 0–1000 目标框及估计深度）、`<Reasoning>`（含用于报告类任务的 `task_gt: <value>` 或 `N/A`）与 `<Action>`（从 8 个方向原语 `move_forward` / `move_backward` / `strafe_left` / `strafe_right` / `move_up` / `move_down` / `turn_left` / `turn_right` 中选 1 个并给出连续幅度，或输出 `navigation done`）。
  - 单步最大位移 $d_{\max}$ 自动设为起点到终点头尾距离的 $1/4$，角度上限为 $3 \times d_{\max}$ 度（提示词中亦标明角度限制 $60^\circ$）；最大步数预算 $T_{\max} = 50$（Patrol）或 $20$（其余任务）。每条任务每个模型运行 3 次取平均，采样温度 $0.7$。

### 2. 评测指标与 LLM-as-a-judge 验证（§3.4, §C.5–C.6）

- **SR（Success Rate）：** Inspection 要求终态位置误差 $\|\mathbf{p}_i - \mathbf{p}^*\| < \tau_d$（$5\text{ m}$）且偏航角误差 $|\psi_i - \psi^*| < \tau_\psi$（$15^\circ$）；Reporting 使用 `Gemini-3-flash-preview` 作为 LLM-as-a-judge 对报告 $\mathcal{R}$ 做语义软匹配（在 150 条人工核验样本上 Precision 1.00、Recall 0.94，而精确字符串匹配 Recall 仅 0.44，会将 Gemini 3.1 Pro 的 SR 从 34.8% 压低至 19.1%）；Manipulation 需同时满足位姿阈值与动作触发；Patrol 对均匀重采样为 $N=100$ 段的预测轨迹与参考轨迹各做半径 $r=20\text{ m}$ 平面膨胀，要求 $\text{IoU}_{\text{patrol}} > 0.5$。
- **OSR（Oracle Success Rate）：** 任意步曾进入目标 $\tau_d = 5\text{ m}$ 范围内即计入，用于区分“未导航到目标附近”与“到达后未完成最后一步感知/对齐/报告”。
- **MP（Mission Progress）：** $\text{MP} = \max(0, \sum_{t=1}^{T}(d_{t-1}-d_t)/d_0)$；Patrol 直接取连续 $\text{IoU}_{\text{patrol}}$。
- **Eff（Step Efficiency）：** $1 - T_{\text{used}}/T_{\max}$。
- **CR（Collision Rate）：** 发生碰撞的步数占比（碰撞不终止回合）。

### 3. 核心实验发现（§4.2–§4.5, Appendix B, D）

- **22 个 MLLM 主评测（Table 3, 30 条 Test 集 × 3 次）：**
  - 最强模型为 **Gemini 3.1 Pro**（SR 34.8%、MP 73.3、OSR 17.2%、CR 4.3%、Eff 53.6%），其次为 **Gemini Robotics 1.6**（SR 32.2%、MP 70.3、OSR 17.8%、CR 0.1%、Eff 24.2%）与开源权重 **Gemma-4-31B-IT**（SR 26.7%、MP 59.3、OSR 11.1%、CR 1.6%、Eff 13.6%）。其余多数模型 SR 低于 10%（如 Qwen 3.5 27B 10.0%、Gemini 2.5 Pro 8.9%、Gemini 3 Flash 7.9%、Qwen 3.6 35B-A3B-Q8 7.9%、Claude Sonnet 4.6 4.7%、Claude Opus 4.6 3.3%、GPT-5.4 2.0%、Random 0.0%）。
  - **人类基线对比：** 人类受试者使用连续键盘控制（$N=3$）达 **84.4% SR / 94.2 MP**；强迫人类使用与 MLLM 完全相同的离散三帧图+文本接口（$N=2$）仍达 **70.0% SR / 79.0 MP**，显著高于最优 MLLM（34.8%）。
- **单帧感知代理无法解释任务成败（§4.3, Table B1）：** 在 10 条 Proxy 集上测量单帧目标框定位 mIoU 与测距 RMSE，与端到端 MP 的皮尔逊相关仅呈弱到中等水平（$r_{\text{mIoU,MP}} = 0.463$，$r_{\text{RMSE,MP}} = -0.315$）。例如 Claude Opus 4.6 测距 RMSE 最低（10.7 m），但端到端 MP 仅 18.9、SR 3.3%；而 Gemini 2.5 Pro mIoU 近零（0.01）却取得 54.2 MP 与 8.9% SR。
- **典型失败模式（§4.4, Table B4）：** GPT-5.4 失败主要由漂移与振荡导致（Drift and Oscillation 36.6%，Unmet Mission Objectives 15.9%，Altitude/Align 8.5%，Premature Termination 7.3%，Continuous Collision 7.3%）；Gemini 3 Flash 则主要死于到达附近却未完成最终目标或高度/朝向未对齐（Unmet Mission Objectives 28.9%，Altitude/Align 24.4%，Premature Termination 14.4%，Continuous Collision 8.9%，Drift 6.7%）。
- **框架消融（Table 4, Gemini 3 Flash on 5-mission Ablation split）：** 默认配置（$T=0.7$、1920×1080 FHD、3 帧历史、含 BBox 与 Reasoning）取得 57.1% SR / 68.4 MP；移除 Reasoning 导致最大跌幅（SR 降至 20.0%、MP 58.9）；移除 BBox 使 SR 降至 42.9%（MP 70.2）；历史缩为 1 帧降至 28.6% SR / 33.1 MP，扩至 5 帧反而因上下文稀释降至 30.8% SR（MP 69.6）；分辨率降至 SD（640×480）时 MP 反升至 75.3 但 SR 跌至 42.9%。
- **API 成本（Table D7）：** 完整 Test 集（30 任务 × 3 次，约 12.8M 输入 token、0.3M 输出 token）单模型费用从 Nova Pro 的 $19.2、GPT-5.4 的 $36.7、Gemini 3.1 Pro 的 $56.9 到 Claude Opus 4.6 的 $71.9 不等。
