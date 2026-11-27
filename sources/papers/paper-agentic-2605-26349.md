# Closing the Loop in Teleoperation: Episode-Level Data Quality Assessment and Feedback for High-Quality Demonstration Collection

- **作者：** Gokul Narayanan*, Yash Shahapurkar*, Melih Erdogan, Brian Zhu, Eugen Solowjow（* Equal contribution）
- **机构：** Siemens Corporation
- **版本 / 载体：** arXiv:2605.26349v1 [cs.RO]（Submitted on 25 May 2026），arXiv HTML 全文
- **原文：** [arXiv HTML v1 全文](https://arxiv.org/html/2605.26349) / [arXiv 摘要页](https://arxiv.org/abs/2605.26349)
- **项目页 / 代码：** 未确认开源代码（截至 2026-11-27，arXiv 原文及项目补查未提供独立项目页或 DQAF 框架官方代码仓库；文中提及的 `unitreerobotics/xr_teleoperate`、`arpitg1304/forge` 与 `UT-Austin-RobIn/datamodels4imitation` 分别为所依赖的遥操作栈、遥测指标库及引用工作 DataMIL 网页仓库）
- **核对范围：** 通读全文 §I–§VII（含 §III-A/B 系统需求与假设、§IV-A–F 四阶段方法与公式 (1)–(18)、§V-A–D 硬件与双实验设计、§VI-A–C 诊断验证与先导用户研究、Table I、Fig. 1–4 题注及 References [1]–[29]）
- **通读：** 已完成

## 原文核对笔记

- **研究动机（§I–II）：** 工业物理 AI（VLA 等）对真机遥操作演示数据需求激增，但新手操作员即使完成任务（nominally successful），也常因动作抖动、反复回退修正、停顿、夹爪频繁开合或贴近关节极限而产出不利于下游策略学习的低质轨迹。现有手段多停留在事后数据清洗（如 SCIZOR、CUPID、DataMIL、`score_lerobot_episodes`）或仅给出成败二值提示，造成“采集后再筛查”的高返工成本。DQAF（Data Quality Assessment and Feedback）将遥操作视为闭环采集过程，在每条 episode 结束后立即给出融合语义进度与机器人遥测的评分、证据轨迹与可操作自然语言反馈。
- **问题建模与系统需求（§III）：** 将遥操作轨迹表示为同步多模态序列 $\tau = \{(I_t, s_t, a_t)\}_{t=1}^T$（相机观测 $I_t$、本体状态 $s_t$、操作指令 $a_t$），假设每项任务有专家示范带标注关键帧 $x^{\mathrm{ref}} = \{(I_t^\star, c_t^\star)\}_{t=1}^R$（A1）。系统输出 $\mathcal{O}(\tau) = (q(\tau), \mathcal{E}(\tau), \mathcal{F}(\tau))$（评分 $q(\tau)\in[0,10]$、时间对齐语义-遥测证据、自然语言反馈），需满足五项需求：R1 可操作反馈（指出何处、何时失败及如何改正）、R2 语义-控制耦合、R3 对瞬时扰动鲁棒、R4 兼容下游筛选排序、R5 可审计性。
- **四阶段方法（§IV, Fig. 1）：**
  1. **计划条件化、上下文感知的语义分析（§IV-B）：** 给定有序子任务计划 $\Pi = \{\pi_1, \dots, \pi_L\}$，按离散更新时刻 $\mathcal{U}=\{t_1,\dots,t_M\}$（实验中每 $2.5\,\text{s}$ 采样一次）向 VLM（实验用 Gemini Flash 1.5）输入 $\mathcal{X}_t = (x^{\mathrm{anchor}}, x_t^{\mathrm{clip}}, \Pi, x^{\mathrm{ref}})$（首帧锚点、近 5 帧片段、子任务列表与专家参考帧），输出 $y_t = (i_t, c_t, r_t, z_t)$（当前子任务编号、子任务完成度、文字理由、异常标记 $z_t\in\{0,1\}$），并换算全局进度 $p_t = \frac{100}{L}((i_t - 1) + \frac{c_t}{100}) \in [0,100]$，得到语义轨迹 $\mathcal{H}(\tau)$。
  2. **遥测质量分析（§IV-C）：** 复用 Forge [28] 与 Sakr 等 [27] 的四项指标：动作幅度饱和率 $s_{\mathrm{overall}}$、基于关节轨迹的运动平滑度 $\mathrm{LDLJ}(\tau)$、二值化夹爪抖动频率 $\rho_{\mathrm{chat}}$、以及基于自适应动作阈值 $\theta_{\mathrm{static}}$ 的停顿静止比例 $f_{\mathrm{static}}$。将各指标映射为归一化子分 $\phi_k \in [0,1]$ 后按权重 $w_k$ 加权汇总为 $q(\tau) \in [0,10]$（Eq. 11）。
  3. **分段阈值诊断与跨模态证据对齐（§IV-D–E）：** 将轨迹划分为 $J$ 个时间段 $\tau_j$（实验每段 $2.5\,\text{s}$），基于专家演示分位数标定的任务阈值 $\theta_k$ 计算超限指示 $V_{j,k}^E$ 与近阈值风险 $V_{j,k}^N$，再把段中心时刻 $\frac{t_j^{\mathrm{start}}+t_j^{\mathrm{end}}}{2}$ 对齐到最近的语义更新 $i^\star(j)$，构建可审计证据集 $\mathcal{E}(\tau)$。
  4. **受约束反馈合成（§IV-F）：** 将 $\mathcal{Y}(\tau) = (\mathcal{H}(\tau), \mathcal{E}(\tau), q(\tau), \mathcal{I}_{\mathrm{exp}})$ 输入受约束 LLM（实验用 Gemini 3 Pro），按优先级生成结构化建议 $\mathcal{F}(\tau)$，每条明确说明失败内容、时间/子任务位置与下次试验改进动作。
- **实验平台与任务（§V）：** Unitree G1 人形机器人 + 顶部 RGB 相机 + Meta Quest 手势遥操作（基于 `unitreerobotics/xr_teleoperate`）。两项真机双臂任务：Task 1（左臂摆盒、右臂抓圆柱入盒并回位）与 Task 2（右臂抓取递送、左臂接取投入垃圾箱）。
- **验证与先导结果（§VI, Table I）：**
  - **诊断验证（Task 1，100 条轨迹，专家标注 72 成功 / 28 失败）：** DQAF 检出 26 条失败（24 条真阳性、2 条假阳性、4 条假阴性），失败检测召回率 **85.7% (24/28)**。约 $50\,\text{s}$ 轨迹分 20 段时，串行后处理耗时 $43\,\text{s}$（20 次 VLM 调用均 $2.0\,\text{s}$ + 反馈生成 $3.0\,\text{s}$；遥测 $2.2\,\text{s}$ 并行隐藏）；若在遥操作期间提前并发语义分析（约 60% 在结束前完成），终末等待可降至平均 **$20\,\text{s}$**（人工专家审核均值 $15\,\text{s}$）。
  - **先导用户研究（3 名新手操作员 × 各 30 次试验，总错 $\le 3$ 判为成功，Table I 均值±SEM）：**
    - Task 1：Op1（有反馈）耗时 $41.6 \pm 2.8\,\text{s}$、评分 $8.8 \pm 0.5$、大错 $1.0 \pm 0.2$、成功率 **93.33%**；Op3（有反馈）耗时 $49.2 \pm 1.3\,\text{s}$、评分 $8.8 \pm 0.4$、大错 $1.25 \pm 0.02$、成功率 **80%**。
    - Task 2：Op3（有反馈）耗时 $51.0 \pm 2.9\,\text{s}$、评分 $9.4 \pm 0.2$、大错 $1.4 \pm 0.2$、成功率 **86.66%**；Op1（无反馈）耗时 $45.4 \pm 2.2\,\text{s}$、评分 $8.2 \pm 0.3$、大错 $2.0 \pm 0.3$、成功率 **66.66%**；Op2（无反馈）耗时 $71.8 \pm 6.7\,\text{s}$、评分 $5.6 \pm 0.6$、大错 $2.9 \pm 0.4$、成功率 **53.33%**。
- **局限与未完成边界（§VI-A, §VII）：** 语义分析受部分遮挡与模糊子任务边界影响；遥测阈值依赖专家数据手工/分位数标定；当前仅支持 episode 结束后反馈而非执行期在线干预；先导研究样本极小（仅 3 名受试者、2 项任务），且尚未直接评测采集数据对下游策略训练成功率的提升；未公布 DQAF 官方开源实现。

对 wiki 的映射：[Closing the Loop in Teleoperation: Episode-Level Data Quality Assessment and Feedback for High-Quality Demonstration Collection](../../wiki/entities/paper-agentic-2605-26349.md)
