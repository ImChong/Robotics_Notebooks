# EmbodiRSI: Recursive Self-Improvement for Data-Efficient Robot Adaptation

- **作者：** Haoran Lang*, Haotao Lu*, Shiyu Sang*, Haoyang Luo, Guo Chen, Qun Li, Jingyi Yu, Ye Shi†, Jingya Wang†（* 同等贡献；† 通讯作者）
- **机构：** 上海科技大学（ShanghaiTech University，基于通讯作者邮箱 `wangjingya@shanghaitech.edu.cn`、`shiye@shanghaitech.edu.cn`；arXiv HTML 原文未完整导出机构角标图例）
- **版本 / 载体：** arXiv:2609.38905v4 [cs.RO]（v1 提交于 2026-09-30，v4 修订于 2026-10-29），CC BY-NC-SA 4.0
- **原文：** https://arxiv.org/abs/2609.38905 / https://arxiv.org/html/2609.38905
- **项目页 / 代码：** 宣称将开源或未列链接（Reproducibility Statement 声明“The full code for the entire pipeline will be released soon.”；截至 2026-11-27 尚未公开独立项目页与 GitHub 代码仓库）
- **核对范围：** 完整通读 arXiv:2609.38905v4 HTML 全文（Abstract、§1 Introduction、§2 Related Work、§3 Method（§3.1–§3.3、Algorithm 1）、§4 Experiments（§4.1–§4.3、Table 1–2、Figure 1–4）、§5 Limitations、§6 Conclusion、AI Use / Reproducibility Statements，以及 Appendix A–F 含 Table C.1、D.1–D.4、E.1–E.5 与全部公式）。
- **通读：** 已完成

## 原文核对笔记

- **核心定位：** 面向桌面操作策略快速自适应（adaptation）的智能体递归自提升系统（EmbodiRSI），将 Real-to-Sim-to-Real（R2S2R）闭环化：在目标部署场景构建的可复用任务仿真中，利用当前策略的执行反馈持续驱动回合内纠错与跨回合定向采数，随后仅用极少真机人工干预（HIL）轨迹完成物理微调部署。
- **支撑基础设施（§3.1–§3.2, Appendix B）：**
  1. **Agentic Scene Compiler（B.1）：** 输入桌面图像 $I_0$（可选目标图或语言指令），VLM 识别桌面与可动物体，SAM 3 分割掩码，SAM 3D 重建带纹理网格得到 $\mathcal{S}_0$；经 **AACR**（Agent-Guided Adaptive Collision Rectification，含并行水平消除碰撞 PHR 与组内关系感知微调 AIR）修正穿插；通过 `on-top-of` / `inside` / `beside` 粗关系与多视角渲染 critic 迭代细化得到目标布局 $\mathcal{S}_{\mathrm{org}}$。
  2. **Agentic Trajectory Generation（B.2）：** 对比 $\mathcal{S}_0$ 与 $\mathcal{S}_{\mathrm{org}}$ 编译有序子任务 $\mathcal{T}$；VLM 规划物体空间路径，**GraspGen** 生成 6-DoF 候选抓取并保持物体相对夹爪变换 $T_{o_i \to g}$，**cuRobo**（cuRobov2）逐段求解运动学与无碰撞路径；经四重验收 $\operatorname{Accept}(\tau) = V_{\mathrm{mesh}} \land V_{\mathrm{view}} \land V_{\mathrm{motion}} \land V_{\mathrm{exec}}$ 过滤并修复后以 LeRobot 格式落盘。
- **RSI 双反馈闭环（§3.3, Algorithm 1, Appendix A）：**
  1. **Agentic Rollout Monitor（§3.3.1）：** 结合时序多视角图像窗口、动作历史和仿真器状态 $x_t$ 输出 `continue` / `recover` / `complete`（仿真真值位姿仅用于训练监控工具，不作为策略输入）。
  2. **Collaborative Error Correction（§3.3.2）：** 回合内触发恢复时暂停仿真，利用 $T_g^{\mathrm{cur}} = T_o^{\mathrm{cur}}(T_o^0)^{-1}T_g^0$ 将标称抓取迁移至当前物体位姿，从当前关节态规划衔接轨迹并可在门控下把控制权交还策略，产出从“策略所达非标称状态”出发的纠错数据。
  3. **Adaptive Data Collection（§3.3.3, A.2）：** 跨回合将最终失败归因至源物体重置、目标锚点重置或两者皆非，在归一化网格 $[0,1]^2$ 上计算归因失败率 $\widehat{r}_{k,b}^s = U_{k,b}^s / N_{k,b}^s$（纠错后成功的回合不计入失败分子），将高失败网格映射回世界重置区间生成困难分布 $p_{\mathrm{hard},k}(c)$，按 $p_{k+1}(c) = (1-\alpha)p_0(c) + \alpha p_{\mathrm{hard},k}(c)$ 混合采样新示教。
  4. **策略更新与调度（§3.3.4, A.3）：** 采用 $\pi_{0.5}$ 预训练 VLA 初始化，配合基于同骨干的价值模型 $V_\phi$ 做 RECAP 式优势条件训练（$R_t^{(n)}=0, n=50$，文本条件 $z_t = \mathbb{I}[\widehat{A}_t^{(n)} > \delta]$）；外层由基于 OpenClaw 与 MCP 接口的监督器按有类型技能库 $\mathcal{K}$ 编排与断点修复。
- **关键实验证据（§4, Appendix C–E）：**
  - **硬件与平台：** 14-DoF 双臂 ALOHA（头相机 + 双腕相机），GPT-5.5 驱动智能体，Isaac Sim 仿真，3 个真实/仿真配对桌面（S1 Office 6 子任务、S2 Kitchen 4 子任务、S3 Household 4 子任务，共 14 子任务），每场景训练 1 个多任务策略。
  - **仿真自迭代提升（Table E.1）：** 跨 R0（50 条/子任务）→ R1（200 条）→ R2（400 条），三场景均衡自主仿真成功率由 **50.4% → 70.7% → 83.5%**。
  - **等预算消融（Table 1, E.3–E.5，S1 & S3 at 400 条）：** EmbodiRSI 相比 Static 静态采数在仿真中提升 **+20.6 pp**（79.9% vs. 59.3%），真机零样本迁移提升 **+12.9 pp**（59.2% vs. 46.3%），叠加 10 条真机 HIL 后仍领先 **+14.2 pp**（82.1% vs. 67.9%）。
  - **真机数据效率（Table 2, E.2）：** R2+10（400 仿真 + 10 条真机 HIL/子任务）达 **83.1%** 场景均衡真机自主成功率（116/140），超越使用 **200 条真机示教/子任务**的 Real-only 200 基线（**75.0%**），物理轨迹仅为其 1/20；R2 零样本真机已有 **62.8%**。
- **对 wiki 的映射：** [EmbodiRSI: Recursive Self-Improvement for Data-Efficient Robot Adaptation](../../wiki/entities/paper-agentic-2609-38905.md)
