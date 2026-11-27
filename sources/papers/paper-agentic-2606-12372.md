# UniIntervene: Agentic Intervention for Efficient Real-World Reinforcement Learning

- **作者：** Haoyuan Deng, Yitong Gao, Yudong Lin, Haichao Liu, Zhenyu Wu, Ziwei Wang
- **机构：** Nanyang Technological University (NTU), Beijing University of Posts and Telecommunications (BUPT)
- **版本 / 载体：** arXiv:2606.12372v1 [cs.RO]（2026-06-10 提交，arXiv HTML 内页日期 2026-08-24）；CoRL 2026 Accepted
- **原文：** https://arxiv.org/abs/2606.12372（全文核对使用 arXiv HTML 版 https://arxiv.org/html/2606.12372）
- **项目页 / 代码：** https://denghaoyuan123.github.io/UniIntervene-project/ | https://github.com/Denghaoyuan123/UniIntervene（MIT License，部分开源：已开源 UniIntervene 离线训练与数据验证管线，包含 NTTG 进度标注、代理价值函数训练、值标注、三规则干预挖掘、VQ1、恢复记忆构建与 VQ2 FAST 恢复策略训练及 Fold Towel 示例配置；UR7e 真机部署与 HIL-SERL 在线训练集成未包含在仓库内，未发布机器人轨迹数据、训练权重与记忆库文件）
- **核对范围：** 通读 arXiv:2606.12372v1 全文（Abstract、Sections 1–6、Appendices A–I、Tables 1–6、Algorithm 1、Figures 1–8 题注）及项目页 https://denghaoyuan123.github.io/UniIntervene-project/ 公开文本，并静态核对官方仓库 https://github.com/Denghaoyuan123/UniIntervene 的 `README.md` 与 `files_sample` 文件列表。
- **通读：** 已完成

## 原文核对笔记

对 wiki 的映射：[UniIntervene 实体页](../../wiki/entities/paper-agentic-2606-12372.md)

1. **核心问题与动机（Sections 1–2, 3.1）**
   - 真实世界人机回环强化学习（HiL-RL）虽然能通过人工纠偏改善精细操作策略，但长期受限于高频人工干预带来的标注负担。现有方法一类聚焦更好地复用人工纠偏数据（如 HG-DAgger、SERL/HIL-SERL、ConRFT、Compliant Residual DAgger、E2HiL），另一类聚焦安全或可重置探索（如 Recovery RL、Reset-Free RL、Failure-Aware RL），均未能解决真实训练中最主要的成本来源——策略虽未发生危险或不可逆失败，却长时间停滞在无进展探索（unproductive exploration）状态，迫使人类介入接管。
   - UniIntervene 将在线干预形式化为内生决策模块 $\mathcal{I}_\psi: (o_t, \ell, a_t) \mapsto (\hat{q}_t, s_t^{\mathrm{int}}, A_t^{\mathrm{rec}})$，统一联合预测当前动作引导的价值 $\hat{q}_t$、干预触发分数 $s_t^{\mathrm{int}} \in [0, 1]$ 以及纠偏动作块 $A_t^{\mathrm{rec}} = \{a_t^{\mathrm{rec}}, \dots, a_{t+H}^{\mathrm{rec}}\}$。当 $s_t^{\mathrm{int}} < \tau_{\mathrm{int}}$ 时执行原策略动作，否则覆盖执行 $A_t^{\mathrm{rec}}$ 并将恢复转移写入回放缓冲区 $\mathcal{B}$，仅将罕见、不安全或严重分布外情况留给人类操作员。

2. **离线代理价值函数 Proxy Value Function（Section 3.2 & Appendix C）**
   - 架构：冻结 SigLIP-SO400M 视觉编码器（1152 维）→ 两层 MLP 投影至 640 维语言空间 → Gemma-3-270M 融合语言指令 $\ell$ → 双头价值网络 $V_\phi(o, \ell) = \min(V_{\phi,1}, V_{\phi,2})$ 抑制单头高估。分两阶段训练：先冻结视觉和语言主干在通用图文数据上对齐投影层，再冻结视觉主干在机器人数据上解冻投影层与价值头。
   - 进度标注（NTTG）：对长度 $T$ 的成功轨迹赋予归一化剩余进度标签 $v_t = t / (T - 1) \in [0, 1]$；失败轨迹全段置 $v_t = 0$ 并配大负原始回报。
   - 训练目标：$\mathcal{L}_{\text{proxy}} = \lambda_{\text{TD}}\mathcal{L}_{\text{TD}} + \lambda_{\text{label}}\mathcal{L}_{\text{label}} + \lambda_{\text{prog}}\mathcal{L}_{\text{mono}} + \lambda_{\text{CQL}}\mathcal{L}_{\text{CQL}}$，权重 $(1.0, 0.3, 1.0, 0.05)$、折扣 $\gamma = 0.99$、前 5 轮仅开进度损失预热、成功与失败轨迹按 $0.5$ 均衡采样。在验证集上成功/失败轨迹均值显著分离（Insertion 0.53 vs 0.09；Pick Eggplant 0.53 vs 0.24，双头分歧 $< 3\times 10^{-3}$）。

3. **未来条件动作价值估计与骨干网络 VQ1（Section 3.2 & Appendix D）**
   - 骨干为 Qwen3-VL-2B-Instruct，配置 LoRA（rank=16, $\alpha=32$, dropout=0.05，作用于 `q_proj, k_proj, v_proj, o_proj`），在提示末尾附加特殊查询 token `<VQ_QUERY>`，取其末层隐状态 $h_t$ 作为共享表征。
   - **潜在后果预测（Latent Future Head）：** 两层 MLP（4 倍隐藏宽度扩展）从 $h_t$ 预测下一步池化潜在表示 $\hat{z}_{t+1}$，与冻结 V-JEPA2 编码器（`vjepa2-vitl-fpc64-256`，1024 维 × 256 tokens 池化）对真实 $o_{t+1}$ 的表征计算 L2 归一化后的 MSE 损失。
   - **双价值头与辅助头：** Twin-Q 头输入 $\hat{z}_{t+1}$ 输出 $\hat{q}_t = \min(\hat{q}_{1,t}, \hat{q}_{2,t})$，以 Smooth-$L_1$ 回归离线代理目标 $q_t^\star \equiv V_\phi(o_{t+1}, \ell)$；另有辅助当前价值头直接从 $h_t$ 回归当前步价值以稳定早期尺度。骨干学习率 $2\times 10^{-5}$，各头学习率 $1\times 10^{-4}$，AdamW，bf16 训练 30 个 epoch。

4. **时序价值风险干预触发器 TVR（Section 3.3 & Appendix E）**
   - 单帧低价值在接触、对齐、重抓阶段很常见，因此引入窗口长度 $K=8$、期望进步率 $\epsilon=0.005$、趋势折扣 $\gamma_r=0.9$ 的时序价值风险：
     $$R_t = (1 - V_t) \sum_{i=0}^{K-1} \gamma_r^i (\epsilon - \Delta V_{t-i}), \qquad \Delta V_i = V_i - V_{i-1}$$
     $(1 - V_t)$ 在接近完成时衰减风险，使 $R_t$ 仅对持续停滞或退化响应。
   - **离线三规则干预点挖掘（Mining）：** 在单集 min-max 归一化后的代理价值曲线上以长度 8 滑窗标记三类模式——持续下降（decline）、平台停滞（plateau）、骤降未恢复（abrupt-drop）。
   - **风险与触发头：** 值历史编码器将近期价值序列及其一阶差分映射为 32 维向量，与 $\hat{z}_{t+1}$ 一起送入 $f_{\text{risk}}$ 回归 $\hat{R}_t$；触发头读入池化潜在特征与已梯度截断（detached）的价值标量，经 $\sigma(\hat{R}_t)$ 输出 $s_t^{\text{int}}$（阈值 $\tau_{\text{int}}=0.5$），以二元 Focal Loss（$\alpha=0.75, \gamma=2.0$）拟合挖掘标签。

5. **记忆引导的目标条件恢复 VQ2（Section 3.4 & Appendix F）**
   - **恢复记忆构建：** 从先验轨迹中挖掘固定跨度（前四任务 `span=8`、`delta >= 0.40`、10×12=120 条；Fold Towel 表 5 为 `span=16`、`delta >= 0.50`、20×12=240 条；代码仓库 `configs/fold_towel.json` 示例采用 `span=32`、`delta=0.40`）的“值提升片段”，按进度分桶定额采样，配对 $(o_j^{\mathrm{fail}}, s_j^{\mathrm{fail}}, \ell_j, o_j^g, s_j^g)$。
   - **目标检索与 FAST 解码：** 在共享嵌入空间 $\phi(\cdot)$ 中按余弦相似度检索最匹配过去失败上下文的高价值目标 $g_t$（表 6 审计：Top-1 相似度 0.957–0.993，目标价值提升 0.339–0.563）。随后条件恢复策略 $\pi_{\mathrm{rec}}$ 通过共享残差 LoRA（rank=16, $\alpha=32$）复用骨干表征，用 `physical-intelligence/fast` 分词器将 $H=8$ 步、7 维末端执行器动作块转为离散频域 token，以逐 token 交叉熵行为克隆训练。

6. **真机平台、实验与消融结论（Section 4 & Appendices B, H, I）**
   - **真机平台：** UR7e 机械臂 + 平行夹爪，腕部相机 + 第三人称固定相机（均缩放至 $224\times 224$ RGB），3D-connexion SpaceMouse 人工干预接口。
   - **5 项真机任务（Table 1）：** Pick Eggplant（多物体抓取）、Tube Insertion、RAM Insertion（亚厘米与视觉微弱变化的富接触插接）、Wipe Whiteboard（恒定接触力擦拭）、Fold Towel（柔性长程折叠）。每任务评测 20 回合成功率和 3 次独立训练运行的人工干预率（IR）。
   - **核心结果：** `HiL-SERL + UniIntervene` 平均成功率 **88%**（相对 HiL-SERL 的 81% 绝对提升 7 个百分点 / 相对提升 **8.6%**）、平均人工干预率降至 **14.6%**（相对 HiL-SERL 的 34.3% 降低 **57%**），在所有 5 个任务上均取得最高成功率与最低人工干预率，且优于 $\pi_{0.5}$ (SFT)（54% SR）与 `HiL-SERL + FA-RL`（77% SR, 24.6% IR）。
   - **消融实验（Table 2，Pick Eggplant 与 RAM Insertion）：** 完整版 95% SR / 11.1% IR / 0.882 Int. F1。去价值预测 F1 降至 0.845、IR 升至 18.7%、SR 降至 85%；去 TVR 导致 F1 最低（0.832）、IR 16.9%、SR 85%；去未来预测（SR 90%, IR 15.8%）和去记忆目标（SR 85%, IR 16.1%）几乎不降离线触发 F1（0.878 / 0.882），但明显损害在线恢复成功率。
   - **局限（Section 6）：** 依赖离线代理价值函数校准质量；对记忆库未覆盖的失败模式恢复可靠性下降；实验局限于单一桌面机械臂平台；未完全消除罕见危险/分布外工况对人类监督的依赖。
