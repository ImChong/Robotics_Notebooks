# HAM-VLN: Harnessing Hierarchical Agentic Memory for Zero-Shot Vision-and-Language Navigation

- **作者：** An Liu, Bingxi Liu, Hongyu Ding, Yixuan Jiang, Yaran Chen, Fulin Tang（通讯作者）, Cong Leng, Hong Zhang, Jian Cheng
- **机构：** 中国科学院自动化研究所（CASIA）；南方科技大学（SUSTech）；鹏城实验室（Peng Cheng Laboratory）；南京大学（NJU）；西交利物浦大学（Xi'an Jiaotong-Liverpool University）；MAICRO
- **版本 / 载体：** arXiv:2607.29600v1 [cs.RO]（Submitted on 31 Jul 2026，arXiv HTML 版）
- **原文：** <https://arxiv.org/abs/2607.29600>（全文阅读载体：<https://arxiv.org/html/2607.29600>）
- **项目页 / 代码：** 未确认开源代码（arXiv v1 原文及 HTML 版本未提供独立项目主页或 GitHub 代码仓库链接；预检索缓存 `github_urls` 与 `candidate_repos` 均为空）
- **核对范围：** 逐节通读 arXiv:2607.29600v1 HTML 全文，包括 Abstract、§1 Introduction、§2 Related Work、§3 Method（§3.1–§3.4、公式 1–5、Algorithm 1）、§4 Experiments（§4.1–§4.5、Table 1–4、Figure 1–4 图注）及 §5 Conclusion 与 References。论文所涉基座模型（Gemini-3.1-Pro、Qwen3.6-35B、bge-large-en-v1.5）与评估基准仅依据论文文本核对，未进行本地仿真复现。
- **通读：** 已完成

## 原文核对笔记

1. **核心定位与动机（§1、§2）：** 针对零样本视觉语言导航（VLN）中多模态大模型（MLLM）的长程记忆瓶颈，指出两条既有路线各有缺陷——稠密语义地图随探索面积增长且与决策解耦，原始视觉历史随步数线性膨胀且无法显式标记死胡同。HAM-VLN 将决策与记忆写入耦合在同一次 MLLM 调用中，维护一个基于深度的稀疏世界图与分层智能体记忆。
2. **双系统架构（§3.1，公式 1）：**
   - **System 2 ($\varphi_2$)：** 慢速推理规划器（实验中为 `Gemini-3.1-Pro`），在每个路点调用一次，读取导航指令 $\mathcal{T}$、当前全景观测 $O_t$ 与记忆上下文 $M_t$，输出结构化 JSON 记录 $J_t=(a_t, w_t, r_t)$，包含离散高层决策 $a_t$（前进方向、回溯目标 $b_t$ 或停止）、写入状态 $w_t$（房间类型、可见物体、指令进度更新）与动作理由 $r_t$。
   - **System 1 ($\varphi_1$ + $\pi_{\mathrm{robot}}$)：** 快速底层执行策略（实验中 $\varphi_1$ 为 `Qwen3.6-35B`），不依赖预训练路点预测器，直接基于目标语义描述在选定视角预测目标边界框，取框内代表像素结合深度图 $D_t$、相机内参 $\mathbf{K}_{\mathrm{cam}}$ 与里程计位姿 $P_t$ 反投影至三维世界坐标，再由基于局部可通行性地图的快速行进规划器（fast-marching planner）执行短程轨迹。
3. **分层智能体记忆与世界图（§3.2，公式 2–4）：**
   - 上下文构造：$M_t = W_t \oplus L_t \oplus \mathcal{R}(G_t, q_t)$。工作记忆 $W_t$ 保留最近 $K$ 个已完成路点转移的原始帧（全实验统一设为 $K=1$），$L_t$ 为有序指令进度表（首个待完成项即当前子目标 $q_t$），$G_t = (\mathcal{V}_t^P, \mathcal{V}_t^O, \mathcal{E}_t^{PP}, \mathcal{E}_t^{PO})$ 为增量单链聚类形成的地点-物体图（使用 Grounding DINO + SAM 提取开放词汇目标并经深度投影融合）。
   - 四个功能视图：Working Memory（$W_t$ 近期帧）、Episodic Memory（访问轨迹与时间近因性）、Semantic Memory（地点-物体语义内容与拓扑相关性）、Reflection Memory（附着在已放弃分支地点上的失败反思笔记，仅随地点被检索时原样注入上下文）。
4. **子目标条件检索与反思回溯（§3.3–§3.4，公式 5、Algorithm 1）：**
   - 本地句子编码器 `bge-large-en-v1.5` 对 $q_t$ 与地点摘要编码（零 API 额外调用），按 $s_t(p)=\alpha\,\mathrm{rel}(q_t,p)+\beta\,\mathrm{rec}_t(p)+\gamma\,\mathrm{sal}(p)$ 打分，其中 $\mathrm{rec}_t(p)=\rho^{t-\tau(p)}$，实验固定 $(\alpha,\beta,\gamma)=(1.0,0.3,0.3)$、衰减 $\rho=0.85$。先取前 $K_s=3$ 个种子节点，经 1 跳邻域扩展并加入当前地点邻域，最终裁剪至至多 $K_p=5$ 个地点序列化为上下文。
   - 回溯 $a_t=\textsc{Backtrack}(b_t)$ 时，将被放弃后缀 $\mathcal{B}_t$ 映射为地点集合 $\mathcal{F}_t$ 并写入失败反思 $r_t$；回溯控制锚点保留在路点层以保证可执行的几何精度，而失败证据挂载在地点层以支持跨次访问复用；该证据为建议性提示而非硬性屏蔽。
5. **实验与消融（§4，Table 1–4）：**
   - 评测基于 R2R-CE、RxR-CE 与 HM3D-v2 ObjectNav 各 100-episode `val-unseen` 子集（遵循 Uni-LaViRA 协议，报告 3 个随机种子均值±标准差）。
   - 主结果：R2R-CE 达 NE `3.92 ± 0.15` m、OSR `78.7 ± 3.1`%、SR `61.0 ± 1.7`%、SPL `48.1 ± 0.2`%；RxR-CE 达 NE `6.27 ± 0.23` m、SR `52.7 ± 1.5`%、SPL `35.7 ± 2.6`%、nDTW `54.3 ± 0.3`%；HM3D-v2 ObjectNav 达 SR `79.7 ± 0.5`%、SPL `43.2 ± 0.6`%。
   - 上下文效率（Table 3）：每步决策 System 2 上下文 `17.9k` tokens，每 episode API 总量 `244.9k` tokens，相比全量原始历史（`57.7k` / `746.1k`）分别降低 `69.0%` 与 `67.2%`，且 SR 高出 7.7 个百分点。
   - 消融与回溯统计（Table 4、§4.5）：移除世界图（保留 $K=1$）使 SR 降至 `51.7`；分别移除 EM / SM / RM 使 SR 降至 `53.3` / `53.3` / `55.7`（移除 RM 使 SPL 从 `48.1` 暴跌至 `36.9`，说明失败反思主要提升路径效率）；同时禁用三视图使 SR 降至 `50.7`、SPL 降至 `31.5`。在 100 个 R2R-CE 回合中共执行 175 次回溯（占 1,288 次路点决策的 13.6%，平均每回合 1.75 次）。

- 对 wiki 的映射：[HAM-VLN 实体页](../../wiki/entities/paper-agentic-2607-29600.md)
