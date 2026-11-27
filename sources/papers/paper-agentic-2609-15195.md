# HarnessVLN: Unifying Training-Free Embodied Navigation through an Agent Harness

- **作者：** Yang Chen, Lirong Che, Zhenyu Huang, Wenbo Fu, Chuang Wang, Xu Cao, Daqi Liu, Yuzhe Yang, Jian Su, Lan-Zhe Guo
- **机构：** 南京大学（Nanjing University）、智元机器人（AGIBOT）、清华大学（Tsinghua University）
- **版本 / 载体：** arXiv:2609.15195v3 [cs.RO]（2026-09-14 提交 v1，2026-09-29 修订 v3）；arXiv HTML 全文 + 官方项目页
- **原文：** https://arxiv.org/html/2609.15195 （摘要页：https://arxiv.org/abs/2609.15195 ，项目页：https://agibot-harnessvln.netlify.app/ ）
- **项目页 / 代码：** **已开源** — 官方项目页 https://agibot-harnessvln.netlify.app/ （可访问）；官方 GitHub 仓库 https://github.com/AgibotGeneral/harnessvln （含仿真评估脚本、`vlnce_baselines/harness/` 核心运行时、多进程评估与轨迹可视化工具，基于 `files_sample` 与 `readme_preview` 静态核对）
- **核对范围：** 通读 arXiv HTML 全文（Abstract、§1 Introduction、§2 Related Work、§3 Method、§4 Experiments、§5 Real-World Deployment、§6 Conclusion、References、Appendix 1–3，含 Table 1–6、Algorithm 1 与 Figure 1–11 图注）、官方项目页文本及官方仓库 `README.md` 与目录文件列表。
- **通读：** 已完成

## 原文核对笔记

对 wiki 的映射：[HarnessVLN](../../wiki/entities/paper-agentic-2609-15195.md)

### 1. 研究动机与系统抽象

- **核心痛点：** 现有免训练 MLLM 导航方法多让模型作为独立规划器输出子目标，缺乏统一机制来联合维护空间证据、任务进度与执行反馈；模型识别出目标不等于到达目标，发出动作也不代表子目标完成。
- **统一状态定义（Eq. 1）：** Harness 在决策步 $t$ 维护共享状态 $\mathcal{S}_t = (x, \omega_t, p_t, z_t, \mathcal{M}_t, \mathcal{G}_t)$，其中 $x$ 为任务规格（路线指令或目标类别），$\omega_t$ 为当前位姿对齐的 RGB-D 观测与局部几何地图，$p_t$ 为智能体位姿，$z_t$ 为任务进度，$\mathcal{M}_t$ 为分层事件记忆，$\mathcal{G}_t$ 为持久化时空图（Spatiotemporal Graph, ST Graph）。
- **统一工具接口（Table 1）：** 通过 8 个工具解耦感知、检索、定位、执行、恢复与终止：`observe`、`observe_panorama`、`retrieve_memory`、`ground_target`、`query_depth`、`navigate_to`、`backtrack`、`request_stop`。Harness 在派发前校验参数、证据来源与时效性、几何可达性及子目标一致性。

### 2. 分层事件记忆与持久化时空图

- **分层事件记忆 $\mathcal{M}_t$：**
  - *Working memory*：保存当前观测、有界全景历史、当前目标假设与近期工具反馈，防止上下文随轨迹无限增长。
  - *Progress memory*：维护带唯一 `todo_id` 的子目标（状态为 `pending` / `active` / `completed` / `blocked`，同一时刻最多 1 个 `active`），完成须有位姿对齐观测或工具成功返回支撑。
  - *Reflection memory*：完整记录不可达目标、定位不一致、碰撞、无进展、回溯失败及被拒绝的停止请求等事件（含位置、时间戳、工具反馈与观测）。
- **时空图 $\mathcal{G}_t = (\mathcal{V}^P_t \cup \mathcal{V}^E_t, \mathcal{E}_t, \mathcal{T}_t)$：**
  - 位置节点 $\mathcal{V}^P_t$（合并半径 $0.75\text{ m}$）通过双向 `NavigableTo` 边连接；实体节点 $\mathcal{V}^E_t$（合并半径 $1.00\text{ m}$）通过 `ObservedFrom` 与 `Contains` 边保留观测视点与时空戳。
  - 任务条件检索公式（Eq. 3）：$s(v_i \mid g_t) = R(v_i, g_t) + \lambda_{\mathrm{rec}} C(v_i) + \lambda_{\mathrm{sal}} A(v_i) - \lambda_{\mathrm{fail}} P(v_i, g_t)$，其中 $\lambda_{\mathrm{rec}}=0.3$、$\lambda_{\mathrm{sal}}=0.3$、单次失败惩罚 $0.5$（上限 $1.0$）；使用局部 IDF 加权余弦相似度检索前 3 个种子位置节点并做 1 跳扩展，最多返回 5 个位置节点。

### 3. 接地执行与证据化终止

- **导航执行器：** 平面运动采用 FMM 规划器，涉及跨层楼梯通行时调用 NavDP；视觉定位采用 GroundingDINO + SAM（或由 MLLM 输出 `[0,1000]` 归一化包围框）。
- **证据化终止（Eq. 4）：** MLLM 只能发出 `request_stop` 请求，由 Harness 校验 $F_{\mathrm{stop}} = F_{\mathrm{semantic}} \land F_{\mathrm{geometric}} \land F_{\mathrm{progress}}$。ObjectNav 几何检查优先使用目标框锚点 $5\times 5$ 深度块中位数（标准差 $\le 0.5\text{ m}$ 且深度 $\le 2.5\text{ m}$），若深度无效则回退到投影目标点距离（$\le 1.0\text{ m}$）。

### 4. 关键定量结果（arXiv v3 正文与附录）

- **主评测（Table 2 & Table 3）：**
  - VLN-CE R2R val-unseen（全量 1,839 episodes，GPT-5.5）：NE $4.08\text{ m}$、OSR $70.7\%$、SR $59.6\%$、SPL $43.6\%$；R2R Rand100：NE $4.24\text{ m}$、OSR $77.0\%$、SR $58.0\%$、SPL $35.8\%$（同用 GPT-5.5 的 AgenticNav 为 SR $55.0\%$、OSR $65.0\%$、SPL $48.4\%$）。
  - VLN-CE RxR val-unseen（全量英文指令，GPT-5.5）：NE $6.50\text{ m}$、SR $51.4\%$、SPL $38.1\%$、nDTW $55.6\%$。
  - HM3D-v2（1,000 episodes，GPT-5.6-luna）：SR $76.0\%$、SPL $37.9\%$。
  - HM3D-OVON（3,000 episodes，GPT-5.6-luna）：SR $59.3\%$、SPL $36.6\%$。
- **累积消融（Table 4，100-episode 子集）：**
  - R2R Rand100：基线 SR $43.0\%$ $\rightarrow$ +Mem $50.0\%$（$+7.0$）$\rightarrow$ +Graph $56.0\%$（$+6.0$）$\rightarrow$ +Stop $58.0\%$（$+2.0$），累计 $+15.0\%$；SPL $24.3\% \rightarrow 35.9\%$，OSR $65.0\% \rightarrow 77.0\%$。
  - HM3D-OVON 子集：基线 SR $45.0\%$ $\rightarrow$ +Mem $52.0\%$（$+7.0$）$\rightarrow$ +Graph $53.0\%$（$+1.0$）$\rightarrow$ +Stop $55.0\%$（$+2.0$），累计 $+10.0\%$；SPL $32.0\% \rightarrow 33.0\%$，OSR $72.0\% \rightarrow 74.0\%$。
- **模型敏感性与同模型对比（Table 6，100-episode 子集）：**
  - HM3D-v2：GPT-5.6-luna $76.0\%$（MSGNav 同模型 $62.0\%$）、GPT-6-astra $86.0\%$、Qwen3.8-flash $72.0\%$。
  - HM3D-OVON：GPT-5.6-luna $55.0\%$（MSGNav 同模型 $37.0\%$）、GPT-6-astra $63.0\%$、Qwen3.8-flash $51.0\%$。
- **步数预算与观测旋转开销（Appendix 1.3 & Fig. 4）：** R2R Rand100 在 1,000 步上限下，观测旋转占全部 $36,291$ 次动作中的 $19,907$ 次（$54.85\%$，平均每回合 $199.07$ 步）；从 500 步放宽至 1,000 步后，SR 从 $58.0\%$ 升至 $61.0\%$，SPL 从 $35.85\%$ 微降至 $35.63\%$。
- **版本数字差异说明：** 官方项目页（含 9 月 9 日 demo 集说明）展示的 R2R / RxR SR 为 $60.8\%$ / $53.9\%$（消融为 $46.0 \rightarrow 54.0 \rightarrow 60.0 \rightarrow 64.0\%$），而 arXiv v3（9 月 29 日）报告为 $59.6\%$ / $51.4\%$（消融 $43.0 \rightarrow 50.0 \rightarrow 56.0 \rightarrow 58.0\%$）；知识库采信论文 v3 主文并明确记录该版本差异。

### 5. 真机部署（§5）

- 平台为 **AGIBOT A3 Ultra** 全尺寸人形机器人（身高 $1.74\text{ m}$，3D LiDAR、多路 RGB-D 与鱼眼相机、NVIDIA Thor 机载算力），主视觉采用头部双目相机配合 **Fast-FoundationStereo** 估计稠密度量深度，机载/调用规划模型为 **Qwen-3.8-27B**，验证了指令跟随（806 室走廊转弯至垃圾桶/饮水机再到冰箱）、开放词汇寻物（打印机）以及「指令跟随 + 寻物」组合任务（入口标识牌处停止后再寻找自动售货机）。
