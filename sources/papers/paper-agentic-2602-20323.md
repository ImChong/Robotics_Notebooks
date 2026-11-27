# PhysMem: Scaling Test-Time Memory for Embodied Physical Reasoning

- **作者：** Haoyang Li, Yang You, Hao Su, Leonidas Guibas
- **机构：** Stanford University（斯坦福大学）；UC San Diego（加州大学圣地亚哥分校）
- **版本 / 载体：** arXiv:2602.20323v6 [cs.RO]（2026-02-23 提交 v1，2026-05-03 修订 v6）
- **原文：** https://arxiv.org/abs/2602.20323 （HTML: https://arxiv.org/html/2602.20323 ；PDF: https://arxiv.org/pdf/2602.20323 ）
- **项目页 / 代码：** 已开源（官方 GitHub 仓库 https://github.com/haoyangli16/PhysMem ；官方项目主页 https://phys-mem.github.io/ ；README 声明 MIT License，仓内未单列 `LICENSE` 文件；注：论文附录 F/H/I 明确说明三项真机任务的完整提示词规范细节在发布版论文中已省略）
- **核对范围：** 全文通读 arXiv:2602.20323v6 HTML 正文第 I–VII 节（Abstract、Introduction、Related Work、Method III-A–III-E、Experimental Setup IV-A–IV-B、Experiments V-A–V-G、Discussion & Limitation、Conclusion）、表 I–IV、图 1–10，以及全部附录 A–I（A-A–A-G 实现细节与被证伪假设案例、B-A–B-C 完整消融及表 V–VI、C-A–C-C 敏感性与失败模式/运行开销分析、D 数量分析及表 VII、E 仿真细节、F 真机细节与 OOD 设置、G 跨模型表 VIII、H–I 提示词模板）；同时核对 `haoyangli16/PhysMem` 仓库 README 与文件树。
- **通读：** 已完成

## 原文核对笔记

- **核心问题（声明式物理知识与物理接地鸿沟）：** 冻结参数的 VLM 规划器虽然具备抽象物理常识，却难以仅凭视觉外观预判具体场景下的隐含物理参数（如球在不同桌面上的滚动阻尼/弹性、不规则部件之间的 3D 互锁间隙、或堆叠石块的表面摩擦与重心分布）。直接复用 RAG / 原始经验检索会把历史特例固化为过时规则，一旦物理条件微调便反复出错。
- **核心机制（Scientific Memory Loop + 三层记忆架构）：**
  - 将高层规划抽象为 Options 框架 $\omega_t = \pi_\theta^H(o_t, \tau, \mathcal{P}_t)$，VLM 参数 $\theta$ 全程冻结，低层控制器 $\pi^L$ 独立执行动作。
  - **三层记忆：** 情景记忆（Episodic Memory，存储带符号状态 `symbolic_state` 的原始交互经验，上限 $N_{\max}=3000$）、工作记忆（Working Memory，存放待检验候选假设）、长期记忆（Long-term Memory，存放经验证升格的原则，按 Ebbinghaus 遗忘曲线 $\gamma=0.995$ 衰减）。
  - **共鸣门控（Resonance Checking）：** 计算经验与当前激活原则的一致性 $\rho(e, \mathcal{P}_{\text{active}})$；仅当 $\rho < 1$（出现预期违背 / Surprise）时触发新假设巩固，$\rho = 1$ 时仅强化已有原则。
  - **聚类与假设生成：** 按符号状态文本嵌入进行凝聚式层次聚类（阈值 $\tau=0.6$ 或 $\tau_c=0.7$，最小簇大小 $n_{\min}=2$），由反思模型 $f_\phi$（真机用异步 Qwen3-VL，每 15s 运行一次）生成类型化假设（`Avoid`、`Prefer`、`Sequence` 等）。
  - **动作级归因与证伪/升格：** 仅使用对应动作 $a^*$ 的经验更新假设置信度；满足 $\text{conf}(h)\ge \tau_p=0.8$、支持回合 $\ge 3$ 且准确率 $\ge 85\%$ 时升格为长期原则，并执行**记忆折叠（Memory Folding）**压缩已覆盖的原始经验（TTL=100 回合）；若 $\text{conf}(h)\le \tau_r=0.3$ 且反例 $\ge 2$ 则证伪剔除（全任务共证伪 87/281 ≈ 31% 的候选假设）。
- **核心实验数据：**
  - **Brick Insertion 仿真对比（Table II, Gemini-3-Flash, 100 回合×3 种子）：** Easy / Med. / Hard 成功率：No Memory 为 83%/53%/28%（0.40× tokens），Direct Retrieval 跌至 48%/23%/8%（0.55×），MemER-style Retrieval 为 85%/53%/19%（1.0×），MemoryVLA-style Token Memory 为 69%/35%/13%（2.5×），Reflexion-style 为 86%/61%/31%（1.2×），**PhysMem (Full) 达 89%/76%/39%（1.0×）**。
  - **真机三项任务（xArm6 + 鳍状软爪 + RealSense D435, 30 分钟+ 部署）：** Parts Organization 分数从 `-1` 提升至 `9.7`（无记忆接近 `0`）；Ball Navigation 达 `14.7` vs `0.7`；Balanced Stacking 在 50% 经验即可接近 100% 表现。共鸣度 $\rho$ 在 10 回合内由 $\approx 0.2$ 升至 $\approx 0.9$，高共鸣回合（$\rho>0.7$）得分是前 3 回合的 `2.3×`。
  - **OOD 迁移（Table III, 10 次试验）：** Parts Organization 完整版 `8.3`（4/10）；Balanced Stacking 仅先验即达 `9.2`（8/10），完整版 `12.3`（9/10）；Ball Navigation 换用网球与月球纹理塑料球后，仅先验失效（`2.9`, 1/10），加入测试时适应后完整版提升至 `7.1`（4/10 = 40%）。
  - **跨 VLM 缩放（Table VIII, Medium）：** Gemini-3-Flash 53%→76%（+23%）、GPT-5.1 43%→57%（+14%）、Qwen3-VL-235B 38%→50%（+12%）、Gemini-ER-1.5 29%→34%（+5%）。
  - **消融与开销（Tables IV/VI & App. C）：** 去掉共鸣过滤在 Medium/Hard 均降 18 个百分点；去掉验证降 12 个百分点；去掉遗忘在 Hard 降 3% 且 token 飙升至 4.8×；去掉记忆折叠 token 增至 2.1×–2.8×。单步检索开销约 180 ms，异步巩固每 15s 约 1.4 s，总墙钟时间开销约占基座 VLM 推理的 8%。
- **已知局限（Sec. VI & App. C-B）：** 当前聚焦离散高层规划，未接入连续底层力/轨迹表征或端到端 VLA 控制；观测依赖视觉与二值/分步结果反馈，缺乏触觉与声学感知；真机连续部署仍需人工干预重置掉落物体；中等难度 24% 失败案例主要来自过度泛化（40%）、延迟纠错（35%，滞后 2–4 回合）与规划器未遵循原则（25%）。
- **对 wiki 的映射：** [PhysMem 实体页](../../wiki/entities/paper-agentic-2602-20323.md)
