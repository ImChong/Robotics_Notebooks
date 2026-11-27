# RoboStream: Weaving Spatio-Temporal Reasoning with Memory in Vision-Language Models for Robotics

- **作者：** Yuzhi Huang, Jie Wu, Weijue Bu, Ziyi Xiong, Gaoyang Jiang, Ye Li, Kangye Ji, Shuzhao Xie, Yue Huang, Chenglei Wu, Jingyan Jiang, Zhi Wang
- **机构：** Shenzhen International Graduate School, Tsinghua University; YuanxingGuangnian Robotics; China University of Mining and Technology; Shenzhen Technology University; Huazhong University of Science and Technology; Xiamen University
- **版本 / 载体：** arXiv:2603.12939v2 [cs.RO]（2026-03-13 提交 v1，2026-07-10 修订 v2）；Accepted by ECCV 2026
- **原文：** https://arxiv.org/abs/2603.12939（全文核对使用 arXiv HTML 版 https://arxiv.org/html/2603.12939）
- **项目页 / 代码：** https://robostream123.github.io/ | https://github.com/yu2hi13/RoboStream（Apache-2.0 License，**已开源**：包含规划/感知/记忆/控制核心代码、vendored RLBench 长时域任务与 SIMPLER/ManiSkill2 仿真栈、GSNet 与 SAM/GroundingDINO 适配器及公开配置矩阵；VLM 权重/API 密钥、SAM3 权重与 GSNet 预训练权重需外部按说明下载配置）
- **核对范围：** 通读 arXiv:2603.12939v2 全文（Abstract、Sections 1–6、References、Appendices 0.A–0.D、Tables 1–12、Figures 1–18 题注），并核查官方 GitHub 仓库页面（https://github.com/yu2hi13/RoboStream）的 README 与部分文件列表。
- **通读：** 已完成

## 原文核对笔记

对 wiki 的映射：[RoboStream 实体页](../../wiki/entities/paper-agentic-2603-12939.md)

1. **核心问题与动机（Sections 1–2）**
   - 现有基于 VLM 的机器人规划器多将每一步视为孤立的观测到动作映射，每轮重新从原始像素推断三维几何，既缺乏跨步几何锚定（Ungrounded Spatial Perception），又缺乏动作引起的环境状态转移记录（Untracked Causal History）。
   - 结果导致在长时域（long-horizon）操作中出现感知误差累积、暂时被遮挡物体被完全遗忘以及前置条件违背等级联失败。
   - RoboStream 采用**免训练（training-free）**框架，将空间-时间几何锚定与因果记忆直接织入 VLM 闭环规划循环。

2. **两项核心机制与四步形式化流程（Section 3）**
   - **目标与感知（Sec. 3.2）**：输入 RGB-D 观测 $(I_t, D_t)$、目标描述 $I_{\text{goal}}$（支持目标图像或自然语言指令）及历史记忆 $\mathcal{M}_{<t}$。首先提示 VLM 结合 $\mathcal{M}_{<t}$ 生成任务相关物体的开放词表描述 $\mathcal{D} = \{d_1, \dots, d_N\}$；再由 $\mathrm{SAM3}(I_t, \mathcal{D})$ 生成像素掩码 $M_i \in \{0,1\}^{H\times W}$，结合深度图 $D_t$ 提取各物体 3D 点云 $\mathbf{P}_i$。
   - **时空融合标记 STF-Tokens（Sec. 3.3）**：$\mathcal{T}_t^{\text{STF}} = \Phi_{\text{enc}}(I_t, D_t \mid \mathcal{M}_{<t}) = \{\tau_i^t\}_{i=1}^N$。对物体 $o_i$，将掩码图像 $\hat{I}_i^t$ 送入 VLM 视觉编码器得到 $16\times 16$ patch tokens，筛选与 $M_i$ 的 IoU 超过阈值 $I_{th}$ 的 patch 并投影聚合为视觉证据 $\mathbf{v}_i^t$；同时计算点云中位数质心 $\mathbf{c}_i^t = \mathrm{median}(\mathbf{P}_i) \in \mathbb{R}^3$ 与轴向高斯+极值形状向量 $\mathbf{s}_i^t = \{(\mu_a, \sigma_a, a_{\min}, a_{\max}) \mid a \in \{x,y,z\}\}$；将 $\mathbf{c}_i^t$ 与 $\mathbf{s}_i^t$ 通过文本分词器序列化到同一语言嵌入空间，构成 $\tau_i^t = \langle \mathbf{v}_i^t, \mathbf{c}_i^t, \mathbf{s}_i^t, t \rangle$。
   - **4D 因果时空图 CSTG（Sec. 3.4）**：$\mathcal{G}_t^{\text{CST}} = \Psi_{\text{graph}}(\mathcal{T}_t^{\text{STF}}, \mathcal{G}_{t-1}^{\text{CST}}) = (\mathcal{G}_t, \mathcal{H}_t^K)$。空间场景图 $\mathcal{G}_t = (\mathcal{V}_t, \mathcal{E}_t)$ 中每个节点维护长度为 $K$ 的滑窗缓冲 $\{\tau_i^{t-K+1}, \dots, \tau_i^t\}$，边编码欧氏距离 $d_{ij} = \|\mathbf{c}_i - \mathbf{c}_j\|_2$ 与方向偏移 $\Delta \mathbf{c}_{ij} = \mathbf{c}_j - \mathbf{c}_i$；跨窗对比检测物体级事件（计划位移、意外碰撞、遮挡）与任务级事件（动作执行、子任务完成），写入带时间戳、位置与因果来源的记忆日志 $\mathcal{H}_t^K$。
   - **VLM 因果推理与解耦 6-DoF 实例化（Sec. 3.1, 3.5）**：将 $\mathcal{G}_t^{\text{CST}}$、带 Set-of-Mark 物体身份标注的当前观测及 $I_{\text{goal}}$ 组装为提示 $\mathcal{P}_t$，VLM 通过思维链（CoT）空间验证输出语义动作指令 $\hat{a}_t = \Pi_\theta(I_{\text{goal}}, \mathcal{P}_t \mid \mathcal{G}_t^{\text{CST}})$，再由 $\Phi_{\text{inst}}(\hat{a}_t, \mathcal{G}_t^{\text{CST}})$ 对齐 STF-Token 几何属性确定性解析为 6-DoF 动作 $a_t$。

3. **实验评测与核心数据（Section 4 & Appendix 0.C）**
   - 评测统一基于 Qwen3-VL-8B / 32B / 235B，全部**零样本免微调**。
   - **真机实验（Franka Research 3 + D435i，21 项任务×3 次，Sec. 4.2, Tab. 6–7）**：
     - Block Building（Easy/Med/Hard）：235B 平均 63.0%（Hard 达 44.4%，SoFar 与 VoxPoser 均 11.1%）；32B 48.1%，8B 33.3%，SoFar 22.2%，VoxPoser 25.9%。
     - Block Disassembly（Easy/Med/Hard）：235B 平均 77.8%（Hard 66.7%）；32B 63.0%，8B 40.7%，SoFar 29.6%，VoxPoser 25.9%。
     - Block Hide and Restore（全遮挡记忆恢复，Hard 3 项任务）：235B **88.9%**、32B 55.6%、8B 33.3%，而 SoFar 与 VoxPoser 均为 **0.0%**。
   - **SIMPLER 零样本跨本体（Tab. 1）**：Google Robot 上 RoboStream-8B 在 Pick Coke Can（VA/VM）达 94.2% / 95.7%、Move Near 达 79.8% / 95.8%（均优于 SoFar 与 OXE 微调的 RT-2-X）；WidowX + Bridge 上 RoboStream-8B 四任务均分 **74.8%**（Spoon 62.5%、Carrot 71.9%、Stack 77.1%、Egg 87.5%），高于零样本 SoFar（58.3%）及 Bridge 微调的 SpatialVLA（42.7%）。
   - **RLBench 长时域（8 任务×25 回合，Tab. 2）**：RoboStream-235B 平均 **90.5%**、32B **82.5%**、8B **58.0%**，显著超越 SoFar（28.0%）与 VoxPoser（26.5%）；在 4 项记忆密集型任务上（Tab. 12），零样本 RoboStream（8B/32B/235B 分别为 48.0%/80.0%/92.0%）优于需训练的记忆方法 SAM2Act（30.0%）与 MemoryVLA（26.0%）。另在 15 项短时域 RLBench（Tab. 11）上 RoboStream-8B 均分 71.2%（VoxPoser 49.9%，SoFar 23.5%）。
   - **6-DoF 空间推理（配合 PointSO，Tab. 3–4）**：6-DoF SpatialBench 总准确率 RoboStream-32B 达 **48.9%**（SoFar-32B 45.3%，GPT-4o 36.2%）；Open6DOR V2 上 RoboStream-32B 达 Pos 93.8% / Rot 58.8% / 6-DoF **52.2%**（SoFar-32B 为 93.1% / 54.6% / 48.4%）。
   - **消融与开销（Tab. 5, 8, 9, 10）**：235B 去掉 CSTG 成功率由 90.5% 骤降至 14.5%，去掉 STF-Tokens 降至 79.5%，两者全去降至 12.0%。滑窗 $K=3$ 下感知（0.25 s）与 CSTG 构建（0.16 s）步间稳定不随步数增长；全真机任务单步总延迟 8B（1×A100）为 3.86 s，32B（2×A100）8.61 s，235B MoE（8×A100）5.15 s。

4. **失败分布与局限（Section 5 & Appendix 0.C.2）**
   - 真机失败分布（Fig. 14）：VLM 任务规划错误（空间幻觉、左右/拆解顺序误判、遮挡遗忘）占 **43.5%**；物体解析与检测误差（如相似颜色干扰与堆叠遮挡下 SAM3 定位偏差）占 **20.7%**；放置精度误差占 **16.3%**；抓取不稳定占 **14.1%**；运动规划（IK 求解/工作空间越界/碰撞）占 **5.4%**。
   - 系统受限于解耦“规划-执行”范式：子模块误差仍会向底层执行传播，尚未在控制环内融合末端柔性 VLA 或端到端可微分轨迹执行。
