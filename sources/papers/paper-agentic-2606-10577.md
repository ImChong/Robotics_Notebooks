# AgenticNav: Zero-Shot Vision-and-Language Navigation as a Tool-Calling Harness

- **作者：** Yijian Li, Changze Li, Han Zheng, Jiyuan Cai, Tong Qin, Ming Yang
- **机构：** Shanghai Jiao Tong University；Yinwang Intelligent Technology Co., Ltd.
- **版本 / 载体：** arXiv:2606.10577v3 [cs.RO]（2026-06-09 提交 v1，2026-10-02 修订 v3；CC BY 4.0）
- **原文：** https://arxiv.org/abs/2606.10577 、https://arxiv.org/html/2606.10577
- **项目页 / 代码：** 未确认开源代码（截至 2026-11-27，arXiv v3 全文、HTML 版本与元数据补查均未提供官方项目页或代码仓库链接）
- **核对范围：** 通读 arXiv HTML 全文（Abstract、§I Introduction、§II Related Works、§III Method、§IV Experiments、§V Failure Analysis、§VI Conclusion，以及 Table I–V 与 Fig. 1–8 图注）
- **通读：** 已完成

## 原文核对笔记

- 对 wiki 的映射：[AgenticNav: Zero-Shot Vision-and-Language Navigation as a Tool-Calling Harness](../../wiki/entities/paper-agentic-2606-10577.md)

### 研究动机与核心接口设计

1. **三类既有零样本 VLN-CE 瓶颈**：
   - **路径点预测器限制动作空间**：如 Open-Nav、SmartWay、EvoNav 依赖额外训练的航点预测网络从全景 RGB-D 中给出少量候选点，指令相关的可见目标若不在候选集中便无法抵达。
   - **深度信息未有效显式暴露给 VLM**：航点预测器在内部消耗深度，或直接向 VLM 输入稠密深度图，导致 VLM 难以对特定位置获取精确米制距离。
   - **上下文与记忆设计失衡**：持续堆叠历史图像/文本会引入大量无关冗余；跨回合经验检索（如 EvoNav）则削弱严格单回合零样本假设。

2. **工具调用接口（§III-A, Eq. 1–2）**：
   - 环境观测 $O_t = \{(I_t^k, D_t^k)\}_{k=0}^{N-1}$，但对齐深度图 $D_t^k$ 仅保留在工具层，不直接输入 VLM。
   - VLM 每步上下文 $X_t = (L, \mathcal{I}_t, B_t, H_t)$（指令、当前多视角 RGB、紧凑 BEV 地图、近 6 步推理与动作文本历史）。
   - 暴露 4 个可调用工具：$\mathcal{T} = \{\texttt{query\_depth}, \texttt{recall}, \texttt{move\_to}, \texttt{stop}\}$。前两者为非终结取证工具，返回结果进入本步下一轮 VLM 推理；后两者（且 `move_to` 通过安全检查）为本步终结动作。

3. **三大工具机制（§III-B–D, Eq. 3–9）**：
   - **Depth Tool（`query_depth`）**：VLM 提交归一化像素坐标批次 $p_i=(k_i, u_i, v_i)$（$u_i, v_i \in [0, 1]$），工具返回米制深度 $d_i$ 及 $5\times 5$ 局部邻域统计量 $s_i^{5\times 5}$（min/mean/median），帮助判断目标是否在稳定表面或深度边界。
   - **Action Tool（`move_to(k, u, v, reason)`）**：反投影像素 $p_t = T_k(d K^{-1}[x,y,1]^\top)$，计算地平面方位角 $\theta$ 与经停靠余量 $m$ 裁剪并限幅的距离 $\rho = \min(\rho_{\max}, \max(0, \|p_{t,xz}\| - m))$。再将机体高度范围 $[y_{\min}, y_{\max}]$ 内的点云投影到地平面障碍物集合 $\mathcal{O}_t$，检查以机器人半径 $r_a$ 为半宽、长度 $\rho + r_a$ 的矩形区域 $\mathcal{R}(\theta, \rho)$ 是否无碰撞；若不安全则返回 `Reselect` 要求 VLM 重选，安全则执行 `Execute(\theta, \rho)`。
   - **Agentic Memory & Recall Tool**：回合内记忆 $M_t = (H_t, B_t, C_t)$。$H_t$ 保留最近 6 步 `(reason, action)`；$B_t$ 由底层 SLAM 模块提供障碍与可通行信息，并标注历史决策节点编号；`recall(p, k)` 从有界缓存 $C_t$ 取回指定节点 $p$ 视角 $k$ 的历史 RGB 图像并临时追加到当前提示词，新决策步重建上下文时不永久堆积。

### 关键实验与消融记录（§IV–V）

- **仿真评测（Habitat / R2R-CE `val_unseen` 100 episodes，Table I–II）**：
  - 同基座对比（Gemini-3.7-Flash）：AgenticNav 达到 **76% SR / 66.50% SPL / 78% OSR / 69.38% nDTW / 3.48m NE**，对比 SmartWay*（54% SR / 47.95% SPL / 60% OSR / 65.93% nDTW / 4.68m NE）提升 +22% SR 与 +18.55% SPL，也超过有监督的 LightNav-0*（67% SR / 62.71% SPL）。
  - 同基座对比（GPT-5.5）：AgenticNav 55% SR / 48.41% SPL vs SmartWay* 44% SR / 35.04% SPL。
  - 同基座对比（Gemini-2.5-Pro）：AgenticNav 58% SR / 42.91% SPL / 76% OSR / 5.00m NE vs SmartWay* 52% SR / 41.70% SPL / 62% OSR / 5.28m NE。
  - 其余 VLM 骨干（Table II）：DeepSeek-V4-Flash Vision-Exp（49% SR / 36.48% SPL）、Qwen3.8-27B（48% SR / 33.57% SPL）。从 Gemini-2.5-Pro 到 Gemini-3.7-Flash，AgenticNav 增益 +18.00 SR / +23.59 SPL，远高于 SmartWay（+2.00 / +6.25）与 Open-Nav（+4.00 / +1.71）。
- **记忆与工具消融（Table III–IV，Gemini-3.7-Flash）**：
  - 去掉推理历史文本 $H_t$：SR 76% → 63%；用结构化任务进度文本替代推理文本：64%（错误进度判断会向后传播）；去掉 BEV 地图：70%；堆叠全部历史图像：66% SR / 54.05% SPL。
  - 用航点预测器替代 Action Tool：66% SR / 57.73% SPL；去掉 Depth Tool：60% SR / 47.14% SPL / 5.03m NE；用深度图输入替代 Depth Tool：69% SR / 55.79% SPL；去掉 Recall Tool：64% SR / 47.47% SPL。
- **真机实验（§IV-B, Table V）**：
  - 平台：Unitree Go2W 四足机器人（局部规划器 SCAN-Planner）与全向四轮平台（局部规划器 TEB），均搭载 4 台 Intel RealSense D435 RGB-D 相机 + LiDAR + FAST-LIO 定位。
  - 35 条真实场景轨迹（实验室、办公室、室外广场，含路牌理解、夜间低光照与 >200m 室内外跨区长程导航）。GPT-5.5 下：SmartWay（12 视角原地旋转）整体 22.9% SR / 3.74m NE；AgenticNav 单视角 37.1% SR / 3.16m NE；AgenticNav 4 视角 **54.3% SR / 2.51m NE**（广场长程场景从 11.1% 提升至 55.6%）。
- **失败分析（§V）**：与 SmartWay 相比，停靠相关失败率下降逾半，真机记忆相关失败率降近一半，航点预测器导致的失败完全消除；残余误差主要来自 VLM 主干自身的推理与决策局限。
