# Critic in the Loop: A Tri-System VLA Framework for Robust Long-Horizon Manipulation

- **作者：** Pengfei Yi, Yingjie Ma, Wenjiang Xu, Yanan Hao, Shuai Gan, Wanting Li*, Shanlin Zhong*（* 共同通信作者）
- **机构：** 中国科学院自动化研究所（Institute of Automation, Chinese Academy of Sciences）、中国科学院大学人工智能学院（School of Artificial Intelligence, University of Chinese Academy of Sciences）
- **版本 / 载体：** arXiv:2603.05185v1 [cs.RO]（2026-03-05 提交），arXiv HTML 全文
- **原文：** https://arxiv.org/abs/2603.05185 （全文阅读载体：https://arxiv.org/html/2603.05185 ）
- **项目页 / 代码：** 未确认开源代码（截至 2026-11-27，论文全文与索引条目均未列出本论文的官方项目页或源码仓库链接；文中 §2.3 参考文献 [1] 所附 `https://github.com/ly-geming/video2tasks` 为 BAAI 于 2025 年发布的 Video2Tasks 前置工作，并非本论文官方实现；文中基于开源 $\pi_{0.5}$（`openpi`）、`Florence-2-base` 与 `Qwen3-VL 32B` 进行扩展与调用）
- **核对范围：** 完整通读 arXiv:2603.05185v1 HTML 全文，覆盖 Abstract、§1 Introduction、§2 Related Work（§2.1–§2.3）、§3 Methodology（§3.1–§3.5、Algorithm 1 与值估计/RDP 公式）、§4 Experiments（§4.1–§4.3、Table 1、Table 2、Fig. 1–6 图注）、§5 Conclusions 及 References [1]–[24]
- **通读：** 已完成

## 原文核对笔记

### 1. 研究动机与核心框架
- 现有双系统 VLA（如 $\pi_{0.5}$、HAMSTER、OneTwoVLA、Hi Robot）普遍采用固定频率或刚性顺序切换机制：在顺利执行阶段重复调用 VLM 造成算力浪费与延迟抖动，而在遭遇外部扰动或长程任务死锁时又难以及时重规划或跳出无限重试循环。
- 本文提出 **Tri-System VLA** 异步分层架构，以开源 $\pi_{0.5}$（`openpi` 框架）为基础扩展为“大脑-小脑-评价器”三系统：
  - **System Two（大脑 / 子任务生成 $\pi_{\theta}^{Brain}$）**：复用共享 VLM 骨干（如 PaliGemma），以提示模板 `"Task: ℓ; Info: m; Current Subtask:"` 结合全局指令 $\ell$、多视角观测 $O_t$ 与短期记忆上下文 $m$ 自回归解码语义子任务 $g_t$。仅在初始时刻或 Critic 触发抢占时同步唤醒，其余时间休眠。
  - **System One（小脑 / 连续动作执行 $\pi_{\theta}^{cerebellum}$）**：保留 $\pi_{0.5}$ 默认流匹配（flow matching）动作专家容量与超参数，以 $(O_t, s_t, g_t)$ 为条件学习向量场 $v_t$，迭代去噪生成动作块 $a_{t:t+H}$。仅当动作缓冲 $A$ 耗尽或被抢占清空时同步生成新动作块。
  - **System Three（Critic / 视觉状态评估 $C_{\phi}(O_t, g_t)$）**：基于约 0.2B 参数的 `Florence-2-base`，以 VQA 形式（`"Evaluate the progress for task: g_t."`）在 ~20 Hz 控制循环中异步运行，输出离散化的进度分箱字符串或异常中断词元 `<aci>`。

### 2. Critic 训练与动态调度机制
- **蒙特卡洛进度估计**：对长为 $L$ 的有效子任务轨迹，定义连续进度 $V_t = \max(-1.0, (t - L) / L_{max}^{g_t}) \in [-1.0, 0.0]$，其中 $L_{max}^{g_t}$ 为训练集中子任务 $g_t$ 轨迹长度的 90 分位数；进一步离散化为 $B = 101$ 个文本分箱。
- **异常检测 `<aci>`**：对异常轨迹最后关键窗口（如末 20 帧）的进度分箱替换为高优先级语义词元 `<aci>`，与进度分箱共用同一文本词表和自回归交叉熵损失，无需额外 BCE 分类头，缓解罕见失败长尾问题。
- **训练设置**：冻结视觉塔，仅微调语言与投影层；AdamW 优化器，基础学习率 $1 \times 10^{-6}$、线性衰减、0 warmup、50 epochs、每卡 batch size 64。
- **三大抢占触发条件（Algorithm 1）**：
  1. 异常检测：$y_t == \text{<aci>}$ $\Rightarrow$ $m \leftarrow \text{"accident happened"}$；
  2. 子任务完成：$V_t > \tau_{succ}$（如 $\tau_{succ} \approx -0.041$）$\Rightarrow$ $m \leftarrow g_t + \text{" completed"}$；
  3. 执行停滞（类人启发规则）：记录 $V_t$ 未刷新历史最大值 $V_{max}$ 的累计帧数 $t_{stag}$，当 $t_{stag} \ge N_{stag}$（$N_{stag} = 180$）时触发物理状态重置（收回机械臂以打破视觉-运动学死锁）并设置 $m \leftarrow \text{"stagnation timeout"}$。
  - 任一条件触发后立即清空过期动作缓冲 $A \leftarrow \emptyset$，重置 $V_{max}, t_{stag}$，唤醒 System 2 生成新子任务 $g_t$。

### 3. 自动化子任务标注流水线（§3.5）
- **运动学关键帧提议**：从遥操作 3D 末端执行器（EE）轨迹出发，采用 Ramer-Douglas-Peucker（RDP）算法提取正交偏差超过阈值 $\epsilon$ 的几何拐点，结合夹爪开合离散状态变化，并通过 $\Delta t \ge 30$ 帧的贪心邻近过滤合并冗余候选帧。
- **VLM 检索接地**：将候选帧输入 `Qwen3-VL 32B` 从预定义子任务词表中检索语义标签，合并连续同标签段落得到子任务分割，随后辅以人工核验保证边界质量。

### 4. 真机平台与关键实验结论
- **硬件与数据**：Cobot Magic ALOHA 双臂 7-DoF 平台 + 3 台 Intel RealSense D435（头视 + 左右腕视）。`Arrange the Tableware` 采集 200 条遥操作轨迹 + 100 条 `Fallen` 恢复轨迹（刻意仅在右侧用右臂操作杯子以构造 OOD 偏差）；`Tidy up the Desk` 采集 200 条轨迹。
- **主实验（Table 1，每项 10 次试验）**：Tri-System $\pi_{0.5}$ 在 `Arrange the Tableware` 的 `Ordered`/`Scattered`/`Left cup`/`Fallen` 分别达到 `10/10`、`9/10`、`7/10`、`7/10`（显著优于 Single-System `8/10, 0/10, 0/10, 2/10` 与 Dual-System `7/10, 6/10, 1/10, 5/10`）；在 `Tidy up the Desk` 软体与多步序列 `Open`/`Bottle1`/`Bottle2`/`Overall` 达到 `9/10, 8/10, 5/10, 4/10`（两基线 `Overall` 均为 `0/10`）。
- **OOD 消融（Table 2，`Left cup`）**：加入左臂碗操作数据使成功率从 `0/10` 升至 `7/10`（验证子任务级技能跨物体迁移）；但采用显式 `<side> <obj> with <arm>` 提示时，GT 提示可达 `9/10`，而 System 2 生成提示降至 `3/10`，暴露 System 2 VLM 在未见属性组合上的分布过拟合仍是当前首要瓶颈。

对 wiki 的映射：[Critic in the Loop: A Tri-System VLA Framework for Robust Long-Horizon Manipulation](../../wiki/entities/paper-agentic-2603-05185.md)
