# EventVLA: Event-Driven Visual Evidence Memory for Long-Horizon Vision-Language-Action Policies

> 来源归档（ingest）

- **标题：** EventVLA: Event-Driven Visual Evidence Memory for Long-Horizon Vision-Language-Action Policies
- **类型：** paper
- **来源：** arXiv abs / HTML；项目页与 GitHub 交叉核对
- **原始链接：**
  - <https://arxiv.org/abs/2606.20092>
  - <https://ganlin-yang.github.io/EventVLA.github.io/>
  - <https://github.com/InternRobotics/EventVLA>
- **作者：** Ganlin Yang, Zhangzheng Tu, Yuqiang Yang, Sitong Mao, Junyi Dong, Tianxing Chen, Jiaqi Peng, Jing Xiong, Jiafei Cao, Jifeng Dai, Wengang Zhou, Yao Mu, Tai Wang 等
- **机构：** 中国科学技术大学（USTC）、上海人工智能实验室（Shanghai AI Lab）、上海交通大学（SJTU）、大连理工大学（DUT）、华为（Huawei）、香港大学（HKU）、清华大学（Tsinghua）、北京大学（PKU）
- **入库日期：** 2026-07-15
- **一句话说明：** 面向 **非马尔可夫长程操作**，提出 **稀疏视觉证据记忆** 端到端 VLA：**基础视觉锚点**（初始帧 + 短期滑窗）+ **前瞻式 KEM 关键帧证据记忆**（从 VLA 隐状态预测未来 chunk 内关键帧概率，以原始图像写入 FIFO 缓冲）；发布 **RoboTwin-MeM** 诊断基准（8 任务、中间关键帧数 $n\in[1,5]$）；**RMBench 67.8%**、**RoboTwin-MeM 75.2%**、真机 ARX ACONE 四项任务最高 **90%**，相对 SOTA 记忆 VLA 平均 **+40%**。

## 核心论文摘录（MVP）

### 1) 问题：非马尔可夫与三类记忆 VLA 局限

- **链接：** <https://arxiv.org/abs/2606.20092> §1
- **摘录要点：** 标准 VLA 假设任务相关信息持续可见；现实中遮挡、位移后需回忆早期视觉证据。现有记忆增强路线三类瓶颈：**双系统 Memory-VLA**（VLM 规划 + 低层控制）延迟高、误差传播；**循环/压缩历史** 信息瓶颈丢细节；**记忆缓冲** 盲目堆帧、冗余淹没稀疏关键证据。核心问题：**何时、存什么视觉证据** 才能在算力约束下最大化成功率。
- **对 wiki 的映射：**
  - [EventVLA（论文实体）](../../wiki/entities/paper-eventvla-visual-evidence-memory.md) — 问题定义与稀疏证据记忆范式。
  - [VLA](../../wiki/methods/vla.md) — 长程记忆增强子路线。

### 2) 基础视觉锚点（Foundational Visual Anchors）

- **链接：** arXiv §3.1
- **摘录要点：** 记忆缓冲 $M_t=A_t\cup E_t$。**锚点** $A_t=\{o_0\}\cup\{o_{t-K},\dots,o_{t-1}\}$：初始帧保留不变全局布局，短期滑窗提供运动与进度线索。对 RMBench 等「布局持久可见」任务，**仅锚点** 即达 **67.8%** SOTA；去掉初始帧或短期历史分别跌至 **33.7%** / **23.8%**。
- **对 wiki 的映射：**
  - [EventVLA](../../wiki/entities/paper-eventvla-visual-evidence-memory.md) — 锚点与 KEM 分工。
  - [Manipulation](../../wiki/tasks/manipulation.md) — 长程桌面/双臂操作语境。

### 3) KEM：前瞻式关键帧证据记忆

- **链接：** arXiv §3.2–3.3
- **摘录要点：**
  - **预测头：** 从 VLA 最后一层隐状态 $h_t\in\mathbb{R}^{H\times d}$（动作 chunk 长度 $H$）经轻量 MLP 得 $\hat{\mathbf{p}}_t\in[0,1]^H$，每维为对应未来步成为任务关键帧的概率。
  - **写入：** $\hat{p}_t^i\geq\tau_{\text{commit}}$ 时将 $t+i$ 原始图像写入事件缓冲 $E_t$；容量 $N_{\max}$、**FIFO** 驱逐。
  - **输入：** $I_{\text{input}}=\text{concat}([A_t,E_{t-1},o_t])$ 送入 VLM 视觉编码器，自注意力跨稀疏历史帧提取时序关联。
  - **训练：** **Qwen3-VL** 离线自动标注关键帧时间戳；**时序平滑软标签 + chunk-wise BCE**（$L_{\text{kem}}$）；与动作损失 $L_{\text{action}}$ 端到端联合优化；**teacher→student 课程** 从 GT 记忆构建过渡到自主预测；推理用 **1D NMS + 冷却** 抑制冗余写入。
- **对 wiki 的映射：**
  - [EventVLA](../../wiki/entities/paper-eventvla-visual-evidence-memory.md) — KEM 流程 Mermaid。
  - [StarVLA](../../wiki/methods/star-vla.md) — QwenOFT 骨干与 EventVLA 实现栈。

### 4) RoboTwin-MeM 诊断基准

- **链接：** arXiv §4；项目页
- **摘录要点：**
  - 基于 **RoboTwin 2.0** / **SAPIEN**；8 项 **真正非马尔可夫** 任务，每 episode **430–1544** 步。
  - 参数 **$n$**：须动态保留的**中间事件关键帧数**（1–5），分层难度。
  - 代表能力：**瞬态记忆**（揭盖见色后遮挡）、**事件计数**（Press Button Keyframe）、**上下文模仿**（Reproduce Route 观察演示后复现随机路径）。
  - 相对 RMBench，现有锚点策略在 MeM 上仅 **18.0%**，凸显中间态记忆缺口。
- **对 wiki 的映射：**
  - [EventVLA](../../wiki/entities/paper-eventvla-visual-evidence-memory.md) — 基准与任务表。
  - [RoboTwin 2.0](../../wiki/entities/robotwin.md) — 底层仿真平台。

### 5) 仿真与真机定量结果

- **链接：** arXiv §5；Table 1–3；Fig. 4
- **摘录要点：**
  - **RMBench（17 任务）：** EventVLA（VA only）**67.8%** avg，超 MemoryVLA-QwenOFT **41.7%**、Mem-0 **42.0%**。
  - **RoboTwin-MeM：** VA only **18.0%** → **VA+KEM 75.2%**；$\pi_{0.5}$ **7.8%**、MemoryVLA-QwenOFT **10.8%**。
  - **RoboTwin 2.0 标准马尔可夫：** Easy **83.8%** / Hard **81.6%**，略优于 QwenOFT **80.0%** / **78.0%**。
  - **消融：** 隐式潜记忆 bank **24.9%**；硬标签 **48.8%**；无 NMS **53.4%**；$N_{\max}=2$ **32.0%**；chunk=15 **13.6%**。
  - **真机 ARX ACONE 双臂 4 任务 ×20 trials：** Find Block Easy/Hard **90%** / **60%**；Pick-X-Times **90%**；Pick in Order **75%**；$\pi_{0.5}$ **0–10%**；$\pi_{MEM}$ **30–50%**。
- **对 wiki 的映射：**
  - [EventVLA](../../wiki/entities/paper-eventvla-visual-evidence-memory.md) — 结果表与消融。
  - [Bimanual Manipulation](../../wiki/tasks/bimanual-manipulation.md) — 双臂真机评测语境。

### 6) 与 KEMO 等相邻工作的定位

- **链接：** arXiv §2；项目页 Built on RoboTwin 2.0, RMBench, StarVLA
- **摘录要点：** 与 [KEMO](../../wiki/entities/paper-kemo-event-driven-keyframe-memory-vla.md) 同属 **事件/关键帧稀疏记忆**，但 EventVLA 用 **学习式前瞻预测头 + 原始图像拼接** 而非运动学启发式选帧 + cross-attention 融合；与 **MemoryVLA**（稠密 token merge）、**π_MEM**（压缩历史）对比，强调 **显式保留原始关键帧** 避免信息瓶颈。局限：有界 FIFO 在 **>10 min** 高密度事件任务可能饱和。
- **对 wiki 的映射：**
  - [EventVLA](../../wiki/entities/paper-eventvla-visual-evidence-memory.md) — 对照与局限。
  - [KEMO](../../wiki/entities/paper-kemo-event-driven-keyframe-memory-vla.md) — 规则/运动学事件选帧路线。
