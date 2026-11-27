# Efficient Vision-Language-Action Management and Serving for Robot Factories

- **作者：** Dionysios Adamopoulos, Nattapol Chanpaisit, Basel Fakhri, Christina Giannoula
- **机构：** Max Planck Institute for Software Systems
- **版本 / 载体：** arXiv:2609.12075v1 [cs.DC]（2026-09-10，CC BY 4.0，HTML 全文共 7 节）
- **原文：** https://arxiv.org/html/2609.12075 （arXiv 摘要页：https://arxiv.org/abs/2609.12075 ）
- **项目页 / 代码：** 未确认开源代码（截至 2026-11-27，arXiv:2609.12075v1 原文、摘要页及索引条目均未给出独立项目页或官方代码仓库链接）
- **核对范围：** 通读 arXiv HTML v1 全文（Abstract、§1 Introduction、§2 Background & Motivation、§3 Robion: Design Overview、§4 Robion: Detailed Design、§5 Evaluation（Fig. 1–14 对应正文分析）、§6 Related Work、§7 Conclusion 及 References）。
- **通读：** 已完成

## 原文核对笔记

- **背景与设计动机（§1–§2）：**
  - 典型 VLA 模型由视觉-语言模型（VLM）与动作扩散 Transformer（ADiT）两阶段构成：VLM 编码图像与文本指令生成隐状态/小规模 KV 缓存，ADiT 执行固定步数（5–10 步，每步 1–2 ms）生成 10–50 步未来动作块（如 $\pi_0$ 每 500 ms / 2 Hz 触发一次，单次推理 $\le 100$ ms）。
  - Roofline 与利用率剖析显示两阶段资源特征互补：VLM 为计算密集型（高算术强度 GEMM），ADiT 为访存密集型（低算术强度小矩阵与逐元素向量算子）。在 RTX 6000 Pro 上以 batch size 4 运行 3B 参数 $\pi_0$，VLM 平均 Tensor Core 利用率为 52%，ADiT 仅为 15%（batch size 16 时也仅 27%）。
  - 机载嵌入式 GPU（如 Jetson AGX Thor，70W 功耗模式）相比本地边缘服务器（RTX 6000 Pro / H100 + WiFi 7）推理延迟高出约 $3\times$，且机载推理会缩短机器人电池续航高达约 45%。而 VLA 模型参数量为 1–5B（如 SmolVLA 450M、Xiaomi-Robotics-0 4.7B），权重仅约 10 GB，远低于服务器级 GPU 显存（80–96 GB），适合单卡多模型共置。
  - 现有系统缺口：ROSA、Kairos、Armory 将 VLA 视作单一流水线且不面向请求级 SLO 做 GPU 内并发共享；vLLM-Omni、M*、SGLang-Omni 等跨卡多阶段服务系统面向百毫秒级吞吐优化，在毫秒级 VLA 阶段上不仅跨卡浪费算力，即便放回单卡，其 ADiT 每步去噪轮询阶段间缓存的开销也占到总延迟的 30–35%。

- **Robion 服务引擎设计（§4.1）：**
  1. **固定批大小卡内阶段解耦（§4.1.1）：** 在单卡内使用两条异步 CUDA 流分别执行 VLM 与 ADiT；仅在阶段起始接纳固定大小批次（2 的幂次，默认 `max_batch=2`），消除每步去噪的缓冲区轮询与张量重打包开销，并在初始化阶段为各 `(stage, batch_size)` 组合预录制 CUDA Graph。
  2. **锁步与受保护动态空间 SM 划分（§4.1.2）：** 由两个 CPU 线程通过线程屏障以锁步（lockstep）同步驱动：第 $k$ 个锁步中，VLM 流处理第 $k$ 批（$b_k$），ADiT 流并行处理第 $k-1$ 批（$b_{k-1}$）。利用 **CUDA Green Contexts**（AMD 侧可用执行上下文）在每个锁步开始前限制 VLM 流可用 SM 数量（避免运行中抢占导致 CUDA Graph 失效或进度丢失），保证小核 ADiT 始终有空闲 SM 立即并发执行。初始化阶段通过离线剖析（枚举 $O(U^2 \cdot \log(\text{max\_batch})^2)$ 种架构/批大小组合，按 GPU 粒度如 8 SM 的倍数搜索）生成查表配置（例如 188 SM 设备上 $(1,1)\to 152$、$(2,1)\to 176$、$(2,2)\to 168$ SMs）。
  3. **SLO 感知双流多模型共置（§4.1.3）：** 单卡上所有共置模型共享同一对 VLM 流与 ADiT 流（避免同类阶段争抢资源以及 8–10 $\mu$s 级短核拦截调度开销）。每个模型维护独立 FIFO 队列，锁步起始采用最早剩余 SLO 优先（Earliest-SLO-First, ESF）策略选择队头剩余时间最短的队列出队。

- **Robion 管理引擎与初始化（§4.2–§4.3）：**
  - **灵活模型放置（§4.2.1）：** 支持异构架构或同构微调变体在单卡共置、跨卡复制，以及将 GPU 划分为不相交分组（每组专用于单一 VLA 架构的微调变体，消除不同架构阶段耗时差异导致的锁步等待）。
  - **自适应流量控制器（§4.2.2）：** 初始化阶段压测确定单卡对模型 $m_i$ 在目标 SLO 达成率（如 98%）下的最大服务机器人数 $C_i$；针对给定规模 $N_i$ 与频率 $f_i$，采用 **Google OR-Tools** 求解整数规划，满足 $\sum_j r_{ij}=N_i$ 与 $\sum_i r_{ij}/C_i \le 1$，并最小化 $\sum_j \sum_{i<k} r_{ij} r_{kj}$，优先将同模型请求集中到单卡队列以提升批大小，仅在超出 $C_i$ 时分流。
  - **初始化开销（§4.3）：** 在 $4\times$ RTX 6000 Pro 上部署 2 种架构共 8 个微调变体（2 个分组、每卡 4 个变体），CUDA Graph 捕获约 4 分钟，SM 限制剖析约 3 分钟，单卡容量压测约 1 小时（跨卡并行），求解器耗时约 10 秒。

- **核心实验结果（§5）：**
  - 评测模型：$\pi_0$（PI0）、SmolVLA（SVLA）、GR00T N1.5（GR15）、Xiaomi-Robotics-0（XR0），图像与指令来自 LIBERO 数据集，默认请求频率 $f=2$ Hz；SLO 期限取 Jetson AGX Thor（70W）本地延迟（PI0 156 ms、XR0 117 ms、SVLA 50 ms、GR15 53 ms），端到端延迟计入 WiFi 7 网络延迟；平台为 $4\times$ RTX 6000 Pro Blackwell（96 GB，188 SMs）与 $4\times$ H100（80 GB）。
  - **单模型单卡（§5.2）：** Monolithic 在 98% SLO-AT 下平均可服务比 vLLM-Omni 高 $4.4\times$ 的机器人负载；Robion 在 Monolithic 饱和前最大负载处 P99 延迟平均低 $1.4\times$，在 98% SLO-AT 下平均承载比 vLLM-Omni 高 **$6.7\times$**、比 Monolithic 高 **$1.5\times$** 的负载，单卡平均可服务 **20 台**机器人（相对每机配 Jetson AGX Thor 硬件成本降低约 $6.8\times$）。消融实验（Fig. 7）显示在 PI0（20 台）/ XR0（24 台）下，Monolithic 为 0%/0%，仅阶段解耦为 56%/39%，不相交 SM 划分为 67%/85%（因 188 非 8 的倍数向下取整至 184 而浪费 4 个 SM），受保护动态 SM 划分达 82%/89%。静态固定 SM（如 96/188）导致可服务负载平均下降约 18%。
  - **单卡多模型共置（§5.3）：** 最多服务 18 台机器人时，Robion 的 SLO 违约率（SLO-MR）平均比 Monolithic 低 **$2.1\times$**（vLLM-Omni 达 100% 违约）；ESF 队列调度相比 RR 与 LQF 的平均 SLO-AT 分别高 **$1.06\times$** 与 **$1.07\times$**。
  - **多卡放置与流量控制（§5.4）：** 4 卡部署 4 个变体（1/2/2/4 Hz）时，Robion 流量控制器相比单模型独享卡（SMG）与均匀切分（EPS）平均分别减少 **$8.2\times$** 与 **$4.4\times$** SLO 违约；8 变体（4 PI0 + 4 XR0）4 卡部署下，按架构分不交分组（`Best-Group`）始终优于混合共置（`Mix`），在 98% SLO-AT 下最高可服务 **64 台**机器人。PI0 4 卡对比跨卡解耦时，Robion 卡内解耦+全卡复制可服务 **72 台**，优于最佳跨卡解耦 `Dis-3:1` 的 **60 台**。
  - **云端 vs 边缘对比（§5.5）：** 4 模型各占 1 卡、每模型 24 台机器人时，云端部署因网络延迟更高，SLO-AT 在 $4\times$ RTX 6000 Pro 与 $4\times$ H100 上分别比边缘（WiFi 7）低 **23.6%** 与 **18.4%**。

- 对 wiki 的映射：[Robion：面向机器人工厂的高效 VLA 管理与服务系统](../../wiki/entities/paper-agentic-2609-12075.md)
