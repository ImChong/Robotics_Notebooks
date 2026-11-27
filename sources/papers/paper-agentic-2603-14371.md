# OxyGen: Unified KV Cache Management for VLA Inference under Multi-Task Parallelism

- **作者：** Xiangyu Li, Huaizhi Tang, Xin Ding, Weijun Wang, Ting Cao, Yunxin Liu
- **机构：** 清华大学智能产业研究院（AIR, Tsinghua University）、清华大学电子工程系、中国科学技术大学（USTC）
- **版本 / 载体：** arXiv:2603.14371v5 [cs.RO]（2026-03-15 提交 v1，2026-10-16 修订 v5；NeurIPS 2026 Main Track Poster）
- **原文：** <https://arxiv.org/html/2603.14371>（摘要页 <https://arxiv.org/abs/2603.14371>）
- **项目页 / 代码：** <https://air-embodied-brain.github.io/OxyGen/> / <https://github.com/air-embodied-brain/OxyGen>（**已开源**，Apache-2.0 协议，Gemma 组件附 `LICENSE_GEMMA.txt`；含 JAX 与 PyTorch 双后端 $\pi_{0.5}$ 推理与评测、LIBERO 文本记忆 LoRA 适配与数据集、Xiaomi-Robotics-0 / StarVLA / Qwen3.5 扩展模块，详见 [源码归档](../repos/paper-agentic-2603-14371.md)）
- **核对范围：** 通读 arXiv HTML 全文，包括 Abstract、§1 引言、§2 相关工作、§3 方法（§3.1–§3.2、Eq. 1–4、Fig. 2–3）、§4 实验（§4.1–§4.7、Table 1、Fig. 4–8）、§5 结论与局限、Appendix A（A.1–A.3、Eq. 5–7、Algorithm 1）、Appendix B（B.1–B.3、Table 2–3、Fig. 9–11）、Appendix C（Table 4–5）、Appendix D（D.1–D.4、Table 6–7、Fig. 12）、Appendix E（Fig. 13）及 References [1]–[71]；同时核对官方 GitHub 仓库主页、`README.md` 与文件列表（静态代码核对，未本地运行）。
- **通读：** 已完成

## 原文核对笔记

1. **问题定义与瓶颈定位（§1, §3.1）：** 具身智能体常需基于同一帧观测 $\mathbf{o}_t$ 并发执行多类时间约束不同的任务（如高频机械控制与跨周期语言/记忆生成），即 **multi-task parallelism（多任务并行）**。Mixture-of-Transformers（MoT）VLA（如 $\pi_{0.5}$、WALL-OSS、Xiaomi-Robotics-0）虽在架构上支持多模态专家路由，但现有推理框架（如 `openpi`）采用 **isolated execution（隔离执行）**，将各任务的 KV cache 孤立管理，导致两类低效：（i）**冗余计算**：同一观测在 VLM 主干被重复 prefill；（ii）**资源竞争**：长周期语言解码与高频动作去噪竞争 GPU，拉高帧延迟。
2. **形式化建模与优化目标（§3.1, Eq. 1–3）：**
   - 观测编码（prefill）产出模态无关的 KV cache：$\mathcal{K}_t = \{(\mathbf{K}_{t,l}, \mathbf{V}_{t,l})\}_{l=1}^L = \Theta_{\text{VLM}}(\mathbf{o}_t)$。
   - 动作专家通过 $S$ 步流匹配/扩散去噪生成 $H$ 步动作块 $\mathbf{A}_t$，语言专家基于 $\mathcal{K}_t$ 自回归生成至多 $N$ 个 token $\mathbf{y}_t$。
   - 若每帧新到 1 个请求、以帧延迟 $T$ 产出 $H$ 个动作并推进批大小为 $B$ 的语言请求各 $k$ 步解码，优化目标为在满足动作硬截止频率 $f = H/T \ge f_{\min}$ 下最大化语言吞吐 $\tau = Bk/T$。
3. **统一 KV 缓存管理（§3.2, Appendix A.2–A.3）：** 提出 **Unified KV Cache Manager** $\mathcal{M}$，将 KV cache 抽象为跨任务、跨控制周期的一等共享资源：
   - **Cross-task KV sharing（跨任务 KV 共享）：** 每帧只做一次 $\text{Prefill}(\mathbf{o}_t)$ 得到 $\mathcal{K}_t$，同时分发给动作专家做 $S$ 步去噪，以及初始化新语言请求的可恢复生成状态 $\sigma_t = (\mathcal{K}_t, \mathbf{y}_t, \delta_t)$。
   - **Cross-frame continuous batching（跨帧连续批处理）：** $\mathcal{M}$ 通过 `Store` / `Retrieve` / `Update` / `Remove` 管理活跃请求集 $\mathcal{R}$，每帧用 `Batch` 将未完成的语言状态 $\sigma_{t_i}$ 沿 batch 维堆叠为 $\hat{\sigma} = (\hat{\mathcal{K}}, \hat{\mathbf{y}}, \hat{\boldsymbol{\delta}})$，共同执行 $k$ 步自回归解码后 `UnBatch`，释放已完成（EOS 或达上限 $N$）的请求。
4. **核心定量结果（§4, Appendix B–D）：**
   - **$\pi_{0.5}$ 端到端与消融（RTX 4090 & Jetson AGX Thor）：** 在 LIBERO/DROID/ALOHA 配置下，OxyGen 相对顺序隔离基线实现 **1.2–3.7×** 加速，在 RTX 4090 上同时达到 **> 200 tokens/s** 语言吞吐与 **70 Hz** 动作频率；LIBERO（$S=10$）上，仅跨任务 KV 共享在 $N=5$ 时将帧延迟从 200.3 ms 降至 145.1 ms（**1.4×**），叠加跨帧连续批处理在 $N=30, k=5$ 时进一步从 469.3 ms 降至 175.8 ms（额外 **2.7×**，频率从 21.3 Hz 升至 56.9 Hz），并在跨 $N$ 增长时在 4090 上稳定维持约 60 Hz、在 Thor 上维持约 27 Hz。
   - **多模型与多专家迁移（Table 1, Table 4–5）：** RTX 4090 上 $N=30, k=5$ 时，Xiaomi-Robotics-0 加速 **3.23×**（$k=1$ 为 5.83×），StarVLA `Qwen3VL-PI_v3`（2/3/4 专家）加速 **3.60× / 3.53× / 3.69×**（$k=1$ 为 6.64× / 6.05× / 6.30×）；Qwen3.5（0.8B–9B）加速 **3.58×–3.79×**。
   - **动作质量与资源代价（Table 2–3, Table 7, D.3）：** LIBERO 10 种子共 20,000 次 rollout 平均成功率 **96.76%**（95% CI [96.55%, 96.97%]），与 `openpi` 官方点估计 96.85% 基本持平；RTX 4090 上（batch=4）峰值显存较基线（6.43 GiB → 7.41 GiB）仅增 **15%**，平均功耗降 **47%**（293.5 W → 154.4 W），单请求能耗降 **78.0%**（117.4 J → 25.8 J）。3,000 帧变负载压测下显存保持有界（JAX 6.57–6.94 GiB，PyTorch 7.29–7.65 GiB）。
   - **人形真机部署（§4.7, Fig. 8）：** 搭载板载 Jetson AGX Thor 的人形机器人实测（3 路 224×224 RGB，$H=10, S=10, N=30, k=5$），动作+语言总推理延迟从隔离基线 **1030 ms**（207.5 + 822.3 ms）降至 **393 ms**（198.0 + 195.4 ms），动作下发后 195.4 ms 的语言解码可隐藏在 333.0 ms 的物理动作执行期间。
5. **自述局限（§5）：** 当前仅针对动作与语言两类模态并发评测，尚未在包含视频等更多输出模态的统一模型、异步控制/模型级压缩协同优化、以及涵盖数据采集-多任务训练-闭环部署的端到端真机应用中完成充分验证。

- **对 wiki 的映射：** [OxyGen: Unified KV Cache Management for VLA Inference under Multi-Task Parallelism](../../wiki/entities/paper-agentic-2603-14371.md)
