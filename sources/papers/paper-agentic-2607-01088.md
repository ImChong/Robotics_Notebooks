# ROSA: A Robotics Foundation Model Serving System for Robot Factories

- **作者：** Wenqi Jiang, Jason Clemons, Rowland O’Flaherty, Hugo Hadfield, Alperen Degirmenci, Shuran Song, Yashraj Narang, Christos Kozyrakis
- **机构：** NVIDIA Research; Stanford University
- **版本 / 载体：** arXiv:2607.01088v1 [cs.RO]（2026-07-01，HTML 全文）
- **原文：** [arXiv HTML 全文 (2607.01088)](https://arxiv.org/html/2607.01088)、[arXiv 摘要页](https://arxiv.org/abs/2607.01088)
- **项目页 / 代码：** 未确认开源代码（截至 2026-11-27 核查 arXiv:2607.01088v1 原文及关联条目，未提供独立项目页或官方 GitHub 仓库链接）
- **核对范围：** 通读 `paper_text.txt` 全文（Abstract、§1 引言、§2 背景与相关工作、§3 ROSA 问题建模/架构/声明式接口/同构与异构调度/准入与容错/Ray Serve 实现、§4 8×H200 合成负载与 Franka Panda 真机实验、Table 1–3、Figure 1–16 说明及 §5 结论）
- **通读：** 已完成

## 原文核对笔记

- 对 wiki 的映射：[ROSA: A Robotics Foundation Model Serving System for Robot Factories](../../wiki/entities/paper-agentic-2607-01088.md)

### 1. 研究动机与三大核心原则
- **打破两项常见误区：**
  1. 将机器人基础模型（RFM）推理视为纯边缘计算（单机片上系统 SoC 或一机一卡专属服务器）会导致算力受限、移动平台续航折损（如 Figure 02 双 RTX GPU 占整机功耗约一半），且因同步执行等待物理动作与缺乏跨机器人批处理导致 GPU 严重空转。
  2. 仅优化单一动作模型（System 1）延迟忽视了工厂生产所需的多模型流水线（System 2 规划、安全检查、任务进度监控）以及面向整厂吞吐的优化目标。
- **三大系统设计原则：**
  1. **共享服务器级 GPU 池：** 利用数据中心 GPU 的高带宽与算力（如对比同代 Jetson Thor 与 NVIDIA B100，后者具备 29× 显存带宽与 3.5× FP8 算力）、延长电池续航并实现跨机器人批处理；机器人端保留轻量本地算力负责 ≥100 Hz 高频执行器控制（如 SONIC）与超时/安全兜底。
  2. **面向机器人的声明式编程抽象：** 统一描述服务器集群、机器人车队、多模型组合与依赖关系、各组件 P99 延迟 SLO 与调用频率，以及安全警告/SLO 超时/任务失败后的重试与人工接管策略。
  3. **工厂目标驱动调度：** 将模型划分为决定动作频率的**目标耦合模型** $\mathcal{G}=\{\mathrm{S1},\mathrm{S2}\}$ 与固定频率运行的**义务模型** $\mathcal{O}=\{\mathrm{safe},\mathrm{mon}\}$；在满足每类任务延迟 SLO、频率约束与最低总吞吐下限 $F_c^{\min}$ 的前提下，最大化全厂加权动作吞吐 $\sum_c v_c K_c f_c$。

### 2. 调度算法与系统实现
- **同构任务调度（§3.4.2）：** 先基于实测延迟分布 $L_{m,h}(b,\lambda)$ 为义务模型分配最小可行 GPU 集合 $S_{\mathrm{obl}}$；再对共享动作频率 $f$ 做二分搜索，在剩余 GPU $S_{\mathrm{goal}}$ 上枚举可行单卡配置 $\mathcal{P}(f)$，用整数线性规划（ILP）求解部署并校验闭环频率上界 $f \le 1 / (t_{act} + \ell_{\mathrm{S1}} + \ell_{\mathrm{S2}}/H)$。
- **异构任务调度（§3.4.3）：** 采用贪心自适应前沿搜索遍历多维频率向量 $\mathbf{f}$，配合「单类任务隔离 ILP 装箱 + 贪心跨类合并压缩」两阶段原语以控制组合爆炸。
- **运行时适配与容错（§3.4.4）：** 新机器人接入优先走保留现有模型放置的准入控制，不足时再全局重调度；容错采用预加载权重的热备 GPU 节点实现瞬时流重定向，避免双活重调度对正常机器人的扰动。
- **实现栈（§3.5）：** 基于 Ray Serve 构建分布式网关与副本管理；自回归 VLM（System 2 / Safety / Monitor）运行于 vLLM（连续批处理），System 1 运行于 PyTorch（`torch.compile` + CUDA Graphs + FlashAttention，FP16）或 JAX（FP16）；调度器使用 Python 与 OR-Tools 求解 ILP。

### 3. 核心实验数据核对（§4，8×NVIDIA H200）
- **评测模型：** System 1 主用 GR00T N1.6（PyTorch）并对比 $\pi_{0.5}$（JAX），System 2 与 Monitor 使用 Qwen2.5-VL-7B，Safety 使用 Qwen2.5-VL-3B。
- **对比独占基线：** 在 P4（四组件装配）任务中，32 台机器人下 ROSA 达到 **89.4** SLO 达标动作/秒，较 1-GPU-per-model（仅支持 2 台，7.4 动作/秒）最高提升 **12.06×**。
- **对比共享静态基线：** 整体 SLO 达标工厂吞吐最高提升 **2.44×**；32 台机器人下 P2 提升 **1.23×**（106.9 vs 87.1），P3/P4 基线均因排队超时跌至 0，ROSA 分别维持 106.9 与 89.4 动作/秒（P4 达标率 99.9%）；异构混合负载 Mix2 与 Mix4 峰值吞吐分别提升 **2.21×** 与 **2.30×**。
- **消融验证：** P4 负载 64 台机器人下，请求速率控制带来 **3.18×** 达标吞吐提升（78.8 vs 24.8，SLO 达标率 97.2% vs 17.0%）；即使基线开启速率控制，均分与按参数量加权分配仍需最多 **5.5×** 与 **8.6×** 的 GPU 才能追平 8 卡 ROSA；System 1 批大小增至 16 在高并发下可比 batch=1 提速 **2.46×**。
- **真机功能验证：** 在 Franka Panda 机械臂上完整跑通类似 P3 的工具收纳任务（Action + Safety VLM + Monitor VLM）。
