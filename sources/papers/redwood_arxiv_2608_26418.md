# Redwood：AI 自主设计的前沿 AI 加速器（Architect Labs）

> 来源归档（ingest）

- **标题：** Redwood: A Frontier AI Accelerator Designed, Verified, and Deployed from Scratch in 2 Weeks by AI
- **类型：** paper
- **原始链接：**
  - <https://arxiv.org/abs/2608.26418>
  - <https://architectlabs.com/blog/redwood>
- **机构：** Architect Labs
- **入库日期：** 2026-09-07
- **一句话说明：** Architect Labs 用端到端 AI 芯片设计系统（ALP）在两周内从高层规格自主生成性能模型、RTL、UVM、形式验证、固件与内核；首颗成果 Redwood 面向 physical AI 单 batch 超低延迟推理，Redwood Nano 已在 FPGA 跑 Qwen3/Llama，投影 Samsung 8 nm 相对 Jetson Orin Nano 达 3.4× 能效。

## 核心摘录（MVP）

### 1) 问题：工作负载与硅片节奏错位

- **摘录要点：** 架构定义领先量产数年，而目标模型数月即变；Moore 定律放缓后，专用化是剩余 perf/W 来源，但传统顺序芯片流无法跟上 workload 节奏。ALP 把软件–硅栈收成单一优化环，硬件与软件在同一目标下共设计与验证。
- **对 wiki 的映射：**
  - [Redwood 加速器实体页](../../wiki/entities/paper-redwood-architectlabs-accelerator.md)
  - [边缘–云端协同](../../wiki/concepts/edge-cloud-robotics.md)

### 2) Redwood 架构：tile 空间数据流 + Transformer 共设计

- **摘录要点：** 面向 **single-batch、低功耗、超低延迟** physical AI。N×M tile mesh + DMA；每 tile 含 RISC-V 控制核、INT8 MAC 阵列的矩阵引擎（systolic GEMM/GEMV）→ 向量引擎（SIMD、转置、FP 激活），512 KB 本地存储；片上网络支持 broadcast/multicast/信用流控。Softmax 复用 FlashAttention-4 仿真算法与 SIMD，无专用大块硬件。控制前端与计算后端分时钟域，kernel 执行时可关断控制逻辑省电。
- **对 wiki 的映射：**
  - [Redwood 加速器实体页](../../wiki/entities/paper-redwood-architectlabs-accelerator.md) — 微架构。
  - [控制/推理频率解耦](../../wiki/concepts/control-inference-frequency-decoupling.md) — 机载低延迟 decode 语境。

### 3) Redwood Nano 实测与 ASIC 投影

- **摘录要点：** 2×2 tile 版 Redwood Nano 在 **AMD Versal VPK180 @ 250 MHz** 端到端跑 **Qwen3-0.6B**（含 host prompt 与逐 token 回传），**12.1 tok/s**。投影到与 Jetson Orin Nano 同工艺级（Samsung 8 nm）：**1.75× decode 吞吐、1.9× 更低功耗 → 3.4× perf/W**；面积效率相对 Jetson 约一个数量级。DAC 现场演示 AI 设计加速器跑 live 推理。
- **对 wiki 的映射：**
  - [Redwood 加速器实体页](../../wiki/entities/paper-redwood-architectlabs-accelerator.md) — 评测。
  - [Jetson Orin NX](../../wiki/entities/jetson-orin-nx.md) — 机器人机载边缘算力对照生态。

### 4) ALP 流程与递归自改进

- **摘录要点：** 两名人类架构师写规格后，**100% RTL/UVM/形式验证/固件/驱动/内核** 由 AI 生成；每块 **≥95% coverage**；规格变更 **48 h** 内重验证并 redeploy FPGA。峰值单日 **115 merge commits**。Qwen3 部署在 Redwood 上作为 API 端点，采样发现 timing/kernel 优化并反馈下一代 Redwood——早期递归自改进示范。
- **对 wiki 的映射：**
  - [Redwood 加速器实体页](../../wiki/entities/paper-redwood-architectlabs-accelerator.md) — 设计方法论。
  - [ONNX Runtime vs MNN vs TensorRT](../../wiki/comparisons/onnxruntime-vs-mnn-vs-tensorrt.md) — 机载推理栈选型背景。

### 5) 开源状态（截至 2026-09-07）

- **摘录要点：** **未开源**。`architectlabs.com` 与 Redwood 博文 **未列 GitHub / Hugging Face / 权重**；ALP 为闭源商业平台。Redwood Nano 仅在 FPGA 演示，ASIC tapeout 路线指向 TSMC，尚无公开 GDS/RTL。
- **对 wiki 的映射：**
  - [Architect Labs Redwood 项目页](../sites/architectlabs-redwood.md)
  - [Redwood 加速器实体页](../../wiki/entities/paper-redwood-architectlabs-accelerator.md) — 局限。

## 当前提炼状态

- [x] arXiv 摘要 + 官方博文架构/评测节已对齐摘录
- [x] 与 1X「Redwood World Model」命名冲突已在实体页显式消歧
- [x] wiki 映射：`wiki/entities/paper-redwood-architectlabs-accelerator.md` 新建
