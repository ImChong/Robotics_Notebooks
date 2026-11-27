# How Fast Can I Run My VLA? Demystifying VLA Inference Performance with VLA-Perf

- **作者：** Wenqi Jiang, Jason Clemons, Karu Sankaralingam, Christos Kozyrakis
- **机构：** NVIDIA Research
- **版本 / 载体：** arXiv:2602.18397v1 [cs.RO]（2026-02-20 提交，arXiv HTML 全文 + 官方 GitHub 仓库）
- **原文：** <https://arxiv.org/html/2602.18397>（摘要页 <https://arxiv.org/abs/2602.18397>）
- **项目页 / 代码：** <https://github.com/NVlabs/vla-perf>（**已开源**，Apache 2.0 协议；含 `vla-perf/` 性能分析与绘图脚本，以及扩展版 `genz/` 硬件性能后端，详见 [源码归档](../repos/paper-agentic-2602-18397.md)）
- **核对范围：** 通读 arXiv HTML 全文，包括 Abstract、§1 引言、§2 背景与动机（§2.1–§2.3）、§3 VLA-Perf 性能模型与保真度验证（公式 1–4、Table 1）、§4 评测设置与全部分析（§4.1–§4.11、Table 2–9、Figure 5–9 图注与正文数据）、§5 结论与未来工作、Appendix A（Table 10–11 硬件与模型规格）及 References [1]–[45]；并结合官方 GitHub 仓库主页与 `README.md`、`files_sample` 核对开源状态与脚本入口（静态核对，未本地运行）。
- **通读：** 已完成

## 原文核对笔记

1. **核心问题与工具定位（§1–§3）：** 论文针对视觉-语言-动作（VLA）模型在真实机器人部署中的实时推理约束（定义 10 Hz 为可接受阈值、100 Hz 为高性能目标），提出基于 Roofline 的解析性能模型 **VLA-Perf**，构建于扩展版 GenZ 后端之上。模型将端到端推理延迟拆解为组件计算延迟与跨设备网络传输延迟：$T_{\text{total}} = \sum_{m\in\mathcal{M}} T_m + \sum_{d\in\mathcal{D}} T_d$，各算子延迟取计算上限与访存上限最大值 $T_o = \max(\text{FLOPs}_o / \text{FLOP/s}_h, \text{Bytes}_o / \text{MemBW}_h)$，网络传输按 $T_d^{\text{net}} = \text{NetLat} + \text{Bytes}_d / \text{NetBW}$ 建模。
2. **模型保真度验证（§3, Table 1）：** 以 Ma et al. (2025) 在 RTX 4090 上针对 $\pi_0$ 的 Triton 优化实现为实测对照（10 步流匹配、action chunk 63、空文本提示），1/2/3 相机下 VLA-Perf 预测为 14.7 / 22.5 / 30.4 ms，实测为 20.0 / 27.3 / 36.8 ms，实测达到理论 Roofline 的 **73.3% / 82.3% / 82.6%**。
3. **基准 $\pi_0$ 与硬件瓶颈（§4.1–§4.2, Table 3–4, 10–11）：** 评估场景为双臂 UR5e（14 DoF、3 路 224×224 相机共 768 视觉 token + 32 语言 token = 800 tokens、action chunk = 50、10 步去噪、BF16/FP16）。$\pi_0$（2.7B，SigLIP 411.19M + Gemma 1.98B + Action Expert 292.63M）在 Jetson Thor / RTX 4090 / A100 / H100 / B100 上无网络端到端延迟分别为 52.57 / 31.06 / 16.20 / 6.15 / 3.18 ms（19.0 / 32.2 / 61.7 / 162.5 / 314.4 Hz）。算子强度（OI）：Vision=321.4、VLM=542.8、Action=54.0；Action Expert 在全部 GPU 上均受内存带宽限制（memory-bound），而 Vision 与 VLM 在 4090/A100/H100/B100 上为计算受限（compute-bound），在采用 270 GB/s LPDDR 的 Jetson Thor（Balance OI=1481.5）上则全部受限于内存带宽。
4. **模型缩放与长上下文（§4.3–§4.4, Table 5–6）：** 将 $\pi_0$ 缩放至 $\pi_0$-L (9.1B)、$\pi_0$-XL (16.7B)、$\pi_0$-XXL (81.3B) 时，各组件延迟近似随参数量线性增长；B100 在 81.3B 下仍可维持 9.6 Hz，RTX 4090 在 16.7B 即 OOM，Jetson Thor 在 16.7B 降至 2.1 Hz。引入历史视觉 KV cache 的长上下文评测表明：1,000 历史步需 18.3 GB 总显存（13.2 GB KV cache），B100 可维持 11.7 Hz（85.2 ms），而 Thor/4090 降至 1.3 / 1.1 Hz（100 步约 8.1 / 8.5 Hz）；10,000 步（137.0 GB 总显存）时 B100 亦跌至 1.2 Hz（823.7 ms）。
5. **去噪步数、Action Chunk 与生成范式对比（§4.5–§4.6）：** 在 B100 上，去噪步数从 10 增至 50 使 Action Expert 延迟增至 5×、端到端增至 2.15×；而 chunk size 从 50 增至 250（5×）仅使 Action Expert 增加 40%、端到端增加 11%（因其受访存主导）。在 chunk=50、14 DoF 下，标准扩散版（3.2 ms）比逐维自回归版（327.6 ms）快 **102.4×**；启用并行解码的自回归变体在 chunk $\le 10$ 时占优，但在 chunk=50 时 OI 由 135.9 升至 477.7（超过 B100 平衡点 218.8），转为 compute-bound，延迟迅速上升。
6. **端边云部署、协同切分、异步与双系统（§4.7–§4.11, Table 7–9）：**
   - 端 vs 边 vs 云：多数网络下边缘/云端服务器（含 WiFi 下的 RTX 4090）快于 Thor 端侧，仅在 4G 或慢速云网络下端侧更有优势。
   - 端边协同（B100 跑 VLM + Thor 跑 Action Expert）：需通过网络下载 VLM KV cache（10G 以太网 12.4 ms、WiFi 7 43.7 ms、5G 257.7 ms），因此始终慢于纯服务端，且在无线条件下反而慢于纯端侧。
   - 异步推理：可隐藏网络传输与 GPU 计算，在 B100 上使 WiFi 7 吞吐从 119.7 提升至 314.4 Hz（2.63×），5G 从 35.9 升至 215.3 Hz（5.99×），4G 从 13.7 升至 50.5 Hz（3.68×），慢速云+4G 从 3.7 升至 50.5 Hz（13.79×），但不降低单次端到端延迟且引入状态陈旧性（staleness）。
   - 双系统流水线（System 1 高频动作 + System 2 低频 VLM 5/10 Hz）：B100 + 10G 以太网在 10 Hz 限频下可达 676.0 Hz（2.24×），但在 5G 下受 S1 网络延迟（26.0 ms）拖累仅提升 1.05×（37.8 Hz）；Thor 端侧在 10 Hz 限频下为 24.7 Hz（1.30×）。
7. **论文自述局限（§5）：** 聚焦机械臂/双臂操作任务，未覆盖需要机载执行与 SLAM 等模块的自动驾驶、四足或无人机；未建模机械执行延迟与相机传感采集延迟；Roofline 假设理想算子执行且异步/双系统不联合建模任务成功率退化。

- **对 wiki 的映射：** [How Fast Can I Run My VLA? Demystifying VLA Inference Performance with VLA-Perf](../../wiki/entities/paper-agentic-2602-18397.md)
