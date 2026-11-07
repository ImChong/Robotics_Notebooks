# VLA-ULAP（云端 VLA + 边缘 ULAP）

> 来源归档（ingest）

- **标题：** VLA-ULAP: Interleaving Cloud VLA Calls with Ultra-Lightweight Local Action Prediction at the Edge
- **类型：** paper
- **原始链接：** <https://arxiv.org/abs/2609.18663>
- **机构：** 东京大学（The University of Tokyo）（一作 Deyu Cao 等）
- **作者：** Deyu Cao、Ryuji Oi、Kosuke Matsushima、Yuxuan Pan、Ziheng Wang、Daichi Fujiki、Atsutake Kosuge
- **入库日期：** 2026-09-18
- **复核日期：** 2026-11-07（重新取得 arXiv 当前版 PDF，36 页）
- **核对范围：** 摘要、引言要点，以及附录里的 A6000–Jetson 成本模型与 Table 4（Orin Nano 实测）、Table 5（A6000 与 Orin Nano Super 规格）；方法、仿真实验、云–边调度附录**未读**。
- **通读：** 未完成
- **一句话说明：** **~7.4M** 参数 **ULAP** 在边缘单 pass 预测 action chunk，与远程 VLA 交错调用；Jetson Orin Nano **19.9 ms / 0.183 J** vs GR00T A6000 **284.3 ms / 50.55 J**；三仿真对保留 **95–97.5%** baseline SR 同时减 **48.8–76.7%** VLA 调用。

## 核心摘录（MVP）

### 1) 云–边交错推理

- **摘录要点：** 十亿参数 VLA 需高 onboard 功耗；纯远程推理受通信延迟限制。VLA-ULAP 用当前视图、本体与已执行动作历史，由 ULAP 独立训练（不需 VLA hidden state、在线验证或 server 往返）。
- **对 wiki 的映射：**
  - [VLA-ULAP](../../wiki/entities/paper-vla-ulap.md) — 系统设定
  - [VLA](../../wiki/methods/vla.md) — 部署与加速对照

### 2) 效率与成功率 operating point

- **摘录要点：** Orin Nano 上 ULAP **19.9 ms**；选定 operating point 在三组 base-policy/benchmark 上保留 **95.0–97.5%** baseline SR，VLA 调用减 **48.8–76.7%**。相对 VLA-JEPA 上 ACT/SP-VLA，成功 episode 级估计 time/energy 更低。
- **对 wiki 的映射：**
  - [VLA-ULAP](../../wiki/entities/paper-vla-ulap.md) — 数字读法
  - [APXInf](../../wiki/entities/apxinf.md) — 另一条端侧 VLA 加速轴

### 3) 真机 SO-101

- **摘录要点：** seen / held-out placement 保留 **95.2–100%** baseline SR；推理时间降 **47.9–58.0%**，推理设备能耗降 **52.1–62.5%**（基于成功 episode 调用计数与实测 device cost）。
- **对 wiki 的映射：**
  - [VLA-ULAP](../../wiki/entities/paper-vla-ulap.md) — 真机

### 4) 动态任务与 latency-aware 仿真

- **摘录要点：** latency-aware LIBERO-Safety 上 VLA-ULAP 较 π₀.₅ 两任务分别 **+11.0 / +15.5 pp**，约减半 VLA 调用。
- **对 wiki 的映射：**
  - [VLA-ULAP](../../wiki/entities/paper-vla-ulap.md) — 动态场景

### 5) 开源状态（截至 2026-09-18）

- **摘录要点：** **截至入库日 arXiv v1 未列 GitHub 或项目页**。
- **对 wiki 的映射：**
  - [VLA-ULAP](../../wiki/entities/paper-vla-ulap.md) — 局限节

## 当前提炼状态

- [x] arXiv 摘要已对齐
- [x] 无代码链接
- [x] wiki 映射：`wiki/entities/paper-vla-ulap.md`

## 2026-11-07 部署相关核对（当前版 PDF）

- **与早先摘录的差异：** 当前版摘要称 ULAP（含冻结视觉编码器约 **7.4M** 参数）在 Jetson Orin Nano 上每次推理 **20.7 ms、0.122 J**（idle 扣除后的能耗），GR00T N1.7 在 RTX A6000 上 **289.3 ms、40.46 J**；早先归档的 19.9 ms / 0.183 J / 284.3 ms / 50.55 J 为旧版数字。VLA 调用减少 45.3–77.3%，保留 95.0–98.5% 基线成功率（四个仿真基线/基准对，选定工作点）。
- **测量条件（附录）：** 300 次同步调用均值（30 次预热后），Orin Nano 用 **15 W DVFS（非最高 Super 模式）、FP32/TF32 CUDA Graphs**；单模型 19.73 ms / 0.1140 J（活跃/空闲功率 11.01 / 5.69 W），ensemble 20.72 ms / 0.1221 J；A6000 为 bf16 计算 + FP32 参数、4 步去噪。Jetson 能耗取 VDD_IN 模组功率（含 CPU、GPU、DRAM），A6000 取 NVML 板级功率，二者统计口径不同。
- Table 5：A6000 48 GB GDDR6、768 GB/s、38.7 TFLOPS(FP32)、300 W；Orin Nano Super 8 GB LPDDR5、102 GB/s、67 TOPS（稀疏 INT8）、7/15/25 W、套件 US$249（2024-12）。
