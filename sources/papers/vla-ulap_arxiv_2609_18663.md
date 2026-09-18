# VLA-ULAP（云端 VLA + 边缘 ULAP）

> 来源归档（ingest）

- **标题：** VLA-ULAP: Interleaving Cloud VLA Calls with Ultra-Lightweight Local Action Prediction at the Edge
- **类型：** paper
- **原始链接：** <https://arxiv.org/abs/2609.18663>
- **机构：** 东京大学（The University of Tokyo）（一作 Deyu Cao 等）
- **作者：** Deyu Cao、Ryuji Oi、Kosuke Matsushima、Yuxuan Pan、Ziheng Wang、Daichi Fujiki、Atsutake Kosuge
- **入库日期：** 2026-09-18
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
