---
type: source
title: "EventVLA: Event-Driven Visual Evidence Memory for Long-Horizon Vision-Language-Action Policies — 原文精读归档"
url: "https://arxiv.org/abs/2606.20092"
date: 2026-11-14
status: ingested
tags: [paper, vla, manipulation, long-horizon, memory, dual-arm, qwen, ustc, shanghai-ai-lab, sjtu, dalian-university-of-technology, huawei, hku, tsinghua, pku]
---

# EventVLA — 原文精读归档

- **论文：** *EventVLA: Event-Driven Visual Evidence Memory for Long-Horizon Vision-Language-Action Policies*
- **arXiv：** [2606.20092](https://arxiv.org/abs/2606.20092)（v3，CoRL 2026）
- **作者与机构：** Ganlin Yang、Zhangzheng Tu、Yuqiang Yang、Sitong Mao、Junyi Dong、Tianxing Chen、Jiaqi Peng、Jing Xiong、Jiafei Cao、Jifeng Dai、Wengang Zhou、Yao Mu、Tai Wang（中国科学技术大学 USTC、上海人工智能实验室 Shanghai AI Lab、上海交通大学 SJTU、大连理工大学 DUT、华为 Huawei、香港大学 HKU、清华大学 Tsinghua、北京大学 PKU）
- **项目页 / 代码：** <https://ganlin-yang.github.io/EventVLA.github.io/> · <https://github.com/InternRobotics/EventVLA>
- **核对范围：** 通读 `arXiv:2606.20092v3` 原文抽取文本（全文 2,042 行）：§1 引言、§2 相关工作、§3 方法（基础视觉锚点 $A_t$、前瞻式关键帧证据记忆 KEM、Qwen3-VL-235B 离线自动标注、平滑软标签与 teacher→student 课程、1D NMS + 冷却推理写入）、§4 RoboTwin-MeM 诊断基准（8 任务 × 50 演示，平均 430–1544 步，$n\in[1,5]$）、§5 实验（Table 1–4 跨 RMBench / RoboTwin-MeM / RoboTwin 2.0 / RoboMME / 真机与记忆基线对比）、§6 结论与局限，以及 Appendix A–D.2（Table 5–11 详细标注误差、任务分解、$H_k$ 消融与真机推理延迟/显存指标）。
- **通读：** 已完成

## 核心技术要点摘录

1. **视觉记忆分解 $M_t = A_t \cup E_t$：**
   - **基础视觉锚点（Foundational Visual Anchors, $A_t$）：** 由初始帧 $o_0$ 与短期历史滑窗 $\{o_{t-K}, \dots, o_{t-1}\}$ 组成（仿真采用 $o_0, o_{t-30}, o_{t-15}$；真机采用 $o_0, o_{t-60}, o_{t-40}, o_{t-20}$）。在布局持久可见的 RMBench 上，仅用 VA 即可达到 **67.8%**；若去除 $o_0$ 或短期历史滑窗，分别掉至 **33.7%** 与 **23.8%**。
   - **前瞻式关键帧证据记忆（Keyframe Evidence Memory, KEM, $E_t$）：** 轻量 MLP 预测头从动作 chunk 最后一层共享隐状态 $h_t \in \mathbb{R}^{H\times d}$（$H=50, H_k=50$）直接前瞻预测未来 chunk 内各步成为事件关键帧的概率 $\hat{\mathbf{p}}_t = \sigma(\mathrm{KEM}_{\mathrm{mlp}}(h_t)) \in [0,1]^H$。当 $\hat{p}_t^i \ge \tau_{\mathrm{commit}}=0.55$ 且通过 1D NMS（半径 $w=8$）与冷却窗口（$C=10$ 步）过滤时，当机器人后续走到对应时刻 $t+i$ 即把该步 **原始 RGB 帧** 写入容量为 $N_{\max}=5$ 的 FIFO 缓冲 $E_t$，避免隐式特征压缩带来的信息丢失。
2. **离线 VLM 自动标注与训练课程：**
   - 用 `Qwen3-VL-235B-A22B-Instruct-FP8`（8×A800 vLLM 部署，每轨迹均匀采样 128 帧多视角 + 2–3-shot ICL JSON 提示）离线自动标定关键帧时刻 $t^*$，与仿真器真值时序偏差 **<10 步**，与人类标注偏差 **<50 步**（在 1500–2000 步长轨迹上，Table 10 证明用自动标注训练反超手工 GT：`75.2%` vs `76.3%` in sim / `78.8%` vs `77.7%` in real）。
   - 采用升余弦软标签 $y_t^i = 0.5(1+\cos(\pi|t+i-t^*|/R))$ 计算 chunk-wise BCE 损失 $L = L_{\mathrm{action}} + \lambda L_{\mathrm{kem}}$（$\lambda=0.1$），并按课程将历史记忆中老师/预测混合比例 $\alpha$ 从 `1.0 -> 0.0` 退火，使策略学会自己“边走边挑、边用边存”。
3. **RoboTwin-MeM 基准与全量评测数据（v3 核对）：**
   - **RoboTwin-MeM（Table 2）：** 8 项非马尔可夫双臂任务（平均 430–1544 步，$n\in[1,5]$），$\pi_{0.5}$ `7.8%`、QwenOFT `3.8%`、MemER `10.5%`、Mem-0 `0.0%`、MemoryVLA(OpenVLA/QwenOFT) `4.9%` / `10.8%`、VA only `18.0%` → **VA+KEM 达到 `75.2%`**（分任务：Rearrange `62%`、Put Back `93%`、Pick Order `90%`、Pick Unhidden `54%`、Cover `94%`、Find Seal `63%`、Reproduce Route `98%`、Press Button `48%`）。
   - **跨基准（Table 1, 3, 4）：** RMBench（Table 1）VA only **67.8%**（VA+KEM **66.7%**，对比 MemoryVLA-QwenOFT `41.7%`、Mem-0 `42.0%`）；RoboTwin 2.0 马尔可夫基准（Table 3）Easy **83.8%** / Hard **81.6%**（超 $\pi_{0.5}$ `82.7%` / `76.8%`）；RoboMME（Table 4）VA+KEM **32.1%**（Oracle `40.5%`）。
   - **关键消融（Table 2, 4, 10）：** 隐式记忆库 `24.9%`、硬标签 `48.8%`、去 NMS `53.4%`、$N_{\max}=2$ `32.0%`；传统启发式写入（Random `10.1%` / Periodic `34.2%` / Visual-change `44.6%` / Reactive selector `28.9%`）均显著低于前瞻式 KEM；前瞻视野 $H_k$ 从 `0 / 1 / 15 / 30 / 50` 逐步提升，RoboTwin-MeM 分别为 **`28.9% / 28.2% / 36.8% / 45.4% / 75.2%`**，真机分别为 **`30.4% / 31.1% / 42.9% / 50.1% / 78.8%`**。
   - **真机 ARX ACONE（Table 11, A100 bs=1）：** 4 任务各 20 trials（Find Block Easy `90%`、Pick-X-Times `90%`、Find Block Hard `60%`、Pick in Order `75%`）均值 **78.8%**（对比 $\pi_{\mathrm{MEM}}$ `37.5%`、MemER `32.3%`），单 chunk 延迟 **0.42 s**、显存约 **7 GB**（MemER `1.43 s` / ~24 GB，$\pi_{\mathrm{MEM}}$ `1.08 s` / ~24 GB）。
4. **原文显式局限（§6）：** 有界 FIFO 容量（$N_{\max}=5$）在超过 10 分钟的高事件密度长程任务上仍可能饱和并过早驱逐早期证据；已习得记忆策略对全新未见任务、物体与强干扰物的零样本跨任务迁移尚未验证。
