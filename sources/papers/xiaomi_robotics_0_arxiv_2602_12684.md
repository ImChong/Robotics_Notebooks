# Xiaomi-Robotics-0: An Open-Sourced Vision-Language-Action Model with Real-Time Execution（arXiv:2602.12684）

> 来源归档（ingest）

- **标题：** Xiaomi-Robotics-0: An Open-Sourced Vision-Language-Action Model with Real-Time Execution
- **类型：** paper（arXiv 技术报告）
- **arXiv：** <https://arxiv.org/abs/2602.12684>（v1 提交 2026-02-13，v2 修订 2026-03-25，`cs.RO`）
- **作者：** Rui Cai, Jun Guo, Xinze He, Piaopiao Jin, Jie Li, Bingxuan Lin, Futeng Liu, Wei Liu, Fei Ma, Kun Ma, Feng Qiu, Heng Qu, Yifei Su, Qiao Sun, Dong Wang, Donghao Wang, Yunhong Wang, Rujie Wu, Diyun Xiang, Yu Yang, Hangjun Ye, Yuan Zhang, Quanyun Zhou（Xiaomi Robotics）
- **代码 / 权重：** <https://github.com/XiaomiRobotics/Xiaomi-Robotics-0>（仓库归档见 [xiaomi-robotics-0.md](../repos/xiaomi-robotics-0.md)）；HuggingFace Collection：<https://huggingface.co/collections/XiaomiRobotics/xiaomi-robotics-0>
- **项目页：** <https://robotics.xiaomi.com/xiaomi-robotics-0.html>（`xiaomi-robotics-0.github.io` 自动跳转至此页；归档见 [xiaomi-robotics-website.md](../sites/xiaomi-robotics-website.md)）
- **入库日期：** 2026-11-14
- **通读：** 已完成
- **核对范围：** `arXiv:2602.12684v2` 全文（Abstract、Sec. 1–5、Table 1–6、Fig. 1–6、Appendix A–C 定性与 VL 基准细节），并核对 HTML 版表格与 `abs` 版本提交历史
- **一句话说明：** 4.7B 参数 MoT 架构 VLA（`Qwen3-VL-4B-Instruct` + 16 层 DiT），经 ~200M 步机器人轨迹与 >80M VL 样本两步预训练、异步执行专项后训练（RoPE 偏移、Λ 形局部因果掩码、前缀随机遮蔽、在线 L1 误差重加权）及 30Hz 同步重采样，兼顾通用 VLM 语义保留、高仿真分数与 RTX 4090 上 80ms 推理延迟的实时双臂操控。

---

## 一、数据配方（Sec. 2.1, Fig. 2）

| 模态 | 规模 | 来源与任务配比 |
|---|---|---|
| **机器人轨迹** | 约 **200M** 时间步（Fig. 2 标明 204.9M） | 开源跨本体数据集（含 DROID、MolmoAct 等）+ 内部双臂遥操作：**Lego Disassembly 338h**、**Towel Folding 400h** |
| **视觉–语言（VL）** | **>80M** 样本（Fig. 2 标明 82.3M） | 通用 VL 语料 + 机器人衍生 VL 语料（腕部/第一人称视角）；覆盖 4 类任务：Visual Grounding（Grounded SAM + Grounding DINO 1.5 + LLMDet 交叉共识）、VQA、Image Captioning、Embodied Reasoning & Planning（EQA、高层规划、点轨迹预测） |

---

## 二、模型与训练架构（Sec. 2.2）

- **总体架构：** Mixture-of-Transformers（MoT），总参数量 **4.7B**。VLM 为 `Qwen3-VL-4B-Instruct`，DiT 共 **16 层**，条件注入取自 VLM **最后 16 层** 的 KV Cache，并通过 **adaLN** 注入 flow-matching 时间步 $\tau$。
- **预训练 Stage 1（Action-capable VLM 共训）：**
  - 采用 **Choice Policies**：输入序列 $[o_t, l, s_t, [A_1], \dots, [A_T], [S]]$，预测 $N$ 组 $T$ 步候选动作块及各自评分；以预测动作块与真值的 $L_1$ 距离监督分数，动作预测采用 **winner-takes-all**（仅对 $L_1$ 最小候选反传）。
  - 为防止 VLM 灾难性遗忘，机器人轨迹（Choice Policies）与 VL 数据（next-token prediction）按 **1 : 6**（VL : Robot）比例混训。
- **预训练 Stage 2（冻结 VLM 训练 DiT）：**
  - 冻结 Stage 1 后 VLM 作为多模态特征条件器（输入仅含 $o_t, l$，不再使用 Stage 1 的 $[A_i]/[S]$ token），从头训练 16 层 DiT，以 rectified flow / flow matching 损失监督：$\tilde{a}_{t:t+T}^\tau = \tau a_{t:t+T} + (1-\tau)\epsilon$，$\tau \in [0, 0.999]$ 从 Beta 分布采样（偏向高噪声段）。
  - DiT 输入序列：`[SINK], s_t, \tilde{a}_t, ..., \tilde{a}_{t+T-1}`，加入可学习 `[SINK]` token 稳定注意力分布，采用因果/局部因果注意力。
  - Pre-train 共 40k steps，batch size 32,768，AdamW + DeepSpeed ZeRO-2。
- **异步后训练与时间协同捷径抑制（Sec. 2.2.2）：**
  - **问题：** 训练期 Training RTC 将 $\Delta t_c$ 步已提交干净动作作为前缀拼接到 DiT 输入（`[SINK], s_t, a_t..a_{t+\Delta t_c-1}, \tilde{a}_{t+\Delta t_c}..\tilde{a}_{t+T-1}`），后期动作 token 容易走「直接抄前缀」的时序捷径，忽视视觉观测（如叠毛巾翻折抓起多层时陷入死循环）。
  - **四项组合对策：**
    1. **RoPE 位置索引偏移：** 为 noisy action tokens 的位置 ID 增加固定偏移 `+10`，使模型明确区分干净动作前缀与待去噪动作段。
    2. **Λ 形（Λ-Shape）局部因果注意力掩码（Fig. 4）：** 紧随前缀的 noisy token 可在窗口 $w$ 内看到动作前缀以平滑过渡，更靠后的 noisy token 看不到早期前缀，被迫关注 VLM KV Cache 与本体状态。
    3. **前缀随机遮蔽（Random Masking）：** 训练时保留前缀末尾若干步以维持边界连续，对其余前缀步随机掩码。
    4. **在线误差自适应损失重加权（Adaptive Loss Re-weighting）：** 训练中随机采样 $\Delta t_c \in \{0, 1, \dots, 6\}$；当 $\Delta t_c > 0$ 时先在线前向推导动作轨迹并与真值算 $L_1$ 误差，按偏差动态放大 flow-matching 损失权重。
  - 后训练超参：batch size 2,048，Lego Disassembly 40k steps、Towel Folding 80k steps，chunk 长度 $T=30$（对应 30Hz 下 1 秒）。

---

## 三、部署方案（Sec. 2.3, Fig. 5）

- **推理积分：** 标准高斯初始噪声下进行 **5 步** flow-matching 欧拉积分（$\tau: 0 \to 1$）。
- **实时性：** 单张 **RTX 4090** 上单次推理延迟 **$\Delta t_{\text{inf}} = 80\,\text{ms}$**；多路相机图像与本体状态按时间戳重采样统一对齐到 **30Hz** 时钟，每个 tick 取最近邻观测输入，消除异步传感漂移。
- **异步流水线：** 执行当前 chunk 前 $T_e$ 步后触发下一次推理；条件前缀取 $[T_e, T_e + \Delta t_c - 1]$，约束 **$\Delta t_c \ge \Delta t_{\text{inf}}$**，使推理完成时新 chunk 恰好从 $\Delta t_{\text{inf}}$ 时刻无缝接管。

---

## 四、核心实验结果（Sec. 3.1–3.3, Table 1–6）

### 1. LIBERO（Table 1，成功率 %）

| 方法 | Spatial | Object | Goal | Long | Average |
|---|---|---|---|---|---|
| OpenVLA-OFT | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| $\pi_0$ | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| $\pi_{0.5}$ | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| FLOWER | 97.5 | 99.1 | 96.1 | 94.9 | 96.9 |
| EO-1 | **99.7** | 99.8 | **99.2** | 94.8 | 98.2 |
| **Xiaomi-Robotics-0** | 98.8 | **100.0** | 98.8 | **97.2** | **98.7** |

### 2. CALVIN（Table 2，连续完成 1–5 项任务及 Avg. Len.）

| 方法 | Setting | 1 | 2 | 3 | 4 | 5 | Avg. Len. |
|---|---|---|---|---|---|---|---|
| UniVLA | ABCD→D | 98.5% | 96.1% | 93.1% | 89.9% | 85.1% | 4.63 |
| FLOWER | ABCD→D | 99.2% | 96.9% | **96.9%** | 92.3% | 88.3% | 4.67 |
| **Xiaomi-Robotics-0** | **ABCD→D** | **99.7%** | **98.0%** | 96.7% | **94.2%** | **91.8%** | **4.80** |
| UniVLA | ABC→D | 98.9% | 94.8% | 89.0% | 82.8% | 75.1% | 4.41 |
| FLOWER | ABC→D | 99.4% | 95.8% | 90.7% | 84.9% | 77.8% | 4.53 |
| **Xiaomi-Robotics-0** | **ABC→D** | **100.0%** | **98.3%** | **96.0%** | **92.6%** | **88.1%** | **4.75** |

### 3. SimplerEnv（Table 4–5，Overall 成功率）

- **WidowX（Table 4）：** Put Spoon on Towel **95.8%**、Put Carrot on Plate **62.5%**、Stack Blocks **75.0%**、Put Eggplant in Basket **83.3%**，**Overall 79.2%**（高于 EO-1 的 72.7% 与 $\pi_0$ 的 69.2%）。
- **Google Robot Visual Matching（Table 5）：** Pick Coke Can **98.7%**、Move Near **88.8%**、Open/Close Drawer **79.6%**、Drawer Apple **75.0%**，**Overall 85.5%**（EO-1 76.5%、$\pi_0$ 71.4%）。
- **Google Robot Visual Aggregation（Table 5）：** Pick Coke Can 88.2%、Move Near 76.8%、Open/Close Drawer 67.2%、Drawer Apple **66.7%**，**Overall 74.7%**（EO-1 63.0%、$\pi_0$ 54.7%）。

### 4. VLM 与具身推理能力保留（Table 3，10 项基准共约 40.5k 样本）

| 模型 | ERQA | SEED | POPE | AI2D | MMBench | MME | MMMU | TextVQA | SciQA | ChartQA |
|---|---|---|---|---|---|---|---|---|---|---|
| $\pi_0$ | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.1 | 0.1 | 1.4 | 0.0 | 0.0 |
| $\pi_{0.5}$ | 0.0 | 21.5 | 0.0 | 14.4 | 22.1 | 0.0 | 19.9 | 0.0 | 28.0 | 0.5 |
| MolmoAct | 33.5 | 72.7 | 86.6 | 72.0 | 80.1 | 69.5 | 38.0 | 67.3 | 91.1 | 57.1 |
| XR-0 (w/o VL data) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| **Xiaomi-Robotics-0** | **40.8** | 78.6 | 88.5 | 78.7 | 84.4 | 81.8 | 46.2 | 72.0 | 79.4 | 59.2 |
| Qwen3-VL-4B-Instruct | 40.0 | 78.8 | 89.7 | 81.6 | 88.7 | 87.1 | 51.7 | 78.0 | 92.7 | 76.8 |

> 关键读点：不混训 VL 数据的 XR-0 在 10 项 VL 基准全部归零（0.0），而 XR-0 预训练后在 **ERQA（40.8 vs 40.0）与 SEED（78.6 vs 78.8）等几乎持平甚至略超原始 `Qwen3-VL-4B-Instruct`**。

### 5. 双臂真机评测（Sec. 3.2, Fig. 6）

- **Lego Disassembly（LA-5/10/20 与 MA-34 砖块分拣）：** 同步变体（$\pi_{0.5}$、XR-0 Sync）成功率略高（异步反应滞后更易产生高张力弹出积木），但异步 XR-0 在吞吐量（正确分拣块数/分钟）上高于 $\pi_{0.5}$、XR-0 (Sync) 与基线 XR-0 (Training RTC)。
- **Towel Folding（两轮 30 分钟连续 rollout，超 2 分钟判失败）：** $\pi_{0.5}$、XR-0 (Sync) 与 XR-0 (Training RTC) 吞吐量均为约 **1 pcs/min**，完整 **Xiaomi-Robotics-0 达到 1.2 pcs/min**，避免 Training RTC 抓到多层毛巾后反复抄前缀抖动卡死的现象。
