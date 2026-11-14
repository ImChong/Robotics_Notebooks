# FlashVLA（流式异步动作解码）

> 来源归档（ingest）

- **标题：** FlashVLA: Streaming Action Decoding for Fast and Asynchronous VLA Inference
- **类型：** paper
- **原始链接：** <https://arxiv.org/abs/2608.27384>
- **机构：** 加州大学圣地亚哥分校（UCSD）；麻省理工学院（MIT）
- **作者：** Zekai Li、Jiaming Tang、Zhijian Liu
- **项目页 / 博客：** <https://z-lab.ai/projects/flashvla/>
- **代码：** <https://github.com/z-lab/flashvla>
- **权重：** <https://huggingface.co/z-lab/flashvla-pi05-libero>（另有 `flashvla-pi05-robotwin`；集合 <https://huggingface.co/collections/z-lab/flashvla>）
- **入库日期：** 2026-08-29
- **核对更新：** 2026-11-14
- **通读：** 已完成
- **核对范围：** 全文 §1–§6（含 Tab. 1–6、Fig. 1–5）、参考文献及附录 A.1–A.4（含 Fig. 6–8、Algorithm 1），并核对 arXiv abs 页元数据（arXiv:2608.27384v1，2026-08-27 提交）。
- **一句话说明：** 对流匹配 VLA 做流式 chunk 解码：交错噪声缓冲 + chunk 级因果注意力，一次前向吐一块可执行动作；LIBERO 异步 2.43×、真机单卡 ≥30 Hz。

## 核心摘录

### 1) 流匹配 VLA 的延迟与异步错配同源（§1–§2）

- **摘录要点：** \(\pi_{0.5}\) 在 RTX 4090（双视角、无系统优化基准）上，10 步串行动作解码占单步推理时间的 **75%**（Fig. 1b）。同步推理在 chunk 边界停机；异步推理在执行当前动作窗期间以陈旧观测启动下次预测，lookahead 越大，预测态与到达状态越错配。两者共同根因是 **每个动作块从纯噪声孤立解码**——既将全部去噪步压入单次调用，又让新块对在飞轨迹无感知。
- **对 wiki 的映射：**
  - [FlashVLA](../../wiki/entities/paper-flashvla.md) — 问题设定。
  - [\(\pi_{0.5}\)](../../wiki/entities/paper-pi05-open-world-vla.md) — 主实验基座策略。

### 2) 交错噪声缓冲 + chunk 级因果注意力 + 多缓冲联合微调（§3 / App. A.3–A.4 / Alg. 1）

- **摘录要点：** 维护包含 \(N\) 个动作块的流式缓冲 \(\mathbf{B}_t=[\mathbf{x}^{(1)}_{\tau_1},\ldots,\mathbf{x}^{(N)}_{\tau_N}]\)，噪声单调递增 \(\tau_1<\cdots<\tau_N\)（Eq. 2 采样 \(u_i\sim\mathrm{Beta}(1.5,1.0)\)，\(\tau_i=0.001+0.999\frac{i-1+u_i}{N}\)）。chunk 级因果掩码只允许高噪未来块 attend 低噪近执行块、禁止反向，使单次前向将整缓冲推进一级（`dt = 1/N`）：稳态每步弹出最前端干净块并尾部补纯噪声（Alg. 1）；冷启动支付 \(N{-}1\) 步暖机并执行安全默认动作（零增量或当前关节角）。训练阶段将同一观测 \(o_t\) 的全部 \(N\) 种冷启动填充状态打入单样本（Eq. 3，掩码隔离各缓冲但共享 \(o_t\) 编码），并用 FiLM 强化多级时间步条件；多任务微调偶发梯度爆炸时，重初始化动作专家归一化层并保留其余权重。经验建议总缓冲跨度 \(N\times C\) 贴近预训练原生 chunk 长（\(\pi_{0.5}\) 为 50）。
- **对 wiki 的映射：**
  - [FlashVLA](../../wiki/entities/paper-flashvla.md) — 方法、系统编译与消融。
  - [VLA](../../wiki/methods/vla.md) — 流匹配动作头加速。

### 3) 仿真与真机实验数字（§4–§5 / App. A.1–A.3）

- **摘录要点：**
  - **LIBERO（Tab. 1 & 3，5 次运行均值 2,000 回合，\(C{=}10, N{=}4, E{=}5\)）：** \(\pi_{0.5}\) 同步 96.9%（53.8 ms/step）；FlashVLA \(d{=}0\) **97.9%**（Long 92.4%→96.2% +3.8），\(d{=}1\) **97.8%**、**22.1 ms/step（2.43×）**，\(d{=}2\) **98.3%**、**20.6 ms/step（2.62×）**，在 \(d{=}1..4\) 全程保持 97.5–98.3%；对照 VLASH \(d{=}1\) 97.2%（46.8 ms, 1.15×）在 \(d{=}4\) 跌至 93.1%，StreamingVLA \(d{=}1\) 94.9%（31.6 ms, 1.70×）。去掉因果掩码仅留缓冲，异步成功率约 **-10 pt**（Fig. 6）。
  - **RoboTwin 2.0 50 任务多任务（Tab. 4 & Fig. 3，\(C{=}20, N{=}4, E{=}16\)）：** \(\pi_{0.5}\) 同步 86.0%（47.4 ms/step，其中仿真渲染占 36.8 ms、通信 2.5 ms、策略 8.1 ms）；FlashVLA \(d{=}0\) 90.5%，\(d{=}1\) **90.6%**（43.8 ms/step），\(d{=}4\) 89.8%（vs VLASH \(d{=}4\) 85.1%）。按视界分组同步结果：短程 93.9%→93.2%（-0.7）、中程 81.1%→85.4%（+4.3）、长程 53.0%→**89.6%（+36.6）**。
  - **推理延迟与反应速度（Tab. 2 & Fig. 4，同步 CUDA Graph + 算子融合）：** 双视角/三视角 RTX 4090 由 \(\pi_{0.5}\) 45.8/55.4 ms 降至 **26.7/36.8 ms**（Realtime-VLA 29.2/38.9 ms）；RTX 5090 由 37.0/44.8 ms 降至 **20.3/27.1 ms**（20.3 ms ≈ 50 Hz）；无优化 profile 动作解码墙钟加速 **19.9×**（GPU 计算降 9.3× + 消除 pass 间发射序列化）、总前向 **3.5×**（Fig. 1b）。30 Hz 目标下 TTFA **37.1 ms**（vs \(\pi_{0.5}\) 80.0 / FASTER 62.1，1.7× FASTER）、期望 TTR **70.4 ms**（vs 130.0 / 112.1，1.6× FASTER）。
  - **跨架构泛化（Tab. 5）：** SmolVLA（LIBERO, \(C{=}10, N{=}5\)）\(d{=}0\) 保 80.1% 且推理 19.7→10.1 ms（1.95×），\(d{=}1\) 79.5%；LingBot-VLA 4B（RoboTwin, \(C{=}20, N{=}4\)）\(d{=}0\) 85.2%→88.6%（70.6→25.1 ms, 2.81×），\(d{=}1\) **89.3%**。
  - **真机 7-DoF Franka + 单卡 RTX A4000（§5 & App. A.2）：** 3 项任务各 50 条 Gello 遥操作演示、每任务 15 次评测（0/1/2 三分制），执行窗 16 步 + \(d{=}2\)（A4000 推理 **67.3 ms** ≈ 2 个 30 Hz 周期）实现 **30 Hz** 控制；三任务均分 **84.4%**（vs \(\pi_{0.5}\) 同步 80.0% / 朴素异步 75.6% / RTC 80.0%），成功回合完成时间平均加速 **1.3×**（vs 同步）/ **1.2×**（vs RTC）。
- **对 wiki 的映射：**
  - [FlashVLA](../../wiki/entities/paper-flashvla.md) — 评测与消融读法。
  - [Manipulation](../../wiki/tasks/manipulation.md) — 桌面操作语境。

### 4) 开源与复现边界（核对截至 2026-11-14）

- **摘录要点：** 算法源码、多缓冲训练、延迟基准及 LIBERO/RoboTwin 2.0 仿真评测（含 `z-lab/flashvla-pi05-libero` 与 `flashvla-pi05-robotwin` HF 权重）已按 Apache-2.0 开源；**Franka 真机控制节点与 50 条/任务 Gello 遥操作演示数据集未随仓库发布**（论文 §5 / App. A.2 仅给出指令文本与训练步数设定）。
- **对 wiki 的映射：**
  - [flashvla 仓库](../repos/flashvla.md)
  - [z-lab FlashVLA 博客](../sites/z-lab-flashvla.md)

## 当前提炼状态

- [x] 全文 §1–§6 与附录 A.1–A.4（含 Tab. 1–6、Alg. 1）已通读核对
- [x] 代码仓库源码、训练/评测脚本与开源边界已实测核对
- [x] wiki 映射：`wiki/entities/paper-flashvla.md`
