# FabriVLA: A Lightweight Vision-Language-Action Model for Precise Multi-Task Manipulation（arXiv:2607.08575）

> 来源归档（ingest）

- **标题：** FabriVLA: A Lightweight Vision-Language-Action Model for Precise Multi-Task Manipulation
- **缩写：** **FabriVLA**
- **类型：** paper / vision-language-action / lightweight-vla / flow-matching / imitation-learning
- **arXiv：** <https://arxiv.org/abs/2607.08575>（v2，2026-07-10；PDF：<https://arxiv.org/pdf/2607.08575>）
- **代码：** <https://github.com/Youi-FabriX/FabriVLA>（Apache-2.0）
- **权重：** <https://huggingface.co/Youi-FabriX/FabriVLA>（`checkpoint_step_93000.pt`）
- **项目页：** 无独立 `*.github.io`；以 GitHub README + Hugging Face 为入口（截至 2026-07-23）
- **作者：** Shiyuan Yang\*、Borong Zhang\*、Jizheng Zhang\*、Zhijia Tao\*、Junfei Guo、Donglai Ran、Xu Bian†、Qingbiao Li†（\*共一；†通讯）
- **机构：** 澳门大学；Mese Technology Limited Co., Ltd.；优艾智合 Youibot Robotics / FabriX 团队
- **入库日期：** 2026-07-23
- **一句话说明：** **0.89B** 轻量 VLA：InternVL3.5-1B（保留 14 层）+ **gated self-attention** flow-matching 动作头 + **shallow VLM layer fusion（layer 6⊕14）**；在公开 **Evo-1 Meta-World** 演示集上 **单阶段联合微调**（无机器人预训练），MT50 **tier-average 90.0%** / episode-level **92.0%**。

## 开源状态（步骤 2.5）

- **已开源：** 训练/评测代码（GitHub）+ 93k 步 checkpoint（HF）；许可 Apache-2.0。
- **数据：** 使用公开 Evo-1 Meta-World 演示集（论文 §3）；README 要求配置为 LeRobot 布局。
- **项目页：** 无单独项目站；入口即仓库与 HF。

## 摘录 1：问题与动机（摘要 / §1）

- **痛点：** 数十亿参数 VLA 算力与推理延迟阻碍实时控制；需要在性能与效率间折中的轻量架构。
- **路线：** 受 [Evo-1](https://arxiv.org/abs/2511.04555) 启发，保留「亚十亿 InternVL + flow-matching 动作专家」骨架，但改为：**动作 token 间 gated self-attention**（门控从 0 缓开）与 **浅层–深层 VLM 特征融合**，并采用 **单阶段全模型联合优化**（相对 Evo-1 的两阶段语义保持配方）。
- **结果主张：** Meta-World MT50 上 **tier-avg 90.0%**，优于表中 LA4VLA（87.5%）、Evo-Depth（84.4%）、Evo-1（80.6%）等对照，且 **无 Robo-Pretrain**。

**对 wiki 的映射：** 与 [VLA](../../wiki/methods/vla.md) 轻量路线、[Evo-1](../../wiki/entities/paper-evo1-lightweight-vla.md) 对照；沉淀 [`wiki/entities/paper-fabrivla.md`](../../wiki/entities/paper-fabrivla.md)。

## 摘录 2：架构与训练（§2–§3）

- **骨干：** InternVL3.5-1B；图像 **448×448**；保留 **14** transformer 层；状态 \(\mathbf{s}\in\mathbb{R}^{24}\) 经 MLP → 1024-d token 前置。
- **动作编码：** 噪声动作块 \(\mathbf{x}_t\in\mathbb{R}^{50\times 24}\)（horizon 50）；`MultiEmbodimentActionEncoder` + horizon 位置编码。
- **动作头：** \(L=8\) 块；每块：**gated self-attn（\(g\) 初值 0）→ cross-attn 到 VLM context → FFN + 时间嵌入**；输出速度场 \(\mathbf{v}\in\mathbb{R}^{50\times 24}\)。
- **Shallow fusion：** layer **6** 与 layer **14** concat → 线性投影 \(2048\to 1024\)，初始化 \([\mathbf{I}|\mathbf{0}]\)，约 **+2.1M** 参数。
- **Flow matching：** \(\mathbf{x}_t=(1-t)\boldsymbol{\epsilon}+t\mathbf{a}\)，\(t\sim\mathrm{Beta}(2,2)\)；损失 \(\|\mathbf{v}_\theta-(\mathbf{a}-\boldsymbol{\epsilon})\|^2\)；推理 **N=50** 欧拉积分；闭环 receding horizon。
- **训练：** Evo-1 公开 MT50 演示（50 traj/任务，共 2500）；**100k** 步、**B=40**、5× RTX PRO 6000；AdamW \(2\times10^{-5}\)，DeepSpeed ZeRO-2 + **FP32 master weights**；评测用 **93k** checkpoint（100k 略降）。

**对 wiki 的映射：** 实体页画流程与源码时序；强调「单阶段 + FP32 master」工程细节。

## 摘录 3：实验与消融（§4）

| 模型 | Params | Robo-Pre. | Easy | Med. | Hard | V.Hard | Tier Avg. |
|------|--------|-----------|------|------|------|--------|-----------|
| TinyVLA | 1.3B | No | 77.6 | 21.5 | 11.4 | 15.8 | 31.6 |
| π₀ | 3.5B | Yes | 71.8 | 48.2 | 41.7 | 30.0 | 47.9 |
| SmolVLA | 2.3B | No | 87.1 | 51.8 | 70.0 | 64.0 | 68.2 |
| Evo-1 | 0.8B | No | 89.2 | 76.8 | 77.2 | 79.2 | 80.6 |
| Evo-Depth | 0.9B | No | 83.1 | 84.7 | 87.3 | 82.4 | 84.4 |
| LA4VLA | 1B | MixPT | 88.9 | 94.5 | 66.7 | 100.0 | 87.5 |
| **FabriVLA** | **0.89B** | **No** | **95.0** | **88.2** | **86.7** | **90.0** | **90.0** |

- **评测协议：** 每任务 10 episode、seed 4042、400 步上限；tier-avg = 四难度均值；overall episode-level = 500 episode 成功比 → **92.0%**。
- **消融：** shallow fusion 相对 deep-only：tier-avg **82.9% → 90.0%**；冻结 VLM 的廉价消融中 **gated SA** 是动作头决定性模块（TR/TC 叠加无收益）。
- **弱项桶：** tool-mediated、coarse reaching/transport、grasp&place 相对较低。

**对 wiki 的映射：** 与 [VLA SOTA Leaderboard](../../wiki/entities/vla-sota-leaderboard.md) Meta-World 条目互证；更新 [Evo-1](../../wiki/entities/paper-evo1-lightweight-vla.md) 对照表。

## 建议 wiki 动作

- 新建 **`wiki/entities/paper-fabrivla.md`**（含流程与源码时序图）。
- 新建 **`wiki/entities/vla-sota-leaderboard.md`**（同步 ingest 榜站）。
- 更新 **`wiki/methods/vla.md`**、**`wiki/entities/paper-evo1-lightweight-vla.md`**、**`wiki/overview/vla-open-source-repro-landscape-2025.md`**、**`wiki/tasks/manipulation.md`**、**`wiki/queries/embodied-eval-benchmark-selection-loop.md`**。
