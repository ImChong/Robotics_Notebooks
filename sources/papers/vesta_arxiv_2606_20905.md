# vesta_arxiv_2606_20905

> 来源归档（ingest）

- **标题：** Vesta: A Generalist Embodied Reasoning Model
- **类型：** paper
- **来源：** arXiv:2606.20905（2026-06）
- **机构：** NVIDIA 等（以 arXiv 作者脚注为准；含 Jan Kautz、Linxi "Jim" Fan、Yuke Zhu、K.R. Zentner 等）
- **入库日期：** 2026-06-23
- **一句话说明：** 在 **Qwen3-VL-8B** 上经 **空间导向 SFT 混合语料** 与 **极简多模态 memory harness**（历史帧 + 子任务文本缓存）统一 **定位 / 导航 / 具身推理 / 长时程子任务规划** 四类能力；四轴 benchmark 平均 **>20 pt** 超最强单基线、**>10 pt** 超 per-category oracle 集成；真机 **Gr00t-N1.6 actor + YAM 双臂** 记忆型任务成功率 **+38.3%**（相对 actor-only）。

## 核心论文摘录（MVP）

### 1) 问题：多 specialist 栈 vs 单一 generalist planner（Abstract / §1）

- **链接：** <https://arxiv.org/abs/2606.20905> §1
- **核心贡献：** 开放世界机器人需同时 **定位、空间推理、导航、长时程规划**；学界常把各能力拆成 **navigation / memory / reasoning specialist**，部署多模型栈带来 **延迟、集成复杂度与级联错误**。Vesta 主张这些能力应 **统一进单一 generalist planner**，而非 patchwork specialist ensemble。
- **对 wiki 的映射：**
  - [Vesta 论文实体](../../wiki/entities/paper-vesta-generalist-embodied-reasoning.md) — 问题定义与相对 RynnBrain / RoboBrain / Qwen3-VL 定位
  - [VLA](../../wiki/methods/vla.md) — planner VLM + actor VLA 分层栈上下文
  - [SayCan](../../wiki/methods/saycan.md) — 高层语言规划 + 低层执行的经典分层叙事

### 2) 四能力训练与 memory harness（§2）

- **链接：** <https://arxiv.org/abs/2606.20905> §2
- **核心贡献：**
  - **Localization**：base（Objects365 / COCO / LVIS）+ tail（egocentric / manipulation 序列）；点、框等结构化输出经 **同一 language head 解码为 text token**。
  - **Navigation**：R2R-style VLN-CE；输出 pixel goal（↓+归一化坐标）、turn 序列或 stop；数据来自 R2R / RxR / ScaleVLN 仿真渲染。
  - **Embodied reasoning**：VQA / detection / pointing 先验 + 具身 affordance、placement、轨迹 waypoint、任务进度估计。
  - **Action planning + memory**：给定目标 $g$，每步预测文本子任务 $a_t$；非 Markov → 用 memory harness 压缩历史 $s_t=\Phi(o_t,\mathcal{M}_t,g)$。每步 **CoT 四段**：Observation / Progress / Reasoning / Action（仅 $a_t$ 写入 memory）。Memory 元组 $m_i=\langle i,\tau_i,o_i,a_i,g\rangle$；历史图像 cap $K$ 帧（uniform 或 recency-biased，首帧必留）。
- **对 wiki 的映射：**
  - [Vesta 论文实体](../../wiki/entities/paper-vesta-generalist-embodied-reasoning.md) — Mermaid 流程与 memory 设计表
  - [Vision-Language Navigation](../../wiki/tasks/vision-language-navigation.md) — VLN-CE 与 R2R 指标语境

### 3) SFT 数据混合与训练配方（§3 / Figure 4）

- **链接：** <https://arxiv.org/abs/2606.20905> §3
- **核心贡献：** 六类混合（占比）：Spatial Intelligence **27.1%**、Navigation **21.8%**、Grounding **20.8%**、General VLM **16.2%**、Embodied Reasoning **9.8%**、Real Robots **4.3%**。1 epoch、lr **1e-5**、wd **0.01**、128×H100、batch **256**。
- **对 wiki 的映射：**
  - [Vesta 论文实体](../../wiki/entities/paper-vesta-generalist-embodied-reasoning.md) — 数据混合表与 generalist vs specialist 消融

### 4) Benchmark 与真机（§4）

- **链接：** <https://arxiv.org/abs/2606.20905> §4
- **核心贡献：**
  - **Table 1 Embodied**：Cognition avg **68.7** vs RynnBrain **64.8** / RoboBrain **56.6** / Qwen3-VL **55.7**；Localization avg **69.9** vs **61.9 / 69.4 / 57.3**。
  - **Table 2 离线 action planning**：AgiBot 五任务 + Egocentric-Human 60 任务；Vesta avg **75.4%** vs RoboBrain **38.5%**；160 episode、零样本、temporal IoU 评分。
  - **Table 3 R2R-CE**：SR **55.5%**（≈ InternVLA-N1 specialist **55.4%**）；generalist 竞品 RynnBrain/RoboBrain/Qwen3-VL **SR=0**（灾难性遗忘，总输出 →→）。
  - **Table 4 generalist vs specialist mix**：统一 mix 在 R2R **+1.4 SR**、embodied **+3.9 avg** 超各自 specialist-only。
  - **真机（§4.4）**：**Gr00t-N1.6** actor；YAM 双臂；Find Object / Count Fruits / Memorize Candy；Vesta planner 相对 actor-only **+38.3%**、相对 Qwen3-VL planner **+25%**（>4σ）。
  - **Table 5 memory 消融**：Image+Text 优于纯 Image 或纯 Text；transition 步 **2×** 过采样最优。
- **对 wiki 的映射：**
  - [Vesta 论文实体](../../wiki/entities/paper-vesta-generalist-embodied-reasoning.md) — 实验表摘要与局限
  - [Manipulation VLA 架构选型](../../wiki/queries/manipulation-vla-architecture-selection.md) — planner+actor 分层对照

### 5) 局限与 broader impacts（Appendix E–F）

- **局限：** 真机仅 **YAM 双臂 + 3 任务**；基座固定 **8B**；memory 为 **固定采样 + 文本 log**（无 learned retriever）；未在 humanoid / mobile manipulator 上压测 planner–actor 接口。
- **安全面：** hierarchical split 可在 **文本子任务** 层加规则/学习过滤器再下发 motor 命令。

## 对 wiki 的映射（汇总）

- [paper-vesta-generalist-embodied-reasoning.md](../../wiki/entities/paper-vesta-generalist-embodied-reasoning.md) — 主沉淀页
- 交叉更新：[vla.md](../../wiki/methods/vla.md)、[vision-language-navigation.md](../../wiki/tasks/vision-language-navigation.md)、[saycan.md](../../wiki/methods/saycan.md)

## 引用（arXiv BibTeX 摘要）

```bibtex
@article{bjorck2026vesta,
  title={Vesta: A Generalist Embodied Reasoning Model},
  author={Bjorck, Johan and Li, Zhiqi and Man, Yunze and Wang, Jing and Cheng, An-Chieh and Liu, Sifei and Wang, Shihao and Yu, Zhiding and others},
  journal={arXiv preprint arXiv:2606.20905},
  year={2026}
}
```

## 当前提炼状态

- [x] 摘要与核心方法摘录（≥5 条）
- [x] wiki 页面映射
- [ ] 官方代码/权重发布（截至 ingest 日未见公开 repo）
