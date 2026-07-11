# ARDY: Autoregressive Diffusion with Hybrid Representation for Interactive Human Motion Generation

> 来源归档（ingest）

- **标题：** ARDY: Autoregressive Diffusion with Hybrid Representation for Interactive Human Motion Generation
- **类型：** paper
- **来源：** ACM Transactions on Graphics（SIGGRAPH 2026）
- **原始链接：**
  - 项目页：<https://research.nvidia.com/labs/sil/projects/ardy/>
  - DOI：<https://doi.org/10.1145/3811284>
- **机构：** NVIDIA Research · ETH Zürich
- **入库日期：** 2026-07-11
- **一句话说明：** **自回归扩散** 交互式人体运动生成框架：在 **交互速度** 下支持 **流式文本提示** 与 **灵活长时域运动学约束**（根部路径/路点、全身关键帧、稀疏关节位姿/旋转），用 **混合表示**（显式 root + 潜空间 body）与 **两阶段自回归 Transformer 去噪器** 兼顾轨迹可控与生成效率。

## 核心论文摘录（MVP）

### 1) 动机：离线可控 vs 在线实时 的鸿沟

- **链接：** <https://research.nvidia.com/labs/sil/projects/ardy/>
- **摘录要点：** 交互应用（动画、仿真、人形机器人）需要 **实时** 3D 人体运动合成。离线方法如 [Kimodo](https://research.nvidia.com/labs/sil/projects/kimodo/) 可用文本与运动学约束做高质量可控生成，但 **推理速度不足以交互**；既有在线方法虽能实时，却常 **牺牲可控性**，或在复杂文本语义与 **长时域目标** 上受限于 **有限上下文窗口**。
- **对 wiki 的映射：**
  - [ARDY](../../wiki/entities/ardy.md) — 「为什么重要」与离线/在线谱系定位
  - [Kimodo](../../wiki/entities/kimodo.md) — 离线高质量可控扩散对照锚点

### 2) 混合表示 + 两阶段自回归去噪器

- **链接：** <https://research.nvidia.com/labs/sil/projects/ardy/>
- **摘录要点：**
  - **混合表示**：patch 化 body 运动经编码器得 **潜空间 body embedding**，与 patch 化 **全局 root** 拼接；解码重建 body。兼顾 **可解释的全局轨迹控制** 与 **低维潜空间高效生成**。
  - **Motion Tokenizer**：编码–解码 hybrid token，平衡场景空间约束与生成学习效率。
  - **两阶段自回归 Transformer 去噪器**：在 **可变长度历史上下文** 下，于当前生成窗口预测 **C 个干净 motion token**；**先预测 root，再以 root 条件预测 body latent**（交错两阶段，保持保真同时满足文本与空间控制）。
  - **长时域约束**：运动学约束可在时间/关节上 **稀疏**，且可落在 **当前生成窗口之外**（如远期路点），支持长程目标到达。
- **对 wiki 的映射：**
  - [ARDY](../../wiki/entities/ardy.md) — 架构、Mermaid 流程与约束类型表
  - [Diffusion-based Motion Generation](../../wiki/methods/diffusion-motion-generation.md) — 交互式扩散生成范式补充

### 3) 训练与条件：文本 + 从 GT 采样的运动学约束

- **链接：** <https://research.nvidia.com/labs/sil/projects/ardy/>
- **摘录要点：** 在 **大规模动捕** 上训练；**直接以文本标签** 与 **从真值姿态采样的运动学约束** 为条件，原生学习 **在线 prompt** 与 **灵活长时域目标** 的可控生成。
- **对 wiki 的映射：**
  - [ARDY](../../wiki/entities/ardy.md) — 训练条件与评测叙事
  - [BONES-SEED / Rigplay 生态](../../wiki/entities/kimodo.md) — 同 NVIDIA 人形运动数据栈

### 4) 交互演示与人形下游：ARDY + SONIC on G1

- **链接：** <https://research.nvidia.com/labs/sil/projects/ardy/>
- **摘录要点：**
  - **交互 Demo**：动态文本控制、关键帧约束、路径跟随、**实时 locomotion**（鼠标路点 + 键盘速度指令）。
  - **人形机器人**：ARDY 实时人形运动生成 + **GEAR SONIC** 物理跟踪策略 → **Unitree G1** 交互控制（流式约束与用户输入）。
- **对 wiki 的映射：**
  - [SONIC](../../wiki/methods/sonic-motion-tracking.md) — 生成→跟踪闭环
  - [Unitree G1](../../wiki/entities/unitree-g1.md) — 演示平台

## 引用（项目页 BibTeX）

```bibtex
@article{zhao2026ardy,
  title   = {ARDY: Autoregressive Diffusion with Hybrid Representation for Interactive Human Motion Generation},
  author  = {Zhao, Kaifeng and Petrovich, Mathis and Zhang, Haotian and Wang, Tingwu and Tang, Siyu and Rempe, Davis},
  journal = {ACM Transactions on Graphics (TOG)},
  year    = {2026},
  volume  = {45},
  number  = {4},
  articleno = {86},
  doi     = {10.1145/3811284}
}
```
