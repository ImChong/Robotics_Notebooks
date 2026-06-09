# LIMMT: Less is More for Motion Tracking

> 来源归档（ingest）

- **标题：** LIMMT: Less is More for Motion Tracking
- **类型：** paper / humanoid / motion-tracking / data-curation
- **venue：** ICML 2026（项目页标注）
- **原始链接：**
  - arXiv：<https://arxiv.org/abs/2606.06953>
  - 项目页：<https://giraffeguan.github.io/limmt/>
- **机构：** 清华大学；GalBot；上海交通大学；北京大学；上海期智研究院
- **入库日期：** 2026-06-09
- **一句话说明：** 首篇面向 **物理人形 motion tracking** 的 **数据中心化** 研究：用 **GQS（General Quality Selection）** 三阶段管线（仿真可行性过滤 → HME 语义嵌入 → 复杂度加权 FPS）从 AMASS 等大规模库中选出 **≈3%** 子集，在 **Any2Track / TWIST2** 等 tracker 上 **plug-and-play** 优于全量数据；真机 **Unitree G1** 用 **10% GQS** 策略无微调部署。

## 核心论文摘录（MVP）

### 1) 问题：盲目堆 MoCap 为何常常无效

- **链接：** <https://arxiv.org/abs/2606.06953> §1
- **摘录要点：** 互联网级 / 视频重建动捕库常含 **脚滑、穿地、不可行接触、关节超速** 等系统性伪影，污染模仿信号并诱发 reward hacking；同时大规模库放大 **采样、课程与长程优化** 成本。作者认为 **数据质量塑造早期优化轨迹**——劣质或冗余片段会注入偏置梯度，使策略早早落入难恢复的吸引子。
- **对 wiki 的映射：**
  - [LIMMT / GQS](../../wiki/methods/limmt-gqs-motion-curation.md) — 「质量 > 数量」的问题定义

### 2) 三维质量定义与 GQS 分阶段设计

- **链接：** <https://arxiv.org/abs/2606.06953> §1–3；项目页 Method Overview
- **摘录要点：**
  - **可行性（physics feasibility）**：刚体仿真回放，复合分 $S_{phy}$；**悬浮、穿地、脚滑** 重罚，**自碰、jerk** 轻罚；保留 $S_{phy}\ge 90$。须 **先于** 表征学习，避免坏动作占据嵌入流形。
  - **多样性（action diversity）**：在可行集上用 **Periodic Autoencoder** 得 **HME（Harmonic Motion Embedding）**；全局描述子 $z_{global}=\mathrm{mean}([A,F])$ 相位不变，支撑 **行为覆盖** 而非 pose 欧氏冗余。
  - **复杂度（action complexity）**：由动能/加速度等物理强度打分；在 **Global Weighted FPS** 中作小偏置 $\alpha\cdot\hat D+(1-\alpha)\cdot\hat C$，在距离可比时偏好 **动态更丰富** 的监督。
  - **顺序约束**：过滤 → 嵌入 → 复杂度加权；颠倒顺序会让高能量伪影被误选。
- **对 wiki 的映射：**
  - [LIMMT / GQS](../../wiki/methods/limmt-gqs-motion-curation.md) — 三阶段 Mermaid 与机制表

### 3) AMASS 主实验：3% 击败全量

- **链接：** 项目页 Main Results on AMASS
- **摘录要点：**
  - 基线 tracker：**Any2Track**、**TWIST2**；对比 **Random 3%**、**PHC** 式过滤、**GQS** 各比例。
  - **Any2Track + GQS @ 3%**：SR **0.956**，MPJPE **0.108 rad**，MPKPE **29.87 mm**；优于 **Full Data（无 GQS）** 与 **Random 3%（SR 崩塌至 0.838）**。
  - **TWIST2 + GQS @ 3%**：SR **0.861**，MPJPE **0.092 rad**；同样显著优于 random 子集。
  - 项目页主张 **~15% MPJPE 改善**、**3% 数据即跨全量 SR 基线**；强调 effect 来自 **「对的数据」** 而非单纯减量。
- **对 wiki 的映射：**
  - [LIMMT / GQS](../../wiki/methods/limmt-gqs-motion-curation.md) — 实验结论与 tracker-agnostic 叙事

### 4) 消融、训练动力学与跨域 PHUMA

- **链接：** 项目页 Component Ablation / Training Dynamics / Generalization
- **摘录要点：**
  - **3% 消融**：去掉物理过滤 SR **0.911→0.956** 全管线；去掉多样性 **0.934**；复杂度加权在 **MPJPE** 上进一步精炼。
  - **训练曲线**：GQS 子集在 **<0.5B steps** 即更高 reward、更低 tracking error——解释为 **更干净的梯度轨迹**，非仅更快收敛。
  - **PHUMA**：**10% GQS** 子集 in-domain 优于全量；零样本迁到 AMASS **92.8% vs 91.0% SR**。
  - **Web mocap 清洗**：对估计源数据同样适用 GQS 框架（摘要 §1 末句）。
- **对 wiki 的映射：**
  - [LIMMT / GQS](../../wiki/methods/limmt-gqs-motion-curation.md) — 消融与跨数据集可迁移性

### 5) 真机 Unitree G1

- **链接：** 项目页 Real-World Deployment
- **摘录要点：** **10% GQS** 训练策略 **无微调** 上真机 G1；舞蹈、竞技、表现力动作等多类参考跟踪；论证策展不仅改善仿真指标，也利于 **sim2real 泛化**。
- **对 wiki 的映射：**
  - [LIMMT / GQS](../../wiki/methods/limmt-gqs-motion-curation.md) — 真机 G1 验证段落

## 对 wiki 的映射（总览）

- 沉淀方法页：[limmt-gqs-motion-curation](../../wiki/methods/limmt-gqs-motion-curation.md) — GQS 三阶段数据策展与「3% 胜全量」的人形 tracking 数据效率入口
- 交叉（正文互链，非本 source 强制映射）：EGM（小高质量集）、Any2Track、TWIST2、PHC、AMASS、Humanoid-GPT（同族 HME）、人形 motion tracking 选型 query

## BibTeX（项目页）

```bibtex
@article{guan2026limmt,
  title={LIMMT: Less Is More for Motion Tracking},
  author={Guan, Yu and Qi, Zekun and Lin, Chenghuai and Chen, Xuchuan and Liu, Dairu and
          Zhang, Wenyao and Wang, Jilong and Yu, Xinqiang and Wang, He and Yi, Li},
  journal={arXiv preprint arXiv:2606.06953},
  year={2026}
}
```
