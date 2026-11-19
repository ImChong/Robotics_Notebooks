# Are Foundation Models the Route to Full-Stack Transfer in Robotics?

> 来源归档（综述全文通读与核对 · `arXiv:2602.22001v1`）

- **标题：** Are Foundation Models the Route to Full-Stack Transfer in Robotics?
- **类型：** paper
- **状态：** 综述 · arXiv 2026（v1，2026-02-25）
- **原始链接：**
  - arXiv：<https://arxiv.org/abs/2602.22001>
- **代码：** 不适用（DLR + Stanford AI Lab 综述研究论文，无配套代码仓库）
- **作者：** Freek Stulp, Samuel Bustamante, João Silvério, Alin Albu-Schäffer (DLR); Jeannette Bohg, Shuran Song (Stanford AI Lab)
- **机构：** German Aerospace Center (DLR); Stanford University (Stanford AI Lab)
- **入库日期：** 2026-08-18
- **更新日期：** 2026-11-19
- **核对范围：** 通读 `arXiv:2602.22001v1` 综述全文（§1–§8 与参考文献）。
- **通读：** 已完成
- **一句话说明：** 联结 **DLR** 与 **Stanford AI Lab** 的全栈迁移综述（arXiv:2602.22001）：以 Jaquier et al. 迁移层次（Mimicry → Imitation → Emulation → Causal understanding / Verbal Instructions / Goal Specifications）为主轴，系统评估 LLM/VLM、扩散/流匹配策略、VLA 知识隔离（$\pi_0$/$\pi_{0.5}$-KI/$\pi_{0.6}$）、跨具身策略与真机评测规范，结论是基础模型是关键主线但必须结合世界模型、显式记忆与闭环 RL。

## 核心论文摘录（全文核对）

### 1. 全栈迁移问题重框与迁移层次

- **核心定义：** 基于 Jaquier et al. 分类框架，区分机器人知识迁移的六个层次：
  - 低层物理与轨迹层面：**Mimicry（动作复制）** → **Imitation（动作对齐语义目标）** → **Emulation（理解目标但自主发现动力学解）**。
  - 高层认知与条件层面：**Causal Understanding（因果物理推断）**、**Verbal Instructions（自然语言指令泛化）**、**Goal Specifications（目标图/子目标规格）**。
- **核心张力：** 大模型（LLM/VLM）带来高层语言–视觉–因果先验，但微调低层动作时容易破坏高层语义表示（知识退化）；物理接触和不同机器人动力学则需要局部反应与记忆。

### 2. VLA 演进与知识隔离（Knowledge Insulation）

- **动作表示分水岭：** OpenVLA / VLA-DAT 的离散自回归动作分词 vs $\pi_0$-FAST 基于 DCT 的高频块压缩 vs $\pi_0$ / $\pi_{0.5}$ / $\pi_{0.6}$ 连续 Flow-Matching Action Expert。
- **知识隔离（KI）与联合修复：** $\pi_{0.5}$-KI 与 $\pi_{0.6}$ 在预训练与微调时对 VLM 主干采用梯度阻断或解耦训练，防止高频控制信号洗掉 VLM 的开放世界语义理解；OG-VLA 等通过图像/特征修复补齐视角缺失。
- **跨具身三条机制：** 1）具身无关共享表征（如 UMI / 人手腕空间）；2）多机联合隐空间对齐；3）先大底座预训练 + 少量目标具身微调或上下文适配。

### 3. 数据范式与基准评测建议

- **数据采集光谱：** 遥操作（ALOHA / DROID 等，高对齐度但高成本） vs 便携无机器人手持/可穿戴采集（UMI / UMI-FT / iPhUMI 等，高多样性） vs VLM 自动重标定与人类视频利用。
- **基准建设准则：** 明确源域–目标域分布划分（避免数据泄露掩盖迁移增益）；以真机开放共享评估（如 RoboArena、euROBIN 协作比赛）替代单一封闭环境指标。

### 4. 核心结论：基础模型是主干但非全部答案

- 仅靠更大规模的静态图文–动作 VLM/VLA **不足以实现完美全栈迁移**：在 **非马尔可夫长时记忆**、**未知物理动力学接触适应**、**在线闭环强化学习提升（超过模仿上限）** 上仍需结合扩散微调、世界模型、显式门控/事件记忆与合规柔顺控制器。

## 对 wiki 的映射

- [`wiki/entities/paper-foundation-models-full-stack-transfer.md`](../../wiki/entities/paper-foundation-models-full-stack-transfer.md)
- [`wiki/overview/realab-14-papers-technology-map-2026.md`](../../wiki/overview/realab-14-papers-technology-map-2026.md)

## 当前提炼状态

- [x] `arXiv:2602.22001v1` 全文通读与核对完成
- [x] 支撑 `wiki/overview/realab-14-papers-technology-map-2026.md` 与 `paper-foundation-models-full-stack-transfer.md`
