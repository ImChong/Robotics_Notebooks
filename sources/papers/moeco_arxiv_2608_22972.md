# MoeCo（知识驱动 MoE 的手术三元组协同优化）

> 来源归档（ingest）

- **标题：** Optimize Surgical Triplet Recognition: A Knowledge-Driven Mixture-of-Experts Solution
- **类型：** paper
- **原始链接：**
  - <https://arxiv.org/abs/2608.22972>
- **代码：** <https://github.com/YIYIZH/MoeCo>（部分发布）
- **机构：** 香港中文大学（CUHK）；香港理工大学（PolyU）
- **入库日期：** 2026-08-26
- **一句话说明：** 用组件定制适配器拆开器械/动作/目标特征冲突，用协调梯度学习重平衡长尾，再用 MLLM 结构先验的 MoE 动态注入领域知识，做手术视频 \(\langle\)instrument, verb, target\(\rangle\) 识别。

## 核心摘录（MVP）

### 1) 两层优化冲突 + 领域先验缺失

- **摘录要点：** 共享 \(\langle\)grasper, retract\(\rangle\) 的两个三元组在 I/V 空间应靠近、在 T/IVT 空间应分开；CholecT45 头类 >40k、尾类仅 8 样本，头类梯度淹没尾类。器械尖端/腕/杆的结构—功能约束（如叉形尖端更像 clipper）被现有方法忽略。
- **对 wiki 的映射：**
  - [MoeCo](../../wiki/entities/paper-moeco.md)

### 2) CTA + CGL + KD-MoE

- **摘录要点：** CTA 在时空两套 latent 上做任务 prompt，避免单一子空间同时服务帧内语义与长程一致。CGL 分解 BCE 正负梯度并按概率丢弃头类过强正梯度。KD-MoE：GPT-4o 挖器械属性 → CLIP 文本嵌入 → 高斯专家，Top-K 激活后拼到视觉嵌入；MLLM **仅离线**，训练/推理不用。
- **对 wiki 的映射：**
  - [MoeCo](../../wiki/entities/paper-moeco.md) — 方法栈。

### 3) CholecT45 / T50

- **摘录要点：** 5-fold CholecT45：MoeCo-T \(AP_{IVT}\) **40.5%**（+4.8 vs TERL-T），MoeCo-B **41.7%**，集成 **42.6%**。CholecT50：MoeCo-B **40.5%**。消融：baseline 38.3 → KD-MoE 40.3 → +CTA+CGL 42.3。相对 4-task 分支，CTA 用更少参数换更大增益。
- **对 wiki 的映射：**
  - [MoeCo](../../wiki/entities/paper-moeco.md) — 评测。

### 4) 开源状态（截至 2026-08-26）

- **摘录要点：** **部分开源**。仓内有 `network.py` / `dataloader.py` / `loss/` / `run.sh`，README 写明完整训练入口、GMM 与预提取特征 **录用后发布**；部分文件仍含实验机绝对路径。
- **对 wiki 的映射：**
  - [仓库归档](../repos/moeco.md)

## 当前提炼状态

- [x] arXiv HTML + GitHub README 已对齐摘录
- [x] wiki 映射：`wiki/entities/paper-moeco.md` 新建
