# CLIFT: Turning Gemini Robotics On-Device into Humanoid Specialists via Non-Invasive Closed-Loop Iterative Fine-Tuning（arXiv:2607.29172）

> 来源归档（ingest）

- **标题：** CLIFT: Turning Gemini Robotics On-Device into Humanoid Specialists via Non-Invasive Closed-Loop Iterative Fine-Tuning
- **缩写 / 框架：** **CLIFT**（Closed-Loop Iterative Fine-Tuning）
- **类型：** paper / humanoid / vla / closed-loop-finetuning / managed-api / reward-model / unitree-g1
- **arXiv：** <https://arxiv.org/abs/2607.29172>（v1，Submitted 2026-07-31，cs.RO；PDF：<https://arxiv.org/pdf/2607.29172>）
- **项目页：** <https://thomaschen98.github.io/clift> — 归档见 [`sources/sites/thomaschen98-clift.md`](../sites/thomaschen98-clift.md)
- **代码：** 项目页列出 Code 条目但**无可用链接**（截至入库日 2026-08-04 未发布）
- **作者：** Yuxin Chen、Hari Srikanth、Nathan Jew、Menglin Wu、Pengcheng Wang、Junli Ren、Masayoshi Tomizuka、Peng Xu、Jinyu Xie、Thomas Tian
- **机构：** 加州大学伯克利分校（UC Berkeley）、谷歌 DeepMind（Google DeepMind）、英伟达（NVIDIA Research）
- **入库日期：** 2026-08-04
- **一句话说明：** 闭权重机器人基础模型正在以「托管 SFT API」形式开放——用户交数据、拿回微调策略，但**拿不到权重 / 梯度 / loss / likelihood**，于是只能纯模仿。CLIFT 把**部署期的奖励反馈转成 API 兼容的监督数据**，在不「打开模型盒子」的前提下做闭环自改进，两个飞轮周期把 GROD 推到接近满分。

## 开源状态（步骤 2.5）

- **项目页核查（2026-08-04）：** `thomaschen98.github.io/clift` 有 flywheel 示意图、演示视频、BibTeX、arXiv 链接；**Code 条目列出但无有效 URL**，论文正文标注 `coming_soon`。
- **结论：** **宣称将开源 / 截至入库日未列可用链接**。且核心依赖是 **GROD 的托管 SFT API 访问权**与真机 Unitree G1，即使代码放出，复现门槛主要在**访问权与硬件**，不在代码。

## 摘录 1：托管 SFT API 这个新访问层

- **形式化：** 黑箱算子 \(\mathcal F_{SFT}\)：训练数据集 → 微调后的策略；**不暴露**权重、梯度、loss、动作 likelihood。
- **定位：** 介于「完全闭源」和「开放权重」之间的中间态；代表是 **Gemini Robotics On-Device（GROD）** 与 Physical Intelligence 的 partner API。
- **代价：** 策略改进被限制在**纯模仿**上——RL 与任何依赖内部训练信号的闭环方法都用不了。
- **为什么对人形尤其致命：** 敏捷、接触丰富的人形操作里，策略输出与真实部署行为之间差距很大（新状态、动作跟踪动力学、时延、控制器特有失败模式），纯模仿补不上。

**对 wiki 的映射：** 升格 [`wiki/entities/paper-clift-closed-loop-iterative-finetuning.md`](../../wiki/entities/paper-clift-closed-loop-iterative-finetuning.md)；与 [gemini-robotics](../../wiki/entities/gemini-robotics.md)、[safe-real-world-rl-fine-tuning](../../wiki/concepts/safe-real-world-rl-fine-tuning.md)、[hub-safe-fine-tuning](../../wiki/overview/hub-safe-fine-tuning.md) 互链。

## 摘录 2：CLIFT 三个组件

1. **偏好校准的稠密奖励模型（select-then-distill）**
   - 两个朴素做法都不行：VLM 零样本打分**可扩展但标定差**；逐步人工标注**贵**。
   - 做法：对每个 rollout **对**，提示 VLM 生成 **K=12** 条候选逐步奖励序列；只保留那些**诱导出的排序与人类成对偏好一致**的候选（用 100 组人工比较）；再把这些候选**蒸馏**成一个可复用的生成式奖励模型（基于 **Qwen3-VL**）。
   - 奖励模型**训练一次后固定**，不随飞轮更新。
2. **基于检索的优势条件（retrieval-based advantage conditioning）**
   - 每个 action chunk 拿到一个**二值 token**（正 / 负），表示它的回报是否高于**视觉相似状态**下的同侪 chunk。
   - 流程：(a) 用冻结 **DINOv3** 编码全部帧；(b) 对查询 chunk，检索初始观测余弦相似度超过阈值 \(\delta\) 的 chunk 作为比较集；(c) 用奖励模型在 **1.8 秒前瞻窗口**内算折扣回报；(d) 回报落在比较集 **前 30%** 则标正。
   - 关键性质：**门槛随状态难度自适应**——难状态里「还行」的 chunk 也能被标正。
3. **迭代自改进（flywheel）**
   - 第 k 轮：部署 \(\pi_k\) → 收 rollout → 打分与优势标注 → 并入累积数据集 \(\mathcal D_k=\mathcal D_{demo}\cup\mathcal D^{1:k}_{rollout}\) → 提交托管 API 得 \(\pi_{k+1}\)。
   - **演示数据始终标正**。

**对 wiki 的映射：** 实体页画飞轮流程图；强调「优势信号被编码成输入侧的 token，而不是改 loss」——这是「非侵入」的技术核心。

## 摘录 3：平台与任务

- **机器人：** Unitree G1 人形，双灵巧臂 + 多指手，头部两路 RGB。
- **控制分层：** 动作拆成上半身（臂 / 手关节角）与下半身（平面速度 + 偏航），由 RL 学到的全身控制器跟踪。
- **数据：** 全身 VR 遥操作，每任务 **2 小时** 演示。
- **三个接触丰富、需要全身平衡的任务：**
  - **Box Packing** — 抓取并放入箱中
  - **Cup Insertion** — 非对称双手协调：一手稳住、一手插入
  - **Bimanual Plate Handover** — 拾取、双臂交接、放置，需要臂间时序与稳定接触

## 摘录 4：结果（两个飞轮周期）

| 方法 | Box Packing | Cup Insertion | Bimanual Plate Handover |
|------|------------|---------------|-------------------------|
| **GROD + CLIFT（稠密优势条件）** | 93% → **100%** | 70% → **98%** | 53% → **96%** |
| GROD + 高回报 episode 筛选 | ~93% → ~98% | ~68% → ~94% | ~53% → ~84% |
| π₀.₅ + CLIFT（同一管线） | 59% → 76% | 50% → 56% | 5% → 30% |
| π₀.₅ + 侵入式 FiLM 适配 | ~70% | 48% | 40% |

**读点：**

- **API 直接 SFT 就已经强于同演示数据训练的开放权重 VLA（π₀.₅）**，但仍够不到部署级掌握。
- **稠密 chunk 级标注 > episode 级筛选**，差距在最难的双臂交接上最大（96% vs ~84%）：失败 rollout 里被良好执行的片段被**重新标为正例**回收了。
- **基础模型强度和访问权限一样重要**：受限 API 下的 GROD 反而胜过对较弱 π₀.₅ 做侵入式适配。
- 两轮后出现**演示中不存在的涌现行为**（预操作调整、失败重试）。

**基线设置：** π₀.₅（PaliGemma 主干）套**完全相同**的 CLIFT 管线（同奖励模型、同 rollout 预算、同优势标注）做受控对照；另测一个把偏好信号经 FiLM 式架构条件**直接注入**的侵入式适配基线。

## 摘录 5：局限

- **每轮都要真机 rollout** 才能拿部署信号 → 贵且涉及安全；论文提出用 **control-aware world model** 减少物理 rollout 作为未来方向。
- **只对比了一个开放权重模型**（π₀.₅），结论的外推需谨慎。
- **强依赖托管 API 与硬件访问权**，社区难独立复现。
- 奖励模型固定不更新：若策略行为分布随飞轮显著漂移，标定可能过期（论文未做这项消融）。

## 建议 wiki 动作

- 新建 **`wiki/entities/paper-clift-closed-loop-iterative-finetuning.md`**（含飞轮流程图 + 结论；源码运行时序图写「不适用」及原因）。
- 新建 **`sources/sites/thomaschen98-clift.md`**。
- 交叉：[`wiki/entities/gemini-robotics.md`](../../wiki/entities/gemini-robotics.md)、[`wiki/entities/unitree-g1.md`](../../wiki/entities/unitree-g1.md)、[`wiki/concepts/safe-real-world-rl-fine-tuning.md`](../../wiki/concepts/safe-real-world-rl-fine-tuning.md)、[`wiki/tasks/bimanual-manipulation.md`](../../wiki/tasks/bimanual-manipulation.md)、[`wiki/concepts/reward-design.md`](../../wiki/concepts/reward-design.md)。
