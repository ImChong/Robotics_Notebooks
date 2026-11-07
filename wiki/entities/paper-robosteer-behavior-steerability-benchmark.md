---

type: entity
tags: [paper, humanoid, benchmark, behavior-foundation-model, steerability, evaluation, unitree-g1, zju, unitree, nus, csu]
status: complete
updated: 2026-11-07
arxiv: "2610.10198"
original:
  - ../../sources/papers/robosteer_arxiv_2610_10198.md
project_id: robosteer-behavior-steerability-benchmark
project: https://robosteer.github.io/
related:
  - ../overview/frontier-humanoid-quadruped-2026-10-05-10-09.md
  - ../concepts/behavior-foundation-model.md
  - ./paper-bfm-zero.md
  - ../methods/sonic-motion-tracking.md
  - ../methods/motion-retargeting-gmr.md
  - ./paper-hy-motion-1-0.md
sources:
  - ../../sources/papers/robosteer_arxiv_2610_10198.md
  - ../../sources/sites/robosteer_project.md
  - ../../sources/repos/robosteer.md
  - ../../sources/blogs/frontier_humanoid_quadruped_2026_10_05_10_09.md
summary: "RoboSteer（arXiv:2610.10198v2）：浙大/宇树/NUS/中南大学提出行为基础模型可操控性基准，三层共 773,716 任务实例、38,522 条源动作，评测 9 组代表性 BFM；v2 全文与附录 A–E 已核对。数据、评测器权重及网页评测入口已公开，完整官方适配/评测源码尚未公开。"
---

# RoboSteer：行为基础模型可操控性基准 / Benchmarking Behavioral Steerability in Behavior Foundation Models

**RoboSteer**（[arXiv:2610.10198v2](https://arxiv.org/html/2610.10198v2)，[项目页](https://robosteer.github.io/)）提出「行为可操控性」——行为基础模型（BFM）能否准确产出符合用户意图的行为——并用三层基准（条件、约束、组合操控）对 9 个 BFM 做首次大规模评测，核心结论是：生成得好不等于控得住。

## 一句话定义

**把「能生成人形动作」与「能按速度/方向/顺序/身体部位/多模态组合要求执行」分开度量：BG 衡量生成质量，BS=BG×∏IR 衡量意图实现后的可操控性；RoboSteer 本身是**评测基准而非新 BFM 模型**。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| BFM | Behavior Foundation Model | 被评测对象 |
| BG | Behavior Generation score | BG=1000·exp(−0.3·FID−1.6·MM-Distance) |
| BS | Behavior Steering score | BS_l=BG×∏_{k≤l}IR_k |
| IR | Intention Realization | 按层评估用户意图实现率，范围 [0,1] |
| FID | Fréchet Inception Distance | 生成分布质量指标 |
| GMR | General Motion Retargeting | 把人体动作重定向到 G1 |
| JRO | Joint-Range Overlap | 空间补全时缺失关节角度范围的重叠分 |
| tIoU | Temporal Intersection-over-Union | 时间补全中生成和参考动作的时长重叠率 |

## 为什么重要

- 现有 BFM 评测偏重生成质量，不能回答「命令里的速度、顺序、次数是否被做到」。
- 基准把多种模型输出统一到 G1 29 DoF、50 Hz 表示，结合 GMR 重定向、SONIC 跟踪与约束检查，尽量让生成质量和意图遵循可对照；**不能等同大规模真机验证**。
- 773,716 实例使其可用于训练或细分诊断，而不仅是小规模对照。

## 核心信息

| 项 | 内容 |
|----|------|
| 机构 | 浙江大学（ZJU）；宇树科技（Unitree Robotics）；新加坡国立大学（NUS）；中南大学（CSU） |
| 论文版本 | arXiv:2610.10198v2（2026-10-11）；正文及附录 A–E 已通读核对 |
| 规模 | 773,716 实例 / 20 任务族；38,522 源动作；91.58 小时配对视频 |
| 机器人 | Unitree G1（GMR 重定向 + SONIC 跟踪验证） |
| 开源 | **部分**：数据集、评测器权重与项目页交互式评测入口可用；完整官方模型适配/评测源码与模型包未公开 |

## 流程总览

```mermaid
flowchart LR
  A["网络人体视频"] --> B["过滤 + 字幕 + 人体动作重建"]
  B --> C["GMR 重定向到 G1 + SONIC 跟踪验证"]
  C --> D["38,522 源动作 -> 773,716 任务实例"]
  D --> E["L1 条件 / L2 约束 / L3 组合"]
  E --> F["9 个 BFM 生成动作"]
  F --> G["机器人执行反馈 -> IR"]
  G --> H["BG 与 BS 打分"]
```

## 方法

- **Level 1 条件操控：** 12 个任务族、630,623 实例（81.51%）——完整条件复现（文本/视频/音频/节奏/旋转）、时间补全（预测/回溯/插值/关键帧）、空间补全（上/下半身到全身、目标到达）。
- **Level 2 约束操控：** 135,329 实例（17.49%），7 类约束：速度、幅度、方向、顺序、次数、轨迹、身体约束；速度/幅度/方向/轨迹/身体约束用 CSV 运动学评估，顺序和次数用视频+VLM 评估（项目页）。
- **Level 3 组合操控：** 7,764 实例（1.00%），文本/音频/视频/关键帧交错的多源条件。
- **数据构造：** 网络规模人体视频经过滤、Gemini-3-Flash-Preview 等时序动作语义标注、人体动作重建、GMR 重定向与 SONIC 验证；语音条件由 Edge-TTS 合成。Order/Times 另使用 BABEL/AMASS 标注片段。
- **指标：** 传统指标之外使用 BG、BS（见缩写速查）；BG 系数为经验选择。单样本形式化公式为 `BS_l = BG × ∏(IR_k, k≤l)`，但跨层表格是不同任务子集的聚合结果。

## 评测指标的原文定义与边界

| 层级 | IR 的计算 | 工程上容易误判的点 |
|---|---|---|
| L1 完整条件复现 | **明确设 IR₁=1**，因此 BS-L1=BG | 是定义导致的相等，不代表模型完美执行所有要求 |
| L1 时间补全 | 生成/参考时长的 tIoU | 评价目标区间长短，不能单凭它确认动作语义正确 |
| L1 空间补全 | 缺失关节角度范围的 JRO | 关注关节活动范围重叠，非逐时刻精确跟踪 |
| L1 关键帧、目标 | G1 运动学的 14 链接姿态误差，±0.04 s 时间窗，指数得分 | 容许轻微时间偏移 |
| L2 约束 | 七类约束的二值通过率宏平均 | Speed 只判断时长变短/变长，**并不核实精确 2 倍/0.5 倍**；Direction 容许相对目标方向 ±60°；Body Restrain 用受限区 RMS 速度 ≤0.10 m/s 且剩余区 ≥0.12 m/s |
| L3 组合 | 按各模态片段时长对段级意图分加权平均 | 反映连续序列中的多源指令切换，不覆盖复杂接触或长期规划 |

BG = `1000 × exp(−0.3 × FID − 1.6 × MM-Distance)`。FID 衡量生成/目标分布，MM-Distance 衡量输入/行为的一致性；分模态使用不同的冻结评测器。Order/Times 需固定 VideoLLM 评估渲染视频，不能只通过关节 CSV 推导。

**重要：** 论文用 `BS_l=BG×∏IR_k` 定义同一评价对象的分数，正文汇总表的 L1/L2/L3 涉及不同任务样本与输入条件，不能直接从三个**汇总值**验算单样本乘积公式。

## 实验与评测（arXiv v2 Table 2）

| 输入 | 模型 | BG | BS-L1 | BS-L2 | BS-L3 |
|---|---|---:|---:|---:|---:|
| 文本 | GEM+GMR | 114.09 | 68.38 | 16.15 | 42.12 |
| 文本 | LoM+GMR | 104.63 | 66.27 | 16.83 | 39.28 |
| 文本 | MotionCraft+GMR | 114.93 | 63.98 | 22.73 | 40.66 |
| 文本 | TextOp | 82.67 | 43.54 | 20.41 | 31.05 |
| 文本 | UniAct | 102.44 | 57.95 | 12.76 | 37.05 |
| 文本 | UH-1 | 96.65 | 48.97 | 18.53 | 38.21 |
| 音频 | MotionCraft+GMR | 108.23 | 57.88 | 19.74 | 40.05 |
| 视频 | Video2Robot+GMR | 119.34 | 91.98 | 23.73 | 34.97 |
| 视频 | GEM+GMR | 114.88 | 78.49 | 20.98 | 36.57 |
| 轨迹 | BFM-Zero | 95.39 | 95.39 | — | — |
| 节奏 | MotionCraft | 96.05 | 96.05 | — | — |

- **版本更正：** 相比旧页使用的 v1 数字，v2 的 GEM 文本 BS-L2 为 **16.15**（原 13.97），MotionCraft 文本为 **22.73**（原 19.93），Video2Robot 视频为 **23.73**（原 23.55），请以 v2 为准。
- **Q1 生成 ≠ 可操控：** 非平凡 steering 任务的 BS 普遍低于对应 BG；完整复现特例 BS-L1=BG。
- **Q2 复杂需求难：** 对并行或交错的多条要求保持一致性更困难；不能把不同任务集的汇总分数当作同一输入上的单调序列。
- **Q3 时空能力：** 附录 **Table B.7** 的多数模型空间补全得分低于时间补全。
- **Q4 最难约束：** 附录 **Table B.8** 显示 Body Restrain 通常很低，例如 GEM 文本为 **0.832**（对应 BS-L2）；强调身体部位级别控制薄弱。
- **Q5 组合操控：** 语言中心的输入不足以详尽表达身体和空间关系，论文倡导交错多模态输入。
- **指标方向：** TextOp 文本行 Contact Sliding = 7.620，**越低越好**，不能误写“最高即最好”。

**实验适配：** 不同基线的文本/音频经过 Qwen2.5-3B-Instruct 与 faster-whisper 规范化；GMR 把人体输出转成 G1，UH-1 将 H1 27 DoF 映射至 G1 29 DoF。作者称 9 组模型家族，但 §5.1 所述 GENMO / PromptHMR 与 Table 2 的 GEM / Video2Robot 在命名上不一致，故此页按表格原名列出，不自行合并。

## 源码运行时序图

不适用：官方仅发布数据集与评测器权重（Hugging Face），项目页无 GitHub 代码仓库，模型适配与评测代码「将发布」，截至 2026-11-07 官方网页提供托管评测服务与数据/权重，但未提供可通读、可执行的官方仓库入口；不以网页表单推导源码调用关系。

## 工程实践

| 项 | 建议 |
|----|------|
| 获取数据 | 从 HF 数据集页下载 `archives/` 并核对 `SHA256SUMS`；留 ≈707 GiB（压缩+解压） |
| 评测 | 评测器权重在 `Leontc/RoboSteer-evaluator`；项目页现有网页评测入口；Speed/Amplitude/Direction/Trajectory/Body Restrain 用动作 CSV，Order/Times 需视频 VLM 评测 |
| 许可 | 数据卡称许可仍在审查；AMASS/BABEL 派生部分有非商用条款，商用前需核对 |
| 报告方式 | 同时报 BG 与 BS，别只用 BG |

## 结论

**RoboSteer 最重要的贡献不是刷新生成榜，而是让“动作像不像”与“是否准确按用户意图做”分别可检验。**

1. **报告 BG 和 BS 两种分数：** 只看 FID 或 BG 会掩盖细粒度指令的执行失败。
2. **完整复现 BS-L1=BG 是定义：** 原文附录 C 明确设 IR₁=1，不能据此推断完美可操控。
3. **优先攻克身体部位约束：** Table B.8 表明 Body Restrain 是明显弱项，适合用于关节/身体部位表示的消融研究。
4. **正视指标宽松性：** Speed 与 Amplitude 看方向不看精确倍率，真实运控应用还应补速度/轨迹误差与安全性。
5. **跨模态结论有适配前提：** 转写与文本改写、重定向、G1/SONIC 统一表示均会影响可比性，不能直接当作原生端到端真机效果。
6. **数据可取，但完整代码未公开：** 适合先做数据分析和线上单样本约束测试，完整基准复现入口仍待官方发布。

## 与其他工作对比

| 对照 | 差异读法 |
|------|----------|
| [BFM-Zero](./paper-bfm-zero.md) | 本基准的被评对象之一（轨迹条件，BG 95.39） |
| [BFM 概念页](../concepts/behavior-foundation-model.md) | 本文给出 BFM 的评测维度 |
| [SONIC](../methods/sonic-motion-tracking.md) | 本文用其跟踪验证 G1 可执行性 |
| [HY-Motion 1.0](./paper-hy-motion-1-0.md) | 文本到动作生成模型，属于同类生成端 |

## 局限与风险

- **开源状态：** 数据、评测器权重及网页评测入口已开放；完整官方代码/模型包未公开；数据许可需要逐项核实。
- 单机器人、单人动作；无物理交互、多机器人、长时程规划；合成语音。
- BG 系数为经验值；Speed/Amplitude/Direction 指标容差较宽，模型适配对输入做了转写/文本规范化；命名在正文与表之间不完全一致。
- **已核对 v2 正文与附录 A–E（包括 B.7/B.8 详细表）**；但官方尚无可复现的完整评测源码，且论文规模化结果不是已证明的真机安全能力。

## 关联页面

- [人形/四足前沿周报 2026-10-05~10-09](../overview/frontier-humanoid-quadruped-2026-10-05-10-09.md)
- [行为基础模型概念页](../concepts/behavior-foundation-model.md)
- [BFM-Zero](./paper-bfm-zero.md)
- [GMR 重定向](../methods/motion-retargeting-gmr.md)
- [具身大模型评测基准选型闭环](../queries/embodied-eval-benchmark-selection-loop.md) — 本页属③层在行为基础模型上的展开：按条件/约束/组合三层测 BFM 可操控性，区分「被引导」与仅「生成合理动作」

## 参考来源

- [robosteer_arxiv_2610_10198.md](../../sources/papers/robosteer_arxiv_2610_10198.md) — 论文来源归档
- [robosteer_project.md](../../sources/sites/robosteer_project.md) — 项目页核查
- [robosteer.md](../../sources/repos/robosteer.md) — 数据集归档
- [人形/四足周报来源](../../sources/blogs/frontier_humanoid_quadruped_2026_10_05_10_09.md)
- [arXiv:2610.10198v2 HTML 全文](https://arxiv.org/html/2610.10198v2)

## 推荐继续阅读

- [项目页](https://robosteer.github.io/)
- [Hugging Face 数据集](https://huggingface.co/datasets/YanCORANV/RoboSteer)
- [arXiv HTML 全文 v2（附录 A–E）](https://arxiv.org/html/2610.10198v2)
