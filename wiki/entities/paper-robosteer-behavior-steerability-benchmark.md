---

type: entity
tags: [paper, humanoid, benchmark, behavior-foundation-model, steerability, evaluation, unitree-g1, zju, unitree, nus, csu]
status: complete
updated: 2026-10-12
arxiv: "2610.10198"
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
summary: "RoboSteer（arXiv:2610.10198）：浙大/宇树/NUS/中南大学提出行为基础模型可操控性基准，三层（条件/约束/组合）共 773,716 实例、38,522 条源动作；评测 9 个 BFM，所有模型 BS 均低于 BG；数据与评测器权重已上 Hugging Face，代码未发布。"
---

# RoboSteer：行为基础模型可操控性基准

**RoboSteer**（[arXiv:2610.10198](https://arxiv.org/abs/2610.10198)，[项目页](https://robosteer.github.io/)）提出「行为可操控性」——行为基础模型（BFM）能否准确产出符合用户意图的行为——并用三层基准（条件、约束、组合操控）对 9 个 BFM 做首次大规模评测，核心结论是：生成得好不等于控得住。

## 一句话定义

**把「能生成人形动作」与「能按速度/方向/顺序/身体部位/多模态组合要求执行」分开度量：BG 衡量生成质量，BS=BG×∏IR 衡量意图实现后的可操控性。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| BFM | Behavior Foundation Model | 被评测对象 |
| BG | Behavior Generation score | BG=1000·exp(−0.3·FID−1.6·MM-Distance) |
| BS | Behavior Steering score | BS_l=BG×∏_{k≤l}IR_k |
| IR | Intention Realization | 来自机器人执行反馈的意图实现率，范围 [0,1] |
| FID | Fréchet Inception Distance | 生成分布质量指标 |
| GMR | General Motion Retargeting | 把生成动作重定向到 G1 |

## 为什么重要

- 现有 BFM 评测偏重生成质量，不能回答「命令里的速度、顺序、次数是否被做到」。
- 基准直接用 Unitree G1 重定向与跟踪反馈计算 IR，把「物理可执行」纳入打分。
- 773,716 实例使其可用于训练或细分诊断，而不仅是小规模对照。

## 核心信息

| 项 | 内容 |
|----|------|
| 机构 | 浙江大学（ZJU）；宇树科技（Unitree）；新加坡国立大学（NUS）；中南大学 |
| 规模 | 773,716 实例 / 20 任务族；38,522 源动作；91.58 小时配对视频 |
| 机器人 | Unitree G1（GMR 重定向 + SONIC 跟踪验证） |
| 开源 | **部分**：数据集与评测器权重在 Hugging Face；无 GitHub 代码；模型包未发布 |

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
- **数据构造：** 网络规模人体视频经过滤、Gemini-3-Flash-Preview 等字幕、人体动作重建、GMR 重定向与 SONIC 验证；语音条件由 Edge-TTS 合成。
- **指标：** 传统指标之外使用 BG、BS（见缩写速查）；BG 系数为经验选择。

## 实验与评测

| 条件 | 模型 | BG | BS-L1 | BS-L2 | BS-L3 |
|------|------|----|-------|-------|-------|
| 文本 | MotionCraft+GMR | 114.93 | 63.98 | 19.93 | 40.66 |
| 文本 | GEM+GMR | 114.09 | 68.38 | 13.97 | 42.12 |
| 文本 | TextOp | 82.67 | 43.54 | 17.95 | 31.05 |
| 音频 | MotionCraft+GMR | 108.23 | 57.88 | 16.76 | 40.05 |
| 视频 | Video2Robot+GMR | 119.34 | 91.98 | 23.55 | 34.97 |
| 视频 | GEM+GMR | 114.88 | 78.49 | 18.09 | 36.57 |
| 轨迹 | BFM-Zero | 95.39 | 95.39 | 未报告 | 未报告 |
| 节奏 | MotionCraft | 96.05 | 96.05 | 未报告 | 未报告 |

- **Q1：** 所有模型 BS 低于 BG，生成与操控是不同能力。
- **Q2：** 复杂度上升时可操控性下降；Q3：空间补全多弱于时间补全（支撑表 B.7 未读到）。
- **Q4：** Body Restrain 最难（表 B.8 未读到）；Q5：Level 3 同时满足多要求困难。
- TextOp 文本行 Contact Sliding 7.620 为最高。
- **一致性疑点：** 多行文本/音频结果 BS-L3 高于 BS-L2（如 GEM 文本 42.12 对 13.97），与「逐层下降」叙述冲突；轨迹/节奏行 BS-L1 与 BG 完全相同，意味着 IR₁=1。

## 源码运行时序图

不适用：官方仅发布数据集与评测器权重（Hugging Face），项目页无 GitHub 代码仓库，模型适配与评测代码「将发布」，截至 2026-10-12 无可对齐的运行入口。

## 工程实践

| 项 | 建议 |
|----|------|
| 获取数据 | 从 HF 数据集页下载 `archives/` 并核对 `SHA256SUMS`；留 ≈707 GiB（压缩+解压） |
| 评测 | 评测器权重在 `Leontc/RoboSteer-evaluator`；Order/Times 需 VLM 评测 |
| 许可 | 数据卡称许可仍在审查；AMASS/BABEL 派生部分有非商用条款，商用前需核对 |
| 报告方式 | 同时报 BG 与 BS，别只用 BG |

## 结论

**BFM 的瓶颈是意图兑现而非生成质量；基准提供了可诊断的三层结构，但个别数字与叙述需谨慎解读。**

1. **BG 与 BS 要分开看：** 所有模型 BS 均低于 BG。
2. **约束比条件更难：** 文本行 BS-L2 在 10.98–19.93 之间，远低于 BS-L1 的 43.54–68.38。
3. **Body Restrain 最难（作者结论）：** 细表未读到。
4. **多模态组合仍开放：** Level 3 仅占 1.00% 实例，覆盖偏薄。
5. **留意数据偏置：** 日常/行走/运动/表演占 83.92%，单机器人。
6. **数据可用，代码未出：** 复现评测需等官方评测代码。

## 与其他工作对比

| 对照 | 差异读法 |
|------|----------|
| [BFM-Zero](./paper-bfm-zero.md) | 本基准的被评对象之一（轨迹条件，BG 95.39） |
| [BFM 概念页](../concepts/behavior-foundation-model.md) | 本文给出 BFM 的评测维度 |
| [SONIC](../methods/sonic-motion-tracking.md) | 本文用其跟踪验证 G1 可执行性 |
| [HY-Motion 1.0](./paper-hy-motion-1-0.md) | 文本到动作生成模型，属于同类生成端 |

## 局限与风险

- **开源状态：** 数据与评测器权重已公开；无 GitHub 代码；模型资源仓库只有文档/manifest；许可审查中。
- 单机器人、单人动作；无物理交互、多机器人、长时程规划；合成语音。
- BG 系数为经验值；命名在正文与表之间不完全一致。
- 本页未读到附录 C–E 与完整 Level 2/3 表。

## 关联页面

- [人形/四足前沿周报 2026-10-05~10-09](../overview/frontier-humanoid-quadruped-2026-10-05-10-09.md)
- [行为基础模型概念页](../concepts/behavior-foundation-model.md)
- [BFM-Zero](./paper-bfm-zero.md)
- [GMR 重定向](../methods/motion-retargeting-gmr.md)

## 参考来源

- [robosteer_arxiv_2610_10198.md](../../sources/papers/robosteer_arxiv_2610_10198.md) — 论文来源归档
- [robosteer_project.md](../../sources/sites/robosteer_project.md) — 项目页核查
- [robosteer.md](../../sources/repos/robosteer.md) — 数据集归档
- [人形/四足周报来源](../../sources/blogs/frontier_humanoid_quadruped_2026_10_05_10_09.md)
- [arXiv:2610.10198](https://arxiv.org/abs/2610.10198)

## 推荐继续阅读

- [项目页](https://robosteer.github.io/)
- [Hugging Face 数据集](https://huggingface.co/datasets/YanCORANV/RoboSteer)
- [arXiv HTML 全文](https://arxiv.org/html/2610.10198v1)
