---
type: entity
project_id: driving-autovla
tags: [paper, curated-index, awesome-world-action-models-rcl, rcl-wam-catalog]
status: complete
updated: 2026-10-06
arxiv: "2506.13757"
venue: "NeurIPS 2025"
code: https://github.com/ucla-mobility/AutoVLA
summary: "RCL Awesome WAM 清单收录（VLA）；细节以原文 PDF / 项目页为准。"
related:
  - paper-rcl-wam-robot-learning-control-survey.md
  - ../overview/rcl-awesome-wam-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
original:
  - ../../sources/papers/ad_top10_2026_autovla.md
sources:
  - ../../sources/papers/ad_top10_2026_autovla.md
  - ../../sources/repos/ad_top10_2026_autovla.md
  - ../../sources/blogs/wechat_shenlan_ai_autonomous_driving_top10_2026.md
  - ../../sources/papers/rcl_awesome_wam_2506_13757_autovla-a-vision-language-action-model-f.md
  - ../../sources/papers/rcl_awesome_wam_catalog.md
  - ../../sources/repos/awesome-world-action-models-rcl.md
---

# AutoVLA

**AutoVLA: A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning** 收录于 [Awesome World-Action Models (RCL)](https://github.com/rcl-robotics/Awesome-World-Action-Models) **第 228/564** 篇，分组 **VLA**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

RCL Awesome WAM 清单收录（VLA）；细节以原文 PDF / 项目页为准。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WAM | World Action Model | 世界预测与动作生成耦合 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| IDM | Inverse Dynamics Model | 先预测未来再反推动作 |
| WM | World Model | 环境前向预测模型 |

## 为什么重要

- RCL Awesome WAM 清单收录（VLA）；细节以原文 PDF / 项目页为准。
- 想横向对照同一分组的其他工作，可以从 [RCL Awesome WAM 技术地图](../overview/rcl-awesome-wam-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World-Action Models](paper-rcl-wam-robot-learning-control-survey.md) 与站内 WAM / VLA 方法页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    subgraph IN["模型输入"]
        CAM["三路相机（前/左前/右前），每路 4 帧 @2 Hz"]
        NAV["高层导航指令 I（如 Turn Left / Go Straight）"]
        EGO["自车状态 S：速度、加速度、历史动作"]
    end
    CAM --> VLM
    NAV --> VLM
    EGO --> VLM
    VLM["Qwen2.5-VL-3B：统一自回归 Transformer 解码器"] --> MODE{"快思考 / 慢思考"}
    MODE -- "快思考：直接输出" --> ACT
    MODE -- "慢思考：先做 CoT（场景分析、关键物体、他车意图、驾驶决策）" --> ACT["物理动作 token（K=2048 码本，每个为 Δx, Δy, Δθ）"]
    ACT --> TRJ["按码本解码为规划轨迹"]
    subgraph SFT["阶段 1：监督微调 SFT"]
        GT["真值轨迹（离散化为动作 token）"]
        COT["Qwen2.5-VL-72B 自动生成 CoT 推理标注<br/>（nuPlan 45.6k、Waymo 7.2k，另并入 DriveLM）"]
        LS["损失：语言建模 L_LM + λa·L_action，含 CoT 样本加权 λcot"]
        GT --> LS
        COT --> LS
    end
    subgraph RFT["阶段 2：强化微调 RFT（GRPO）"]
        SMP["对同一场景采样 G 个候选输出"]
        RWD["奖励 r = r_Driving − λr·r_CoT<br/>nuPlan 用 PDMS；Waymo 用 ADE；CoT 长度惩罚"]
        GRP["组内归一化优势 + 裁剪 + KL 约束于 SFT 参考策略"]
        SMP --> RWD --> GRP
    end
    LS --> RFT
    RFT -. "优化得到的策略" .-> VLM
```

图依据：论文 §3 概览与 Fig.3（两阶段训练）、§3.1（输入、动作 tokenization、快慢思考）、§3.2（推理数据）、§3.3（SFT）、§3.4（GRPO 强化微调）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 228/564 |
| 分组 | VLA |
| 出处 | NeurIPS 2025 |
| 论文 | <https://proceedings.neurips.cc/paper_files/paper/2025/hash/2843fccca5bedd369a4764848b9bd546-Abstract-Conference.html> |
| 代码/项目 | <https://github.com/ucla-mobility/AutoVLA> |
| 子类 / 象限 | 自回归VLA · VLA后训练与数据增强 · 不适用 |

## 核心机制（归纳）

### 策展导读要点

RCL Awesome WAM 清单收录（VLA）；细节以原文 PDF / 项目页为准。

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内 [World Action Models（WAM）](../concepts/world-action-models.md) 等概念页。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://proceedings.neurips.cc/paper_files/paper/2025/hash/2843fccca5bedd369a4764848b9bd546-Abstract-Conference.html) 为准。
- 横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 的 **VLA** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：清单 Contribution 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「AutoVLA」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **VLA**，以及 Contribution 点出的问题设定；本页不自行推导新结论。
- 适用边界：本页不能替代原文 PDF；开源状态以项目页实际链接为准（清单可能滞后）。
- 要深读这篇，建议直接从原文入手，再回到下方关联的方法 / 任务页对照。

## 常见误区

1. 不要把 Awesome 条目的 Contribution 当成完整方法证明——它只是策展导读。
2. 若站内已有这篇的深读页，以那一页为准——本页只是清单入口，不含实验数据。

## 关联页面

- 列表实体：[Awesome World-Action Models（RCL）](paper-rcl-wam-robot-learning-control-survey.md)
- 技术地图：[RCL Awesome WAM 技术地图](../overview/rcl-awesome-wam-technology-map.md)
- 方法/任务：[generative-world-models.md](../methods/generative-world-models.md)、[manipulation.md](../tasks/manipulation.md)

## 参考来源

- [`sources/papers/rcl_awesome_wam_2506_13757_autovla-a-vision-language-action-model-f.md`](../../sources/papers/rcl_awesome_wam_2506_13757_autovla-a-vision-language-action-model-f.md) — 本条目策展摘录
- [`sources/papers/rcl_awesome_wam_catalog.md`](../../sources/papers/rcl_awesome_wam_catalog.md) — 列表总表
- [`sources/repos/awesome-world-action-models-rcl.md`](../../sources/repos/awesome-world-action-models-rcl.md)
- [`docs/PAPERS.md`](https://github.com/RCL-Robotics/Awesome-World-Action-Models/blob/main/docs/PAPERS.md) — 上游论文目录
- 论文：<https://proceedings.neurips.cc/paper_files/paper/2025/hash/2843fccca5bedd369a4764848b9bd546-Abstract-Conference.html>
- [arXiv PDF](https://arxiv.org/pdf/2506.13757)：已通读方法部分（实验与附录以略读为准）：§3 方法（§3.1–3.4）；§4 实验与补充材料仅略读。

## 推荐继续阅读

- [Awesome World-Action Models (RCL) 仓库](https://github.com/rcl-robotics/Awesome-World-Action-Models)
- [原文](https://proceedings.neurips.cc/paper_files/paper/2025/hash/2843fccca5bedd369a4764848b9bd546-Abstract-Conference.html)


## 2026-11-18 官方原文复核与复现边界

**按需快慢推理、离散动作token及GRPO。** 将轨迹增量映射为动作codebook；简单场景直接产生动作token，复杂场景先生成驾驶CoT再生成动作；SFT之后用GRPO强化微调。

- **原始指标：** 官方开放NAVSIM与nuScenes等训练和评测脚本，不能跨基准混用PDMS与轨迹误差。
- **开源边界：** 动作token的离散误差及额外CoT的延时需单独评估；离线规划指标不是实车安全证明。
- **官方运行入口：** ```bash scripts/action_token_cluster.sh```、```bash scripts/run_sft.sh```、```bash scripts/run_rft.sh```、```bash navsim/scripts/evaluation/run_autovla_agent_pdm_score_evaluation.sh```。

### 复核流程图

```mermaid
flowchart LR
  Obs[相机导航与自车状态] --> VLM[Qwen2.5 VL]
  VLM --> Mode{快/慢推理}
  Mode -->|快| Token[动作token]
  Mode -->|慢| CoT[因果解释]
  CoT --> Token
  Token --> Dec[轨迹解码]
  GRPO[GRPO后训练] -.优化.-> VLM
```

## 源码运行时序图（源码入口核对）

```mermaid
sequenceDiagram
  autonumber
  actor Dev as 开发者
  participant Code as scripts/action_token_cluster.sh
  participant SFT as scripts/run_sft.sh
  participant RFT as scripts/run_rft.sh
  participant Eval as navsim/scripts/evaluation/run_autovla_agent_pdm_score_evaluation.sh
  Dev->>Code: 根据轨迹构造动作codebook
  Code-->>SFT: 离散动作token
  Dev->>SFT: 监督微调Qwen2.5-VL骨干
  SFT-->>RFT: 初始化模型权重
  Dev->>RFT: GRPO强化微调
  RFT-->>Eval: 训练后检查点
  Dev->>Eval: NAVSIM评分
  Eval-->>Dev: PDMS输出
```

本次核验是静态分析，不声称在GPU上执行。详见[论文核对](../../sources/papers/ad_top10_2026_autovla.md)和[代码入口](../../sources/repos/ad_top10_2026_autovla.md)。

## 工程实践

官方运行入口及数据限制见上一节；需按实际硬件与数据版本配置。

## 局限与风险

动作token的离散误差及额外CoT的延时需单独评估；离线规划指标不是实车安全证明。
