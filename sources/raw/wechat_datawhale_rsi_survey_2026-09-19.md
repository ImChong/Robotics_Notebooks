---
title: 这是一篇把"RSI"讲明白的科普级综述！
author: 赵志民（Datawhale）
date: 2026-09-19
source: https://mp.weixin.qq.com/s/rlfTKyWhALsNONhAwGih1A
---

# 这是一篇把"RSI"讲明白的科普级综述！

> 原始落盘（WebFetch；`--no-images`）

- **作者：** 赵志民，Datawhale 成员（加拿大皇后大学博士；主页 https://zhimin-z.github.io）
- **公众号：** Datawhale
- **链接：** https://mp.weixin.qq.com/s/rlfTKyWhALsNONhAwGih1A
- **入库日期：** 2026-09-19
- **抓取方式：** WebFetch（本环境未预装 `wechat-article-for-ai`）

## 正文（WebFetch 摘录）

2026 年 7 月 OpenAI 发布 GPT-5.6，成绩单首次出现 **Self-improvement** 科目：GPT-5.6 Sol **57.9%**，GPT-5.5 **41.7%**（OpenAI 称 **RSI Index**）。同一时期：OpenAI 将 Agent 称为「自动化研究实习生」；快手 **AgentX** 参与方案、代码与线上 A/B；Google 称长期 Agent 循环参与下一版模型精炼；**Motus2** 把预测、评估与策略更新带进真机。

### 四层 RSI 标准（文内框架）

1. **持久改进** — 变化进入权重、记忆、工具、harness 或训练流程，而非一次回答即消失。
2. **自主闭环（有界 RSI）** — 在划定边界内发现问题、修改、实验、评估并接纳更好版本；新版本可参与下一轮改进。
3. **递归增益（ignition）** — 新版本不只在任务分数更高，还**更擅长产生下一次改进**；改进能力本身开始复利。
4. **稳健与可控（开放式 RSI）** — 增益在固定资源与隐藏评测下持续、泛化，且不以评估污染、复杂度失控、对齐退化或不可审计为代价。

文内判断：**前两层已有证据；第三层尚无充分公开证据；第四层更远。**

### 前史与五次边界推进（压缩）

- **1981 EURISKO** — 规则自改 + 奖励黑客（把名字写进「发现者」）。
- **2017 AlphaZero** — 自对弈权重级闭环，但沙盒边界人类写死。
- **第一次（2022–23）Reflexion** — 情景记忆错题本；HumanEval pass@1 **91%**；参数未改。
- **第二次（2022–24）STaR / SPIN** — 自生成数据写进权重；人类仍定目标与过滤。
- **第三次（2022–25）CAI → Self-Rewarding → Meta-Rewarding** — AI 打分；Meta-Rewarding 后期「评委的评委」退化。
- **第四次（2023–26）驾具（harness）** — OPRO / ADAS / AFlow / Self-Harness / **DGM**（SWE-bench Verified **20%→50%**）/ MetaClaw / **AgentX**（线上 A/B）。
- **第五次（2025–26）学习与研究过程** — **SEAL**、**WebEvolver**、**Motus2**（真机 **65%→75%**）；OpenAI/Google 智能体参与模型研发但未证完整自主闭环。
- **2026 夏** — **Bilevel Autoresearch**、**RHI**、**AIDE²**（Weco：100 步重写内层 harness，称 Level 1 有界 RSI；**未过 ignition test**）；**AI4AI-Bench** 反向证据（最强 **0.250** / 1.0）。

### 四道门（文内统一视角）

1. 验证器可靠且系统改不动（隐藏测试、编译器、证明器）。
2. 改进能否跨出训练分布（模型坍缩、题库过拟合）。
3. 任务分数 ↑ ≠ 更会设计下一轮改进（ignition 未证）。
4. 能力、复杂度与控制能否同步扩张（AIDE² 死代码预警；AutoResearchEval 45 类失败）。

### 结语

有界、可测量的自我改进闭环已出现并可能净正收益，但**尚无充分公开证据**表明改进后的系统持续提升自身改进能力。

## 参考文献（文内列表，已整理为可点击）

见 [`sources/blogs/wechat_datawhale_rsi_survey_2026-09-19.md`](../blogs/wechat_datawhale_rsi_survey_2026-09-19.md) 末尾「推荐继续阅读」。
