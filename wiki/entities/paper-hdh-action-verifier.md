---
type: entity
project_id: hdh-action-verifier
tags: [paper, robotics, ingest-2026-11]
status: draft
updated: 2026-11-14
arxiv: "2611.14834"
project: https://wen-hanhsieh.github.io/hdh/
code: https://github.com/Wen-HanHsieh/hdh_real
related:
  - ./paper-carryon-dual-loop.md
  - ./paper-moss-mode-source-shaping.md
original:
  - ../../sources/papers/hdh-action-verifier_arxiv_2611_14834.md
sources:
  - ../../sources/papers/hdh-action-verifier_arxiv_2611_14834.md
  - ../../sources/sites/hdh-action-verifier.md
  - ../../sources/repos/hdh-action-verifier.md
summary: "冻结 VLA 的双候选动作验证"
---

# HDH：冻结 VLA 的低延迟动作验证

## 一句话定义

**从冻结 VLA 的自主执行成败训练紧凑验证器，以近期观测与动作历史给候选动作块评分，选择更可能成功的动作。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉语言动作策略 |
| HDH | Hindsight Decision Model with History | 基于历史与结果监督的动作验证 |
| LIBERO | Language-Instructed Benchmark for Robot Manipulation | 机器人操作评测 |

## 为什么重要

冻结 VLA 的双候选动作验证，关注现有控制、数据或机器人学习方法在复杂任务中的失效点。这里单独建论文实体，避免与其他论文共用详情页；当前是待完成全文与源码复核的草稿。

## 核心机制

冻结 π0.5 收集自主 rollouts，按回合终止成败标注当前观测与执行动作前缀；训练带历史条件的轻量 verifier；推理时采样两个候选块，按成功分数选择，并执行获选块的前 8 步。

## 方法流程（待全文逐节复核）

以下流程仅依据已核对的论文 HTML 方法描述，不代替源代码调用图。

```mermaid
flowchart LR
  obs[观测与历史动作] --> actor[冻结 VLA]
  actor --> choices[采样 2 个动作块]
  history[自主 rollout 终止成败] --> verifier[训练轻量验证器]
  obs --> verifier
  choices --> verifier
  verifier --> action[选高分候选前 8 步执行]
```

## 源码运行时序图

**待核实，不绘制未经代码阅读确认的时序图。** 官方存在公开仓库，但本次仅核对了 README 和部分发布说明；需继续阅读真实运行入口及关键模块后补充。

## 实验与评测

四个 LIBERO-10 任务平均成功率 82.3%；Franka Panda 三项真机任务的平均成功率由 44.7% 升至 67.3%（每方法每任务 50 次）。5.4% 额外延迟为仿真统计，真机延迟需要独立测量。

## 局限与工程实践

- 验证器依赖任务自身的 rollout 成败标签；候选只来自原冻结策略，无法保证发现策略从未采样的动作。
- **源码状态：** 已公开 Franka 真机代码；另外有 hdh_sim 仿真仓库。
- **原文状态：** 本轮已核查 arXiv 部分章节，尚未通读全部正文/附录，保持 draft，不宣称健康度满分。

## 结论

**HDH 的收益来自冻结策略之外的候选选择，而非修改主模型；必须对照同一候选集的随机选择。**

1. 使用与论文一致的任务、硬件和评测协议复验，不跨实验口径比较数据。
2. 把方法改善的指标与额外的数据、计算或控制成本分开记录。
3. 对未经独立核查的复现入口，暂不声称完成验证。

## 关联页面

- [CarryOn：具备记忆、安全与失败恢复的双循环机器人 Harness](./paper-carryon-dual-loop.md)
- [MOSS：从初始噪声选定 Flow Matching 动作模式](./paper-moss-mode-source-shaping.md)

## 参考来源

- [论文归档](../../sources/papers/hdh-action-verifier_arxiv_2611_14834.md)
- [官方项目页归档](../../sources/sites/hdh-action-verifier.md)
- [官方仓库归档](../../sources/repos/hdh-action-verifier.md)

## 推荐继续阅读

- [arXiv 原文](https://arxiv.org/abs/2611.14834)
- [官方项目页](https://wen-hanhsieh.github.io/hdh/)
- [公开仓库](https://github.com/Wen-HanHsieh/hdh_real)
