# Automating eval design and hillclimbing with Claude

> 来源归档

- **标题：** Automating eval design and hillclimbing with Claude
- **类型：** blog（claude.dev）
- **作者：** Lance Martin
- **链接：** https://claude.dev/blog/automating-eval-design-and-hillclimbing/
- **原文（用户给定）：** 同上
- **发布日期：** 2026-09-28
- **入库日期：** 2026-09-30
- **一句话说明：** 好 eval 四要素（分布像生产、强模型+高 effort 应更好、前沿模型仍有 passable headroom、低 run 方差）与 **对抗采样** 陷阱；`/claude-api build-eval` 与 `hillclimb` 工作流（输入审批、grader 校验、train/test 防过拟合、stall 时分桶根因）。
- **沉淀到 wiki：** 是 → [`wiki/concepts/ai-agent-evaluation.md`](../../wiki/concepts/ai-agent-evaluation.md)、[`wiki/entities/anthropic-claude-api-skill.md`](../../wiki/entities/anthropic-claude-api-skill.md)

---

## 好 eval 四要素

1. Task 分布像 **生产**（勿只选易生成/易打分题）。  
2. 更强模型 + 更多 thinking 应 **更高分**（否则 task/grader 有问题）。  
3. 最强配置应 **明显低于 100%** 且非「永远失败」的坏题。  
4. **低 run 方差**（题意清、grader 稳定、环境无泄漏、effort 一致）。

## 对抗采样

- 只挑「当前模型会挂」的题 → 测的是 **该模型失败指纹**，非任务内在难度。  
- 应包含：人判难例、生产/工单失败、手写 5–10 例 + 代码合成（锚定真实例）。

## build-eval 流程

- 采样顺序：生产 transcript（敏感/留存）→ bug/ticket → 手写 → 代码合成。  
- Grader：**程序化**（约束输出空间）优先；开放输出用 **LLM judge + 可检查 claim rubric**（非 1–5）；可 pairwise vs baseline。  
- 人工确认样本打分；baseline + 置信区间；诊断 grader 双跑一致性、超时/截断、~95% 饱和警告。

## hillclimb 原则

- 改 **便宜可回滚** 面：prompt、skills、工具描述、model/effort、（谨慎）harness。  
- **可归因** 指标（如 skill 触发率 ↔ skill 描述）。  
- **成本** 可在饱和 eval 上仍优化。

## 防过拟合

- Train / test 拆分；**勿把失败 transcript 粘贴进 prompt**；答案勿 structurally 可被模型直接读到。  
- 每轮：读 train 失败 → 单 patch → 若 train↑ test 平则 revert；stall 2–3 轮则 **按根因分桶**（含修 grader/题面）。

## 案例摘要

- **成本：** 44 ticket 客服；Opus 4.8 high 74.4% / 4.6¢ → audit + Opus 5.5 low 87.8% / 1.9¢ → Sonnet 5 low + 路由 98.9% / ~1¢；hold-out 90.5% vs 78.6%。  
- **性能：** claude-api skill 文档 eval 66%→~88%；补 8 个 API 特性节、修 C#/Java 表、priors→当前 API 对照表、修 grader/题面矛盾。

## 子命令

- `/claude-api build-eval` — 无 eval 时  
- `/claude-api hillclimb` — 已有 runnable eval
