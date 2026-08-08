# HarnessBank: Semantic Gene-Bank Search with Gated Verification for Agent-Harness Self-Evolution（arXiv:2607.13683）

> 来源归档（ingest）

- **标题：** HarnessBank: Semantic Gene-Bank Search with Gated Verification for Agent-Harness Self-Evolution
- **缩写 / 框架：** **HarnessBank**；**HGB**（Harness Gene Bank）；**Gated Harness Screening**
- **类型：** paper / llm-agents / agent-harness / self-evolution / quality-diversity
- **arXiv：** <https://arxiv.org/abs/2607.13683>（v2 2026-07-30；PDF：<https://arxiv.org/pdf/2607.13683>）
- **项目页：** 无独立项目页（截至入库日）
- **代码：** 论文写 “code will be publicly available upon acceptance”；GitHub 检索无公开仓（核查日 2026-08-08）
- **作者：** Xiaotian Luo、Dizhan Xue、Fengxingyu Wang、Chuanrui Hu、Yafeng Deng\*（\* corresponding）
- **机构：** EverMind / 恒心智能（EverMind）；盛大集团（Shanda Group）
- **入库日期：** 2026-08-08
- **一句话说明：** 在冻结任务模型权重下，用独立 evolver + **语义 Harness Gene Bank** + **门控筛选**做可信 agent-harness 自进化，七基准测试 Pass@1 提升约 **5.1%–15.4%**。

## 开源状态（步骤 2.5）

- **项目页：** 无。
- **仓库核查（2026-08-08）：** GitHub 搜索 `HarnessBank` 无匹配公开仓；摘要承诺 acceptance 后开源。
- **结论：** **宣称将开源 / 尚未发布**。wiki 不得写「已可复现」；源码运行时序图标 **不适用**。

## 摘录 1：问题与主张（§I / Abstract）

- **痛点：** Agent 表现由 harness（prompt / 知识 / 工具 / 控制环 / 恢复策略 / 配置）主导；现有自进化常贪心提拔少量子编辑，且用单次增益或模型自评筛候选 → **搜索坍缩、任务过拟合、增益不可信**。
- **主张：** **HarnessBank** 分离 task agent 与 evolver；用语义坐标的 **Harness Gene Bank** 保留多样高质 harness；用 **Gated Harness Screening**（有效性→激活→配对显著性→增益）廉价筛子代后再全训练集评估。
- **结果概览：** 七域测试 Pass@1 相对 vanilla **+5.1%–15.4%**；跨模型实验显示进化 harness **模型特异**而非普适最优。

**对 wiki 的映射：** 升格 [`wiki/entities/paper-harnessbank.md`](../../wiki/entities/paper-harnessbank.md)；与 [OpenClaw](../../wiki/entities/openclaw.md)、[Darwin Skill](../../wiki/entities/darwin-skill.md)、[AI Auto-Research](../../wiki/concepts/ai-auto-research.md)、[SkillCorpus](../../wiki/entities/paper-skillcorpus.md) 互链。

## 摘录 2：方法栈（§III）

| 模块 | 要点 |
|------|------|
| **形式化** | \(A_H=M\circ H\)；\(H=\mathcal{K}\cup\mathcal{X}\)，只改 mutable surface \(\mathcal{X}\) |
| **语义细胞** | 坐标 \((w,y)\)：\(w\in\{\texttt{prompt},\texttt{knowledge},\texttt{runtime},\texttt{config}\}\)，\(y\) 为失败病理（如 thinking-runaway） |
| **遗传/变异** | 失败轨迹驱动 **reinvent**；跨细胞机制 **recombine**；质量偏置选亲本 |
| **门控筛选** | 子集 rollout → validity / activation / paired significance / gain → 全训练集再评 → 竞争入银行 |
| **对照** | 同协议下相对 GEPA（prompt-only）、DGM（开放自改但无门控）更可信；TB2 消融显示去掉 \(2\sigma\) 会引入 false elites 且停不下来 |

**对 wiki 的映射：** 实体页画「诊断→生成→门控→入银行」流程图；强调 **可验证增益** 与 **语义多样性** 两轴。

## 摘录 3：实验（§IV）

| 设定 | 读点 |
|------|------|
| **骨干** | 主实验冻结 Qwen3.6-27B 作 task + proposer |
| **七基准** | TB2、LiveCode、Omni-MATH、BrowseComp+、GDPval、AppWorld、SWE-bench |
| **主表（Test Pass@1）** | 如 AppWorld **+15.4**、BrowseComp+ **+13.9**、LiveCode **+13.7**；SWE-bench 测试 \(n{=}26\) 记为 preliminary（+5.1 未过 bar） |
| **病理→补丁** | thinking-runaway → selective recovery；premature finalize → verify-finalize；跨模型匹配律（错配补丁近零或有害） |
| **消融（TB2）** | 完整 HarnessBank：Test 45.4、0 false elites、10 轮停；无 \(2\sigma\)：false elites↑、跑满 cap |

**对 wiki 的映射：** 「结论」写清：真影响是 **模型特异 harness 修正 + 可统计验证**；代价是 rollout 预算与代码未发布。

## 局限

- 进化依赖可打分训练任务与确定性评估器；小 \(n\) 域（SWE-bench）统计力不足。
- `why` 病理标签由 LLM 假设，错标最多浪费候选（门控兜底），但探索效率会受损。
- 代码尚未公开，无法核对 evolver prompt 与门控实现细节。

## 建议 wiki 动作

- 新建 **`wiki/entities/paper-harnessbank.md`**（源码时序图标不适用）。
- 交叉更新 OpenClaw / Darwin / AI Auto-Research / SkillCorpus。

## 当前提炼状态

- [x] 论文摘要填写
- [x] wiki 页面映射确认
- [x] 开源状态核查（acceptance 后开源）
- [ ] 官方代码发布后补源码运行时序图
