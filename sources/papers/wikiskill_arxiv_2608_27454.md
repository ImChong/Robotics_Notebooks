# WikiSkill: Compiling Agent Experience into Persistent Knowledge for Skill Evolution（arXiv:2608.27454）

> 来源归档（ingest）

- **标题：** WikiSkill: Compiling Agent Experience into Persistent Knowledge for Skill Evolution
- **类型：** paper / agent-skills / skill-evolution / persistent-knowledge / LLM-agents
- **arXiv：** <https://arxiv.org/abs/2608.27454>（PDF：<https://arxiv.org/pdf/2608.27454.pdf>）
- **作者：** Liyan Tang、Cyrus Rashtchian、Chun-Sung Ferng、Andrew Tomkins、Da-Cheng Juan、Tu Vu（通讯）
- **机构：** 谷歌研究院（Google Research）；弗吉尼亚理工学院（Virginia Tech，Tu Vu）
- **入库日期：** 2026-08-31
- **一句话说明：** 在 agent 工作区引入 **持久 Wiki 层**，把执行轨迹编译为可复利知识，再驱动 **可执行 skill** 的提出与 validation gating；在五套 benchmark、五类推理模型上稳定优于 EvoSkill / SkillOpt / Trace2Skill，并展示跨模型 skill 迁移与「技能进化补模型规模」现象。

## 开源状态（步骤 2.5，2026-08-31）

- **项目页：** arXiv 预印本 **未列** 独立 `*.github.io` / lab 项目页链接。
- **官方代码：** 论文正文与 arXiv 页 **未列** Google Research 官方 GitHub / Hugging Face 发布；截至入库日视为 **未开源**（方法论文）。
- **社区实现（非官方）：** PyPI [`wikiskill`](https://pypi.org/project/wikiskill/) 指向第三方 [`ashutoshsinghpr7/wikiskill`](https://github.com/ashutoshsinghpr7/wikiskill)，自述对齐 Algorithm 1；**不等同**作者官方发布，本站不建 `sources/repos/` 归档，仅在 wiki 局限节作对照提示。

## 摘要级要点

- **问题：** 自动 skill 进化方法（EvoSkill、Trace2Skill、SkillOpt）把经验散落在优化历史里，难以跨迭代系统复用「为何改 skill、哪些失败模式复发」。
- **方法：** 工作区三层——`raw/` 不可变轨迹、`wiki/` 持久模式库与演化日志、`skills/` 可执行 `SKILL.md`；每轮 **Inference Agent → Wiki Maintainer → Skill Proposer → Gating/Rollback**，且 **wiki 永不回滚**。
- **主结果：** 五模型平均准确率 WikiSkill 均为最高；Qwen 族随规模增益递增（+12.3 / +17.5 / +23.9 pt）；Qwen-3.5-9B+WikiSkill **47.4%** 超过无 skill 的 Qwen-3.6-27B **39.4%**。
- **迁移：** 他模型演化 skill 常优于自演化；也存在模型特定 workaround 导致负迁移（如 SpreadSheet 上小模型 skill 伤害 Gemini）。
- **消融：** Skill Proposer 读持久 wiki 平均 +15.0 pt；训练 rollout 时让 Inference Agent 读 wiki 反而降分。

## 核心论文摘录（MVP）

### 1) 三层知识架构

- **链接：** §3.1；Fig. 2
- **摘录要点：**
  - **Raw Layer (`raw/`)：** 每轮训练轨迹，只增不改。
  - **Wiki Layer (`wiki/`)：** `patterns/*.md` 记录失败模式/成功策略；`logs.md` 演化叙事；`skill-impact.md` 记录 proposal diff、验证分与接受/拒绝审计。
  - **Skills Layer (`skills/`)：** 每 skill 含 `SKILL.md`（规程）与 `PURPOSE.md`（回溯 motivating wiki patterns）。
- **对 wiki 的映射：**
  - [WikiSkill（论文实体）](../../wiki/entities/paper-wikiskill.md) — 架构与运行时合约。
  - [LLM Wiki（Karpathy）](../../wiki/references/llm-wiki-karpathy.md) — 论文显式援引的「编译经验为持久知识」范式。

### 2) 演化闭环四组件

- **链接：** §3.2；Algorithm 1（附录）
- **摘录要点：**
  1. **Inference Agent：** 用当前 `S_{k-1}` 在训练集 rollout；**训练阶段禁止读 wiki**（消融验证）。
  2. **Wiki Maintainer：** 采样成功/失败轨迹，根因分析并 patch 更新 `patterns/` 与 `index.md`、`logs.md`。
  3. **Skill Proposer：** ReAct + `read_file` 按需读 pattern 与原始轨迹；每轮 **原子** 提案（新建或 patch 单一 skill）。
  4. **Gating：** 在 `D_val` 上评估候选 `S'_k`，仅当分数超过历史最佳才接受；拒绝则 skill 回滚，**wiki 保留**。
- **对 wiki 的映射：**
  - [WikiSkill](../../wiki/entities/paper-wikiskill.md) — 流程总览与工程读法。
  - [Superpowers（obra）](../../wiki/entities/superpowers-obra.md) — 对照「可执行 skill 文件 + 维护流程」生态。

### 3) 主实验与跨模型迁移

- **链接：** §4；Table 1–2
- **摘录要点：**
  - **Benchmark：** LiveMath、SealQA、SpreadSheet、OfficeQA、ALFWorld。
  - **模型：** Qwen-3.5-4B/9B、Qwen-3.6-27B、Gemma-4-31B、Gemini-3.5-Flash。
  - **对照：** Trace2Skill、EvoSkill、SkillOpt、无 skill；skill 全文注入 system prompt（与 prior work 对齐，隔离检索失败）。
  - **迁移：** Qwen-3.6-27B 演化 skill 使 Qwen-3.5-9B 在 SpreadSheet 上 24.3%→50.5%（自演化仅 33.6%）。
- **对 wiki 的映射：**
  - [Darwin Skill](../../wiki/entities/darwin-skill.md) — 另一类 skill 迭代优化（validation-gated edits，单 skill 域）。
  - [AI Auto-Research](../../wiki/concepts/ai-auto-research.md) — agent 经验复利与知识编译边界。

### 4) 消融与局限

- **链接：** §5.1；Limitations
- **摘录要点：**
  - 无 wiki 累积（去掉 Maintainer）时 Gemini 平均 48.7%；完整 WikiSkill 63.7%。
  - 局限：未评 skill 检索/触发；严格单调 validation gating 可能拒绝「中性但为后续铺路」的提案；wiki 无自动剪枝；未覆盖数百步超长程在线适应。
- **对 wiki 的映射：**
  - [WikiSkill](../../wiki/entities/paper-wikiskill.md) — 结论与部署边界。

## BibTeX

```bibtex
@misc{tang2026wikiskill,
  author       = {Liyan Tang and Cyrus Rashtchian and Chun-Sung Ferng and
                  Andrew Tomkins and Da-Cheng Juan and Tu Vu},
  title        = {{WikiSkill}: Compiling Agent Experience into Persistent Knowledge for Skill Evolution},
  year         = {2026},
  eprint       = {2608.27454},
  archivePrefix= {arXiv},
  primaryClass = {cs.AI},
  url          = {https://arxiv.org/abs/2608.27454}
}
```

## 对 wiki 的映射

- 升格实体页：[WikiSkill（论文实体）](../../wiki/entities/paper-wikiskill.md)
- 交叉补强：[LLM Wiki（Karpathy）](../../wiki/references/llm-wiki-karpathy.md)、[Superpowers（obra）](../../wiki/entities/superpowers-obra.md)、[Darwin Skill](../../wiki/entities/darwin-skill.md)
