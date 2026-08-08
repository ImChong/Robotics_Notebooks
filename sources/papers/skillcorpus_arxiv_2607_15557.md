# SkillCorpus: Consolidating and Evaluating the Open Skill Ecosystem for Real-World LLM Agents（arXiv:2607.15557）

> 来源归档（ingest）

- **标题：** SkillCorpus: Consolidating and Evaluating the Open Skill Ecosystem for Real-World LLM Agents
- **缩写 / 框架：** **SkillCorpus**；三面质量（utility / robustness / safety）；16-class taxonomy；retrieve→rerank→LLM select
- **类型：** paper / llm-agents / agent-skills / corpus / retrieval
- **arXiv：** <https://arxiv.org/abs/2607.15557>（v5 2026-08-06；PDF：<https://arxiv.org/pdf/2607.15557>）
- **项目页：** 无独立项目页（截至入库日）
- **代码 / 数据：** 论文写 dataset、models、code **will be released upon acceptance**；公开检索无 SkillCorpus 仓（核查日 2026-08-08）
- **作者：** Yanze Wang\*、Pengfei Yao\*、Tianyi Sun、Chuanrui Hu†（project leader）、Yan Xiao、Xiaotian Luo、Yunyun Han、Yifan Chen、Jun Sun‡、Yafeng Deng‡（\* equal；‡ corresponding）
- **机构：** EverMind；盛大集团（Shanda Group）；北京大学（PKU）
- **入库日期：** 2026-08-08
- **一句话说明：** 把约 **82.1 万** 社区 `SKILL.md` 经六阶段漏斗策展为 **96,401** 可再分发技能语料，配微调检索-选择栈，在 SkillsBench / GDPVal / QwenClawBench 上相对无技能基线一致增益（SkillsBench 池化 **+7.5 pp**）。

## 开源状态（步骤 2.5）

- **项目页：** 无。
- **仓库核查（2026-08-08）：** GitHub 搜索 `SkillCorpus` 无匹配公开仓；承诺 acceptance 后释放 **OSI-permissive** 语料、微调检索栈与策展代码。
- **结论：** **宣称将开源 / 尚未发布**。wiki 不得写「语料已可下载」；源码运行时序图标 **不适用**。

## 摘录 1：问题与主张（§I / Abstract）

- **痛点：** 社区 `SKILL.md` 生态碎片化、冗余、质量不均；既有语料缺许可审计或未在真实任务+可部署检索栈上端到端评测。
- **主张：** **SkillCorpus** 统一聚合→策展→匹配→真实任务评测；释放许可审计后的大规模语料 + 微调 retrieve/rerank + LLM selector。
- **边界发现：** 增益受 **覆盖边界** 与 **harness 边界** 调制——无覆盖则 \(\Delta\approx 0\)（非负尾），同技能在 Raven vs OpenClaw 兑现差异大。

**对 wiki 的映射：** 升格 [`wiki/entities/paper-skillcorpus.md`](../../wiki/entities/paper-skillcorpus.md)；强链 [OpenClaw](../../wiki/entities/openclaw.md)、各 Agent Skills 实体、[HarnessBank](../../wiki/entities/paper-harnessbank.md)。

## 摘录 2：策展与检索栈（§III）

| 模块 | 要点 |
|------|------|
| **聚合** | 62 源注册表；~821k 原始文件；25,159 仓入六阶段漏斗 |
| **去重** | 精确指纹 + 语义余弦；边界对 LLM 裁决；\(283{,}844\to 101{,}111\) |
| **三面质量** | utility（描述）/ robustness（正文一致性）/ safety（危害）；19 flags；5 个硬门控 |
| **分数** | \(\mathtt{content\_q}=0.50u+0.35r+0.15s\) + 源先验收缩；安全边际衰减 |
| **Stage 5** | 安全硬门 + OSI 宽松许可 → **96,401** active set |
| **匹配** | Qwen3-Emb/Rank 0.6B 微调 recall/rerank → LLM selector 注入 0–2 技能 |

**对 wiki 的映射：** 实体页画「爬取→漏斗→检索→注入 harness」流程图；强调安全/许可门与 harness 兑现。

## 摘录 3：实验（§IV）

| 设定 | 读点 |
|------|------|
| **基准** | SkillsBench（87）、GDPVal（220）、QwenClawBench（100） |
| **Harness × 骨干** | OpenClaw / Raven × Qwen3.5-27B / 397B；Opus 4.7 稳健性检查 |
| **池化 \(\Delta\)** | SkillsBench **+7.5±2.3**；GDPVal **+1.51±0.49**；QwenClawBench **+2.79±0.70** |
| **最强单元** | Raven×397B SkillsBench \(9.2\to 22.6\)（单跑 +13.4） |
| **消融** | 换货架检索或原始爬取 → 约 14% pass（仍高于无技能 9.2，但远低于完整管线） |
| **覆盖** | 检索匹配分与 \(\Delta\) 正相关（\(r\approx 0.31\)–\(0.40\)）；薄覆盖类 \(\Delta\approx 0\) |

**对 wiki 的映射：** 「结论」写清：真影响是 **策展+检索的可部署社区技能层**；上限由覆盖与 harness 执行环共同决定。

## 局限

- 质量/安全以文本 LLM judge 为主，非沙箱执行验证。
- 高基线 LLM-judge 基准上单格噪声大，主结论在池化层。
- 英文主导快照；社区持续演化需周期重爬。
- 资源尚未公开发布。

## 建议 wiki 动作

- 新建 **`wiki/entities/paper-skillcorpus.md`**。
- 交叉更新 OpenClaw、Darwin/Nuwa/MattPocock Skills、HarnessBank。

## 当前提炼状态

- [x] 论文摘要填写
- [x] wiki 页面映射确认
- [x] 开源状态核查（acceptance 后开源）
- [ ] 官方语料/代码发布后补下载入口与源码运行时序图
