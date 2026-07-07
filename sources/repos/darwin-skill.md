# 达尔文.skill（alchaincyf/darwin-skill）

> 来源归档

- **标题：** 达尔文.skill（Darwin Skill）2.0
- **类型：** repo
- **作者：** 花叔 Huashu
- **链接：** https://github.com/alchaincyf/darwin-skill
- **安装：** `npx skills add alchaincyf/darwin-skill`
- **入库日期：** 2026-07-07
- **一句话说明：** 受 [karpathy/autoresearch](karpathy-autoresearch.md) 启发的 **Agent Skill 自主优化系统**：9 维加权评分 + 独立子 agent 评委 + 棘轮 keep/revert + 人在回路 checkpoint；v2.0 吸收微软 SkillLens / SkillOpt 论文。
- **为什么值得保留：** 把 autoresearch 的 **只保留可测量改进** 机制映射到 `SKILL.md` 优化域；与 [nuwa-skill](nuwa-skill.md)、[cangjie-skill](cangjie-skill.md) 形成 **造 skill → 进化 skill** 闭环；微软 SkillOpt 官方集成名单收录；对本站理解 **如何迭代 `schema/` 与 agent 规约文件** 有方法论参照。
- **沉淀到 wiki：** 是 → [`wiki/entities/darwin-skill.md`](../wiki/entities/darwin-skill.md)

## README 要点（归纳）

- **定位：** *像训练模型一样优化你的 Agent Skills* — 60+ skills 时手动维护不可持续，需 **结构质量 + 实际效果** 双重评估。
- **autoresearch 映射：**

| autoresearch | 达尔文.skill |
|:---|:---|
| `program.md` | 本 SKILL.md（评估标准与约束） |
| `train.py` | 待优化的 SKILL.md |
| `val_bpb` | 9 维加权总分（满分 100） |
| git ratchet | keep / revert |
| test set | `test-prompts.json` |
| 全自主 | **人在回路**（Skill 好坏比 loss 更微妙） |

- **五条原则：** 单一可编辑资产；双重评估（结构 + 实测）；棘轮机制；独立评分（禁止自改自评，SkillLens 实证 LLM 自评准确率 46.4%）；人在回路。
- **v2.0 升级（2026-05-28）：** 8→9 维（失败模式编码、可执行具体性、高风险行动黑名单）；多评委不复用、早停（单轮涨幅 <1 分）、干跑比例告警；8 条反例黑名单（含禁用 `git reset --hard` 回滚）。
- **优化五阶段：** Phase 1 基线评估 → Phase 2 单维度优化（🔴 CHECKPOINT）→ Phase 2.5 测试提示词 → Phase 3 回归测试（🛑 STOP）→ 下一 skill。
- **实测：** huashu-gpt-image 80.8→91.65；darwin-skill 自评 86.05→92.7。
- **生态：** [nuwa-skill](https://github.com/alchaincyf/nuwa-skill) 造 skill；**达尔文** 进化；[cangjie-skill](https://github.com/kangarooking/cangjie-skill) 蒸馏书。
- **学术引用：** SkillLens (arXiv:2605.23899)、SkillOpt (arXiv:2605.23904)、[karpathy/autoresearch](https://github.com/karpathy/autoresearch)。
- **协议：** MIT。

## 对 wiki 的映射

| 主题 | 目标页面 |
|------|----------|
| 实体页（主） | [`wiki/entities/darwin-skill.md`](../../wiki/entities/darwin-skill.md) |
| 灵感来源 | [`wiki/entities/karpathy-autoresearch.md`](../../wiki/entities/karpathy-autoresearch.md) |
| 造 skill | [`wiki/entities/nuwa-skill.md`](../../wiki/entities/nuwa-skill.md) |
| 蒸馏内容 | [`wiki/entities/cangjie-skill.md`](../../wiki/entities/cangjie-skill.md) |

## 与本站 sources 的其它锚点

- autoresearch 源归档：[karpathy-autoresearch.md](karpathy-autoresearch.md)
- 姊妹生态：[nuwa-skill.md](nuwa-skill.md)、[cangjie-skill.md](cangjie-skill.md)
