# Physical RSI — SAIL Research Lab 项目仓库

> 来源归档（GitHub 项目 / 研究地图）

- **标题：** Physical RSI — Physical Recursive Self-Improvement
- **类型：** GitHub 项目与文献地图
- **URL：** <https://github.com/SAIL-Research-Lab/Physical-RSI>
- **维护方：** SAIL-Research-Lab
- **仓库创建：** 2026-09-29（GitHub 仓库元数据）
- **核查日期：** 2026-10-14；2026-11-07 重新浅克隆（HEAD `e365a418`，2026-11-02）并通读，见「2026-11-07 复核」
- **核对范围：** 2026-11-07 读了 `README.md` 全文（约 70 KB：最新论文表、Evolution 引言、Boundary、L0–L5、五大主题说明、全部 325 条目录条目的标题与一句话注记、Open Questions、Citation、License）、`docs/model_training.md` 全文（约 26 KB，四节训练专题）、三张 README 图（overview、update-targets-and-tasks、verification-evaluation，用图像阅读）、`docs/paper-reference-index.md` 的文件头与 Coverage notes 以及对全部 276 个条目的「标题 / 目录归属 / 论文引用小节」逐条提取表。**未逐条读** 该索引中每条的 bibliography 原文块（作者全名与链接，共 5450 行，为稿件参考文献的镜像）；**未读** 目录中任一被收录论文本身（只读到仓库对它们的一句话描述）；**未读** 配套论文 arXiv:2610.18842 全文（本归档的对象是仓库，不是该论文）
- **通读：** 已完成
- **许可：** MIT
- **一句话说明：** 面向具身系统递归改进研究的证据可审计阅读地图，整理论文目录、L0–L5 能力分类、harness 适应、模型训练、安全、验证与评测资源；它是研究索引与框架资料，不是一个机器人策略或单项实验论文。

## 资源入口

| 资源 | 链接 | 说明 |
|------|------|------|
| 项目仓库 | <https://github.com/SAIL-Research-Lab/Physical-RSI> | 主入口与 README |
| 论文索引 | <https://github.com/SAIL-Research-Lab/Physical-RSI/blob/main/docs/paper-reference-index.md> | 维护的参考文献索引 |
| 模型训练专题 | <https://github.com/SAIL-Research-Lab/Physical-RSI/blob/main/docs/model_training.md> | 模型训练与适应专题材料 |
| License | <https://github.com/SAIL-Research-Lab/Physical-RSI/blob/main/LICENSE> | MIT；外链论文及第三方材料各循其许可 |

## README 要点（2026-10-14 核查）

1. **范围：** 收集 Physical Recursive Self-Improvement（Physical RSI）相关的具身研究，强调物理交互对后续改进机制的因果作用，以及被采纳改动的验证、保留和继承。
2. **证据边界：** 仅有更多数据、单次失败恢复或策略微调，不自动等于 RSI；重点在于保留的变化是否影响未来改进的生成、评估或选择机制。
3. **能力分类：** L0–L5 同时区分更新对象（执行系统或改进程序）与收益迁移范围（同任务或跨任务）；L5 还要求在匹配资源预算和受保护的跨任务评估下，后续改进能力持续变好。
4. **研究地图：** 将相关工作组织为 harness / execution、模型训练与适应、安全与保障、验证与评测、基准 / 数据集 / 平台等主题。
5. **目录规模：** README 徽章标示 301 项 works/resources；这是维护者的目录规模说明，不代表统一评测或质量排名。

## 2026-11-07 复核（与 2026-10-14 版有出入）

- **规模与配套论文：** README 徽章现为 **325** works/resources（此前核查为 301）；README 明确列出配套论文 [arXiv:2610.18842](https://arxiv.org/abs/2610.18842) *Who Improves the Robot? Toward Physical Recursive Self-Improvement*（Hai Zhou 等 35 位作者，通讯联系 shangding.gu@sjtu.edu.cn）。README 自称与稿件 v35 对齐；`paper-reference-index.md` 对齐 v34（54 页，参考文献 292 条合并为 278 条，另有 2026-10-18 起的增补不在 v34 编号内）。
- **框架：** 执行系统 M=(Θ,H)（参数化模型 + harness）与改进过程 Φ=(Adapt, Train, Ver)；图中另有 Evaluator E、经验反馈 D、Safety and Control。L1/L3 为 capability-level 更新（同任务 / 跨任务），L2/L4 为 meta-level 更新，L5 为在 E 下验证的递归增益。验证器含候选构建、实现检查、有效性检查、保持性检查（不损伤旧能力）与元验证。图 (b) 说明「新场景不等于新任务」：同一任务集 κ 相同、ξ 不同，而同一批技能换顺序属不同任务。
- **目录结构：** 六大主题——基础与综述、Agent Harness 与执行、模型训练与适应、安全与保障、验证与评测方法、基准 / 数据集 / 平台（含 RSI 相关基准）；`model_training.md` 把训练拆成数据准备、训练方法、学习模型、知识迁移与训练程序改进四节。
- **最新论文表（README 页首，2026-09-29 至 10-13）：** EmbodiedEvo、RSI-Anything、Embodied Turing Machines、EmbodiedRSI、Video Prediction Policy 2、PhysEvo、RoboHarn-Evo。
- **开放问题（README 引稿件第 7 节）：** 不确定物理经验中的学习信号、harness 组件的因果归因、改进回路的非平稳性、奖励投机与评测泄漏、递归更新中的安全、对「已验证的递归增益」做基准。
- **许可：** MIT；README 称外链论文与第三方材料各循其许可。

## 与同名项目的区分

本仓库与香港大学 MMLab × Kinetix AI 的 **Physical RSI 1.0: Recursive Self-Harness for Scaling Embodied Skills** 是不同项目：后者是具体的 self-harness 系统与 RoboDojo-Sim 评测展示；本仓库则是 SAIL Research Lab 维护的递归改进研究地图和文献目录。名称相同，不应合并为同一实体。

## 来源与限制

- 本归档基于公开 README、仓库文件列表和 LICENSE；目录内容会随仓库更新。
- README 提到 accompanying paper，但当前 README 未提供可直接识别的论文标题或 PDF 链接；不据此虚构论文条目。
- 目录收录本身不代表其中每篇工作都证明了物理 RSI，也不代表作者对相关工作作出统一复现或独立验证。
