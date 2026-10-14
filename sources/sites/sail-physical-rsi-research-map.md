# Physical RSI — SAIL Research Lab 项目仓库

> 来源归档（GitHub 项目 / 研究地图）

- **标题：** Physical RSI — Physical Recursive Self-Improvement
- **类型：** GitHub 项目与文献地图
- **URL：** <https://github.com/SAIL-Research-Lab/Physical-RSI>
- **维护方：** SAIL-Research-Lab
- **仓库创建：** 2026-09-29（GitHub 仓库元数据）
- **核查日期：** 2026-10-14
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

## 与同名项目的区分

本仓库与香港大学 MMLab × Kinetix AI 的 **Physical RSI 1.0: Recursive Self-Harness for Scaling Embodied Skills** 是不同项目：后者是具体的 self-harness 系统与 RoboDojo-Sim 评测展示；本仓库则是 SAIL Research Lab 维护的递归改进研究地图和文献目录。名称相同，不应合并为同一实体。

## 来源与限制

- 本归档基于公开 README、仓库文件列表和 LICENSE；目录内容会随仓库更新。
- README 提到 accompanying paper，但当前 README 未提供可直接识别的论文标题或 PDF 链接；不据此虚构论文条目。
- 目录收录本身不代表其中每篇工作都证明了物理 RSI，也不代表作者对相关工作作出统一复现或独立验证。
