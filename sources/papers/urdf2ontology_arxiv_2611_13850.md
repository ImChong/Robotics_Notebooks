# Taxonomy-Agnostic Neurosymbolic Grounding of Robot Kinematics｜URDF2Ontology 机器人运动学语义接地

- **类型：** paper
- **原文：** https://arxiv.org/abs/2611.13850
- **PDF：** https://arxiv.org/pdf/2611.13850
- **核对范围：** 核对论文 arXiv v1 摘要、用户 2026-11-12 资讯附件、官方源码 README 与公开入口；未通读全文方法、实验与附录，未执行实验。
- **通读：** 未完成
- **收录日期：** 2026-11-14
- **代码或数据：** https://github.com/RIS-WITH/urdf2ontology

## 核心摘录

1. 读取关节/连杆图和用户提供的 taxonomy。
   - **对 wiki 的映射：** [方法与评测](../../wiki/entities/paper-urdf2ontology.md)
2. 分支结构用 bottom-up Tree Track，串联结构用 top-down Chain Track。
   - **对 wiki 的映射：** [方法与评测](../../wiki/entities/paper-urdf2ontology.md)
3. 对已解析子图逐步清理，通过符号规则约束 LLM 输出语义能力本体。
   - **对 wiki 的映射：** [方法与评测](../../wiki/entities/paper-urdf2ontology.md)

## 作者报告的结果与限制

摘要称异构机器人上能抽取结构、实现 schema 对齐；文章提及五类机器人，复杂传感器和语义分类仍有局限。

URDF 缺少任务语义，LLM 语义推断可能幻觉；不等同自动认证机器人能力。

## 待补

需通读论文图表、正文和附录并逐项校验结果；当前内容为原始摘要及资讯整理，不替代原文。
