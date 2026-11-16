# Taxonomy-Agnostic Neurosymbolic Grounding of Robot Kinematics｜URDF2Ontology 机器人运动学语义接地

- **类型：** paper
- **原文：** https://arxiv.org/abs/2611.13850
- **PDF：** https://arxiv.org/pdf/2611.13850
- **核对范围：** arXiv:2611.13850v1 全文的可访问文本镜像：论文结构、图 2 流程说明、Phase I–III、实验与限制。原站 PDF 请求失败，未把镜像图示当作官方图像复现；另核对作者 GitHub README 的代码入口与 ROS 2 Ontologenius 前置条件。
- **通读：** 已完成
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

## 原文核对摘录（2026-11-14）

URDF2Ontology 输入 URDF 的 link/joint 图与用户指定的 TBox 类层次，将可推断的本体功能映射到 OWL/ABox 实例断言。图 2 表明三阶段循环：Phase I 提取尚未处理的子链，Phase II 用 Tree Track 自底向上聚合多分支的传感器、末端执行器与移动组件，用 Chain Track 自顶向下拆分串联肢体，Phase III 完成实体命名、拓扑检验与 ABox 写入；外层循环逐类处理并清掉已经标注的节点，降低 LLM 上下文负担。文章展示五类机器人形态，不代表对所有高密度传感器及语义角色都正确。官方仓提供 `scripts/main.py` 类入口及 ROS 2 Ontologenius 启动说明，依赖正确 URDF/本体与 LLM 配置；本轮没有运行。

## 复现边界

尚未独立运行数据、训练、评测或机器人试验；论文数字为作者报告。对 wiki 的映射：[论文详情](../../wiki/entities/paper-urdf2ontology.md)。
