# SMART：Design Docs Are All You Need 论文来源归档

## 元数据

- **题目：** Design Docs Are All You Need: An AI-native Machine-Learning Performance Tool
- **系统：** SMART（symbolic ML performance-modeling library）
- **作者：** Samuel Kushnir, Kimia Noorbakhsh, Kavya Sreedhar, Liqun Cheng, Ming Liu, Parthasarathy Ranganathan, Mohammad Alizadeh, Fred Kjolstad, Suvinay Subramanian
- **单位（HTML 页作者栏）：** MIT（Noorbakhsh、Alizadeh）；Google（Sreedhar、Cheng、Liu、Ranganathan）；Stanford（Kjolstad）；Google DeepMind（Subramanian）；Kushnir 未列单位
- **预印本：** https://arxiv.org/abs/2609.05364（cs.PL，交叉 cs.AI，提交于 2026-09-04）
- **HTML：** https://arxiv.org/html/2609.05364v1
- **代码：** 摘要页与 HTML 页均未给出代码仓库或项目页；截至 2026-10-12 未核实到公开实现（SMART 主分支本身即设计文档，论文未声明开放）。
- **入库日期：** 2026-10-12（经 WaytoAGI 周报线索，技术事实取自 arXiv）

## 论文要解决的问题

ML 性能建模工具（分析屋顶线、通信、调度）的抽象会随每代硬件与模型架构失效，手工补丁累积技术债。作者的判断是：编码智能体已足够快、足够便宜，**从自然语言设计文档重新生成整套实现**的成本低于长期修补代码。因此需要一种让“重新生成”可靠的文档与 IR 组织方式。

## 方法与训练信息

1. **仓库形态：** 主分支几乎不含代码，是一组 markdown 设计文档构成的有向无环图（约 50 篇、约 9,000 行规格说明）。依赖边由只读智能体推断，不手工维护。
2. **再生流水线：** 一个 orchestrator 按拓扑序遍历文档图，给每篇自包含文档分配一个编码子智能体；基础文档可路由给更强的 LLM（动态模型路由）。全量重建使用 Claude Code。中央日志记录子智能体在哪些文字上卡住、哪里出 bug，用来指导人修改文档。人只改文档，不改代码。
3. **文档写法：** 以逐步“工作示例”充当上下文内演示；含数字的文档末尾附“对账锚点”（reconciliation anchor）——一个小预设加精确期望输出，由生成的测试强制检查。
4. **算子 IR：** 递归定义的 Op，含符号输入输出、SymPy 成本表达式（计算/访存/通信）、资源预约表（RRT）和参数；参数三类：InnerLoop、GraphParams、LeafParams。叶子面向 TPU（MXU 矩阵块、VMEM 加载、ICI all-gather）。构建 DSL 是嵌入 Python 的追踪 DSL（`@smart_loop`、`@smart_map_loop`），分布式由分片标注推断集合通信。
5. **两种评估模式：** 快速分析模式（闭式汇总 + 屋顶线式重叠变换，适合扫数千设计点）；慢速模调度模式（迭代 modulo scheduling 对 RRT 排程，把达成的 initiation interval 逐层汇总）。
6. **校验：** 参考模型对账、参数护栏、单元测试和修复步骤。

## 论文报告的结果

- 再生实现与手工审计的参考模型“复现到舍入精度”，包含 DeepSeek-V3 在 TPU pod 切片上的服务建模；HTML 页未给出误差数值。
- 全新重建耗时约 1.5–3 小时，API 费用约 100 美元（约占高档订阅周额度的 20%）。
- 未报告：token 数、代码行数、逐文档错误率、与其他方法的对比。

## 复现与解释边界

- 结论基于单一库（SMART）与作者自有参考模型，缺少对照与第三方复现。
- “舍入精度”为作者陈述，页内无量化误差。
- 无公开代码，外部无法重跑再生流水线；结果依赖特定编码智能体与订阅额度。
- 未读到独立的 limitations 章节（HTML 页无）。

## Robotics_Notebooks 映射

- 详情节点：wiki/entities/paper-smart-design-docs-ai-native-performance-model.md
- 概念入口：wiki/concepts/agentic-coding-software-fundamentals.md、wiki/concepts/recursive-self-improvement.md

## 一手来源

- arXiv 摘要：<https://arxiv.org/abs/2609.05364>
- arXiv HTML：<https://arxiv.org/html/2609.05364v1>
