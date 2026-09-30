# PPTBench: Can Coding Agents Reconstruct the Visual World through Structured, Editable Slides（arXiv:2609.29718）

> 来源归档（ingest）

- **标题：** PPTBench: Can Coding Agents Reconstruct the Visual World through Structured, Editable Slides
- **短名：** PPTBench
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2609.29718>
- **项目页：** <https://lab.einsia.ai/pptbench>
- **代码：** <https://github.com/Einsia/PPTBench> — [`sources/repos/pptbench.md`](../repos/pptbench.md)
- **评测 Session 示例：** <https://agent-git.com/@einsia/ppt-bench>
- **机构：** Einsia.AI / Navers Lab；清华大学（Tsinghua University）
- **入库日期：** 2026-09-30
- **一句话说明：** 500 项科学流程图→单页可编辑 PPTX 重建任务 + 四阶段 Agentic Judge；测 coding agent 从视觉结构到程序化对象图的端到端能力，最佳配置 Kimi K3 仅 67.80 分。

## 开源状态（步骤 2.5，2026-09-30）

- **已开源**：GitHub `Einsia/PPTBench` 提供 benchmark 材料化、Judge 与评测协议；源论文像素不随仓再分发。

## 核心摘录（面向 wiki 编译）

- 任务：给定 arXiv 论文流程图 raster，重建为 **native editable** 单页 PPTX（禁止贴参考图当退化解）。
- 500 任务经自动检索 + 人工筛 frozen；覆盖 50 个 arXiv 主类 / 10 展示域；难例含多 panel、层级块、密连线、矢量图标。
- Judge：artifact 有效性（确定性）× 语义正确 × 渲染质量 × 细粒度视觉；**乘法门控**，语义错即零分。
- 31 配置 / 46.5k 判决：artifact 失败仅 2.08%，**70.43%** 在可读 deck 后丢语义；文本细节占扣分 51.6%。
- **对 wiki 的映射：** [paper-pptbench](../../wiki/entities/paper-pptbench.md)；与 [RLE-Bench](../../wiki/entities/rle-bench.md) 同属 coding agent 评测轴（视觉结构化输出 vs 机器人学习工程）
