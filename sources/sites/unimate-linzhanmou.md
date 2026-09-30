# UniMate（项目页 — linzhanmou.com）

> 来源归档（site / 步骤 2.5 核查）

- **标题：** UniMate — One Unified Model to Animate Diverse Skeletons
- **类型：** site（SIGGRAPH Asia 2026 项目页）
- **链接：** <https://linzhanmou.com/unimate/>
- **交互 Demo：** <https://linzhanmou.com/unimate/interactive.html>
- **论文：** <https://arxiv.org/abs/2609.05415>
- **入库日期：** 2026-09-30
- **一句话说明：** 展示跨拓扑 text 驱动骨骼动画、UniML3D 构建管线与零样本编辑/补间/扩展应用。
- **沉淀到 wiki：** [`wiki/entities/paper-unimate.md`](../../wiki/entities/paper-unimate.md)

## 开源状态（步骤 2.5，2026-09-30）

| 资源 | URL | 状态 |
|------|-----|------|
| 论文 PDF | arXiv:2609.05415 | **已公开** |
| **代码** | <https://github.com/Friedrich-M/UniMate> | **已开源**（训练 + 推理 + `data_process/`） |
| **数据集** | <https://huggingface.co/datasets/Linzhan/UniML3D>（及 Mixamo/Objaverse/Truebones 注释子集） | **已发布**（Truebones **动作本体** 需商业授权，页内说明） |
| **Checkpoints** | <https://huggingface.co/Linzhan/UniMate> | **Preview 权重已发布**（README 称持续同步） |
| 交互结果浏览 | interactive.html | **在线 Demo** |

- **核查结论：** 项目页 Footer / News 与 GitHub README 一致；2026-09-06 释训练推理代码，2026-08-30 释 UniML3D 与处理管线。

## 项目页核心摘录

1. **输入：** rigged 3D asset + text prompt → 单模型输出时序连贯、prompt 对齐的关节动画。
2. **UniML3D：** 13,006 sequences；七类拓扑 + articulated rigid；16 步 filter → annotate → canonicalize（项目页表格）。
3. **零样本任务：** motion editing（固定子集 joint）、in-betweening（起止 pose + prompt）、expansion（分段 prompt 续写）。
4. **Baselines 语境（论文）：** 相对 SMPL 系 T2M、AnyTop（需目标 skeleton 统计 + 无 text）、mesh 顶点回归等。

## 对 wiki 的映射

- [`wiki/entities/paper-unimate.md`](../../wiki/entities/paper-unimate.md)
- [`sources/repos/unimate-friedrich-m.md`](../repos/unimate-friedrich-m.md)
- [`sources/papers/unimate_arxiv_2609_05415.md`](../papers/unimate_arxiv_2609_05415.md)
