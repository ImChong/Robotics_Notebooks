# From Where to How: Continuous 4D Interaction Forecasting from Egocentric Video（arXiv:2609.08636）

> 来源归档（ingest）

- **标题：** From Where to How: Continuous 4D Interaction Forecasting from Egocentric Video
- **短名：** Coherent4D / HIGFlow
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2609.08636>
- **PDF：** <https://arxiv.org/pdf/2609.08636>
- **项目/代码：** <https://corrineqiu.github.io/from-where-to-how/>
- **入库日期：** 2026-09-09
- **索引来源：** [sources/blogs/wechat_embodied_station_visual_focus_10_papers_2026-09-09.md](../blogs/wechat_embodied_station_visual_focus_10_papers_2026-09-09.md)
- **一句话说明：** Coherent4D + HIGFlow（arXiv:2609.08636）：233K 样本连续 4D 交互预测数据集；先预测未来 3D 手部位点再条件化 residual flow matching 生成全身姿态。

## 开源状态（步骤 2.5，2026-09-09）

- **结论：** **待核实** — 项目/仓库见 `https://corrineqiu.github.io/from-where-to-how/`。
- **核查：** 公众号列 GitHub 404 或项目页未给可运行入口；复现前需再核。

## 核心摘录（面向 wiki 编译）

- Coherent4D：233,828 样本，Cooking/Health/Bike Repair 三域
- Stage1 Qwen3-VL + V-JEPA 预测连续手部位点；Stage2 flow matching 姿态
- 三域 interaction location 与 pose 指标均优于 FIction 等基线

**对 wiki 的映射：** [paper-from-where-to-how](../../wiki/entities/paper-from-where-to-how.md)
