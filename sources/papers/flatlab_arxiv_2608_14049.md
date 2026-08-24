# FlatLab: A Unified Methodology Framework and Simulation-Based Benchmark for Robotic Manipulation of Flat Objects（arXiv:2608.14049）

> 来源归档（ingest）

- **标题：** FlatLab: A Unified Methodology Framework and Simulation-Based Benchmark for Robotic Manipulation of Flat Objects
- **类型：** paper / manipulation / benchmark / sim2real
- **arXiv abs：** <https://arxiv.org/abs/2608.14049>
- **PDF：** <https://arxiv.org/pdf/2608.14049>
- **项目页：** <https://flatlab-web.github.io/>（归档见 [`sources/sites/flatlab-web-github-io.md`](../sites/flatlab-web-github-io.md)）
- **机构：** 吉林大学、北京大学、中科院、伯明翰大学等
- **入库日期：** 2026-08-24
- **索引来源：** [具身智能小站 9 篇盘点](../blogs/wechat_embodied_station_9_papers_vla_predict_grasp_2026-08-24.md)

## 开源状态（步骤 2.5，2026-08-24）

- 项目页与摘要写明 code **将公开发布**（<https://flatlab-web.github.io/>），截至入库日 **无 GitHub URL**。
- **结论：** **待发布**。

## 摘录 1：方法

- **策略生成器** — 从点云学习策略中心、物体不变表示（对比学习 + 仿真数据变换）。
- **动作执行模块** — 长时序拆为可复用动作原语并动态组合轨迹。

## 摘录 2：FlatLab 基准

- Isaac Sim 上 **100+** 刚/可变形平面物体；自动多模态采集；标准任务与评测协议；一键部署脚本。

## 摘录 3：结论读法

- 未见物体/类别泛化优于单策略启发式与现有基线。

**对 wiki 的映射：** [`wiki/entities/paper-flatlab.md`](../../wiki/entities/paper-flatlab.md)；交叉 [Manipulation](../../wiki/tasks/manipulation.md)、[Sim2Real](../../wiki/concepts/sim2real.md)。

## 当前提炼状态

- [x] 项目页核查（待发布）
- [x] 升格 `wiki/entities/paper-flatlab.md`
