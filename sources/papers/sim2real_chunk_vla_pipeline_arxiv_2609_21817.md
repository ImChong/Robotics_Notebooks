# Sim2Real Chunk VLA Pipeline（arXiv:2609.21817）

> 来源归档（paper）

- **标题：** A Sim-to-Real Integration Pipeline for Training and Deployment of Chunk-Based VLA Manipulation Policies
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2609.21817>
- **PDF：** <https://arxiv.org/pdf/2609.21817>
- **代码：** https://gitlab.isir.upmc.fr/kappel/sim2real_public_chunk_control
- **入库日期：** 2026-09-21
- **一句话说明：** 仿真生成 expert trajectories，真机 Franka FR3 open-loop replay 记录视觉/本体感知，同一部署栈闭环评估 chunk VLA；配对数据直接量化 sim-to-real gap。

## 开源状态

- **已开源**（步骤 2.5 核查，2026-09-21）
- 项目页/arXiv 截至入库日的可运行代码链接核查结论见上。

## 核心摘录

1. **策展档位：** 扫读（[具身小站 10 篇盘点](../blogs/wechat_embodied_station_contact_rich_sim_10_papers_2026-09-21.md)）
2. **机构：** 索邦大学 ISIR（UPMC）
3. **导读要点：** chunk VLA 的 sim-to-real gap 缺少统一硬件闭环测量；该协议用同一栈贯通训练与部署评测。

## 对 wiki 的映射

- [paper-sim2real-chunk-vla-pipeline](../../wiki/entities/paper-sim2real-chunk-vla-pipeline.md)
- [10 篇技术地图](../../wiki/overview/contact-rich-sim-10-papers-technology-map.md)
