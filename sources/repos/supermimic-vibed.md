# supermimic_vibed

> 来源归档

- **标题：** SuperMimic reproduction on Isaac Lab 3（仓库名：supermimic_vibed）
- **类型：** repo（非官方独立复现）
- **链接：** <https://github.com/zita-ch/supermimic_vibed>
- **论文：** <https://arxiv.org/abs/2610.28480>（SuperMimic）
- **机构：** 作者标注 ETH Zurich（个人项目；README 声明与论文作者无隶属关系）
- **许可证：** MIT
- **入库日期：** 2026-10-22
- **再核日期：** 2026-10-22
- **一句话说明：** 基于 Isaac Lab 3 kit-less 与 MuJoCo-Warp 重实现 SuperMimic 的 G1/LAFAN1 动作跟踪实验，附训练、评测、渲染和失败分析工具。
- **沉淀到 wiki：** [SuperMimic](../../wiki/entities/paper-supermimic.md)

---

## 开源状态与结果声明（2026-10-22）

| 项 | 状态 |
|---|---|
| 实现代码 | **已开源**；仓库含 supermimic/ 算法模块及 tools/、训练和评测脚本 |
| 运行环境 | Linux、NVIDIA GPU、Docker 与 Isaac Lab 3 kit-less 镜像；LAFAN1/G1 资产需另行下载 |
| README 报告结果 | RTX PRO 6000 上训练 1.47 小时，40 段 LAFAN1 动作评测回报 0.829、失败 0 段；这是仓库作者自报结果，尚未独立复核 |
| 作者自述限制 | README 称项目由 AI 助手在约 1 小时内快速生成，代码未经审查，并提醒使用者自行承担风险 |

## 实现线索

README 将论文方法映射到 supermimic/robot.py、motion.py、mdp.py、env.py、networks.py、train.py 与 evaluate.py。流程采用 Isaac Lab 3 的 kit-less Docker 环境、MuJoCo-Warp solver，运动数据和 G1 模型从外部数据集下载后生成仿真资产。它适合作为快速阅读与复现实验入口；引用其回报数据时需连同硬件、环境和作者自述的审查边界一起说明。

## 对 wiki 的映射

- [SuperMimic：论文与研究实体](../../wiki/entities/paper-supermimic.md) — 与官方论文共用实体节点，明确区分官方论文与个人复现。
- [SuperMimic 论文来源](../papers/supermimic_arxiv_2610_28480.md) — 原始方法与评测主张。
- [官方项目页](../sites/supermimic-project.md) — 官方项目页未列作者维护代码；本仓库不是论文官方实现。
