# ReFORM（mit-realm.github.io/reform）

> 来源归档（ingest）

- **标题：** ReFORM: Reflected Flows for On-support Offline RL via Noise Manipulation
- **类型：** project site
- **官方入口：** <https://mit-realm.github.io/reform/>
- **论文（OpenReview）：** <https://openreview.net/forum?id=YvFsyRReeN>
- **PDF：** <https://openreview.net/pdf?id=YvFsyRReeN>
- **代码：** <https://github.com/MIT-REALM/reform>
- **入库日期：** 2026-09-30
- **一句话说明：** ICLR 2026 **MIT REALM** 项目主页：BC flow + reflected noise 示意图、与 FQL/DSRL/IFQL 的 support 可视化对比、OGBench 四类环境 GIF、performance profile 与 IQM 结果、BibTeX。

## 源码开放核查（2026-09-30）

| 项 | 结论 |
|----|------|
| **代码** | **已开源** — 页内链至 <https://github.com/MIT-REALM/reform>（Jax 官方实现，README 含 train/test 与论文复现命令） |
| **权重** | 未在主页单独列出 checkpoint 托管；训练脚本本地产出 `logs/<env>/reform/...` |
| **数据** | 依赖 [OGBench](https://seohong.me/projects/ogbench/) 标准 offline 数据集（经 `singletask` 环境接口） |

## 页面结构要点

| 区块 | 内容 |
|------|------|
| Overview | support 约束、reflected flow 噪声、40 任务 + 固定超参 vs hand-tuned 基线 |
| Challenges | OOD、距离正则的局限、多模态表达需求 |
| Method | BC flow（有界超球源）+ reflected flow 噪声；与基线 action 分布对比图 |
| Tasks / Results | antmaze-large / cube-* / scene；clean vs noisy；profile + IQM |
| Abstract / BibTeX | 与 OpenReview 一致 |

## 对 wiki 的映射

- [ReFORM（论文实体）](../../wiki/entities/paper-reform-iclr-2026.md)
- [sources/papers/reform_iclr_2026.md](../papers/reform_iclr_2026.md)
