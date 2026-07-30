# Roboparty/INTACT-JEPA

> 来源归档（ingest 配套仓库 · RoboParty 组织镜像）

- **URL：** <https://github.com/Roboparty/INTACT-JEPA>
- **对应论文：** [arXiv:2607.26056](https://arxiv.org/abs/2607.26056)
- **规范仓（上游）：** <https://github.com/zju3dv/INTACT-JEPA> → 见 [intact-jepa.md](intact-jepa.md)
- **项目页：** <https://zju3dv.github.io/INTACT-JEPA/>
- **Lab：** <https://lab.roboparty.com/>
- **许可：** MIT（与上游一致）
- **入库日期：** 2026-07-30
- **最后更新：** 2026-07-30
- **一句话说明：** RoboParty 组织对 `zju3dv/INTACT-JEPA` 的 **fork 镜像**（作者单位含 RoboParty Lab）；内容与上游同为文档/站点预览，**训练代码 Coming Soon**。
- **代码：** <https://github.com/Roboparty/INTACT-JEPA>（fork；非独立可跑训练入口）

## 开源核查（步骤 2.5，2026-07-30）

| 项 | 状态 |
|----|------|
| GitHub API | `fork: true`，`parent` / `source` = `zju3dv/INTACT-JEPA` |
| 树内容 | 与上游同构：`docs/`（METHOD/RESULTS/RELEASE）、`assets/`、站点工具 `tools/*.py`；**无** `train`/`eval` 训练入口 |
| Code / Models badge | **Coming Soon**（`docs/RELEASE.md` Stage 0–2） |
| 结论 | **部分开源镜像**：便于从 [Roboparty](https://github.com/Roboparty) 组织导航；复现仍以规范仓 `zju3dv/INTACT-JEPA` 与 Stage 2+ 发布为准 |

## 与 Lab 叙事的关系

- README / README_CN 将作者单位标为 **RoboParty Lab**（与 ZJU CAD&CG、清华 AIR、InSpatio 并列）。
- 中文动机句与宣传口径一致：LeWM 类前向模型学「动作→效果」；INTACT 补「意图→动作」同构接口，使 Direct 条件均值成为无搜索策略（四任务宏 SR ~95.33%，延迟 2.9–5.5 ms）。

## 对 wiki 的映射

- [INTACT 论文实体](../../wiki/entities/paper-intact.md)
- [规范仓归档](intact-jepa.md)
- [论文归档](../papers/intact_arxiv_2607_26056.md)
- [RoboParty Lab 门户](../sites/lab_roboparty_com.md)
- [RoboParty 实体](../../wiki/entities/roboparty.md)
