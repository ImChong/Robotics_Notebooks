# graph-robots/graph-as-policy — GaP 官方代码仓库

> 来源归档

- **标题：** Graph-as-Policy（GaP）官方实现
- **类型：** repo
- **组织：** graph-robots（UC Berkeley / NVIDIA / CMU / Bosch 署名网络）
- **代码：** <https://github.com/graph-robots/graph-as-policy>
- **项目页：** <https://graph-robots.github.io/gap/>
- **技能库：** <https://github.com/graph-robots/open-robot-skills>（MORSL / Anthropic Agent Skills 格式，与主仓并列 clone 自动发现）
- **控制器：** <https://github.com/graph-robots/controllers>（Franka / 双臂 I2RT YAM 等真机与遥操作接口）
- **论文：** arXiv:2607.05369
- **入库日期：** 2026-07-08
- **许可证：** Apache-2.0
- **一句话说明：** GaP Beta 代码：自然语言 VA 任务 → 类型检查计算图 → 仿真自学习改图 → edge 解释器在 LIBERO 仿真与 Franka/UR5 真机执行；`gap.agent` / `gap.execute` / `gap.viz` 与 `open-robot-skills` 技能包并列部署。

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [GaP 论文实体](../../wiki/entities/paper-gap-graph-as-policy.md) | 论文 Graph-as-Policy + MORSL + 8 VA benchmark 的工程入口 |
| [变体自动化（VA）](../../wiki/concepts/variational-automation.md) | FA/VA/GR 任务谱与 benchmark 族 |
| [ASPIRE](../../wiki/methods/aspire.md) | 姊妹 agentic code-as-policy 路线（程序技能库 vs 计算图） |
| [VLA](../../wiki/methods/vla.md) | GaP staging 与 MORSL 内 model-free 原语 |

## 为何值得保留

- **2026-07-01 Beta 发布**：`graph-as-policy` 主仓 + `open-robot-skills` 技能仓已公开；项目页 `graph-robots/gap` 仅为静态站。
- 复现 **LIBERO quickstart、杂货打包 workflow.json、仿真排练与真机 strip demo** 的直接入口。
- 与 [Humanoid_Robot_Learning_Paper_Notebooks](https://github.com/ImChong/Humanoid_Robot_Learning_Paper_Notebooks) 单篇深读互补：本库负责 **跨主题 agentic manipulation / VA** 索引。

## 对 wiki 的映射

- [paper-gap-graph-as-policy.md](../../wiki/entities/paper-gap-graph-as-policy.md)
- [gap_arxiv_2607_05369.md](../papers/gap_arxiv_2607_05369.md)
- [gap-graph-robots-project.md](../sites/gap-graph-robots-project.md)
