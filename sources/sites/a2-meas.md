# A²-MEAS 官方项目主页

> 来源归档（site）；核查日：2026-11-11

- **项目：** A²-MEAS / Toward the Era of Agentic Embodied Intelligence
- **原文：** <https://a2-meas.github.io/>
- **论文：** <https://arxiv.org/abs/2611.09124v1>
- **代码：** <https://github.com/AutoLab-SAI-SJTU/A2MEAS>（默认 `evaluation` 分支）
- **模型权重：** <https://huggingface.co/shibohuang/policy-weights>
- **机构：** 上海交通大学人工智能学院 AutoLab、Anyverse Dynamics 研究实验室
- **核对范围：** 官方项目页 System、Model、Infra、Simulation、Real World、Authors 和 Citation 全部正文，及跳转的 GitHub README 与公开代码入口。
- **通读：** 已完成

## 官方架构

项目将 MetaBrain 的任务级推理、感知/交互/可行性诊断代理、纠错上下文编码器与残差 Adapter 接入 A²-WLA 连续动作策略。动作策略使用近期多视角历史、运动感知特征及未来潜空间场景特征；恢复时记录已执行控制而不是仅使用未经执行的动作提案。

## 论文报告数据（不是当前开源仓库复现表）

| 设置 | 结果 |
|---|---|
| BEHAVIOR 2026，公开 100 任务 × 10 实例 | 完整系统 Q-score **0.6411**，完整成功率 **45.40%** |
| RoboTwin 2.0 | A²-WLA **93.98%**，完整系统 **95.05%** |
| 真机 12 任务，关闭高层在线纠错 | A²-WLA **57.92%**，π0.5 **26.67%** |
| 咖啡任务回退后的两步恢复 | 基础重试 **14%/8%**，历史 Adapter **94%/82%** |

项目页说明：50 任务子集的 0.5755 与 2025 冠军 0.2605 属于不同基准版本与训练配置，不能当作严格同协议对照。

## 开放程度与复现边界

官方仓库现已发布模型推理、BEHAVIOR 2026 和 RoboTwin 2.0 评测代码及权重入口；README 的最新发布版复现表写明 **RoboTwin 95.05%**、**BEHAVIOR Q-score 0.5287 / 成功率 32.40%**，与论文完整系统的 BEHAVIOR 0.6411 / 45.40% 不同。README 明确：此次只包含一个 Adapter，完整 Adapter 配置及部分任务特化执行策略（如导航流程）未发布；训练脚本和完整训练基础设施亦非 README 所列公开复现路径。不可将论文整套系统的成绩当作当前开源版本复现值。

## 交叉引用

- [原论文归档](../papers/a2-meas-agentic-embodied-intelligence_arxiv_2611_09124.md)
- [官方源码归档](../repos/a2meas.md)
- [唯一主 Wiki 节点](../../wiki/entities/paper-a2-meas-agentic-embodied-intelligence.md)
