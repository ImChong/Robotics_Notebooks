# NavGPT-3: Harnessing Context in a Hierarchical Navigation Runtime

- **类型：** 论文
- **作者：** Gengze Zhou 等
- **机构：** 阿德莱德大学澳大利亚机器学习研究所；Metacognition；澳大利亚国立大学；Roblox；北京大学；上海交通大学；北卡罗来纳大学教堂山分校；新南威尔士大学
- **论文：** <https://arxiv.org/abs/2610.10787>（v1：2026-10-07；v2：2026-10-12）
- **项目页：** <https://metacognitionai.github.io/NavGPT3/>
- **代码：** <https://github.com/metacognitionai/NavGPT-3>
- **入库日期：** 2026-10-29

## 摘要要点

- 用具备线程、上下文、工具和权限的运行时协调语言模型推理、动作执行与监控；中断和控制权切换用于响应事件。
- NavGPT VLA 使用 19.28M 样本训练，并按场景变化分配视觉 token。8B 单模型在 R2R-CE / RxR-CE Val-Unseen 的 SR 为 76.94% / 82.39%。
- 完整系统分别达到 81.51% / 90.43% SR；后者的人类参照采用离散导航，设置与 RxR-CE 不同。
- 论文报告动作策略单步 0.5–1 秒，相对语言模型决策 3–19 秒；这不是底层电机控制周期。

**对 wiki 的映射：** [NavGPT-3 实体页](../../wiki/entities/paper-navgpt-3.md)。

## 开放状态

截至 2026-10-29，项目页与 [官方仓库](../repos/navgpt-3.md) 提供 VLA 推理、harness、仿真评测和 4B / 8B 权重；Unitree 的 ZEOS 部署代码仍列为待发布。详见 [项目页归档](../sites/navgpt-3.md)。
