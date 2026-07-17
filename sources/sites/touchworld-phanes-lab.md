# phanes-lab.github.io/TouchWorld-website（TouchWorld 项目页）

- **标题：** TouchWorld: A Predictive and Reactive Tactile Foundation Model for Dexterous Manipulation
- **类型：** site / project-page
- **URL：** <https://phanes-lab.github.io/TouchWorld-website/>
- **入库日期：** 2026-07-16
- **配套论文：** [TouchWorld（arXiv:2607.07287）](https://arxiv.org/abs/2607.07287) — 归档见 [`sources/papers/touchworld_arxiv_2607_07287.md`](../papers/touchworld_arxiv_2607_07287.md)
- **机构：** 哈尔滨工业大学（深圳）；PHANES AI
- **通讯作者：** Shuo Yang（[主页](https://homepage.hit.edu.cn/yangshuohit)）
- **代码 / 数据（截至 2026-07-16 项目页核查）：** **未列** GitHub、Hugging Face、Zenodo 或数据集下载链接；资源区仅有 **Paper**（链至 arXiv PDF）。

## 一句话摘要

TouchWorld 官方项目页：展示 **预测–反应式四层触觉层级策略**（SP / TWM / VLA / TRT）、**Wuji 灵巧手 + JQ-Industries 触觉手套** 硬件栈、**六任务真机 benchmark** 与人为扰动结果、TWM 预测可视化、子任务规划器消融，以及遥操数据采集 rollout 画廊。

## 公开信息要点（截至入库日）

- **核心叙事：** 触觉应同时承担 **未来接触子目标预测** 与 **高频在线残差纠偏**；单体 VLA 把慢推理与快反馈绑在同一环路会损害接触丰富长程任务。
- **架构侧栏（SP / TWM / VLA / TRT）：** 子任务分解 → 视触觉子目标预测 → 名义动作块 → 触觉残差精修，对应论文多时钟层级。
- **硬件：** 人形 + Wuji 灵巧手 + JQ-Industries 触觉手套；遥操 Meta Quest + Touch Plus + Wuji Glove。
- **实验：** 六任务（浇花、清桌面、杯插入、插头插入、擦锅、抽纸巾）干净 / 人为扰动双设置；Table 1 与论文一致（TouchWorld **65.0% / 53.7%** 平均成功率）。
- **TWM 评测：** Table 2 触觉子目标预测指标；交互式 prediction demo 区块。
- **Planner 评测：** Table 3 记忆增强 Qwen3-VL-4B 子任务规划优于 zero-shot 32B。
- **数据画廊：** 遥操采集视频（浇花、插插头、叠杯等），**非模型推理输出**。
- **局限：** 任务多样性仍有限；跨触觉传感器 / 本体需标定与小量适应数据。

## 为何值得保留

- **非 PDF 证据：** 轨迹样本浏览器、TWM 预测对比 demo、推理过程定性图比表格更直观呈现 **子任务–子目标–残差** 闭环。
- **开源状态锚点：** 页上 **无代码链接**，避免 wiki 误写「已开源」；后续 lint 可据此跟进。

## 关联资料

- 论文归档：[`sources/papers/touchworld_arxiv_2607_07287.md`](../papers/touchworld_arxiv_2607_07287.md)
- Wiki 实体：[`wiki/entities/paper-touchworld-tactile-foundation-dexterous-manipulation.md`](../../wiki/entities/paper-touchworld-tactile-foundation-dexterous-manipulation.md)
