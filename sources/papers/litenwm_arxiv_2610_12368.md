# LiteNWM：高效潜空间导航世界模型（arXiv:2610.12368）

> 来源归档（ingest · arXiv v1）

- **标题：** LiteNWM: Efficient Latent World Models for Onboard Visual Navigation in the Wild
- **类型：** paper / visual navigation / latent world model / trajectory selection
- **版本：** arXiv:2610.12368v1，2026-10-08
- **作者：** Linkai Liu、Yuntian Zhang、Zhenshan Bing、Chen Chen、Lingjuan Lyu、Shangguang Wang、Mengwei Xu、Dongqi Cai
- **机构：** 南京大学；Imperial College London；Sony AI；北京邮电大学
- **论文：** [arXiv 摘要](https://arxiv.org/abs/2610.12368) · [HTML 正文](https://arxiv.org/html/2610.12368v1) · [PDF](https://arxiv.org/pdf/2610.12368)
- **项目页 / 官方代码：** 截至 2026-10-14，arXiv 论文未提供独立项目页或官方代码仓库链接；不据此推断代码已经公开。
- **平台：** 离线评测使用 RECON、SCAND、SACSoN；真机使用 Unitree Go2 EDU（Jetson Orin）。
- **入库日期：** 2026-10-14
- **一句话说明：** 在冻结的视觉导航策略提出的候选轨迹上共享视觉编码、联合预测 H1–H4 的动作条件潜特征，再由相对评分器选择轨迹，避免为每个候选生成 RGB 未来图像。

## 核心摘录（面向 Wiki 编译）

### 1. 把高成本的 RGB rollout 换成共享潜空间评估

论文定位的是导航流水线中的候选轨迹评估瓶颈，而不是替换轨迹生成策略。默认设置用冻结 NoMaD 产生 16 条候选轨迹，并保留一个 incumbent；历史观测与目标图像由冻结 DINOv3 编码一次后供所有候选共享。动作条件世界模型在潜空间联合预测四个短期时域的未来特征，不做 RGB 解码，也不把前一时域预测自回归反馈到下一时域。

**对 Wiki 的映射：** [LiteNWM 实体页](../../wiki/entities/paper-litenwm-efficient-latent-world-model-navigation.md)「核心方法」；[生成式世界模型](../../wiki/methods/generative-world-models.md)中导航候选的潜空间未来评分。

### 2. 相对 incumbent 的评分与安全回退

评分器综合预测未来特征、当前状态、视觉目标、动作和候选集合信息，估计候选相对 incumbent 的 ATE、平移 RPE、FDE 与综合目标变化。论文用三成员预测器 / 评分器集成、不确定性校准和 harm / dual-improvement 概率构成接纳门；候选不满足收益、物理或候选支持条件时继续保留 incumbent。选出的短段执行后，机器人更新视觉历史并重新规划。

**对 Wiki 的映射：** [LiteNWM 实体页](../../wiki/entities/paper-litenwm-efficient-latent-world-model-navigation.md)「候选筛选与闭环执行」；[模型预测控制](../../wiki/methods/model-predictive-control.md)中的滚动重规划思路。

### 3. 效率与真机结果的读法

在 RTX 5090 上，完整端到端流程相较 NoMaD+NWM-XL 快 **128.00 倍**；RECON、SCAND、SACSoN 三域宏平均轨迹 ATE 降低 **17.56%**。冻结评估器经确定性动作适配器迁移到 MBRA，无需按 proposer 重新训练，宏平均 ATE 降低 **16.2%**。Go2 EDU 的三个未见场景共 30 次试验中，NoMaD 成功率为 **43.3%**，LiteNWM 为 **83.3%**；这是有限场景和试次下的真机证据，不应与离线 SR@1 混为一谈。

**对 Wiki 的映射：** [LiteNWM 实体页](../../wiki/entities/paper-litenwm-efficient-latent-world-model-navigation.md)「实验与评测」和「结论」。

## 归档说明

- 本归档以 arXiv v1 正文为一手来源；机构、作者和发表日期依论文首页核对。
- 截至 2026-10-14，正文未指向独立项目页或官方可运行代码仓库；因此不创建 `sources/sites/` 或 `sources/repos/` 条目，也不登记为已开源。
- 论文摘要与正文将 RTX 5090 的端到端加速写为 128.00×；个别结论段落出现不同速度比数字，归档以摘要与正文 IV-E 的明确主比较为准。
