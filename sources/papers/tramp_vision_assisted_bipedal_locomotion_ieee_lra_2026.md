# Vision-Assisted Bipedal Locomotion on Challenging Terrains via Terrain-Related Adversarial Motion Priors（IEEE RA-L 2026）

> 来源归档（ingest · ResearchGate 用户提供入口）

- **标题：** Vision-Assisted Bipedal Locomotion on Challenging Terrains via Terrain-Related Adversarial Motion Priors
- **简称（本库）：** **TRAMP**（Terrain-Related Adversarial Motion Priors）
- **类型：** paper / humanoid / perceptive-locomotion / depth / amp / moe / single-stage-rl
- **DOI：** [10.1109/LRA.2026.3707326](https://doi.org/10.1109/LRA.2026.3707326)
- **IEEE Xplore：** <https://ieeexplore.ieee.org/document/11578326/>
- **ResearchGate（ingest 入口）：** <https://www.researchgate.net/publication/408100590_Vision-Assisted_Bipedal_Locomotion_on_Challenging_Terrains_via_Terrain-Related_Adversarial_Motion_Priors>
- **作者：** Yunpeng Liang、Kaiqi Yang、Zhenyu Fang、Yanzheng Zhao、Weixin Yan（上海交通大学机械工程学院）
- **期刊：** IEEE Robotics and Automation Letters，Vol. 11, No. 8, pp. 9622–9629，2026-08（online 2026-06-25）
- **资助：** Mscape Technology Co. Ltd（Crossref 元数据）
- **入库日期：** 2026-08-20
- **一句话说明：** 单阶段视觉辅助人形 RL：机载本体 + 低成本深度；层次特征提取器压缩动力学与地形上下文，经 **MoE actor** 做地形感知行为调制；用平地与楼梯示范构造 **地形相关对抗运动先验**，在坡/楼梯/高台/宽沟与户外杂乱场景实现鲁棒节能行走。

## 开源状态（步骤 2.5）

- **核查日：** 2026-08-20。检索 ResearchGate 条目、IEEE Xplore、Semantic Scholar、作者 ORCID 与公开 GitHub；**未发现独立项目页或官方代码仓**。
- **ResearchGate：** 用户提供全文学下载入口；Cloud Agent 环境访问该站触发反爬限制，**未能在此环境拉取 PDF**，摘要与元数据以 DOI / Semantic Scholar 为准。
- **已发布：** IEEE RA-L 正式论文（闭源 OA）；ResearchGate 作者自托管全文（需人工下载）。
- **未发布：** 训练/推理代码、权重、数据集；论文摘要与元数据未列 GitHub / Hugging Face 等链接。
- **结论：** **确认未开源**（截至入库日）。若后续作者发布项目页，应补 `sources/sites/` 与 `sources/repos/` 并更新实体页「源码运行时序图」。

## 摘录 1：问题与单阶段框架（摘要）

复杂地形人形行走需要可靠感知与地形自适应步态调节。近年感知 locomotion 虽进展显著，但不少方法依赖**显式地形表征**、**几何相关辅助监督**或**多阶段训练管线**，系统复杂度与训练成本偏高。

本文提出**轻量单阶段 RL** 框架：仅使用机载**本体感知**与**低成本深度**实现视觉辅助人形行走。核心组件：

1. **层次特征提取器（hierarchical feature extractor）** — 学习机器人动力学与地形上下文的紧凑潜表示；
2. **MoE actor** — 将上述表示用于地形感知的行为调制；
3. **地形相关对抗运动先验（terrain-related adversarial motion prior）** — 由**平地**与**楼梯**行走示范构造，在统一策略内鼓励地形相容的运动模式。

**对 wiki 的映射：** 升格 [`wiki/entities/paper-tramp-vision-assisted-bipedal-locomotion.md`](../../wiki/entities/paper-tramp-vision-assisted-bipedal-locomotion.md)；对照 [MoRE](../../wiki/entities/paper-amp-survey-08-more.md)（两阶段 + 多判别器 AMP）、[T-GMP](../../wiki/entities/paper-motion-cerebellum-t-gmp.md)（CVAE 生成式地形先验）、[CReF](../../wiki/entities/paper-cref.md)（单阶段深度交叉注意）、[DPL](../../wiki/entities/paper-notebook-dpl-depth-only-perceptive-humanoid-locomotion-vi.md)（深度→高程重建）。

## 摘录 2：实验场景与部署结论（摘要）

仿真与**物理人形真机**实验表明：所学策略在**坡道、楼梯、高台、宽沟**及**户外杂乱场景**中实现鲁棒且**节能**的行走，并保持稳定**足–地形**接触。

摘要未给出定量成功率、能耗数值或具体机器人型号；完整 benchmark、消融与网络超参需阅读 IEEE RA-L 全文或 ResearchGate 下载稿。

**对 wiki 的映射：** [楼梯/障碍感知 locomotion](../../wiki/tasks/stair-obstacle-perceptive-locomotion.md)、[Humanoid Locomotion](../../wiki/tasks/humanoid-locomotion.md)、[AMP 奖励](../../wiki/methods/amp-reward.md)。

## 摘录 3：文献脉络（Crossref references 节选）

论文引用簇包含：足式/人形感知行走（Extreme Parkour、Hiking in the Wild 等）、**AMP**（Escontrela et al., IROS 2022）、**MoRE**（Wang et al., 2025）、地形条件 AMP 四足工作、PRIOR 等。定位是**视觉 + 单阶段 + 地形相关 AMP + MoE** 的 RA-L 工程向组合，与多阶段深度 teacher–student 或显式高程图管线形成对照。

**对 wiki 的映射：** [地形适应](../../wiki/concepts/terrain-adaptation.md)、[Privileged Training](../../wiki/concepts/privileged-training.md)（若正文采用非对称 critic，待 PDF 核实）。

## 对 wiki 的映射（汇总）

- 新建实体页：[`wiki/entities/paper-tramp-vision-assisted-bipedal-locomotion.md`](../../wiki/entities/paper-tramp-vision-assisted-bipedal-locomotion.md)
- 交叉更新：[`wiki/tasks/stair-obstacle-perceptive-locomotion.md`](../../wiki/tasks/stair-obstacle-perceptive-locomotion.md)、[`wiki/tasks/humanoid-locomotion.md`](../../wiki/tasks/humanoid-locomotion.md)、[`wiki/methods/amp-reward.md`](../../wiki/methods/amp-reward.md)

## 参考来源（原始）

- DOI：<https://doi.org/10.1109/LRA.2026.3707326>
- ResearchGate：<https://www.researchgate.net/publication/408100590_Vision-Assisted_Bipedal_Locomotion_on_Challenging_Terrains_via_Terrain-Related_Adversarial_Motion_Priors>
- Semantic Scholar 摘要 API（2026-08-20 检索）
