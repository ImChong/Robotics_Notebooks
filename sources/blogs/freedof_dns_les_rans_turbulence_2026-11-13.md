# 从 DNS 到 RANS：湍流求解的不同方法，其实在做同一道选择题

> 来源归档（公众号长文 / 用户提供的 94 页 PDF；保留出处并提炼知识，不转载全文）

- **标题：** 从 DNS 到 RANS：湍流求解的不同方法，其实在做同一道选择题
- **作者：** 自由度 FreeDof（PDF 页眉标注 freedof / 自由度）
- **发布日期：** 2026-11-13（PDF 首屏标注）
- **原文：** <https://mp.weixin.qq.com/s/bB03r_WZvOjFNpkhWeAQVQ>
- **资料类型：** blog / CFD 教程 / 湍流方法对比
- **取得方式：** 用户上传的 94 页 PDF，打印于 2026-11-14；PDF 原文件不提交仓库，避免未经许可整篇再分发
- **核对范围：** 通读 PDF 94 页（正文四部分与末尾 26 条参考资料）：① NS 方程、Kolmogorov 尺度与 DNS 代价；② Reynolds 平均、应力封闭、混合长度、k–ε/k–ω/SST、SA、RSM、URANS；③ 空间过滤、SGS、Smagorinsky/WALE/动态模型/SGS-k/ILES、WRLES/WMLES、DES/DDES/IDDES；④ 圆柱绕流计算域、初边值、近壁网格、压力/黏性力积分及验证。公式按原文所注明的不可压缩、常密度、常运动黏度、无体力假设理解；原文并未提供可运行代码或新的实验数据。
- **通读：** 已完成

## 一句话信息

**DNS、RANS/URANS、LES 与 DES 家族是对“哪些尺度需要解析、哪些尺度交由封闭模型”的不同取舍，而非统一精度旋钮。** 选择应从目标统计量、瞬态载荷或空间湍流结构出发，而不是从求解器里的模型名字出发。

## 原文四部分的知识主线

1. **瞬时 NS → DNS 成本。** DNS 直接解析含能范围至耗散范围；Kolmogorov 尺度及网格/时间采样决定成本，原文用 0.1 m 域与 0.1 mm 耗散尺度给出约 10⁹ 个均匀网格单元的量级例子（并非任意工况的通用成本）。
2. **Reynolds 平均 → 封闭缺口。** 非线性对流项平均后，速度脉动二阶相关构成 Reynolds 应力。Boussinesq 涡黏度关系将偏应力与平均应变率联系；经典混合长度、SA、k–ε、k–ω、SST 计算涡黏度，而 RSM 直接输运应力分量。
3. **URANS 的时间含义。** URANS 保留**所选平均场**及其湍流模型变量的物理时间项；能描述特定相干非定常响应，但仅减小时间步不能转为 LES，且不可由 k、ε 等统计场重建完整宽频载荷。
4. **LES 过滤 → SGS 应力。** 空间过滤后出现“先乘后滤”和“先滤后乘”的差；SGS 可用 Smagorinsky、WALE、动态 Smagorinsky、SGS-k 或经过验证的 ILES 数值耗散封闭。WRLES 与 WMLES 的近壁成本、边界处理不同。
5. **混合 DES。** DES 用网格关联长度与 RANS 尺度选择改变模型作用；DDES 的屏蔽旨在避免附着边界层过早降低模型应力；IDDES 包含壁面建模 LES 的分支。模型应力下降却无已解析脉动补足，会有应力缺口。
6. **圆柱绕流的落地。** 目标量为平均阻力、周期升力及三维尾流；对照初边值、近壁处理与 y+、空间/时间收敛、采样、谱及实验工况，区分 verification 与 validation。

## 原文中的代表性选择

| 需要回答的问题 | 原文建议的候选路线（需结合 Re、网格、壁面与验证） |
|---|---|
| 圆柱平均阻力与平均尾流 | 稳态 RANS，必要时非定常计算后统计 |
| 启动瞬态与平均场随时间变化 | URANS |
| 主导脱涡频率与相干载荷 | URANS 可作为起点；也可考虑 LES/DES |
| 三维尾流与宽频压力谱 | LES/DES 并与数据对照 |
| 全尺度耗散细节 / 模型基准 | 在合适工况下 DNS 或实验 |

## 来源限制与核验边界

- 这是**方法讲解与综述性教程**，并非提出一个新 CFD 模型的研究论文，也没有独立的数值 benchmark 数据表。
- PDF 中若干公式以图片/排版呈现；知识页侧重可核对的方法含义和工程决策，不移植未经单独校验的完整模型系数实现。
- PDF 第 92–94 页列出 26 项原始参考文献，其中包括 NASA Turbulence Modeling Resource 的 SA、SST、Wilcox 模型规范，OpenFOAM Foundation 的 LRR、kEqn 与力积分实现，以及 Nicoud–Ducros WALE 和 Germano–Lilly 动态封闭等论文；这些是**原文引用**，本次未逐篇独立复核。

## 对 wiki 的映射

- [DNS / LES / RANS 湍流选型对比](../../wiki/comparisons/dns-les-rans-turbulence-resolution-selection.md)
- [CAE / CFD 代理工具与技能项目总览](../../wiki/overview/cae-cfd-agent-skills-landscape.md)（工程求解器与自动化的衔接）
- [HydroGym：流体动力学强化学习平台](../../wiki/entities/paper-hydrogym.md)（CFD 与流控学习的应用例）
