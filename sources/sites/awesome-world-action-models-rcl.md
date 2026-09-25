# Awesome World-Action Models（RCL Robotics · MBZUAI 综述站点）

- **类型**：项目静态站点 / 综述可视化导航
- **收录日期**：2026-09-16
- **站点**：<https://rcl-robotics.github.io/Awesome-World-Action-Models/>
- **同源仓库**：<https://github.com/rcl-robotics/Awesome-World-Action-Models>
- **综述标题**：*World-Action Models for Robot Learning and Control: A Survey*
- **arXiv**：[2609.16074](https://arxiv.org/abs/2609.16074)（2026-09-25 发布）
- **机构**：MBZUAI（通讯 Xingxing Zuo）等；联合 Caltech、Amazon FAR、UVA、Georgia Tech、NYU、UC Berkeley

## 一句话

把 **World-Action Models（WAM）** 综述的核心定义、**2×2 架构 taxonomy**（One Model / Dual-system × Joint / IDM）、训练数据金字塔、应用域与 **564 条** 可检索论文库，整理成章节化网页与交互筛选器。

## 为什么值得保留

- **边界定义**与 OpenMOSS 2605.12090 互补：强调 **control utility**（动作接地、时空一致、闭环改进、实时预算）四项准则，而非仅 Cascaded/Joint 二分。
- **架构轴更细**：独立区分 **架构**（One Model vs Dual-system）与 **预测–动作接口**（Joint prediction vs IDM），形成 Q1–Q4 四象限索引。
- **规模与证据**：**564 entries / 8 major categories**（截至 2026-09-13 目录更新），含 **Reading reports** 与 `papers.json` 机器可读书目；适合作 WAM 领域 **持续更新入口**。

## 开源边界（步骤 2.5）

| 已发布 | 不适用 |
|--------|--------|
| Markdown 策展清单、`papers.json`、静态站源码（MIT License） | 训练/推理代码、模型权重（清单性质） |

**结论：已开源（策展与站点模板）**；综述 PDF 见 **arXiv:2609.16074**。

## 站点摘录（2026-09-16 抓取要点）

来源：<https://rcl-robotics.github.io/Awesome-World-Action-Models/>

- **统一视图**：\((\widehat{\mathbf{O}}, \widehat{\mathbf{A}}) = f_{\mathrm{WAM}}(h_t, \ell)\) — 历史观测 + 语言指令 → 未来观测块与动作块。
- **VLA**：观测与语言 → 动作；缺显式物理演化。
- **World model**：\(p(o' \mid o, a)\)；预测未来，不单独构成可执行策略。
- **WAM**：在共享学习/推理过程中耦合世界预测与动作生成，使动作 **inform by predicted consequences**。
- **Joint prediction**：\(p_{\mathrm{joint}}(\mathbf{O}, \mathbf{A} \mid h_t, \ell)\)。
- **IDM**：\(p_{\mathrm{plan}}(\mathbf{O} \mid h_t, \ell) \cdot p_{\mathrm{IDM}}(\mathbf{A} \mid h_t, \mathbf{O})\)；部分系统在推理时不显式滚完整未来。
- **架构四象限**：(a) One Model + Joint；(b) One Model + IDM；(c) Dual-system + Joint；(d) Dual-system + IDM。
- **训练**：两阶段 — 互联网/自我中心视频预训练（时空与动作表征）→ 微调 / 数据增强 / RL 后训练；数据金字塔强调 action-free 视频 与 embodied 轨迹互补。
- **应用**：操纵、导航、自动驾驶；预测角色分 representation / look-ahead / synthetic trajectory 三类。
- **开放挑战**：动作对齐、世界–动作因子分解、空间/多视角一致、长程记忆、神经仿真闭环、高效推理。

### 八大类（Major categories）

| 类别 | 条目数（入库日） |
|------|------------------|
| Foundational work | 42 |
| VLA | 46 |
| WAMs | 296 |
| Datasets | 38 |
| Evaluation metrics | 15 |
| Benchmarks & simulators | 63 |
| Components of WAMs | 30 |
| Related resources | 31 |

## 对 wiki 的映射

- 实体页：[Awesome World-Action Models（RCL）](../../wiki/entities/awesome-world-action-models-rcl.md)
- 概念交叉：[World Action Models（WAM）](../../wiki/concepts/world-action-models.md) — 补 2×2 taxonomy 与 control utility 准则
- 对照策展：[Awesome-WAM（OpenMOSS）](../repos/awesome-wam-openmoss.md) · [Awesome World Models（sun254667）](../repos/awesome-world-models.md)
- 综述 source：[rcl_wam_robot_learning_survey.md](../papers/rcl_wam_robot_learning_survey.md)
