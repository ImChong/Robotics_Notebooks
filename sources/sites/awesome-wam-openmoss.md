# Awesome-WAM（OpenMOSS GitHub Pages）

- **类型**：项目静态站点 / 综述可视化导航
- **收录日期**：2026-05-14
- **站点**：<https://openmoss.github.io/Awesome-WAM>
- **同源仓库**：<https://github.com/OpenMOSS/Awesome-WAM>
- **综述论文**：<https://arxiv.org/abs/2605.12090>

## 一句话

把 **World Action Models（WAM）** 综述的核心定义（VLA vs WM vs WAM）、**Cascaded / Joint** 架构叙事、数据生态、评测轴与开放挑战，整理成 **可扫读的章节化网页**，并挂载可搜索论文库（与仓库同步）。

## 为什么值得保留

- **边界定义**写得很直白：WAM 要求 **前向预测未来状态** 与 **动作生成** 在策略内耦合，目标为 \(p(o', a \mid o, l)\)。
- **架构**与 **评测**两块的条目名（基准族、指标族）适合作为 wiki 页写「延伸阅读」时的 **外部索引**，避免在 wiki 内复制长列表。

## 站点摘录（2026-05-14 抓取要点）

来源：<https://openmoss.github.io/Awesome-WAM>

- **VLA**：\(p(a \mid o, l)\)，语义强、但缺显式物理演化。
- **World model**：\(p(o' \mid o, a)\)，预测未来观测，不单独构成可执行策略。
- **WAM**：\(p(o', a \mid o, l)\)，未来合成与可执行控制在 **同一 embodied policy 框架** 内联合学习。
- **Cascaded WAM**：先 `future plan` 再 `action`；显式/隐式规划载体二分。
- **Joint WAM**：`future + action` 共享模型；自回归、扩散/流匹配等耦合方式并列。
- **数据**：机器人遥操作对齐、便携人类示教（UMI 类）、仿真特权、人/自我中心视频——强调 **混合设计**。
- **评测**：世界建模（视觉保真 / 物理常识 / 动作可推断性）与策略能力（通用操作、双臂人形、移动操作、软体接触、真机套件）双轴。
- **开放挑战**：耦合对比、多模态物理状态、数据混合、长程规划、推理效率、联合安全评测等。

## 对 wiki 的映射

- 主沉淀：[World Action Models（WAM）](../../wiki/concepts/world-action-models.md)
- 原始论文档：[world_action_models_survey_2605.md](../papers/world_action_models_survey_2605.md)
- 列表维护入口：[awesome-wam-openmoss.md](../repos/awesome-wam-openmoss.md)
