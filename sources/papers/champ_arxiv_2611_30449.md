# CHAMP：以全身运动先验协调人形移动操作 / CHAMP: Coordinated Humanoid Loco-Manipulation with Whole-Body Motion Priors

- **类型：** 论文预印本
- **作者：** Lijun Han、Muyuan Ma、Zichen Wang、Siyuan Liu、Yuxi Liu、Yutong Wang、Bohao Zhang、Heng Zhou、Wei Sui
- **机构：** 地瓜机器人（D-Robotics）、中国科学院自动化研究所（CASIA）
- **日期：** 2026-11-23（arXiv v1 提交）
- **原文：** [arXiv](https://arxiv.org/abs/2611.30449) · [HTML](https://arxiv.org/html/2611.30449v1) · [PDF](https://arxiv.org/pdf/2611.30449v1)
- **项目页：** [CHAMP](https://han-li-jun.github.io/CHAMP/index.html) · [项目页归档](../sites/champ-2611-30449.md)
- **核对范围：** arXiv v1 全文正文 §1–5，方法图与表 1–3，附录 A–D 的坐标表示、训练与数据筛选、评测和路由统计。
- **通读：** 已完成

## 论文摘录（意译）

1. §3.2：用约 2,298 小时异构人类与机器人记录做 VLM 未来运动预测，不必为缺失的身体或手部标注伪造目标。
   - **对 wiki 的映射：** [CHAMP：方法与移动操作](../../wiki/entities/paper-champ-2611-30449.md)。
2. §3.3 与附录 A/B：SONIC 闭环回放筛选得到 40,016 段后训练轨迹；CAE 从冻结 VLM 多层读出并用条件流匹配联合生成身体与手部动作。
   - **对 wiki 的映射：** [CHAMP：回放接地与路由](../../wiki/entities/paper-champ-2611-30449.md#方法)。
3. §4：七个仿真任务 178/210、七个真机任务 64/70；完整模型在 SIMPLE 消融中优于四种固定层变体。
   - **对 wiki 的映射：** [CHAMP：评测](../../wiki/entities/paper-champ-2611-30449.md#评测)。这些数字是作者报告，不是独立复现。
