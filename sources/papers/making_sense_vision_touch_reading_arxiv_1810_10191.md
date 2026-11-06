# Making Sense of Vision and Touch 论文通读（视触觉自监督表征）

> 来源归档（原文通读 · 支撑 `wiki/concepts/visuo-tactile-fusion.md`）

- **标题：** Making Sense of Vision and Touch: Self-Supervised Learning of Multimodal Representations for Contact-Rich Tasks
- **作者：** Michelle A. Lee, Yuke Zhu 等（Stanford；ICRA 2019）
- **类型：** paper / multimodal / contact-rich
- **原文：** <https://arxiv.org/abs/1810.10191>（通过 arXiv PDF v2 获取全文）
- **一句话说明：** 用自监督预测任务（动作条件光流、下一步是否接触、多模态时间对齐）把 RGB、腕部 F/T、本体感受压成 128 维表征，冻结后供 TRPO 学 peg-in-hole 策略。
- **核对范围：** 通读摘要、§I–§VIII 全文（相关工作、表征网络、策略与控制器、实验设置、仿真消融、真机实验、结论）；参考文献列表仅浏览。本文无附录。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **动机（§I）：** 视觉给语义与几何，触觉在遮挡下给接触状态，二者互补；多数学习型策略只围绕单一模态（多为视觉）闭环。
- **编码器（§IV-A）：** 128×128 RGB 经 6 层 CNN 得 128 维；最近 32 次六轴 F/T 读数（32×6）经 5 层因果卷积（stride 2）得 64 维；末端位置速度经 2 层 MLP 得 32 维；拼接后过 2 层 MLP 融合为 128 维表征。
- **自监督目标（§IV-B）：** (i) 给定下一动作预测光流（真值由本体与运动学自动生成，EPE 损失）；(ii) 预测下一控制周期是否接触（标签由 F/T 启发式生成）；(iii) 预测多模态输入是否时间对齐（随机错位采样，交叉熵）；三损失端到端求和。
- **策略与控制（§V）：** 冻结表征，仅训练 2 层 MLP 策略（约占总参数 3%），输出 3D 笔端位移 Δx，TRPO 优化；策略 20 Hz，轨迹插值后经阻抗 PD + 操作空间控制以 200 Hz 输出力矩，保持接触柔顺。
- **奖励（§VI）：** 分阶段奖励：reaching、alignment、insertion、completion。
- **数据（§VI）：** 100k 状态，随机策略 + 启发式策略采集，约 90–120 分钟；表征训练 20 epoch。
- **仿真消融（§VII-A，Fig. 4）：** 方形 peg，1.2k 回合 TRPO。Full 约 80% 完成插入，三个基线（无视觉 / 无触觉 / 两者皆无）完成率均低于 5%；No haptics 基线最强，能定位孔但很少完成插入——作者解释为视觉主导前期 reaching 与对位，触觉在接触前无信息，接触后对插入决定性。
- **真机（§VII-B）：** Kuka IIWA + OptoForce + Kinect v2；圆、三角、半圆 peg，每个 300 回合（约 5 小时）；每个策略评估 100 回合。迁移：用三角 peg 的表征与策略直接用于六边形、方形 peg 成功率超过 60%；重训策略（保留表征）再提升 19%（六边形）与 30%（方形）。遮挡相机与推扰动下可恢复。
- **局限与展望（§VIII）：** 仅 3 自由度笛卡尔位移，未覆盖 6-DoF；未纳入深度、声音等模态。

## 对 wiki 的映射

- [`wiki/concepts/visuo-tactile-fusion.md`](../../wiki/concepts/visuo-tactile-fusion.md)

## 参考来源

- 论文：<https://arxiv.org/abs/1810.10191>
