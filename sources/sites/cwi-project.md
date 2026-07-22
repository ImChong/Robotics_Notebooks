# CWI 项目页

> 来源归档（site / project page）

- **标题：** CWI: Composite Humanoid Whole-Body Imitation System for Loco-manipulation
- **类型：** project page
- **URL：** <https://cwi-ral.github.io/CWI-RAL-Webpage>
- **论文：** <https://arxiv.org/abs/2606.27676>
- **代码：** 未列出（2026-07-22 核查）
- **视频：** YouTube <https://youtu.be/BLckpzHSI0w>；Bilibili 链接见项目页
- **机构：** LimX Dynamics、HKU、SUSTech、HKUST、ZJU-UIUC
- **核查日期：** 2026-07-22
- **一句话说明：** CWI 项目页展示复合全身模仿系统：上身使用 AMASS 多样操作参考，下身使用 walk/squat AMP 先验，multi-critic 与 teacher-student 蒸馏后部署到 LimX Oli。

## 核心摘录（归纳，非全文）

- 项目页 Abstract 强调：CWI decouples MoCap data use for upper-body manipulation and lower-body locomotion。
- Framework 区强调：full AMASS upper-body references + compact expert walking/squatting clips + AMP。
- Method 区强调：locomotion rewards、upper-body tracking rewards、AMP style rewards 用 multi-critic PPO 优化。
- Deployment 区强调：student policy 只由 bimanual hand poses 加 velocity/height commands 驱动。
- Evaluation 区强调：31-DoF LimX Oli，IsaacLab 与真机；ablate multi-critic、distillation、AMP。
- 页面未列 GitHub、Code、Dataset 或下载入口。

## 对 wiki 的映射

- [CWI 实体页](../../wiki/entities/paper-cwi-composite-humanoid-whole-body-imitation.md)
- [Loco-Manip 接触分类 02：接触表示](../../wiki/overview/loco-manip-contact-category-02-contact-representation.md)

## 参考来源（原始）

- 项目页：<https://cwi-ral.github.io/CWI-RAL-Webpage>
- arXiv：<https://arxiv.org/abs/2606.27676>
