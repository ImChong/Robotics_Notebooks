# ingrid789.github.io/SkillMimic（SkillMimic 项目页）

- **标题：** SkillMimic: Learning Basketball Interaction Skills from Demonstrations
- **类型：** site / project-page
- **URL：** <https://ingrid789.github.io/SkillMimic/>
- **配套论文：** [SkillMimic（arXiv:2408.15270）](https://arxiv.org/abs/2408.15270) — 归档见 [`sources/papers/skillmimic_arxiv_2408_15270.md`](../papers/skillmimic_arxiv_2408_15270.md)
- **代码：** <https://github.com/wyhuai/SkillMimic> — 归档见 [`sources/repos/skillmimic.md`](../repos/skillmimic.md)
- **入库日期：** 2026-07-28

## 一句话摘要

CVPR 2025 Highlight 官方项目页：展示物理仿真人形从人–球演示学习 **多样篮球交互技能**，以及用 **高层控制器（HLC）** 组合技能完成连续得分等长程任务。

## 公开信息要点（截至入库日）

- **机构：** HKUST、Unitree Robotics、Peking University、Tsinghua University、IDEA、Tencent、CMU（通讯作者含 Ailing Zeng、Jian Zhang 等）。
- **页首卖点：** 统一配置学运球/上篮/投篮等；技能可复用与组合；连续得分（运球→上篮→篮板→重复）。
- **系统三部分：** (a) HOI 数据采集；(b) 统一 HOI 模仿奖励训 skill policy；(c) HLC 用极简任务奖励复用技能。
- **Learned Skills：** Jump Shot、Turnaround Layup、Layup、Rebound、多向 Dribble、Catch、Pass、Pickup 等。
- **Skill Switching：** Layup↔Rebound、Pickup→Dribble→Layup→Catch 等（含参考中未出现的切换）。
- **HLC 任务：** Heading、Circling、Scoring、Throwing；另有 Get Up / Pick Up 鲁棒性演示。
- **3D Visualization：** 下拉选择 HLC/LLC 运动并拖动视角。
- **资源链接：** 页内指向 arXiv / 视频；**代码入口为官方 GitHub `wyhuai/SkillMimic`**（Apache-2.0，训练评测已发布）。

## 为何值得保留

- **非 PDF 证据：** 技能切换与 HLC 长程任务比表格更直观。
- **开源三角互证：** 项目页任务名与仓库 `HRLCircling` / `HRLHeadingEasy` / `HRLScoringLayup` / `HRLThrowing` 一致。
- **与 Paper Notebooks 分类 13** 的篮球物理动画线（Learning to Ball 等）对照入口。

## 关联资料

- 论文归档：[`sources/papers/skillmimic_arxiv_2408_15270.md`](../papers/skillmimic_arxiv_2408_15270.md)
- 代码仓库：[`sources/repos/skillmimic.md`](../repos/skillmimic.md)
- Paper Notebooks 锚点：[`sources/papers/humanoid_pnb_skillmimic-learning-basketball-interaction-skill.md`](../papers/humanoid_pnb_skillmimic-learning-basketball-interaction-skill.md)
