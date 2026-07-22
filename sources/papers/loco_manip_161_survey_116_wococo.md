# WoCoCo

> 来源归档（ingest · 人形 Loco-Manip 161 篇长文 第 116/161）

- **标题：** WoCoCo: Learning Whole-Body Humanoid Control with Sequential Contacts
- **类型：** paper
- **Loco-Manip 161 分类：** 05 动捕、人类视频与交互动作规划
- **机构：** ETH Zurich、⋆Work was done at Carnegie Mellon University、Carnegie Mellon University
- **项目页：** https://lecar-lab.github.io/wococo/
- **发表日期：** 2024年11月7日
- **入库日期：** 2026-06-26
- **一句话说明：** WoCoCo 先从本体状态与关节序列、人类视频/动捕轨迹、仿真交互数据恢复场景、目标或运动表征，再用PPO/RL 策略训练、AMP/运动先验、扩散策略/流匹配生成全身轨迹/动作序列、末端执行器/腕手目标。关键点是把动作生成看成条件生成问题，用扩散或流匹配在多模态动作分布里采样可执行轨迹。

## 核心摘录（策展，非全文）

- **在 161 篇地图中的位置：** 05 动捕、人类视频与交互动作规划，编号 **116/161**。
- **算法实现总结（公众号）：** WoCoCo 先从本体状态与关节序列、人类视频/动捕轨迹、仿真交互数据恢复场景、目标或运动表征，再用PPO/RL 策略训练、AMP/运动先验、扩散策略/流匹配生成全身轨迹/动作序列、末端执行器/腕手目标。关键点是把动作生成看成条件生成问题，用扩散或流匹配在多模态动作分布里采样可执行轨迹。

## 对 wiki 的映射

- [paper-loco-manip-161-116-wococo](../../wiki/entities/paper-loco-manip-161-116-wococo.md)
- [loco-manip-161-category-05-mocap-human-video](../../wiki/overview/loco-manip-161-category-05-mocap-human-video.md)

## 参考来源（原始）

- 微信公众号编译：[wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md](../blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md)
- 原始抓取：[wechat_humanoid_loco_manip_161_2026-06-26.md](../raw/wechat_humanoid_loco_manip_161_2026-06-26.md)

## 项目页与开源状态核查（2026-07-22）

- **论文/项目：** arXiv:2406.06005 / CoRL 2024 Oral；项目页 <https://lecar-lab.github.io/wococo/>。
- **代码：** <https://github.com/LeCAR-Lab/wococo>；CC BY-NC 4.0 + inherited licenses；含 `legged_gym`、`rsl_rl`、`scripts/train.py`、`scripts/play.py`。
- **开放边界：** README 提供 clap-and-dance 示例，明确鼓励用户按具体任务工程化 reward/MDP 与 sim-to-real。
- **wiki 深化：** [paper-loco-manip-161-116-wococo](../../wiki/entities/paper-loco-manip-161-116-wococo.md) 已补源码运行时序图。
