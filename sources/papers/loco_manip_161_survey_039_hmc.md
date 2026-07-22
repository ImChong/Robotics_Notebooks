# HMC

> 来源归档（ingest · 人形 Loco-Manip 161 篇长文 第 039/161）

- **标题：** HMC: Learning Heterogeneous Meta-Control for Contact-Rich Loco-Manipulation
- **类型：** paper
- **Loco-Manip 161 分类：** 02 上半身中心控制与移动操作接口
- **机构：** HMC: Learning Heterogeneous Meta-Control for Contact-Rich、UC San Diego
- **项目页：** https://loco-hmc.github.io
- **发表日期：** 2025年11月18日
- **入库日期：** 2026-06-26
- **一句话说明：** HMC 主要解决数据闭环：用遥操作/外骨骼数据、接触力/触觉信号采集人类操作和机器人状态，再通过ACT/行为克隆模仿学习、分层技能/专家策略转成可训练、可复用的关节位置/力矩命令、全身轨迹/动作序列、低层控制器目标。关键点是把任务拆成可路由的技能或专家策略，再用高层模块在执行中选择和组合。

## 核心摘录（策展，非全文）

- **在 161 篇地图中的位置：** 02 上半身中心控制与移动操作接口，编号 **039/161**。
- **算法实现总结（公众号）：** HMC 主要解决数据闭环：用遥操作/外骨骼数据、接触力/触觉信号采集人类操作和机器人状态，再通过ACT/行为克隆模仿学习、分层技能/专家策略转成可训练、可复用的关节位置/力矩命令、全身轨迹/动作序列、低层控制器目标。关键点是把任务拆成可路由的技能或专家策略，再用高层模块在执行中选择和组合。

## 对 wiki 的映射

- [paper-loco-manip-161-039-hmc](../../wiki/entities/paper-loco-manip-161-039-hmc.md)
- [loco-manip-161-category-02-upper-body-interface](../../wiki/overview/loco-manip-161-category-02-upper-body-interface.md)

## 参考来源（原始）

- 微信公众号编译：[wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md](../blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md)
- 原始抓取：[wechat_humanoid_loco_manip_161_2026-06-26.md](../raw/wechat_humanoid_loco_manip_161_2026-06-26.md)

## 项目页与开源状态核查（2026-07-22）

- **论文/项目：** <https://loco-hmc.github.io>。
- **代码：** 未确认官方训练/部署仓库；可见 GitHub 账户仅项目页仓库。
- **关键结论：** HMC-Controller 在 torque space 混合 position、impedance、hybrid force-position；HMC-Policy 用 MoE routing 从大量位置示范和少量力感知示范学习，真机 challenging tasks 相对基线提升超过 50%。
- **wiki 深化：** [paper-loco-manip-161-039-hmc](../../wiki/entities/paper-loco-manip-161-039-hmc.md)。
