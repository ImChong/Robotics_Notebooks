# FLORES 仓库

> 来源归档（repo）

- **标题：** FLORES
- **类型：** repo
- **链接：** <https://github.com/ZhichengSong6/FLORES>
- **入库日期：** 2026-11-01
- **线索来源：** [wechat_guyueju_quadruped_opensource_deep_dive_2026-11-01.md](../blogs/wechat_guyueju_quadruped_opensource_deep_dive_2026-11-01.md)（古月居转载，原作者敢敢AUTOHUB）
- **一句话说明：** FLORES 对应 RA-L 2026 论文「A Reconfigured Wheel-Legged Robot for Enhanced Steering and Adaptability」：通过关节 DoF 与可重构腿设计增强转向与适应性。
- **最近提交：** 2026-05-22
- **核查：** 仓库可达（`git ls-remote`，浅克隆读 README，不运行代码）
- **文章星标（文章表述）：** 25
- **沉淀到 wiki：** [`wiki/entities/flores-wheel-legged-robot.md`](../../wiki/entities/flores-wheel-legged-robot.md)

## README 摘录（入库日人工阅读）

README 标注发表于 IEEE RA-L 2026（请引用同行评审版本），并给出机械设计与训练流程图：HIM 策略先在 Isaac Gym 带奖励塑形训练，再迁移到 MuJoCo 做 sim-to-sim，第二阶段为实机部署。
