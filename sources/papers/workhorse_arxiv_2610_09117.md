# Workhorse 论文来源归档

- **标题：** *Workhorse: Learning Robust Whole-Body Humanoid Loco-Manipulation from Human Data*
- **arXiv：** https://arxiv.org/abs/2610.09117
- **HTML v1：** https://arxiv.org/html/2610.09117v1
- **项目页：** https://hybridrobotics.github.io/workhorse/
- **作者：** Songbo Hu, Qiayuan Liao, Yufeng Chi, Kevin Zakka, Yakun Sophia Shao, Pieter Abbeel, Koushil Sreenath
- **机构：** University of California, Berkeley
- **提交日期：** 2026-10-06
- **代码：** 项目页声明 code 尚未发布；论文/项目页均未提供可核验的官方 GitHub 仓库。

## 方法摘录

- 人类演示由 5 个可穿戴 tracker（胸部、双手、双脚）和胸前相机记录，约 60 Hz；同一批记录的五链路人类姿态同时训练 planner 与 tracker，无需机器人遥操作，也无需将动作 retarget 到机器人关节轨迹。
- 输入人体目标表示为躯干、左右腕、左右脚五个 link 的位姿。
- 视觉 planner 接收机器人 egocentric 图像与一段五链路历史，使用 flow matching 预测未来约 1.16 s 的目标 chunk；论文描述以 5 Hz replanning。
- RL whole-body tracker 跟踪近期的五链路 chunk 并输出机器人关节目标；其更新窗口约 0.2 s。
- 两者交替数据增强：tracker 增加目标跟踪漂移，模拟部署时会遇到的误差；planner 用人体分割 / inpainting / 机器人渲染及历史漂移增广，缩小视觉计划与 tracker 能力之间的分布差异。
- 部署端按图像与目标时间戳处理异步 chunk，避免控制滞后或重复执行过期动作。

## 论文报告结果

- Unitree G1 真机展示箱子分拣（手与脚）、接住抛掷箱子，以及推动 14 kg 行李箱、翻倒并攀上 0.39 m 行李箱等任务。
- 模拟演示室中箱子分拣成功率 77%；施加 40 N·s 推扰动时为 64%。
- 同一示范重新训练后，第二种人形机器人 H2 在无推扰动的模拟分拣成功率为 83%。
- 结果分别对应真实机器人演示和论文模拟协议，不能混作同一测试分布上的统计结果。

## 项目页与代码状态

官网给出方法、数据采集、评测和部署细节。官网页面当前仍显示 arXiv 链接后续补充的旧提示；现以已发布 arXiv v1 为论文依据。官网说明代码尚未发布；未找到可核验的官方代码仓库。
