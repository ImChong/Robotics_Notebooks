# Workhorse 项目页来源归档

- **项目页：** https://hybridrobotics.github.io/workhorse/
- **论文：** https://arxiv.org/abs/2610.09117
- **HTML v1：** https://arxiv.org/html/2610.09117v1
- **作者：** Songbo Hu*, Qiayuan Liao*, Yufeng Chi, Kevin Zakka, Yakun Sophia Shao, Pieter Abbeel, Koushil Sreenath
- **机构：** University of California, Berkeley
- **入库日期：** 2026-10-07；依据新论文更新：2026-10-09

## 项目简介

Workhorse 从机器人免示范学习 whole-body humanoid loco-manipulation。采集使用胸部、双手、双脚五个可穿戴 tracker 与胸前相机；视觉 planner 和 RL whole-body tracker 分别在相同的人类五链路示范上训练，不做机器人关节轨迹 retargeting。官网与 arXiv v1 共同描述 G1 箱子分拣、接抛掷箱子与行李箱交互等任务。

## 可核验论文结果

- G1 真机：手/脚配合分拣箱子、接住抛掷箱子、推动 14 kg 行李箱后翻倒并爬上 0.39 m 行李箱取物。
- 仿真复刻演示室：box sorting 77%；受到 40 N·s push 时 64%。
- H2 在同一示范重新训练、无 push 仿真条件下 box sorting 83%。
- 训练中 planner / tracker 使用互相条件化的数据增强，模拟两端之间的部署误差。

## 开放状态与证据边界

官网当前仍保留“arXiv link follows shortly”的旧提示，但 arXiv:2610.09117 v1 已发布，本归档以论文为方法与定量结果依据。官网明确说明 code 尚未 release，且本次未找到可核验的官方 GitHub 代码库。论文结果不应误写成开放代码已可复现。

## 对 wiki 的映射

- 项目实体：[Workhorse](../../wiki/entities/workhorse-humanoid-loco-manipulation.md)
- 任务入口：[Loco-Manipulation](../../wiki/tasks/loco-manipulation.md)
- 论文归档：[Workhorse arXiv 论文](../papers/workhorse_arxiv_2610_09117.md)
