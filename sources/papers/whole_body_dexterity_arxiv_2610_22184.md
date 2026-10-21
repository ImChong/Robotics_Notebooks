# Whole-Body Dexterity 论文来源归档

> 来源归档（ingest）— 根据 arXiv 摘要、HTML 与作者项目页整理；非论文全文镜像。

## 元数据

- **题目：** Whole-Body Dexterity
- **作者：** Conor Mc Gartoll、An Dang、Pranay Thangeda、Arjun Gupta、Carolina Higuera、Nima Fazeli、Mustafa Mukadam、Manikantan Nambi
- **机构：** Amazon；加州大学圣地亚哥分校（UC San Diego）；密歇根大学（University of Michigan）
- **arXiv：** https://arxiv.org/abs/2610.22184
- **HTML：** https://arxiv.org/html/2610.22184v1
- **版本：** v1，2026-10-15，cs.RO / cs.AI / cs.LG / eess.SY
- **项目页：** https://www.wholebodydex.com/（见 [wholebodydex-project.md](../sites/wholebodydex-project.md)）
- **代码与数据：** 截至 2026-10-17，项目页未列公开仓库或数据集链接
- **沉淀到 wiki：** [paper-whole-body-dexterity.md](../../wiki/entities/paper-whole-body-dexterity.md)

## 问题与方法

论文关注人形机器人操作能力通常停留在末端执行器和简单抓取的问题。Whole-Body Dexterity（WBD）把手臂、躯干等身体表面纳入可切换的物体支撑接触，使机器人能把大件物体从一个身体接触区域重定位到另一个区域。

论文摘要给出的仿真到真机流程为：先生成可行的全身抓取数据集；再从数据集中采样起始与目标抓取，训练用于通用任意到任意重定位目标的强化学习专家；最后把专家蒸馏为单一策略。论文摘要称部署策略使用本体感觉与物体几何信息。项目页进一步称其为 proprioception-only 策略，读者应以论文正文的具体输入定义为准。

## 论文报告的真机结果

- **物体与接触：** Unitree G1 对篮球和纸箱进行躯干附近的重定位；展示把纸箱夹在单臂下、从货架搬运到桌面并放下的长时程任务。
- **控制方式：** 人类遥操作员提供粗粒度动作参考，低层控制器执行全身重定位。
- **总体完成率：** 蒸馏策略在项目页汇总的真实全身操作试验中完成 **45/50（90%）**；直接运动重定向为 **9/50（18%）**；通用全身控制器 SONIC 为 **2/50（4%）**。

## 对 wiki 的映射

- [Whole-Body Dexterity 实体页](../../wiki/entities/paper-whole-body-dexterity.md) — 归纳身体接触重定位流程、真机结果与复现边界。
- [Loco-Manipulation 任务页](../../wiki/tasks/loco-manipulation.md) — 作为全身接触丰富操作的代表实例。

## 参考资料

- 论文：<https://arxiv.org/abs/2610.22184>
- 作者项目页：<https://www.wholebodydex.com/>
