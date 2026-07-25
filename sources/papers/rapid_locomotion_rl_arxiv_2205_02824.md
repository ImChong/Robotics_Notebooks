# Rapid Locomotion via Reinforcement Learning（Agile Locomotion via Model-free Learning）

> 来源归档

- **标题：** Rapid Locomotion via Reinforcement Learning
- **博文别名：** Agile Locomotion via Model-free Learning（Robot Daycare 清单用名）
- **类型：** paper
- **作者：** Gabriel B. Margolis, Ge Yang, Kartik Paigwar, Tao Chen, Pulkit Agrawal
- **机构：** MIT CSAIL（Improbable AI）
- **链接：** https://arxiv.org/abs/2205.02824
- **PDF：** https://arxiv.org/pdf/2205.02824
- **项目页：** https://agility.csail.mit.edu/ ；https://sites.google.com/view/model-free-speed/
- **代码：** https://github.com/Improbable-AI/rapid-locomotion-rl
- **入库日期：** 2026-07-25
- **一句话说明：** 端到端 RL 控制器使 Mini Cheetah 野外高速奔跑/转向，持续速度至 **3.9 m/s**；关键为速度命令自适应课程 + 在线系统辨识式 Sim2Real。
- **开源状态：** **已开源**（Improbable-AI/rapid-locomotion-rl）
- **沉淀到 wiki：** [paper-rapid-locomotion-rl](../../wiki/entities/paper-rapid-locomotion-rl.md)

---

## 核心贡献（摘录）

1. Mini Cheetah 记录级敏捷：3.9 m/s，草地/冰/砾石等。
2. adaptive curriculum on velocity commands。
3. online system identification for sim-to-real（承接先验工作）。

## 对 wiki 的映射

- [paper-rapid-locomotion-rl](../../wiki/entities/paper-rapid-locomotion-rl.md)
- [sim2real](../../wiki/concepts/sim2real.md)
- [curriculum-learning](../../wiki/concepts/curriculum-learning.md)
- [mit-mini-cheetah](../../wiki/entities/mit-mini-cheetah.md)
