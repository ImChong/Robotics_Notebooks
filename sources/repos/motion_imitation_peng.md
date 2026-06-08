# motion_imitation（Peng et al.）

> 来源归档

- **标题：** Learning Agile Robotic Locomotion Skills by Imitating Animals
- **类型：** repo
- **链接：** https://github.com/erwincoumans/motion_imitation
- **论文项目页：** https://xbpeng.github.io/projects/Robotic_Imitation/index.html
- **入库日期：** 2026-06-08
- **一句话说明：** 四足模仿动物 MoCap 的奠基开源实现（PyBullet）；将动物参考运动转为四足机器人可跟踪轨迹并训练模仿策略，是后续 AMP / legged_gym 生态的重要前序。
- **沉淀到 wiki：** 是 → [`wiki/entities/motion-imitation-quadruped.md`](../../wiki/entities/motion-imitation-quadruped.md)

## 要点

- `paper` 分支含论文原始代码；`motion_imitation/data/motions/` 内置多种动物参考片段。
- 训练：`python motion_imitation/run.py --mode train --motion_file ...`
