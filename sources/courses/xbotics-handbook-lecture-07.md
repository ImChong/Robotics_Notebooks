# Xbotics 第7讲：机器人操作技能 / Xbotics Lecture 7: Robot Manipulation Skills: Pick, Place, and Failure Recovery

- **类型：** 官方教程章节 / course
- **维护者：** Xbotics 具身智能社区
- **官方章节：** https://xbotics-embodied-ai-club.github.io/Xbotics-Embodied-AI-Handbook/part2-vision-manipulation/07-manipulation-skills.html
- **固定版本 Markdown：** https://github.com/Xbotics-Embodied-AI-club/Xbotics-Embodied-AI-Handbook/blob/342247fb71dd0ee15386b6eab80543c50593d3c1/docs/part2-vision-manipulation/07-manipulation-skills.md
- **核对范围：** 官方章节完整正文及参考资料列表；不把课程所引第三方工作计为独立通读。
- **通读：** 已完成
- **代码入口：** https://github.com/Xbotics-Embodied-AI-club/Xbotics-Embodied-AI-Handbook/blob/342247fb71dd0ee15386b6eab80543c50593d3c1/code/lecture07/robot_pick_place/run_demo.py
- **验证边界：** 文稿核验与源码静态走读，不代表本次重训、复现实验或真机运行。

## 编译线索

用顶抓方块和侧抓直立瓶对照同一抓放状态机。先由任务参数生成五个位姿，再通过夹持、抬升、放置三道物理验证，而不是把运动完成当任务成功。

run_demo 默认只接 MockBackend，模拟结果不能替代真机成功率。教材列出的偏移量是工程教学示例，不是可直接照搬的真机标定值。

## 对 wiki 的映射

- [第 7 讲独立知识节点](../../wiki/entities/xbotics-handbook-lecture-07.md)
- [课程主节点](../../wiki/entities/xbotics-embodied-ai-handbook.md)
