# Xbotics 第20讲：世界模型 —— 预测、规划与数据飞轮 / Xbotics Lecture 20: World Models

- **类型：** 官方教程章节 / course
- **维护者：** Xbotics 具身智能社区
- **官方章节：** https://xbotics-embodied-ai-club.github.io/Xbotics-Embodied-AI-Handbook/part6-agent-world-model/20-world-model.html
- **固定版本 Markdown：** https://github.com/Xbotics-Embodied-AI-club/Xbotics-Embodied-AI-Handbook/blob/342247fb71dd0ee15386b6eab80543c50593d3c1/docs/part6-agent-world-model/20-world-model.md
- **核对范围：** 官方章节完整正文及参考资料列表；不把课程所引第三方工作计为独立通读。
- **通读：** 已完成
- **代码入口：** https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/experiments/libero/eval_libero_single.py
- **验证边界：** 文稿核验与源码静态走读，不代表本次重训、复现实验或真机运行。

## 编译线索

从预测动作后果切入，区分面向策略学习的世界模型、学习型仿真器、机器人视频世界建模，最后以 Fast-WAM 的 LIBERO 实验拆解训练和部署预算。

动作侧提速依赖该架构的首帧条件、掩码和缓存设计，不能外推到一切 WAM；小样本 smoke test 仅检验入口与配置，不支持论文级结论。

## 对 wiki 的映射

- [第 20 讲独立知识节点](../../wiki/entities/xbotics-handbook-lecture-20.md)
- [课程主节点](../../wiki/entities/xbotics-embodied-ai-handbook.md)
