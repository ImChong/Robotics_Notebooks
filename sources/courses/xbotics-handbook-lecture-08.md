# Xbotics 第8讲：端到端策略导论 —— 从 Pipeline 到 Policy / Xbotics Lecture 8: Introduction to End-to-End Policies: From Pipeline to Policy

- **类型：** 官方教程章节 / course
- **维护者：** Xbotics 具身智能社区
- **官方章节：** https://xbotics-embodied-ai-club.github.io/Xbotics-Embodied-AI-Handbook/part3-end-to-end/08-%E7%AB%AF%E5%88%B0%E7%AB%AF%E7%AD%96%E7%95%A5%E5%AF%BC%E8%AE%BA.html
- **固定版本 Markdown：** https://github.com/Xbotics-Embodied-AI-club/Xbotics-Embodied-AI-Handbook/blob/342247fb71dd0ee15386b6eab80543c50593d3c1/docs/part3-end-to-end/08-%E7%AB%AF%E5%88%B0%E7%AB%AF%E7%AD%96%E7%95%A5%E5%AF%BC%E8%AE%BA.md
- **核对范围：** 官方章节完整正文及参考资料列表；不把课程所引第三方工作计为独立通读。
- **公众号入口（该版本正文未核验）：** https://mp.weixin.qq.com/s/jmDRNwsFD_Uv9-6f3q6pxw
- **通读：** 已完成
- **代码入口：** https://github.com/Xbotics-Embodied-AI-club/Xbotics-Embodied-AI-Handbook/blob/342247fb71dd0ee15386b6eab80543c50593d3c1/code/vla/1_policy_rollout/1_2_pi0_libero_rollout/pi0_demo.py
- **验证边界：** 文稿核验与源码静态走读，不代表本次重训、复现实验或真机运行。

## 编译线索

对比显式感知、规划、控制模块和直接观测到动作的策略。以 LeRobot 的 LIBERO 微调版 π0 演示从任务观测、预处理、执行到下一观测的闭环。

演示选了匹配的 LIBERO 微调权重与验证过的初态，不是基座零样本成绩；单步裁剪也不能修复绝对/相对口径或坐标系用错。

## 对 wiki 的映射

- [第 8 讲独立知识节点](../../wiki/entities/xbotics-handbook-lecture-08.md)
- [课程主节点](../../wiki/entities/xbotics-embodied-ai-handbook.md)
