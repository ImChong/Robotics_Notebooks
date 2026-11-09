# End to End Learning for Self-Driving Cars（arXiv:1604.07316）

- **类型：** paper / primary-source reading
- **标题：** End to End Learning for Self-Driving Cars
- **作者：** Mariusz Bojarski, Davide Del Testa, Daniel Dworakowski et al.
- **arXiv：** https://arxiv.org/abs/1604.07316
- **HTML：** https://arxiv.org/html/1604.07316v1
- **原文 PDF：** https://arxiv.org/pdf/1604.07316
- **核对范围：** 论文摘要、§1 以及 §4–6 的数据、网络和仿真流程；§7–8 的实验逐项核对与论文图像细节待补齐。
- **通读：** 未完成
- **一句话摘录：** NVIDIA DAVE-2 将驾驶员前视图像与转向指令配对，CNN 联合学习感知与转向；以仿射扰动增强恢复偏航能力。

## 基于当前已核实原文的要点

- 训练信号：驾驶员转向指令，人工移动/旋转图像后的恢复转向指令。
- 网络：YUV 图像、归一化、5 层卷积与 3 层全连接，输出转弯半径倒数。
- 仿真：利用已记录视频和车体动力学按策略的偏离量变换下一帧；离路中心超过 1m 触发模拟接管。
- 边界：主要评测车道保持，不等于开放道路完整自动驾驶。

## 开放资源核查

- arXiv 论文及 NVIDIA 官方技术博客为一手入口；本轮没有核实到与论文逐项对应的公开可运行实现，不能据此断言永远未开源。
- 补充一手入口：https://developer.nvidia.com/blog/deep-learning-self-driving-cars/

## 对 wiki 的映射

- [paper-dave2-end-to-end-driving.md](../../wiki/entities/paper-dave2-end-to-end-driving.md)
- [11 篇论文路线](../../wiki/overview/robot-learning-foundations-11-papers.md)

## 未完成事项

- 继续阅读尚未核实的正文/附录、表格及可复现资产；完成后方可将“通读”改成“已完成”。
