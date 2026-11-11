# Motion Graphs

- **类型：** 作者一手论文
- **作者：** Lucas Kovar, Michael Gleicher, Frédéric Pighin
- **会议：** SIGGRAPH 2002
- **机构：** 威斯康星大学麦迪逊分校（University of Wisconsin–Madison）；南加州大学（University of Southern California）
- **论文：** https://graphics.cs.wisc.edu/Papers/2002/KGP02/
- **原始版本：** SIGGRAPH 2002 论文作者存档
- **项目页：** https://research.cs.wisc.edu/graphics/Gallery/kovar.vol/MoGraphs/
- **代码：** 作者主页未列可核实官方代码
- **核对范围：** 核对论文官方页面/HTML 可见的方法与结果、作者项目主页；未核实到可运行实现；全文细节、所有附录未逐段完成交叉核对。
- **通读：** 未完成

## 方法与应用

基于动捕片段建立有向连接和可平滑的过渡边；把用户的行进轨迹、风格要求转成图搜索目标。交互方向控制可预计算节点与方向对应的后续片段。

## 实验与结论

原作者演示覆盖沿任意路径行走、混合动作风格与交互改变方向；示例中角色响应转向可能有延迟，不是即时重规划的动力学机器人。

## 局限 / 对 wiki 的映射

不能直接证明重定向到 G1 后的物理稳定性；路径跟踪受动捕库覆盖程度与转场限制。

- [Motion Graphs：动捕片段图与路径控制](../../wiki/entities/paper-motion-graphs.md)
